#!/usr/bin/env python3
"""Measure the wide camera's framing by replaying a REAL recorded trace.

The framing constants in ``render_frames`` are the output of this script, not a
choice.  It replays the recorded per-tick trace through that module's OWN easing
and distance ramp - so the camera path it scores is the camera path the render
will fly - and grades each candidate on five things a viewer cares about:

* **duck on screen** - is the robot inside the frame at all;
* **duck unoccluded** - can it be SEEN, or is a shelf between it and the camera;
* **subject on screen** - during an investigation, is the thing it is looking at
  in frame too, since that relationship is the shot;
* **duck clear of the HUD** - does it fall behind a panel, where it cannot be
  watched;
* **duck size** - how many pixels across, at the median and at the worst.

Run:
    ../../microduck_rl/.venv/bin/python tools/probe_framing.py \\
        --trace /tmp/pt_trace.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from patrol_facility import FLOOR_HALF  # noqa: E402
from patrol_states import DUCK_EXACT_PLANAR_RADIUS  # noqa: E402
from policy_runtime import load_scene  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# The HUD's occupied rectangles, in fractions of the frame.  From
# ``video_overlay.compose``: the left column, the right column and the timeline.
HUD_RECTS = (
    (10 / 960, 44 / 640, 278 / 960, 510 / 640),     # left column
    (650 / 960, 44 / 640, 950 / 960, 584 / 640),    # right column + PiP
    (10 / 960, 584 / 640, 950 / 960, 632 / 640),    # timeline
)


def project(point, camera, width, height):
    """World point to pixel, using the same pinhole MuJoCo renders with."""
    elevation = math.radians(camera.elevation)
    azimuth = math.radians(camera.azimuth)
    forward = np.array([math.cos(elevation) * math.cos(azimuth),
                        math.cos(elevation) * math.sin(azimuth),
                        math.sin(elevation)])
    eye = np.asarray(camera.lookat, dtype=np.float64) - forward * camera.distance
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)

    delta = np.asarray(point, dtype=np.float64) - eye
    depth = float(delta @ forward)
    if depth <= 1e-6:
        return None, depth
    # MuJoCo's free camera uses a 45 deg vertical FOV by default.
    tan_v = math.tan(math.radians(45.0) * 0.5)
    tan_h = (width / height) * tan_v
    x = float(delta @ right) / (depth * tan_h)
    y = float(delta @ up) / (depth * tan_v)
    return (0.5 * (x + 1.0) * width, 0.5 * (1.0 - y) * height), depth


def in_hud(px, py, width, height) -> bool:
    for x0, y0, x1, y1 in HUD_RECTS:
        if x0 * width <= px <= x1 * width and y0 * height <= py <= y1 * height:
            return True
    return False


def score(trace, azimuth, elevation, near, far, width, height, model, data):
    """Replay the trace through render_frames' own easing and grade the shot."""
    import render_frames as RF

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.elevation = elevation
    camera.azimuth = azimuth
    camera.distance = near

    lookat = np.array([trace[0]["duck_xy"][0], trace[0]["duck_xy"][1],
                       RF.LOOKAT_Z])
    distance = near

    on_screen = unoccluded = subject_on = clear_of_hud = 0
    subject_steps = 0
    sizes: list[float] = []
    total = 0

    for record in trace:
        duck = np.array(record["duck_xy"], dtype=np.float64)
        subject_xy = record["actor_xy"].get(record["subject"])
        if subject_xy is None:
            target = np.array([duck[0], duck[1], RF.LOOKAT_Z])
            separation = 0.0
        else:
            subject_xy = np.array(subject_xy, dtype=np.float64)
            target = np.array([
                RF.LOOKAT_SUBJECT_BIAS * duck[0]
                + (1 - RF.LOOKAT_SUBJECT_BIAS) * subject_xy[0],
                RF.LOOKAT_SUBJECT_BIAS * duck[1]
                + (1 - RF.LOOKAT_SUBJECT_BIAS) * subject_xy[1],
                RF.LOOKAT_Z])
            separation = float(np.linalg.norm(duck - subject_xy))
        lookat += RF.LOOKAT_EASE * (target - lookat)
        wanted = near + (far - near) * min(
            max(separation / RF.SEPARATION_FOR_FAR_M, 0.0), 1.0)
        distance += RF.LOOKAT_EASE * (wanted - distance)

        camera.azimuth = azimuth + RF.AZIMUTH_SWING_DEG * math.sin(
            record["t"] / RF.AZIMUTH_SWING_PERIOD_S)
        camera.distance = distance
        camera.lookat[0], camera.lookat[1] = float(lookat[0]), float(lookat[1])
        camera.lookat[2] = RF.LOOKAT_Z

        total += 1
        head = np.array([duck[0], duck[1], 0.12])
        pixel, depth = project(head, camera, width, height)
        if pixel is None:
            continue
        px, py = pixel
        if not (0 <= px < width and 0 <= py < height):
            continue
        on_screen += 1
        if not in_hud(px, py, width, height):
            clear_of_hud += 1

        # Angular size of the duck, in pixels.
        tan_v = math.tan(math.radians(45.0) * 0.5)
        sizes.append(2.0 * DUCK_EXACT_PLANAR_RADIUS / (depth * tan_v)
                     * (height * 0.5))

        # Occlusion: cast from the eye to the duck through the real scene.
        elevation_r = math.radians(camera.elevation)
        azimuth_r = math.radians(camera.azimuth)
        forward = np.array([math.cos(elevation_r) * math.cos(azimuth_r),
                            math.cos(elevation_r) * math.sin(azimuth_r),
                            math.sin(elevation_r)])
        eye = np.asarray(camera.lookat) - forward * camera.distance
        span = head - eye
        length = float(np.linalg.norm(span))
        direction = span / max(length, 1e-9)
        geom_id = np.zeros(1, dtype=np.int32)
        hit = mujoco.mj_ray(model, data, eye, direction, None, 1, -1, geom_id)
        if geom_id[0] < 0 or hit < 0.0 or hit >= length - 0.25:
            unoccluded += 1

        if subject_xy is not None:
            subject_steps += 1
            spixel, _ = project(
                np.array([subject_xy[0], subject_xy[1], 0.30]),
                camera, width, height)
            if spixel and 0 <= spixel[0] < width and 0 <= spixel[1] < height:
                subject_on += 1

    return {
        "azimuth": azimuth,
        "elevation": elevation,
        "near": near,
        "far": far,
        "duck_on_screen": on_screen / max(total, 1),
        "duck_unoccluded": unoccluded / max(total, 1),
        "subject_on_screen": subject_on / max(subject_steps, 1),
        "duck_clear_of_hud": clear_of_hud / max(total, 1),
        "duck_px_median": float(np.median(sizes)) if sizes else 0.0,
        "duck_px_min": float(np.min(sizes)) if sizes else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="/tmp/pt_trace.json")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--stride", type=int, default=5,
                        help="sample every Nth tick, for speed")
    args = parser.parse_args()

    trace = json.loads(Path(args.trace).read_text())[::args.stride]
    model = load_scene()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"{len(trace)} sampled ticks from {args.trace}")
    print()
    print(f"  {'azim':>5} {'elev':>5} {'near':>5} {'far':>5} "
          f"{'onscr':>6} {'unocc':>6} {'subj':>6} {'clear':>6} "
          f"{'px50':>6} {'pxmin':>6}")

    best = None
    for elevation in (-38.0, -46.0, -52.0, -58.0, -64.0):
        for azimuth in (38.0, 55.0, 90.0, 125.0):
            for near, far in ((3.20, 4.40), (3.60, 5.00), (4.20, 5.60)):
                result = score(trace, azimuth, elevation, near, far,
                               args.width, args.height, model, data)
                print(f"  {azimuth:5.0f} {elevation:5.0f} {near:5.2f} "
                      f"{far:5.2f} {result['duck_on_screen']:6.3f} "
                      f"{result['duck_unoccluded']:6.3f} "
                      f"{result['subject_on_screen']:6.3f} "
                      f"{result['duck_clear_of_hud']:6.3f} "
                      f"{result['duck_px_median']:6.1f} "
                      f"{result['duck_px_min']:6.1f}")
                # Rank on visibility first, then on how much of the time the
                # duck is somewhere a viewer can actually watch it.
                key = (round(result["duck_on_screen"], 2),
                       round(result["duck_unoccluded"], 2),
                       round(result["subject_on_screen"], 2),
                       result["duck_clear_of_hud"])
                if best is None or key > best[0]:
                    best = (key, result)

    print()
    print("BEST")
    for key, value in best[1].items():
        print(f"  {key:<20} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
