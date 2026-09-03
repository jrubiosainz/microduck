#!/usr/bin/env python3
"""Choose the wide camera by MEASUREMENT, replaying the real recorded trace.

Framing is the one part of this behavior that cannot be graded by the
acceptance gate, so it is graded here instead of by eye.  The probe replays the
ACTUAL trace through ``render_frames``' own easing and distance ramp - so the
camera path it scores is the camera path the render will fly - and reports, per
candidate:

* ``duck_on_screen``     - the duck's trunk projects inside the frame;
* ``duck_unoccluded``    - and a ray from the camera eye to it reaches it, so it
  is not behind a crate;
* ``subject_on_screen``  - the body it is negotiating with is in frame too,
  which is what makes a decision legible;
* ``duck_px``            - the duck's apparent size in pixels, because a duck
  that is technically on screen at four pixels across is not visible;
* ``clear_of_hud``       - it does not sit under the HUD panels, whose pixel
  boxes are taken from ``video_overlay`` rather than guessed.

A candidate that wins on visibility but puts the duck under the DECISION panel
has not won.

Run:
    ../../microduck_rl/.venv/bin/python tools/probe_framing.py --trace /tmp/sl_trace.json
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

from policy_runtime import load_scene  # noqa: E402
from slalom_course import GOAL_XY  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# The HUD's occupied pixel boxes, as a fraction of frame width/height.  Taken
# from ``video_overlay.compose``: the left column is 10..278 px of a 960 px
# frame, the right column starts at width-310, the title is the top 40 px and
# the timeline the bottom 56.
HUD_LEFT_FRAC = 278.0 / 960.0
HUD_RIGHT_FRAC = 1.0 - 310.0 / 960.0
HUD_TOP_FRAC = 40.0 / 640.0
HUD_BOTTOM_FRAC = 1.0 - 56.0 / 640.0

# The duck's exact planar half-extent, for the apparent-size estimate.
DUCK_HALF_M = 0.0827


def project(point, eye, forward, right, up, tan_h, tan_v):
    """Normalised device coordinates of a world point, or None if behind."""
    delta = np.asarray(point, dtype=np.float64) - eye
    depth = float(delta @ forward)
    if depth <= 1e-6:
        return None
    x = float(delta @ right) / (depth * tan_h)
    y = float(delta @ up) / (depth * tan_v)
    return x, y, depth


def camera_basis(lookat, azimuth_deg, elevation_deg, distance):
    """Eye position and orthonormal basis for a MuJoCo free camera."""
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    forward = np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation)])
    forward /= np.linalg.norm(forward)
    eye = np.asarray(lookat, dtype=np.float64) - forward * distance
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    return eye, forward, right, up


def occluded(model, data, eye, target) -> bool:
    """Does scene geometry block the straight line from ``eye`` to ``target``?"""
    delta = np.asarray(target, dtype=np.float64) - eye
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        return False
    direction = delta / distance
    geom_id = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, eye, direction, None, 1, -1, geom_id)
    return bool(geom_id[0] >= 0 and 0.0 < hit < distance - 0.06)


def score(records, model, data, *, azimuth, elevation, near, far, fovy=45.0,
          width=960, height=640, subject_bias=0.62, ease=0.045,
          lookat_z=0.42, separation_far=3.40, stride=10) -> dict:
    """Replay the trace through the render's own easing and score the shot."""
    tan_v = math.tan(math.radians(fovy) * 0.5)
    tan_h = (width / height) * tan_v

    lookat = np.array([records[0]["duck_xy"][0], records[0]["duck_xy"][1],
                       lookat_z])
    distance = near
    totals = {"duck_on": 0, "duck_clear": 0, "subject_on": 0,
              "clear_of_hud": 0, "samples": 0}
    duck_px = []

    for index, record in enumerate(records):
        duck = np.array(record["duck_xy"], dtype=np.float64)
        name = record["subject"]
        subject = (np.asarray(GOAL_XY, dtype=np.float64) if name == "goal"
                   else np.array(record["actor_xy"].get(name, record["duck_xy"]),
                                 dtype=np.float64))
        target = np.array([
            subject_bias * duck[0] + (1.0 - subject_bias) * subject[0],
            subject_bias * duck[1] + (1.0 - subject_bias) * subject[1],
            lookat_z])
        lookat += ease * (target - lookat)
        separation = float(np.linalg.norm(duck - subject))
        wanted = near + (far - near) * min(
            max(separation / separation_far, 0.0), 1.0)
        distance += ease * (wanted - distance)

        if index % stride:
            continue

        eye, forward, right, up = camera_basis(lookat, azimuth, elevation,
                                               distance)
        duck_world = np.array([duck[0], duck[1], 0.116])
        projected = project(duck_world, eye, forward, right, up, tan_h, tan_v)
        totals["samples"] += 1
        if projected is None:
            continue
        x, y, depth = projected
        on_screen = abs(x) <= 1.0 and abs(y) <= 1.0
        if on_screen:
            totals["duck_on"] += 1
            # Pixel half-width of the duck at this depth.
            half_px = (DUCK_HALF_M / (depth * tan_h)) * (width / 2.0)
            duck_px.append(2.0 * half_px)
            # Frame position, with +x right and +y UP in NDC.
            fx = (x + 1.0) * 0.5
            fy = (1.0 - y) * 0.5
            if HUD_LEFT_FRAC < fx < HUD_RIGHT_FRAC \
                    and HUD_TOP_FRAC < fy < HUD_BOTTOM_FRAC:
                totals["clear_of_hud"] += 1
            data.qpos[0], data.qpos[1] = duck[0], duck[1]
            mujoco.mj_forward(model, data)
            if not occluded(model, data, eye, duck_world):
                totals["duck_clear"] += 1
        subject_projected = project(
            np.array([subject[0], subject[1], 0.35]), eye, forward, right, up,
            tan_h, tan_v)
        if subject_projected and abs(subject_projected[0]) <= 1.0 \
                and abs(subject_projected[1]) <= 1.0:
            totals["subject_on"] += 1

    samples = max(totals["samples"], 1)
    return {
        "azimuth": azimuth, "elevation": elevation,
        "near": near, "far": far,
        "duck_on_screen": totals["duck_on"] / samples,
        "duck_unoccluded": totals["duck_clear"] / samples,
        "subject_on_screen": totals["subject_on"] / samples,
        "clear_of_hud": totals["clear_of_hud"] / samples,
        "duck_px_median": float(np.median(duck_px)) if duck_px else 0.0,
        "duck_px_min": float(np.min(duck_px)) if duck_px else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="/tmp/sl_trace.json")
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    records = json.loads(Path(args.trace).read_text())
    model = load_scene()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("=" * 104)
    print("FRAMING CANDIDATES  (replayed through the render's own easing)")
    print("=" * 104)
    print(f"  {'az':>5} {'elev':>6} {'near':>5} {'far':>5} {'duck on':>8} "
          f"{'unoccl':>7} {'subj on':>8} {'clear HUD':>10} {'duck px':>8} "
          f"{'min px':>7}")

    results = []
    for elevation in (-52.0, -58.0, -64.0, -70.0):
        for azimuth in (38.0, 55.0, 90.0):
            for near, far in ((3.40, 4.60), (4.00, 5.20), (4.60, 6.20)):
                result = score(records, model, data, azimuth=azimuth,
                               elevation=elevation, near=near, far=far,
                               stride=args.stride)
                results.append(result)
                print(f"  {azimuth:5.0f} {elevation:6.0f} {near:5.2f} "
                      f"{far:5.2f} {result['duck_on_screen']:8.3f} "
                      f"{result['duck_unoccluded']:7.3f} "
                      f"{result['subject_on_screen']:8.3f} "
                      f"{result['clear_of_hud']:10.3f} "
                      f"{result['duck_px_median']:8.1f} "
                      f"{result['duck_px_min']:7.1f}")

    print()
    print("BEST by (unoccluded, clear of HUD, subject on screen, duck size):")
    best = max(results, key=lambda r: (round(r["duck_unoccluded"], 3),
                                       round(r["clear_of_hud"], 2),
                                       round(r["subject_on_screen"], 2),
                                       r["duck_px_median"]))
    for key, value in best.items():
        print(f"  {key:20} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
