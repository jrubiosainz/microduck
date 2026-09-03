#!/usr/bin/env python3
"""Choose the wide camera by REPLAYING THE REAL TRACE, not by eye.

The first framing attempt used an azimuth copied from the sibling promenade
behavior and produced a video in which the duck is behind a partition for much
of the run.  That is not a taste problem: this concourse is divided by two
2.05 m full-height slabs, and a camera whose eye sits below them looks straight
into one for every leg of the route that runs behind it.

So the camera is SCORED.  For each candidate (azimuth, elevation, distance) this
replays the recorded trace through the SAME easing, distance ramp and look-at
clamp that ``render_frames`` applies, and measures per sampled frame:

* is the duck inside the frame at all;
* is the duck unoccluded in 3D, against the partitions, screen and columns as
  real solid boxes and every person as a standing cylinder;
* is the FOLLOWER in frame and unoccluded — this behavior is about a pair, and a
  shot that loses her is a shot of the wrong thing;
* is the camera's own eye inside the concourse, so MuJoCo cannot draw the near
  wall as a slab across the lens;
* is the duck clear of the HUD panels, whose rectangles are imported from the
  overlay rather than restated.

Run:
    ../../microduck_rl/.venv/bin/python tools/probe_framing.py --trace /tmp/t.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

from guide_cast import BY_NAME  # noqa: E402
from guide_layout import FLOOR_HALF, OBSTACLES  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

WIDTH, HEIGHT = 960, 640
FOVY_DEG = 45.0

# The HUD rectangles the duck must stay clear of, imported in spirit from
# ``video_overlay.compose``: the left column, the PiP and the plan view.
PANELS = [
    (12, 38, 316, 594),          # the whole left column
    (WIDTH - 312, 38, WIDTH - 12, 254),   # the PiP
    (WIDTH - 320, 262, WIDTH - 12, HEIGHT - 92),  # the plan view
    (324, HEIGHT - 86, WIDTH - 12, HEIGHT - 12),  # the timeline
]

# A person, as a standing cylinder, for the occlusion test.
PERSON_RADIUS = 0.24
PERSON_TOP = 0.72


def look_matrix(lookat, azimuth_deg, elevation_deg, distance):
    """MuJoCo free-camera geometry: eye, forward, right, up."""
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    forward = np.array([math.cos(elevation) * math.cos(azimuth),
                        math.cos(elevation) * math.sin(azimuth),
                        math.sin(elevation)])
    eye = np.asarray(lookat, dtype=np.float64) - forward * distance
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    return eye, forward, right, up


def project(point, eye, forward, right, up):
    """World point to pixels.  Returns ``None`` behind the camera."""
    delta = np.asarray(point, dtype=np.float64) - eye
    depth = float(delta @ forward)
    if depth <= 0.05:
        return None
    tan_v = math.tan(math.radians(FOVY_DEG) * 0.5)
    tan_h = tan_v * (WIDTH / HEIGHT)
    x = float(delta @ right) / (depth * tan_h)
    y = float(delta @ up) / (depth * tan_v)
    return (0.5 * WIDTH * (1.0 + x), 0.5 * HEIGHT * (1.0 - y))


def on_screen(px) -> bool:
    return px is not None and 0 <= px[0] < WIDTH and 0 <= px[1] < HEIGHT


def clear_of_panels(px) -> bool:
    if px is None:
        return False
    for x0, y0, x1, y1 in PANELS:
        if x0 <= px[0] <= x1 and y0 <= px[1] <= y1:
            return False
    return True


def blocked_by_scenery(eye, target) -> bool:
    """Does a full-height body stand in the line from the eye to the target?

    The obstacles are boxes; the segment is sampled finely and each sample
    tested against the box in 3D, so a camera looking OVER a 2.05 m partition
    correctly sees past it while one looking THROUGH it does not.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    steps = 120
    for obstacle in OBSTACLES:
        half = np.array([obstacle.half[0], obstacle.half[1],
                         0.5 * obstacle.height_m])
        centre = np.array([obstacle.center[0], obstacle.center[1],
                           0.5 * obstacle.height_m])
        for index in range(steps + 1):
            point = eye + (target - eye) * (index / steps)
            local = np.abs(point - centre)
            if obstacle.kind == "circle":
                if (float(np.linalg.norm(point[:2] - centre[:2]))
                        <= obstacle.radius and local[2] <= half[2]):
                    return True
            elif np.all(local <= half):
                return True
    return False


def blocked_by_person(eye, target, people, ignore=()) -> bool:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    for name, xy in people.items():
        if name in ignore:
            continue
        centre = np.array([xy[0], xy[1]])
        for index in range(1, 60):
            point = eye + (target - eye) * (index / 60)
            if point[2] > PERSON_TOP:
                continue
            if float(np.linalg.norm(point[:2] - centre)) <= PERSON_RADIUS:
                return True
    return False


def eye_inside_hall(eye, margin: float = 0.30) -> bool:
    return (abs(float(eye[0])) <= FLOOR_HALF[0] - margin
            and abs(float(eye[1])) <= FLOOR_HALF[1] - margin
            and float(eye[2]) > 0.15)


def score(records, azimuth, elevation, near, far, *, stride=25):
    """Replay the trace through this candidate and score it."""
    from render_frames import (
        AZIMUTH_SWING_DEG,
        AZIMUTH_SWING_PERIOD_S,
        EYE_WALL_MARGIN_M,
        LOOKAT_DUCK_BIAS,
        LOOKAT_EASE,
        LOOKAT_Z,
        SEPARATION_FOR_FAR_M,
    )

    lookat = np.array([-2.5, -1.4, LOOKAT_Z])
    distance = near
    totals = dict(frames=0, duck_on=0, duck_clear=0, duck_unoccluded=0,
                  follower_on=0, follower_unoccluded=0, eye_in=0,
                  duck_px=0.0, pair_px=0.0)

    for index, record in enumerate(records):
        duck = np.array(record["duck_xy"], dtype=np.float64)
        follower = np.array(record["follower_xy"], dtype=np.float64)
        target = np.array([
            LOOKAT_DUCK_BIAS * duck[0] + (1 - LOOKAT_DUCK_BIAS) * follower[0],
            LOOKAT_DUCK_BIAS * duck[1] + (1 - LOOKAT_DUCK_BIAS) * follower[1],
            LOOKAT_Z])
        lookat = lookat + LOOKAT_EASE * (target - lookat)
        separation = float(record["follower_range_m"])
        wanted = near + (far - near) * min(max(
            separation / SEPARATION_FOR_FAR_M, 0.0), 1.0)
        distance += LOOKAT_EASE * (wanted - distance)
        if index % stride:
            continue

        swing = azimuth + AZIMUTH_SWING_DEG * math.sin(
            record["t"] / AZIMUTH_SWING_PERIOD_S)
        el = math.radians(elevation)
        head = math.radians(swing)
        half_x = FLOOR_HALF[0] - EYE_WALL_MARGIN_M
        half_y = FLOOR_HALF[1] - EYE_WALL_MARGIN_M
        shift_x = math.cos(el) * math.cos(head) * distance
        shift_y = math.cos(el) * math.sin(head) * distance
        clamped = np.array([
            float(np.clip(lookat[0], -half_x + shift_x, half_x + shift_x)),
            float(np.clip(lookat[1], -half_y + shift_y, half_y + shift_y)),
            float(lookat[2])])

        eye, forward, right, up = look_matrix(clamped, swing, elevation,
                                              distance)
        duck_3d = np.array([duck[0], duck[1], 0.14])
        follower_3d = np.array([follower[0], follower[1], 0.55])
        duck_px = project(duck_3d, eye, forward, right, up)
        follower_px = project(follower_3d, eye, forward, right, up)

        totals["frames"] += 1
        if eye_inside_hall(eye):
            totals["eye_in"] += 1
        if on_screen(duck_px):
            totals["duck_on"] += 1
            if clear_of_panels(duck_px):
                totals["duck_clear"] += 1
        if on_screen(follower_px):
            totals["follower_on"] += 1
        people = {n: xy for n, xy in record["person_xy"].items()}
        if not blocked_by_scenery(eye, duck_3d) and not blocked_by_person(
                eye, duck_3d, people):
            totals["duck_unoccluded"] += 1
        if not blocked_by_scenery(eye, follower_3d) and not blocked_by_person(
                eye, follower_3d, people, ignore=(record["follower"],)):
            totals["follower_unoccluded"] += 1
        # Apparent size, so the shot does not win by being far away.
        top = project(np.array([duck[0], duck[1], 0.30]), eye, forward, right,
                      up)
        if duck_px and top:
            totals["duck_px"] += abs(duck_px[1] - top[1])
        if duck_px and follower_px:
            totals["pair_px"] += float(np.linalg.norm(
                np.array(duck_px) - np.array(follower_px)))

    n = max(totals["frames"], 1)
    return {
        "azimuth": azimuth, "elevation": elevation,
        "near": near, "far": far,
        "duck_on": totals["duck_on"] / n,
        "duck_clear": totals["duck_clear"] / n,
        "duck_unoccluded": totals["duck_unoccluded"] / n,
        "follower_on": totals["follower_on"] / n,
        "follower_unoccluded": totals["follower_unoccluded"] / n,
        "eye_in": totals["eye_in"] / n,
        "duck_px": totals["duck_px"] / n,
        "pair_px": totals["pair_px"] / n,
    }


def rank(result) -> float:
    """One number.  Occlusion dominates, because it is what went wrong."""
    return (3.0 * result["duck_unoccluded"]
            + 2.0 * result["follower_unoccluded"]
            + 1.5 * result["duck_on"]
            + 1.0 * result["follower_on"]
            + 1.0 * result["duck_clear"]
            + 1.0 * result["eye_in"]
            + 0.010 * min(result["duck_px"], 60.0)
            + 0.002 * min(result["pair_px"], 260.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="/tmp/lms-trace.json")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    records = json.loads(Path(args.trace).read_text())
    print(f"{len(records)} recorded ticks from {args.trace}")

    results = []
    for azimuth in range(0, 360, 15):
        for elevation in (-24.0, -32.0, -40.0, -48.0, -56.0):
            for near, far in ((3.15, 4.35), (3.8, 5.2), (4.6, 6.0)):
                results.append(score(records, float(azimuth), elevation,
                                     near, far))
    results.sort(key=rank, reverse=True)

    print()
    print(f"{'az':>5} {'el':>6} {'near':>5} {'far':>5} {'duckOn':>7} "
          f"{'duckClr':>8} {'duckVis':>8} {'folOn':>7} {'folVis':>7} "
          f"{'eyeIn':>6} {'px':>5} {'pair':>6} {'score':>6}")
    for result in results[:args.top]:
        print(f"{result['azimuth']:5.0f} {result['elevation']:6.1f} "
              f"{result['near']:5.2f} {result['far']:5.2f} "
              f"{result['duck_on']:7.3f} {result['duck_clear']:8.3f} "
              f"{result['duck_unoccluded']:8.3f} {result['follower_on']:7.3f} "
              f"{result['follower_unoccluded']:7.3f} {result['eye_in']:6.3f} "
              f"{result['duck_px']:5.1f} {result['pair_px']:6.1f} "
              f"{rank(result):6.3f}")

    best = results[0]
    print()
    print("BEST:")
    for key, value in best.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
