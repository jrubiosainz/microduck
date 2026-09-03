#!/usr/bin/env python3
"""Score wide-camera candidates by replaying the REAL recorded trace.

The framing of this video is a MEASUREMENT, not a taste.  The building is
divided by 1.35 m partitions and the lift car has 1.15 m walls, so a shallow
camera films the outside of a box for the whole second half of the run - the
part the video exists to show.

So this tool replays the actual recorded trace through ``render_frames``' own
easing, distance ramp and look-at clamp, and scores each candidate on:

* is the DUCK on screen, and unoccluded in 3D against every wall, jamb, cabin
  panel, door leaf and person, treated as solid;
* is its current SUBJECT on screen and unoccluded, since every decision in this
  behavior is about somebody else;
* does the camera's own EYE stay out of the scenery;
* is the duck clear of the HUD panels, which cover the left column and the
  right column of every frame.

Occlusion is a real MuJoCo ray cast through the compiled model at the recorded
pose, not a planar approximation - the door leaves move, and a candidate that
looked fine at t=0 can be blocked by a shut lift door at t=60.

Run:
    ../../microduck_rl/.venv/bin/python tools/probe_framing.py --trace /tmp/dee_trace.json
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

from etiquette_actors import people_at, pose_people  # noqa: E402
from etiquette_cast import BY_NAME  # noqa: E402
from etiquette_markers import pose_leaves  # noqa: E402
from lobby_doors import doors_at  # noqa: E402
from lobby_layout import FLOOR_HALF  # noqa: E402
from policy_runtime import load_scene  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# The HUD's own panel columns, as fractions of the frame.  A duck drawn under a
# panel is a duck the viewer cannot see, however well framed it is.
HUD_LEFT_FRAC = 278.0 / 960.0
HUD_RIGHT_FRAC = 1.0 - 310.0 / 960.0
HUD_TOP_FRAC = 40.0 / 640.0
HUD_BOTTOM_FRAC = 1.0 - 58.0 / 640.0


def visible_point(model, data, eye, look_dir, target, fovy_deg, aspect,
                  ignore_bodies):
    """Is ``target`` inside the frustum and unoccluded from ``eye``?

    Returns (on_screen, unoccluded, ndc_x, ndc_y).

    ``look_dir`` IS THE CAMERA'S OWN FORWARD AXIS AND MUST BE PASSED IN.
    An earlier version built the basis from ``target - eye``, which points the
    camera AT whatever it is testing: every NDC came back as exactly (0, 0),
    ``on_screen`` was vacuously true for every candidate, and the framing sweep
    was silently scoring occlusion alone.  The symptom was a "pair spread"
    metric that measured 0.0000 at every azimuth, which is what exposed it.

    The ray cast ignores the bodies belonging to whatever is being looked at,
    because a ray to a point on somebody's centreline necessarily strikes their
    own surface first.
    """
    delta = target - eye
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return False, False, 0.0, 0.0

    forward = np.asarray(look_dir, dtype=np.float64)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-9:
        return False, False, 0.0, 0.0
    right /= float(np.linalg.norm(right))
    up = np.cross(right, forward)

    tan_v = math.tan(math.radians(fovy_deg) * 0.5)
    tan_h = tan_v * aspect
    depth = float(delta @ forward)
    if depth <= 0.0:
        return False, False, 0.0, 0.0
    ndc_x = float(delta @ right) / (depth * tan_h)
    ndc_y = float(delta @ up) / (depth * tan_v)
    on_screen = abs(ndc_x) <= 1.0 and abs(ndc_y) <= 1.0

    direction = delta / distance
    geom_id = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, eye, direction, None, 1, -1, geom_id)
    unoccluded = True
    if geom_id[0] >= 0 and hit >= 0.0:
        body = int(model.geom_bodyid[int(geom_id[0])])
        if body not in ignore_bodies and hit < distance - 0.05:
            unoccluded = False
    return on_screen, unoccluded, ndc_x, ndc_y


def body_subtree(model, root):
    bodies = {root}
    for body in range(model.nbody):
        parent = body
        while parent > 0:
            if parent == root:
                bodies.add(body)
                break
            parent = int(model.body_parentid[parent])
    return bodies


def score(model, data, records, azimuth, elevation, near, far, sep_far,
          lookat_z, ease, width, height, fovy=45.0, stride=10):
    """Replay the trace through the renderer's own easing and score it."""
    trunk = model.body("trunk_base").id
    duck_bodies = body_subtree(model, trunk)
    aspect = width / height

    lookat = np.array([records[0]["duck_xy"][0], records[0]["duck_xy"][1],
                       lookat_z])
    distance = near
    counts = {"duck_on": 0, "duck_clear": 0, "subject_on": 0,
              "subject_clear": 0, "eye_inside": 0, "duck_clear_hud": 0,
              "samples": 0}
    duck_px = []
    separations: list[float] = []

    for index, record in enumerate(records):
        duck = np.array(record["duck_xy"], dtype=np.float64)
        subject_name = record["subject"]
        subject = np.array(record["person_xy"][subject_name], dtype=np.float64)
        target = np.array([0.5 * duck[0] + 0.5 * subject[0],
                           0.5 * duck[1] + 0.5 * subject[1], lookat_z])
        lookat += ease * (target - lookat)
        separation = float(np.linalg.norm(duck - subject))
        wanted = near + (far - near) * min(max(separation / sep_far, 0.0), 1.0)
        distance += ease * (wanted - distance)
        if index % stride:
            continue

        # Pose the world exactly as the renderer would see it at this tick.
        t = record["t"]
        pose_people(model, data, people_at(t), t)
        pose_leaves(model, data, doors_at(t))
        data.qpos[0], data.qpos[1] = duck
        mujoco.mj_forward(model, data)

        elev = math.radians(elevation)
        head = math.radians(azimuth)
        forward = np.array([math.cos(elev) * math.cos(head),
                            math.cos(elev) * math.sin(head),
                            math.sin(elev)])
        eye = lookat - forward * distance

        counts["samples"] += 1
        if (abs(eye[0]) <= FLOOR_HALF[0] + 2.5
                and abs(eye[1]) <= FLOOR_HALF[1] + 2.5):
            counts["eye_inside"] += 1

        duck_target = np.array([duck[0], duck[1], 0.14])
        on, clear, ndc_x, ndc_y = visible_point(
            model, data, eye, forward, duck_target, fovy, aspect, duck_bodies)
        counts["duck_on"] += int(on)
        counts["duck_clear"] += int(on and clear)
        if on:
            sx = 0.5 * (ndc_x + 1.0)
            sy = 0.5 * (1.0 - ndc_y)
            if (HUD_LEFT_FRAC < sx < HUD_RIGHT_FRAC
                    and HUD_TOP_FRAC < sy < HUD_BOTTOM_FRAC):
                counts["duck_clear_hud"] += 1
            duck_px.append((sx, sy))

        spec = BY_NAME[subject_name]
        person_body = model.body(f"person_{subject_name}").id
        subject_target = np.array([subject[0], subject[1],
                                   spec.origin_z + 0.16 * spec.stature])
        on_s, clear_s, sub_x, sub_y = visible_point(
            model, data, eye, forward, subject_target, fovy, aspect,
            body_subtree(model, person_body))
        counts["subject_on"] += int(on_s)
        counts["subject_clear"] += int(on_s and clear_s)
        if on and on_s:
            # HORIZONTAL separation of the two bodies on screen, in frame
            # widths.  This is what picks the azimuth: the camera tracks the
            # duck, so the duck sits near frame centre at every azimuth and any
            # measure of ITS screen position is constant by construction.  What
            # does vary is where the OTHER body falls - an azimuth aligned with
            # the route stacks them front-to-back and they overlap; one across
            # the route spreads them apart and both stay readable.
            separations.append(abs(0.5 * (ndc_x - sub_x)))

    samples = max(counts["samples"], 1)
    # HOW FAR APART THE PAIR SITS ON SCREEN.  See the accumulation above: this
    # is what picks the AZIMUTH, because visibility does not.
    spread = float(np.mean(separations)) if separations else 0.0
    return {
        "pair_spread": spread,
        "azimuth": azimuth, "elevation": elevation,
        "near": near, "far": far, "lookat_z": lookat_z,
        "duck_on": counts["duck_on"] / samples,
        "duck_clear": counts["duck_clear"] / samples,
        "subject_on": counts["subject_on"] / samples,
        "subject_clear": counts["subject_clear"] / samples,
        "eye_inside": counts["eye_inside"] / samples,
        "duck_clear_hud": counts["duck_clear_hud"] / samples,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    records = json.loads(Path(args.trace).read_text())
    model = load_scene()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    print(f"replaying {len(records)} recorded ticks, every {args.stride}")
    results = []
    # THE SWEEP WAS EXTENDED ONCE, BECAUSE THE FIRST ANSWER SAT ON ITS EDGE.
    # A grid whose optimum is at a boundary has not found an optimum, it has
    # found the edge of the grid.  The first pass topped out at azimuth 150 and
    # elevation -58, both extremes, so both were extended until the best result
    # sits strictly inside the range.
    for azimuth in (30.0, 60.0, 90.0, 120.0, 150.0, 180.0,
                    210.0, 240.0, 270.0, 300.0, 330.0):
        for elevation in (-34.0, -42.0, -50.0, -58.0, -66.0):
            for near, far in ((3.60, 4.80), (3.90, 5.20)):
                results.append(score(
                    model, data, records, azimuth, elevation, near, far,
                    3.20, 0.58, 0.045, args.width, args.height,
                    stride=args.stride))

    # THE TIE-BREAK IS AS IMPORTANT AS THE SCORE, BECAUSE THE SCORE SATURATES.
    # Once the camera is high enough to clear the partitions and the cabin
    # walls, dozens of candidates all reach duck-clear 1.000 and subject-clear
    # 0.993, and picking the arithmetic maximum among them lands on the STEEPEST
    # and most extreme azimuth in the grid - a near-top-down shot in which the
    # building has no depth and the duck is a dot.
    #
    # So visibility is rounded to 2 dp to form a plateau, and within that
    # plateau the SHALLOWEST elevation wins: the lowest camera that still sees
    # everything is the one that shows the most building.  Distance breaks any
    # remaining tie towards the nearer shot, which makes the duck larger.
    def key(entry):
        return (round(entry["duck_clear"], 2),
                round(entry["subject_clear"], 2),
                round(entry["duck_clear_hud"], 2),
                entry["elevation"],              # shallowest that still sees all
                round(entry["pair_spread"], 2),  # the pair does not overlap
                -entry["near"])                  # nearer makes the duck larger

    results.sort(key=key, reverse=True)
    print()
    print(f"  {'az':>6} {'el':>6} {'near':>5} {'far':>5} "
          f"{'duckOn':>7} {'duckClr':>8} {'subjOn':>7} {'subjClr':>8} "
          f"{'eyeIn':>6} {'noHUD':>6} {'spread':>7}")
    for entry in results[:args.top]:
        print(f"  {entry['azimuth']:6.0f} {entry['elevation']:6.0f} "
              f"{entry['near']:5.2f} {entry['far']:5.2f} "
              f"{entry['duck_on']:7.3f} {entry['duck_clear']:8.3f} "
              f"{entry['subject_on']:7.3f} {entry['subject_clear']:8.3f} "
              f"{entry['eye_inside']:6.3f} {entry['duck_clear_hud']:6.3f} "
              f"{entry['pair_spread']:7.3f}")
    best = results[0]
    print()
    print("BEST:")
    print(f"  CAM_AZIMUTH = {best['azimuth']:.1f}")
    print(f"  CAM_ELEVATION = {best['elevation']:.1f}")
    print(f"  CAM_DISTANCE_NEAR = {best['near']:.2f}")
    print(f"  CAM_DISTANCE_FAR = {best['far']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
