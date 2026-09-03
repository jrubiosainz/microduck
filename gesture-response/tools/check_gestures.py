#!/usr/bin/env python3
"""Pose every gesture template on the REAL compiled model and MEASURE it.

This is the tool that set every window edge in ``gest_gesture``.  It drives each
template's joint angles into the real scene, runs ``mj_forward``, reads the
world positions of the six keypoint bodies, and prints the features
``gest_pose`` computes from them - then asks the classifier what it makes of
them.

Two failures it exists to catch, both of which are invisible from the source:

* **A clamped joint.**  A template that asks for an angle outside a joint's
  range silently gets the clamped value, producing a pose the windows were never
  tuned on.  It shows up here as a feature outside its own window.
* **A sign error.**  The animation and the perception sides share nothing but
  the model, so a flipped abduction sign makes POINT_LEFT_ARM classify as
  TURN_LEFT instead of TURN_RIGHT.  Reading the printed lateral component
  catches it immediately; reading the source does not.

Run:
    ../../microduck_rl/.venv/bin/python tools/check_gestures.py
    ../../microduck_rl/.venv/bin/python tools/check_gestures.py --person teo
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gest_arm import (  # noqa: E402
    ANIMATED,
    JOINT_KEYS,
    RAISE_S,
    REST,
    arm_targets,
)
from gest_cast import ALL_NAMES, INSTRUCTOR  # noqa: E402
from gest_gesture import MOTION_WINDOW_S, classify  # noqa: E402
from gest_pose import measure_body  # noqa: E402
from policy_runtime import load_scene  # noqa: E402

KEYPOINTS = ("shoulder", "elbow", "hand")


def keypoints_for(model, data, person: str) -> dict[str, np.ndarray]:
    """The six world keypoints, read from the bodies MuJoCo actually placed."""
    out: dict[str, np.ndarray] = {}
    for side in ("l", "r"):
        out[f"{side}_shoulder"] = data.xpos[
            model.body(f"{person}_shoulder_{side}").id].copy()
        out[f"{side}_elbow"] = data.xpos[
            model.body(f"{person}_fore_{side}").id].copy()
        out[f"{side}_hand"] = data.xpos[
            model.body(f"{person}_hand_{side}").id].copy()
    return out


def place(model, data, person: str, position, yaw: float) -> None:
    body = model.body(f"actor_{person}")
    mocap = int(model.body_mocapid[body.id])
    data.mocap_pos[mocap] = (float(position[0]), float(position[1]),
                             float(position[2]))
    data.mocap_quat[mocap] = np.array(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def pose_arms(model, data, person: str, targets: dict[str, float]) -> None:
    for key in JOINT_KEYS:
        joint = model.joint(f"{person}_{key}").id
        data.qpos[int(model.jnt_qposadr[joint])] = targets[key]
    mujoco.mj_forward(model, data)


def sample(model, data, person: str, gesture: str, elapsed: float,
           span: float, yaw: float, position) -> tuple:
    """Measure one instant of one gesture, and classify it."""
    targets, weight = arm_targets(gesture, elapsed, span, 0.0)
    place(model, data, person, position, yaw)
    pose_arms(model, data, person, targets)
    pose = measure_body(person, yaw, keypoints_for(model, data, person),
                        {"l": True, "r": True})

    # Hand travel over the classifier's own window, accumulated as PATH and as
    # NET displacement by re-posing the SAME arm at each intermediate instant -
    # which is exactly what the runtime detector accumulates tick by tick from
    # its own history.
    #
    # PATH, NOT ENDPOINT DISTANCE, AND THE DIFFERENCE IS THE WHOLE FEATURE.  A
    # beckon at 1.15 Hz sampled 0.60 s apart can land almost exactly where it
    # started: MEASURED, an endpoint version reported 0.07 for a hand that had
    # swept 0.40 of its span and rejected every COME in the run.
    #
    # WANDER = PATH / NET is what separates an oscillation from an arm on its
    # way up, and it is why the still, half-raised PARTIAL pose is refused at
    # the instant its raise completes.
    from policy_runtime import CTRL_HZ
    dt = 1.0 / CTRL_HZ
    steps = max(1, int(round(MOTION_WINDOW_S / dt)))
    travel_m = 0.0
    previous = None
    first = None
    last = None
    for index in range(steps + 1):
        when = elapsed - MOTION_WINDOW_S + index * dt
        stage, _ = arm_targets(gesture, max(when, 0.0), span, 0.0)
        pose_arms(model, data, person, stage)
        here = keypoints_for(model, data, person)
        if previous is not None:
            left = float(np.linalg.norm(here["l_hand"] - previous["l_hand"]))
            right = float(np.linalg.norm(here["r_hand"] - previous["r_hand"]))
            travel_m += max(left, right)
        if first is None:
            first = here
        last = here
        previous = here
    net_m = max(
        float(np.linalg.norm(last["l_hand"] - first["l_hand"])),
        float(np.linalg.norm(last["r_hand"] - first["r_hand"])))
    pose_arms(model, data, person, targets)
    from gest_cast import BY_NAME
    travel = travel_m / BY_NAME[person].arm_span
    wander = travel_m / max(net_m, 1e-6)
    return pose, travel, wander, weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", default=INSTRUCTOR, choices=ALL_NAMES)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()

    model = load_scene()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    mujoco.mj_forward(model, data)

    person = args.person
    yaw = math.radians(-90.0)
    position = (0.0, 1.30, 0.36)
    span = 5.6
    failures = 0

    print("=" * 100)
    print(f"GESTURE TEMPLATES MEASURED ON THE COMPILED MODEL  (person={person})")
    print("all lengths normalised by this person's own arm span")
    print("=" * 100)

    for gesture in (REST,) + ANIMATED:
        print()
        print(f"-- {gesture} " + "-" * (96 - len(gesture)))
        print(f"  {'t_held':>7} {'arms':>5} {'ext':>7} {'elev':>8} "
              f"{'fwd':>7} {'lat':>7} {'path':>7} {'wander':>7} -> "
              f"{'template':<16} {'command':<11} {'conf':>5}")
        readings = []
        for index in range(args.samples):
            held = index * (span - RAISE_S - 0.7) / max(args.samples - 1, 1)
            elapsed = RAISE_S + held
            if gesture == REST:
                elapsed, gesture_span = 0.0, 0.0
            else:
                gesture_span = span
            pose, travel, wander, _ = sample(
                model, data, person, gesture, elapsed, gesture_span, yaw,
                position)
            reading = classify(pose, travel, wander)
            readings.append(reading)
            primary = pose.primary
            if primary is None:
                print(f"  {held:7.2f} {pose.raised_count:5d} "
                      f"{'-':>7} {'-':>8} {'-':>7} {'-':>7} {travel:7.3f} "
                      f"{wander:7.2f} -> "
                      f"{reading.template or '(none)':<16} "
                      f"{reading.command or '-':<11} {reading.confidence:5.2f}")
            else:
                print(f"  {held:7.2f} {pose.raised_count:5d} "
                      f"{primary.extension:7.3f} {primary.elevation_deg:8.1f} "
                      f"{primary.forward:7.3f} {primary.lateral:7.3f} "
                      f"{travel:7.3f} {wander:7.2f} -> "
                      f"{reading.template or '(none)':<16} "
                      f"{reading.command or '-':<11} {reading.confidence:5.2f}")

        accepted = {r.template for r in readings if r.accepted}
        if gesture in ("REST", "PARTIAL"):
            ok = not accepted
            note = ("correctly recognised as nothing" if ok
                    else f"WRONGLY accepted as {sorted(accepted)}")
        else:
            expected = {"POINT_L_ARM": "POINT_LEFT_ARM",
                        "POINT_R_ARM": "POINT_RIGHT_ARM"}.get(gesture, gesture)
            ok = accepted == {expected}
            note = (f"accepted as {expected} in "
                    f"{sum(1 for r in readings if r.accepted)}/{len(readings)} "
                    "samples" if ok else
                    f"EXPECTED {{{expected}}} but got {sorted(accepted)}")
        print(f"  => {'OK  ' if ok else 'FAIL'} {note}")
        failures += 0 if ok else 1

    print()
    print("=" * 100)
    print("ALL TEMPLATES MEASURE AS INTENDED" if not failures
          else f"{failures} TEMPLATE(S) DO NOT MEASURE AS INTENDED")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
