#!/usr/bin/env python3
"""Measure every template's feature envelope over its REAL held animation.

``check_gestures.py`` samples nine instants and asks whether each classifies.
That is necessary but not sufficient: a gesture is acted on only after it holds
for the whole CONFIRM window, so what actually matters is the WORST tick over
the hold, not a sample of good ones.  This tool sweeps every control tick of
each template's hold and reports, per feature, the measured range and the worst
per-feature margin - which is what a window edge and its margin must be set
from.

It exists because of a real failure.  The first run classified COME correctly at
most instants and still never confirmed it: at the top of each beckon the
extension reached 0.948 against a ceiling of 0.97 with a margin of 0.10, scoring
0.22, and the confirm window reset twice per second forever.  A nine-sample
check could not see that; this does.

Run:
    ../../microduck_rl/.venv/bin/python tools/probe_templates.py
    ../../microduck_rl/.venv/bin/python tools/probe_templates.py --window 1.0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gest_arm import ANIMATED, JOINT_KEYS, RAISE_S, arm_targets  # noqa: E402
from gest_cast import BY_NAME  # noqa: E402
from gest_gesture import (  # noqa: E402
    BY_TEMPLATE,
    MIN_CONFIDENCE,
    MOTION_WINDOW_S,
    classify,
    score,
)
from gest_pose import measure_body  # noqa: E402
from policy_runtime import CTRL_HZ, load_scene  # noqa: E402

# Which template each ANIMATION must satisfy.  The two names differ - the
# animation says which ARM, the template says which SIDE - and conflating them
# was a real bug (see ``gest_script.TEMPLATE_FOR_GESTURE``).
TEMPLATE_FOR = {
    "COME": "COME",
    "STOP": "STOP",
    "POINT_L_ARM": "POINT_LEFT_ARM",
    "POINT_R_ARM": "POINT_RIGHT_ARM",
    "BACK_UP": "BACK_UP",
    "WAVE": "WAVE",
    "PARTIAL": "",
}


class Poser:
    """Drives one person's arms on the real compiled model and reads keypoints."""

    def __init__(self, person: str = "mira", yaw_deg: float = -90.0):
        self.model = load_scene()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data,
                                    self.model.key("STAND").id)
        mujoco.mj_forward(self.model, self.data)
        self.person = person
        self.yaw = math.radians(yaw_deg)
        mocap = int(self.model.body_mocapid[
            self.model.body(f"actor_{person}").id])
        self.data.mocap_pos[mocap] = (0.0, 1.30, 0.36)
        self.data.mocap_quat[mocap] = np.array(
            [math.cos(self.yaw / 2.0), 0.0, 0.0, math.sin(self.yaw / 2.0)])
        self.joints = {k: self.model.joint(f"{person}_{k}").id
                       for k in JOINT_KEYS}

    def pose(self, gesture: str, elapsed: float, span: float) -> dict:
        targets, _ = arm_targets(gesture, elapsed, span, 0.0)
        for key, joint in self.joints.items():
            self.data.qpos[int(self.model.jnt_qposadr[joint])] = targets[key]
        mujoco.mj_forward(self.model, self.data)
        out = {}
        for side in ("l", "r"):
            out[f"{side}_shoulder"] = self.data.xpos[
                self.model.body(f"{self.person}_shoulder_{side}").id].copy()
            out[f"{side}_elbow"] = self.data.xpos[
                self.model.body(f"{self.person}_fore_{side}").id].copy()
            out[f"{side}_hand"] = self.data.xpos[
                self.model.body(f"{self.person}_hand_{side}").id].copy()
        return out


def motion_features(poser: Poser, gesture: str, elapsed: float, span: float,
                    window_s: float, dt: float) -> tuple[float, float]:
    """Hand PATH and WANDER over the window ending at ``elapsed``.

    Accumulated by re-posing the SAME arm at each intermediate instant, which is
    exactly what the runtime detector accumulates tick by tick from its own
    observation history.
    """
    steps = int(round(window_s / dt))
    frames = [poser.pose(gesture, max(elapsed - window_s + i * dt, 0.0), span)
              for i in range(steps + 1)]
    best_path, best_net = 0.0, 0.0
    for side in ("l", "r"):
        track = [f[f"{side}_hand"] for f in frames]
        path = sum(float(np.linalg.norm(b - a))
                   for a, b in zip(track, track[1:]))
        if path >= best_path:
            best_path = path
            best_net = float(np.linalg.norm(track[-1] - track[0]))
    return (best_path / BY_NAME[poser.person].arm_span,
            best_path / max(best_net, 1e-6))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=float, default=MOTION_WINDOW_S,
                        help="motion window in seconds")
    parser.add_argument("--hold", type=float, default=4.0,
                        help="how much of the hold to sweep")
    parser.add_argument("--span", type=float, default=6.4)
    args = parser.parse_args()

    poser = Poser()
    dt = 1.0 / CTRL_HZ
    failures = 0

    print("=" * 102)
    print(f"TEMPLATE ENVELOPES OVER THE WHOLE HOLD  "
          f"(motion window {args.window:.2f}s, {int(args.hold / dt)} ticks each)")
    print("the WORST STEADY tick is what the confirm window is graded on")
    print()
    print("STEADY means the motion window lies entirely inside the hold.  The")
    print("first window's worth of ticks after the arm arrives still contains")
    print("the RAISE, whose hand path is one-way, so a 'moving' template is")
    print("correctly unrecognised there - that is the wander rule working, not")
    print("a failure.  A gesture is acted on only after CONFIRM_S of sustained")
    print("reading, which begins after that transient either way.")
    print("=" * 102)

    for gesture in ANIMATED:
        template_name = TEMPLATE_FOR[gesture]
        template = BY_TEMPLATE.get(template_name)
        rows = []
        for index in range(int(args.hold / dt)):
            elapsed = RAISE_S + index * dt
            travel, wander = motion_features(
                poser, gesture, elapsed, args.span, args.window, dt)
            points = poser.pose(gesture, elapsed, args.span)
            pose = measure_body(poser.person, poser.yaw, points,
                                {"l": True, "r": True})
            reading = classify(pose, travel, wander)
            value = (score(template, pose, travel, wander)
                     if template is not None else 0.0)
            arms = pose.raised_arms
            rows.append({
                "score": value, "travel": travel, "wander": wander,
                "accepted": reading.accepted, "template": reading.template,
                "arms": pose.raised_count,
                # A tick is STEADY once the whole motion window sits inside the
                # hold rather than straddling the raise that led into it.
                "steady": index * dt >= args.window,
                "ext": [a.extension for a in arms],
                "elev": [a.elevation_deg for a in arms],
                "fwd": [a.forward for a in arms],
                "lat": [a.lateral for a in arms],
            })

        steady = [r for r in rows if r["steady"]]
        scores = np.array([r["score"] for r in steady])
        print()
        print(f"-- {gesture} -> {template_name or '(must match nothing)'} "
              + "-" * max(0, 70 - len(gesture) - len(template_name)))
        for label, key in (("extension", "ext"), ("elevation", "elev"),
                           ("forward", "fwd"), ("lateral", "lat")):
            values = [v for r in steady for v in r[key]]
            if values:
                print(f"   {label:<10} {min(values):9.4f} .. {max(values):9.4f}")
        print(f"   {'travel':<10} {min(r['travel'] for r in steady):9.4f} .. "
              f"{max(r['travel'] for r in steady):9.4f}")
        print(f"   {'wander':<10} {min(r['wander'] for r in steady):9.4f} .. "
              f"{max(r['wander'] for r in steady):9.4f}")

        if template is None:
            # The refusing pose is graded over EVERY tick, transient included:
            # there is no instant at which it may be accepted.
            accepted = sum(1 for r in rows if r["accepted"])
            ok = accepted == 0
            print(f"   score      always 0 by design; accepted on {accepted} "
                  f"of {len(rows)} ticks (transient included)")
            print(f"   => {'OK  ' if ok else 'FAIL'} "
                  + ("refused on every tick of the hold" if ok
                     else f"WRONGLY accepted on {accepted} ticks"))
        else:
            wrong = sum(1 for r in rows
                        if r["accepted"] and r["template"] != template_name)
            ok = bool(scores.min() >= MIN_CONFIDENCE and wrong == 0)
            print(f"   score      min {scores.min():.4f}  p05 "
                  f"{np.percentile(scores, 5):.4f}  median "
                  f"{np.median(scores):.4f}  (bar {MIN_CONFIDENCE:.2f}, "
                  f"{len(steady)} steady ticks)")
            print(f"   => {'OK  ' if ok else 'FAIL'} "
                  + ("holds above the bar on every steady tick, 0 mis-reads"
                     if ok else
                     f"worst steady tick {scores.min():.4f} < "
                     f"{MIN_CONFIDENCE:.2f}"
                     f" or {wrong} ticks read as another template"))
        failures += 0 if ok else 1

    print()
    print("=" * 102)
    print("EVERY TEMPLATE HOLDS ITS OWN WINDOW FOR THE WHOLE STEADY HOLD"
          if not failures else
          f"{failures} TEMPLATE(S) DO NOT HOLD THEIR WINDOW FOR THE WHOLE HOLD")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
