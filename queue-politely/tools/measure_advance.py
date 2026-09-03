#!/usr/bin/env python3
"""Measure the three quantities a QUEUE ADVANCE is made of.

``sweep_commands.py`` answers "does this command walk, and how sharply does it
turn".  It does not answer the questions an advance actually poses, which are
all about starting, stopping and staying on a curve:

1. **Gait-onset latency.**  How far the duck has travelled at each 0.5 s mark
   after a walk command is issued from standstill.  An advance is only 0.55 m
   long, so if the first second is nearly free the controller has to release
   the command earlier than the distance alone suggests.

2. **Coast distance after the command is set to EXACTLY zero.**  Every
   stationary state in this behavior commands exact zero, and the duck must
   come to rest inside a 0.30 m standoff band.  If the coast is a large
   fraction of that band, the controller must brake early - and by how much is
   a measured quantity, not a guess.

3. **Turn radius per (vx, wz) over a SHORT window, per sign.**  The 6 s sweep
   wraps past 180 deg for the strongest commands, which makes the sign
   ambiguous and the radius meaningless.  Three-second windows keep every
   measurement inside one revolution, and the queue's fold radius has to be a
   radius the policy can actually hold.

Usage:

    python tools/measure_advance.py --policy onnx/alpha_walking.onnx \\
        --out /tmp/advance.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene  # noqa: E402

START_XY = (0.30, -1.30)
START_YAW_DEG = 180.0


def _fresh(model, policy, start_xy=START_XY, yaw_deg=START_YAW_DEG):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1] = start_xy
    half = math.radians(yaw_deg) * 0.5
    data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
    runner = policy.reset(model, data)
    return data, runner, model.body("trunk_base").id


def _advance(model, data, runner, trunk, command, steps, decimation):
    for _ in range(steps):
        runner.step(data, np.asarray(command, dtype=np.float32))
        for _ in range(decimation):
            mujoco.mj_step(model, data)
    return data.xpos[trunk][:2].copy()


def onset_profile(model, policy, command, seconds=4.0):
    """Distance from the start point at each 0.5 s mark after release."""
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    data, runner, trunk = _fresh(model, policy)
    start = data.xpos[trunk][:2].copy()
    marks: dict[str, float] = {}
    for half in range(1, int(seconds * 2) + 1):
        _advance(model, data, runner, trunk, command,
                 int(0.5 * CTRL_HZ), decimation)
        marks[f"{0.5 * half:.1f}s"] = round(
            float(np.linalg.norm(data.xpos[trunk][:2] - start)), 4)
    return marks


def coast_distance(model, policy, command, cruise_s=4.0, coast_s=2.5):
    """Extra distance travelled after the command is set to EXACTLY zero."""
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    data, runner, trunk = _fresh(model, policy)
    _advance(model, data, runner, trunk, command,
             int(cruise_s * CTRL_HZ), decimation)
    release = data.xpos[trunk][:2].copy()
    marks: dict[str, float] = {}
    for tenth in range(1, int(coast_s * 10) + 1):
        _advance(model, data, runner, trunk, (0.0, 0.0, 0.0),
                 int(0.1 * CTRL_HZ), decimation)
        marks[f"{0.1 * tenth:.1f}s"] = round(
            float(np.linalg.norm(data.xpos[trunk][:2] - release)), 4)
    return {"command": [float(v) for v in command],
            "coast_profile_m": marks,
            "total_coast_m": marks[f"{coast_s:.1f}s"]}


def turn_radius(model, policy, command, seconds=3.0):
    """Path length, yaw sweep and the implied radius over one short window."""
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    data, runner, trunk = _fresh(model, policy)
    yaw0 = runner.yaw(data)
    previous = data.xpos[trunk][:2].copy()
    path = 0.0
    for _ in range(int(seconds * CTRL_HZ)):
        runner.step(data, np.asarray(command, dtype=np.float32))
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        current = data.xpos[trunk][:2].copy()
        path += float(np.linalg.norm(current - previous))
        previous = current
    sweep = math.degrees(
        (runner.yaw(data) - yaw0 + math.pi) % (2 * math.pi) - math.pi)
    return {
        "command": [float(v) for v in command],
        "seconds": seconds,
        "path_m": round(path, 4),
        "yaw_deg": round(sweep, 2),
        "yaw_rate_dps": round(sweep / seconds, 2),
        "speed_mps": round(path / seconds, 4),
        "turn_radius_m": (round(abs(path / math.radians(sweep)), 4)
                          if abs(sweep) > 2.0 else None),
    }


# The commands the behavior is built from, named here so the report is
# readable and the constants module can quote this file directly.
APPROACH = (0.46, 0.0, 0.0)
ADVANCE = (0.38, 0.0, 0.0)
SETTLE = (0.24, 0.0, 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="onnx/alpha_walking.onnx")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = load_scene(args.scene, args.robot_dir)
    policy = PolicyRunner(args.policy)
    report: dict = {}

    print("GAIT-ONSET LATENCY (distance travelled since release)")
    report["onset"] = {}
    for label, command in (("approach", APPROACH), ("advance", ADVANCE),
                           ("settle", SETTLE)):
        marks = onset_profile(model, policy, command)
        report["onset"][label] = {"command": list(command), "marks_m": marks}
        print(f"  {label:9s} {command}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in marks.items()))

    print("\nCOAST AFTER EXACT ZERO")
    report["coast"] = {}
    for label, command in (("approach", APPROACH), ("advance", ADVANCE),
                           ("settle", SETTLE)):
        entry = coast_distance(model, policy, command)
        report["coast"][label] = entry
        profile = entry["coast_profile_m"]
        print(f"  {label:9s} {command}  total={entry['total_coast_m']:.4f} m  "
              + "  ".join(f"{k}={profile[k]:.3f}"
                          for k in ("0.2s", "0.5s", "1.0s", "2.0s")))

    print("\nTURN RADIUS OVER 3 s, BOTH SIGNS")
    report["turns"] = []
    grid = []
    for vx in (0.28, 0.34, 0.38):
        for wz in (-0.55, -0.42, -0.34, -0.26, -0.18, 0.18, 0.26, 0.34, 0.42):
            grid.append((vx, 0.0, wz))
    header = (f"  {'vx':>6} {'wz':>6} | {'path':>7} {'yaw':>8} {'dps':>7} "
              f"{'m/s':>7} {'R':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for command in grid:
        entry = turn_radius(model, policy, command)
        report["turns"].append(entry)
        radius = entry["turn_radius_m"]
        print(f"  {command[0]:+6.2f} {command[2]:+6.2f} | "
              f"{entry['path_m']:7.4f} {entry['yaw_deg']:+8.2f} "
              f"{entry['yaw_rate_dps']:+7.2f} {entry['speed_mps']:7.4f} "
              f"{(f'{radius:8.3f}' if radius else '       -')}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
