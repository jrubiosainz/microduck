#!/usr/bin/env python3
"""Measure the three approach quantities the standoff gate depends on.

``sweep_commands.py`` answers "does this command walk"; it does not answer the
questions a *recall* actually poses, which are all about starting and stopping:

1. **Gait-onset latency.**  How far the duck has travelled at 0.5/1.0/1.5/2.0 s
   after a walk command is issued from standstill.  The approach controller has
   to release the command early enough that the duck is already moving when it
   matters, and the arrival predictor has to know that the first second is
   nearly free.

2. **Coast distance.**  How far the duck keeps travelling after the command is
   set to exactly zero.  The standoff band is only ~0.30 m wide, so a stop
   command issued at the band centre lands outside it if the coast is large.
   The controller must therefore brake at ``band_centre + coast``.

3. **Yaw rate versus wz, per sign, over a short window.**  The 6 s sweep wraps
   past +/-180 deg for the strongest commands, which makes the sign ambiguous.
   3 s windows keep every measurement inside one revolution.

Usage:

    python tools/measure_approach.py --policy onnx/alpha_walking.onnx \\
        --out /tmp/approach.json
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


def _fresh(model, policy):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = policy.reset(model, data)
    return data, runner, model.body("trunk_base").id


def _advance(model, data, runner, command, steps, decimation):
    """Step ``steps`` control ticks at ``command``; return the xy path taken."""
    path = [data.xpos[model.body("trunk_base").id][:2].copy()]
    for _ in range(steps):
        runner.step(data, np.asarray(command, dtype=np.float32))
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        path.append(data.xpos[model.body("trunk_base").id][:2].copy())
    return np.asarray(path)


def onset_profile(model, policy, command, seconds=3.0):
    """Distance from the start point at each 0.5 s mark."""
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    data, runner, trunk = _fresh(model, policy)
    start = data.xpos[trunk][:2].copy()
    marks: dict[str, float] = {}
    for half in range(1, int(seconds * 2) + 1):
        _advance(model, data, runner, command, int(0.5 * CTRL_HZ), decimation)
        marks[f"{0.5 * half:.1f}s"] = round(
            float(np.linalg.norm(data.xpos[trunk][:2] - start)), 4
        )
    return marks


def coast_distance(model, policy, command, cruise_s=4.0, coast_s=2.5):
    """Extra distance travelled after the command is set to exactly zero."""
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    data, runner, trunk = _fresh(model, policy)
    _advance(model, data, runner, command, int(cruise_s * CTRL_HZ), decimation)
    release = data.xpos[trunk][:2].copy()
    zero = np.zeros(3, dtype=np.float32)
    profile: dict[str, float] = {}
    for half in range(1, int(coast_s * 2) + 1):
        _advance(model, data, runner, zero, int(0.5 * CTRL_HZ), decimation)
        profile[f"+{0.5 * half:.1f}s"] = round(
            float(np.linalg.norm(data.xpos[trunk][:2] - release)), 4
        )
    return {
        "command": [round(float(v), 3) for v in command],
        "cruise_s": cruise_s,
        "coast_profile_m": profile,
        "total_coast_m": profile[f"+{coast_s:.1f}s"],
        "final_z": round(float(data.xpos[trunk][2]), 4),
    }


def yaw_rate(model, policy, command, seconds=3.0):
    """Signed yaw delta over a window short enough to avoid +/-180 wrapping."""
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    data, runner, trunk = _fresh(model, policy)
    yaw0 = runner.yaw(data)
    unwrapped = 0.0
    previous = yaw0
    for _ in range(int(seconds * CTRL_HZ)):
        runner.step(data, np.asarray(command, dtype=np.float32))
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        current = runner.yaw(data)
        unwrapped += (current - previous + math.pi) % (2 * math.pi) - math.pi
        previous = current
    start = np.zeros(2)
    return {
        "command": [round(float(v), 3) for v in command],
        "seconds": seconds,
        "yaw_deg": round(math.degrees(unwrapped), 2),
        "yaw_deg_per_s": round(math.degrees(unwrapped) / seconds, 2),
        "net_m": round(float(np.linalg.norm(data.xpos[trunk][:2] - start)), 4),
        "min_ok": bool(data.xpos[trunk][2] > 0.09),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="onnx/alpha_walking.onnx")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = load_scene(args.scene, args.robot_dir)
    policy = PolicyRunner(args.policy)

    report: dict = {"onset": {}, "coast": [], "yaw": []}
    for command in ((0.24, 0.0, 0.0), (0.28, 0.0, 0.0), (0.32, 0.0, 0.0)):
        key = f"vx={command[0]:.2f}"
        report["onset"][key] = onset_profile(model, policy, command)
        print(f"onset {key}: {report['onset'][key]}")

    for command in ((0.24, 0.0, 0.0), (0.28, 0.0, 0.0), (0.28, 0.0, -0.45)):
        row = coast_distance(model, policy, command)
        report["coast"].append(row)
        print(f"coast {row['command']}: total={row['total_coast_m']:.4f} m "
              f"{row['coast_profile_m']}")

    for wz in (0.25, -0.25, 0.45, -0.45, 0.60, -0.60, 0.85, -0.85):
        for vx in (0.24, 0.28):
            row = yaw_rate(model, policy, (vx, 0.0, wz))
            report["yaw"].append(row)
            print(f"yaw vx={vx:.2f} wz={wz:+.2f}: "
                  f"{row['yaw_deg']:+8.2f} deg  ({row['yaw_deg_per_s']:+6.2f} deg/s)"
                  f"  net={row['net_m']:.3f} m")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
