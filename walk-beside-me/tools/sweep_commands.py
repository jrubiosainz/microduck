#!/usr/bin/env python3
"""Measure walking commands on THIS scene with THIS model.

Nothing about the gait is inherited from a sibling behavior.  Gait onset is a
CLIFF rather than a ramp, and the two yaw signs are not symmetric, so a constant
copied from another scene is a guess wearing a measurement's clothes.

Run with:  ../../microduck_rl/.venv/bin/python tools/sweep_commands.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from policy_runtime import (  # noqa: E402
    CTRL_HZ,
    FALLEN_TRUNK_Z,
    PolicyRunner,
    load_scene,
)

REPO = Path(__file__).resolve().parents[1]


def rollout(model, policy, twist, seconds: float) -> dict:
    """One no-render rollout from the STAND keyframe, reporting real motion."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = policy.reset(model, data)
    trunk = model.body("trunk_base").id
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    steps = int(seconds * CTRL_HZ)

    command = np.asarray(twist, dtype=np.float32)
    start = data.xpos[trunk][:2].copy()
    start_yaw = runner.yaw(data)
    min_z = float(data.xpos[trunk][2])
    fallen = 0
    path = 0.0
    previous = start.copy()
    for _ in range(steps):
        runner.step(data, command)
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        here = data.xpos[trunk][:2].copy()
        path += float(np.linalg.norm(here - previous))
        previous = here
        z = float(data.xpos[trunk][2])
        min_z = min(min_z, z)
        if z < FALLEN_TRUNK_Z:
            fallen += 1

    end = data.xpos[trunk][:2].copy()
    end_yaw = runner.yaw(data)
    displacement = end - start
    yaw_delta = math.atan2(math.sin(end_yaw - start_yaw),
                           math.cos(end_yaw - start_yaw))
    c, s = math.cos(start_yaw), math.sin(start_yaw)
    local = np.array([c * displacement[0] + s * displacement[1],
                      -s * displacement[0] + c * displacement[1]])
    return {
        "twist": [float(v) for v in command],
        "net_m": float(np.linalg.norm(displacement)),
        "path_m": path,
        "local_fwd_m": float(local[0]),
        "local_left_m": float(local[1]),
        "yaw_deg": math.degrees(yaw_delta),
        "yaw_rate_dps": math.degrees(yaw_delta) / seconds,
        "speed_mps": float(np.linalg.norm(displacement)) / seconds,
        "min_z": min_z,
        "final_z": float(data.xpos[trunk][2]),
        "fallen_steps": fallen,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--turn-seconds", type=float, default=3.0)
    args = parser.parse_args()

    model = load_scene()
    policy = PolicyRunner(args.policy)

    print("=" * 92)
    print("FORWARD SWEEP — locating gait onset (a cliff, not a ramp)")
    print("=" * 92)
    print(f"{'vx':>6} {'net m':>8} {'path m':>8} {'speed':>8} "
          f"{'fwd m':>8} {'left m':>8} {'yaw°':>7} {'min z':>7} {'fall':>5}")
    for vx in (0.14, 0.18, 0.20, 0.22, 0.24, 0.26, 0.30, 0.34, 0.38, 0.42,
               0.46, 0.52):
        r = rollout(model, policy, (vx, 0.0, 0.0), args.seconds)
        print(f"{vx:6.2f} {r['net_m']:8.3f} {r['path_m']:8.3f} "
              f"{r['speed_mps']:8.3f} {r['local_fwd_m']:8.3f} "
              f"{r['local_left_m']:8.3f} {r['yaw_deg']:7.1f} "
              f"{r['min_z']:7.3f} {r['fallen_steps']:5d}")

    print()
    print("=" * 92)
    print("YAW SWEEP — each sign measured independently")
    print("=" * 92)
    print(f"{'vx':>6} {'wz':>6} {'yaw°':>8} {'°/s':>8} "
          f"{'net m':>8} {'fwd m':>8} {'min z':>7} {'fall':>5}")
    for vx in (0.0, 0.26, 0.34, 0.42):
        for wz in (-0.55, -0.42, -0.30, -0.22, -0.16, -0.10, 0.10, 0.16,
                   0.22, 0.30, 0.42, 0.55):
            r = rollout(model, policy, (vx, 0.0, wz), args.turn_seconds)
            print(f"{vx:6.2f} {wz:6.2f} {r['yaw_deg']:8.1f} "
                  f"{r['yaw_rate_dps']:8.1f} {r['net_m']:8.3f} "
                  f"{r['local_fwd_m']:8.3f} {r['min_z']:7.3f} "
                  f"{r['fallen_steps']:5d}")

    print()
    print("=" * 92)
    print("LATERAL SWEEP — is vy usable on this policy at all?")
    print("=" * 92)
    print(f"{'vx':>6} {'vy':>6} {'left m':>8} {'fwd m':>8} {'yaw°':>8} "
          f"{'min z':>7} {'fall':>5}")
    for vx in (0.0, 0.30):
        for vy in (-0.34, -0.28, -0.22, -0.18, 0.18, 0.22, 0.28, 0.34):
            r = rollout(model, policy, (vx, vy, 0.0), args.seconds)
            print(f"{vx:6.2f} {vy:6.2f} {r['local_left_m']:8.3f} "
                  f"{r['local_fwd_m']:8.3f} {r['yaw_deg']:8.1f} "
                  f"{r['min_z']:7.3f} {r['fallen_steps']:5d}")

    print()
    print("=" * 92)
    print("COAST — distance travelled after the command goes to EXACT zero")
    print("=" * 92)
    for vx in (0.26, 0.34, 0.42):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        runner = policy.reset(model, data)
        trunk = model.body("trunk_base").id
        decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
        for _ in range(int(4.0 * CTRL_HZ)):
            runner.step(data, np.array([vx, 0.0, 0.0], dtype=np.float32))
            for _ in range(decimation):
                mujoco.mj_step(model, data)
        mark = data.xpos[trunk][:2].copy()
        for _ in range(int(1.5 * CTRL_HZ)):
            runner.step(data, np.zeros(3, dtype=np.float32))
            for _ in range(decimation):
                mujoco.mj_step(model, data)
        coast = float(np.linalg.norm(data.xpos[trunk][:2] - mark))
        print(f"  vx={vx:.2f} -> coast {coast:.4f} m in 1.5 s of exact zero")

    print()
    print("=" * 92)
    print("ZERO HOLD — 10 s of exact zero from STAND")
    print("=" * 92)
    r = rollout(model, policy, (0.0, 0.0, 0.0), 10.0)
    print(f"  drift {r['net_m']:.4f} m, min z {r['min_z']:.4f}, "
          f"final z {r['final_z']:.4f}, yaw {r['yaw_deg']:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
