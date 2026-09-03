#!/usr/bin/env python3
"""Second sweep: the yaw dead band, lateral authority, and the head's reach.

``sweep_commands.py`` located gait onset and the yaw asymmetry.  This tool
answers the three questions THIS behavior turns on, which that sweep does not
cover:

1. **Where is the yaw dead band?**  The controller must emit either a command
   that turns the duck or exact zero, never a number in between that shows up in
   the HUD and does nothing on the floor.

2. **Can the duck turn while standing still?**  Measured answer: no — at
   ``vx = 0`` even ``wz = ±0.55`` yields about 2 °/s, which over a whole search
   would be a handful of degrees.  This is not a limitation to work around; it
   is *why* the search sweep has to be done with the head while the locomotion
   command is exactly zero, which is also what the acceptance gate demands.

3. **How far can the head actually look?**  The scan's amplitude has to come
   from the model's own joint range, not from a number that looked reasonable.

Run with:  ../../microduck_rl/.venv/bin/python tools/measure_scan.py
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
    HEAD_PITCH_ACT,
    HEAD_YAW_ACT,
    PolicyRunner,
    load_scene,
)
from sweep_commands import rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO / "onnx" / "alpha_walking.onnx"))
    args = parser.parse_args()

    model = load_scene()
    policy = PolicyRunner(args.policy)

    print("=" * 88)
    print("YAW DEAD BAND at the two speeds the controller actually uses")
    print("=" * 88)
    print(f"{'vx':>6} {'wz':>6} {'yaw°/3s':>9} {'°/s':>8} {'net m':>8}")
    for vx in (0.30, 0.42):
        for wz in (-0.20, -0.14, -0.10, -0.06, 0.06, 0.10, 0.14, 0.20):
            r = rollout(model, policy, (vx, 0.0, wz), 3.0)
            print(f"{vx:6.2f} {wz:6.2f} {r['yaw_deg']:9.1f} "
                  f"{r['yaw_rate_dps']:8.1f} {r['net_m']:8.3f}")

    print()
    print("=" * 88)
    print("TURNING ON THE SPOT — is it possible at all?")
    print("=" * 88)
    for wz in (-0.55, -0.40, 0.40, 0.55):
        r = rollout(model, policy, (0.0, 0.0, wz), 6.0)
        print(f"  vx=0.00 wz={wz:+.2f} -> {r['yaw_deg']:+6.1f}° in 6 s "
              f"({r['yaw_rate_dps']:+.1f} °/s), net {r['net_m']:.3f} m")
    print("  => the body cannot scan; the HEAD must, at exact-zero command.")

    print()
    print("=" * 88)
    print("LATERAL AUTHORITY")
    print("=" * 88)
    print(f"{'vy':>6} {'left m':>8} {'fwd m':>8} {'yaw°':>7}")
    for vy in (-0.40, -0.28, -0.18, 0.18, 0.28, 0.40):
        r = rollout(model, policy, (0.0, vy, 0.0), 4.0)
        print(f"{vy:6.2f} {r['local_left_m']:8.3f} {r['local_fwd_m']:8.3f} "
              f"{r['yaw_deg']:7.1f}")

    print()
    print("=" * 88)
    print("HEAD JOINT RANGE, from the model")
    print("=" * 88)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for label, actuator in (("yaw", HEAD_YAW_ACT), ("pitch", HEAD_PITCH_ACT)):
        joint = int(model.actuator_trnid[actuator, 0])
        low, high = model.jnt_range[joint]
        print(f"  head {label:<5} joint range "
              f"{math.degrees(low):+7.1f}° .. {math.degrees(high):+7.1f}°")

    print()
    print("=" * 88)
    print("HOLD AT EXACT ZERO — drift over a long stationary phase")
    print("=" * 88)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = policy.reset(model, data)
    trunk = model.body("trunk_base").id
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    start = data.xpos[trunk][:2].copy()
    zero = np.zeros(3, dtype=np.float32)
    min_z = float(data.xpos[trunk][2])
    for _ in range(int(10.0 * CTRL_HZ)):
        runner.step(data, zero)
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        min_z = min(min_z, float(data.xpos[trunk][2]))
    drift = float(np.linalg.norm(data.xpos[trunk][:2] - start))
    print(f"  10 s of exact zero from STAND -> drift {drift:.4f} m, "
          f"min z {min_z:.4f}, final z {float(data.xpos[trunk][2]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
