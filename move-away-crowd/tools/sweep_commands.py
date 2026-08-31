#!/usr/bin/env python3
"""Measure stock-policy locomotion commands on scene_move_away_crowd.xml.

Every constant used by the behavior must be MEASURED on this scene with this
model, because the backward/forward gait has a hard onset threshold: a small
command does not give small motion, it gives NO motion.  PR #22 measured its
constants against ``robot_allcollisions.xml``; this scene is built on
``robot_walk.xml``, so the numbers are re-measured here rather than inherited.

    python tools/sweep_commands.py --policy onnx/alpha_walking.onnx

Prints one row per command: net displacement, path length, yaw delta, minimum
and final trunk height.
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
from policy_runtime import (  # noqa: E402
    ACTION_SCALE,
    CTRL_HZ,
    DEFAULT_POSE,
    PolicyRunner,
    load_scene,
)


def rollout(model, policy: PolicyRunner, command, seconds: float) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = policy.reset(model, data)
    trunk = model.body("trunk_base").id
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    steps = int(seconds * CTRL_HZ)

    start = data.xpos[trunk][:2].copy()
    yaw0 = runner.yaw(data)
    previous = start.copy()
    path = 0.0
    min_z = float(data.xpos[trunk][2])
    for _ in range(steps):
        runner.step(data, np.asarray(command, dtype=np.float32))
        for _ in range(decimation):
            mujoco.mj_step(model, data)
            min_z = min(min_z, float(data.xpos[trunk][2]))
        current = data.xpos[trunk][:2].copy()
        path += float(np.linalg.norm(current - previous))
        previous = current
    end = data.xpos[trunk][:2].copy()
    yaw1 = runner.yaw(data)
    delta = end - start
    # Displacement expressed in the STARTING body frame: +x forward, +y left.
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    body_dx = c * delta[0] - s * delta[1]
    body_dy = s * delta[0] + c * delta[1]
    return {
        "command": [round(float(v), 3) for v in command],
        "seconds": seconds,
        "net_m": round(float(np.linalg.norm(delta)), 4),
        "body_dx_m": round(float(body_dx), 4),
        "body_dy_m": round(float(body_dy), 4),
        "path_m": round(path, 4),
        "yaw_deg": round(math.degrees((yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi), 2),
        "min_z": round(min_z, 4),
        "final_z": round(float(data.xpos[trunk][2]), 4),
        "walked": bool(np.linalg.norm(delta) > 0.05),
        "upright": bool(data.xpos[trunk][2] > 0.09 and min_z > 0.09),
    }


DEFAULT_GRID = [
    # forward gait onset
    (0.10, 0.0, 0.0), (0.16, 0.0, 0.0), (0.20, 0.0, 0.0), (0.24, 0.0, 0.0),
    (0.28, 0.0, 0.0), (0.32, 0.0, 0.0), (0.36, 0.0, 0.0), (0.42, 0.0, 0.0),
    # backward gait onset (PR #22 reported ~-0.32 onset on the other model)
    (-0.24, 0.0, 0.0), (-0.30, 0.0, 0.0), (-0.34, 0.0, 0.0), (-0.36, 0.0, 0.0),
    (-0.40, 0.0, 0.0),
    # pure lateral
    (0.0, 0.15, 0.0), (0.0, 0.22, 0.0), (0.0, 0.30, 0.0), (0.0, 0.38, 0.0),
    (0.0, -0.22, 0.0), (0.0, -0.30, 0.0), (0.0, -0.38, 0.0),
    # turn in place vs turn while walking
    (0.0, 0.0, 0.6), (0.0, 0.0, -0.6),
    (0.24, 0.0, 0.6), (0.24, 0.0, -0.6),
    (0.24, 0.0, 0.35), (0.24, 0.0, -0.35),
    # diagonal escapes
    (0.24, 0.22, 0.0), (0.24, -0.22, 0.0),
    (0.28, 0.30, 0.0), (0.28, -0.30, 0.0),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="onnx/alpha_walking.onnx")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--commands", default=None,
                        help='JSON list of [vx,vy,wz] triples; default = built-in grid')
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = load_scene(args.scene, args.robot_dir)
    policy = PolicyRunner(args.policy)
    grid = json.loads(args.commands) if args.commands else DEFAULT_GRID

    print(f"action_scale={ACTION_SCALE}  ctrl_hz={CTRL_HZ}  "
          f"decimation={round((1.0 / CTRL_HZ) / model.opt.timestep)}  "
          f"seconds={args.seconds}  default_pose_z={DEFAULT_POSE[0]:.3f}")
    header = (f"{'vx':>6} {'vy':>6} {'wz':>6} | {'net':>7} {'dx':>7} {'dy':>7} "
              f"{'path':>7} {'yaw':>8} {'minz':>6} {'finz':>6}  walked upright")
    print(header)
    print("-" * len(header))
    rows = []
    for command in grid:
        row = rollout(model, policy, command, args.seconds)
        rows.append(row)
        print(f"{command[0]:+6.2f} {command[1]:+6.2f} {command[2]:+6.2f} | "
              f"{row['net_m']:7.4f} {row['body_dx_m']:+7.4f} {row['body_dy_m']:+7.4f} "
              f"{row['path_m']:7.4f} {row['yaw_deg']:+8.2f} {row['min_z']:6.3f} "
              f"{row['final_z']:6.3f}  {'YES' if row['walked'] else 'no ':>6} "
              f"{'YES' if row['upright'] else 'FALL':>7}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
