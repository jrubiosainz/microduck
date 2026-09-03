#!/usr/bin/env python3
"""Measure stock-policy locomotion commands on scene_queue_politely.xml.

Every constant this behavior uses is MEASURED on THIS scene with THIS model.
Numbers are not inherited from a sibling behavior, because gait onset is a
CLIFF rather than a ramp in every axis and each axis carries its own left/right
asymmetry.

What this behavior specifically needs measured, and why
-------------------------------------------------------
* **A slow, controllable forward creep.**  An advance in a queue is 0.55 m
  long.  At the corridor behavior's cruise speed that is over in two seconds
  with a long coast, and the standoff band is only 0.30 m wide, so the advance
  needs the slowest command that still crosses gait onset.
* **Turn authority WHILE WALKING.**  The queue folds through 180 deg, so the
  duck has to walk an arc.  The required yaw rate is ``v / R``; the fold radius
  in the scene was chosen from what this sweep reports, not the other way
  round.  Both signs are measured independently because they are not symmetric.
* **Coast after the command is set to exactly zero.**  Every stationary state
  in this behavior commands exact zero, and the duck must STOP inside a 0.30 m
  standoff band.  If the coast is a large fraction of that band, the controller
  has to brake early, and by how much is a measured quantity.

    python tools/sweep_commands.py --policy onnx/alpha_walking.onnx --seconds 6

Prints one row per command: net displacement, body-frame dx/dy, path length,
yaw delta, minimum and final trunk height, plus the derived speeds.
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
    PolicyRunner,
    load_scene,
)

# Where the duck stands for the sweep: on the lane's return leg, on the
# straight, exactly the ground the real advances happen on.
START_XY = (0.30, -1.30)


def rollout(model, policy: PolicyRunner, command, seconds: float,
            start_xy=START_XY, start_yaw_deg: float = 180.0) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1] = start_xy
    half = math.radians(start_yaw_deg) * 0.5
    data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
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
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    body_dx = c * delta[0] - s * delta[1]
    body_dy = s * delta[0] + c * delta[1]
    yaw_delta = math.degrees((yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi)
    return {
        "command": [round(float(v), 3) for v in command],
        "seconds": seconds,
        "net_m": round(float(np.linalg.norm(delta)), 4),
        "body_dx_m": round(float(body_dx), 4),
        "body_dy_m": round(float(body_dy), 4),
        "path_m": round(path, 4),
        "ground_speed_mps": round(float(np.linalg.norm(delta)) / seconds, 4),
        "forward_speed_mps": round(float(body_dx) / seconds, 4),
        "lateral_speed_mps": round(float(body_dy) / seconds, 4),
        "yaw_deg": round(yaw_delta, 2),
        "yaw_rate_dps": round(yaw_delta / seconds, 2),
        # The turn radius this command actually produces while walking, which
        # is the quantity the queue's fold has to match.
        "turn_radius_m": (
            round(abs(path / math.radians(yaw_delta)), 4)
            if abs(yaw_delta) > 1.0 else None),
        "min_z": round(min_z, 4),
        "final_z": round(float(data.xpos[trunk][2]), 4),
        "walked": bool(np.linalg.norm(delta) > 0.05),
        "upright": bool(data.xpos[trunk][2] > 0.09 and min_z > 0.09),
    }


DEFAULT_GRID = [
    # Forward gait onset, finely, because an advance is a SLOW manoeuvre and
    # the slowest command that actually walks is the one this behavior wants.
    (0.10, 0.0, 0.0), (0.14, 0.0, 0.0), (0.16, 0.0, 0.0), (0.18, 0.0, 0.0),
    (0.20, 0.0, 0.0), (0.22, 0.0, 0.0), (0.24, 0.0, 0.0), (0.28, 0.0, 0.0),
    (0.32, 0.0, 0.0), (0.38, 0.0, 0.0), (0.46, 0.0, 0.0),
    # Turning while walking: the fold.  Both signs, several rates, at the
    # advance speed and at approach speed.
    (0.20, 0.0, 0.20), (0.20, 0.0, -0.20),
    (0.20, 0.0, 0.30), (0.20, 0.0, -0.30),
    (0.20, 0.0, 0.42), (0.20, 0.0, -0.42),
    (0.20, 0.0, 0.55), (0.20, 0.0, -0.55),
    (0.28, 0.0, 0.30), (0.28, 0.0, -0.30),
    (0.28, 0.0, 0.42), (0.28, 0.0, -0.42),
    (0.28, 0.0, 0.55), (0.28, 0.0, -0.55),
    (0.34, 0.0, 0.42), (0.34, 0.0, -0.42),
    # Small heading trims while creeping, to hold the lane on the straights.
    (0.20, 0.0, 0.10), (0.20, 0.0, -0.10),
    (0.28, 0.0, 0.12), (0.28, 0.0, -0.12),
    # Pure yaw: expected to produce almost nothing; measured to confirm that
    # every heading change in this behavior must be made while walking.
    (0.0, 0.0, 0.45), (0.0, 0.0, -0.45),
    (0.0, 0.0, 0.85), (0.0, 0.0, -0.85),
    # A little lateral, for the final lane-centring nudge on the join.
    (0.0, 0.36, 0.0), (0.0, -0.36, 0.0),
    (0.0, 0.46, 0.0), (0.0, -0.46, 0.0),
    (0.20, 0.30, 0.0), (0.20, -0.30, 0.0),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="onnx/alpha_walking.onnx")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--commands", default=None,
                        help='JSON list of [vx,vy,wz]; default = built-in grid')
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = load_scene(args.scene, args.robot_dir)
    policy = PolicyRunner(args.policy)
    grid = json.loads(args.commands) if args.commands else DEFAULT_GRID

    print(f"action_scale={ACTION_SCALE}  ctrl_hz={CTRL_HZ}  "
          f"decimation={round((1.0 / CTRL_HZ) / model.opt.timestep)}  "
          f"seconds={args.seconds}  start={START_XY}")
    header = (f"{'vx':>6} {'vy':>6} {'wz':>6} | {'net':>7} {'dx':>7} {'dy':>7} "
              f"{'path':>7} {'m/s':>6} {'yaw':>8} {'dps':>7} {'R':>7} "
              f"{'minz':>6} {'finz':>6}  walked upright")
    print(header)
    print("-" * len(header))
    rows = []
    for command in grid:
        row = rollout(model, policy, command, args.seconds)
        rows.append(row)
        radius = row["turn_radius_m"]
        print(f"{command[0]:+6.2f} {command[1]:+6.2f} {command[2]:+6.2f} | "
              f"{row['net_m']:7.4f} {row['body_dx_m']:+7.4f} "
              f"{row['body_dy_m']:+7.4f} {row['path_m']:7.4f} "
              f"{row['ground_speed_mps']:6.3f} {row['yaw_deg']:+8.2f} "
              f"{row['yaw_rate_dps']:+7.2f} "
              f"{(f'{radius:7.3f}' if radius else '      -')} "
              f"{row['min_z']:6.3f} {row['final_z']:6.3f}  "
              f"{'YES' if row['walked'] else 'no ':>6} "
              f"{'YES' if row['upright'] else 'FALL':>7}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
