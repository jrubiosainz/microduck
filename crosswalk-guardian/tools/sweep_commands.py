#!/usr/bin/env python3
"""Measure stock-policy locomotion commands on scene_crosswalk_guardian.xml.

Every constant this behavior uses must be MEASURED on THIS scene with THIS
model.  Two properties of the stock walking policy make inheriting numbers from
another behavior unsafe:

* **Forward gait onset is a cliff, not a ramp.**  A command below onset produces
  no motion at all, so "walk slowly across" cannot be expressed by shrinking
  ``vx``.
* **Effective ground speed is far below the commanded ``vx``.**  The crossing
  duration estimate that the whole gap decision depends on is a *measured*
  metres-per-second, not the command value.  Getting that wrong would make the
  conflict predictor confidently wrong.

The grid covers exactly what this behavior needs:

* forward speeds from below onset to the fastest stable command, to fix the
  approach speed and — critically — the crossing speed and its duration;
* forward + small yaw trims in both signs, because the duck must hold a
  straight line down the zebra rather than drifting into a lane;
* pure yaw, to confirm (again, on this scene) that the duck cannot turn in
  place and therefore that every heading is set while walking.

    python tools/sweep_commands.py --policy onnx/alpha_walking.onnx --seconds 6

Prints one row per command: net displacement, body-frame dx/dy, path length,
yaw delta, minimum and final trunk height, plus the derived ground speed.
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

# Where the duck stands for the sweep: on the near pavement, exactly where the
# real rollout starts, so the measured constants come from the same ground.
START_XY = (-1.90, 0.0)


def rollout(model, policy: PolicyRunner, command, seconds: float,
            start_xy=START_XY) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1] = start_xy
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
    return {
        "command": [round(float(v), 3) for v in command],
        "seconds": seconds,
        "net_m": round(float(np.linalg.norm(delta)), 4),
        "body_dx_m": round(float(body_dx), 4),
        "body_dy_m": round(float(body_dy), 4),
        "path_m": round(path, 4),
        # The number the crossing-duration estimate is built on.
        "ground_speed_mps": round(float(np.linalg.norm(delta)) / seconds, 4),
        "forward_speed_mps": round(float(body_dx) / seconds, 4),
        "yaw_deg": round(
            math.degrees((yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi), 2),
        "yaw_rate_dps": round(
            math.degrees((yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi)
            / seconds, 2),
        "min_z": round(min_z, 4),
        "final_z": round(float(data.xpos[trunk][2]), 4),
        "walked": bool(np.linalg.norm(delta) > 0.05),
        "upright": bool(data.xpos[trunk][2] > 0.09 and min_z > 0.09),
    }


DEFAULT_GRID = [
    # forward gait onset, then every speed a crossing could plausibly use
    (0.16, 0.0, 0.0), (0.20, 0.0, 0.0), (0.24, 0.0, 0.0), (0.28, 0.0, 0.0),
    (0.32, 0.0, 0.0), (0.36, 0.0, 0.0), (0.42, 0.0, 0.0), (0.46, 0.0, 0.0),
    (0.52, 0.0, 0.0), (0.58, 0.0, 0.0), (0.65, 0.0, 0.0), (0.75, 0.0, 0.0),
    # turn in place: expected to produce almost nothing, measured to confirm
    (0.0, 0.0, 0.45), (0.0, 0.0, -0.45), (0.0, 0.0, 0.85), (0.0, 0.0, -0.85),
    # heading trims while crossing: the duck must hold the zebra line
    (0.46, 0.0, 0.20), (0.46, 0.0, -0.20),
    (0.46, 0.0, 0.30), (0.46, 0.0, -0.30),
    (0.46, 0.0, 0.45), (0.46, 0.0, -0.45),
    (0.28, 0.0, 0.30), (0.28, 0.0, -0.30),
    (0.28, 0.0, 0.45), (0.28, 0.0, -0.45),
    # lateral trim, as an alternative to yaw for holding the line
    (0.46, 0.10, 0.0), (0.46, -0.10, 0.0),
    (0.46, 0.20, 0.0), (0.46, -0.20, 0.0),
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
              f"{'path':>7} {'m/s':>6} {'yaw':>8} {'°/s':>7} {'minz':>6} "
              f"{'finz':>6}  walked upright")
    print(header)
    print("-" * len(header))
    rows = []
    for command in grid:
        row = rollout(model, policy, command, args.seconds)
        rows.append(row)
        print(f"{command[0]:+6.2f} {command[1]:+6.2f} {command[2]:+6.2f} | "
              f"{row['net_m']:7.4f} {row['body_dx_m']:+7.4f} "
              f"{row['body_dy_m']:+7.4f} {row['path_m']:7.4f} "
              f"{row['ground_speed_mps']:6.3f} {row['yaw_deg']:+8.2f} "
              f"{row['yaw_rate_dps']:+7.2f} {row['min_z']:6.3f} "
              f"{row['final_z']:6.3f}  {'YES' if row['walked'] else 'no ':>6} "
              f"{'YES' if row['upright'] else 'FALL':>7}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
