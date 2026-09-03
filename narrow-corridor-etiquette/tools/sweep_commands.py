#!/usr/bin/env python3
"""Measure stock-policy locomotion commands on scene_narrow_corridor.xml.

Every constant this behavior uses must be MEASURED on THIS scene with THIS
model.  Two properties of the stock walking policy make inheriting numbers from
a sibling behavior unsafe:

* **Gait onset is a cliff, not a ramp**, in every axis.  A command below onset
  produces no motion at all, so "ease gently into the alcove" cannot be
  expressed by shrinking the command.
* **The axes are not symmetric.**  Forward, lateral and yaw all have their own
  onset and their own left/right asymmetry, and the sideways step this behavior
  depends on has never been measured in this lab before.

The grid covers exactly what this behavior needs:

* forward speeds from below onset to a comfortable corridor cruise, to fix the
  cruise speed and the ground speed the encounter predictor is built on;
* **pure lateral commands in both signs**, which is the primitive the whole
  pull-over rests on: the duck must translate into a side recess without
  turning, because turning inside a 0.5 m corridor would swing its nose into a
  wall;
* forward + lateral together, since the pull-over may need to keep making
  progress while stepping aside;
* small yaw trims in both signs, to hold the corridor axis while cruising;
* pure yaw, to confirm (again, on this scene) that the duck cannot turn in
  place, and therefore that a pull-over must be a translation.

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

# Where the duck stands for the sweep: on the corridor centreline near the
# start, exactly where the real rollout begins, so the measured constants come
# from the same ground.
START_XY = (-2.60, 0.0)


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
    max_abs_dy = 0.0
    for _ in range(steps):
        runner.step(data, np.asarray(command, dtype=np.float32))
        for _ in range(decimation):
            mujoco.mj_step(model, data)
            min_z = min(min_z, float(data.xpos[trunk][2]))
        current = data.xpos[trunk][:2].copy()
        path += float(np.linalg.norm(current - previous))
        max_abs_dy = max(max_abs_dy, abs(float(current[1] - start[1])))
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
        "world_dy_m": round(float(delta[1]), 4),
        # The lateral excursion a corridor cruise actually produces, which is
        # what makes a nominal passing gap meaningful or meaningless.
        "max_abs_world_dy_m": round(max_abs_dy, 4),
        "path_m": round(path, 4),
        "ground_speed_mps": round(float(np.linalg.norm(delta)) / seconds, 4),
        "forward_speed_mps": round(float(body_dx) / seconds, 4),
        "lateral_speed_mps": round(float(body_dy) / seconds, 4),
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
    # forward gait onset, then every speed a corridor cruise could use
    (0.16, 0.0, 0.0), (0.20, 0.0, 0.0), (0.24, 0.0, 0.0), (0.28, 0.0, 0.0),
    (0.34, 0.0, 0.0), (0.40, 0.0, 0.0), (0.46, 0.0, 0.0), (0.52, 0.0, 0.0),
    # PURE LATERAL, both signs.  The primitive the pull-over is built on and
    # the one no sibling behavior in this lab has measured.
    (0.0, 0.12, 0.0), (0.0, -0.12, 0.0),
    (0.0, 0.20, 0.0), (0.0, -0.20, 0.0),
    (0.0, 0.28, 0.0), (0.0, -0.28, 0.0),
    (0.0, 0.36, 0.0), (0.0, -0.36, 0.0),
    (0.0, 0.46, 0.0), (0.0, -0.46, 0.0),
    (0.0, 0.60, 0.0), (0.0, -0.60, 0.0),
    # forward + lateral: stepping aside while still making progress
    (0.24, 0.36, 0.0), (0.24, -0.36, 0.0),
    (0.28, 0.46, 0.0), (0.28, -0.46, 0.0),
    # turn in place: expected to produce almost nothing, measured to confirm
    (0.0, 0.0, 0.45), (0.0, 0.0, -0.45), (0.0, 0.0, 0.85), (0.0, 0.0, -0.85),
    # heading trims while cruising: the duck must hold the corridor axis
    (0.28, 0.0, 0.10), (0.28, 0.0, -0.10),
    (0.28, 0.0, 0.20), (0.28, 0.0, -0.20),
    (0.28, 0.0, 0.30), (0.28, 0.0, -0.30),
    (0.34, 0.0, 0.15), (0.34, 0.0, -0.15),
    # lateral with a yaw trim, since the pull-over holds heading too
    (0.0, 0.46, 0.10), (0.0, -0.46, -0.10),
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
              f"{'wdy':>7} {'path':>7} {'m/s':>6} {'lat':>7} {'yaw':>8} "
              f"{'minz':>6} {'finz':>6}  walked upright")
    print(header)
    print("-" * len(header))
    rows = []
    for command in grid:
        row = rollout(model, policy, command, args.seconds)
        rows.append(row)
        print(f"{command[0]:+6.2f} {command[1]:+6.2f} {command[2]:+6.2f} | "
              f"{row['net_m']:7.4f} {row['body_dx_m']:+7.4f} "
              f"{row['body_dy_m']:+7.4f} {row['world_dy_m']:+7.4f} "
              f"{row['path_m']:7.4f} {row['ground_speed_mps']:6.3f} "
              f"{row['lateral_speed_mps']:+7.4f} {row['yaw_deg']:+8.2f} "
              f"{row['min_z']:6.3f} {row['final_z']:6.3f}  "
              f"{'YES' if row['walked'] else 'no ':>6} "
              f"{'YES' if row['upright'] else 'FALL':>7}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
