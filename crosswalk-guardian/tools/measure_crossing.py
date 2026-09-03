#!/usr/bin/env python3
"""Measure the crossing primitive that the gap decision is built on.

``conflict.CROSS_SPEED_MPS`` is the single most load-bearing constant in this
behavior: every gap accept/reject compares vehicle arrival times against a
predicted crossing schedule derived from it.  A wrong value would not make the
duck look wrong — it would make the duck confidently take a gap that is not
there.

So it is not guessed, and it is not read off a straight-line command sweep
either.  This tool runs the **exact crossing primitive** used by the rollout —
``GuardianController`` in the ``CROSSING`` state, closed-loop on heading, from
the real kerb stop position — and times the true lane occupancies:

* when the duck's inflated footprint enters and leaves each lane;
* the resulting per-lane crossing schedule, to compare against
  ``conflict.duck_lane_intervals``;
* the net x-speed over the whole crossing, which is the number to paste into
  ``CROSS_SPEED_MPS``;
* lateral drift and yaw excursion, which is what justifies the heading loop.

    python tools/measure_crossing.py --policy onnx/alpha_walking.onnx

Run this again and re-check the constant whenever the crossing command, the
heading gains or the observation change.
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
from conflict import (  # noqa: E402
    CROSS_DURATION_PESSIMISM,
    CROSS_SPEED_MPS,
    VX_CROSS,
    duck_lane_intervals,
)
from contact_geometry import duck_planar_radius  # noqa: E402
from guardian_model import GuardianController  # noqa: E402
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene  # noqa: E402
from street import (  # noqa: E402
    CROSS_GOAL_X,
    CURB_STOP_X,
    DUCK_PLANAR_RADIUS,
    LANE_SPANS,
    ROAD_EXIT_X,
    in_lane,
)


def measure(model, policy: PolicyRunner, *, start_x: float, seconds: float,
            use_controller: bool = True) -> dict:
    """One crossing from ``start_x``, timed lane by lane."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1] = start_x, 0.0
    runner = policy.reset(model, data)
    controller = GuardianController(ctrl_hz=CTRL_HZ)
    trunk = model.body("trunk_base").id
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    steps = int(seconds * CTRL_HZ)
    dt = 1.0 / CTRL_HZ

    lane_enter: dict[str, float | None] = {lane: None for lane in LANE_SPANS}
    lane_exit: dict[str, float | None] = {lane: None for lane in LANE_SPANS}
    start = data.xpos[trunk][:2].copy()
    previous = start.copy()
    path = 0.0
    min_z = float(data.xpos[trunk][2])
    max_abs_y = 0.0
    max_abs_yaw = 0.0
    reached_at: float | None = None
    exit_at: float | None = None

    for index in range(steps):
        t = index * dt
        pos = data.xpos[trunk]
        x, y = float(pos[0]), float(pos[1])
        yaw = runner.yaw(data)
        for lane in LANE_SPANS:
            inside = in_lane(x, lane)
            if inside and lane_enter[lane] is None:
                lane_enter[lane] = t
            if not inside and lane_enter[lane] is not None and lane_exit[lane] is None:
                lane_exit[lane] = t
        if exit_at is None and x >= ROAD_EXIT_X:
            exit_at = t
        if reached_at is None and x >= CROSS_GOAL_X:
            reached_at = t
            break
        command = (controller.update("CROSSING", x, yaw, y)
                   if use_controller
                   else np.array([VX_CROSS, 0.0, 0.0], dtype=np.float32))
        runner.step(data, command)
        for _ in range(decimation):
            mujoco.mj_step(model, data)
            min_z = min(min_z, float(data.xpos[trunk][2]))
        current = data.xpos[trunk][:2].copy()
        path += float(np.linalg.norm(current - previous))
        previous = current
        max_abs_y = max(max_abs_y, abs(float(current[1])))
        max_abs_yaw = max(max_abs_yaw, abs(math.degrees(runner.yaw(data))))

    end = data.xpos[trunk][:2].copy()
    elapsed = reached_at if reached_at is not None else steps * dt
    dx = float(end[0] - start[0])
    return {
        "start_x": start_x,
        "closed_loop": use_controller,
        "reached_goal": reached_at is not None,
        "goal_time_s": reached_at,
        "road_exit_time_s": exit_at,
        "elapsed_s": elapsed,
        "dx_m": round(dx, 4),
        "dy_m": round(float(end[1] - start[1]), 4),
        "path_m": round(path, 4),
        "x_speed_mps": round(dx / elapsed, 4) if elapsed > 0 else 0.0,
        "lane_enter_s": {k: (round(v, 3) if v is not None else None)
                         for k, v in lane_enter.items()},
        "lane_exit_s": {k: (round(v, 3) if v is not None else None)
                        for k, v in lane_exit.items()},
        "max_abs_y_m": round(max_abs_y, 4),
        "max_abs_yaw_deg": round(max_abs_yaw, 2),
        "min_z": round(min_z, 4),
        "final_z": round(float(data.xpos[trunk][2]), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="onnx/alpha_walking.onnx")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = load_scene(args.scene, args.robot_dir)
    policy = PolicyRunner(args.policy)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    radius = duck_planar_radius(model, data, model.body("trunk_base").id)
    print(f"measured duck planar radius = {radius:.4f} m "
          f"(street.DUCK_PLANAR_RADIUS = {DUCK_PLANAR_RADIUS})")
    print(f"crossing command vx={VX_CROSS}, goal x={CROSS_GOAL_X}, "
          f"kerb stop x={CURB_STOP_X}\n")

    rows = []
    for start_x, closed in ((CURB_STOP_X, True), (CURB_STOP_X, False),
                            (CURB_STOP_X - 0.05, True),
                            (CURB_STOP_X + 0.05, True)):
        row = measure(model, policy, start_x=start_x, seconds=args.seconds,
                      use_controller=closed)
        rows.append(row)
        print(f"start_x={start_x:+.3f} closed_loop={str(closed):<5s} "
              f"goal={'YES' if row['reached_goal'] else 'NO '} "
              f"t={row['elapsed_s']:5.2f}s  dx={row['dx_m']:+.3f} "
              f"speed={row['x_speed_mps']:.4f} m/s  "
              f"|dy|max={row['max_abs_y_m']:.3f}  "
              f"|yaw|max={row['max_abs_yaw_deg']:5.1f}°  "
              f"minz={row['min_z']:.3f}")
        print(f"    lane enter {row['lane_enter_s']}  "
              f"exit {row['lane_exit_s']}  road exit "
              f"{row['road_exit_time_s']}")

    closed_rows = [r for r in rows if r["closed_loop"] and r["reached_goal"]]
    if closed_rows:
        speeds = [r["x_speed_mps"] for r in closed_rows]
        slowest = min(speeds)
        print(f"\nclosed-loop x-speeds: {speeds}")
        print(f"slowest = {slowest:.4f} m/s   "
              f"conflict.CROSS_SPEED_MPS = {CROSS_SPEED_MPS}")
        # Compare the PREDICTED lane schedule against the measured one.
        predicted = duck_lane_intervals(CURB_STOP_X)
        base = closed_rows[0]
        print("\nlane schedule, predicted (pessimism "
              f"{CROSS_DURATION_PESSIMISM}) vs measured:")
        for lane in LANE_SPANS:
            window = predicted[lane]
            enter = base["lane_enter_s"][lane]
            exit_ = base["lane_exit_s"][lane]
            print(f"  {lane:>4}: predicted [{window.start:5.2f}, "
                  f"{window.end:5.2f}]  measured [{enter}, {exit_}]")
        print("\nThe predicted window must CONTAIN the measured one, or the "
              "gap decision is optimistic.")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
