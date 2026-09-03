#!/usr/bin/env python3
"""Measure THIS scene's locomotion constants with short no-render rollouts.

Nothing in this behavior may quote a speed, a yaw rate or a drift figure that
this script did not produce ON THIS SCENE with THIS model.  The sibling
behaviors' numbers are a starting point for what to sweep, never an answer.

Run:
    ../../microduck_rl/.venv/bin/python tools/sweep_commands.py
    ../../microduck_rl/.venv/bin/python tools/sweep_commands.py --what yaw
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
    PolicyRunner,
    load_scene,
)

REPO = Path(__file__).resolve().parents[1]


def run(runner, model, data, twist, seconds: float, decimation: int) -> dict:
    """One open-loop command held for ``seconds``.  Returns measured motion."""
    trunk = model.body("trunk_base").id
    start = data.xpos[trunk][:2].copy()
    start_yaw = runner.yaw(data)
    min_z = float(data.xpos[trunk][2])
    path = 0.0
    previous = start.copy()
    for _ in range(int(seconds * CTRL_HZ)):
        runner.step(data, np.asarray(twist, dtype=np.float32))
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        here = data.xpos[trunk][:2].copy()
        path += float(np.linalg.norm(here - previous))
        previous = here
        min_z = min(min_z, float(data.xpos[trunk][2]))
    end = data.xpos[trunk][:2].copy()
    delta_yaw = math.degrees(
        math.atan2(math.sin(runner.yaw(data) - start_yaw),
                   math.cos(runner.yaw(data) - start_yaw)))
    return {
        "net_m": float(np.linalg.norm(end - start)),
        "path_m": path,
        "speed_mps": float(np.linalg.norm(end - start)) / seconds,
        "yaw_deg": delta_yaw,
        "yaw_dps": delta_yaw / seconds,
        "min_z": min_z,
        "final_z": float(data.xpos[trunk][2]),
    }


def fresh(policy, model, seconds_settle: float = 0.6):
    """A settled robot at the STAND keyframe, ready for one measurement."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    runner = policy.reset(model, data)
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    for _ in range(int(seconds_settle * CTRL_HZ)):
        runner.step(data, np.zeros(3, dtype=np.float32))
        for _ in range(decimation):
            mujoco.mj_step(model, data)
    return runner, data, decimation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--what", default="all",
                        choices=("all", "forward", "yaw", "ceiling", "lateral",
                                 "spin", "zero"))
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()

    model = load_scene()
    policy = PolicyRunner(args.policy)

    if args.what in ("all", "forward"):
        print("=" * 78)
        print(f"FORWARD  ({args.seconds:.0f} s per command, vy=0, wz=0)")
        print("=" * 78)
        print(f"  {'vx':>6} {'net m':>8} {'m/s':>8} {'yaw deg':>9} "
              f"{'min z':>7} {'final z':>8}")
        for vx in (0.18, 0.20, 0.22, 0.24, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46):
            runner, data, dec = fresh(policy, model)
            r = run(runner, model, data, (vx, 0.0, 0.0), args.seconds, dec)
            print(f"  {vx:6.2f} {r['net_m']:8.3f} {r['speed_mps']:8.3f} "
                  f"{r['yaw_deg']:9.1f} {r['min_z']:7.4f} {r['final_z']:8.4f}")

    if args.what in ("all", "yaw"):
        print()
        print("=" * 78)
        print("YAW WHILE WALKING  (3 s per command)")
        print("=" * 78)
        print(f"  {'vx':>6} {'wz':>6} {'net m':>8} {'yaw deg':>9} "
              f"{'deg/s':>8} {'min z':>7}")
        for vx in (0.26, 0.34, 0.42):
            for wz in (-0.34, -0.22, -0.16, -0.10, 0.10, 0.16, 0.22, 0.34):
                runner, data, dec = fresh(policy, model)
                r = run(runner, model, data, (vx, 0.0, wz), 3.0, dec)
                print(f"  {vx:6.2f} {wz:6.2f} {r['net_m']:8.3f} "
                      f"{r['yaw_deg']:9.1f} {r['yaw_dps']:8.1f} "
                      f"{r['min_z']:7.4f}")

    if args.what in ("all", "ceiling"):
        print()
        print("=" * 78)
        print("YAW CEILING  (3 s) - how far out does the axis keep responding?")
        print("=" * 78)
        print(f"  {'vx':>6} {'wz':>6} {'net m':>8} {'yaw deg':>9} "
              f"{'deg/s':>8} {'radius m':>9} {'min z':>7}")
        for vx in (0.30, 0.34):
            for wz in (-0.75, -0.68, -0.58, -0.50, -0.42, 0.42, 0.50, 0.58,
                       0.68, 0.75):
                runner, data, dec = fresh(policy, model)
                r = run(runner, model, data, (vx, 0.0, wz), 3.0, dec)
                rate = abs(r["yaw_dps"])
                radius = (r["path_m"] / 3.0) / math.radians(max(rate, 1e-6))
                print(f"  {vx:6.2f} {wz:6.2f} {r['net_m']:8.3f} "
                      f"{r['yaw_deg']:9.1f} {r['yaw_dps']:8.1f} "
                      f"{radius:9.3f} {r['min_z']:7.4f}")

    if args.what in ("all", "lateral"):
        print()
        print("=" * 78)
        print("LATERAL BUDGET  - THE measurement this behavior turns on.")
        print("There is no strafe on this policy, so every sidestep is a")
        print("turn-out / run / turn-back.  How much |dy| does that buy, and")
        print("how much |dx| does it cost?")
        print("=" * 78)
        print(f"  {'wz':>6} {'out s':>6} {'dx m':>7} {'dy m':>7} "
              f"{'yaw end':>8} {'secs':>6} {'min z':>7}")
        for wz in (-0.42, -0.58, 0.42, 0.58):
            for out_s in (1.2, 1.8, 2.4):
                runner, data, dec = fresh(policy, model)
                trunk = model.body("trunk_base").id
                start = data.xpos[trunk][:2].copy()
                start_yaw = runner.yaw(data)
                # turn out, run straight, turn back the same amount
                run(runner, model, data, (0.34, 0.0, wz), out_s, dec)
                run(runner, model, data, (0.34, 0.0, 0.0), 1.0, dec)
                r = run(runner, model, data, (0.34, 0.0, -wz), out_s, dec)
                end = data.xpos[trunk][:2].copy()
                delta = end - start
                # express in the START frame, so dy is true lateral travel
                c, s = math.cos(-start_yaw), math.sin(-start_yaw)
                dx = float(delta[0] * c - delta[1] * s)
                dy = float(delta[0] * s + delta[1] * c)
                yaw_end = math.degrees(math.atan2(
                    math.sin(runner.yaw(data) - start_yaw),
                    math.cos(runner.yaw(data) - start_yaw)))
                print(f"  {wz:6.2f} {out_s:6.1f} {dx:7.3f} {dy:7.3f} "
                      f"{yaw_end:8.1f} {2 * out_s + 1.0:6.1f} "
                      f"{r['min_z']:7.4f}")

    if args.what in ("all", "spin"):
        print()
        print("=" * 78)
        print("TURN IN PLACE  (vx=0, 3 s per command) - why the cabin is a through-car")
        print("=" * 78)
        print(f"  {'wz':>6} {'drift m':>9} {'yaw deg':>9} {'deg/s':>8} "
              f"{'min z':>7}")
        for wz in (-0.42, -0.34, -0.30, -0.22, -0.16, 0.16, 0.22, 0.30, 0.34,
                   0.42):
            runner, data, dec = fresh(policy, model)
            r = run(runner, model, data, (0.0, 0.0, wz), 3.0, dec)
            print(f"  {wz:6.2f} {r['net_m']:9.4f} {r['yaw_deg']:9.1f} "
                  f"{r['yaw_dps']:8.1f} {r['min_z']:7.4f}")

    if args.what in ("all", "zero"):
        print()
        print("=" * 78)
        print("EXACT ZERO  - the measurement every yield and the RIDE rest on")
        print("=" * 78)
        for seconds in (3.0, 10.0):
            runner, data, dec = fresh(policy, model)
            r = run(runner, model, data, (0.0, 0.0, 0.0), seconds, dec)
            print(f"  {seconds:5.1f} s of exact zero: drift {r['net_m']:.4f} m, "
                  f"path {r['path_m']:.4f} m, yaw {r['yaw_deg']:+.2f} deg, "
                  f"min z {r['min_z']:.4f}, final z {r['final_z']:.4f}")
        print()
        print("COAST AFTER STOPPING  (walk 4 s, then exact zero for 1.5 s)")
        for vx in (0.30, 0.42):
            runner, data, dec = fresh(policy, model)
            run(runner, model, data, (vx, 0.0, 0.0), 4.0, dec)
            r = run(runner, model, data, (0.0, 0.0, 0.0), 1.5, dec)
            print(f"  from vx={vx:.2f}: coasted {r['net_m']:.4f} m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
