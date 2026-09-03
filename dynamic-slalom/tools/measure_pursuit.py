#!/usr/bin/env python3
"""Measure what the CLOSED-LOOP controller actually achieves, not the open loop.

``tools/sweep_commands.py`` measures open-loop commands: hold ``wz`` for N
seconds and see what happens.  That is the right way to find the gait onset and
the yaw ceiling, but it is the WRONG way to find the duck's lateral rate,
because the duck never holds a fixed ``wz`` — it chases a pursuit point on an
offset line with :class:`slalom_control.SlalomController`.

THE MEASUREMENT THIS FIXES
----------------------------
The open-loop lateral figure is a ROUND TRIP: turn out, run, turn back, and
measure the net sideways displacement.  At the ceiling that is 0.34 m in 5.8 s,
which averages 0.059 m/s.  Deriving the planner's lateral rate from it is wrong
in the pessimistic direction, because half of that manoeuvre is spent turning
BACK — undoing lateral progress the duck wants to keep while it passes.

What the planner actually needs is: **how fast does the duck converge onto an
offset line it is pursuing?**  That is a closed-loop question, and this script
answers it by running the real controller against the real policy.

Run:
    ../../microduck_rl/.venv/bin/python tools/measure_pursuit.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from policy_runtime import CTRL_HZ, PolicyRunner, load_scene  # noqa: E402
from slalom_control import SlalomController  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def fresh(policy, model, settle_s: float = 0.6):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    runner = policy.reset(model, data)
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    for _ in range(int(settle_s * CTRL_HZ)):
        runner.step(data, np.zeros(3, dtype=np.float32))
        for _ in range(decimation):
            mujoco.mj_step(model, data)
    return runner, data, decimation


def chase_offset(policy, model, offset_m: float, seconds: float,
                 lookahead_m: float = 0.40, careful: bool = False) -> dict:
    """Run the REAL controller chasing a line offset ``offset_m`` to the side.

    The duck starts on the lane heading +x and pursues a point ``lookahead_m``
    ahead on the offset line — exactly what ``rollout_slalom`` does during a
    pass.  Reports when it first got within 0.12 m of the line (the controller's
    own ``ON_CORRIDOR_M``) and what that cost in forward travel.
    """
    runner, data, decimation = fresh(policy, model)
    controller = SlalomController(ctrl_hz=CTRL_HZ)
    trunk = model.body("trunk_base").id
    start = data.xpos[trunk][:2].copy()

    reached_s = None
    path = 0.0
    previous = start.copy()
    min_z = float(data.xpos[trunk][2])
    samples: list[tuple[float, float, float]] = []

    for step in range(int(seconds * CTRL_HZ)):
        t = step / CTRL_HZ
        here = data.xpos[trunk][:2].copy()
        yaw = runner.yaw(data)
        # The pursuit point: lookahead ahead in x, on the offset line.
        target = np.array([float(here[0]) + lookahead_m,
                           float(start[1]) + offset_m])
        command = controller.update("PASS", here, yaw, target_xy=target,
                                    remaining_m=1e9, careful=careful)
        runner.step(data, command)
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        now = data.xpos[trunk][:2].copy()
        path += float(np.linalg.norm(now - previous))
        previous = now
        min_z = min(min_z, float(data.xpos[trunk][2]))
        lateral = abs(float(now[1]) - (float(start[1]) + offset_m))
        if reached_s is None and lateral <= 0.12:
            reached_s = t
        if step % 25 == 0:
            samples.append((t, float(now[0] - start[0]),
                            float(now[1] - start[1])))

    end = data.xpos[trunk][:2].copy()
    dy = float(end[1] - start[1])
    dx = float(end[0] - start[0])
    return {
        "offset_m": offset_m,
        "dx_m": dx,
        "dy_m": dy,
        "reached_s": reached_s,
        "lateral_rate_mps": (abs(dy) / reached_s if reached_s else
                             abs(dy) / seconds),
        "path_m": path,
        "min_z": min_z,
        "final_yaw_deg": math.degrees(runner.yaw(data)),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--seconds", type=float, default=9.0)
    args = parser.parse_args()

    model = load_scene()
    policy = PolicyRunner(args.policy)

    print("=" * 84)
    print("CLOSED-LOOP LATERAL CONVERGENCE  (the real controller, real policy)")
    print("=" * 84)
    print(f"  {'offset':>7} {'careful':>8} {'reached':>8} {'dy':>7} "
          f"{'dx':>7} {'rate m/s':>9} {'yaw':>7} {'min z':>7}")
    best: dict[float, float] = {}
    for careful in (False, True):
        for offset in (0.26, -0.26, 0.38, -0.38, 0.50, -0.50):
            r = chase_offset(policy, model, offset, args.seconds,
                             careful=careful)
            reached = ("never" if r["reached_s"] is None
                       else f"{r['reached_s']:.2f}s")
            print(f"  {offset:+7.2f} {str(careful):>8} {reached:>8} "
                  f"{r['dy_m']:+7.3f} {r['dx_m']:+7.3f} "
                  f"{r['lateral_rate_mps']:9.4f} {r['final_yaw_deg']:+7.1f} "
                  f"{r['min_z']:7.4f}")
            if r["reached_s"] is not None:
                key = abs(offset)
                best[key] = max(best.get(key, 0.0), r["lateral_rate_mps"])

    print()
    print("SUSTAINED LATERAL RATE, per offset (worst of the two signs is what")
    print("the planner must assume):")
    for offset in sorted(best):
        print(f"  {offset:.2f} m -> {best[offset]:.4f} m/s")
    if best:
        print(f"\n  PLANNER FIGURE (min across offsets): "
              f"{min(best.values()):.4f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
