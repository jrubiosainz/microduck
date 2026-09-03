#!/usr/bin/env python3
"""Measure how long the duck actually takes to walk each leg of its route.

Every time in ``lobby_doors.DOOR_SCHEDULE`` and every ``start_t`` and hold window
in ``etiquette_actors.ROUTES`` has to be pinned to the instant the duck really
arrives somewhere.  Guessing those from the route's arc length and the nominal
cruise speed is wrong by tens of seconds, because the duck slows through every
aperture, eases into every holding point, and loses time to the policy's own
right-hand yaw bias on the straight legs.

So this tool walks the WHOLE route with the real controller and the real policy
and NO state machine at all: every leg is released the instant the previous one
completes.  What it prints is the floor - the fastest the choreography could
possibly need - and the schedule is then built with the yields and waits added
on top of it.

Run:
    ../../microduck_rl/.venv/bin/python tools/measure_legs.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from etiquette_control import LEG_ARRIVED_M, EtiquetteController  # noqa: E402
from etiquette_path import (  # noqa: E402
    LEG_NAMES,
    build_route,
    careful_bands,
    in_careful_band,
    leg_bounds,
)
from etiquette_states import (  # noqa: E402
    DUCK_START_XY,
    DUCK_START_YAW_DEG,
)
from etiquette_tracker import RouteTracker  # noqa: E402
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--seconds", type=float, default=140.0)
    args = parser.parse_args()

    model = load_scene()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1] = DUCK_START_XY
    half = math.radians(DUCK_START_YAW_DEG) * 0.5
    data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]

    policy = PolicyRunner(args.policy)
    runner = policy.reset(model, data)
    trunk = model.body("trunk_base").id
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    dt = 1.0 / CTRL_HZ

    route = build_route()
    bounds = leg_bounds(route)
    bands = careful_bands(route)
    tracker = RouteTracker(route)
    controller = EtiquetteController(ctrl_hz=CTRL_HZ)

    leg = 0
    tracker.set_leg_end(bounds[0])
    arrivals: list[tuple[str, float, float]] = []
    previous_t = 0.0

    print(f"route {route.length:.4f} m, {len(bounds)} legs, "
          f"arrival tolerance {LEG_ARRIVED_M} m")
    print(f"  {'leg':<16} {'ends s (m)':>11} {'len (m)':>8} "
          f"{'arrives (s)':>12} {'leg (s)':>8} {'m/s':>7}")

    for index in range(int(args.seconds * CTRL_HZ)):
        t = index * dt
        duck_xy = data.xpos[trunk][:2].copy()
        duck_yaw = runner.yaw(data)
        tracker.project(duck_xy)

        if tracker.remaining_m <= LEG_ARRIVED_M:
            start_s = 0.0 if leg == 0 else bounds[leg - 1]
            length = bounds[leg] - start_s
            arrivals.append((LEG_NAMES[leg], t, length))
            print(f"  {LEG_NAMES[leg]:<16} {bounds[leg]:11.4f} {length:8.4f} "
                  f"{t:12.2f} {t - previous_t:8.2f} "
                  f"{length / max(t - previous_t, 1e-9):7.4f}")
            previous_t = t
            leg += 1
            if leg >= len(bounds):
                break
            tracker.set_leg_end(bounds[leg])

        careful = in_careful_band(bands, tracker.arc_s)
        command = controller.update(
            "APPROACH_DOOR", duck_xy, duck_yaw,
            target_xy=tracker.pursuit_point(),
            remaining_m=tracker.remaining_m, careful=careful)
        runner.step(data, command)
        for _ in range(decimation):
            mujoco.mj_step(model, data)

    if leg < len(bounds):
        print(f"  DID NOT FINISH: stalled on leg {LEG_NAMES[leg]} at "
              f"arc {tracker.arc_s:.4f} of {bounds[leg]:.4f}")
        return 1

    print()
    print(f"  total walking time {arrivals[-1][1]:.2f} s "
          f"over {route.length:.4f} m")
    print()
    print("  Add on top of this, per the scenario:")
    print("    YIELD_EXITERS      until both exiters are measured clear")
    print("    WAIT_SIDE          until the lift doors begin to open")
    print("    DOORS_OPEN +")
    print("    LET_OCCUPANTS_EXIT until the last occupant is measured clear")
    print("    RIDE               until the rear doors begin to open")
    print("    DOORS_OPEN_TARGET  until the guardian is measured through them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
