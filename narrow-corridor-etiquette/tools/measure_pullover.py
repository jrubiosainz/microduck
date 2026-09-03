#!/usr/bin/env python3
"""Measure the real pull-over primitive: entry, settle, rejoin, and cruise drift.

Every timing constant the alcove scorer depends on comes from here, run with
the EXACT controller the rollout uses rather than from an open-loop sweep:

* **cruise tracking** — the duck's peak lateral excursion while trying to hold
  the centreline.  This is what justifies ``corridor.SAFE_PASSING_GAP_M``: a
  nominal passing gap smaller than the robot's own tracking error can be closed
  by tracking error alone, so it is not a gap.
* **lateral entry** — how long the closed-loop pull-over actually takes from
  the centreline to a park point on each side, including the gait-onset dead
  time, and the yaw excursion it costs.
* **settle** — how long after the command is released the trunk keeps drifting
  before it is stationary, which is the constant that decides whether YIELD
  begins on a duck that is still coasting.
* **rejoin** — the same measurement in the opposite direction.

    python tools/measure_pullover.py --policy onnx/alpha_walking.onnx

Prints the measured numbers and the constants they must agree with.
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
from contact_geometry import (  # noqa: E402
    duck_planar_radius,
    exact_lateral_half_width,
    exact_planar_radius,
)
from corridor import (  # noqa: E402
    ALCOVES,
    ADULT_LATERAL_HALF,
    DUCK_LATERAL_HALF,
    DUCK_PLANAR_RADIUS,
    REJOIN_TOLERANCE_M,
    SAFE_PASSING_GAP_M,
    START_X,
    corridor_passing_geometry,
)
from encounter import (  # noqa: E402
    APPROACH_SPEED_MPS,
    CRUISE_SPEED_MPS,
    LATERAL_DEAD_TIME_S,
    SETTLE_S,
    VY_SPEED_MPS,
)
from etiquette_model import PARK_TOLERANCE_M, EtiquetteController  # noqa: E402
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene  # noqa: E402


def _make(model, policy, start_x=START_X, start_y=0.0):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1] = start_x, start_y
    runner = policy.reset(model, data)
    return data, runner


def measure_cruise(model, policy, seconds: float = 12.0) -> dict:
    """Closed-loop corridor cruise: forward speed and lateral tracking error."""
    data, runner = _make(model, policy)
    trunk = model.body("trunk_base").id
    controller = EtiquetteController(ctrl_hz=CTRL_HZ)
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    start = data.xpos[trunk][:2].copy()
    worst_y = 0.0
    worst_yaw = 0.0
    for _ in range(int(seconds * CTRL_HZ)):
        pos = data.xpos[trunk]
        command = controller.update(
            "CRUISE", float(pos[0]), float(pos[1]), runner.yaw(data))
        runner.step(data, command)
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        worst_y = max(worst_y, abs(float(data.xpos[trunk][1])))
        worst_yaw = max(worst_yaw, abs(math.degrees(runner.yaw(data))))
    end = data.xpos[trunk][:2].copy()
    return {
        "seconds": seconds,
        "forward_m": float(end[0] - start[0]),
        "forward_speed_mps": float(end[0] - start[0]) / seconds,
        "max_abs_y_m": worst_y,
        "max_abs_yaw_deg": worst_yaw,
        "final_y_m": float(end[1]),
        "final_z_m": float(data.xpos[trunk][2]),
    }


def measure_pull_over(model, policy, alcove_name: str,
                      seconds: float = 14.0) -> dict:
    """Closed-loop entry into one alcove, then the settle, then the rejoin.

    The duck starts on the centreline at the alcove's own station, so the
    measurement isolates the LATERAL leg: the along-corridor approach is a
    plain cruise and is measured separately.
    """
    alcove = {a.name: a for a in ALCOVES}[alcove_name]
    data, runner = _make(model, policy, start_x=alcove.center_x, start_y=0.0)
    trunk = model.body("trunk_base").id
    controller = EtiquetteController(ctrl_hz=CTRL_HZ)
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    dt = 1.0 / CTRL_HZ
    park_y = alcove.park_y

    entry_done = None
    move_started = None
    settle_done = None
    rejoin_started = None
    rejoin_done = None
    max_abs_yaw = 0.0
    previous = data.xpos[trunk][:2].copy()
    path = 0.0
    start_xy = previous.copy()

    for step in range(int(seconds * CTRL_HZ)):
        t = step * dt
        pos = data.xpos[trunk]
        x, y = float(pos[0]), float(pos[1])
        if entry_done is None:
            state = "PULL_OVER"
        elif settle_done is None:
            state = "YIELD"
        else:
            state = "REJOIN"
            if rejoin_started is None:
                rejoin_started = t
        command = controller.update(
            state, x, y, runner.yaw(data),
            park_y=park_y, target_x=alcove.center_x)
        runner.step(data, command)
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        current = data.xpos[trunk][:2].copy()
        speed = float(np.linalg.norm(current - previous)) / dt
        path += float(np.linalg.norm(current - previous))
        max_abs_yaw = max(max_abs_yaw, abs(math.degrees(runner.yaw(data))))
        y_now = float(current[1])

        if move_started is None and abs(y_now - float(start_xy[1])) > 0.01:
            move_started = t
        if entry_done is None and abs(y_now - park_y) <= PARK_TOLERANCE_M:
            entry_done = t
            entry_xy = current.copy()
        elif entry_done is not None and settle_done is None:
            if speed <= 0.05 and t - entry_done >= 0.10:
                settle_done = t
                settle_xy = current.copy()
        elif settle_done is not None and rejoin_done is None:
            if abs(y_now) <= REJOIN_TOLERANCE_M:
                rejoin_done = t
                break
        previous = current

    end = data.xpos[trunk][:2].copy()
    return {
        "alcove": alcove_name,
        "side": alcove.side,
        "park_y": park_y,
        "lateral_distance_m": abs(park_y),
        "dead_time_s": (move_started or 0.0),
        "entry_s": entry_done,
        "entry_after_onset_s": (
            None if entry_done is None or move_started is None
            else entry_done - move_started),
        "lateral_speed_mps": (
            None if entry_done is None or move_started is None
            else abs(park_y) / max(entry_done - move_started, 1e-6)),
        "settle_s": (None if settle_done is None or entry_done is None
                     else settle_done - entry_done),
        "settle_y_m": (None if settle_done is None else float(settle_xy[1])),
        "settle_drift_m": (
            None if settle_done is None or entry_done is None
            else abs(float(settle_xy[1]) - float(entry_xy[1]))),
        "rejoin_s": (None if rejoin_done is None or rejoin_started is None
                     else rejoin_done - rejoin_started),
        "max_abs_yaw_deg": max_abs_yaw,
        "path_m": path,
        "final_xy": [float(end[0]), float(end[1])],
        "final_z_m": float(data.xpos[trunk][2]),
        "x_drift_m": float(end[0] - start_xy[0]),
    }


def measure_approach(model, policy, alcove_name: str, lead_m: float,
                     seconds: float = 20.0) -> dict:
    """Time the WHOLE pull-over from ``lead_m`` before the alcove's mouth.

    This is the measurement the reachability estimate must be built on, because
    the controller drives the forward and lateral axes AT THE SAME TIME.
    Charging the two legs sequentially is not conservatism — it is a different
    manoeuvre, and it rejects alcoves the duck can comfortably reach.

    The duck starts on the centreline ``lead_m`` short of its target station and
    runs the real ``PULL_OVER`` controller until its footprint is parked and
    stationary.
    """
    alcove = {a.name: a for a in ALCOVES}[alcove_name]
    low, high = alcove.x_span
    target_x = max(low + DUCK_PLANAR_RADIUS, min(alcove.center_x, high))
    data, runner = _make(model, policy, start_x=target_x - lead_m, start_y=0.0)
    trunk = model.body("trunk_base").id
    controller = EtiquetteController(ctrl_hz=CTRL_HZ)
    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    dt = 1.0 / CTRL_HZ
    park_y = alcove.park_y

    parked_at = None
    previous = data.xpos[trunk][:2].copy()
    max_abs_yaw = 0.0
    for step in range(int(seconds * CTRL_HZ)):
        t = step * dt
        pos = data.xpos[trunk]
        x, y = float(pos[0]), float(pos[1])
        command = controller.update(
            "PULL_OVER", x, y, runner.yaw(data),
            park_y=park_y, target_x=alcove.center_x)
        runner.step(data, command)
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        current = data.xpos[trunk][:2].copy()
        speed = float(np.linalg.norm(current - previous)) / dt
        max_abs_yaw = max(max_abs_yaw, abs(math.degrees(runner.yaw(data))))
        inside = alcove.footprint_inside(float(current[0]), float(current[1]))
        if (abs(float(current[1]) - park_y) <= PARK_TOLERANCE_M
                and speed <= 0.05):
            parked_at = t
            break
        previous = current

    end = data.xpos[trunk][:2].copy()
    lateral = abs(park_y)
    return {
        "alcove": alcove_name,
        "lead_m": lead_m,
        "target_x": target_x,
        "parked_s": parked_at,
        "forward_leg_s": lead_m / APPROACH_SPEED_MPS,
        "lateral_leg_s": lateral / VY_SPEED_MPS,
        "max_leg_s": max(lead_m / APPROACH_SPEED_MPS,
                         lateral / VY_SPEED_MPS),
        "ratio_to_max_leg": (
            None if parked_at is None
            else parked_at / max(lead_m / APPROACH_SPEED_MPS,
                                 lateral / VY_SPEED_MPS)),
        "final_xy": [float(end[0]), float(end[1])],
        "inside_mouth": alcove.footprint_inside(float(end[0]), float(end[1])),
        "max_abs_yaw_deg": max_abs_yaw,
        "final_z_m": float(data.xpos[trunk][2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="onnx/alpha_walking.onnx")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = load_scene(args.scene, args.robot_dir)
    policy = PolicyRunner(args.policy)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0] = START_X
    mujoco.mj_forward(model, data)
    trunk = model.body("trunk_base").id
    person = model.body("person_chen").id
    data.mocap_pos[int(model.body_mocapid[person])] = [0.0, 0.0, 0.36]
    mujoco.mj_forward(model, data)

    radii = {
        "duck_bounding_planar_radius_m": duck_planar_radius(model, data, trunk),
        "duck_exact_planar_radius_m": exact_planar_radius(model, data, trunk),
        "duck_exact_lateral_half_m": exact_lateral_half_width(
            model, data, trunk),
        "adult_exact_planar_radius_m": exact_planar_radius(model, data, person),
        "adult_exact_lateral_half_m": exact_lateral_half_width(
            model, data, person),
    }
    print("MEASURED RADII")
    for key, value in radii.items():
        print(f"  {key:38s} {value:.4f}")
    print(f"  corridor.DUCK_PLANAR_RADIUS            {DUCK_PLANAR_RADIUS:.4f}")
    print(f"  corridor.DUCK_LATERAL_HALF             {DUCK_LATERAL_HALF:.4f}")
    print(f"  corridor.ADULT_LATERAL_HALF            {ADULT_LATERAL_HALF:.4f}")

    print("\nPASSING GEOMETRY")
    geometry = corridor_passing_geometry()
    for key, value in geometry.items():
        print(f"  {key:38s} {value}")

    print("\nCLOSED-LOOP CRUISE")
    cruise = measure_cruise(model, policy)
    for key, value in cruise.items():
        print(f"  {key:38s} {value}")
    print(f"  corridor.SAFE_PASSING_GAP_M            {SAFE_PASSING_GAP_M:.4f}"
          f"   (must be >= max_abs_y_m)")
    print(f"  encounter.CRUISE_SPEED_MPS             {CRUISE_SPEED_MPS:.4f}")

    pull_overs = []
    for name in ("bay_open", "bay_far"):
        print(f"\nCLOSED-LOOP PULL-OVER: {name}")
        result = measure_pull_over(model, policy, name)
        pull_overs.append(result)
        for key, value in result.items():
            print(f"  {key:38s} {value}")
    print(f"\n  encounter.VY_SPEED_MPS                 {VY_SPEED_MPS:.4f}")
    print(f"  encounter.SETTLE_S                     {SETTLE_S:.4f}")
    print(f"  encounter.LATERAL_DEAD_TIME_S          {LATERAL_DEAD_TIME_S:.4f}")
    print(f"  encounter.APPROACH_SPEED_MPS           {APPROACH_SPEED_MPS:.4f}")

    approaches = []
    print("\nWHOLE-MANOEUVRE APPROACH (forward and lateral together)")
    header = (f"  {'alcove':12s} {'lead':>6} {'parked':>7} {'fwd leg':>8} "
              f"{'lat leg':>8} {'max leg':>8} {'ratio':>6} {'inside':>7} "
              f"{'yaw':>7}")
    print(header)
    for name in ("bay_open", "bay_far"):
        for lead in (0.10, 0.30, 0.50, 0.70, 0.90):
            result = measure_approach(model, policy, name, lead)
            approaches.append(result)
            parked = result["parked_s"]
            ratio = result["ratio_to_max_leg"]
            print(f"  {name:12s} {lead:6.2f} "
                  f"{'None' if parked is None else f'{parked:7.2f}'} "
                  f"{result['forward_leg_s']:8.2f} "
                  f"{result['lateral_leg_s']:8.2f} "
                  f"{result['max_leg_s']:8.2f} "
                  f"{'  None' if ratio is None else f'{ratio:6.2f}'} "
                  f"{str(result['inside_mouth']):>7s} "
                  f"{result['max_abs_yaw_deg']:7.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "radii": radii, "passing_geometry": geometry, "cruise": cruise,
            "pull_overs": pull_overs, "approaches": approaches,
        }, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
