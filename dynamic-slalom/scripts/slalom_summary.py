#!/usr/bin/env python3
"""Turning a finished rollout into the summary the gate grades and the README
quotes.

Measurement lives here; JUDGING lives in ``slalom_metrics``.  Keeping them apart
means a gate cannot quietly redefine what it measures, and it is why
``tests/test_gate_counterexamples.py`` can mutate a real summary once per gate
and require the named gate to reject it.

THREE COMPUTATIONS HERE ARE THE INTERESTING ONES
--------------------------------------------------
* :func:`prediction_bracketing` compares, per encounter, the clearance the
  planner PREDICTED for the corridor it committed to against the clearance that
  was actually MEASURED during that pass.  A prediction is CONSERVATIVE when the
  measured value came out at least as large - the planner under-promised.  This
  is the honest test of a deliberately naive constant-velocity model, and it can
  fail.

* :func:`turning_path` measures how much of the duck's motion was actually
  lateral, and on which hands.  Because the policy has no strafe, a real pass
  shows up as path length exceeding net displacement AND a signed lane offset
  that goes both ways.  A duck that walked a straight line would have almost
  equal path and net, and a lane offset that never left zero.

* :func:`state_cycles` counts the ADVANCE -> THREAT -> {CHOOSE|WAIT} -> PASS ->
  REPLAN cycles that actually ran, which is what makes "it replanned after each
  pass" a count rather than a caption.
"""

from __future__ import annotations

import math

import numpy as np

from policy_runtime import ACTION_SCALE, GYRO_SENSOR, OBS_DIM
from slalom_actors import (
    ROUTES,
    lane_crossings,
    max_heading_step,
    moving_fraction,
    route_records,
)
from slalom_cast import (
    ALL_NAMES,
    BY_NAME,
    CROSSING_NAMES,
    ENCOUNTER_ORDER,
    EXPECTED_PASS_SIDES,
)
from slalom_course import GOAL_XY, STATIC_OBSTACLES
from slalom_states import (
    DUCK_EXACT_LATERAL_HALF_WIDTH,
    DUCK_PLANAR_RADIUS,
    FORBIDDEN_STATES,
    LATERAL_OFFSETS,
    PREDICT_HORIZON_S,
    SAFE_CLEARANCE_M,
    STATES,
    ZERO_COMMAND_STATES,
    ZERO_PATH_10S_M,
)


def prediction_bracketing(tally, passes: list[dict]) -> list[dict]:
    """Per encounter: what was predicted, what happened, and by how much.

    ``conservative`` is True when the MEASURED clearance was at least the
    PREDICTED one, i.e. the planner under-promised.  That is the direction a
    safety prediction has to err in, and a naive constant-velocity model
    genuinely might not - which is what makes this a test rather than a
    tautology.
    """
    out: list[dict] = []
    for index, entry in sorted(tally.predictions.items()):
        measured = entry["measured_m"]
        predicted = entry["predicted_m"]
        record = passes[index] if index < len(passes) else {}
        out.append({
            "index": index,
            "threat": entry["threat"],
            "side": entry["side"],
            "predicted_clearance_m": round(float(predicted), 4),
            "measured_clearance_m": (None if not np.isfinite(measured)
                                     else round(float(measured), 4)),
            "margin_m": (None if not np.isfinite(measured)
                         else round(float(measured - predicted), 4)),
            "conservative": bool(np.isfinite(measured)
                                 and measured >= predicted),
            "measured_positive": bool(np.isfinite(measured) and measured > 0.0),
            "rejected_side": record.get("rejected_side", ""),
            "rejected_clearance_m": record.get("rejected_clearance_m"),
        })
    return out


def turning_path(records: list[dict], tally) -> dict:
    """Evidence that the duck really moved sideways, on both hands.

    The policy has NO STRAFE (MEASURED), so lateral displacement can only come
    from a turning path.  Three numbers together make that case: path length
    against net displacement, the signed lane offset's extremes, and the total
    yaw travel.
    """
    if not records:
        return {}
    start = np.asarray(records[0]["duck_xy"], dtype=np.float64)
    end = np.asarray(records[-1]["duck_xy"], dtype=np.float64)
    net = float(np.linalg.norm(end - start))
    yaw_travel = 0.0
    previous = records[0]["duck_yaw_deg"]
    for record in records[1:]:
        delta = record["duck_yaw_deg"] - previous
        yaw_travel += abs((delta + 180.0) % 360.0 - 180.0)
        previous = record["duck_yaw_deg"]
    return {
        "path_m": round(float(tally.path_m), 4),
        "net_m": round(net, 4),
        "excess_over_net_m": round(float(tally.path_m) - net, 4),
        "max_left_offset_m": (None if not np.isfinite(tally.max_left_offset_m)
                              else round(float(tally.max_left_offset_m), 4)),
        "max_right_offset_m": (
            None if not np.isfinite(tally.max_right_offset_m)
            else round(float(tally.max_right_offset_m), 4)),
        "lateral_span_m": (
            None if not (np.isfinite(tally.max_left_offset_m)
                         and np.isfinite(tally.max_right_offset_m))
            else round(float(tally.max_left_offset_m
                             - tally.max_right_offset_m), 4)),
        "lateral_path_m": round(float(tally.lateral_path_m), 4),
        "yaw_travel_deg": round(yaw_travel, 2),
    }


def state_cycles(transitions: list[dict]) -> dict:
    """How many full encounter cycles ran, and in what order the states went.

    A cycle is ADVANCE -> THREAT -> (CHOOSE_* | WAIT) -> PASS -> REPLAN.  Counted
    from the transition log rather than assumed, so "it replanned after every
    pass" is arithmetic.
    """
    order = [t["to"] for t in transitions]
    cycles = 0
    index = 0
    while index < len(order):
        if order[index] == "THREAT":
            window = order[index:index + 5]
            if "PASS" in window and "REPLAN" in window:
                cycles += 1
        index += 1
    return {
        "state_order": order,
        "encounter_cycles": cycles,
        "replans_after_pass": sum(1 for i, s in enumerate(order)
                                  if s == "REPLAN" and i > 0
                                  and order[i - 1] == "PASS"),
    }


def summarize(rollout) -> dict:
    """Everything the gate needs, measured from a finished rollout."""
    records = rollout.records
    tally = rollout.tally
    machine = rollout.machine
    dt = rollout.dt

    machine_summary = machine.summary()
    passes = machine_summary["passes"]

    state_seconds = {state: steps * dt
                     for state, steps in tally.state_steps.items()}
    zero_state_path = {state: round(tally.state_path_m.get(state, 0.0), 5)
                       for state in ZERO_COMMAND_STATES
                       if state in tally.state_steps}

    start = np.asarray(records[0]["duck_xy"], dtype=np.float64)
    end = np.asarray(records[-1]["duck_xy"], dtype=np.float64)

    subject_visibility = {}
    for subject, steps in tally.subject_steps.items():
        los = tally.subject_los.get(subject, 0)
        seen = tally.subject_visible_los.get(subject, 0)
        subject_visibility[subject] = {
            "role": "goal" if subject == "goal" else BY_NAME[subject].kind,
            "steps": steps,
            "los_steps": los,
            "visible_with_los": seen,
            "fraction_with_los": (seen / los) if los else 1.0,
        }

    moving = moving_fraction(rollout.seconds)
    moved = [name for name in ALL_NAMES if moving.get(name, 0.0) > 0.02]

    return {
        # -- the physical contract -----------------------------------------
        "gyro_sensor": GYRO_SENSOR,
        "observation_dim": OBS_DIM,
        "action_scale": ACTION_SCALE,
        "policy_sha256": rollout.policy_sha256,
        "control_hz": 1.0 / dt,
        "decimation": rollout.decimation,
        "seconds": rollout.seconds,
        "steps": len(records),

        # -- locomotion ------------------------------------------------------
        "path_m": round(float(tally.path_m), 4),
        "net_m": round(float(np.linalg.norm(end - start)), 4),
        "start_xy": [round(float(start[0]), 4), round(float(start[1]), 4)],
        "final_xy": [round(float(end[0]), 4), round(float(end[1]), 4)],
        "min_trunk_z_m": round(float(tally.min_trunk_z), 5),
        "final_trunk_z_m": round(float(records[-1]["trunk_z"]), 5),
        "fallen_steps": tally.fallen_steps,
        "contact_steps": tally.contact_steps,
        "walk_path_m": round(float(tally.walk_path_m), 4),
        "walk_seconds": round(tally.walk_steps * dt, 3),

        # -- the turning path ------------------------------------------------
        "turning_path": turning_path(records, tally),

        # -- the decisions ----------------------------------------------------
        "passes": passes,
        "pass_sides": machine_summary["pass_sides"],
        "expected_pass_sides": list(EXPECTED_PASS_SIDES),
        "alternating": machine_summary["alternating"],
        "waits": machine_summary["waits"],
        "replans": machine_summary["replans"],
        "pass_count": len(passes),
        "wait_count": len(machine_summary["waits"]),
        "prediction_bracketing": prediction_bracketing(tally, passes),
        "predict_horizon_s": PREDICT_HORIZON_S,
        "safe_clearance_bar_m": SAFE_CLEARANCE_M,
        "lateral_offsets_m": list(LATERAL_OFFSETS),
        **state_cycles(machine_summary["transitions"]),

        # -- the goal ----------------------------------------------------------
        "goal_xy": list(GOAL_XY),
        "reached_goal_at_s": tally.reached_goal_at_s,
        "goal_steps": tally.goal_steps,
        "goal_seconds": round(tally.goal_steps * dt, 3),
        "min_goal_distance_m": round(float(tally.min_goal_distance_m), 4),
        "goal_visible_fraction": (
            tally.goal_visible_steps / max(len(records), 1)),
        "goal_visible_fraction_with_los": (
            tally.goal_visible_with_los / tally.goal_los_steps
            if tally.goal_los_steps else 1.0),
        "goal_los_steps": tally.goal_los_steps,

        # -- clearance ---------------------------------------------------------
        "min_body_clearance_m": round(float(tally.min_body_clearance), 4),
        "min_body_clearance_name": tally.min_body_name,
        "min_clearance_by_body_m": {
            name: round(float(gap), 4)
            for name, gap in sorted(tally.min_clearance_by_body.items())},
        "min_scenery_clearance_m": round(float(tally.min_scenery_clearance), 4),
        "min_scenery_clearance_geom": tally.min_scenery_geom,
        "duck_planar_radius_m": round(float(rollout.duck_radius), 4),
        "duck_exact_planar_radius_m": round(float(rollout.duck_exact_radius), 4),
        "duck_exact_lateral_half_width_m": round(
            float(rollout.duck_lateral_half), 4),
        "actor_lateral_half_m": round(float(rollout.actor_lateral_half), 4),

        # -- the states ---------------------------------------------------------
        "declared_states": list(STATES),
        "states_visited": sorted(tally.state_steps),
        "state_steps": dict(tally.state_steps),
        "state_seconds": {k: round(v, 3) for k, v in state_seconds.items()},
        "state_command_max": {k: round(float(v), 6)
                              for k, v in tally.state_command_max.items()},
        "state_path_m": {k: round(float(v), 5)
                         for k, v in tally.state_path_m.items()},
        "zero_command_states": list(ZERO_COMMAND_STATES),
        "zero_command_violations": list(tally.zero_command_violations),
        "max_abs_vy_command": round(float(tally.max_abs_vy_command), 9),
        "zero_state_path_m": zero_state_path,
        "zero_episodes": list(tally.zero_episodes),
        "worst_zero_episode_path_m": (
            max((e["path_m"] for e in tally.zero_episodes), default=0.0)),
        "worst_zero_episode_net_m": (
            max((e["net_m"] for e in tally.zero_episodes), default=0.0)),
        "zero_drift_reference_m": ZERO_PATH_10S_M,
        "longest_illegal_zero_run": tally.longest_illegal_zero_run,
        "illegal_zero_windows": list(tally.illegal_zero_windows),
        "forbidden_state_steps": {
            state: tally.state_steps.get(state, 0)
            for state in FORBIDDEN_STATES},
        "transitions": machine_summary["transitions"],
        "timeouts": machine_summary["timeouts"],

        # -- visibility ----------------------------------------------------------
        "subject_visibility": subject_visibility,
        "subject_sequence": list(tally.subject_sequence),
        "visible_fraction_with_los": (
            tally.visible_with_los / tally.los_steps
            if tally.los_steps else 1.0),
        "monitor_steps": tally.monitor_steps,
        "monitor_los_steps": tally.monitor_los_steps,
        "monitor_visible_fraction_with_los": (
            tally.monitor_visible_with_los / tally.monitor_los_steps
            if tally.monitor_los_steps else 1.0),
        "blocked_by": dict(tally.blocked_by),

        # -- the interlock --------------------------------------------------------
        "interlock_holds": tally.interlock_holds,
        "interlock_reasons": dict(tally.interlock_reasons),

        # -- the scenario ----------------------------------------------------------
        "actor_names": list(ALL_NAMES),
        "crossing_actor_names": list(CROSSING_NAMES),
        "actor_routes": route_records(),
        "encounter_order": list(ENCOUNTER_ORDER),
        "lane_crossings": lane_crossings(rollout.seconds),
        "moving_actors": len(moved),
        "moving_fraction": {k: round(v, 3) for k, v in moving.items()},
        "max_actor_heading_step_deg": round(
            max_heading_step(rollout.seconds)[0], 4),
        "max_bodies_in_lane": tally.max_bodies_in_lane,
        "lane_occupied_seconds": round(tally.lane_occupied_steps * dt, 3),
        "static_obstacle_names": [o.name for o in STATIC_OBSTACLES],
        "static_obstacle_count": len(STATIC_OBSTACLES),
        "actor_count": len(ALL_NAMES),
        "obstacle_and_actor_count": len(STATIC_OBSTACLES) + len(ALL_NAMES),

        # -- the tracker -----------------------------------------------------------
        "tracker_max_raw_speed_mps": round(
            float(rollout.tracker.max_raw_speed_mps), 4),
        "tracker_max_filtered_speed_mps": round(
            float(rollout.tracker.max_filtered_speed_mps), 4),
        # What the filter ACTUALLY bounds: how fast the velocity estimate may
        # change.  The speed extremes coincide on this cast because the actors
        # walk analytic constant-speed routes; the rate of change does not.
        "tracker_max_raw_accel_mps2": round(
            float(rollout.tracker.max_raw_accel_mps2), 4),
        "tracker_max_filtered_accel_mps2": round(
            float(rollout.tracker.max_filtered_accel_mps2), 4),
    }
