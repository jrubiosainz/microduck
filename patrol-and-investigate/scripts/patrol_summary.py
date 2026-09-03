#!/usr/bin/env python3
"""Turning a finished rollout into the summary the gate grades and the README
quotes.

Measurement lives here; JUDGING lives in ``patrol_metrics``.  Keeping them apart
means a gate cannot quietly redefine what it measures, and it is why
``tests/test_gate_counterexamples.py`` can mutate a real summary once per gate
and require the named gate to reject it.

THREE COMPUTATIONS HERE ARE THE INTERESTING ONES
--------------------------------------------------
* :func:`checkpoint_order` compares the sequence of checkpoints the duck
  actually recorded against the order the facility declares, and reports both.
  It is the gate that a patrol is a patrol rather than a wander, and it can only
  be checked because the two are computed in different modules.

* :func:`route_memory` reports, per interruption, the checkpoint the duck was
  walking to when it broke off, the checkpoint it was walking to when it
  resumed, and the MEASURED distance between the point it left and the point it
  came back to.  Those are three separately-recorded quantities, which is what
  makes "it remembered its route" checkable rather than a caption.

* :func:`standoff_report` compares, per investigation, the range the approach
  STARTED at against the range it ENDED at and the closest the duck's own
  contact probe ever measured to that body.  The first pair proves the approach
  was physical; the third proves it never touched anything.
"""

from __future__ import annotations

import numpy as np

from policy_runtime import ACTION_SCALE, GYRO_SENSOR, OBS_DIM
from patrol_actors import (
    appearance_times,
    max_heading_step,
    moving_fraction,
    route_records,
    zone_occupancy,
)
from patrol_cast import (
    ALL_NAMES,
    ANOMALY_NAMES,
    BY_NAME,
    DISMISSED,
    EXPECTED_VERDICTS,
    INVESTIGATED,
    OBJECT_NAMES,
    PERSON_NAMES,
)
from patrol_facility import (
    CHECKPOINT_NAMES,
    FIXTURES,
    HOME,
    RESTRICTED_ZONE,
)
from patrol_plan import circuit_length_m, corner_turns_deg
from patrol_states import (
    DUCK_EXACT_LATERAL_HALF_WIDTH,
    DUCK_PLANAR_RADIUS,
    FORBIDDEN_STATES,
    OBSERVE_ANGLES_DEG,
    SCAN_ARC_COMPLETE_DEG,
    SCAN_SWEEP_DEG,
    SPIN_BEST_RATE_DPS,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
    STATES,
    ZERO_COMMAND_STATES,
    ZERO_PATH_10S_M,
)


def checkpoint_order(visits: list[dict]) -> dict:
    """Did the duck visit all five checkpoints, in order, exactly once?

    Reported as three separate facts - the sequence, whether it matches, and
    whether anything was repeated - so a failure says WHICH property broke.
    """
    visited = [v["name"] for v in visits]
    return {
        "declared_order": list(CHECKPOINT_NAMES),
        "visited_order": visited,
        "in_declared_order": visited == list(CHECKPOINT_NAMES),
        "all_visited": set(visited) == set(CHECKPOINT_NAMES),
        "no_repeats": len(visited) == len(set(visited)),
        "count": len(visited),
    }


def scan_report(visits: list[dict]) -> dict:
    """What each checkpoint stop and sweep actually was, on the floor."""
    completed = [v for v in visits if v["result"] == "clear"]
    return {
        "scans": [
            {
                "checkpoint": v["name"],
                "stopped_s": v["stopped_s"],
                "scan_s": v["scan_s"],
                "scan_arc_deg": v["scan_arc_deg"],
                "bodies_seen": v["bodies_seen"],
                "result": v["result"],
                "detected": v["detected"],
                "still_path_m": v["still_path_m"],
                "arrival_error_m": v["arrival_error_m"],
            } for v in visits],
        "completed_scan_arcs_deg": [v["scan_arc_deg"] for v in completed],
        "min_completed_scan_arc_deg": (
            min((v["scan_arc_deg"] for v in completed), default=0.0)),
        "min_stopped_s": min((v["stopped_s"] for v in visits), default=0.0),
        "max_still_path_m": max((v["still_path_m"] for v in visits),
                                default=0.0),
        "max_arrival_error_m": max((v["arrival_error_m"] for v in visits),
                                   default=0.0),
        "scan_sweep_deg": SCAN_SWEEP_DEG,
        "scan_arc_complete_deg": SCAN_ARC_COMPLETE_DEG,
    }


def route_memory(plan, investigations: list[dict]) -> dict:
    """Was the interrupted route preserved, and did the duck physically return?"""
    entries = [i.as_record() for i in plan.interruptions]
    return {
        "interruptions": entries,
        "count": len(entries),
        "all_preserved": bool(entries) and all(
            e["route_preserved"] for e in entries),
        "max_return_error_m": max(
            (e["return_error_m"] for e in entries
             if e["return_error_m"] is not None), default=None),
        "resumed_targets": [e["resumed_target_name"] for e in entries],
        "interrupted_targets": [e["target_name"] for e in entries],
    }


def standoff_report(investigations: list[dict]) -> dict:
    """Per investigation: was the approach real, and did it stop safely?"""
    return {
        "investigations": investigations,
        "count": len(investigations),
        "range_reductions_m": [i["range_reduction_m"] for i in investigations],
        "approach_paths_m": [i["approach_path_m"] for i in investigations],
        "final_standoffs_m": [i["approach_end_range_m"]
                              for i in investigations],
        "min_clearances_m": [i["min_clearance_m"] for i in investigations],
        "angles_held": [i["angles_held"] for i in investigations],
        "band_m": [STANDOFF_MIN_M, STANDOFF_MAX_M],
        "declared_angles_deg": list(OBSERVE_ANGLES_DEG),
    }


def verdict_report(machine, detector) -> dict:
    """What the duck concluded about each anomaly, against what it should have.

    ``expected`` comes from the CAST and the duck never reads it, so comparing
    the two is evidence rather than a tautology.
    """
    verdicts = {v["target"]: v["verdict"] for v in machine.verdicts}
    return {
        "verdicts": list(machine.verdicts),
        "verdict_by_target": verdicts,
        "expected": dict(EXPECTED_VERDICTS),
        "all_correct": all(verdicts.get(k) == v
                           for k, v in EXPECTED_VERDICTS.items()),
        "investigated": [v["target"] for v in machine.verdicts
                         if v["verdict"] in ("suspicious", "intrusion")],
        "dismissed": [v["target"] for v in machine.verdicts
                      if v["verdict"] == "benign"],
        "expected_investigated": list(INVESTIGATED),
        "expected_dismissed": list(DISMISSED),
        "camera_gate_ticks": dict(detector.gate_ticks),
        "first_in_camera_gate_s": {
            k: round(v, 3) for k, v in detector.first_gate_s.items()},
        "observations": detector.summary()["observations"],
    }


def summarize(rollout) -> dict:
    """Everything the gate needs, measured from a finished rollout."""
    records = rollout.records
    tally = rollout.tally
    machine = rollout.machine
    dt = rollout.dt

    machine_summary = machine.summary()
    visits = machine_summary["visits"]
    investigations = machine_summary["investigations"]

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
            "role": "route" if subject == "route" else BY_NAME[subject].kind,
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
        "spin_rate_measured_dps": SPIN_BEST_RATE_DPS,

        # -- the patrol --------------------------------------------------------
        **{f"checkpoint_{k}": v for k, v in checkpoint_order(visits).items()},
        "visits": visits,
        "checkpoint_arrival_error_m": {
            k: round(v, 4) for k, v in tally.checkpoint_arrival_error_m.items()},
        "circuit_length_m": round(circuit_length_m(), 4),
        "corner_turns_deg": corner_turns_deg(),
        "reached_home_at_s": tally.reached_home_at_s,
        "home_seconds": round(tally.home_steps * dt, 3),
        "min_home_distance_m": round(float(tally.min_home_distance_m), 4),
        "home_xy": list(HOME.xy),
        **{f"scan_{k}": v for k, v in scan_report(visits).items()},

        # -- the route memory ----------------------------------------------------
        **{f"memory_{k}": v
           for k, v in route_memory(rollout.plan, investigations).items()},

        # -- the investigations ---------------------------------------------------
        **{f"standoff_{k}": v
           for k, v in standoff_report(investigations).items()},
        **{f"verdict_{k}": v
           for k, v in verdict_report(machine, rollout.detector).items()},

        # -- clearance -------------------------------------------------------------
        "min_body_clearance_m": round(float(tally.min_body_clearance), 4),
        "min_body_clearance_name": tally.min_body_name,
        "min_clearance_by_body_m": {
            name: round(float(gap), 4)
            for name, gap in sorted(tally.min_clearance_by_body.items())},
        "min_scenery_clearance_m": round(float(tally.min_scenery_clearance), 4),
        "min_scenery_clearance_geom": tally.min_scenery_geom,
        "min_zone_gap_m": round(float(tally.min_zone_gap_m), 4),
        "zone_breach_steps": tally.zone_breach_steps,
        "restricted_zone": {
            "name": RESTRICTED_ZONE.name,
            "center": list(RESTRICTED_ZONE.center),
            "half": list(RESTRICTED_ZONE.half),
        },
        "duck_planar_radius_m": round(float(rollout.duck_radius), 4),
        "duck_exact_planar_radius_m": round(float(rollout.duck_exact_radius), 4),
        "duck_exact_lateral_half_width_m": round(
            float(rollout.duck_lateral_half), 4),

        # -- the states -------------------------------------------------------------
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
        "camera_active_steps": tally.camera_active_steps,
        "camera_active_fraction": (
            tally.camera_active_steps / max(len(records), 1)),
        "blocked_by": dict(tally.blocked_by),
        "max_bodies_visible": tally.max_bodies_visible,
        "bodies_ever_seen": sorted(tally.bodies_ever_seen),

        # -- the interlock --------------------------------------------------------
        "interlock_holds": tally.interlock_holds,
        "interlock_reasons": dict(tally.interlock_reasons),

        # -- the facility ----------------------------------------------------------
        "body_names": list(ALL_NAMES),
        "person_names": list(PERSON_NAMES),
        "object_names": list(OBJECT_NAMES),
        "anomaly_names": list(ANOMALY_NAMES),
        "body_routes": route_records(),
        "appearance_times_s": appearance_times(),
        "moving_bodies": len(moved),
        "moving_fraction": {k: round(v, 3) for k, v in moving.items()},
        "max_actor_heading_step_deg": round(
            max_heading_step(rollout.seconds)[0], 4),
        "zone_occupancy_s": zone_occupancy(rollout.seconds),
        "fixture_names": [f.name for f in FIXTURES],
        "fixture_count": len(FIXTURES),
        "body_count": len(ALL_NAMES),
    }
