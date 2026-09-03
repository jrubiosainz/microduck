#!/usr/bin/env python3
"""Flatten one rollout into the summary dict every gate and the README read.

Split from ``guide_metrics`` so that MEASURING a run and JUDGING it are separate
files.  Nothing here decides whether anything passed: it only reports what
happened, which is why a gate can never quietly change a measurement to suit
itself.

THE STALL WINDOWS AND THE EPISODES ARE COMPARED, NOT CONFLATED
----------------------------------------------------------------
``guide_follower.STALLS`` declares when she slows down.  The machine cannot see
it.  :func:`summarize` reports BOTH the declared windows and the episodes the
duck opened from its own measurements, side by side, so the gate can require
each declared stall to have produced a detection — which is a much stronger
statement than "two episodes happened".
"""

from __future__ import annotations

import numpy as np

from guide_cast import CROWD_NAMES, FOLLOWER, OTHER_NAMES, PEOPLE
from guide_follower import MIN_TRAIL_GAP_M, stall_windows
from guide_layout import DESTINATION_KEYS
from guide_states import (
    CATCHUP_DISTANCE_M,
    FACE_TOLERANCE_DEG,
    FINAL_PERSON_NEAR_M,
    FORBIDDEN_STATES,
    INDICATE_SECONDS,
    LAG_CONFIRM_S,
    LAG_DISTANCE_M,
    LOST_CONFIRM_S,
    MONITOR_STATES,
    RESUME_CONFIRM_S,
    SAFETY_MAX_DISTANCE_M,
    SAFETY_MAX_INTERVAL_S,
    STATES,
    ZERO_COMMAND_STATES,
)
from guide_thresholds import CHECK_STILL_PATH_M, FINAL_DISTANCE_BAND_M


def _fraction(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def summarize(rollout) -> dict:
    """Every measured quantity the gate and the README quote."""
    records = rollout.records
    machine = rollout.machine
    plan = rollout.plan
    dt = rollout.dt

    states = [r["state"] for r in records]
    final = records[-1]

    # -- the lead ------------------------------------------------------------
    lead_records = [r for r in records if r["state"] in ("LEAD", "RESUME")]
    lead_start = (np.array(lead_records[0]["duck_xy"])
                  if lead_records else np.zeros(2))
    lead_end = (np.array(lead_records[-1]["duck_xy"])
                if lead_records else np.zeros(2))
    lead_net = float(np.linalg.norm(lead_end - lead_start))

    # -- the episodes --------------------------------------------------------
    episodes = []
    for episode in machine.episodes:
        index = episode["index"]
        wait_steps = rollout.tally.episode_wait_steps.get(index, 0)
        los_steps = rollout.tally.episode_los_steps.get(index, 0)
        seen_steps = rollout.tally.episode_visible_steps.get(index, 0)
        wait_xy = rollout.tally.episode_wait_xy.get(index)
        episodes.append({
            **episode,
            "monitor_seconds": round(wait_steps * dt, 3),
            "wait_only_seconds": round(
                rollout.tally.episode_wait_only_steps.get(index, 0) * dt, 3),
            "max_command_while_waiting": round(
                rollout.tally.episode_wait_command_peak.get(index, 0.0), 6),
            "duck_moved_while_waiting_m": round(
                rollout.tally.episode_wait_moved_m.get(index, 0.0), 4),
            "duck_moved_while_monitoring_m": round(
                rollout.tally.episode_duck_moved_m.get(index, 0.0), 4),
            "squaring_up_path_m": round(
                rollout.tally.episode_check_path_m.get(index, 0.0), 4),
            "follower_closed_m": round(
                rollout.tally.episode_follower_closed_m.get(index, 0.0), 4),
            "waiting_spot_xy": (None if wait_xy is None else
                                [round(float(wait_xy[0]), 4),
                                 round(float(wait_xy[1]), 4)]),
            "waiting_spot_scenery_clearance_m": round(
                rollout.tally.episode_wait_scenery.get(index, float("nan")), 4),
            "los_steps": los_steps,
            "visible_with_los_steps": seen_steps,
            "visible_fraction_with_los": round(
                _fraction(seen_steps, los_steps), 4),
        })

    # -- the declared stalls, for comparison ---------------------------------
    stalls = stall_windows()
    matched = []
    for stall in stalls:
        hits = [e for e in episodes
                if stall["start_s"] <= e["detected_at_s"]
                <= stall["end_s"] + 8.0]
        matched.append({
            **stall,
            "episode_indices": [e["index"] for e in hits],
            "detected": bool(hits),
            "detection_lag_s": (round(hits[0]["detected_at_s"]
                                      - stall["start_s"], 3)
                                if hits else None),
        })

    # -- the arrival ---------------------------------------------------------
    final_destination_distance = final.get("destination_distance_m")
    final_follower_distance = final["follower_range_m"]
    indicate_seconds = rollout.tally.state_steps.get("INDICATE", 0) * dt

    # -- the actors ----------------------------------------------------------
    moving_adults = sum(
        1 for name in OTHER_NAMES
        if max(float(np.linalg.norm(
            np.array(records[-1]["person_xy"][name])
            - np.array(records[0]["person_xy"][name]))), 0.0) > 0.5)

    # -- visibility in the monitor states ------------------------------------
    monitor_records = [r for r in records if r["state"] in MONITOR_STATES]

    return {
        "seconds": rollout.seconds,
        "control_steps": len(records),
        "control_hz": 1.0 / dt,
        "follower": machine.follower,

        # physics
        "observation_dim": 61,
        "action_scale": 0.9,
        "gyro_sensor": "imu_ang_vel",
        "policy_sha256": rollout.policy_sha256,
        "fallen_steps": rollout.tally.fallen_steps,
        "contact_steps": rollout.tally.contact_steps,
        "min_trunk_z_m": round(rollout.tally.min_trunk_z, 5),
        "final_trunk_z_m": round(float(final["trunk_z"]), 5),
        "path_m": round(rollout.tally.path_m, 4),
        "min_person_clearance_m": round(rollout.tally.min_person_clearance, 4),
        "min_person_clearance_name": rollout.tally.min_person_name,
        "min_follower_clearance_m": round(rollout.tally.min_follower_clearance, 4),
        "min_scenery_clearance_m": round(rollout.tally.min_scenery_clearance, 4),
        "min_scenery_clearance_geom": rollout.tally.min_scenery_geom,
        "duck_planar_radius_m": round(rollout.duck_radius, 4),
        "duck_exact_radius_m": round(rollout.duck_exact_radius, 4),
        "adult_half_extent_m": round(rollout.adult_half_extent, 4),
        "adult_half_extent_basis": "pose-zero sample; not a gait maximum",

        # states
        "states_visited": sorted(set(states)),
        "declared_states": list(STATES),
        "forbidden_states": list(FORBIDDEN_STATES),
        "forbidden_state_steps": {
            state: rollout.tally.state_steps.get(state, 0)
            for state in FORBIDDEN_STATES},
        "state_steps": dict(rollout.tally.state_steps),
        "state_seconds": {k: round(v * dt, 3)
                          for k, v in rollout.tally.state_steps.items()},
        "state_command_max": {k: round(v, 6)
                              for k, v in rollout.tally.state_command_max.items()},
        "zero_command_states": list(ZERO_COMMAND_STATES),
        "zero_command_violations": list(rollout.tally.zero_command_violations),
        "transitions": machine.transitions,

        # the request
        "requested_destination": machine.requested_key,
        "request_t_s": machine.request_t,
        "destination_candidates": list(machine.candidates),
        "destination_candidate_count": len(machine.candidates),
        "all_destination_keys": list(DESTINATION_KEYS),
        "resolved_destination": (None if machine.destination is None
                                 else machine.destination.key),
        "resolution_correct": (machine.destination is not None
                               and machine.destination.key
                               == machine.requested_key),

        # the plan
        "plan": None if plan is None else plan.as_record(),
        "planned": machine.planned,

        # the lead
        "lead_seconds": round(len(lead_records) * dt, 3),
        "lead_path_m": round(rollout.tally.lead_path_m, 4),
        "lead_net_m": round(lead_net, 4),
        "max_cross_track_m": round(rollout.tally.max_cross_track_m, 4),
        "route_progress_final": final.get("route_progress"),
        "route_remaining_final_m": final.get("route_remaining_m"),

        # the follower
        "follower_walked_m": round(rollout.follower.walked_m, 4),
        "follower_trail_gap_final_m": round(rollout.follower.trail_gap_m, 4),
        "min_lead_gap_m": round(rollout.tally.min_lead_gap_m, 4),
        "follower_ahead_steps": rollout.tally.follower_ahead_steps,
        "trail_gap_floor_m": MIN_TRAIL_GAP_M,
        "max_follower_range_m": round(rollout.tally.max_follower_range_m, 4),
        "safety_max_distance_m": SAFETY_MAX_DISTANCE_M,
        "max_safety_breach_s": round(rollout.tally.max_safety_breach_s, 3),
        "safety_max_interval_s": SAFETY_MAX_INTERVAL_S,

        # the episodes
        "max_check_path_m": round(rollout.tally.max_check_path_m, 4),
        "check_still_path_m": CHECK_STILL_PATH_M,
        "episodes": episodes,
        "episode_count": len(episodes),
        "lag_threshold_m": LAG_DISTANCE_M,
        "lag_confirm_s": LAG_CONFIRM_S,
        "lost_confirm_s": LOST_CONFIRM_S,
        "catchup_threshold_m": CATCHUP_DISTANCE_M,
        "resume_confirm_s": RESUME_CONFIRM_S,
        "declared_stalls": matched,
        "stalls_detected": sum(1 for s in matched if s["detected"]),

        # the arrival
        "arrival": dict(machine.arrival),
        "arrived": machine.arrived,
        "final_destination_distance_m": final_destination_distance,
        "final_destination_band_m": list(FINAL_DISTANCE_BAND_M),
        "final_facing_error_deg": final.get("facing_error_deg"),
        "face_tolerance_deg": FACE_TOLERANCE_DEG,
        "final_follower_distance_m": final_follower_distance,
        "final_person_near_m": FINAL_PERSON_NEAR_M,
        "indicate_seconds": round(indicate_seconds, 3),
        "indicate_required_s": INDICATE_SECONDS,

        # the cast
        "people": [p.name for p in PEOPLE],
        "other_adults": list(OTHER_NAMES),
        "moving_adults": moving_adults,
        "crowd_names": list(CROWD_NAMES),

        # visibility
        "visible_steps": rollout.tally.visible_steps,
        "visible_fraction": round(
            _fraction(rollout.tally.visible_steps, len(records)), 4),
        "los_steps": rollout.tally.los_steps,
        "visible_with_los_steps": rollout.tally.visible_with_los,
        "visible_fraction_with_los": round(
            _fraction(rollout.tally.visible_with_los, rollout.tally.los_steps), 4),
        "monitor_steps": rollout.tally.monitor_steps,
        "monitor_los_steps": rollout.tally.monitor_los_steps,
        "monitor_visible_with_los_steps": rollout.tally.monitor_visible_with_los,
        "monitor_visible_fraction_with_los": round(
            _fraction(rollout.tally.monitor_visible_with_los,
                      rollout.tally.monitor_los_steps), 4),
        "monitor_records": len(monitor_records),
        "blocked_by": dict(rollout.tally.blocked_by),

        # timeouts
        "timeouts": machine.timeouts,
    }
