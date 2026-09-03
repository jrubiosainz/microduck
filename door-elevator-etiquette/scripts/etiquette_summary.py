#!/usr/bin/env python3
"""Flatten one rollout into the summary dict every gate and the README read.

Split from ``etiquette_metrics`` so that MEASURING a run and JUDGING it are
separate files.  Nothing here decides whether anything passed: it only reports
what happened, which is why a gate can never quietly change a measurement to suit
itself.

THE SCHEDULE AND THE MEASUREMENTS ARE COMPARED, NOT CONFLATED
---------------------------------------------------------------
``lobby_doors.DOOR_SCHEDULE`` declares when each door opens, and
``etiquette_actors.ROUTES`` declares when each person walks.  The duck cannot see
either.  :func:`summarize` reports BOTH the declared schedule and what the duck
actually measured and did, side by side, so the gate can require the two to line
up - which is a much stronger statement than "the run finished".
"""

from __future__ import annotations

import numpy as np

from etiquette_actors import ACTOR_NAMES, moving_fraction, route_records
from etiquette_aim import expected_subject_order, role_of
from etiquette_cast import (
    DOOR_EXITER_NAMES,
    GUARDIAN,
    OCCUPANT_NAMES,
    OTHER_NAMES,
    PEOPLE,
)
from etiquette_path import (
    LEG_NAMES,
    aperture_crossings,
    route_bend_report,
)
from etiquette_states import (
    DUCK_EXACT_LATERAL_HALF_WIDTH,
    FORBIDDEN_STATES,
    MIN_OCCUPANTS_EXITED,
    MONITOR_STATES,
    SPIN_180_SECONDS,
    SPIN_BEST_RATE_DPS,
    STATES,
    WALKING_STATES,
    ZERO_COMMAND_STATES,
    ZERO_PATH_10S_M,
)
from etiquette_thresholds import ZERO_STATE_PATH_M
from lobby_doors import APERTURE_NAMES, DOOR_PASSABLE_FRACTION, schedule_windows
from lobby_layout import ABREAST_MARGIN_M, DOOR_CLEAR_W, LIFT_CLEAR_W


def _fraction(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def _distinct_roles(sequence: list[dict]) -> list[str]:
    """The sequence of DISTINCT subject ROLES the run produced.

    Roles rather than names because which individual is least clear at a given
    instant is a measurement; what the behavior claims is the ROLE order.
    """
    roles: list[str] = []
    for entry in sequence:
        role = role_of(entry["subject"])
        if not roles or roles[-1] != role:
            roles.append(role)
    return roles


def summarize(rollout) -> dict:
    """Every measured quantity the gate and the README quote."""
    records = rollout.records
    machine = rollout.machine
    tally = rollout.tally
    dt = rollout.dt

    states = [r["state"] for r in records]
    final = records[-1]
    first = records[0]

    walk_records = [r for r in records if r["state"] in WALKING_STATES]
    start = np.array(first["duck_xy"])
    end = np.array(final["duck_xy"])

    # -- the yields ----------------------------------------------------------
    yields = []
    for entry in machine.yields:
        yields.append({
            **entry,
            "state_path_m": round(
                tally.state_path_m.get("YIELD_EXITERS", 0.0), 4),
            "max_command": round(
                tally.state_command_max.get("YIELD_EXITERS", 0.0), 6),
        })

    # -- the doors, declared beside measured --------------------------------
    crossings = []
    for name in APERTURE_NAMES:
        record = tally.crossings.get(name)
        crossings.append(record or {"aperture": name, "entered_at_s": None})

    # -- the people ----------------------------------------------------------
    moving_adults = sum(
        1 for name in OTHER_NAMES
        if float(np.linalg.norm(
            np.array(final["person_xy"][name])
            - np.array(first["person_xy"][name]))) > 0.5)

    # Which exiters actually used the doorway, measured from the trace rather
    # than from their declared routes.
    exiters_used_door = sorted({
        name for r in records
        for name in r["aperture_occupancy"]["concourse_door"]["others"]
        if name in DOOR_EXITER_NAMES})
    occupants_used_lift = sorted({
        name for r in records
        for name in r["aperture_occupancy"]["lift_front"]["others"]
        if name in OCCUPANT_NAMES})
    max_occupants_exited = max((r["occupants_exited"] for r in records),
                               default=0)

    # -- the zero-command states --------------------------------------------
    zero_state_path = {
        state: round(tally.state_path_m.get(state, 0.0), 4)
        for state in ZERO_COMMAND_STATES if state in tally.state_steps}

    # -- the cabin -----------------------------------------------------------
    ride_records = [r for r in records if r["state"] == "RIDE"]

    # -- visibility ----------------------------------------------------------
    subject_visibility = {
        subject: {
            "steps": tally.subject_steps.get(subject, 0),
            "los_steps": tally.subject_los.get(subject, 0),
            "visible_with_los": tally.subject_visible_los.get(subject, 0),
            "fraction_with_los": round(
                _fraction(tally.subject_visible_los.get(subject, 0),
                          tally.subject_los.get(subject, 0)), 4),
            "role": role_of(subject),
        }
        for subject in sorted(tally.subject_steps)
    }

    # -- the abreast budget, so the non-vacuity is visible -------------------
    widest_adult = rollout.adult_lateral_half
    abreast = {
        "concourse_door": round(
            DOOR_CLEAR_W - 2 * DUCK_EXACT_LATERAL_HALF_WIDTH
            - 2 * widest_adult, 4),
        "lift_front": round(
            LIFT_CLEAR_W - 2 * DUCK_EXACT_LATERAL_HALF_WIDTH
            - 2 * widest_adult, 4),
    }

    return {
        "seconds": rollout.seconds,
        "control_steps": len(records),
        "control_hz": 1.0 / dt,
        "guardian": machine.guardian,

        # physics
        "observation_dim": 61,
        "action_scale": 0.9,
        "gyro_sensor": "imu_ang_vel",
        "policy_sha256": rollout.policy_sha256,
        "fallen_steps": tally.fallen_steps,
        "contact_steps": tally.contact_steps,
        "min_trunk_z_m": round(tally.min_trunk_z, 5),
        "final_trunk_z_m": round(float(final["trunk_z"]), 5),
        "path_m": round(tally.path_m, 4),
        "net_m": round(float(np.linalg.norm(end - start)), 4),
        "walk_path_m": round(tally.walk_path_m, 4),
        "walk_seconds": round(len(walk_records) * dt, 3),
        "max_cross_track_m": round(tally.max_cross_track_m, 4),

        # clearance
        "min_person_clearance_m": round(tally.min_person_clearance, 4),
        "min_person_clearance_name": tally.min_person_name,
        "min_clearance_by_person_m": {
            name: round(value, 4)
            for name, value in sorted(tally.min_clearance_by_person.items())},
        "min_scenery_clearance_m": round(tally.min_scenery_clearance, 4),
        "min_scenery_clearance_geom": tally.min_scenery_geom,
        "duck_planar_radius_m": round(rollout.duck_radius, 4),
        "duck_exact_radius_m": round(rollout.duck_exact_radius, 4),
        "duck_lateral_half_m": round(rollout.duck_lateral_half, 4),
        "adult_lateral_half_m": round(rollout.adult_lateral_half, 4),
        "adult_half_extent_basis": "pose-zero sample; not a gait maximum",
        "abreast_slack_m": abreast,
        "abreast_margin_required_m": ABREAST_MARGIN_M,

        # states
        "states_visited": sorted(set(states)),
        "declared_states": list(STATES),
        "state_order": [t["to"] for t in machine.transitions],
        "forbidden_states": list(FORBIDDEN_STATES),
        "forbidden_state_steps": {
            state: tally.state_steps.get(state, 0)
            for state in FORBIDDEN_STATES},
        "state_steps": dict(tally.state_steps),
        "state_seconds": {k: round(v * dt, 3)
                          for k, v in tally.state_steps.items()},
        "state_command_max": {k: round(v, 6)
                              for k, v in tally.state_command_max.items()},
        "state_path_m": {k: round(v, 4) for k, v in tally.state_path_m.items()},
        "zero_command_states": list(ZERO_COMMAND_STATES),
        "zero_command_violations": list(tally.zero_command_violations),
        "zero_state_path_m": zero_state_path,
        "zero_state_path_bound_m": ZERO_STATE_PATH_M,
        "zero_drift_reference_m": ZERO_PATH_10S_M,
        "transitions": machine.transitions,
        "timeouts": machine.timeouts,

        # the route
        "route_length_m": round(rollout.route.length, 4),
        "route_bends": route_bend_report(rollout.route),
        "route_crossings": aperture_crossings(rollout.route),
        "leg_names": list(LEG_NAMES),
        "leg_bounds_m": [round(b, 4) for b in rollout.leg_bounds],
        "final_arc_s_m": final.get("route_arc_s_m"),
        "final_route_remaining_m": final.get("route_route_remaining_m"),

        # the doorway yields
        "yields": yields,
        "yield_count": len(yields),
        "exiters_used_door": exiters_used_door,
        "door_exiter_names": list(DOOR_EXITER_NAMES),

        # the lift
        "boarding": dict(machine.boarding),
        "occupants_used_lift": occupants_used_lift,
        "occupant_names": list(OCCUPANT_NAMES),
        "max_occupants_exited": max_occupants_exited,
        "min_occupants_required": MIN_OCCUPANTS_EXITED,
        "ride_seconds": round(len(ride_records) * dt, 3),
        "cabin_seconds": round(tally.cabin_steps * dt, 3),
        "min_cabin_margin_m": (round(tally.min_cabin_margin_m, 4)
                               if tally.cabin_steps else None),
        "cabin_outside_while_riding_steps": tally.cabin_outside_steps,

        # the zones
        "zone_worst": {
            name: {**entry, "worst_m": round(entry["worst_m"], 4)}
            for name, entry in sorted(tally.zone_worst.items())},
        "zone_violation_steps": dict(tally.zone_violation_steps),
        "aperture_steps": dict(tally.aperture_steps),
        "aperture_shared_steps": dict(tally.aperture_shared_steps),
        "aperture_shared_with": {
            name: sorted(names)
            for name, names in tally.aperture_shared_with.items()},
        "crossings": crossings,
        "door_schedule": schedule_windows(),
        "door_passable_fraction": DOOR_PASSABLE_FRACTION,

        # order relative to the guardian
        "min_guardian_gap_m": (round(tally.min_guardian_gap_m, 4)
                               if tally.guardian_gap_samples else None),
        "max_guardian_gap_m": (round(tally.max_guardian_gap_m, 4)
                               if tally.guardian_gap_samples else None),
        "overtake_steps": tally.overtake_steps,
        "guardian_gap_samples": tally.guardian_gap_samples,

        # the interlock
        "interlock_holds": tally.interlock_holds,
        "interlock_reasons": dict(tally.interlock_reasons),
        "controller_interlock_holds": rollout.controller.interlock_holds,

        # the cast
        "people": [p.name for p in PEOPLE],
        "other_adults": list(OTHER_NAMES),
        "moving_adults": moving_adults,
        "actor_moving_fraction": {
            name: round(value, 3)
            for name, value in moving_fraction(rollout.seconds).items()},
        "actor_routes": route_records(),
        "actor_names": list(ACTOR_NAMES),

        # visibility
        "visible_steps": tally.visible_steps,
        "los_steps": tally.los_steps,
        "visible_with_los_steps": tally.visible_with_los,
        "visible_fraction_with_los": round(
            _fraction(tally.visible_with_los, tally.los_steps), 4),
        "monitor_steps": tally.monitor_steps,
        "monitor_los_steps": tally.monitor_los_steps,
        "monitor_visible_with_los_steps": tally.monitor_visible_with_los,
        "monitor_visible_fraction_with_los": round(
            _fraction(tally.monitor_visible_with_los,
                      tally.monitor_los_steps), 4),
        "monitor_states": list(MONITOR_STATES),
        "blocked_by": dict(tally.blocked_by),
        "subject_visibility": subject_visibility,
        "subject_sequence": list(tally.subject_sequence),
        "subject_role_order": _distinct_roles(tally.subject_sequence),
        "expected_subject_role_order": list(expected_subject_order()),

        # the finding that shaped the building
        "spin_best_rate_dps": SPIN_BEST_RATE_DPS,
        "spin_180_seconds": round(SPIN_180_SECONDS, 1),
        "guardian_name": GUARDIAN.name,
    }
