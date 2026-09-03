#!/usr/bin/env python3
"""Acceptance gates for queue-politely.

Every gate is evaluated from the rollout records and the machine's own logs;
nothing here re-simulates or re-derives physics.  A gate never softens: if the
behavior cannot meet one, the BEHAVIOR is changed, not the threshold.

Gate summary
------------
``state_order``             the nine states occur in the required order and the
                            rollout ends in DONE
``order_inferred``          the inferred queue order equals the true order on
                            every sampled tick
``tail_correct``            the true tail is identified on every sampled tick
``no_wrong_locks``          zero ticks in which the duck was queueing behind
                            somebody who was not its actual predecessor
``naive_would_fail``        both coordinate heuristics name a DIFFERENT person
                            as the tail at decision time, so the projection is
                            load-bearing rather than decorative
``bystanders_excluded``     both off-path people are excluded, by measured
                            distance, on every tick
``rejected_enough``         at least two gaps refused that the duck could
                            PHYSICALLY have occupied
``joined_behind_tail``      the accepted gap is the one behind the true tail
``join_band``               the physical join is inside the lateral and
                            longitudinal bands behind the tail
``no_overtaking``           the duck is never ahead of a queue member still in
                            line
``wait_advance_cycles``     at least three completed WAIT->ADVANCE cycles
``stationary_command_zero`` command EXACTLY zero in every stationary state
``advances_are_real``       every advance produced real path length, real net
                            arc progress, and stopped 0.45-0.75 m behind its
                            predecessor
``bend_followed``           cross-track stayed inside its budget for the whole
                            of every advance, with no corner cutting
``predecessor_visible``     the person ahead was in the PiP for >= 95 % of every
                            advance in which the sightline existed
``counter_last``            the duck reached the counter only after every
                            predecessor had been served
``person_clearance``        minimum geometric duck/person clearance is positive
``scenery_clearance``       minimum geometric duck/scenery clearance is positive
``no_contacts``             zero steps with any non-positive clearance
``no_decorative_commands``  every nonzero forward command clears the measured
                            gait onset
``no_falls``                fallen_steps == 0
``min_trunk_z``             >= 0.09 m throughout
``final_trunk_z``           within 0.012 m of the nominal 0.116 m
``no_timeouts``             no phase hit its ceiling
"""

from __future__ import annotations

import numpy as np

from policy_runtime import FALLEN_TRUNK_Z, NOMINAL_TRUNK_Z
from queue_analysis import cycle_rows
from queue_constants import STATES, STATIONARY_STATES, VX_ONSET
from queue_geometry import (
    CORNER_CUT_LIMIT_M,
    CROSS_TRACK_LIMIT_M,
    queue_geometry_summary,
)
from queue_people import BYSTANDER_NAMES, QUEUE_NAMES, departures
from queue_report import format_report  # noqa: F401  (re-exported)

REQUIRED_ORDER = ("APPROACH", "OBSERVE_QUEUE", "IDENTIFY_TAIL",
                  "EVALUATE_GAPS", "JOIN", "WAIT", "ADVANCE",
                  "AT_COUNTER", "DONE")
MIN_WAIT_ADVANCE_CYCLES = 3
MIN_REJECTED_AVAILABLE = 2
TRACKING_FRACTION = 0.95
FINAL_Z_TOLERANCE = 0.012


def _states_in_order(records: list[dict]) -> list[str]:
    seen: list[str] = []
    for record in records:
        if not seen or seen[-1] != record["state"]:
            seen.append(record["state"])
    return seen


def summarize(rollout) -> dict:
    """Every measured quantity and every gate verdict, in one dictionary."""
    records = rollout.records
    machine = rollout.machine
    sequence = _states_in_order(records)

    # -- ordering ------------------------------------------------------
    samples = [s for s in rollout.order_samples if s["truth"]]
    order_wrong = [s for s in samples if not s["correct"]]
    tail_wrong = [s for s in samples if not s["tail_correct"]]

    # The naive readings, evaluated at the moment the duck actually decided.
    decision = rollout.first_reading or {}
    naive_tails = decision.get("naive_tails", {})
    true_tail_at_decision = (
        decision.get("truth", [None])[-1] if decision.get("truth") else None)
    naive_would_fail = bool(
        true_tail_at_decision is not None
        and naive_tails
        and all(value != true_tail_at_decision
                for value in naive_tails.values()))

    # -- gaps ----------------------------------------------------------
    rejected_available = decision.get("rejected_available", [])
    accepted = machine.accepted_gap or {}

    # -- wrong locks ---------------------------------------------------
    # A "wrong lock" is a tick in which the duck is queueing behind somebody who
    # is not the person immediately ahead of it in the true queue.
    wrong_locks = 0
    for record in records:
        subject = record["predecessor"]
        if subject is None or record["state"] not in ("JOIN", "ADVANCE", "WAIT"):
            continue
        truth = record["true_order"]
        ahead = [n for n in truth
                 if record["person_xy"].get(n) is not None
                 and record["person_in_queue"].get(n)]
        if ahead and subject != ahead[-1]:
            wrong_locks += 1

    # -- overtaking ----------------------------------------------------
    overtakes = []
    for record in records:
        if record["state"] in ("APPROACH", "OBSERVE_QUEUE", "IDENTIFY_TAIL",
                               "EVALUATE_GAPS"):
            continue
        for name in record["true_order"]:
            arc = record.get("person_arc_m", {}).get(name)
            if arc is None:
                continue
            if record["duck_arc_m"] < arc - 1e-6:
                overtakes.append({"t": record["t"], "person": name})
    # The duck's own remaining count is the authoritative statement of how many
    # people are still ahead of it, and it must never increase after the join.
    joined_index = next(
        (i for i, r in enumerate(records) if r["state"] == "WAIT"), None)
    remaining_series = [r["predecessors_remaining"]
                        for r in records[joined_index:]] if joined_index else []
    remaining_monotone = all(
        b <= a for a, b in zip(remaining_series, remaining_series[1:]))

    # -- cycles --------------------------------------------------------
    advance_cycles = cycle_rows(rollout, records)
    wait_advance = [c for c in advance_cycles if c["kind"] == "advance"]
    to_counter = [c for c in advance_cycles if c["kind"] == "to_counter"]
    # -- physics -------------------------------------------------------
    trunk_z = np.array([r["trunk_z_m"] for r in records], dtype=np.float64)
    fallen_steps = int(np.sum(trunk_z < FALLEN_TRUNK_Z))
    clearances = np.array([r["nearest_clearance_m"] for r in records])
    scenery = np.array([r["scenery_clearance_m"] for r in records])
    contact_steps = int(np.sum((clearances <= 0.0) | (scenery <= 0.0)))

    # -- commands ------------------------------------------------------
    stationary_peak = dict(rollout.stationary_command_max)
    decorative = [
        {"t": r["t"], "state": r["state"], "command": r["command"]}
        for r in records
        if 0.0 < abs(r["command"][0]) < VX_ONSET - 1e-9
    ]

    # -- counter -------------------------------------------------------
    counter_records = [r for r in records if r["state"] == "AT_COUNTER"]
    counter_t = counter_records[0]["t"] if counter_records else None
    served = departures(rollout.seconds)
    last_service = max((e["served_at_s"] for e in served), default=None)
    counter_after_all = bool(
        counter_t is not None and last_service is not None
        and counter_t >= last_service)

    # -- bystanders ----------------------------------------------------
    bystander_included = [
        {"t": r["t"], "who": name}
        for r in records for name in BYSTANDER_NAMES
        if name in r["inferred_order"]
    ]

    # -- tracking ------------------------------------------------------
    tracked_fractions = [c["tracked_fraction"] for c in advance_cycles
                         if c["tracked_fraction"] is not None]

    summary = {
        "seconds": rollout.seconds,
        "control_steps": len(records),
        "policy_sha256": _sha256(rollout.policy.path),
        "action_scale": 0.9,
        "observation_dim": 61,
        "gyro_sensor": "imu_ang_vel",
        "duck_planar_radius_m": round(rollout.duck_radius, 4),
        "duck_exact_planar_radius_m": round(rollout.duck_exact_radius, 4),
        "adult_half_extent_m": round(rollout.adult_half_extent, 4),
        "geometry": queue_geometry_summary(),
        "state_sequence": sequence,
        "transitions": machine.transitions,
        "timeouts": machine.timeouts,
        "decision_reading": decision,
        "accepted_gap": accepted,
        "rejected_gaps": machine.rejected_gaps,
        "rejected_available_gaps": rejected_available,
        "join_evidence": rollout.join_evidence,
        "order_samples_total": len(samples),
        "order_wrong_samples": len(order_wrong),
        "tail_wrong_samples": len(tail_wrong),
        "first_wrong_order": order_wrong[0] if order_wrong else None,
        "naive_tails_at_decision": naive_tails,
        "true_tail_at_decision": true_tail_at_decision,
        "wrong_lock_steps": wrong_locks,
        "overtaking_events": overtakes[:8],
        "remaining_monotone": bool(remaining_monotone),
        "cycles": advance_cycles,
        "wait_advance_cycles": len(wait_advance),
        "to_counter_cycles": len(to_counter),
        "stationary_command_peak": stationary_peak,
        "decorative_commands": decorative[:8],
        "counter_reached_s": counter_t,
        "last_service_s": last_service,
        "services": served,
        "bystanders_wrongly_included": bystander_included[:8],
        "path_m": round(rollout.path_m, 4),
        "min_person_clearance_m": round(rollout.min_person_clearance, 4),
        "min_person_clearance_who": rollout.min_person_name,
        "min_scenery_clearance_m": round(rollout.min_scenery_clearance, 4),
        "min_scenery_clearance_geom": rollout.min_scenery_geom,
        "contact_steps": contact_steps,
        "fallen_steps": fallen_steps,
        "min_trunk_z_m": round(float(trunk_z.min()), 4),
        "final_trunk_z_m": round(float(trunk_z[-1]), 4),
        "tracked_fractions": tracked_fractions,
    }

    summary["gates"] = {
        "state_order": _order_ok(sequence),
        "order_inferred": len(samples) > 0 and not order_wrong,
        "tail_correct": len(samples) > 0 and not tail_wrong,
        "no_wrong_locks": wrong_locks == 0,
        "naive_would_fail": naive_would_fail,
        "bystanders_excluded": not bystander_included,
        "rejected_enough": len(rejected_available) >= MIN_REJECTED_AVAILABLE,
        "joined_behind_tail": bool(
            accepted.get("gap") == "behind_tail"
            and accepted.get("ahead") == true_tail_at_decision),
        "join_band": bool((rollout.join_evidence or {}).get("in_band")),
        "no_overtaking": not overtakes and remaining_monotone,
        "wait_advance_cycles": len(wait_advance) >= MIN_WAIT_ADVANCE_CYCLES,
        "stationary_command_zero": all(
            value == 0.0 for value in stationary_peak.values()),
        "advances_are_real": bool(advance_cycles) and all(
            c["path_m"] > 0.10 and (c["arc_progress_m"] or 0.0) > 0.10
            for c in advance_cycles) and all(
            c["standoff_in_band"] for c in wait_advance
            if c["standoff_gradeable"]),
        "bend_followed": bool(advance_cycles) and all(
            c["max_cross_track_m"] <= CROSS_TRACK_LIMIT_M
            and c["max_inside_cut_m"] <= CORNER_CUT_LIMIT_M
            for c in advance_cycles),
        "predecessor_visible": bool(tracked_fractions) and all(
            value >= TRACKING_FRACTION for value in tracked_fractions),
        "counter_last": counter_after_all,
        "person_clearance": rollout.min_person_clearance > 0.0,
        "scenery_clearance": rollout.min_scenery_clearance > 0.0,
        "no_contacts": contact_steps == 0,
        "no_decorative_commands": not decorative,
        "no_falls": fallen_steps == 0,
        "min_trunk_z": float(trunk_z.min()) >= FALLEN_TRUNK_Z,
        "final_trunk_z": abs(
            float(trunk_z[-1]) - NOMINAL_TRUNK_Z) <= FINAL_Z_TOLERANCE,
        "no_timeouts": not machine.timeouts,
    }
    summary["all_gates_pass"] = all(summary["gates"].values())
    return summary


def _order_ok(sequence: list[str]) -> bool:
    """Every required state occurs, in order, and the rollout ends in DONE."""
    if not sequence or sequence[-1] != "DONE":
        return False
    cursor = 0
    for state in sequence:
        if state not in STATES:
            return False
        if cursor < len(REQUIRED_ORDER) and state == REQUIRED_ORDER[cursor]:
            cursor += 1
    return cursor == len(REQUIRED_ORDER)


def _sha256(path) -> str:
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
