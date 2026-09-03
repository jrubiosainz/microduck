#!/usr/bin/env python3
"""Acceptance gates for come-here-recall.

Every gate is evaluated from the rollout records and the closed cycles; nothing
here re-simulates or re-derives physics.  A gate never softens: if the behavior
cannot meet one, the behavior is changed, not the threshold.

Gate summary
------------
``recalls``             >= 3 completed recall cycles
``distinct_callers``    >= 3 different adults called the duck
``caller_order``        the completed order matches the intended order exactly
``distinct_bearings``   the call bearings are genuinely spread around the duck
``no_wrong_locks``      every lock was the adult who was actually calling
``locks_were_seen``     every lock happened with the caller inside the camera
                        acquisition gate - never on world geometry alone
``approach_visibility`` caller visible for >= 95% of APPROACH steps
``arrived_visibility``  caller visible at the ARRIVED instant of every cycle
``approach_moved``      every approach has real path length and net displacement
``approach_closed``     every approach measurably reduced range to its caller
``standoff_band``       every arrival settles inside the safe standoff band
``faces_caller``        the TRUNK points at the caller at the end of each cycle
``still_when_still``    command is EXACTLY zero outside APPROACH
``no_caller_change``    the locked identity never changes inside a cycle
``no_person_contact``   the duck never touches any adult's surface
``no_falls``            fallen_steps == 0
``min_trunk_z``         >= 0.09 m throughout
``final_trunk_z``       within 0.012 m of the nominal 0.116 m
``refused_interrupt``   a call arriving mid-cycle was refused, not obeyed
"""

from __future__ import annotations

import math

import numpy as np

MIN_RECALLS = 3
MIN_DISTINCT_CALLERS = 3
MIN_BEARING_SEPARATION_DEG = 60.0
FALLEN_Z = 0.09
NOMINAL_Z = 0.116
FINAL_Z_TOLERANCE = 0.012
MIN_APPROACH_PATH_M = 0.60
MIN_APPROACH_NET_M = 0.40
# Range must fall by at least this much across an approach, so a cycle cannot
# pass by starting already close.
MIN_RANGE_REDUCTION_M = 0.50
MIN_APPROACH_VISIBLE_PCT = 95.0
# Trunk-yaw error at ARRIVED.  The duck must END UP FACING the caller, not
# merely near them.  This is measured on the PHYSICAL TRUNK, not on head gaze:
# gaze lives in an isolated rendering MjData and could be aimed anywhere
# without the body having turned at all, so grading the body is the honest
# test.  30 deg keeps the caller well inside the duck's forward arc while
# allowing the arrival the measured turn authority can actually achieve - the
# policy cannot turn in place, so the final heading is whatever the closing arc
# produced when the standoff range was reached.
MAX_FACING_ERROR_DEG = 30.0
STATIONARY_STATES = ("LISTEN", "SEARCH", "CALLER_LOCK", "ARRIVED")


def summarize(rollout, *, expected_order: tuple[str, ...],
              standoff_min: float, standoff_max: float) -> dict:
    records = rollout.records
    cycles = list(rollout.machine.cycles)
    completed = len(cycles)

    stationary = [r for r in records if r["state"] in STATIONARY_STATES]
    stationary_max = max(
        (float(np.linalg.norm(r["command"])) for r in stationary), default=0.0)
    approaching = [r for r in records if r["state"] == "APPROACH"]

    per_cycle = []
    for cycle in cycles:
        path = float(cycle.get("approach_path_m", 0.0))
        net = float(cycle.get("approach_net_m", 0.0))
        start_range = float(cycle.get("approach_start_range_m") or 0.0)
        min_range = float(cycle.get("approach_min_range_m", float("inf")))
        final_range = float(cycle.get("final_range_m") or 0.0)
        approach_steps = int(cycle.get("approach_steps", 0))
        approach_visible = int(cycle.get("approach_visible_steps", 0))
        arrived_steps = int(cycle.get("arrived_steps", 0))
        arrived_visible = int(cycle.get("arrived_visible_steps", 0))
        visible_pct = (
            100.0 * approach_visible / approach_steps if approach_steps else 0.0)
        arrived_pct = (
            100.0 * arrived_visible / arrived_steps if arrived_steps else 0.0)
        per_cycle.append({
            "cycle": cycle["cycle"],
            "caller": cycle["caller"],
            "call_start_s": cycle.get("call_start_s"),
            "search_start_s": cycle.get("search_start_s"),
            "search_duration_s": cycle.get("search_duration_s"),
            "lock_s": cycle.get("lock_s"),
            "lock_range_m": cycle.get("lock_range_m"),
            "lock_off_axis_deg": cycle.get("lock_off_axis_deg"),
            "lock_gate_open": bool(cycle.get("lock_gate_open", False)),
            "lock_caller_visible": bool(cycle.get("lock_caller_visible", False)),
            "lock_is_active_caller": bool(cycle.get("lock_is_active_caller", False)),
            "call_bearing_deg": cycle.get("call_bearing_deg"),
            "approach_start_s": cycle.get("approach_start_s"),
            "approach_end_s": cycle.get("approach_end_s"),
            "approach_duration_s": cycle.get("approach_duration_s"),
            "approach_timeout": bool(cycle.get("approach_timeout", False)),
            "approach_start_range_m": start_range,
            "approach_min_range_m": min_range,
            "arrival_range_m": cycle.get("arrival_range_m"),
            "final_range_m": final_range,
            "range_reduction_m": start_range - min_range,
            "approach_path_m": path,
            "approach_net_m": net,
            "approach_visible_pct": visible_pct,
            "arrived_visible_pct": arrived_pct,
            "min_caller_clearance_m": cycle.get("min_caller_clearance_m"),
            "final_facing_error_deg": cycle.get("final_facing_error_deg"),
            "caller_changed": bool(cycle.get("caller_changed", False)),
            "moved": path >= MIN_APPROACH_PATH_M and net >= MIN_APPROACH_NET_M,
            "closed": (start_range - min_range) >= MIN_RANGE_REDUCTION_M,
            "in_band": standoff_min <= final_range <= standoff_max,
            "facing": abs(float(cycle.get("final_facing_error_deg", 180.0)))
            <= MAX_FACING_ERROR_DEG,
        })

    order = tuple(c["caller"] for c in per_cycle)
    distinct = sorted({c["caller"] for c in per_cycle})
    bearings = [c["call_bearing_deg"] for c in per_cycle
                if c["call_bearing_deg"] is not None]
    bearing_gaps = [
        abs((bearings[i] - bearings[j] + 180.0) % 360.0 - 180.0)
        for i in range(len(bearings)) for j in range(i + 1, len(bearings))
    ]
    min_bearing_gap = min(bearing_gaps) if bearing_gaps else 0.0

    min_clearance_overall = min(
        (r["nearest_clearance_m"] for r in records), default=float("inf"))
    final_z = records[-1]["trunk_z_m"] if records else 0.0
    min_z = min((r["trunk_z_m"] for r in records), default=0.0)
    refusals = list(rollout.machine.refused_calls)

    gates = {
        "recalls": completed >= MIN_RECALLS,
        "distinct_callers": len(distinct) >= MIN_DISTINCT_CALLERS,
        "caller_order": order == tuple(expected_order[:len(order)])
        and len(order) >= MIN_RECALLS,
        "distinct_bearings": min_bearing_gap >= MIN_BEARING_SEPARATION_DEG,
        "no_wrong_locks": bool(per_cycle) and all(
            c["lock_is_active_caller"] for c in per_cycle),
        "locks_were_seen": bool(per_cycle) and all(
            c["lock_gate_open"] and c["lock_caller_visible"] for c in per_cycle),
        "approach_visibility": bool(per_cycle) and all(
            c["approach_visible_pct"] >= MIN_APPROACH_VISIBLE_PCT
            for c in per_cycle),
        "arrived_visibility": bool(per_cycle) and all(
            c["arrived_visible_pct"] >= 100.0 for c in per_cycle),
        "approach_moved": bool(per_cycle) and all(c["moved"] for c in per_cycle),
        "approach_closed": bool(per_cycle) and all(c["closed"] for c in per_cycle),
        "standoff_band": bool(per_cycle) and all(c["in_band"] for c in per_cycle),
        "faces_caller": bool(per_cycle) and all(c["facing"] for c in per_cycle),
        "still_when_still": stationary_max == 0.0,
        "no_caller_change": bool(per_cycle) and not any(
            c["caller_changed"] for c in per_cycle),
        "no_person_contact": min_clearance_overall > 0.0,
        "no_falls": sum(r["trunk_z_m"] < FALLEN_Z for r in records) == 0,
        "min_trunk_z": min_z >= FALLEN_Z,
        "final_trunk_z": abs(final_z - NOMINAL_Z) <= FINAL_Z_TOLERANCE,
        "refused_interrupt": len(refusals) >= 1,
    }

    return {
        "duration_s": rollout.seconds,
        "control_steps": len(records),
        "ctrl_hz": 1.0 / rollout.dt,
        "decimation": rollout.decimation,
        "duck_planar_radius_m": rollout.duck_radius,
        "standoff_band_m": [standoff_min, standoff_max],
        "max_facing_error_deg": MAX_FACING_ERROR_DEG,
        "expected_order": list(expected_order),
        "completed_order": list(order),
        "recalls_completed": completed,
        "distinct_callers": distinct,
        "min_call_bearing_gap_deg": min_bearing_gap,
        "cycles": per_cycle,
        "transitions": rollout.transitions,
        "refused_calls": refusals,
        "min_trunk_z_m": min_z,
        "final_trunk_z_m": final_z,
        "nominal_trunk_z_m": NOMINAL_Z,
        "fallen_steps": sum(r["trunk_z_m"] < FALLEN_Z for r in records),
        "min_clearance_any_adult_m": min_clearance_overall,
        "contact_steps": sum(r["nearest_clearance_m"] <= 0.0 for r in records),
        "stationary_command_max": stationary_max,
        "stationary_steps": len(stationary),
        "approach_steps": len(approaching),
        "approach_pct": 100.0 * len(approaching) / len(records) if records else 0.0,
        "state_step_counts": {
            state: sum(r["state"] == state for r in records)
            for state in ("LISTEN", "SEARCH", "CALLER_LOCK", "APPROACH", "ARRIVED")
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def format_report(summary: dict) -> str:
    lines = [
        f"duration={summary['duration_s']:.1f}s  steps={summary['control_steps']}  "
        f"recalls={summary['recalls_completed']}  "
        f"order={'→'.join(summary['completed_order']) or '-'}  "
        f"min_bearing_gap={summary['min_call_bearing_gap_deg']:.1f}°",
        f"min_z={summary['min_trunk_z_m']:.4f}  "
        f"final_z={summary['final_trunk_z_m']:.4f}  "
        f"falls={summary['fallen_steps']}  "
        f"min_clearance={summary['min_clearance_any_adult_m']:.4f}  "
        f"still_cmd_max={summary['stationary_command_max']:.6f}  "
        f"refused={len(summary['refused_calls'])}",
        "",
        f"{'cyc':>3} {'caller':>7} {'call@':>6} {'srch':>5} {'lock@':>6} "
        f"{'off°':>5} {'appr':>5} {'r0':>5} {'rmin':>5} {'rfin':>5} "
        f"{'path':>5} {'net':>5} {'face°':>6} {'vis%':>5} {'arr%':>5} ok",
    ]
    for cycle in summary["cycles"]:
        ok = "OK" if (cycle["moved"] and cycle["closed"] and cycle["in_band"]
                      and cycle["facing"]) else "FAIL"
        lines.append(
            f"{cycle['cycle']:>3} {cycle['caller']:>7} "
            f"{cycle['call_start_s']:6.2f} {cycle['search_duration_s']:5.2f} "
            f"{cycle['lock_s']:6.2f} {cycle['lock_off_axis_deg']:5.1f} "
            f"{cycle['approach_duration_s']:5.2f} "
            f"{cycle['approach_start_range_m']:5.2f} "
            f"{cycle['approach_min_range_m']:5.2f} "
            f"{cycle['final_range_m']:5.2f} {cycle['approach_path_m']:5.2f} "
            f"{cycle['approach_net_m']:5.2f} "
            f"{cycle['final_facing_error_deg']:+6.1f} "
            f"{cycle['approach_visible_pct']:5.1f} "
            f"{cycle['arrived_visible_pct']:5.1f} {ok}"
        )
    if summary["refused_calls"]:
        lines.append("")
        for entry in summary["refused_calls"]:
            lines.append(
                f"  refused call from {entry['caller']} at "
                f"t={entry['refused_at_s']:.2f}s while {entry['state']} "
                f"with {entry['busy_with']}")
    lines.append("")
    for name, passed in summary["gates"].items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    lines.append(f"\nALL GATES: {'PASS' if summary['all_gates_pass'] else 'FAIL'}")
    return "\n".join(lines)
