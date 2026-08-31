#!/usr/bin/env python3
"""Acceptance gates for move-away-crowd.

Every gate is evaluated from the rollout records and the closed cycles; nothing
here re-simulates or re-derives physics.  A gate never softens: if the behavior
cannot meet one, the behavior is changed, not the threshold.

Gate summary
------------
``cycles``            >= 4 completed avoidance cycles
``distinct_adults``   >= 3 different adults triggered them
``approach_sectors``  > 1 distinct approach bearing sector
``no_falls``          fallen_steps == 0
``min_trunk_z``       >= 0.09 m throughout
``final_trunk_z``     within 0.012 m of the nominal 0.116 m
``no_contact``        min clearance to any adult torso OR carried box > 0
``still_when_still``  command is EXACTLY zero outside EVADING
``evade_moved``       every evasion has real path length and net displacement
``evade_improved``    every evasion beats its own counterfactual clearance
``no_wrong_locks``    every lock was a genuine, top-ranked, tightest threat
``lock_visibility``   the locked adult was measurably seen during the encounter
"""

from __future__ import annotations

import math

import numpy as np

MIN_CYCLES = 4
MIN_DISTINCT_ADULTS = 3
MIN_APPROACH_SECTORS = 2
FALLEN_Z = 0.09
NOMINAL_Z = 0.116
FINAL_Z_TOLERANCE = 0.012
MIN_EVADE_PATH_M = 0.25
MIN_EVADE_NET_M = 0.15
MIN_CLEARANCE_GAIN_M = 0.05
MIN_LOCK_VISIBLE_PCT = 40.0
STATIONARY_STATES = ("SCANNING", "THREAT_LOCK", "SETTLING", "CLEAR")


def summarize(rollout) -> dict:
    records = rollout.records
    cycles = list(rollout.machine.cycles)
    completed = len(cycles)

    stationary = [r for r in records if r["state"] in STATIONARY_STATES]
    stationary_max = max(
        (float(np.linalg.norm(r["command"])) for r in stationary), default=0.0)
    evading = [r for r in records if r["state"] == "EVADING"]

    per_cycle = []
    for cycle in cycles:
        evade_path = float(cycle.get("evade_path_m", 0.0))
        evade_net = float(cycle.get("evade_net_m", 0.0))
        counterfactual = float(cycle.get("counterfactual_clearance_m", 0.0))
        achieved = float(cycle.get("end_predicted_clearance_m", 0.0))
        lock_steps = int(cycle.get("lock_steps", 0))
        lock_visible = int(cycle.get("lock_visible_steps", 0))
        evade_steps = int(cycle.get("evade_steps", 0))
        evade_visible = int(cycle.get("evade_visible_steps", 0))
        per_cycle.append({
            "cycle": cycle["cycle"],
            "threat": cycle["threat"],
            "carries_box": bool(cycle.get("carries_box", False)),
            "approach_sector": cycle.get("approach_sector"),
            "lock_s": cycle["lock_s"],
            "lock_bearing_deg": cycle.get("lock_bearing_deg"),
            "lock_clearance_m": cycle.get("lock_clearance_m"),
            "lock_ttc_s": cycle.get("lock_ttc_s"),
            "lock_range_m": cycle.get("lock_range_m"),
            "evade_start_s": cycle.get("evade_start_s"),
            "evade_end_s": cycle.get("evade_end_s"),
            "evade_duration_s": cycle.get("evade_duration_s"),
            "evade_timeout": bool(cycle.get("evade_timeout", False)),
            "evade_path_m": evade_path,
            "evade_net_m": evade_net,
            "counterfactual_clearance_m": counterfactual,
            "achieved_predicted_clearance_m": achieved,
            "clearance_gain_m": achieved - counterfactual,
            "actual_min_clearance_m": cycle.get("actual_min_clearance_m"),
            "lock_visible_pct": (
                100.0 * lock_visible / lock_steps if lock_steps else 0.0),
            "evade_visible_pct": (
                100.0 * evade_visible / evade_steps if evade_steps else 0.0),
            "lock_is_threat": bool(cycle.get("lock_is_threat", False)),
            "lock_top_ranked": bool(cycle.get("lock_top_ranked", False)),
            "lock_tightest_clearance": bool(
                cycle.get("lock_tightest_clearance", False)),
            "moved": evade_path >= MIN_EVADE_PATH_M and evade_net >= MIN_EVADE_NET_M,
            "improved": achieved - counterfactual >= MIN_CLEARANCE_GAIN_M,
        })

    distinct = sorted({c["threat"] for c in per_cycle})
    sectors = sorted({c["approach_sector"] for c in per_cycle if c["approach_sector"]})
    min_clearance_overall = min(
        (r["nearest_clearance_m"] for r in records), default=float("inf"))
    final_z = records[-1]["trunk_z_m"] if records else 0.0
    min_z = min((r["trunk_z_m"] for r in records), default=0.0)

    gates = {
        "cycles": completed >= MIN_CYCLES,
        "distinct_adults": len(distinct) >= MIN_DISTINCT_ADULTS,
        "approach_sectors": len(sectors) >= MIN_APPROACH_SECTORS,
        "no_falls": sum(r["trunk_z_m"] < FALLEN_Z for r in records) == 0,
        "min_trunk_z": min_z >= FALLEN_Z,
        "final_trunk_z": abs(final_z - NOMINAL_Z) <= FINAL_Z_TOLERANCE,
        "no_contact": min_clearance_overall > 0.0,
        "still_when_still": stationary_max == 0.0,
        "evade_moved": bool(per_cycle) and all(c["moved"] for c in per_cycle),
        "evade_improved": bool(per_cycle) and all(c["improved"] for c in per_cycle),
        "no_wrong_locks": bool(per_cycle) and all(
            c["lock_is_threat"] and c["lock_top_ranked"]
            and c["lock_tightest_clearance"] for c in per_cycle),
        "lock_visibility": bool(per_cycle) and all(
            c["lock_visible_pct"] >= MIN_LOCK_VISIBLE_PCT for c in per_cycle),
    }

    return {
        "duration_s": rollout.seconds,
        "control_steps": len(records),
        "ctrl_hz": 1.0 / rollout.dt,
        "decimation": rollout.decimation,
        "duck_planar_radius_m": rollout.duck_radius,
        "cycles_completed": completed,
        "distinct_adults": distinct,
        "approach_sectors": sectors,
        "cycles": per_cycle,
        "transitions": rollout.transitions,
        "min_trunk_z_m": min_z,
        "final_trunk_z_m": final_z,
        "nominal_trunk_z_m": NOMINAL_Z,
        "fallen_steps": sum(r["trunk_z_m"] < FALLEN_Z for r in records),
        "min_clearance_any_adult_m": min_clearance_overall,
        "contact_steps": sum(r["nearest_clearance_m"] <= 0.0 for r in records),
        "stationary_command_max": stationary_max,
        "stationary_steps": len(stationary),
        "evading_steps": len(evading),
        "evading_pct": 100.0 * len(evading) / len(records) if records else 0.0,
        "state_step_counts": {
            state: sum(r["state"] == state for r in records)
            for state in ("SCANNING", "THREAT_LOCK", "EVADING", "SETTLING", "CLEAR")
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def format_report(summary: dict) -> str:
    lines = [
        f"duration={summary['duration_s']:.1f}s  steps={summary['control_steps']}  "
        f"cycles={summary['cycles_completed']}  "
        f"adults={','.join(summary['distinct_adults'])}  "
        f"sectors={','.join(summary['approach_sectors'])}",
        f"min_z={summary['min_trunk_z_m']:.4f}  final_z={summary['final_trunk_z_m']:.4f}  "
        f"falls={summary['fallen_steps']}  "
        f"min_clearance={summary['min_clearance_any_adult_m']:.4f}  "
        f"still_cmd_max={summary['stationary_command_max']:.6f}",
        "",
        f"{'cyc':>3} {'threat':>7} {'box':>4} {'sec':>4} {'lock@s':>7} {'evade_s':>7} "
        f"{'path':>6} {'net':>6} {'cf':>6} {'got':>6} {'gain':>6} {'clr':>6} "
        f"{'vis%':>5} {'ev%':>5} ok",
    ]
    for cycle in summary["cycles"]:
        ok = "OK" if cycle["moved"] and cycle["improved"] else "FAIL"
        lines.append(
            f"{cycle['cycle']:>3} {cycle['threat']:>7} "
            f"{'YES' if cycle['carries_box'] else 'no':>4} "
            f"{str(cycle['approach_sector']):>4} {cycle['lock_s']:7.2f} "
            f"{cycle['evade_duration_s']:7.2f} {cycle['evade_path_m']:6.3f} "
            f"{cycle['evade_net_m']:6.3f} {cycle['counterfactual_clearance_m']:6.3f} "
            f"{cycle['achieved_predicted_clearance_m']:6.3f} "
            f"{cycle['clearance_gain_m']:+6.3f} "
            f"{cycle['actual_min_clearance_m']:6.3f} "
            f"{cycle['lock_visible_pct']:5.1f} {cycle['evade_visible_pct']:5.1f} {ok}"
        )
    lines.append("")
    for name, passed in summary["gates"].items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    lines.append(
        f"\nALL GATES: {'PASS' if summary['all_gates_pass'] else 'FAIL'}")
    return "\n".join(lines)
