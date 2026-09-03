#!/usr/bin/env python3
"""Acceptance gates for crosswalk-guardian.

Every gate is evaluated from the rollout records and the machine's own logs;
nothing here re-simulates or re-derives physics.  A gate never softens: if the
behavior cannot meet one, the behavior is changed, not the threshold.

Gate summary
------------
``state_order``          the eight states occur exactly once, in order
``stops_before_line``    the duck stops with its footprint behind the wait line
``no_early_encroach``    zero encroachment on the wait line before CROSSING
``scan_phases``          all three LOOK phases occurred, in order
``scan_sectors_seen``    the required road sector was genuinely visible in the
                         exact PiP camera during each LOOK phase
``rejected_unsafe_gap``  at least one unsafe gap was explicitly refused
``commit_margin``        the accepted gap cleared SAFETY_MARGIN_S for every road
                         user across the whole estimated crossing
``crossing_continuous``  no stop and no zero-command plateau inside the road
``crossing_moved``       real path length and net displacement while crossing
``reaches_safe_zone``    the duck ends inside the far-side safe zone
``final_state_safe``     the last state is SAFE
``no_reverse``           the duck never moves back toward the road after SAFE
``no_vehicle_contact``   zero geometric overlap with any road user
``positive_clearance``   minimum measured clearance is strictly positive
``still_when_still``     command is EXACTLY zero in every stationary state
``no_falls``             fallen_steps == 0
``min_trunk_z``          >= 0.09 m throughout
``final_trunk_z``        within 0.012 m of the nominal 0.116 m
``no_timeouts``          no phase hit its ceiling
"""

from __future__ import annotations

import numpy as np

from conflict import SAFETY_MARGIN_S, STATES, STATIONARY_STATES
from guardian_model import LOOK_PHASES
from street import DUCK_PLANAR_RADIUS, WAIT_LINE_X

FALLEN_Z = 0.09
NOMINAL_Z = 0.116
FINAL_Z_TOLERANCE = 0.012
# Evidence thresholds for the crossing being a real physical traverse rather
# than a state label.  The road plus both verges is 2.05 m wide, so a genuine
# crossing cannot be much under 1.8 m of net displacement.
MIN_CROSSING_PATH_M = 1.80
MIN_CROSSING_NET_M = 1.70
# A "zero-command plateau" is the failure this behavior exists to avoid: the
# duck standing still in a live traffic lane.  Any run of consecutive in-road
# steps with a zero locomotion command longer than this fails the gate.  Two
# ticks (40 ms) tolerates a single transition tick without tolerating a stop.
MAX_ZERO_COMMAND_STEPS_IN_ROAD = 2
# Likewise for physical motion: the duck must actually be advancing while in
# the road.  A plateau is a run of in-road steps whose total x-advance is below
# this, which catches a duck that is commanded to walk but is not moving.
MIN_IN_ROAD_ADVANCE_M = 0.020
PLATEAU_WINDOW_STEPS = 25          # 0.5 s at 50 Hz
# After SAFE, the duck must not drift back toward the road.
MAX_REVERSE_M = 0.05


def _state_runs(records: list[dict]) -> list[tuple[str, int, int]]:
    """Contiguous runs of the same state: ``(state, first_index, last_index)``."""
    runs: list[tuple[str, int, int]] = []
    for index, record in enumerate(records):
        if runs and runs[-1][0] == record["state"]:
            runs[-1] = (runs[-1][0], runs[-1][1], index)
        else:
            runs.append((record["state"], index, index))
    return runs


def _longest_zero_command_run_in_road(records: list[dict]) -> int:
    longest = current = 0
    for record in records:
        in_road_now = record["in_road"]
        zero = all(abs(v) < 1e-9 for v in record["command"])
        if in_road_now and zero:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _worst_in_road_advance(records: list[dict],
                           window: int = PLATEAU_WINDOW_STEPS) -> float:
    """Smallest x-advance over any sliding window entirely inside the road.

    This is the physical counterpart of the zero-command test: a duck that is
    commanded to walk but has stalled would pass the command test and fail this
    one.
    """
    xs = [r["duck_xy"][0] for r in records]
    flags = [r["in_road"] for r in records]
    worst = float("inf")
    for start in range(len(records) - window):
        stop = start + window
        if not all(flags[start:stop + 1]):
            continue
        worst = min(worst, xs[stop] - xs[start])
    return worst


def summarize(rollout) -> dict:
    records = rollout.records
    machine = rollout.machine
    runs = _state_runs(records)
    run_states = [state for state, _, _ in runs]

    stationary = [r for r in records if r["state"] in STATIONARY_STATES]
    stationary_max = max(
        (float(np.linalg.norm(r["command"])) for r in stationary), default=0.0)
    crossing = [r for r in records if r["state"] == "CROSSING"]
    in_road_records = [r for r in records if r["in_road"]]

    # Where the duck actually came to rest before the scan, and how much room
    # its leading surface had.
    stop_records = [r for r in records if r["state"] == "STOP"]
    stop_x = stop_records[-1]["duck_xy"][0] if stop_records else float("nan")
    stop_leading = stop_x + DUCK_PLANAR_RADIUS
    stop_margin = -WAIT_LINE_X - stop_leading

    scan_log = list(machine.scan_log)
    scan_order = [entry["phase"] for entry in scan_log]
    expected_scan = [phase for phase, _, _ in LOOK_PHASES]

    # Per-LOOK-phase measured sector visibility, from the records rather than
    # from the machine's own confirmation counter, so the two can disagree and
    # be caught.
    scan_evidence = []
    for phase, _, sector in LOOK_PHASES:
        phase_records = [r for r in records if r["state"] == phase]
        key = f"{sector}_visible"
        fraction_key = f"{sector}_fraction"
        seen = sum(bool(r[key]) for r in phase_records)
        scan_evidence.append({
            "phase": phase,
            "sector": sector,
            "steps": len(phase_records),
            "sector_visible_steps": seen,
            "sector_visible_pct": (
                100.0 * seen / len(phase_records) if phase_records else 0.0),
            "peak_sector_fraction": max(
                (r[fraction_key] for r in phase_records), default=0.0),
            "start_s": phase_records[0]["t"] if phase_records else None,
            "end_s": phase_records[-1]["t"] if phase_records else None,
            "occurred": bool(phase_records),
            "sector_seen": seen > 0,
        })

    rejected = machine.rejected_gaps
    commit = dict(machine.commit)

    min_clearance = min(
        (r["nearest_clearance_m"] for r in records), default=float("inf"))
    contact_steps = sum(r["nearest_clearance_m"] <= 0.0 for r in records)
    final_z = records[-1]["trunk_z_m"] if records else 0.0
    min_z = min((r["trunk_z_m"] for r in records), default=0.0)

    safe_records = [r for r in records if r["state"] == "SAFE"]
    if safe_records:
        max_x_in_safe = max(r["duck_xy"][0] for r in safe_records)
        final_x = safe_records[-1]["duck_xy"][0]
        reverse_m = max_x_in_safe - final_x
    else:
        reverse_m = 0.0
        final_x = records[-1]["duck_xy"][0] if records else float("nan")

    longest_zero_run = _longest_zero_command_run_in_road(records)
    worst_advance = _worst_in_road_advance(records)
    crossing_path = rollout.crossing_path_m
    crossing_net = rollout.crossing_net_m

    gates = {
        "state_order": run_states == list(STATES),
        "stops_before_line": bool(stop_records) and stop_margin > 0.0,
        "no_early_encroach": rollout.min_wait_line_margin > 0.0,
        "scan_phases": scan_order == expected_scan,
        "scan_sectors_seen": bool(scan_evidence) and all(
            e["occurred"] and e["sector_seen"] for e in scan_evidence),
        "rejected_unsafe_gap": len(rejected) >= 1,
        "commit_margin": bool(commit) and float(
            commit.get("worst_margin_s", -1.0)) >= SAFETY_MARGIN_S,
        "crossing_continuous": longest_zero_run <= MAX_ZERO_COMMAND_STEPS_IN_ROAD
        and (worst_advance == float("inf")
             or worst_advance >= MIN_IN_ROAD_ADVANCE_M),
        "crossing_moved": crossing_path >= MIN_CROSSING_PATH_M
        and crossing_net >= MIN_CROSSING_NET_M,
        "reaches_safe_zone": bool(records) and records[-1]["in_safe_zone"],
        "final_state_safe": bool(records) and records[-1]["state"] == "SAFE",
        "no_reverse": reverse_m <= MAX_REVERSE_M,
        "no_vehicle_contact": contact_steps == 0,
        "positive_clearance": min_clearance > 0.0,
        "still_when_still": stationary_max == 0.0,
        "no_falls": sum(r["trunk_z_m"] < FALLEN_Z for r in records) == 0,
        "min_trunk_z": min_z >= FALLEN_Z,
        "final_trunk_z": abs(final_z - NOMINAL_Z) <= FINAL_Z_TOLERANCE,
        "no_timeouts": not machine.timeouts,
    }

    return {
        "duration_s": rollout.seconds,
        "control_steps": len(records),
        "ctrl_hz": 1.0 / rollout.dt,
        "decimation": rollout.decimation,
        "duck_planar_radius_m": rollout.duck_radius,
        "safety_margin_s": SAFETY_MARGIN_S,
        "state_sequence": run_states,
        "expected_state_sequence": list(STATES),
        "state_step_counts": {
            state: sum(r["state"] == state for r in records) for state in STATES
        },
        "transitions": rollout.transitions,
        "stop_x_m": stop_x,
        "stop_leading_edge_x_m": stop_leading,
        "stop_wait_line_margin_m": stop_margin,
        "min_wait_line_margin_m": rollout.min_wait_line_margin,
        "max_x_before_crossing_m": rollout.max_x_before_crossing,
        "wait_line_x_m": -WAIT_LINE_X,
        "scan": scan_evidence,
        "scan_log": scan_log,
        "rejected_gaps": rejected,
        "rejected_gap_count": len(rejected),
        "gap_decisions": machine.gap_decisions,
        "commitment": commit,
        "crossing_path_m": crossing_path,
        "crossing_net_m": crossing_net,
        "crossing_duration_s": commit.get("crossing_duration_s"),
        "crossing_estimate_s": commit.get("crossing_duration_estimate_s"),
        "wait_duration_s": commit.get("wait_duration_s"),
        "total_path_m": rollout.path_m,
        "longest_zero_command_run_in_road": longest_zero_run,
        "worst_in_road_advance_m": (
            None if worst_advance == float("inf") else worst_advance),
        "in_road_steps": len(in_road_records),
        "crossing_steps": len(crossing),
        "final_x_m": final_x,
        "reverse_after_safe_m": reverse_m,
        "min_vehicle_clearance_m": min_clearance,
        "contact_steps": contact_steps,
        "stationary_command_max": stationary_max,
        "stationary_steps": len(stationary),
        "min_trunk_z_m": min_z,
        "final_trunk_z_m": final_z,
        "nominal_trunk_z_m": NOMINAL_Z,
        "fallen_steps": sum(r["trunk_z_m"] < FALLEN_Z for r in records),
        "timeouts": list(machine.timeouts),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def format_report(summary: dict) -> str:
    lines = [
        f"duration={summary['duration_s']:.1f}s  "
        f"steps={summary['control_steps']}  "
        f"states={'→'.join(summary['state_sequence'])}",
        f"min_z={summary['min_trunk_z_m']:.4f}  "
        f"final_z={summary['final_trunk_z_m']:.4f}  "
        f"falls={summary['fallen_steps']}  "
        f"min_clearance={summary['min_vehicle_clearance_m']:.4f}  "
        f"still_cmd_max={summary['stationary_command_max']:.6f}",
        f"stop_x={summary['stop_x_m']:.3f} "
        f"(leading edge {summary['stop_leading_edge_x_m']:+.3f}, "
        f"wait line {summary['wait_line_x_m']:+.3f}, "
        f"margin {summary['stop_wait_line_margin_m']:+.3f} m)  "
        f"min margin before crossing {summary['min_wait_line_margin_m']:+.3f} m",
        "",
        "scan phases:",
    ]
    for entry in summary["scan"]:
        lines.append(
            f"  {entry['phase']:<16s} sector={entry['sector']:<5s} "
            f"{entry['start_s']:6.2f}-{entry['end_s']:6.2f}s  "
            f"steps={entry['steps']:4d}  "
            f"sector visible {entry['sector_visible_pct']:5.1f}% "
            f"(peak fraction {entry['peak_sector_fraction']:.2f})")
    lines.append("")
    lines.append(f"rejected gaps: {summary['rejected_gap_count']}")
    for entry in summary["rejected_gaps"]:
        blocking = ", ".join(
            f"{b['vehicle']}({b['lane']}) margin={b['margin_s']:+.2f}s"
            for b in entry["blocking"][:3])
        lines.append(
            f"  {entry['first_rejected_at_s']:6.2f}-"
            f"{entry['last_rejected_at_s']:6.2f}s  "
            f"worst={entry['worst_margin_s']:+.2f}s  {blocking}")
    commit = summary["commitment"]
    if commit:
        lines.append("")
        lines.append(
            f"committed at t={commit.get('committed_at_s', float('nan')):.2f}s "
            f"after waiting {commit.get('wait_duration_s', float('nan')):.2f}s  "
            f"margin={commit.get('worst_margin_s', float('nan')):+.2f}s "
            f"(limiting {commit.get('limiting_vehicle')})  "
            f"estimate={commit.get('crossing_duration_estimate_s', float('nan')):.2f}s "
            f"actual={commit.get('crossing_duration_s', float('nan')):.2f}s")
    lines.append(
        f"crossing: path={summary['crossing_path_m']:.3f} m  "
        f"net={summary['crossing_net_m']:.3f} m  "
        f"longest zero-command run in road="
        f"{summary['longest_zero_command_run_in_road']} steps  "
        f"worst 0.5 s advance in road={summary['worst_in_road_advance_m']}")
    lines.append(
        f"final x={summary['final_x_m']:.3f} m  "
        f"reverse after SAFE={summary['reverse_after_safe_m']:.4f} m")
    if summary["timeouts"]:
        lines.append(f"TIMEOUTS: {', '.join(summary['timeouts'])}")
    lines.append("")
    for name, passed in summary["gates"].items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    lines.append(f"\nALL GATES: {'PASS' if summary['all_gates_pass'] else 'FAIL'}")
    return "\n".join(lines)
