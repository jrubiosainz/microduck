#!/usr/bin/env python3
"""Human-readable rendering of the acceptance summary.

Presentation only: every number here was computed by ``queue_metrics``.  Kept
separate so the gate logic can be read, and tested, without the printing code
around it.
"""

from __future__ import annotations

from queue_geometry import (
    JOIN_LATERAL_BAND_M,
    JOIN_LONGITUDINAL_MAX_M,
    JOIN_LONGITUDINAL_MIN_M,
)


def format_report(summary: dict) -> str:
    lines = ["QUEUE-POLITELY ACCEPTANCE REPORT", "=" * 64]
    lines.append(
        f"steps={summary['control_steps']}  seconds={summary['seconds']}  "
        f"path={summary['path_m']:.3f} m")
    lines.append(f"policy sha256={summary['policy_sha256'][:16]}...  "
                 f"action_scale={summary['action_scale']}  "
                 f"obs={summary['observation_dim']}D  "
                 f"gyro={summary['gyro_sensor']}")
    lines.append(f"states: {' -> '.join(summary['state_sequence'])}")
    lines.append("")
    lines.append(f"ORDER: {summary['order_samples_total']} samples, "
                 f"{summary['order_wrong_samples']} wrong order, "
                 f"{summary['tail_wrong_samples']} wrong tail")
    lines.append(f"  naive tails at decision: {summary['naive_tails_at_decision']}"
                 f"  true tail: {summary['true_tail_at_decision']}")
    lines.append(f"  wrong locks: {summary['wrong_lock_steps']}")
    lines.append("")
    lines.append("GAPS REFUSED THAT THE DUCK COULD HAVE TAKEN:")
    for gap in summary["rejected_available_gaps"]:
        lines.append(f"  {gap['gap']:<26s} {gap['kind']:<7s} "
                     f"sep={gap['separation_m']:.2f} m  {gap['reason']}")
    accepted = summary["accepted_gap"]
    if accepted:
        lines.append(f"  ACCEPTED {accepted['gap']} behind {accepted['ahead']}")
    join = summary["join_evidence"] or {}
    if join:
        lines.append(f"  JOIN at t={join.get('t', 0):.2f}s behind "
                     f"{join.get('behind')}: longitudinal="
                     f"{join.get('longitudinal_m', 0):.3f} m "
                     f"lateral={join.get('lateral_m', 0):.3f} m "
                     f"(bands {JOIN_LONGITUDINAL_MIN_M}-"
                     f"{JOIN_LONGITUDINAL_MAX_M} / {JOIN_LATERAL_BAND_M})")
    lines.append("")
    lines.append("CYCLES:")
    header = (f"  {'#':>2} {'kind':<11s} {'behind':<9s} {'path':>6} "
              f"{'arc':>6} {'stand':>6} {'xtrk':>6} {'cut':>6} {'track':>6}")
    lines.append(header)
    for cycle in summary["cycles"]:
        standoff = cycle["final_standoff_m"]
        tracked = cycle["tracked_fraction"]
        lines.append(
            f"  {cycle['index']:>2} {cycle['kind']:<11s} "
            f"{str(cycle['behind'] or '-'):<9s} {cycle['path_m']:6.3f} "
            f"{(cycle['arc_progress_m'] or 0.0):6.3f} "
            f"{(standoff if standoff is not None else float('nan')):6.3f} "
            f"{cycle['max_cross_track_m']:6.3f} {cycle['max_inside_cut_m']:6.3f} "
            f"{(tracked if tracked is not None else float('nan')):6.3f}")
    lines.append("")
    lines.append(f"counter reached at {summary['counter_reached_s']}s, "
                 f"last service at {summary['last_service_s']}s")
    lines.append(f"stationary command peaks: {summary['stationary_command_peak']}")
    lines.append(
        f"min person clearance {summary['min_person_clearance_m']:.4f} m "
        f"({summary['min_person_clearance_who']}), "
        f"min scenery {summary['min_scenery_clearance_m']:.4f} m "
        f"({summary['min_scenery_clearance_geom']})")
    lines.append(
        f"falls={summary['fallen_steps']}  contacts={summary['contact_steps']}  "
        f"min_z={summary['min_trunk_z_m']:.4f}  "
        f"final_z={summary['final_trunk_z_m']:.4f}")
    lines.append("")
    lines.append("GATES")
    for name, passed in summary["gates"].items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    lines.append("")
    lines.append("ALL GATES PASS" if summary["all_gates_pass"]
                 else "GATES FAILED")
    return "\n".join(lines)
