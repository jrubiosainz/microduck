#!/usr/bin/env python3
"""Headless gate: run the whole behavior with no rendering and grade it.

This is the loop that development happens in.  It imports nothing from a
rendering stack, so a validation run has no PIL, imageio or GPU dependency at
all — which is also the claim ``test_the_headless_gate_imports_no_rendering_stack``
pins.

Run with:
    ../../microduck_rl/.venv/bin/python scripts/validate_guide.py --seconds 82
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--seconds", type=float, default=82.0)
    parser.add_argument("--destination", default=None,
                        help="override the requested destination key")
    parser.add_argument("--json", default="")
    parser.add_argument("--trace", default="",
                        help="write the full per-tick record stream here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from guide_metrics import report, summarize
    from guide_states import REQUESTED_DESTINATION
    from rollout_guide import GuideRollout

    rollout = GuideRollout(args.policy, args.seconds,
                           requested=args.destination or REQUESTED_DESTINATION)

    def progress(index, record):
        if args.quiet or index % 250:
            return
        print(f"  t={record['t']:6.2f}s  {record['state']:<20} "
              f"rem={record.get('route_remaining_m', float('nan')):5.2f} "
              f"her={record['follower_range_m']:5.2f}m "
              f"{'VIS' if record['follower_visible'] else '---'} "
              f"{'los' if record['los_available'] else 'LOS-X'}  "
              f"cmd={record['command_peak']:.2f}  "
              f"z={record['trunk_z']:.4f}")

    rollout.run(progress=progress)
    summary = summarize(rollout)
    passed, results = report(summary)

    print()
    print("=" * 100)
    print("TRANSITIONS")
    print("=" * 100)
    for transition in summary["transitions"]:
        detail = {k: v for k, v in transition.items()
                  if k not in ("t", "from", "to")}
        print(f"  {transition['t']:6.2f}s  {transition['from']:<20} -> "
              f"{transition['to']:<20} {detail}")

    print()
    print("=" * 100)
    print("PLAN")
    print("=" * 100)
    plan = summary["plan"] or {}
    print(f"  requested {summary['requested_destination']!r} out of "
          f"{summary['destination_candidates']} -> resolved "
          f"{summary['resolved_destination']!r}")
    print(f"  route {plan.get('length_m')} m, straight line "
          f"{plan.get('straight_line_m')} m (x{plan.get('detour_ratio')}), "
          f"blocked by {plan.get('straight_line_blocked_by')}")
    print(f"  bends: " + ", ".join(f"{b['hand']} {b['turn_deg']:+.1f} deg"
                                   for b in plan.get("bends", [])))
    print(f"  waypoints {plan.get('waypoints')}")
    print(f"  cells: {plan.get('free_cells')} free, "
          f"{plan.get('static_blocked_cells')} static-blocked, "
          f"{plan.get('crowd_blocked_cells')} crowd-blocked "
          f"{plan.get('crowd_blockers')}")
    print(f"  min planned clearance {plan.get('min_planned_clearance_m')} m "
          f"(needs {plan.get('route_clearance_required_m')} m), corner radius "
          f"{plan.get('corner_radius_m')} m")

    print()
    print("=" * 100)
    print("EPISODES  (detected from measurement; the stall windows are NOT "
          "visible to the machine)")
    print("=" * 100)
    for episode in summary["episodes"]:
        print(f"  episode {episode['index']}  cause={episode['cause']:<5} "
              f"detected {episode['detected_at_s']}s at "
              f"{episode['distance_at_detect_m']} m "
              f"(visible={episode['visible_at_detect']}, "
              f"los={episode['los_available_at_detect']}, "
              f"lagging {episode['lagging_for_s']}s, "
              f"unseen {episode['unseen_for_s']}s)")
        print(f"              waited {episode['wait_duration_s']}s at "
              f"{episode['waiting_spot_xy']} "
              f"(max command {episode['max_command_while_waiting']}, "
              f"moved {episode['duck_moved_while_waiting_m']} m)")
        print(f"              resumed {episode['resumed_at_s']}s at "
              f"{episode['distance_at_resume_m']} m, she closed "
              f"{episode['follower_closed_m']} m, visible "
              f"{episode['visible_fraction_with_los'] * 100:.1f}% of "
              f"{episode['los_steps']} LOS steps")
    print("  declared stalls (the scenario's own script, for comparison):")
    for stall in summary["declared_stalls"]:
        print(f"    {stall['start_s']:5.1f}-{stall['end_s']:5.1f}s  "
              f"factor {stall['speed_factor']:.2f}  {stall['label']!r} -> "
              f"episodes {stall['episode_indices']} "
              f"(lag {stall['detection_lag_s']}s)")

    print()
    print("=" * 100)
    print("ACCEPTANCE GATES")
    print("=" * 100)
    for label, ok, evidence in results:
        print(f"  [{' OK ' if ok else 'FAIL'}] {label}")
        print(f"         {evidence}")

    print()
    print(f"lead={summary['lead_path_m']}m  "
          f"episodes={summary['episode_count']}  "
          f"she_walked={summary['follower_walked_m']}m  "
          f"final_dest={summary['final_destination_distance_m']}m  "
          f"vis_los={summary['monitor_visible_fraction_with_los']}")
    print("ALL GATES PASS" if passed else "GATES FAILED")

    summary["all_gates_pass"] = passed
    summary["gate_results"] = [
        {"gate": label, "pass": ok, "evidence": evidence}
        for label, ok, evidence in results]
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json}")
    if args.trace:
        Path(args.trace).write_text(json.dumps(rollout.records))
        print(f"wrote {args.trace}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
