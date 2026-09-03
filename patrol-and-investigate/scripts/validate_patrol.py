#!/usr/bin/env python3
"""Headless gate: run the whole behavior with no rendering and grade it.

This is the loop that development happens in.  It imports nothing from a
rendering stack, so a validation run has no PIL, imageio or GPU dependency at
all - which is the claim ``test_the_headless_gate_imports_no_rendering_stack``
pins by blocking those modules in ``sys.meta_path``.

Run with:
    ../../microduck_rl/.venv/bin/python scripts/validate_patrol.py --seconds 150
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
    parser.add_argument("--seconds", type=float, default=150.0)
    parser.add_argument("--json", default="")
    parser.add_argument("--trace", default="",
                        help="write the full per-tick record stream here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from patrol_metrics import report, summarize
    from rollout_patrol import PatrolRollout

    rollout = PatrolRollout(args.policy, args.seconds)
    last_state = [None]

    def progress(index, record):
        if args.quiet:
            return
        if record["state"] != last_state[0]:
            print(f"  t={record['t']:7.2f}s  {record['state']:<17} "
                  f"x={record['duck_xy'][0]:+6.2f} y={record['duck_xy'][1]:+6.2f} "
                  f"to={record['target_name']:<12} "
                  f"done={record['completed']}/5 "
                  f"cand={record['candidate'] or '-':<8} "
                  f"{record['candidate_verdict'] or '':<11} "
                  f"zone={record['zone_gap_m']:+.2f}")
            last_state[0] = record["state"]
        elif index % 1000 == 0:
            print(f"  t={record['t']:7.2f}s  {record['state']:<17} "
                  f"cmd={record['command_peak']:.2f} "
                  f"z={record['trunk_z']:.4f} "
                  f"rem={record['target_remaining_m']:5.2f}")

    rollout.run(progress=progress)
    summary = summarize(rollout)
    passed, results = report(summary)

    print()
    print("=" * 100)
    print("THE PATROL")
    print("=" * 100)
    print(f"  circuit {summary['circuit_length_m']} m, corners "
          f"{summary['corner_turns_deg']} deg")
    print(f"  declared {summary['checkpoint_declared_order']}")
    print(f"  visited  {summary['checkpoint_visited_order']}")
    for scan in summary["scan_scans"]:
        print(f"    {scan['checkpoint']:<12} arrived within "
              f"{scan['arrival_error_m']:.3f} m, stopped "
              f"{scan['stopped_s']:.2f}s, swept "
              f"{scan['scan_arc_deg']:6.1f} deg, path "
              f"{scan['still_path_m']:.4f} m -> {scan['result'].upper()}"
              + (f" ({scan['detected']})" if scan["detected"] else ""))
    print(f"  home at {summary['reached_home_at_s']}s, stood "
          f"{summary['home_seconds']}s, closest "
          f"{summary['min_home_distance_m']} m")

    print()
    print("=" * 100)
    print("WHAT IT FOUND  (identity is a semantic proxy, confidence a rule "
          "margin)")
    print("=" * 100)
    for verdict in summary["verdict_verdicts"]:
        print(f"  {verdict['target']:<9} -> {verdict['verdict'].upper():<11} "
              f"conf {verdict['confidence']:.2f}  at {verdict['t']:7.2f}s")
        print(f"             {verdict['rule']}")
    print(f"  expected {summary['verdict_expected']}")
    print(f"  investigated {summary['verdict_investigated']}, dismissed "
          f"{summary['verdict_dismissed']}")

    print()
    print("=" * 100)
    print("THE INVESTIGATIONS")
    print("=" * 100)
    for entry in summary["standoff_investigations"]:
        print(f"  {entry['target']:<9} detected {entry['detected_at_s']:7.2f}s "
              f"at {entry['detect_range_m']:.3f} m, broke off toward "
              f"{entry['interrupted_checkpoint']}")
        print(f"             approach {entry['approach_start_range_m']:.3f} -> "
              f"{entry['approach_end_range_m']:.3f} m "
              f"({entry['range_reduction_m']:+.3f} m) over "
              f"{entry['approach_path_m']:.3f} m of path")
        print(f"             standoff {entry['standoff_xy']}, "
              f"{entry['rejected_standoffs']} candidates rejected; closest "
              f"MEASURED clearance {entry['min_clearance_m']:.4f} m")
        for observation in entry["observations"]:
            print(f"               angle {observation['angle_deg']:+6.1f} deg "
                  f"held {observation['held_s']:.2f}s, target in frame "
                  f"{observation['visible_fraction'] * 100:5.1f}%")
        print(f"             resumed {entry['resumed_at_s']:7.2f}s toward "
              f"{entry['resumed_checkpoint']}, back within "
              f"{entry['return_error_m']:.3f} m  "
              f"route_preserved={entry['route_preserved']}")

    print()
    print("=" * 100)
    print("SAFETY AND STILLNESS")
    print("=" * 100)
    print(f"  restricted zone: closest approach "
          f"{summary['min_zone_gap_m']:+.4f} m, "
          f"{summary['zone_breach_steps']} breaches")
    print(f"  min clearance {summary['min_body_clearance_m']} m to "
          f"{summary['min_body_clearance_name']}")
    print(f"  per body {summary['min_clearance_by_body_m']}")
    print(f"  min scenery {summary['min_scenery_clearance_m']} m to "
          f"{summary['min_scenery_clearance_geom']}")
    print(f"  interlock held the duck {summary['interlock_holds']} tick(s)")
    print(f"  worst zero episode {summary['worst_zero_episode_path_m']} m path, "
          f"{summary['worst_zero_episode_net_m']} m net")
    print(f"  longest illegal zero run "
          f"{summary['longest_illegal_zero_run']} tick(s)")
    print(f"  max |vy| {summary['max_abs_vy_command']}")

    print()
    print("=" * 100)
    print("WHAT THE CAMERA WATCHED")
    print("=" * 100)
    print(f"  camera active {summary['camera_active_fraction'] * 100:.2f}% of "
          f"{summary['steps']} ticks")
    for subject, entry in summary["subject_visibility"].items():
        print(f"    {subject:<8} ({entry['role']:<9}) {entry['steps']:5d} "
              f"steps, {entry['fraction_with_los'] * 100:6.2f}% visible of "
              f"{entry['los_steps']} LOS steps")

    print()
    print("=" * 100)
    print("ACCEPTANCE GATES")
    print("=" * 100)
    for label, ok, evidence in results:
        print(f"  [{' OK ' if ok else 'FAIL'}] {label}")
        print(f"         {evidence}")

    print()
    print(f"path={summary['path_m']}m  checkpoints="
          f"{summary['checkpoint_count']}/5  "
          f"investigations={summary['standoff_count']}  "
          f"verdicts={len(summary['verdict_verdicts'])}  "
          f"home={summary['reached_home_at_s']}s")
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
