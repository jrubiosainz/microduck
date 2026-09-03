#!/usr/bin/env python3
"""Headless gate: run the whole behavior with no rendering and grade it.

This is the loop that development happens in.  It imports nothing from a
rendering stack, so a validation run has no PIL, imageio or GPU dependency at
all - which is the claim ``test_the_headless_gate_imports_no_rendering_stack``
pins by blocking those modules in ``sys.meta_path``.

Run with:
    ../../microduck_rl/.venv/bin/python scripts/validate_slalom.py --seconds 92
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
    parser.add_argument("--seconds", type=float, default=92.0)
    parser.add_argument("--json", default="")
    parser.add_argument("--trace", default="",
                        help="write the full per-tick record stream here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from rollout_slalom import SlalomRollout
    from slalom_metrics import report, summarize

    rollout = SlalomRollout(args.policy, args.seconds)

    last_state = [None]

    def progress(index, record):
        if args.quiet:
            return
        if record["state"] != last_state[0]:
            print(f"  t={record['t']:6.2f}s  {record['state']:<13} "
                  f"x={record['duck_xy'][0]:+6.2f} y={record['duck_xy'][1]:+6.2f} "
                  f"thr={record['threat'] or '-':<6} "
                  f"side={record['decision_side'] or '-':<8} "
                  f"chosen={record['chosen_clearance_m']:5.2f} "
                  f"rej={record['rejected_clearance_m']:5.2f} "
                  f"clr={record['min_body_clearance_m']:5.2f}")
            last_state[0] = record["state"]
        elif index % 500 == 0:
            print(f"  t={record['t']:6.2f}s  {record['state']:<13} "
                  f"cmd={record['command_peak']:.2f} "
                  f"z={record['trunk_z']:.4f} "
                  f"goal={record['goal_remaining_m']:5.2f} "
                  f"y={record['duck_xy'][1]:+5.2f}")

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
        print(f"  {transition['t']:6.2f}s  {transition['from']:<13} -> "
              f"{transition['to']:<13} {detail}")

    print()
    print("=" * 100)
    print("THE ENCOUNTERS  (the choreography is NOT visible to the planner)")
    print("=" * 100)
    for crossing in summary["lane_crossings"]:
        print(f"  declared: {crossing['actor']:<7} ({crossing['encounter']}) "
              f"crosses the lane at t={crossing['t_s']:6.2f}s, "
              f"x={crossing['x_m']:+.3f}, "
              f"{'northbound' if crossing['northbound'] else 'southbound'}")
    print()
    for p in summary["passes"]:
        print(f"  measured: pass {p['index']} on {p['threat']:<7} -> "
              f"{p['side'].upper():<5} began {p['began_at_s']:6.2f}s ended "
              f"{p['ended_at_s']}s  chose {p['chosen_clearance_m']:.3f} m "
              f"over {p['rejected_side']} {p['rejected_clearance_m']:.3f} m"
              + (f"  (after {p['waited_s']:.2f}s of WAIT)"
                 if p['waited_s'] else ""))
    print(f"  sides {summary['pass_sides']}  alternating="
          f"{summary['alternating']}  expected {summary['expected_pass_sides']}")

    print()
    print("=" * 100)
    print("WAITS  (neither corridor predicted safe)")
    print("=" * 100)
    for w in summary["waits"]:
        print(f"  wait {w['index']} on {w.get('threat')}: began "
              f"{w.get('began_at_s')}s for {w.get('duration_s')}s; best "
              f"rejected corridor was the {w.get('rejected_side')} at "
              f"{w.get('rejected_clearance_m')} m, resolved "
              f"{w.get('resolved_side')} at {w.get('resolved_clearance_m')} m")
    if not summary["waits"]:
        print("  none")

    print()
    print("=" * 100)
    print("PREDICTION vs REALITY  (constant-velocity, finite-differenced)")
    print("=" * 100)
    for b in summary["prediction_bracketing"]:
        print(f"  {b['threat']:<7} {b['side']:<5} predicted "
              f"{b['predicted_clearance_m']:6.3f} m   measured "
              f"{b['measured_clearance_m']}   margin {b['margin_m']}   "
              f"{'CONSERVATIVE' if b['conservative'] else 'OPTIMISTIC'}")
    print(f"  tracker: raw max {summary['tracker_max_raw_speed_mps']} m/s, "
          f"filtered max {summary['tracker_max_filtered_speed_mps']} m/s")

    print()
    print("=" * 100)
    print("THE TURNING PATH  (there is NO strafe on this policy)")
    print("=" * 100)
    t = summary["turning_path"]
    print(f"  path {t.get('path_m')} m against net {t.get('net_m')} m "
          f"= {t.get('excess_over_net_m')} m of excess")
    print(f"  lane offset {t.get('max_right_offset_m')} .. "
          f"{t.get('max_left_offset_m')} m (span {t.get('lateral_span_m')} m), "
          f"lateral path {t.get('lateral_path_m')} m")
    print(f"  accumulated yaw {t.get('yaw_travel_deg')} deg")

    print()
    print("=" * 100)
    print("THE GOAL")
    print("=" * 100)
    print(f"  band at {summary['goal_xy']}, first reached "
          f"{summary['reached_goal_at_s']}s, stood in it "
          f"{summary['goal_seconds']}s, closest "
          f"{summary['min_goal_distance_m']} m")
    print(f"  visible in {summary['goal_visible_fraction_with_los'] * 100:.2f}% "
          f"of {summary['goal_los_steps']} LOS steps through the head camera")

    print()
    print("=" * 100)
    print("SAFETY AND STILLNESS")
    print("=" * 100)
    print(f"  min clearance {summary['min_body_clearance_m']} m to "
          f"{summary['min_body_clearance_name']}")
    print(f"  per body {summary['min_clearance_by_body_m']}")
    print(f"  min scenery {summary['min_scenery_clearance_m']} m to "
          f"{summary['min_scenery_clearance_geom']}")
    print(f"  interlock held the duck {summary['interlock_holds']} tick(s): "
          f"{summary['interlock_reasons']}")
    print(f"  zero-state drift: {summary['zero_state_path_m']}")
    print(f"  longest illegal zero run: "
          f"{summary['longest_illegal_zero_run']} tick(s)")

    print()
    print("=" * 100)
    print("WHAT THE CAMERA WATCHED")
    print("=" * 100)
    for subject, entry in summary["subject_visibility"].items():
        print(f"    {subject:<8} ({entry['role']:<10}) {entry['steps']:5d} "
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
    print(f"path={summary['path_m']}m  net={summary['net_m']}m  "
          f"passes={summary['pass_count']}  waits={summary['wait_count']}  "
          f"sides={summary['pass_sides']}  "
          f"goal={summary['reached_goal_at_s']}s")
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
