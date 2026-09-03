#!/usr/bin/env python3
"""Headless gate: run the whole behavior with no rendering and grade it.

This is the loop that development happens in.  It imports nothing from a
rendering stack, so a validation run has no PIL, imageio or GPU dependency at
all - which is also the claim
``test_the_headless_gate_imports_no_rendering_stack`` pins.

Run with:
    ../../microduck_rl/.venv/bin/python scripts/validate_etiquette.py --seconds 96
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
    parser.add_argument("--seconds", type=float, default=96.0)
    parser.add_argument("--json", default="")
    parser.add_argument("--trace", default="",
                        help="write the full per-tick record stream here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from etiquette_metrics import report, summarize
    from rollout_etiquette import EtiquetteRollout

    rollout = EtiquetteRollout(args.policy, args.seconds)

    last_state = [None]

    def progress(index, record):
        if args.quiet:
            return
        if record["state"] != last_state[0]:
            print(f"  t={record['t']:6.2f}s  {record['state']:<20} "
                  f"gap={record['guardian_gap_m']:+6.2f} "
                  f"subj={record['subject']:<7} "
                  f"{'VIS' if record['subject_visible'] else '---'}  "
                  f"door={record['door_fraction']['concourse_door']:.2f} "
                  f"lift={record['door_fraction']['lift_front']:.2f} "
                  f"rear={record['door_fraction']['lift_rear']:.2f}")
            last_state[0] = record["state"]
        elif index % 500 == 0:
            print(f"  t={record['t']:6.2f}s  {record['state']:<20} "
                  f"cmd={record['command_peak']:.2f} "
                  f"z={record['trunk_z']:.4f} "
                  f"rem={record.get('route_remaining_m', 0.0):5.2f}")

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
    print("THE ROUTE")
    print("=" * 100)
    print(f"  {summary['route_length_m']} m, "
          f"{len(summary['route_bends'])} bends, legs at "
          f"{summary['leg_bounds_m']}")
    for bend in summary["route_bends"]:
        print(f"    {bend['hand']:<5} {bend['turn_deg']:+7.1f} deg at "
              f"r={bend['radius_m']} m (needs "
              f"{bend['min_radius_for_hand_m']} m) "
              f"{'OK' if bend['walkable'] else 'UNWALKABLE'}")
    for crossing in summary["route_crossings"]:
        print(f"    crosses {crossing['aperture']:<15} at "
              f"{crossing['offset_from_centre_m']:+.4f} m off centre, "
              f"{crossing['margin_m']} m to the jamb")

    print()
    print("=" * 100)
    print("THE DOORWAY  (the schedule is NOT visible to the machine)")
    print("=" * 100)
    for entry in summary["door_schedule"]:
        print(f"  declared: {entry['door']:<15} opens {entry['opens_at_s']}s "
              f"closes {entry['closes_at_s']}  ({entry['label']})")
    for y in summary["yields"]:
        print(f"  measured: yield {y['index']} ({y['kind']}) stopped "
              f"{y['began_at_s']}s with {y.get('exiters_pending_at_stop')} "
              f"pending, released {y.get('ended_at_s')}s after "
              f"{y.get('duration_s')}s "
              f"(clear sustained {y.get('clear_sustained_s')}s)")
    print(f"  exiters that used the door: {summary['exiters_used_door']}")
    print(f"  aperture ticks: {summary['aperture_steps']}  shared: "
          f"{summary['aperture_shared_steps']}")
    print(f"  abreast slack (two bodies could fit): "
          f"{summary['abreast_slack_m']}")

    print()
    print("=" * 100)
    print("THE LIFT")
    print("=" * 100)
    b = summary["boarding"]
    print(f"  waited beside the doors {summary['state_seconds'].get('WAIT_SIDE')}s")
    print(f"  {b.get('occupants_exited_before_entry')} occupants cleared by "
          f"{b.get('cleared_at_s')}s "
          f"(of {summary['max_occupants_exited']} total, "
          f"{summary['occupants_used_lift']} used the aperture)")
    print(f"  duck entered {b.get('duck_entered_at_s')}s, she was inside: "
          f"{b.get('guardian_inside_at_entry')}, gap "
          f"{b.get('guardian_gap_at_entry_m')} m")
    print(f"  positioned {b.get('positioned_at_s')}s, rode "
          f"{summary['ride_seconds']}s at max command "
          f"{summary['state_command_max'].get('RIDE')}")
    print(f"  cabin: {summary['cabin_seconds']}s inside, min face margin "
          f"{summary['min_cabin_margin_m']} m")
    print(f"  she exited {b.get('guardian_exited_at_s')}s, duck "
          f"{b.get('duck_exited_at_s')}s")
    print(f"  crossings: ")
    for c in summary["crossings"]:
        print(f"    {c['aperture']:<15} entered {c.get('entered_at_s')}s at "
              f"{c.get('open_fraction_at_entry')} open "
              f"({c.get('effective_gap_at_entry_m')} m clear)")

    print()
    print("=" * 100)
    print("ZONES, ORDER AND STILLNESS")
    print("=" * 100)
    for name, entry in summary["zone_worst"].items():
        print(f"  {name:<28} worst {entry['worst_m']} m at {entry['at_s']}s "
              f"over {entry['steps']} step(s); early "
              f"{summary['zone_violation_steps'].get(name, 0)}")
    print(f"  guardian gap {summary['min_guardian_gap_m']} .. "
          f"{summary['max_guardian_gap_m']} m over "
          f"{summary['guardian_gap_samples']} samples, "
          f"{summary['overtake_steps']} overtaking step(s)")
    print(f"  interlock held the duck {summary['interlock_holds']} tick(s): "
          f"{summary['interlock_reasons']}")
    print(f"  zero-state drift: {summary['zero_state_path_m']}")

    print()
    print("=" * 100)
    print("WHAT THE CAMERA WATCHED")
    print("=" * 100)
    print(f"  role order {summary['subject_role_order']} vs expected "
          f"{summary['expected_subject_role_order']}")
    for subject, entry in summary["subject_visibility"].items():
        print(f"    {subject:<8} ({entry['role']:<11}) {entry['steps']:5d} "
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
          f"yields={summary['yield_count']}  ride={summary['ride_seconds']}s  "
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
