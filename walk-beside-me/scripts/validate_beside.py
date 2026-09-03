#!/usr/bin/env python3
"""Headless gate: run the whole behavior with no rendering and grade it.

This is the loop that development happens in.  It imports nothing from a
rendering stack, so a validation run has no PIL, imageio or GPU dependency at
all — which is also the claim ``test_no_render_imports`` pins.

Run with:
    ../../microduck_rl/.venv/bin/python scripts/validate_beside.py --seconds 86
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
    parser.add_argument("--seconds", type=float, default=86.0)
    parser.add_argument("--json", default="")
    parser.add_argument("--trace", default="",
                        help="write the full per-tick record stream here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from beside_metrics import report, summarize
    from rollout_beside import BesideRollout

    rollout = BesideRollout(args.policy, args.seconds)

    def progress(index, record):
        if args.quiet or index % 250:
            return
        print(f"  t={record['t']:5.2f}s  {record['state']:<16} "
              f"side={str(record['side_name']):<5} "
              f"lat={record['lateral_m']:+.3f} "
              f"lon={record['longitudinal_m']:+.3f}  "
              f"cmd={record['command_peak']:.2f}  "
              f"vis={'Y' if record['guardian_visible'] else 'n'}  "
              f"L={'ok' if record['verdict_left']['usable'] else 'X'} "
              f"R={'ok' if record['verdict_right']['usable'] else 'X'}")

    rollout.run(progress=progress)
    summary = summarize(rollout)
    passed, results = report(summary)

    print()
    print("=" * 96)
    print("TRANSITIONS")
    print("=" * 96)
    for transition in summary["transitions"]:
        detail = {k: v for k, v in transition.items()
                  if k not in ("t", "from", "to")}
        print(f"  {transition['t']:6.2f}s  {transition['from']:<16} -> "
              f"{transition['to']:<16} {detail}")

    print()
    print("=" * 96)
    print("SIDE DECISIONS")
    print("=" * 96)
    for decision in summary["side_decisions"]:
        print(f"  {decision['t']:6.2f}s  {decision['kind']:<14} -> "
              f"{decision['side_name']:<5}  {decision['reason']}")
        print(f"           left  usable={decision['left']['usable']} "
              f"static={decision['left']['static_gap_m']} "
              f"({decision['left']['static_name']}) "
              f"person={decision['left']['person_gap_m']} "
              f"({decision['left']['person_name']})")
        print(f"           right usable={decision['right']['usable']} "
              f"static={decision['right']['static_gap_m']} "
              f"({decision['right']['static_name']}) "
              f"person={decision['right']['person_gap_m']} "
              f"({decision['right']['person_name']})")

    print()
    print("=" * 96)
    print("ACCEPTANCE GATES")
    print("=" * 96)
    for label, ok, evidence in results:
        print(f"  [{' OK ' if ok else 'FAIL'}] {label}")
        print(f"         {evidence}")

    print()
    print(f"beside={summary['beside_seconds']}s  "
          f"switches={summary['completed_switches']}  "
          f"decisions={summary['side_decision_count']}  "
          f"lateral={summary['beside_lateral_min_m']}-"
          f"{summary['beside_lateral_max_m']}m  "
          f"vis_los={summary['visible_fraction_with_los']}")
    print("ALL GATES PASS" if passed else "GATES FAILED")

    summary["all_gates_pass"] = passed
    summary["gate_results"] = [
        {"gate": label, "pass": ok, "evidence": evidence}
        for label, ok, evidence in results]
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json}")
    if args.trace:
        Path(args.trace).write_text(json.dumps(rollout.records))
        print(f"wrote {args.trace}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
