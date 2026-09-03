#!/usr/bin/env python3
"""Headless gate: run the whole behavior with no rendering and grade it.

This is the loop that development happens in.  It imports nothing from the
rendering stack, so a validation run has no PIL, imageio or GPU dependency at
all — which is also the claim ``test_no_render_imports`` pins.

Run with:
    ../../microduck_rl/.venv/bin/python scripts/validate_lost.py --seconds 52
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
    parser.add_argument("--policy", default=str(REPO / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--seconds", type=float, default=52.0)
    parser.add_argument("--json", default="")
    parser.add_argument("--trace", default="",
                        help="write the full per-tick record stream here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from lost_metrics import report, summarize
    from rollout_lost import LostRollout

    rollout = LostRollout(args.policy, args.seconds)

    def progress(index, record):
        if args.quiet or index % 250:
            return
        print(f"  t={record['t']:5.2f}s  {record['state']:<13} "
              f"cmd={record['command_peak']:.3f}  "
              f"vis={'Y' if record['guardian_visible'] else 'n'}  "
              f"range={record['guardian_range_m']}  "
              f"blocked={record['guardian_blocked_by'] or '-'}")

    rollout.run(progress=progress)
    summary = summarize(rollout)
    passed, results = report(summary)

    print()
    print("=" * 94)
    print("TRANSITIONS")
    print("=" * 94)
    for transition in summary["transitions"]:
        detail = {k: v for k, v in transition.items()
                  if k not in ("t", "from", "to")}
        print(f"  {transition['t']:6.2f}s  {transition['from']:<13} -> "
              f"{transition['to']:<13} {detail}")

    print()
    print("=" * 94)
    print("ACCEPTANCE GATES")
    print("=" * 94)
    for label, ok, evidence in results:
        print(f"  [{' OK ' if ok else 'FAIL'}] {label}")
        print(f"         {evidence}")

    print()
    print(f"cycles={summary['cycle_count']}  "
          f"rejected={summary['distinct_rejected']}  "
          f"longest_occlusion={summary['longest_geometric_occlusion_s']}s "
          f"({summary['longest_geometric_occluder']})  "
          f"final_range={summary['final_range_m']}m")
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
