#!/usr/bin/env python3
"""Run one headless rollout and grade every acceptance gate on it.

The entry point only: the summary is assembled by :mod:`gest_summary` and the
gates are declared by :mod:`gest_gates`, so this file stays about ORCHESTRATION.

Imports no rendering stack at all - proved by
``test_the_headless_gate_imports_no_rendering_stack``, which blocks ``PIL``,
``imageio`` and ``matplotlib`` in ``sys.meta_path`` and imports this module.

Run:
    ../../microduck_rl/.venv/bin/python scripts/validate_gesture.py \\
        --json /tmp/gr_final.json --trace /tmp/gr_trace.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gest_gates import gates  # noqa: E402,F401
from gest_script import session_end_s  # noqa: E402
from gest_summary import POLICY_SHA256, build_summary  # noqa: E402,F401
from rollout_gesture import GestureRollout  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(
        Path(__file__).resolve().parents[1] / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--seconds", type=float, default=session_end_s() + 8.0)
    parser.add_argument("--json", default="")
    parser.add_argument("--trace", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    rollout = GestureRollout(args.policy, args.seconds)

    def progress(index: int, record: dict) -> None:
        if args.quiet or index % 500:
            return
        print(f"  t={record['t']:6.2f}  {record['state']:<18} "
              f"cmd={record['command_peak']:.3f}  "
              f"acc={len(record['accepted_commands'])}", flush=True)

    rollout.run(progress=progress)
    summary = build_summary(rollout)
    results = gates(summary)
    summary["gates"] = [
        {"name": n, "passed": p, "detail": d} for n, p, d in results]
    summary["gates_passed"] = sum(1 for _, p, _ in results if p)
    summary["gates_total"] = len(results)
    summary["all_gates_passed"] = all(p for _, p, _ in results)

    print()
    print("=" * 92)
    print(f"GESTURE RESPONSE - {len(results)} ACCEPTANCE GATES ON A REAL "
          f"{args.seconds:.0f} s ROLLOUT")
    print("=" * 92)
    for name, passed, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<38} {detail}")
    print("=" * 92)
    print(f"{summary['gates_passed']}/{summary['gates_total']} gates passed")

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"summary -> {args.json}")
    if args.trace:
        Path(args.trace).write_text(json.dumps(rollout.records))
        print(f"trace   -> {args.trace}")

    return 0 if summary["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
