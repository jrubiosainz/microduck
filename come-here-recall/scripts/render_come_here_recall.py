#!/usr/bin/env python3
"""Run the come-here-recall scenario, with or without rendering.

    # gate first, no rendering at all
    python scripts/render_come_here_recall.py --no-render --seconds 46

    # low-fps preview for visual inspection
    python scripts/render_come_here_recall.py --seconds 46 --fps 5 \
        --out /tmp/preview-frames

    # final
    python scripts/render_come_here_recall.py --seconds 46 --fps 50 \
        --out /tmp/final-frames --metrics media/metrics.json

The rollout, the gates and the overlay live in their own modules; this file
only wires them together, owns the CALL SCRIPT, and decides what to write
where.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recall_metrics import format_report, summarize  # noqa: E402
from recall_model import STANDOFF_MAX, STANDOFF_MIN, Call  # noqa: E402
from rollout_recall import RecallRollout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# THE CALL SCRIPT.  Exactly one adult calls at a time, from three widely
# separated bearings, in this order.  A call is a scripted EVENT with an
# identity - a semantic proxy for "an adult calls the robot", not audio.
#
# Durations are generous enough that a call is still sounding while the duck
# searches for it, and they never overlap between the three genuine callers, so
# "exactly one adult calls at a time" is a property of the script rather than a
# hope.
#
# BLUE's call at t=22.0 s is the INTERRUPTION test: it lands while the duck is
# mid-cycle with yellow.  The no-steal rule must refuse it, and the gate checks
# the refusal was recorded.  Blue is never served.
CALLS: tuple[Call, ...] = (
    Call(caller="red", start_s=1.5, duration_s=11.0),
    Call(caller="yellow", start_s=15.5, duration_s=12.0),
    Call(caller="blue", start_s=22.0, duration_s=2.5, expected=False),
    Call(caller="green", start_s=30.0, duration_s=13.5),
)
EXPECTED_ORDER: tuple[str, ...] = ("red", "yellow", "green")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO_ROOT / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--seconds", type=float, default=46.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--out", default="/tmp/come-here-recall-frames")
    parser.add_argument("--metrics", default="/tmp/come-here-recall-metrics.json")
    parser.add_argument("--records", default=None,
                        help="optional path for the full per-step record dump")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--allow-fail", action="store_true",
                        help="write metrics and exit 0 even if a gate fails")
    args = parser.parse_args()

    # The PiP pixel geometry sets the attention camera's horizontal FOV, so it
    # is owned by attention_camera and imported by BOTH the gate and the
    # overlay.  Measuring visibility through one frustum and drawing another
    # would let the reported percentage disagree with the picture.
    from attention_camera import PIP_H, PIP_W

    rollout = RecallRollout(
        args.policy, args.seconds, calls=CALLS, scene=args.scene,
        robot_dir=args.robot_dir, pip_size=(PIP_W, PIP_H),
    )
    print(f"come-here-recall: {rollout.total_steps} control steps, "
          f"decimation={rollout.decimation}, duck_radius="
          f"{rollout.duck_radius:.4f} m, render={not args.no_render}")

    writer = None
    if not args.no_render:
        from render_frames import FrameWriter
        writer = FrameWriter(rollout, args, PIP_W, PIP_H,
                             calls=CALLS, expected_order=EXPECTED_ORDER)

    last_key = None

    def progress(index, record):
        nonlocal last_key
        key = (record["cycle"], record["state"], record["locked"])
        if key != last_key:
            print(f"  t={record['t']:6.2f}s cycle={record['cycle']} "
                  f"{record['state']:<11s} caller={str(record['caller']):<7s} "
                  f"locked={str(record['locked']):<7s} "
                  f"seen={record['caller_visible']!s:<5s} "
                  f"gate={record['gate_open']!s:<5s} "
                  f"rng={record['caller_range_m']:5.2f} "
                  f"z={record['trunk_z_m']:.3f}")
            last_key = key
        elif index % 250 == 0:
            print(f"  t={record['t']:6.2f}s {record['state']:<11s} "
                  f"cmd=({record['command'][0]:+.2f},{record['command'][1]:+.2f},"
                  f"{record['command'][2]:+.2f}) "
                  f"xy=({record['duck_xy'][0]:+.2f},{record['duck_xy'][1]:+.2f}) "
                  f"rng={record['caller_range_m']:5.2f} "
                  f"err={record['heading_error_deg']:+6.1f} "
                  f"z={record['trunk_z_m']:.3f}")

    rollout.run(on_frame=writer.write if writer else None, progress=progress)

    summary = summarize(
        rollout, expected_order=EXPECTED_ORDER,
        standoff_min=STANDOFF_MIN, standoff_max=STANDOFF_MAX,
    )
    summary["calls"] = [
        {"caller": call.caller, "start_s": call.start_s,
         "duration_s": call.duration_s, "expected": call.expected}
        for call in CALLS
    ]
    if writer is not None:
        summary["frames"] = writer.frames
        summary["fps"] = args.fps
        summary["width"] = args.width
        summary["height"] = args.height
    print("\n" + format_report(summary))

    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.metrics}")
    if args.records:
        Path(args.records).write_text(json.dumps(rollout.records) + "\n")
        print(f"wrote {args.records}")

    if not summary["all_gates_pass"] and not args.allow_fail:
        failed = [k for k, v in summary["gates"].items() if not v]
        raise SystemExit(f"GATES FAILED: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
