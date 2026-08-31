#!/usr/bin/env python3
"""Run the move-away-crowd scenario, with or without rendering.

    # gate first, no rendering at all
    python scripts/render_move_away_crowd.py --no-render --seconds 52

    # low-fps preview for visual inspection
    python scripts/render_move_away_crowd.py --seconds 52 --fps 5 \
        --out /tmp/preview-frames

    # final
    python scripts/render_move_away_crowd.py --seconds 52 --fps 50 \
        --out /tmp/final-frames --metrics media/metrics.json

The rollout, the gates and the overlay live in their own modules; this file
only wires them together and decides what to write where.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crowd_metrics import format_report, summarize  # noqa: E402
from rollout_crowd import CrowdRollout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(REPO_ROOT / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--seconds", type=float, default=52.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--out", default="/tmp/move-away-crowd-frames")
    parser.add_argument("--metrics", default="/tmp/move-away-crowd-metrics.json")
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

    rollout = CrowdRollout(
        args.policy, args.seconds, scene=args.scene, robot_dir=args.robot_dir,
        pip_size=(PIP_W, PIP_H),
    )
    print(f"move-away-crowd: {rollout.total_steps} control steps, "
          f"decimation={rollout.decimation}, duck_radius="
          f"{rollout.duck_radius:.4f} m, render={not args.no_render}")

    writer = None
    if not args.no_render:
        from render_frames import FrameWriter
        writer = FrameWriter(rollout, args, PIP_W, PIP_H)

    last_key = None

    def progress(index, record):
        nonlocal last_key
        key = (record["cycle"], record["state"], record["locked"])
        if key != last_key:
            print(f"  t={record['t']:6.2f}s cycle={record['cycle']} "
                  f"{record['state']:<11s} locked={str(record['locked']):<7s} "
                  f"seen={record['locked_visible']!s:<5s} "
                  f"nearest={record['nearest_adult']:<7s} "
                  f"clr={record['nearest_clearance_m']:+.3f} "
                  f"z={record['trunk_z_m']:.3f}")
            last_key = key
        elif index % 250 == 0:
            print(f"  t={record['t']:6.2f}s {record['state']:<11s} "
                  f"cmd=({record['command'][0]:+.2f},{record['command'][1]:+.2f},"
                  f"{record['command'][2]:+.2f}) "
                  f"xy=({record['duck_xy'][0]:+.2f},{record['duck_xy'][1]:+.2f}) "
                  f"z={record['trunk_z_m']:.3f}")

    rollout.run(on_frame=writer.write if writer else None, progress=progress)

    summary = summarize(rollout)
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
