#!/usr/bin/env python3
"""Run the crosswalk-guardian scenario, with or without rendering.

    # gate first, no rendering dependencies at all
    python scripts/render_crosswalk_guardian.py --no-render --seconds 46

    # low-fps preview for visual inspection
    python scripts/render_crosswalk_guardian.py --seconds 46 --fps 4 \
        --out /tmp/preview-frames

    # final
    python scripts/render_crosswalk_guardian.py --seconds 46 --fps 50 \
        --out /tmp/final-frames --metrics media/metrics.json

The rollout, the gates and the overlay live in their own modules; this file
only wires them together and decides what to write where.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from guardian_metrics import format_report, summarize  # noqa: E402
from rollout_guardian import GuardianRollout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--out", default="/tmp/crosswalk-guardian-frames")
    parser.add_argument("--metrics",
                        default="/tmp/crosswalk-guardian-metrics.json")
    parser.add_argument("--records", default=None,
                        help="optional path for the full per-step record dump")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--allow-fail", action="store_true",
                        help="write metrics and exit 0 even if a gate fails")
    args = parser.parse_args()

    # The PiP pixel geometry sets the camera's horizontal FOV, so it is owned
    # by guardian_camera and imported by BOTH the gate and the overlay.
    # Measuring visibility through one frustum and drawing another would let
    # the reported percentages disagree with the picture.
    from guardian_camera import PIP_H, PIP_W

    rollout = GuardianRollout(
        args.policy, args.seconds, scene=args.scene,
        robot_dir=args.robot_dir, pip_size=(PIP_W, PIP_H),
    )
    print(f"crosswalk-guardian: {rollout.total_steps} control steps, "
          f"decimation={rollout.decimation}, duck_radius="
          f"{rollout.duck_radius:.4f} m, render={not args.no_render}")

    writer = None
    if not args.no_render:
        from render_frames import FrameWriter
        writer = FrameWriter(rollout, args, PIP_W, PIP_H)

    last_state = None

    def progress(index, record):
        nonlocal last_state
        if record["state"] != last_state:
            print(f"  t={record['t']:6.2f}s {record['state']:<16s} "
                  f"x={record['duck_xy'][0]:+.3f} "
                  f"y={record['duck_xy'][1]:+.3f} "
                  f"L={record['left_fraction']:.2f} "
                  f"R={record['right_fraction']:.2f} "
                  f"gap={record['gap_margin_s']:+6.2f}s "
                  f"({record['gap_limiting']}) "
                  f"z={record['trunk_z_m']:.3f}")
            last_state = record["state"]
        elif index % 250 == 0:
            print(f"  t={record['t']:6.2f}s {record['state']:<16s} "
                  f"cmd=({record['command'][0]:+.2f},{record['command'][1]:+.2f},"
                  f"{record['command'][2]:+.2f}) "
                  f"x={record['duck_xy'][0]:+.3f} y={record['duck_xy'][1]:+.3f} "
                  f"yaw={record['duck_yaw_deg']:+6.1f} "
                  f"gap={record['gap_margin_s']:+6.2f}s "
                  f"clear={record['nearest_clearance_m']:.3f} "
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
