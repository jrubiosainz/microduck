#!/usr/bin/env python3
"""Render the gesture-response video: run the rollout and stream frames.

The rollout is the SAME one the headless gate grades - same module, same
constants, same policy - with a frame callback attached.  Rendering never feeds
back into the physics: the camera work happens in an isolated ``MjData`` inside
:class:`gest_camera.GestureCamera`.

The gates are re-graded on the RENDERED run and printed, so the video and the
numbers are the same rollout rather than two runs that happen to agree.

Run a low-fps preview first and inspect every phase, then the final:
    ../../microduck_rl/.venv/bin/python scripts/render_gesture.py \
        --fps 4 --out /tmp/preview.mp4
    ../../microduck_rl/.venv/bin/python scripts/render_gesture.py \
        --fps 50 --out media/gesture-response.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    from gest_script import session_end_s

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--seconds", type=float, default=session_end_s() + 8.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--out", default=str(
        REPO / "media" / "gesture-response.mp4"))
    parser.add_argument("--frames", default="",
                        help="also write individual PNGs here (for stills)")
    parser.add_argument("--manifest", default="",
                        help="write the frame->time manifest here")
    parser.add_argument("--json", default="",
                        help="write the run summary here")
    parser.add_argument("--trace", default="",
                        help="write the per-tick trace here")
    args = parser.parse_args()

    from gest_camera import PIP_H, PIP_W
    from render_frames import FrameWriter
    from rollout_gesture import GestureRollout
    from validate_gesture import build_summary, gates

    rollout = GestureRollout(args.policy, args.seconds)
    writer = FrameWriter(rollout, args, PIP_W, PIP_H)

    started = time.time()
    last_state = [None]

    def progress(index, record):
        if record["state"] != last_state[0]:
            print(f"  t={record['t']:7.2f}s  {record['state']:<19} "
                  f"x={record['duck_xy'][0]:+6.2f} "
                  f"y={record['duck_xy'][1]:+6.2f} "
                  f"cmd={record['command_peak']:5.3f} "
                  f"done={len(record['accepted_commands'])}/6 "
                  f"read={record['candidate_command'] or '-'}")
            last_state[0] = record["state"]
        elif index % 1000 == 0:
            elapsed = time.time() - started
            print(f"  t={record['t']:7.2f}s  {record['state']:<19} "
                  f"frames={writer.frames}  {elapsed:5.1f}s elapsed")

    rollout.run(on_frame=writer.write, progress=progress)
    status = writer.close()
    if status != 0:
        print(f"ffmpeg exited {status}")
        return status

    out = Path(args.out)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.2f} MB, "
          f"{writer.frames} frames at {args.fps} fps)")

    if args.manifest:
        writer.write_manifest(Path(args.manifest))
        print(f"wrote {args.manifest}")

    # THE GATES ARE RE-GRADED ON THE RENDERED RUN.  Accepting a video whose
    # rollout was never graded would make the numbers a claim about a different
    # execution than the one on screen.
    summary = build_summary(rollout)
    results = gates(summary)
    summary["gates"] = [
        {"name": n, "passed": p, "detail": d} for n, p, d in results]
    summary["gates_passed"] = sum(1 for _, p, _ in results if p)
    summary["gates_total"] = len(results)
    summary["all_gates_passed"] = all(p for _, p, _ in results)
    summary["render"] = {
        "out": str(out), "fps": args.fps, "width": args.width,
        "height": args.height, "crf": args.crf, "frames": writer.frames,
        "bytes": out.stat().st_size,
    }

    print("\nACCEPTANCE GATES ON THE RENDERED RUN")
    for name, passed, detail in results:
        if not passed:
            print(f"  [FAIL] {name}\n         {detail}")
    print(f"{summary['gates_passed']}/{summary['gates_total']} gates passed")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json}")
    if args.trace:
        Path(args.trace).write_text(json.dumps(rollout.records))
        print(f"wrote {args.trace}")
    return 0 if summary["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
