#!/usr/bin/env python3
"""Render the dynamic-slalom video: run the rollout and stream frames to ffmpeg.

The rollout is the SAME one the headless gate grades - same module, same
constants, same policy - with a frame callback attached.  Rendering never feeds
back into the physics: the camera work happens in an isolated ``MjData`` inside
:class:`slalom_camera.SlalomCamera`.

Run a low-fps preview first and inspect every phase, then the final:
    ../../microduck_rl/.venv/bin/python scripts/render_slalom.py \
        --seconds 92 --fps 4 --width 960 --height 640 --out /tmp/preview.mp4
    ../../microduck_rl/.venv/bin/python scripts/render_slalom.py \
        --seconds 92 --fps 50 --width 960 --height 640 \
        --out media/dynamic-slalom.mp4
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--seconds", type=float, default=92.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--out", default=str(REPO / "media"
                                             / "dynamic-slalom.mp4"))
    parser.add_argument("--frames", default="",
                        help="also write individual PNGs here (for stills)")
    parser.add_argument("--manifest", default="",
                        help="write the frame->time manifest here")
    parser.add_argument("--json", default="",
                        help="write the run summary here")
    args = parser.parse_args()

    from render_frames import FrameWriter
    from rollout_slalom import SlalomRollout
    from slalom_camera import PIP_H, PIP_W
    from slalom_metrics import report, summarize

    rollout = SlalomRollout(args.policy, args.seconds)
    writer = FrameWriter(rollout, args, PIP_W, PIP_H)

    started = time.time()
    last_state = [None]

    def progress(index, record):
        if record["state"] != last_state[0]:
            print(f"  t={record['t']:6.2f}s  {record['state']:<13} "
                  f"x={record['duck_xy'][0]:+6.2f} "
                  f"y={record['duck_xy'][1]:+6.2f} "
                  f"thr={record['threat'] or '-':<6} "
                  f"side={record['decision_side'] or '-'}")
            last_state[0] = record["state"]
        elif index % 500 == 0:
            elapsed = time.time() - started
            print(f"  t={record['t']:6.2f}s  {record['state']:<13} "
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

    summary = summarize(rollout)
    passed, results = report(summary)
    summary["all_gates_pass"] = passed
    summary["gate_results"] = [
        {"gate": label, "pass": ok, "evidence": evidence}
        for label, ok, evidence in results]
    print("\nACCEPTANCE GATES ON THE RENDERED RUN")
    for label, ok, evidence in results:
        if not ok:
            print(f"  [FAIL] {label}\n         {evidence}")
    print("ALL GATES PASS" if passed else "GATES FAILED")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
