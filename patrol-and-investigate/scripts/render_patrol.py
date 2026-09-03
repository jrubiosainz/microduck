#!/usr/bin/env python3
"""Render the patrol-and-investigate video: run the rollout and stream frames.

The rollout is the SAME one the headless gate grades - same module, same
constants, same policy - with a frame callback attached.  Rendering never feeds
back into the physics: the camera work happens in an isolated ``MjData`` inside
:class:`patrol_camera.PatrolCamera`.

Run a low-fps preview first and inspect every phase, then the final:
    ../../microduck_rl/.venv/bin/python scripts/render_patrol.py \
        --seconds 148 --fps 4 --out /tmp/preview.mp4
    ../../microduck_rl/.venv/bin/python scripts/render_patrol.py \
        --seconds 148 --fps 50 --out media/patrol-and-investigate.mp4
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
    parser.add_argument("--seconds", type=float, default=148.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--out", default=str(
        REPO / "media" / "patrol-and-investigate.mp4"))
    parser.add_argument("--frames", default="",
                        help="also write individual PNGs here (for stills)")
    parser.add_argument("--manifest", default="",
                        help="write the frame->time manifest here")
    parser.add_argument("--json", default="",
                        help="write the run summary here")
    args = parser.parse_args()

    from patrol_camera import PIP_H, PIP_W
    from patrol_metrics import report, summarize
    from render_frames import FrameWriter
    from rollout_patrol import PatrolRollout

    rollout = PatrolRollout(args.policy, args.seconds)
    writer = FrameWriter(rollout, args, PIP_W, PIP_H)

    started = time.time()
    last_state = [None]

    def progress(index, record):
        if record["state"] != last_state[0]:
            print(f"  t={record['t']:7.2f}s  {record['state']:<17} "
                  f"x={record['duck_xy'][0]:+6.2f} "
                  f"y={record['duck_xy'][1]:+6.2f} "
                  f"to={record['target_name']:<12} "
                  f"done={record['completed']}/5 "
                  f"cand={record['candidate'] or '-'}")
            last_state[0] = record["state"]
        elif index % 1000 == 0:
            elapsed = time.time() - started
            print(f"  t={record['t']:7.2f}s  {record['state']:<17} "
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
