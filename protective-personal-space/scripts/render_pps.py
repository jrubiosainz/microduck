#!/usr/bin/env python3
"""Render the Protective Personal Space video: run the rollout, stream frames.

The rollout is the SAME one the headless gate grades - same module, same
constants, same policy - with a frame callback attached.  ``PpsRollout.run``
takes ``on_frame`` as a pure observer: it is called after each step with the
record that was just appended and returns nothing that is read back, so the
deterministic sequence is identical with or without it.

The gates are re-graded on the RENDERED run and printed, so the video and the
numbers are the same rollout rather than two runs that happen to agree.

Run a low-fps preview first and inspect every phase, then the final:

    ../../microduck_rl/.venv/bin/python scripts/render_pps.py \\
        --fps 4 --out /tmp/pps_preview.mp4 --frames /tmp/pps_preview_frames
    ../../microduck_rl/.venv/bin/python scripts/render_pps.py \\
        --fps 50 --out media/protective-personal-space.mp4
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
    from pps_script import session_end_s

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--seconds", type=float, default=session_end_s())
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--out", default=str(
        REPO / "media" / "protective-personal-space.mp4"))
    parser.add_argument("--frames", default="",
                        help="also write individual PNGs here (for stills)")
    parser.add_argument("--manifest", default="",
                        help="write the frame->time manifest here")
    parser.add_argument("--json", default="",
                        help="write the run summary here")
    parser.add_argument("--trace", default="",
                        help="write the per-tick trace here")
    args = parser.parse_args()

    from pps_camera import PIP_H, PIP_W
    from pps_metrics import gates, summarize
    from pps_render_frames import PpsFrameWriter
    from rollout_pps import PpsRollout

    rollout = PpsRollout(args.policy, args.seconds)
    writer = PpsFrameWriter(rollout, args, PIP_W, PIP_H)

    started = time.time()
    last_state = [None]

    def log(index, record):
        elapsed = time.time() - started
        print(f"  t={record['t']:7.2f}s  {record['state']:<18} "
              f"active={str(rollout.machine.selected):<7} "
              f"ward={record['ward_range_m']:5.2f} "
              f"cmd={record['command_peak']:5.3f} "
              f"frames={writer.frames:<5} {elapsed:6.1f}s", flush=True)

    # The frame callback fires on EVERY control tick, so state changes are
    # logged the moment they happen rather than at the next 250-tick progress
    # boundary.  Writing the frame first keeps the log honest about how many
    # frames existed when the state changed.
    def on_frame(index, record):
        writer.write(index, record)
        if record["state"] != last_state[0] or index % 1000 == 0:
            log(index, record)
            last_state[0] = record["state"]

    rollout.run(on_frame=on_frame)
    status = writer.close()
    if status != 0:
        print(f"ffmpeg exited {status}")
        return status

    out = Path(args.out)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.2f} MB, "
          f"{writer.frames} frames at {args.fps} fps)")

    # THE GATES ARE RE-GRADED ON THE RENDERED RUN.  Accepting a video whose
    # rollout was never graded would make the numbers a claim about a different
    # execution than the one on screen.
    summary = summarize(rollout)
    results = gates(summary)
    summary["gate_results"] = [
        {"gate": n, "pass": p, "evidence": e} for n, p, e in results]
    summary["gates_passed"] = sum(1 for _, p, _ in results if p)
    summary["gates_total"] = len(results)
    summary["all_gates_pass"] = all(p for _, p, _ in results)

    print("\nACCEPTANCE GATES ON THE RENDERED RUN")
    for name, passed, detail in results:
        print(f"  [{'OK' if passed else 'FAIL'}] {name}: {detail}")
    print(f"{summary['gates_passed']}/{summary['gates_total']} gates passed")

    if args.manifest:
        writer.write_manifest(Path(args.manifest))
        print(f"wrote {args.manifest}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json}")
    if args.trace:
        Path(args.trace).write_text(json.dumps(rollout.records))
        print(f"wrote {args.trace}")
    return 0 if summary["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
