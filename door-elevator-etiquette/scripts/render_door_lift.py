#!/usr/bin/env python3
"""Render the door-and-lift rollout, with or without frames.

    # low-fps preview, for inspecting every phase
    ../../microduck_rl/.venv/bin/python scripts/render_door_lift.py \\
        --seconds 110 --fps 4 --out /tmp/dee-preview

    # final
    ../../microduck_rl/.venv/bin/python scripts/render_door_lift.py \\
        --seconds 110 --fps 50 --out /tmp/dee-final \\
        --metrics media/door-elevator-etiquette-metrics.json

This file only wires modules together and decides what is written where.  The
rollout, the gates and the overlay are unchanged by rendering: the SAME
``EtiquetteRollout`` and the SAME ``summarize``/``report`` grade the run whether
or not a frame is drawn, and the run still exits non-zero if any gate fails.  A
video that disagreed with the gate would be worse than no video.

The headless gate is ``scripts/validate_etiquette.py``, which imports no
rendering stack at all.  This entry point is the only place ``imageio`` and
``PIL`` enter the process, and it imports them lazily INSIDE the render branch
so that ``--no-render`` really is dependency-free.
``test_the_renderer_is_the_only_place_the_rendering_stack_enters`` pins that.
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
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--seconds", type=float, default=110.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--out", default="/tmp/dee-frames")
    parser.add_argument("--metrics", default="/tmp/dee-render-metrics.json")
    parser.add_argument("--trace", default="")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # The PiP pixel geometry sets the camera's horizontal FOV and therefore
    # every visibility measurement, so it is owned by etiquette_camera and
    # imported by BOTH the gate and the renderer.  Measuring through one frustum
    # and drawing another would let the reported percentages disagree with the
    # picture.
    from etiquette_camera import PIP_H, PIP_W
    from etiquette_metrics import report, summarize
    from rollout_etiquette import EtiquetteRollout

    rollout = EtiquetteRollout(args.policy, args.seconds, scene=args.scene,
                               robot_dir=args.robot_dir,
                               pip_size=(PIP_W, PIP_H))
    print(f"door-elevator-etiquette: {rollout.total_steps} control steps, "
          f"decimation={rollout.decimation}, "
          f"duck_radius={rollout.duck_radius:.4f} m, "
          f"{len(rollout.scenery_geoms)} scenery geoms, "
          f"route={rollout.route.length:.4f} m, "
          f"render={not args.no_render}")

    writer = None
    if not args.no_render:
        from render_frames import FrameWriter
        writer = FrameWriter(rollout, args, PIP_W, PIP_H)

    last_state = [None]

    def progress(index, record):
        if args.quiet:
            return
        if record["state"] != last_state[0]:
            print(f"  t={record['t']:6.2f}s  {record['state']:<20} "
                  f"gap={record['guardian_gap_m']:+5.2f} "
                  f"subj={record['subject']:<7} "
                  f"{'VIS' if record['subject_visible'] else '---'}")
            last_state[0] = record["state"]
        elif index % 500 == 0:
            print(f"  t={record['t']:6.2f}s  {record['state']:<20} "
                  f"cmd={record['command_peak']:.2f} "
                  f"z={record['trunk_z']:.4f}")

    rollout.run(on_frame=writer.write if writer else None, progress=progress)
    summary = summarize(rollout)
    passed, results = report(summary)
    summary["all_gates_pass"] = passed
    summary["gate_results"] = [
        {"gate": label, "pass": ok, "evidence": evidence}
        for label, ok, evidence in results]

    print()
    for label, ok, evidence in results:
        print(f"  [{' OK ' if ok else 'FAIL'}] {label}")
        if not ok:
            print(f"         {evidence}")
    print("ALL GATES PASS" if passed else "GATES FAILED")

    if writer is not None:
        manifest = writer.write_manifest()
        print(f"wrote {writer.frames} frames at {args.fps} fps to {args.out}")
        print(f"wrote {manifest}")

    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.metrics}")
    if args.trace:
        Path(args.trace).write_text(json.dumps(rollout.records))
        print(f"wrote {args.trace}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
