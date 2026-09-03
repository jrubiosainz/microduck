#!/usr/bin/env python3
"""Choose the wide camera by REPLAYING the recorded trace, not by eye.

A 25 cm robot on a 12.4 x 6.4 m promenade is easy to lose, and this video may
never hide the duck behind the very kiosk whose blockage it demonstrates.  So
the framing is measured: every candidate camera is replayed against the REAL
recorded per-tick trace, with the SAME look-at easing, swing and derived bounds
the renderer uses, and scored per sampled frame on

* is the duck on screen at all;
* is the duck clear of the HUD panels, which are fixed rectangles;
* is the duck unoccluded against real solid volumes -- every obstacle at its
  true footprint AND height, the four perimeter walls, and every person as a
  standing cylinder;
* is the CAMERA'S OWN EYE inside the promenade, because a camera outside the
  wall renders that wall as a slab across the shot;
* is the guardian in shot, since a formation with only one body in frame cannot
  be judged;
* is the kiosk in shot DURING the switch window, because that is the object the
  refusal names;
* how large the duck appears, and how far apart the duck and the guardian
  appear -- the term this behavior specifically needs, since "beside" is a
  LATERAL offset that a badly chosen azimuth collapses to nothing.

The geometry primitives live in ``tools/framing_geometry.py``; this module is
the scoring policy and the search stages.

Run:
    ../../microduck_rl/.venv/bin/python tools/probe_framing.py \
        --trace /tmp/wbm/trace.json --stage shortlist
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from framing_geometry import (  # noqa: E402
    Camera,
    DUCK_TOP_M,
    _blocked,
    _clear_of_panels,
    _on_screen,
    _volumes,
    eye_safe_lookat_bounds,
)
from promenade_layout import FLOOR_HALF, OBSTACLES  # noqa: E402


def score(records, azimuth, elevation, distance, bias,
          swing_deg, swing_period, lookat_z=0.30, ease=0.045, stride=10):
    """Replay the trace through one candidate camera and score sampled frames.

    THE EASE ADVANCES ON EVERY TICK, THE SCORING ON EVERY ``stride``-th TICK.
    ``render_frames.FrameWriter`` advances its look-at once per control tick and
    renders once per output frame, so easing here per SAMPLED frame would
    measure a camera path no render ever flies — slower by exactly ``stride``.
    Separating the two loops is what makes a probe score evidence about the
    video rather than about the probe.

    There is no clamp parameter.  Containment is DERIVED by
    ``eye_safe_lookat_bounds`` from the camera's own geometry, exactly as
    ``render_frames`` derives it, so the two cannot drift apart.
    """
    camera = Camera(azimuth, elevation, distance)
    lookat = np.array([-3.0, -2.0, lookat_z])
    on, clear, unoccluded, guardian_ok, kiosk_ok, heights = 0, 0, 0, 0, 0, []
    separations: list[float] = []
    inside_hall = 0
    beside_frames = 0
    switch_frames = 0
    total = 0
    kiosk = next(o for o in OBSTACLES if o.name == "kiosk")

    for index, record in enumerate(records):
        duck = np.array(record["duck_xy"], dtype=np.float64)
        guardian = np.array(record["person_xy"]["nadia"], dtype=np.float64)
        target = np.array([bias * duck[0] + (1.0 - bias) * guardian[0],
                           bias * duck[1] + (1.0 - bias) * guardian[1],
                           lookat_z])
        lookat += ease * (target - lookat)
        if index % stride:
            continue
        # Containment is enforced by the DERIVED bounds, so the camera follows
        # the duck as far as it can and is pulled back only when its own eye
        # would leave the hall.
        (lo_x, hi_x), (lo_y, hi_y) = eye_safe_lookat_bounds(camera)
        live = np.array([float(np.clip(lookat[0], lo_x, hi_x)),
                         float(np.clip(lookat[1], lo_y, hi_y)),
                         lookat_z])
        offset = swing_deg * math.sin(record["t"] / swing_period)

        total += 1
        duck_px = camera.project(np.append(duck, 0.115), live, offset)
        top_px = camera.project(np.append(duck, DUCK_TOP_M), live, offset)
        guardian_px = camera.project(np.append(guardian, 0.36), live, offset)
        eye = camera.eye(live, offset)
        volumes = _volumes(record)

        # THE CAMERA MUST STAND INSIDE THE PROMENADE.  A free camera orbiting a
        # look-at near the perimeter puts its own eye beyond the wall, and MuJoCo
        # then renders the near wall as a slab across the shot.  Scored, not
        # assumed: this is what the first rendered frame actually showed.
        if abs(eye[0]) <= FLOOR_HALF[0] - 0.05 \
                and abs(eye[1]) <= FLOOR_HALF[1] - 0.05 and eye[2] > 0.15:
            inside_hall += 1

        if _on_screen(duck_px):
            on += 1
            if _clear_of_panels(duck_px):
                clear += 1
            if not _blocked(eye, np.append(duck, 0.16), volumes, ignore=duck):
                unoccluded += 1
            if duck_px is not None and top_px is not None:
                heights.append(abs(duck_px[1] - top_px[1]))
        if _on_screen(guardian_px, margin=0) and not _blocked(
                eye, np.append(guardian, 0.40), volumes, ignore=guardian):
            guardian_ok += 1
        # THE CRITERION THIS BEHAVIOR NEEDS AND THE SIBLINGS DID NOT.  "Beside"
        # is a LATERAL offset, so a camera looking along her direction of travel
        # collapses the whole formation into a few pixels and the video stops
        # showing the thing it is about.  Scored only while the duck is actually
        # in a formation state, because during the crossing the separation is
        # supposed to be large.
        if record["state"] in ("BESIDE_LEFT", "BESIDE_RIGHT") \
                and duck_px is not None and guardian_px is not None:
            beside_frames += 1
            separations.append(float(np.hypot(duck_px[0] - guardian_px[0],
                                              duck_px[1] - guardian_px[1])))
        # The kiosk only has to be in shot while the refusal it causes is live.
        if 6.0 <= record["t"] <= 32.0:
            switch_frames += 1
            kiosk_px = camera.project(
                np.array([kiosk.center[0], kiosk.center[1], 0.5]), live, offset)
            if _on_screen(kiosk_px, margin=0):
                kiosk_ok += 1

    denominator = max(total, 1)
    return {
        "azimuth": azimuth, "elevation": elevation, "distance": distance,
        "bias": bias, "lookat_z": lookat_z, "ease": ease,
        "on_screen": on / denominator,
        "clear_of_panels": clear / denominator,
        "unoccluded": unoccluded / denominator,
        "guardian": guardian_ok / denominator,
        "kiosk_in_switch": kiosk_ok / max(switch_frames, 1),
        "inside_hall": inside_hall / denominator,
        "duck_px": float(np.mean(heights)) if heights else 0.0,
        "beside_px": float(np.mean(separations)) if separations else 0.0,
        "beside_px_min": float(np.min(separations)) if separations else 0.0,
    }


def rank(result) -> float:
    """One number.  Seeing the duck dominates; legibility of the formation next.

    ``beside_px`` is normalised against 150 px and capped: past that the
    formation is perfectly readable and more separation only shrinks both
    bodies, so it stops earning score.  ``inside_hall`` is weighted as heavily
    as occlusion because a camera outside the wall renders the wall, which is
    the worst possible failure and is invisible to every other term.
    """
    return (4.0 * result["unoccluded"] + 4.0 * result["inside_hall"]
            + 3.0 * result["clear_of_panels"]
            + 2.5 * min(result["beside_px"] / 150.0, 1.0)
            + 2.0 * result["guardian"] + 1.5 * result["kiosk_in_switch"]
            + 1.0 * result["on_screen"] + 0.02 * min(result["duck_px"], 60.0))


# Complete candidate parameter sets for ``--stage shortlist``, as
# (azimuth, elevation, distance, bias, swing_deg, swing_period, lookat_z, ease).
SHORTLIST: dict[str, tuple] = {
    "az10_d28": (10.0, -26.0, 2.8, 1.00, 3.0, 19.0, 0.30, 0.045),
    "az10_d32": (10.0, -26.0, 3.2, 1.00, 3.0, 19.0, 0.30, 0.045),
    "az20_d28": (20.0, -24.0, 2.8, 1.00, 3.0, 19.0, 0.30, 0.045),
    "az20_d32": (20.0, -24.0, 3.2, 1.00, 3.0, 19.0, 0.30, 0.045),
    "az00_d28": (0.0, -26.0, 2.8, 1.00, 3.0, 19.0, 0.30, 0.045),
    "az30_d28": (30.0, -26.0, 2.8, 1.00, 3.0, 19.0, 0.30, 0.045),
    "azm20_d28": (-20.0, -30.0, 2.8, 1.00, 3.0, 19.0, 0.30, 0.045),
    "az10_d24": (10.0, -26.0, 2.4, 1.00, 3.0, 19.0, 0.30, 0.045),
    "az10_d28_e32": (10.0, -32.0, 2.8, 1.00, 3.0, 19.0, 0.30, 0.045),
    "az10_d28_b055": (10.0, -26.0, 2.8, 0.55, 3.0, 19.0, 0.30, 0.045),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="/tmp/wbm/trace.json")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--stage", choices=("pose", "lookat", "shortlist"),
                        default="pose")
    parser.add_argument("--azimuth", type=float, default=-20.0)
    parser.add_argument("--elevation", type=float, default=-32.0)
    parser.add_argument("--distance", type=float, default=3.2)
    args = parser.parse_args()

    records = json.loads(Path(args.trace).read_text())
    print(f"{len(records)} ticks, stage={args.stage}")

    results = []
    if args.stage == "shortlist":
        # Head-to-head between complete parameter sets, each the winner of an
        # earlier stage.  The grid stages optimise one group at a time and can
        # therefore prefer a pose that only wins at the look-at default; this
        # scores whole candidates against each other, which is what actually
        # gets shipped.
        for name, params in SHORTLIST.items():
            result = score(records, *params, stride=10)
            result["name"] = name
            results.append(result)
    elif args.stage == "pose":
        # Stage one: where the camera stands, at the default look-at policy.
        for azimuth in (-30.0, -20.0, -10.0, 0.0, 10.0, 15.0, 20.0, 30.0, 40.0):
            for elevation in (-24.0, -28.0, -32.0, -36.0, -40.0):
                for distance in (2.8, 3.2, 3.6, 4.0, 4.4):
                    results.append(score(records, azimuth, elevation, distance,
                                         1.00, 3.0, 19.0, stride=20))
    else:
        # Stage two: how the camera LOOKS, at the pose stage one chose.
        #
        # ``ease`` IS SWEPT HERE AND THAT MATTERS.  It applies once per control
        # tick, so at 50 Hz an ease of 0.045 is a 0.44 s time constant and the
        # camera pins the duck to the centre of frame — which sounds good and is
        # not, because the centre of frame is where the duck is most often
        # behind a HUD panel.  A slower ease lets the duck drift across the free
        # band between the panels.
        for bias in (0.55, 0.72, 0.88, 1.00):
            for ease in (0.006, 0.012, 0.025, 0.045):
                for distance in (2.4, 2.8, 3.2):
                    results.append(score(
                        records, args.azimuth, args.elevation, distance,
                        bias, 3.0, 19.0, lookat_z=0.30, ease=ease, stride=20))

    results.sort(key=rank, reverse=True)
    for result in results[:args.top]:
        print(f"  {result.get('name', ''):>16} "
              f"az {result['azimuth']:6.1f} el {result['elevation']:6.1f} "
              f"d {result['distance']:.1f} bias {result['bias']:.2f} "
              f"ease {result['ease']:.3f}  "
              f"unocc {result['unoccluded']:.3f}  "
              f"inhall {result['inside_hall']:.3f}  "
              f"clear {result['clear_of_panels']:.3f}  "
              f"guard {result['guardian']:.3f}  "
              f"kiosk {result['kiosk_in_switch']:.3f}  "
              f"on {result['on_screen']:.3f}  "
              f"duckpx {result['duck_px']:.1f}  "
              f"besidepx {result['beside_px']:.0f}/{result['beside_px_min']:.0f}"
              f"   rank {rank(result):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
