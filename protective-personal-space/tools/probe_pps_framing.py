#!/usr/bin/env python3
"""MEASURE the wide plaza camera against the real recorded trace.

The framing this behavior needs is not the framing a patrol needs.  Three
bodies matter at once - the duck, the protected person, and whoever is
currently walking at her - and the HUD leaves only a narrow vertical band of
screen unobstructed, so a camera that keeps the duck nicely centred can still
hide the intruder behind an opaque panel for a whole encounter.

WHAT IS SCORED, AND WHY EACH TERM EXISTS
------------------------------------------
Every candidate is replayed through ``PlazaCamera``'s OWN easing over the real
9500-tick trace, then scored per tick on:

* ``duck_clear``    - the duck inside the HUD's clear band.  Any candidate
  below 0.98 is rejected outright: the robot is the subject.
* ``ward_clear``    - the protected person inside it.  A frame without her
  cannot show what the duck was protecting.
* ``threat_clear``  - the selected intruder inside it, graded ONLY over the
  ticks where one is selected, because in MONITOR there is nobody to frame.
* ``duck_px``       - the duck's projected size in pixels, so the robot is not
  a dot in a wide plaza.
* ``pair_px``       - the median smallest on-screen gap between any two
  subjects.  THIS IS THE TERM THAT DECIDES THE ANSWER, and it exists because
  containment alone does not: a camera placed along the Aina-to-intruder axis
  keeps all three inside the band and SUPERIMPOSES them, which hides the exact
  geometry - the duck standing between two people - that the behavior claims.
  Containment saturates at 1.000 across a wide range of azimuths, so without a
  separation term the choice would come down to rounding.
* ``eye_z``         - the camera's own height, so it never sits inside a wall.

THE CLEAR BAND IS THE BINDING CONSTRAINT AND IT IS READ FROM THE LAYOUT
------------------------------------------------------------------------
It is not a taste parameter: at 960x640 the left HUD column ends at x=278, the
PiP and right column begin at x=650, the title bar ends at y=42 and the caption
strip begins at y=552.  Those four numbers come from ``pps_overlay.compose``
and are restated here as constants, so a layout change that moved a panel would
be caught by re-running the probe rather than by looking at frames.

Run:
    ../../microduck_rl/.venv/bin/python tools/probe_pps_framing.py \\
        --trace /tmp/pps_trace_baseline.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pps_render_camera as prc  # noqa: E402
from pps_actors import bodies_at  # noqa: E402
from pps_render_camera import PlazaCamera  # noqa: E402

# READ FROM pps_overlay.compose.  See the module docstring.
CLEAR_X0, CLEAR_X1 = 278, 650
CLEAR_Y0, CLEAR_Y1 = 42, 552
WIDTH, HEIGHT = 960, 640
# MuJoCo's default free-camera vertical FOV.
FOVY_DEG = 45.0
# A person's chest, and the duck's trunk: the points that must stay in band.
PERSON_Z = 0.36
DUCK_Z = 0.12
# The plaza's walls are 1.00 m half-height, so an eye above this is provably
# outside every one of them and no look-at clamp is ever needed.
EYE_CLEARS_SCENE_Z = 2.40


def in_band(x: float, y: float, in_front: bool) -> bool:
    return bool(in_front and CLEAR_X0 <= x <= CLEAR_X1
                and CLEAR_Y0 <= y <= CLEAR_Y1)


def replay(records, azimuth, elevation, near, far, spread_for_far):
    """Fly one candidate over the whole trace and return its scores."""
    prc.CAM_AZIMUTH = azimuth
    prc.CAM_ELEVATION = elevation
    prc.CAM_DISTANCE_NEAR = near
    prc.CAM_DISTANCE_FAR = far
    prc.SPREAD_FOR_FAR_M = spread_for_far

    rig = PlazaCamera(records[0]["duck_xy"])
    duck_hits = ward_hits = threat_hits = threat_ticks = 0
    duck_px: list[float] = []
    pair_px: list[float] = []
    min_eye_z = float("inf")

    for record in records:
        threat_xy = record["threat_xy"]
        rig.advance(record["duck_xy"], record["ward_xy"], threat_xy)
        t = record["t"]
        min_eye_z = min(min_eye_z, float(rig.eye(t)[2]))

        duck = [record["duck_xy"][0], record["duck_xy"][1], DUCK_Z]
        dx, dy, front = rig.project(duck, t, WIDTH, HEIGHT, FOVY_DEG)
        duck_hits += in_band(dx, dy, front)

        # The duck's projected size, from two points 0.23 m apart across the
        # view: the robot's own measured width.
        head = [record["duck_xy"][0], record["duck_xy"][1], DUCK_Z + 0.23]
        hx, hy, hfront = rig.project(head, t, WIDTH, HEIGHT, FOVY_DEG)
        if front and hfront:
            duck_px.append(float(np.hypot(hx - dx, hy - dy)))

        ward = [record["ward_xy"][0], record["ward_xy"][1], PERSON_Z]
        wx, wy, wfront = rig.project(ward, t, WIDTH, HEIGHT, FOVY_DEG)
        ward_hits += in_band(wx, wy, wfront)

        if threat_xy is not None:
            threat_ticks += 1
            point = [threat_xy[0], threat_xy[1], PERSON_Z]
            tx, ty, tfront = rig.project(point, t, WIDTH, HEIGHT, FOVY_DEG)
            threat_hits += in_band(tx, ty, tfront)
            # THE SEPARATION TERM.  Measured only while an intruder is
            # selected, because that is exactly when the three-body geometry is
            # the thing on trial.
            if front and wfront and tfront:
                pair_px.append(min(
                    float(np.hypot(wx - dx, wy - dy)),
                    float(np.hypot(tx - dx, ty - dy)),
                    float(np.hypot(tx - wx, ty - wy))))

    total = len(records)
    return {
        "azimuth": azimuth, "elevation": elevation, "near": near, "far": far,
        "spread_for_far": spread_for_far,
        "duck_clear": duck_hits / total,
        "ward_clear": ward_hits / total,
        "threat_clear": threat_hits / max(threat_ticks, 1),
        "duck_px": float(np.median(duck_px)) if duck_px else 0.0,
        "pair_px": float(np.median(pair_px)) if pair_px else 0.0,
        "min_eye_z": min_eye_z,
    }


def score(entry) -> float:
    """One number per candidate, with the rejections applied first.

    The duck being visible is not tradeable, and neither is a camera inside a
    wall, so both are hard rejections rather than weighted terms.  Among the
    survivors the trade is real and it is NOT mainly about containment, which
    saturates: it is between a closer camera that makes the duck bigger and an
    angle that keeps the three subjects from superimposing on each other.  The
    separation term is capped at 200 px because beyond that the subjects are
    already unambiguous and further spread only costs duck size.
    """
    if entry["duck_clear"] < 0.98 or entry["min_eye_z"] < EYE_CLEARS_SCENE_Z:
        return -1.0
    return (2.0 * entry["ward_clear"] + 2.0 * entry["threat_clear"]
            + 0.010 * min(entry["duck_px"], 60.0)
            + 0.006 * min(entry["pair_px"], 200.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    records = json.loads(Path(args.trace).read_text())
    # The trace stores the active person's NAME.  Their POSITION is recovered
    # from ``bodies_at`` at the same display time the record was written for,
    # which is exactly what the renderer reads from ``previous_states`` - the
    # actors are pure functions of t, so this is the same number and not an
    # approximation of it.
    for record in records:
        active = record["active"]
        if active is None:
            record["threat_xy"] = None
            continue
        state = bodies_at(record["t"])[active]
        record["threat_xy"] = ([float(state.pos[0]), float(state.pos[1])]
                               if state.present else None)
    selected_ticks = sum(r["threat_xy"] is not None for r in records)
    print(f"replaying {len(records)} ticks; "
          f"{selected_ticks} have a selected intruder")

    grid = itertools.product(
        (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0,
         210.0, 240.0, 270.0, 300.0, 330.0),   # azimuth, the full circle
        (-24.0, -28.0, -32.0, -36.0),          # elevation
        (4.20, 4.60, 5.00),                    # near distance
        (8.20, 9.00),                          # far distance
        (3.60,),                               # spread for far
    )
    results = [replay(records, *combo) for combo in grid]
    results.sort(key=score, reverse=True)

    print(f"\n{'az':>6} {'el':>6} {'near':>5} {'far':>5} {'spr':>4} "
          f"{'duck':>6} {'ward':>6} {'threat':>7} {'px':>6} {'pair':>6} "
          f"{'eyez':>6} {'score':>7}")
    for entry in results[:args.top]:
        print(f"{entry['azimuth']:6.1f} {entry['elevation']:6.1f} "
              f"{entry['near']:5.2f} {entry['far']:5.2f} "
              f"{entry['spread_for_far']:4.1f} "
              f"{entry['duck_clear']:6.3f} {entry['ward_clear']:6.3f} "
              f"{entry['threat_clear']:7.3f} {entry['duck_px']:6.1f} "
              f"{entry['pair_px']:6.1f} "
              f"{entry['min_eye_z']:6.2f} {score(entry):7.4f}")

    best = results[0]
    print(f"\nBEST: azimuth {best['azimuth']}, elevation {best['elevation']}, "
          f"distance {best['near']}-{best['far']}, "
          f"spread_for_far {best['spread_for_far']}")
    print(f"  duck in clear band     {best['duck_clear']:.4f}")
    print(f"  ward in clear band     {best['ward_clear']:.4f}")
    print(f"  threat in clear band   {best['threat_clear']:.4f}")
    print(f"  duck median size       {best['duck_px']:.1f} px")
    print(f"  median subject spacing {best['pair_px']:.1f} px")
    print(f"  minimum eye height     {best['min_eye_z']:.2f} m "
          f"(clears {EYE_CLEARS_SCENE_Z} m: "
          f"{'yes' if best['min_eye_z'] >= EYE_CLEARS_SCENE_Z else 'NO'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
