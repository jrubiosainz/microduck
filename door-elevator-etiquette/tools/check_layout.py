#!/usr/bin/env python3
"""Prove the building is walkable, the zones are consistent, and the etiquette
gates are NOT vacuous.

Run before any rollout.  Everything here is pure geometry against the same
modules the behavior uses - no MuJoCo, no policy, no physics - so a failure names
a building problem rather than a control one.

Five things are checked, and the fifth is the one that matters most:

1. **Every bend fits the duck's MEASURED turning circle for its own sign.**
2. **The route keeps positive clearance to every static surface**, with the
   duck's conservative planar radius.
3. **The route passes through the middle of every aperture**, not past a jamb.
4. **The holding points are where they claim to be**: the door hold outside the
   threshold band, the lift hold outside the exit passage, the cabin hold inside
   the cabin's inset interior.
5. **The apertures are wide enough for two bodies abreast.**  If they were not,
   "the duck never went through side by side" would be a fact about the wall
   rather than about the robot, and the gate would pass whatever the state
   machine did.  This check exists to keep that gate honest, and it FAILS if an
   opening is ever narrowed to the point of triviality.

Run:
    ../../microduck_rl/.venv/bin/python tools/check_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

from etiquette_actors import ROUTES, max_heading_step  # noqa: E402
from etiquette_cast import PEOPLE  # noqa: E402
from etiquette_path import (  # noqa: E402
    LEG_NAMES,
    aperture_crossings,
    build_route,
    door_hold_xy,
    leg_bounds,
    route_bend_report,
)
from etiquette_states import (  # noqa: E402
    DUCK_EXACT_LATERAL_HALF_WIDTH,
    DUCK_PLANAR_RADIUS,
)
from etiquette_zones import (  # noqa: E402
    CABIN_HOLD_XY,
    CABIN_INTERIOR,
    DOOR_APERTURE,
    DOOR_THRESHOLD,
    LIFT_APERTURE,
    LIFT_PASSAGE,
    LIFT_THRESHOLD,
    WAIT_SIDE_XY,
    cabin_contains,
    cabin_margin_m,
)
from lobby_doors import APERTURES  # noqa: E402
from lobby_layout import (  # noqa: E402
    ABREAST_MARGIN_M,
    STATIC_OBSTACLES,
    static_gap,
)

# The widest adult's lateral half-width, taken from the stature scaling rather
# than from a compiled model so this tool stays MuJoCo-free.  The person block
# in ``tools/build_scene.py`` puts the widest geom - an arm capsule - at
# ``0.078 * stature`` from the centreline with a ``0.026 * stature`` radius.
WIDEST_ADULT_LATERAL = max(0.078 * p.stature + 0.026 * p.stature
                           for p in PEOPLE)


def main() -> int:
    failures: list[str] = []

    def check(ok: bool, label: str, evidence: str) -> None:
        print(f"  [{' OK ' if ok else 'FAIL'}] {label}")
        print(f"         {evidence}")
        if not ok:
            failures.append(label)

    route = build_route()
    bounds = leg_bounds(route)

    print("=" * 92)
    print("THE ROUTE")
    print("=" * 92)
    print(f"  {route.length:.4f} m in {len(bounds)} legs")
    previous = 0.0
    for name, end in zip(LEG_NAMES, bounds):
        print(f"    {name:<16} ends {end:7.4f} m  (leg {end - previous:6.4f} m)")
        previous = end

    print()
    print("=" * 92)
    print("1. EVERY BEND FITS THE MEASURED TURNING CIRCLE FOR ITS OWN SIGN")
    print("=" * 92)
    bends = route_bend_report(route)
    for bend in bends:
        print(f"    {bend['hand']:<5} {bend['turn_deg']:+7.1f} deg  "
              f"r={bend['radius_m']:.4f} m  needs "
              f"{bend['min_radius_for_hand_m']:.4f} m  "
              f"{'OK' if bend['walkable'] else 'UNWALKABLE'}")
    check(all(b["walkable"] for b in bends) and len(bends) >= 4,
          "every bend is walkable, and the route has at least 4 of them",
          f"{len(bends)} bends, "
          f"{sum(1 for b in bends if not b['walkable'])} unwalkable")

    print()
    print("=" * 92)
    print("2. THE ROUTE KEEPS POSITIVE CLEARANCE TO EVERY STATIC SURFACE")
    print("=" * 92)
    worst_gap, worst_at, worst_name = float("inf"), None, ""
    for index in range(4001):
        s = route.length * index / 4000
        point, _ = route.pose_at_arc(s)
        name, gap = static_gap(point)
        if gap < worst_gap:
            worst_gap, worst_at, worst_name = gap, point.copy(), name
    margin = worst_gap - DUCK_PLANAR_RADIUS
    check(margin > 0.0,
          "the route clears every wall, jamb, cabin panel and obstacle",
          f"closest approach {worst_gap:.4f} m to {worst_name} at "
          f"({worst_at[0]:.3f}, {worst_at[1]:.3f}); with the duck's "
          f"{DUCK_PLANAR_RADIUS} m radius that leaves {margin:.4f} m")

    print()
    print("=" * 92)
    print("3. THE ROUTE PASSES THROUGH THE MIDDLE OF EVERY APERTURE")
    print("=" * 92)
    crossings = aperture_crossings(route)
    for crossing in crossings:
        print(f"    {crossing['aperture']:<15} "
              f"{crossing['offset_from_centre_m']:+.4f} m off centre, "
              f"{crossing['margin_m']:.4f} m to the jamb")
    check(all(c.get("crossed") and c["margin_m"] > DUCK_PLANAR_RADIUS
              for c in crossings),
          "every aperture is crossed with the duck's whole footprint inside it",
          f"{len(crossings)} crossings, tightest margin "
          f"{min(c['margin_m'] for c in crossings):.4f} m against a "
          f"{DUCK_PLANAR_RADIUS} m radius")

    print()
    print("=" * 92)
    print("4. THE HOLDING POINTS ARE WHERE THEY CLAIM TO BE")
    print("=" * 92)
    hold = door_hold_xy()
    door_depth = DOOR_THRESHOLD.depth_into(hold, DUCK_PLANAR_RADIUS)
    check(door_depth == 0.0,
          "the door holding point is OUTSIDE the threshold band",
          f"({hold[0]:.3f}, {hold[1]:.3f}) penetrates the band by "
          f"{door_depth:.4f} m; band x={DOOR_THRESHOLD.x_range}")

    passage_depth = LIFT_PASSAGE.depth_into(WAIT_SIDE_XY, DUCK_PLANAR_RADIUS)
    lift_threshold_depth = LIFT_THRESHOLD.depth_into(
        WAIT_SIDE_XY, DUCK_PLANAR_RADIUS)
    check(passage_depth == 0.0 and lift_threshold_depth == 0.0,
          "the lift holding point is BESIDE the doors, out of the exit passage",
          f"({WAIT_SIDE_XY[0]:.3f}, {WAIT_SIDE_XY[1]:.3f}) penetrates the "
          f"passage by {passage_depth:.4f} m and the threshold by "
          f"{lift_threshold_depth:.4f} m; passage y={LIFT_PASSAGE.y_range}")

    check(cabin_contains(CABIN_HOLD_XY, DUCK_PLANAR_RADIUS),
          "the cabin holding point is INSIDE the cabin, whole footprint",
          f"({CABIN_HOLD_XY[0]:.3f}, {CABIN_HOLD_XY[1]:.3f}) with "
          f"{cabin_margin_m(CABIN_HOLD_XY):.4f} m to the nearest face; "
          f"interior x={tuple(round(v, 3) for v in CABIN_INTERIOR.x_range)} "
          f"y={tuple(round(v, 3) for v in CABIN_INTERIOR.y_range)}")

    # The cabin hold must also be clear of BOTH aperture centrelines, or the
    # duck would be standing in the doorway it just came through.
    front_depth = LIFT_APERTURE.depth_into(CABIN_HOLD_XY, DUCK_PLANAR_RADIUS)
    check(front_depth == 0.0,
          "the cabin holding point is clear of the front aperture",
          f"penetrates it by {front_depth:.4f} m")

    print()
    print("=" * 92)
    print("5. THE APERTURES ARE WIDE ENOUGH FOR TWO ABREAST  (NON-VACUITY)")
    print("=" * 92)
    print("   If they were not, 'never side by side' would be a fact about the")
    print("   wall rather than about the robot.")
    vacuous = []
    for name, spec in APERTURES.items():
        clear = float(spec["clear_w"])
        slack = clear - 2 * DUCK_EXACT_LATERAL_HALF_WIDTH \
            - 2 * WIDEST_ADULT_LATERAL
        ok = slack >= ABREAST_MARGIN_M
        print(f"    {name:<15} clear {clear:.3f} m, duck+adult abreast leaves "
              f"{slack:.4f} m  {'(two could pass)' if ok else '(TOO NARROW)'}")
        if not ok:
            vacuous.append(name)
    check(not vacuous,
          "every aperture could physically fit the duck and an adult abreast",
          f"required {ABREAST_MARGIN_M} m of slack; "
          f"{'all pass' if not vacuous else 'too narrow: ' + str(vacuous)}")

    print()
    print("=" * 92)
    print("THE SCRIPTED PEOPLE")
    print("=" * 92)
    worst_step, worst_who, worst_when = max_heading_step(100.0)
    check(worst_step <= 6.0,
          "no scripted person turns faster than a walking person could",
          f"largest single-tick heading change {worst_step:.2f} deg by "
          f"{worst_who} at {worst_when:.2f}s")
    for name, actor_route in ROUTES.items():
        print(f"    {name:<8} {actor_route.length:6.3f} m at "
              f"{actor_route.speed:.3f} m/s, starts {actor_route.start_t:5.2f}s, "
              f"finishes {actor_route.finish_t():6.2f}s"
              + (f", holds {actor_route.hold_windows}"
                 if actor_route.hold_windows else ""))

    print()
    print("=" * 92)
    print(f"{len(STATIC_OBSTACLES)} static obstacles, {len(PEOPLE)} people, "
          f"{len(APERTURES)} apertures")
    if failures:
        print(f"LAYOUT FAILED: {len(failures)} check(s)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("LAYOUT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
