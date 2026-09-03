#!/usr/bin/env python3
"""Check the facility layout against every geometric requirement, numerically.

The scenario has to satisfy several constraints at once, and getting any of them
wrong shows up far away - as an investigation that never fires, an approach with
no walking in it, or a body the camera can never resolve.  Checking them here
means the layout is SOLVED rather than nudged until a run looked right.

WHAT IS CHECKED, AND WHY EACH ONE MATTERS
-------------------------------------------
* **The circuit is walkable.**  Every point on every leg clears every fixture by
  more than the duck's own conservative planar radius, and every corner is a
  60 deg LEFT turn the MEASURED yaw ceiling can carry.
* **Each approach is a real walk.**  The range from the checkpoint the anomaly
  is found at, to the planned standoff point, must be long enough that the
  approach measurably reduces range - not a body already standing at its own
  observation distance.
* **The standoff band is reachable.**  A standoff point must exist that clears
  the fixtures, stays out of the restricted zone, and is not behind a wall.
* **Each anomaly is inside the camera gate from somewhere on the circuit**, and
  the trolley and crate are far enough apart that one detection cannot be
  mistaken for the other.
* **Only one person ever enters the restricted zone.**

Run:
    ../../microduck_rl/.venv/bin/python tools/check_layout.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

from patrol_actors import (  # noqa: E402
    APPEARANCES,
    ROUTES,
    bodies_at,
    zone_occupancy,
)
from patrol_cast import ANOMALY_NAMES, BY_NAME  # noqa: E402
from patrol_facility import (  # noqa: E402
    CHECKPOINTS,
    CIRCUIT,
    FIXTURES,
    FLOOR_HALF,
    HOME,
    RESTRICTED_ZONE,
    occluder_between,
    static_gap,
    stowed_on,
)
from patrol_investigate import plan_standoff, range_for_standoff  # noqa: E402
from patrol_plan import circuit_length_m, corner_turns_deg  # noqa: E402
from patrol_states import (  # noqa: E402
    ATTENDED_RADIUS_M,
    DETECT_MAX_RANGE_M,
    DUCK_PLANAR_RADIUS,
    SPEED_AT_APPROACH,
    SPEED_AT_PATROL,
    STANDOFF_TARGET_M,
)

# The duck's eye height, for the planar occlusion check.
EYE_Z = 0.20
# An approach must reduce the range by at least this to count as a physical
# approach rather than a body already at its own observation distance.
MIN_APPROACH_WALK_M = 0.45


def check_circuit() -> list[tuple[str, bool, str]]:
    out = []
    worst = (9.9, "", (0.0, 0.0))
    previous = HOME.position
    for checkpoint in CIRCUIT:
        for index in range(121):
            point = previous + (checkpoint.position - previous) * (index / 120)
            name, gap = static_gap(point)
            if gap < worst[0]:
                worst = (gap, name, (round(float(point[0]), 3),
                                     round(float(point[1]), 3)))
        previous = checkpoint.position
    out.append((
        "every point on the circuit clears every fixture and wall",
        worst[0] > DUCK_PLANAR_RADIUS,
        f"worst {worst[0]:.4f} m to {worst[1]} at {worst[2]}, against the "
        f"duck's {DUCK_PLANAR_RADIUS:.4f} m planar radius"))

    turns = corner_turns_deg()
    out.append((
        "every circuit corner is a LEFT turn of about 60 deg",
        all(55.0 <= t <= 65.0 for t in turns),
        f"turns {turns}"))

    length = circuit_length_m()
    out.append((
        "the circuit takes a sensible fraction of the video to walk",
        30.0 <= length / SPEED_AT_PATROL <= 60.0,
        f"{length:.3f} m at {SPEED_AT_PATROL} m/s = "
        f"{length / SPEED_AT_PATROL:.1f}s of walking"))
    return out


def nearest_checkpoint(xy):
    """The circuit place nearest a point, and the range to it."""
    best, best_m = None, float("inf")
    for checkpoint in CIRCUIT:
        gap = float(np.linalg.norm(checkpoint.position
                                   - np.asarray(xy, dtype=np.float64)))
        if gap < best_m:
            best, best_m = checkpoint, gap
    return best, best_m


def check_anomalies() -> list[tuple[str, bool, str]]:
    out = []
    states = bodies_at(120.0)
    for name in ANOMALY_NAMES:
        position = states[name].pos
        checkpoint, range_m = nearest_checkpoint(position)
        plan = plan_standoff(name, position, checkpoint.position)
        standoff_range = range_for_standoff(name, STANDOFF_TARGET_M)
        walk = range_m - standoff_range

        out.append((
            f"{name}: an approach from {checkpoint.name} is a real walk",
            walk >= MIN_APPROACH_WALK_M,
            f"{range_m:.3f} m from {checkpoint.name} down to a "
            f"{standoff_range:.3f} m standoff range = {walk:.3f} m of "
            f"approach, {walk / SPEED_AT_APPROACH:.1f}s at the approach "
            f"command (bar {MIN_APPROACH_WALK_M} m)"))

        out.append((
            f"{name}: a legal standoff point exists",
            plan.ok,
            (f"chose {plan.chosen.xy} on bearing "
             f"{plan.chosen.bearing_deg:+.0f} deg, "
             f"{plan.chosen.fixture_gap_m:.3f} m from the nearest fixture, "
             f"{plan.chosen.zone_gap_m:+.3f} m outside the zone; "
             f"{sum(1 for c in plan.candidates if not c.ok)} of "
             f"{len(plan.candidates)} candidates rejected")
            if plan.ok else "no candidate survived"))

        out.append((
            f"{name}: inside the camera gate from {checkpoint.name}",
            range_m <= DETECT_MAX_RANGE_M,
            f"{range_m:.3f} m against the {DETECT_MAX_RANGE_M} m gate"))

        blocker = occluder_between(checkpoint.position, position)
        out.append((
            f"{name}: an unobstructed sightline from {checkpoint.name}",
            blocker is None,
            f"blocked by {blocker}" if blocker else "clear"))
    return out


def check_zone() -> list[tuple[str, bool, str]]:
    out = []
    occupancy = zone_occupancy(120.0)
    inside = {k: v for k, v in occupancy.items() if v > 0.0}
    out.append((
        "exactly one person ever enters the restricted zone",
        list(inside) == ["visitor"],
        f"seconds inside: {inside or 'nobody'}"))

    states = bodies_at(120.0)
    depth = RESTRICTED_ZONE.depth_inside(states["visitor"].pos)
    out.append((
        "the intruder ends up well inside the zone, not clipping its edge",
        depth >= 0.10,
        f"{depth:.3f} m inside the marked rectangle at "
        f"({states['visitor'].pos[0]:.3f}, {states['visitor'].pos[1]:.3f})"))

    # The duck must be able to observe the intruder from outside the zone.
    checkpoint, _ = nearest_checkpoint(states["visitor"].pos)
    plan = plan_standoff("visitor", states["visitor"].pos, checkpoint.position)
    out.append((
        "the intruder can be observed WITHOUT the duck entering the zone",
        plan.ok and plan.chosen.zone_gap_m > 0.0,
        (f"standoff {plan.chosen.xy} sits {plan.chosen.zone_gap_m:+.3f} m "
         f"outside the marked rectangle") if plan.ok else "no legal standoff"))
    return out


def check_population() -> list[tuple[str, bool, str]]:
    out = []
    # Nobody may be inside a fixture at any time, and no staff route may pass
    # through one: a person walking through a shelf is a scenario bug that
    # would surface as a bizarre occlusion rather than as an obvious one.
    #
    # AN OBJECT ON ITS OWN DESIGNATED STOW AREA IS EXEMPT, AND THAT EXEMPTION IS
    # THE SCENARIO.  The trolley stands ON the stow pallet - that is the rule
    # that makes it benign - so a check forbidding every overlap would forbid
    # the distractor this behavior exists to dismiss.  The exemption is narrow:
    # it applies only to a body whose own measured position ``stowed_on``
    # returns a stow area for, which is the identical predicate the duck's
    # classifier uses.
    worst = (9.9, "", "", 0.0)
    for step in range(0, 1201):
        t = step * 0.1
        for name, state in bodies_at(t).items():
            if not state.present:
                continue
            if not BY_NAME[name].is_person and stowed_on(state.pos):
                continue
            fixture, gap = static_gap(state.pos)
            if gap < worst[0]:
                worst = (gap, name, fixture, t)
    out.append((
        "no body ever stands inside a fixture or a wall, except an object on "
        "its own designated stow area",
        worst[0] > 0.0,
        f"worst {worst[0]:.4f} m: {worst[1]} against {worst[2]} at "
        f"{worst[3]:.1f}s"))

    states = bodies_at(120.0)
    # The trolley really is on the stow pallet, which is one of the two rules
    # that makes it benign.  Checked positively rather than left implied by the
    # exemption above.
    stow = stowed_on(states["trolley"].pos)
    out.append((
        "the trolley stands on a designated stow area",
        bool(stow),
        f"stow area {stow!r} at "
        f"({states['trolley'].pos[0]:.3f}, {states['trolley'].pos[1]:.3f})"))

    # And a member of staff really is standing beside it, which is the other.
    gap = float(np.linalg.norm(states["emil"].pos - states["trolley"].pos))
    out.append((
        "a member of staff stands beside the trolley",
        gap <= ATTENDED_RADIUS_M,
        f"emil is {gap:.3f} m from the trolley, against the "
        f"{ATTENDED_RADIUS_M} m attendance radius"))

    # The crate must have NOBODY near it while the duck is working on it, or the
    # suspicious call would be wrong.  Checked over the whole window in which
    # the crate is present and the patrol is still running.
    worst_crate = (9.9, "", 0.0)
    for step in range(0, 901):
        t = step * 0.1
        states_t = bodies_at(t)
        if not states_t["crate"].present:
            continue
        for person in ("rosa", "emil", "nadia", "visitor"):
            gap = float(np.linalg.norm(
                states_t[person].pos - states_t["crate"].pos))
            if gap < worst_crate[0]:
                worst_crate = (gap, person, t)
    out.append((
        "nobody comes within the attendance radius of the crate in the first "
        "90 s",
        worst_crate[0] > ATTENDED_RADIUS_M,
        f"nearest was {worst_crate[1]} at {worst_crate[0]:.3f} m, "
        f"t={worst_crate[2]:.1f}s, against the {ATTENDED_RADIUS_M} m radius"))

    separations = []
    for a in ANOMALY_NAMES:
        for b in ANOMALY_NAMES:
            if a >= b:
                continue
            separations.append((
                round(float(np.linalg.norm(states[a].pos - states[b].pos)), 3),
                a, b))
    out.append((
        "the three anomalies are far apart, so one detection cannot be "
        "mistaken for another",
        min(s[0] for s in separations) >= 1.5,
        f"pairwise separations {sorted(separations)}"))
    return out


def main() -> int:
    print("=" * 92)
    print("FACILITY LAYOUT")
    print("=" * 92)
    print(f"  floor {2 * FLOOR_HALF[0]:.2f} x {2 * FLOOR_HALF[1]:.2f} m, "
          f"{len(FIXTURES)} fixtures, {len(CHECKPOINTS)} checkpoints, "
          f"circuit {circuit_length_m():.3f} m")
    print(f"  home {HOME.xy}")
    for checkpoint in CHECKPOINTS:
        print(f"    {checkpoint.name:<12} {checkpoint.xy}  watch "
              f"{checkpoint.watch_deg:+.0f} deg")
    print(f"  restricted zone {RESTRICTED_ZONE.center} half "
          f"{RESTRICTED_ZONE.half}")
    states = bodies_at(120.0)
    for name in ANOMALY_NAMES:
        checkpoint, range_m = nearest_checkpoint(states[name].pos)
        print(f"  {name:<9} ({BY_NAME[name].role:<10}) at "
              f"({states[name].pos[0]:+.2f}, {states[name].pos[1]:+.2f}), "
              f"{range_m:.3f} m from {checkpoint.name}, appears at "
              f"{APPEARANCES.get(name, {}).get('at_s', ROUTES[name].start_t if name in ROUTES else 0.0):.1f}s")

    results = (check_circuit() + check_anomalies() + check_zone()
               + check_population())
    print()
    print("=" * 92)
    print("LAYOUT CHECKS")
    print("=" * 92)
    for label, ok, evidence in results:
        print(f"  [{' OK ' if ok else 'FAIL'}] {label}")
        print(f"         {evidence}")
    passed = all(ok for _, ok, _ in results)
    print()
    print("LAYOUT OK" if passed else "LAYOUT FAILS")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
