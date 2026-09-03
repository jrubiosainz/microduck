#!/usr/bin/env python3
"""Derive the scripted choreography from the duck's MEASURED leg times.

Every ``start_t`` and hold window in ``etiquette_actors.ROUTES``, and every edge
in ``lobby_doors.DOOR_SCHEDULE``, has to line up with the instant the duck really
arrives somewhere.  ``tools/measure_legs.py`` supplies those instants; this tool
turns them into the timeline and CHECKS the result against the invariants the
acceptance gate will grade, before a single frame is rendered.

The point is that the schedule is a CONSEQUENCE of the measurement rather than a
set of numbers tuned until a run passed.  Anything this tool reports as
infeasible is infeasible, and the building or the pace has to change - not the
gate.

WHAT IT SOLVES FOR
-------------------
* the guardian must be AHEAD of the duck on the duck's own route at every
  instant, and never more than ``MAX_GUARDIAN_GAP_M`` ahead;
* she must be clear of each aperture before the duck enters it;
* the two door exiters must be through the doorway and clear of the duck's
  corridor before the duck's yield can end;
* the lift doors must open after the duck has stood beside them for at least
  ``MIN_WAIT_SIDE_S``, and the three occupants must all be out before it boards;
* the front doors must shut after the duck is inside, and the rear doors open
  after at least ``MIN_RIDE_S`` of riding.

Run:
    ../../microduck_rl/.venv/bin/python tools/tune_phasing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

from etiquette_actors import ROUTES  # noqa: E402
from etiquette_cast import DOOR_EXITER_NAMES, OCCUPANT_NAMES  # noqa: E402
from etiquette_path import LEG_NAMES, build_route, leg_bounds  # noqa: E402
from etiquette_sense import (  # noqa: E402
    guardian_arc_on_duck_route,
    past_plane,
)
from etiquette_states import (  # noqa: E402
    EXITER_CLEAR_M,
    GUARDIAN_THROUGH_M,
    MIN_RIDE_S,
    MIN_WAIT_SIDE_S,
    OCCUPANT_EXITED_M,
)
from etiquette_thresholds import MAX_GUARDIAN_GAP_M  # noqa: E402
from lobby_doors import DOOR_RAMP_S, doors_at, schedule_windows  # noqa: E402

# The duck's MEASURED arrival time at the end of each leg, walking the whole
# route with the real policy and no state machine.  Produced by
# ``tools/measure_legs.py``; re-run it after any geometry change.
MEASURED_LEG_ARRIVALS_S = (10.48, 23.70, 36.34, 53.88, 68.26)

# The scripted pauses the scenario inserts between those legs, each of which the
# state machine enters because of a MEASUREMENT rather than a clock.  These are
# what the schedule below has to make happen at the right moment.
YIELD_S = 9.26          # stopped outside the door while two people come out
WAIT_SIDE_S = 3.16      # standing beside the lift before the doors open
EXIT_S = 12.44          # doors travelling, then three occupants stepping out
RIDE_S = 8.06           # sealed car, exactly still; ended by the rear doors
TARGET_DOORS_S = 3.20   # rear doors opening, guardian stepping out first
DONE_TAIL_S = 3.60      # standing on the target floor at the end


def timeline() -> dict[str, float]:
    """The instant each phase begins, built from the measured leg times."""
    walk = list(MEASURED_LEG_ARRIVALS_S)
    leg_len = [walk[0]] + [walk[i] - walk[i - 1] for i in range(1, len(walk))]
    # MEASURED_LEG_ARRIVALS_S comes from tools/measure_legs.py, which walks the
    # route with no pauses at all, so its entries are cumulative WALKING time
    # and the differences above are the true per-leg durations.

    marks: dict[str, float] = {}
    t = 0.0
    marks["approach_door_from"] = t
    t += leg_len[0]
    marks["yield_from"] = t
    t += YIELD_S
    marks["follow_through_from"] = t
    t += leg_len[1]
    marks["approach_lift_from"] = t
    t += leg_len[2]
    marks["wait_side_from"] = t
    t += WAIT_SIDE_S
    marks["doors_open_from"] = t
    t += EXIT_S
    marks["board_from"] = t
    t += leg_len[3]
    marks["ride_from"] = t
    t += RIDE_S
    marks["target_doors_from"] = t
    t += TARGET_DOORS_S
    marks["follow_out_from"] = t
    t += leg_len[4]
    marks["done_from"] = t
    marks["total"] = t + DONE_TAIL_S
    return marks


def main() -> int:
    failures: list[str] = []

    def check(ok: bool, label: str, evidence: str) -> None:
        print(f"  [{' OK ' if ok else 'FAIL'}] {label}")
        print(f"         {evidence}")
        if not ok:
            failures.append(label)

    marks = timeline()
    route = build_route()
    bounds = leg_bounds(route)

    print("=" * 92)
    print("THE TIMELINE, BUILT FROM THE MEASURED LEG TIMES")
    print("=" * 92)
    for name, value in marks.items():
        print(f"    {name:<24} {value:7.2f} s")
    print(f"    route {route.length:.4f} m, legs {[round(b, 3) for b in bounds]}")

    print()
    print("=" * 92)
    print("THE DECLARED DOOR SCHEDULE, AGAINST THAT TIMELINE")
    print("=" * 92)
    for entry in schedule_windows():
        print(f"    {entry['door']:<15} opens {entry['opens_at_s']:6.2f}s "
              f"closes {str(entry['closes_at_s']):>6}  ramp {entry['ramp_s']}s")

    lift_opens = next(e["opens_at_s"] for e in schedule_windows()
                      if e["door"] == "lift_front")
    lift_closes = next(e["closes_at_s"] for e in schedule_windows()
                       if e["door"] == "lift_front")
    rear_opens = next(e["opens_at_s"] for e in schedule_windows()
                      if e["door"] == "lift_rear")
    door_opens = next(e["opens_at_s"] for e in schedule_windows()
                      if e["door"] == "concourse_door")

    check(door_opens + DOOR_RAMP_S <= marks["yield_from"],
          "the concourse door is fully open before the duck reaches its hold",
          f"opens {door_opens}s + {DOOR_RAMP_S}s ramp vs arrival "
          f"{marks['yield_from']:.2f}s")
    check(lift_opens >= marks["wait_side_from"] + MIN_WAIT_SIDE_S,
          f"the lift opens only after {MIN_WAIT_SIDE_S}s of standing beside it",
          f"duck arrives {marks['wait_side_from']:.2f}s, doors open "
          f"{lift_opens}s ({lift_opens - marks['wait_side_from']:.2f}s later)")
    check(lift_closes is not None
          and lift_closes >= marks["ride_from"],
          "the front doors shut only after the duck is aboard and positioned",
          f"duck positioned {marks['ride_from']:.2f}s, doors close "
          f"{lift_closes}s")
    check(rear_opens >= marks["ride_from"] + MIN_RIDE_S,
          f"the rear doors open only after {MIN_RIDE_S}s of riding",
          f"ride from {marks['ride_from']:.2f}s, rear opens {rear_opens}s "
          f"({rear_opens - marks['ride_from']:.2f}s later)")

    # THE DUCK'S RIDE IS BOUNDED BELOW BY THE DOOR SCHEDULE, NOT BY A TIMER.
    # RIDE ends when the rear doors are MEASURED to open, so the ride's real
    # duration is ``rear_opens - ride_from`` whatever ``RIDE_S`` says here.
    # Reporting both keeps the estimate honest against the thing that ends it.
    print(f"         (the ride the machine will actually measure: "
          f"{rear_opens - marks['ride_from']:.2f}s, ended by the doors rather "
          f"than by a countdown)")

    print()
    print("=" * 92)
    print("THE GUARDIAN IS AHEAD OF THE DUCK AT EVERY INSTANT")
    print("=" * 92)
    # The duck's own arc as a function of time, piecewise from the measured leg
    # arrivals and the scripted pauses.  Linear within a leg, which is close
    # enough for a scheduling check: the acceptance gate measures the real thing.
    phases = [
        (marks["approach_door_from"], marks["yield_from"], 0.0, bounds[0]),
        (marks["yield_from"], marks["follow_through_from"], bounds[0], bounds[0]),
        (marks["follow_through_from"], marks["approach_lift_from"],
         bounds[0], bounds[1]),
        (marks["approach_lift_from"], marks["wait_side_from"],
         bounds[1], bounds[2]),
        (marks["wait_side_from"], marks["board_from"], bounds[2], bounds[2]),
        (marks["board_from"], marks["ride_from"], bounds[2], bounds[3]),
        (marks["ride_from"], marks["follow_out_from"], bounds[3], bounds[3]),
        (marks["follow_out_from"], marks["done_from"], bounds[3], bounds[4]),
    ]

    def duck_arc(t: float) -> float:
        for start, end, arc0, arc1 in phases:
            if t <= end:
                if end <= start:
                    return arc1
                u = max(0.0, (t - start) / (end - start))
                return arc0 + (arc1 - arc0) * min(u, 1.0)
        return bounds[-1]

    worst_gap, worst_t = float("inf"), 0.0
    widest_gap, widest_t = -float("inf"), 0.0
    for step in range(int(marks["total"] * 20) + 1):
        t = step / 20.0
        gap = guardian_arc_on_duck_route(
            route, ROUTES["nadia"].pos_at(t)) - duck_arc(t)
        if gap < worst_gap:
            worst_gap, worst_t = gap, t
        if gap > widest_gap:
            widest_gap, widest_t = gap, t
    check(worst_gap > 0.0,
          "the guardian is never behind the duck on the duck's own route",
          f"minimum gap {worst_gap:+.4f} m at {worst_t:.2f}s"
          + ("" if worst_gap > 0.0 else
             "  (NOTE: this estimate walks the duck at a constant rate within "
             "each leg; the rollout measures the real thing)"))
    check(widest_gap <= MAX_GUARDIAN_GAP_M,
          f"and never more than {MAX_GUARDIAN_GAP_M} m ahead",
          f"maximum gap {widest_gap:.4f} m at {widest_t:.2f}s")

    print()
    print("=" * 92)
    print("THE DOOR EXITERS CLEAR BEFORE THE DUCK'S YIELD ENDS")
    print("=" * 92)
    for name in DOOR_EXITER_NAMES:
        actor = ROUTES[name]
        cleared = None
        entered = None
        for step in range(int(marks["total"] * 20) + 1):
            t = step / 20.0
            through = past_plane(actor.pos_at(t), "concourse_door", -1.0)
            if entered is None and abs(through) <= 0.22:
                entered = t
            if cleared is None and through >= EXITER_CLEAR_M:
                cleared = t
        print(f"    {name:<8} in the aperture ~{entered}s, "
              f"{EXITER_CLEAR_M} m clear at {cleared}s")
        check(cleared is not None
              and cleared <= marks["follow_through_from"] - 0.6,
              f"{name} is clear before the duck moves off",
              f"clear {cleared}s vs duck moving "
              f"{marks['follow_through_from']:.2f}s")

    print()
    print("=" * 92)
    print("THE GUARDIAN IS THROUGH EACH APERTURE BEFORE THE DUCK ENTERS IT")
    print("=" * 92)
    guard = ROUTES["nadia"]
    for aperture, sign, duck_enters, waits in (
            ("concourse_door", -1.0, marks["follow_through_from"], False),
            ("lift_front", -1.0, marks["board_from"], False),
            ("lift_rear", +1.0, marks["follow_out_from"], True)):
        through_at = None
        for step in range(int(marks["total"] * 20) + 1):
            t = step / 20.0
            value = past_plane(guard.pos_at(t), aperture, sign)
            if through_at is None and (value <= -GUARDIAN_THROUGH_M
                                       if sign < 0
                                       else value >= GUARDIAN_THROUGH_M):
                through_at = t
                break
        print(f"    {aperture:<15} she is {GUARDIAN_THROUGH_M} m through at "
              f"{through_at}s; the duck's leg begins {duck_enters:.2f}s")
        if waits:
            # DOORS_OPEN_TARGET ENDS ON HER, NOT ON A CLOCK.  The duck holds at
            # exactly zero until it MEASURES her through the rear aperture, so
            # the only thing to check here is that she gets through at all and
            # that the wait it implies is bounded.  Comparing her exit against a
            # fixed estimate of the duck's leg start would be checking this
            # tool's own arithmetic, not the scenario.
            implied = through_at - (marks["follow_out_from"] - TARGET_DOORS_S)
            print(f"                    the duck's DOORS_OPEN_TARGET wait ends "
                  f"on HER, implying {implied:.2f}s of holding")
            check(through_at is not None and 0.0 < implied <= 12.0,
                  "the duck's wait for her at the target floor is bounded",
                  f"she is through at {through_at}s, implying a "
                  f"{implied:.2f}s wait against the 12.0s ceiling")
        else:
            check(through_at is not None and through_at <= duck_enters,
                  f"she is through {aperture} before the duck's leg begins",
                  f"she clears it {through_at}s, the duck starts "
                  f"{duck_enters:.2f}s")

    print()
    print("=" * 92)
    print("THE GUARDIAN HERSELF NEVER WALKS THROUGH A CLOSED DOOR")
    print("=" * 92)
    print("   She is scripted, so nothing stops her - which is exactly why it")
    print("   has to be checked.  A guardian who strolled through a sealed lift")
    print("   would make the duck's own no-closed-doors gate look arbitrary.")
    print("   Graded at the instant she CROSSES each plane, not across the whole")
    print("   0.44 m aperture box: a door still finishing its ramp while she is")
    print("   approaching the sill is a door she walks through open.")
    for aperture, sign in (("concourse_door", -1.0), ("lift_front", -1.0),
                           ("lift_rear", +1.0)):
        crossed_at, fraction_then = None, None
        previous = None
        for step in range(int(marks["total"] * 20) + 1):
            t = step / 20.0
            side = past_plane(guard.pos_at(t), aperture, -1.0)
            if previous is not None and previous > 0.0 >= side:
                crossed_at = t
                fraction_then = doors_at(t)[aperture].fraction
                break
            previous = side
        if crossed_at is None:
            print(f"    {aperture:<15} she never crosses this plane")
            continue
        print(f"    {aperture:<15} crosses at {crossed_at:.2f}s with the door "
              f"{fraction_then:.3f} open")
        check(fraction_then >= 0.55,
              f"the guardian crossed {aperture} only while it was open",
              f"open fraction {fraction_then:.3f} at her crossing "
              f"({crossed_at:.2f}s)")

    print()
    print("=" * 92)
    print("THE LIFT OCCUPANTS ARE OUT BEFORE THE DUCK BOARDS")
    print("=" * 92)
    for name in OCCUPANT_NAMES:
        actor = ROUTES[name]
        cleared = None
        for step in range(int(marks["total"] * 20) + 1):
            t = step / 20.0
            if past_plane(actor.pos_at(t), "lift_front", -1.0) \
                    >= OCCUPANT_EXITED_M:
                cleared = t
                break
        print(f"    {name:<8} {OCCUPANT_EXITED_M} m out of the car at {cleared}s")
        check(cleared is not None and cleared <= marks["board_from"] - 0.6,
              f"{name} is out before the duck boards",
              f"clear {cleared}s vs boarding {marks['board_from']:.2f}s")

    print()
    print("=" * 92)
    print(f"TOTAL RUN {marks['total']:.2f} s")
    if failures:
        print(f"PHASING FAILED: {len(failures)} check(s)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PHASING OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
