#!/usr/bin/env python3
"""Solve each encounter's crossing time against the duck's MEASURED progress.

THE PROBLEM THIS EXISTS TO SOLVE
----------------------------------
An encounter is only an encounter if the crossing body is on the duck's lane
while the duck is close enough for it to matter.  Both halves are physics: the
body's arrival is its route length over its speed, and the DUCK's arrival is
whatever its measured 0.129 m/s cruise, its 0.097 m/s careful command, the
MEASURED 0.64 m of course each sidestep costs and any waiting add up to.

Guessing the duck's half is how the first draft of this behavior failed.  Its
crossing times were chosen from the cruise speed alone, so the duck reached
``x = -2.60`` at 11.6 s while ``mara`` did not cross until 15.4 s: it walked
through the crossing point 3.8 s early, the 7.0 s prediction horizon never saw
a conflict, and the encounter that did eventually happen was a late surprise
that produced a MEASURED -0.038 m overlap.  The symptom looked like a planner
that chose the wrong side; the cause was a scenario whose timing was wishful.

HOW IT IS SOLVED
-----------------
This tool runs the REAL rollout and records the time at which the duck's trunk
passes each encounter's ``cross_x``.  Those measured arrival times are what
``slalom_actors.ENCOUNTERS`` should be phased against: set each ``cross_t`` so
the body is on the lane when the duck is about ``ENGAGE_LEAD_M`` short of the
crossing point, which at the measured cruise is the lead the planner needs to
commit to a corridor and converge onto it.

IT ITERATES, AND THAT IS NOT A WEAKNESS
-----------------------------------------
Changing the phasing changes what the duck does, which changes its arrival
times.  Two or three passes converge, because each encounter costs a bounded and
MEASURED amount of time.  The alternative - deriving the duck's schedule
analytically - would require modelling the controller, which is the thing under
test.

Run:
    ../../microduck_rl/.venv/bin/python tools/tune_phasing.py --seconds 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

from slalom_actors import ENCOUNTERS, ROUTES  # noqa: E402
from slalom_cast import BY_ENCOUNTER, ENCOUNTER_ORDER  # noqa: E402
from slalom_states import SPEED_AT_CAREFUL  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# How far AHEAD of the duck each body crosses its lane, in seconds, PER BODY.
#
# THIS IS THE NUMBER THAT MAKES THE SCENARIO LEGIBLE, AND THE FIRST DRAFT HAD IT
# WRONG TWICE.
#
# Phasing each body to reach the lane at the same instant as the duck is the
# maximally AMBIGUOUS case: both corridors score within a few centimetres of
# each other, the planner takes whichever is marginally better, and the sides
# have no relation to where the traffic came from.  Measured over a full run
# that produced left, right, right, left, left, left - no alternation, and two
# passes that ran to their ceilings because the duck ended up walking alongside
# a body going the same way.
#
# A single global lead was the second mistake.  A body that crosses AHEAD of the
# duck vacates the side it came from, so the duck passes behind it - but how far
# ahead it must be depends on HOW BIG IT IS.  ``tools/solve_leads.py`` sweeps
# each body's own geometry through the real planner and reports the smallest
# lead that yields a decisive, wait-free, correct-side sequence:
#
#   mara   pedestrian  planning r=0.26  ->  5.5 s
#   ines   carries box planning r=0.36  ->  6.5 s
#   noor   carries box planning r=0.36  ->  6.5 s
#   tobin  pushes cart planning r=0.48  ->  7.5 s
#
# The lead scales with the planning radius, which is what one would expect: a
# wider body has to travel further before it has genuinely vacated a corridor.
LEAD_S: dict[str, float] = {
    "mara": 5.5, "tobin": 7.5, "ines": 6.5, "noor": 6.5,
}
# E4 is deliberately NOT in that table: its pair exists to make BOTH corridors
# unsafe at once, so its phasing is set by the two bodies' relationship to each
# other rather than by either one's clean-pass lead.
E4_LEAD_S = 2.0


def arrival_times(records: list[dict]) -> dict[str, float]:
    """When the duck's trunk first passed each encounter's crossing x."""
    out: dict[str, float] = {}
    for key in ENCOUNTER_ORDER:
        cross_x = float(ENCOUNTERS[key]["cross_x"])
        for record in records:
            if float(record["duck_xy"][0]) >= cross_x:
                out[key] = float(record["t"])
                break
    return out


def wait_before(records: list[dict], cross_x: float) -> float:
    """Seconds the duck spent stopped before it reached ``cross_x``.

    THIS IS WHAT MAKES THE SOLVER CONVERGE INSTEAD OF RUNNING AWAY.

    The naive loop - measure when the duck arrived, phase the body a lead
    earlier, repeat - DIVERGES whenever the duck waits.  A wait delays the
    arrival, which pushes the solved crossing later, which makes the body arrive
    later still, which makes the duck wait longer.  Measured over three
    iterations the crossing times marched from 15.2 s to 15.9 s to 16.6 s while
    the waits grew from 3.3 s to 9.5 s.

    A wait is a symptom of BAD PHASING, not a property of the schedule the
    phasing should be solved against.  So the target is the duck's UNIMPEDED
    arrival: what it measured, minus the time it spent standing still on the way
    there.  Solving against that removes the feedback loop, because the quantity
    no longer depends on the waits it is trying to eliminate.
    """
    stopped = 0.0
    previous_t = 0.0
    for record in records:
        t = float(record["t"])
        if float(record["duck_xy"][0]) >= cross_x:
            break
        if record["state"] in ("WAIT", "THREAT") \
                and float(record["command_peak"]) == 0.0:
            stopped += t - previous_t
        previous_t = t
    return stopped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",
                        default=str(REPO / "onnx" / "alpha_walking.onnx"))
    parser.add_argument("--seconds", type=float, default=100.0)
    parser.add_argument("--trace", default="",
                        help="read a trace instead of running the rollout")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    if args.trace:
        records = json.loads(Path(args.trace).read_text())
    else:
        from rollout_slalom import SlalomRollout
        rollout = SlalomRollout(args.policy, args.seconds)
        last = [None]

        def progress(index, record):
            if record["state"] != last[0]:
                print(f"  t={record['t']:6.2f}s  {record['state']:<13} "
                      f"x={record['duck_xy'][0]:+6.2f} "
                      f"y={record['duck_xy'][1]:+6.2f} "
                      f"thr={record['threat'] or '-'}")
                last[0] = record["state"]
        rollout.run(progress=progress)
        records = rollout.records

    arrivals = arrival_times(records)

    print()
    print("=" * 92)
    print("MEASURED DUCK PROGRESS versus the DECLARED crossing times")
    print("=" * 92)
    print(f"  {'enc':>4} {'cross_x':>8} {'declared':>9} {'duck at x':>10} "
          f"{'waited':>7} {'unimpeded':>10} {'wanted':>8}   solved cross_t")
    solved: dict[str, float] = {}
    for key in ENCOUNTER_ORDER:
        cross_x = float(ENCOUNTERS[key]["cross_x"])
        declared = float(ENCOUNTERS[key]["cross_t"])
        arrival = arrivals.get(key)
        if arrival is None:
            print(f"  {key:>4} {cross_x:8.2f} {declared:9.2f} "
                  f"{'never':>10} {'-':>7} {'-':>10} {'-':>8}   (duck never "
                  f"got there)")
            continue
        # Solve against the UNIMPEDED arrival - see ``wait_before``.
        waited = wait_before(records, cross_x)
        unimpeded = arrival - waited
        first = BY_ENCOUNTER[key][0]
        lead = LEAD_S.get(first, E4_LEAD_S)
        wanted = unimpeded - lead
        solved[key] = round(wanted, 2)
        print(f"  {key:>4} {cross_x:8.2f} {declared:9.2f} {arrival:10.2f} "
              f"{waited:7.2f} {unimpeded:10.2f} {wanted:8.2f}   "
              f"{solved[key]:.2f}")

    print()
    print("SUGGESTED slalom_actors.ENCOUNTERS (each body crosses its own")
    print("measured lead AHEAD of the duck, so the duck passes BEHIND it):")
    print("ENCOUNTERS: dict[str, dict] = {")
    for key in ENCOUNTER_ORDER:
        cross_x = float(ENCOUNTERS[key]["cross_x"])
        value = solved.get(key, float(ENCOUNTERS[key]["cross_t"]))
        print(f'    "{key}": {{"cross_x": {cross_x}, "cross_t": {value}}},')
    print("}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"arrivals": arrivals, "solved": solved}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
