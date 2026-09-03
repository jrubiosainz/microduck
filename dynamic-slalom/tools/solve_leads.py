#!/usr/bin/env python3
"""Solve each crossing body's LEAD from its own geometry, through the real planner.

THE QUESTION THIS ANSWERS
---------------------------
A crossing body has to reach the duck's lane some seconds BEFORE the duck gets
there, so that it has genuinely vacated the side it came from and the duck can
pass behind it.  How many seconds?  Not a taste: it depends on how big the body
is, because a wider body has to travel further before a corridor is really
clear.

Rather than guess, this sweeps candidate leads through the SAME
``slalom_plan.choose_corridor`` the rollout uses, with the body on its own real
filleted route and the duck approaching at its MEASURED 0.129 m/s cruise, and
reports the smallest lead that produces a sequence which is:

* DECISIVE   - the planner names a side at every sampled instant;
* CORRECT    - that side is the one behind the body (south for a northbound
  body, north for a southbound one);
* WAIT-FREE  - no instant where neither corridor is safe.

MEASURED RESULT (this is where ``tune_phasing.LEAD_S`` comes from):

    mara   pedestrian   planning r = 0.26 m  ->  5.5 s
    ines   carries box  planning r = 0.36 m  ->  6.5 s
    noor   carries box  planning r = 0.36 m  ->  6.5 s
    tobin  pushes cart  planning r = 0.48 m  ->  7.5 s

The lead scales with the planning radius, which is the expected shape and is
why a single global figure could not work for a cast that mixes pedestrians,
box carriers and cart pushers.

E4's pair is deliberately excluded: those two exist to make BOTH corridors
unsafe at once, so their phasing comes from their relationship to each other
rather than from either one's clean-pass lead.

Run:
    ../../microduck_rl/.venv/bin/python tools/solve_leads.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

import slalom_actors as actors_module  # noqa: E402
from slalom_cast import BY_NAME  # noqa: E402
from slalom_cast import planning_radius  # noqa: E402
from slalom_plan import Track, choose_corridor, nearest_threat  # noqa: E402
from slalom_states import SPEED_AT_WALK  # noqa: E402

# The instants before the duck's arrival at which the decision is sampled.
SAMPLE_DT_S: tuple[float, ...] = (12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0)


def decision_sequence(name: str, northbound: bool, speed: float,
                      cross_x: float, lead_s: float,
                      cross_t: float = 30.0) -> list[str]:
    """What the real planner says as the duck closes on one crossing body."""
    route = actors_module._solve_start(
        actors_module._crossing_corners(cross_x, northbound), speed, cross_t)
    arrival = cross_t + lead_s
    sequence: list[str] = []
    for dt in SAMPLE_DT_S:
        t = arrival - dt
        duck_x = cross_x - SPEED_AT_WALK * dt
        position = route.pos_at(t)
        yaw = route.yaw_at(t)
        body_speed = route.speed_at(t)
        velocity = np.array([np.cos(yaw), np.sin(yaw)]) * body_speed
        tracks = [Track(name, np.asarray(position, dtype=np.float64),
                        velocity, planning_radius(name))]
        duck = np.array([duck_x, 0.0])
        threat, ttc, range_m = nearest_threat(duck, 0.0, tracks)
        if not threat:
            continue
        sequence.append(choose_corridor(
            duck, tracks, ttc_s=ttc, threat=threat,
            threat_range_m=range_m).side)
    return sequence


def solve_lead(name: str, northbound: bool, speed: float, cross_x: float,
               lo: float = 3.0, hi: float = 14.0,
               step: float = 0.5) -> tuple[float | None, list[str]]:
    """Smallest lead giving a decisive, correct, wait-free sequence."""
    want = "right" if northbound else "left"
    candidate = lo
    while candidate <= hi + 1e-9:
        sequence = decision_sequence(name, northbound, speed, cross_x,
                                     candidate)
        if sequence and all(side == want for side in sequence):
            return candidate, sequence
        candidate += step
    return None, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=0.5)
    args = parser.parse_args()

    # name, northbound, speed, cross_x - taken from the cast and the encounter
    # table so this tool cannot drift from the scenario it is solving for.
    subjects = [
        ("mara", True, 0.255, actors_module.ENCOUNTERS["E1"]["cross_x"]),
        ("tobin", False, 0.208, actors_module.ENCOUNTERS["E2"]["cross_x"]),
        ("ines", True, 0.232, actors_module.ENCOUNTERS["E3"]["cross_x"]),
        ("noor", True, 0.238, actors_module.ENCOUNTERS["E5"]["cross_x"]),
    ]

    print("=" * 86)
    print("SOLVED LEADS  (smallest decisive, correct-side, wait-free lead)")
    print("=" * 86)
    print(f"  {'body':>7} {'kind':>11} {'plan r':>7} {'want':>6} {'lead':>6}  "
          f"sequence")
    results: dict[str, float] = {}
    for name, northbound, speed, cross_x in subjects:
        lead, sequence = solve_lead(name, northbound, speed, float(cross_x),
                                    step=args.step)
        want = "right" if northbound else "left"
        results[name] = lead if lead is not None else float("nan")
        shown = "none found" if lead is None else f"{lead:.1f}s"
        print(f"  {name:>7} {BY_NAME[name].kind:>11} "
              f"{planning_radius(name):7.2f} {want:>6} {shown:>6}  {sequence}")

    print()
    print("LEAD_S for tools/tune_phasing.py:")
    print("LEAD_S: dict[str, float] = {")
    print("    " + ", ".join(f'"{n}": {v}' for n, v in results.items()) + ",")
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
