#!/usr/bin/env python3
"""Sweep per-actor start offsets until the planning instant is feasible AND the
requested route has the shape the scenario needs.

Kept in the tree so the chosen offsets in ``guide_actors.START_OFFSETS`` can be
re-derived rather than taken on trust.  It is a search over the SCENARIO, not
over the planner: the planner is never relaxed to make a route appear.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import guide_actors as ga  # noqa: E402
from guide_cast import FOLLOWER  # noqa: E402
from guide_layout import DESTINATION_BY_KEY  # noqa: E402
import guide_planner as gp  # noqa: E402
from guide_states import DUCK_START_XY, REQUEST_T_S  # noqa: E402

NAMES = ("noor", "pablo", "ivan", "sena", "omar", "tessa")
BASE = {name: ga.ROUTES[name].corners for name in NAMES}
SPEED = {name: ga.ROUTES[name].speed for name in NAMES}
GRID = (0.0, 4.0, 8.0, 12.0, 16.0)


def apply(offsets: dict[str, float]) -> None:
    for name in NAMES:
        ga.ROUTES[name] = ga.Route(name, BASE[name], SPEED[name],
                                   start_t=-offsets[name])


def score() -> tuple | None:
    people = ga.actors_at(REQUEST_T_S)
    tubes = gp.tubes_from_states(people, FOLLOWER.name)
    planner = gp.Planner()
    try:
        plans = {key: planner.plan(DUCK_START_XY, DESTINATION_BY_KEY[key], tubes)
                 for key in ("LIFTS", "CAFE", "HELPDESK")}
    except RuntimeError:
        return None
    lifts = plans["LIFTS"]
    if len(lifts.bends) < 3 or lifts.crowd_blocked_cells <= 0:
        return None
    if lifts.detour_ratio < 1.25 or not lifts.straight_blocked_by:
        return None
    return (len(lifts.bends), round(lifts.length_m, 3),
            lifts.crowd_blocked_cells, round(lifts.min_clearance_m, 3),
            round(lifts.detour_ratio, 3), lifts.corner_radius_m,
            [b["hand"] for b in lifts.bends],
            round(plans["CAFE"].length_m, 2),
            round(plans["HELPDESK"].length_m, 2))


def main() -> int:
    # Coordinate descent: six actors over a 5-value grid is 15625 full plans,
    # which is minutes of A*.  Descent over one actor at a time reaches the same
    # feasible region in about 30 plans.
    offsets = {name: 0.0 for name in NAMES}
    best = None
    for sweep in range(3):
        for name in NAMES:
            for value in GRID:
                trial = dict(offsets, **{name: value})
                apply(trial)
                result = score()
                if result is None:
                    continue
                key = (result[0], -abs(result[1] - 9.0), result[3])
                if best is None or key > best[0]:
                    best = (key, dict(trial), result)
        if best is not None:
            offsets = dict(best[1])
    if best is None:
        print("no feasible offsets on this grid")
        return 1
    print("offsets:", best[1])
    print("bends, length_m, crowd_cells, min_clr, detour, radius, hands, "
          "cafe_len, helpdesk_len")
    print(best[2])
    # Prove the neighbouring instants are feasible too, so the scenario is not
    # balanced on a single lucky tick.
    apply(best[1])
    planner = gp.Planner()
    for t in (REQUEST_T_S - 0.4, REQUEST_T_S, REQUEST_T_S + 0.4,
              REQUEST_T_S + 0.8):
        people = ga.actors_at(t)
        tubes = gp.tubes_from_states(people, FOLLOWER.name)
        try:
            plan = planner.plan(DUCK_START_XY, DESTINATION_BY_KEY["LIFTS"],
                                tubes)
            print(f"  t={t:5.2f}s  bends={len(plan.bends)}  "
                  f"len={plan.length_m:.3f} m  crowd={plan.crowd_blocked_cells}")
        except RuntimeError:
            print(f"  t={t:5.2f}s  SEALED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
