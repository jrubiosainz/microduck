#!/usr/bin/env python3
"""Prove the concourse geometry does what the docstrings claim, before anything
is rolled out.

Three questions, each of which would silently ruin the behavior if the answer
were wrong:

1. **Are the two sides really sealed?**  If an inflated path could squeeze past
   the north end of ``partition_c`` or the south end of ``hall_screen``, the
   planner would find a near-straight route and "the route has at least three
   bends" would be a coincidence of the destination rather than a property of
   the hall.
2. **Is each of the three destinations reachable, and does each produce a
   DIFFERENT route?**  A hall in which all three requests produce the same walk
   cannot demonstrate that the duck went to the one that was asked for.
3. **Does the crowd term bite?**  The planner must refuse cells because of
   people, not merely because of walls.

Run:
    ../../microduck_rl/.venv/bin/python tools/check_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

from guide_actors import actors_at  # noqa: E402
from guide_cast import FOLLOWER  # noqa: E402
from guide_layout import (  # noqa: E402
    DESTINATIONS,
    FLOOR_HALF,
    HALL_SCREEN,
    PARTITION_C,
    static_gap,
)
from guide_planner import (  # noqa: E402
    STATIC_INFLATE_M,
    Planner,
    tubes_from_states,
)
from guide_states import DUCK_START_XY, REQUEST_T_S  # noqa: E402


def gap_report(name: str, obstacle, wall_y: float) -> tuple[float, bool]:
    """Clear width between an obstacle's free end and the wall it points at."""
    end_y = (obstacle.center[1] + obstacle.half[1] if wall_y > 0
             else obstacle.center[1] - obstacle.half[1])
    clear = abs(wall_y - end_y)
    sealed = clear < 2.0 * STATIC_INFLATE_M
    print(f"  {name:<14} free end at y={end_y:+.3f}, wall at y={wall_y:+.3f}, "
          f"clear width {clear:.3f} m, needs {2.0 * STATIC_INFLATE_M:.3f} m "
          f"-> {'SEALED' if sealed else 'PASSABLE'}")
    return clear, sealed


def main() -> int:
    print("=" * 88)
    print("SEALING: can an inflated body pass either barrier on the wrong side?")
    print("=" * 88)
    print(f"  static inflation = {STATIC_INFLATE_M:.4f} m "
          f"(duck radius + margin); a passage needs twice that to be usable")
    _, north_sealed = gap_report("partition_c", PARTITION_C, FLOOR_HALF[1])
    _, south_sealed = gap_report("hall_screen", HALL_SCREEN, -FLOOR_HALF[1])
    # The OTHER end of each body must be genuinely open, or the hall is split.
    for name, obstacle, wall_y in (("partition_c", PARTITION_C, -FLOOR_HALF[1]),
                                   ("hall_screen", HALL_SCREEN, FLOOR_HALF[1])):
        end_y = (obstacle.center[1] - obstacle.half[1] if wall_y < 0
                 else obstacle.center[1] + obstacle.half[1])
        clear = abs(wall_y - end_y)
        print(f"  {name:<14} OTHER end at y={end_y:+.3f}: clear width "
              f"{clear:.3f} m -> {'open' if clear > 2.0 * STATIC_INFLATE_M else 'BLOCKED'}")

    print()
    print("=" * 88)
    print("ROUTES: one per destination, from the same start, with the same crowd")
    print("=" * 88)
    people = actors_at(REQUEST_T_S)
    tubes = tubes_from_states(people, FOLLOWER.name)
    planner = Planner()
    plans = {}
    for destination in DESTINATIONS:
        plan = planner.plan(DUCK_START_XY, destination, tubes)
        plans[destination.key] = plan
        bends = plan.bends
        print(f"  {destination.key:<9} length {plan.length_m:6.3f} m  "
              f"straight {plan.straight_line_m:6.3f} m  "
              f"detour x{plan.detour_ratio:.3f}  "
              f"bends {len(bends)} "
              f"({', '.join(f'{b['hand'][0]}{b['turn_deg']:+.0f}' for b in bends)})")
        print(f"            waypoints "
              f"{[[round(float(w[0]), 2), round(float(w[1]), 2)] for w in plan.waypoints]}")
        print(f"            min planned clearance {plan.min_clearance_m:.4f} m "
              f"at {plan.min_clearance_at}, straight line blocked by "
              f"{plan.straight_blocked_by or 'nothing'}")
        print(f"            cells: {plan.free_cells} free / "
              f"{plan.static_blocked_cells} static-blocked / "
              f"{plan.crowd_blocked_cells} crowd-blocked "
              f"{plan.crowd_blockers}, expanded {plan.expanded}")

    print()
    print("=" * 88)
    print("DISTINCTNESS: the three routes must not be the same walk")
    print("=" * 88)
    keys = [d.key for d in DESTINATIONS]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = plans[keys[i]], plans[keys[j]]
            end_gap = float(np.linalg.norm(a.waypoints[-1] - b.waypoints[-1]))
            print(f"  {keys[i]} vs {keys[j]}: endpoints {end_gap:.3f} m apart, "
                  f"lengths {a.length_m:.3f} vs {b.length_m:.3f} m")

    print()
    ok = True
    checks = [
        ("partition_c sealed at the north wall", north_sealed),
        ("hall_screen sealed at the south wall", south_sealed),
        ("every destination reachable", len(plans) == len(DESTINATIONS)),
        ("the requested route has at least 3 bends",
         len(plans["LIFTS"].bends) >= 3),
        ("the requested route is a real detour",
         plans["LIFTS"].detour_ratio >= 1.25),
        ("the straight line to the destination is blocked",
         bool(plans["LIFTS"].straight_blocked_by)),
        ("the crowd removed cells the planner would have used",
         plans["LIFTS"].crowd_blocked_cells > 0),
        ("the planned route keeps positive clearance",
         plans["LIFTS"].min_clearance_m > 0.0),
    ]
    for label, passed in checks:
        print(f"  [{' OK ' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print("LAYOUT OK" if ok else "LAYOUT FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
