#!/usr/bin/env python3
"""The planning WORLD: inflation figures, the crowd's swept tubes, and the Plan.

Split from ``guide_planner`` so that WHAT the search is searching — free space,
margins, moving people — is a separate file from HOW it searches.  Nothing here
runs A*; nothing here knows about grids.

THE INFLATION IS DERIVED FROM THE ROBOT, NOT PICKED
-----------------------------------------------------
``STATIC_INFLATE_M`` is the duck's conservative planar radius (measured from the
built scene at 0.1303 m, bounding-sphere based and therefore over-stating the
robot) plus a working margin.  A path planned with it can be walked by a body of
that radius with the margin to spare, which is what makes the executed clearance
gate a check on the CONTROLLER's tracking rather than on the planner's optimism.

THE CROWD TERM IS REQUIRED TO BITE, AND THAT IS GRADED
-------------------------------------------------------
A planner that "avoids the crowd" in an empty corridor has proved nothing.
:class:`Plan` therefore reports ``crowd_blocked_cells`` — the number of grid
cells that were free of every static body and were refused ONLY because a
person's swept tube covered them — and the acceptance gate requires it to be
positive.  It also reports which people caused those refusals, so the claim
names somebody.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guide_cast import PLANNING_HALF_EXTENT_M
from guide_layout import FLOOR_HALF, OBSTACLES

# -- planning parameters ----------------------------------------------------
# Grid resolution.  0.09 m is finer than a third of the narrowest gap the route
# must use, so a passage that exists is never missed for want of a sample.
GRID_M = 0.09
# Static inflation: the duck's conservative planar radius (0.1303 m, measured)
# plus a 0.17 m working margin.  ``test_static_inflation_matches_the_duck``
# pins the first term against the built scene.
DUCK_PLANNING_RADIUS_M = 0.1303
STATIC_MARGIN_M = 0.17
STATIC_INFLATE_M = DUCK_PLANNING_RADIUS_M + STATIC_MARGIN_M
# Crowd inflation: their planning half-extent plus the duck's radius plus a
# margin.  Larger than the static figure because a person moves and the
# extrapolation is linear and therefore wrong in detail.
CROWD_INFLATE_M = PLANNING_HALF_EXTENT_M + DUCK_PLANNING_RADIUS_M + 0.14
# Fractions of that margin the planner will fall back through, in order.
#
# WHY A FALLBACK EXISTS, AND WHY IT IS NOT A RELAXATION OF SAFETY.  In a hall
# whose passages are a little over a metre of usable width, ONE adult standing
# in the one way through covers it at full margin, and the search correctly
# reports no route.  That is the right answer to the question "is there a path
# that keeps 0.55 m from everybody" and the wrong answer to "what should the
# guide do": a person walking across a doorway is not a wall, and a guide that
# refused to move until the hall was empty would be useless.
#
# So the planner retries at reduced crowd margin and REPORTS which tier it
# needed.  Two things keep this honest:
#
#   * the STATIC inflation never changes — the duck's clearance to walls,
#     partitions and fixtures is not negotiable, and the executed clearance gate
#     measures it every tick against the real geoms anyway;
#   * the tier used is in the metrics and in the HUD, so a run that needed the
#     0.45 tier says so rather than looking identical to one that did not.
#
# The floor is the duck's own radius plus a person's half-extent, which is the
# point at which "avoid them" stops meaning anything at all.
CROWD_MARGIN_TIERS = (1.00, 0.80, 0.62, 0.48, 0.36)
CROWD_INFLATE_FLOOR_M = PLANNING_HALF_EXTENT_M + DUCK_PLANNING_RADIUS_M
# How far ahead each person is swept when their tube is built.  MEASURED
# against the cast rather than picked: at ``ivan``'s 0.205 m/s a 4 s sweep is
# 0.82 m, which inflates his tube to about 1.9 m long.  Sweeping 9 s instead —
# the first value tried — made every walking adult a 2.4 m wall and SEALED the
# concourse outright, which is not caution but a planner that has stopped
# describing the hall.  Four seconds is also about as far as a straight-line
# extrapolation of a walking person is worth making.
CROWD_HORIZON_S = 4.0
CROWD_SAMPLES = 8
# Cells nearer than this to a static surface pay a soft cost, so the search
# prefers the middle of a passage over scraping its edge.  It is a PREFERENCE:
# a cell is never made unreachable by it, or a narrow but usable gap would be
# refused for being narrow.
COMFORT_M = 0.42
COMFORT_WEIGHT = 0.85

# The filleted route must keep at least this much clearance from every static
# surface.  DERIVED: the duck's conservative planar radius plus 0.06 m, so a
# body of that radius tracking the centreline exactly stays off every surface
# with margin.
ROUTE_CLEARANCE_M = DUCK_PLANNING_RADIUS_M + 0.06
# Corner radii tried, largest first.  THE FILLET IS NOT FREE: it cuts inside the
# corner it rounds, so a radius that suits an open boulevard drives the route
# through the very body the corner was going round.  MEASURED here: at r = 0.62
# the LIFTS route's corner past the screen's north end reported a planned
# clearance of -0.0025 m — the centreline was inside the screen.  The planner
# therefore takes the LARGEST radius whose filleted route clears
# :data:`ROUTE_CLEARANCE_M`, and reports which one it took.
CORNER_RADII = (0.62, 0.52, 0.44, 0.36, 0.30, 0.24, 0.18)
# How far back along the fixture's own axis the final approach waypoint sits.
# Long enough that the fillet before it does not eat the straight run, so the
# duck is genuinely heading at the fixture when it arrives.
APPROACH_OFFSET_M = 0.74


def _clearance(xy) -> tuple[float, str]:
    """Gap from ``xy`` to the nearest static surface, and its name."""
    best, name = float("inf"), ""
    for obstacle in OBSTACLES:
        gap = obstacle.distance_to(xy)
        if gap < best:
            best, name = gap, obstacle.name
    walls = float(min(FLOOR_HALF[0] - abs(float(xy[0])),
                      FLOOR_HALF[1] - abs(float(xy[1]))))
    return (best, name) if best <= walls else (walls, "wall")


@dataclass(frozen=True)
class CrowdTube:
    """One person's swept planning tube: measured pose, extrapolated forward.

    Built from a position and a velocity — what the duck could measure — and
    never from a route object, so the planner cannot consult the scenario's own
    schedule.
    """

    name: str
    pos: np.ndarray
    velocity: np.ndarray

    def blocks(self, xy, inflate: float = CROWD_INFLATE_M,
               horizon: float = CROWD_HORIZON_S,
               samples: int = CROWD_SAMPLES) -> bool:
        point = np.asarray(xy, dtype=np.float64)
        for index in range(samples):
            dt = horizon * index / max(samples - 1, 1)
            if float(np.linalg.norm(self.pos + self.velocity * dt - point)) \
                    < inflate:
                return True
        return False

    def as_record(self) -> dict:
        return {
            "name": self.name,
            "pos": [round(float(self.pos[0]), 4), round(float(self.pos[1]), 4)],
            "velocity": [round(float(self.velocity[0]), 4),
                         round(float(self.velocity[1]), 4)],
            "speed_mps": round(float(np.linalg.norm(self.velocity)), 4),
        }


@dataclass
class Plan:
    """A searched route, with the evidence that it was searched.

    Everything the acceptance gate needs about the PLAN — as opposed to about
    the walk — is here, so a gate never has to re-derive it and can never grade
    a different route from the one that was flown.
    """

    destination_key: str
    waypoints: list[np.ndarray]
    route: Route
    grid_cells: int
    free_cells: int
    static_blocked_cells: int
    crowd_blocked_cells: int
    crowd_blockers: dict[str, int]
    tubes: list[CrowdTube]
    expanded: int
    raw_vertices: int
    min_clearance_m: float
    min_clearance_at: tuple[float, float]
    straight_line_m: float
    detour_ratio: float
    straight_blocked_by: str
    corner_radius_m: float
    radii_tried: int
    crowd_inflate_used_m: float
    crowd_tier_used: float
    crowd_tiers_tried: int

    @property
    def length_m(self) -> float:
        return float(self.route.length)

    @property
    def bends(self) -> list[dict]:
        return self.route.corner_report()

    def as_record(self) -> dict:
        return {
            "destination": self.destination_key,
            "waypoints": [[round(float(w[0]), 4), round(float(w[1]), 4)]
                          for w in self.waypoints],
            "length_m": round(self.length_m, 4),
            "straight_line_m": round(self.straight_line_m, 4),
            "detour_ratio": round(self.detour_ratio, 4),
            "straight_line_blocked_by": self.straight_blocked_by,
            "bends": self.bends,
            "bend_count": len(self.bends),
            "grid_cells": self.grid_cells,
            "free_cells": self.free_cells,
            "static_blocked_cells": self.static_blocked_cells,
            "crowd_blocked_cells": self.crowd_blocked_cells,
            "crowd_blockers": dict(self.crowd_blockers),
            "crowd_tubes": [t.as_record() for t in self.tubes],
            "cells_expanded": self.expanded,
            "raw_vertices": self.raw_vertices,
            "simplified_vertices": len(self.waypoints),
            "min_planned_clearance_m": round(self.min_clearance_m, 4),
            "min_planned_clearance_at": [
                round(self.min_clearance_at[0], 4),
                round(self.min_clearance_at[1], 4)],
            "route_clearance_required_m": round(ROUTE_CLEARANCE_M, 4),
            "corner_radius_m": round(self.corner_radius_m, 4),
            "corner_radii_tried": self.radii_tried,
            "static_inflate_m": round(STATIC_INFLATE_M, 4),
            "crowd_inflate_m": round(self.crowd_inflate_used_m, 4),
            "crowd_inflate_full_m": round(CROWD_INFLATE_M, 4),
            "crowd_tier_used": self.crowd_tier_used,
            "crowd_tiers_tried": self.crowd_tiers_tried,
            "crowd_inflate_floor_m": round(CROWD_INFLATE_FLOOR_M, 4),
            "grid_m": GRID_M,
        }
