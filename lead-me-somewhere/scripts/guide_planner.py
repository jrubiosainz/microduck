#!/usr/bin/env python3
"""The route search: A* over inflated free space, then a shortcut and a fillet.

WHY A SEARCH AND NOT A WAYPOINT LIST
-------------------------------------
Every earlier behavior in this lab moved along a route somebody wrote down.  A
guide cannot: the destination is chosen at run time out of three, and the hall is
sealed on two different sides by full-height bodies, so the shape of the route is
a CONSEQUENCE of which destination was asked for.  Writing the answer down would
make "the duck planned a route to the requested place" unfalsifiable, because the
same polyline would be produced whatever was requested.

So the plan is searched, every time, from the duck's measured pose, the resolved
destination, the inflated static bodies, and the crowd's measured swept tubes.

WHY THE PATH IS SHORTCUT AND THEN FILLETED
-------------------------------------------
Raw 8-connected A* output is a staircase: dozens of 45-degree kinks that are an
artifact of the grid, not features of the hall.  Filleting a staircase produces
dozens of tiny arcs and "the route has three bends" becomes meaningless.  The
path is therefore reduced by greedy line-of-sight shortcutting against the SAME
inflated obstacle set the search used, so every surviving vertex is a corner the
geometry actually required.  Only then is it filleted, and the bends it reports
are the bends the hall imposed.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from guide_layout import OBSTACLES, Destination
from guide_planning_space import (
    APPROACH_OFFSET_M,
    COMFORT_M,
    COMFORT_WEIGHT,
    CORNER_RADII,
    CROWD_INFLATE_FLOOR_M,
    CROWD_INFLATE_M,
    CROWD_MARGIN_TIERS,
    DUCK_PLANNING_RADIUS_M,
    GRID_M,
    ROUTE_CLEARANCE_M,
    STATIC_INFLATE_M,
    CrowdTube,
    Plan,
    _clearance,
)
from guide_layout import FLOOR_HALF
from guide_route import Route


class Planner:
    """Grid A* over inflated free space, then greedy shortcutting.

    Holds no scenario knowledge: it is given a start, a destination and a set of
    measured crowd tubes, and it returns the route those inputs imply.
    """

    def __init__(self, grid_m: float = GRID_M,
                 static_inflate: float = STATIC_INFLATE_M,
                 crowd_inflate: float = CROWD_INFLATE_M):
        self.grid_m = float(grid_m)
        self.static_inflate = float(static_inflate)
        self.crowd_inflate = float(crowd_inflate)
        self.nx = int(round(2.0 * FLOOR_HALF[0] / self.grid_m)) + 1
        self.ny = int(round(2.0 * FLOOR_HALF[1] / self.grid_m)) + 1

    # -- grid helpers -----------------------------------------------------
    def cell_xy(self, ix: int, iy: int) -> np.ndarray:
        return np.array([-FLOOR_HALF[0] + ix * self.grid_m,
                         -FLOOR_HALF[1] + iy * self.grid_m])

    def nearest_cell(self, xy) -> tuple[int, int]:
        point = np.asarray(xy, dtype=np.float64)
        ix = int(round((float(point[0]) + FLOOR_HALF[0]) / self.grid_m))
        iy = int(round((float(point[1]) + FLOOR_HALF[1]) / self.grid_m))
        return (int(np.clip(ix, 0, self.nx - 1)),
                int(np.clip(iy, 0, self.ny - 1)))

    def static_free(self, xy) -> bool:
        gap, _ = _clearance(xy)
        return gap >= self.static_inflate

    # -- the search --------------------------------------------------------
    def _build_masks(self, tubes: list[CrowdTube],
                     crowd_inflate: float | None = None):
        """Free / static-blocked / crowd-blocked masks, and the blame counts.

        The two blocked masks are kept SEPARATE rather than merged, because the
        acceptance gate has to be able to say how many cells the crowd alone
        removed.  Merging them would make the crowd term unfalsifiable.
        """
        inflate = self.crowd_inflate if crowd_inflate is None else crowd_inflate
        free = np.zeros((self.nx, self.ny), dtype=bool)
        static_blocked = 0
        crowd_blocked = 0
        blockers: dict[str, int] = {}
        clearance = np.zeros((self.nx, self.ny), dtype=np.float64)
        for ix in range(self.nx):
            for iy in range(self.ny):
                xy = self.cell_xy(ix, iy)
                gap, _ = _clearance(xy)
                clearance[ix, iy] = gap
                if gap < self.static_inflate:
                    static_blocked += 1
                    continue
                blocked_by = ""
                for tube in tubes:
                    if tube.blocks(xy, inflate):
                        blocked_by = tube.name
                        break
                if blocked_by:
                    crowd_blocked += 1
                    blockers[blocked_by] = blockers.get(blocked_by, 0) + 1
                    continue
                free[ix, iy] = True
        return free, clearance, static_blocked, crowd_blocked, blockers

    def _astar(self, free, clearance, start, goal) -> tuple[list, int]:
        """8-connected A* with a soft comfort cost.  Returns cells and expansions."""
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

        def heuristic(cell) -> float:
            return self.grid_m * math.hypot(cell[0] - goal[0], cell[1] - goal[1])

        def comfort(cell) -> float:
            gap = float(clearance[cell[0], cell[1]])
            if gap >= COMFORT_M:
                return 0.0
            return COMFORT_WEIGHT * (COMFORT_M - gap)

        open_heap = [(heuristic(start), 0.0, start)]
        best_cost = {start: 0.0}
        came_from: dict = {}
        expanded = 0
        while open_heap:
            _, cost, cell = heapq.heappop(open_heap)
            if cost > best_cost.get(cell, float("inf")) + 1e-12:
                continue
            expanded += 1
            if cell == goal:
                path = [cell]
                while path[-1] in came_from:
                    path.append(came_from[path[-1]])
                return list(reversed(path)), expanded
            for dx, dy in moves:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not (0 <= nxt[0] < self.nx and 0 <= nxt[1] < self.ny):
                    continue
                if not free[nxt[0], nxt[1]]:
                    continue
                # A diagonal move may not cut a corner between two blocked cells.
                if dx and dy and not (free[cell[0] + dx, cell[1]]
                                      and free[cell[0], cell[1] + dy]):
                    continue
                step = self.grid_m * math.hypot(dx, dy)
                new_cost = cost + step + comfort(nxt) * step
                if new_cost < best_cost.get(nxt, float("inf")) - 1e-12:
                    best_cost[nxt] = new_cost
                    came_from[nxt] = cell
                    heapq.heappush(
                        open_heap, (new_cost + heuristic(nxt), new_cost, nxt))
        return [], expanded

    # -- simplification ----------------------------------------------------
    def _walkable(self, a, b, tubes: list[CrowdTube],
                  crowd_inflate: float | None = None) -> bool:
        """Is the straight segment ``a→b`` clear of everything the search used?

        Sampled at half the grid resolution, which is finer than the search
        itself, so a shortcut can never pass through something the grid marked
        blocked.
        """
        inflate = self.crowd_inflate if crowd_inflate is None else crowd_inflate
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        steps = max(2, int(float(np.linalg.norm(b - a)) / (0.5 * self.grid_m)))
        for index in range(steps + 1):
            point = a + (b - a) * (index / steps)
            if not self.static_free(point):
                return False
            for tube in tubes:
                if tube.blocks(point, inflate):
                    return False
        return True

    def _simplify(self, points: list[np.ndarray],
                  tubes: list[CrowdTube],
                  crowd_inflate: float | None = None) -> list[np.ndarray]:
        """Greedy line-of-sight shortcutting.  Grid staircases become corners."""
        if len(points) <= 2:
            return list(points)
        out = [points[0]]
        index = 0
        while index < len(points) - 1:
            furthest = index + 1
            for candidate in range(len(points) - 1, index, -1):
                if self._walkable(points[index], points[candidate], tubes,
                                  crowd_inflate):
                    furthest = candidate
                    break
            out.append(points[furthest])
            index = furthest
        return out

    # -- public ------------------------------------------------------------
    def plan(self, start_xy, destination: Destination,
             tubes: list[CrowdTube], *,
             radii: tuple[float, ...] = CORNER_RADII,
             tiers: tuple[float, ...] = CROWD_MARGIN_TIERS) -> Plan:
        """Search a route from ``start_xy`` to ``destination``'s standing point.

        Tries each crowd-margin tier in turn and returns the first route it
        finds, reporting which tier that was.  The STATIC inflation is never
        relaxed: see :data:`CROWD_MARGIN_TIERS` for why the two are different.
        """
        last_error = None
        for tier_index, tier in enumerate(tiers):
            inflate = max(CROWD_INFLATE_FLOOR_M, self.crowd_inflate * tier)
            try:
                return self._plan_at(start_xy, destination, tubes, radii,
                                     inflate, tier, tier_index + 1)
            except RuntimeError as error:
                last_error = error
        raise RuntimeError(
            f"no route to {destination.key} from "
            f"{np.asarray(start_xy).round(3).tolist()} at any crowd margin tier "
            f"{tiers}: {last_error}")

    def _plan_at(self, start_xy, destination: Destination,
                 tubes: list[CrowdTube], radii: tuple[float, ...],
                 crowd_inflate: float, tier: float, tiers_tried: int) -> Plan:
        """One search at one crowd margin."""
        free, clearance, static_blocked, crowd_blocked, blockers = \
            self._build_masks(tubes, crowd_inflate)

        start = self.nearest_cell(start_xy)
        goal = self.nearest_cell(destination.stand)
        # The duck's own cell, and the destination's, may be inside a crowd tube
        # or a comfort margin at plan time.  Opening exactly those two cells is
        # honest — the duck IS there, and the destination IS where it is — and
        # opening anything else would let the search tunnel.
        free[start[0], start[1]] = True
        free[goal[0], goal[1]] = True

        cells, expanded = self._astar(free, clearance, start, goal)
        if not cells:
            raise RuntimeError(
                f"no route to {destination.key} from "
                f"{np.asarray(start_xy).round(3).tolist()}: the concourse is "
                f"sealed for a body of radius {self.static_inflate:.3f} m at "
                f"crowd inflation {crowd_inflate:.3f} m")

        raw = [self.cell_xy(*cell) for cell in cells]
        raw[0] = np.asarray(start_xy, dtype=np.float64).copy()
        raw[-1] = destination.stand.copy()
        waypoints = self._simplify(raw, tubes, crowd_inflate)

        # THE ARRIVAL HEADING IS BUILT INTO THE ROUTE, NOT TURNED INTO.
        # This model's turn-in-place is MEASURED at 1.6 deg/s maximum, so the
        # duck cannot pivot to face the fixture once it has stopped.  The final
        # approach is therefore rebuilt to come at the standing point from
        # DIRECTLY OPPOSITE the fixture, which means the duck's own walking
        # heading as it arrives already points at what it led her to.  The
        # inserted point is validated like any other, so a hall that has no room
        # for the approach fails loudly instead of arriving facing away.
        approach = self._approach_point(destination, crowd_inflate, tubes)
        if approach is not None:
            waypoints = self._insert_approach(waypoints, approach)

        corners = tuple((float(w[0]), float(w[1])) for w in waypoints)

        # THE FILLET IS CHOSEN, NOT ASSUMED.  Minimum clearance is measured
        # along the FILLETED route rather than along the polyline, because the
        # fillet cuts corners and can therefore come nearer a body than any
        # vertex does — grading the polyline would miss exactly that.  The
        # largest radius whose filleted route clears ROUTE_CLEARANCE_M wins.
        chosen = None
        tried = 0
        for radius in radii:
            tried += 1
            candidate = Route(f"guide_{destination.key.lower()}", corners,
                              speed=1.0, radius=radius)
            worst, worst_at = self._route_clearance(candidate)
            chosen = (candidate, radius, worst, worst_at)
            if worst >= ROUTE_CLEARANCE_M:
                break
        route, radius, worst, worst_at = chosen
        if worst < ROUTE_CLEARANCE_M:
            raise RuntimeError(
                f"no corner radius in {radii} keeps the route to "
                f"{destination.key} clear of the concourse: best is "
                f"{worst:.4f} m at {worst_at}, needs {ROUTE_CLEARANCE_M:.4f} m")

        start_point = np.asarray(start_xy, dtype=np.float64)
        straight = float(np.linalg.norm(destination.stand - start_point))
        blocked_by = ""
        for obstacle in OBSTACLES:
            if obstacle.segment_hits(start_point, destination.stand,
                                     self.static_inflate):
                blocked_by = obstacle.name
                break

        return Plan(
            destination_key=destination.key,
            waypoints=waypoints,
            route=route,
            grid_cells=self.nx * self.ny,
            free_cells=int(free.sum()),
            static_blocked_cells=static_blocked,
            crowd_blocked_cells=crowd_blocked,
            crowd_blockers=blockers,
            tubes=list(tubes),
            expanded=expanded,
            raw_vertices=len(raw),
            min_clearance_m=worst,
            min_clearance_at=worst_at,
            straight_line_m=straight,
            detour_ratio=(float(route.length) / straight) if straight else 0.0,
            straight_blocked_by=blocked_by,
            corner_radius_m=radius,
            radii_tried=tried,
            crowd_inflate_used_m=crowd_inflate,
            crowd_tier_used=tier,
            crowd_tiers_tried=tiers_tried,
        )

    @staticmethod
    def _route_clearance(route: Route) -> tuple[float, tuple[float, float]]:
        """Worst static clearance along a filleted route, and where it occurs.

        Sampled at 1 cm or finer: 320 samples over a route no longer than the
        12 m the concourse can hold.
        """
        worst, worst_at = float("inf"), (0.0, 0.0)
        for index in range(321):
            point = route.pose_at_arc(route.length * index / 320.0)[0]
            gap, _ = _clearance(point)
            if gap < worst:
                worst, worst_at = gap, (float(point[0]), float(point[1]))
        return worst, worst_at

    def _approach_point(self, destination: Destination, crowd_inflate: float,
                        tubes: list[CrowdTube]) -> np.ndarray | None:
        """A point behind the standing point, on the fixture's own axis.

        Walking from here to the standing point aims the duck straight at the
        fixture, which is how arrival facing is achieved on a robot that cannot
        turn in place.  Returns ``None`` if no offset in the sweep is both free
        and walkable, so a cramped destination fails the facing gate honestly
        rather than silently skipping the approach.
        """
        stand = destination.stand
        axis = stand - destination.position
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return None
        axis = axis / norm
        for offset in (APPROACH_OFFSET_M, 0.62, 0.52, 0.44):
            candidate = stand + axis * offset
            if not self.static_free(candidate):
                continue
            if not self._walkable(candidate, stand, tubes, crowd_inflate):
                continue
            return candidate
        return None

    @staticmethod
    def _insert_approach(waypoints: list[np.ndarray],
                         approach: np.ndarray) -> list[np.ndarray]:
        """Put ``approach`` immediately before the standing point.

        Any earlier waypoint that the approach point has made redundant — one
        closer to the end than the approach itself — is dropped, so the final
        leg is a single straight run at the fixture rather than a wiggle.
        """
        stand = waypoints[-1]
        keep = [w for w in waypoints[:-1]
                if float(np.linalg.norm(w - stand))
                > float(np.linalg.norm(approach - stand)) + 1e-6]
        if not keep:
            keep = [waypoints[0]]
        return keep + [approach.copy(), stand.copy()]


def tubes_from_states(people, exclude: str) -> list[CrowdTube]:
    """Build swept planning tubes from a people snapshot, excluding the follower.

    Uses each person's CURRENT position and velocity only.  Nothing here reads a
    route, a waypoint list or a schedule, which is what keeps the plan a
    consequence of measurement rather than a lookup.
    """
    return [
        CrowdTube(name, np.asarray(state.pos, dtype=np.float64).copy(),
                  np.asarray(state.velocity, dtype=np.float64).copy())
        for name, state in people.items() if name != exclude
    ]
