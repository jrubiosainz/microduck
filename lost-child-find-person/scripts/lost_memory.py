#!/usr/bin/env python3
"""Last-known footprint, the world-space trail, and the rejoin route planner.

Two responsibilities, kept in one module because the second consumes the first.

THE TRAIL IS WORLD-SPACE AND IT SURVIVES THE LOSS
--------------------------------------------------
While the guardian is visible, the duck records where she was, in world
coordinates, at a fixed spatial interval.  When she disappears, that trail is
what remains: the last confirmed footprint plus the path that led to it.  It is
NOT derived from her live pose — the whole point is that the duck no longer has
her live pose — so the trail is retained across the entire loss and is what the
search and the rejoin are anchored to.

Recording at a fixed spatial interval rather than every tick matters: a
time-sampled trail bunches up wherever the guardian slowed down and says nothing
about where she went, while a distance-sampled one is an even record of her
route.

THE ROUTE IS PLANNED AROUND REAL GEOMETRY
------------------------------------------
The rejoin is not "walk at the target".  A straight line from the duck to the
reacquired guardian runs through the kiosk, which is exactly the body that hid
her in the first place.  :func:`plan_route` builds a short visibility graph over
the INFLATED corners of the real obstacles and searches it, so the route it
returns bends round the kiosk the way a person would.

The graph is deliberately small — obstacle corners plus the two endpoints — for
two reasons.  It is enough: with six convex obstacles in an open hall, the
shortest obstacle-free polyline between two points bends only at obstacle
corners, so the corner set contains an optimal path whenever one exists.  And it
is auditable: every waypoint is a named point on a named obstacle, so a route can
be explained rather than merely trusted.

CROWD AVOIDANCE IS TIME-AWARE, AND THAT IS THE HARD PART
---------------------------------------------------------
The people are moving, so "is this segment clear of people" has no answer
without a time.  :func:`plan_route` therefore takes a predicted arrival time for
each candidate waypoint, computed from the measured walking speed, and tests the
crowd at the time the duck would actually be there.  Treating the crowd as a
static snapshot is the error this avoids: it would route the duck through the
space somebody is about to occupy while carefully avoiding where they were.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from lost_geometry import PERSON_CLEARANCE_M, PLAN_INFLATE_M
from plaza_layout import FLOOR_HALF, OBSTACLES, occluder_between

# Spatial interval at which the guardian's footprints are recorded.
TRAIL_STEP_M = 0.22
# How many footprints are kept.  At 0.22 m that is about 3.1 m of history, which
# spans the whole approach to an occluder and well beyond it.
TRAIL_MAX = 14


@dataclass
class GuardianTrail:
    """World-space record of where the guardian was, while she could be seen."""

    points: list[np.ndarray] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    last_seen_xy: np.ndarray | None = None
    last_seen_t: float | None = None
    last_seen_heading: float | None = None

    def observe(self, t: float, xy) -> None:
        """Record a CONFIRMED sighting.  Only ever called when the camera sees her."""
        point = np.asarray(xy, dtype=np.float64).copy()
        if not self.points or float(
                np.linalg.norm(point - self.points[-1])) >= TRAIL_STEP_M:
            self.points.append(point)
            self.times.append(t)
            if len(self.points) > TRAIL_MAX:
                self.points.pop(0)
                self.times.pop(0)
        if self.last_seen_xy is not None:
            delta = point - self.last_seen_xy
            if float(np.linalg.norm(delta)) > 1e-3:
                self.last_seen_heading = math.atan2(
                    float(delta[1]), float(delta[0]))
        self.last_seen_xy = point
        self.last_seen_t = t

    def age(self, t: float) -> float:
        return float("inf") if self.last_seen_t is None else t - self.last_seen_t

    def extrapolated(self, t: float, speed: float = 0.175,
                     max_s: float = 4.0) -> np.ndarray | None:
        """Where she might be if she kept going, capped and clearly labelled.

        Used ONLY to bias the search sweep's centre and to seed the first rejoin
        waypoint — never as a target the duck walks to, and never as evidence
        that she is there.  Capped at ``max_s`` because a linear extrapolation
        of a walking person stops meaning anything after a few seconds.
        """
        if self.last_seen_xy is None or self.last_seen_heading is None:
            return None
        elapsed = min(self.age(t), max_s)
        return self.last_seen_xy + speed * elapsed * np.array(
            [math.cos(self.last_seen_heading), math.sin(self.last_seen_heading)])

    def as_record(self) -> dict:
        return {
            "points": [[round(float(p[0]), 4), round(float(p[1]), 4)]
                       for p in self.points],
            "times": [round(float(t), 3) for t in self.times],
            "last_seen_xy": (None if self.last_seen_xy is None else
                             [round(float(self.last_seen_xy[0]), 4),
                              round(float(self.last_seen_xy[1]), 4)]),
            "last_seen_t": (None if self.last_seen_t is None
                            else round(float(self.last_seen_t), 3)),
            "last_seen_heading_deg": (
                None if self.last_seen_heading is None
                else round(math.degrees(self.last_seen_heading), 2)),
            "length_m": self.length_m(),
        }

    def length_m(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(sum(
            np.linalg.norm(b - a) for a, b in zip(self.points, self.points[1:])))


# -- route planning ---------------------------------------------------------
def _people_clear(a, b, people_xy: dict, clearance: float,
                  samples: int = 40) -> str:
    """Name of the first person a segment passes too close to, or ``""``.

    ``people_xy`` maps a name to the position that person is PREDICTED to
    occupy while the duck traverses this segment; the caller is responsible for
    evaluating it at the right time.
    """
    start = np.asarray(a, dtype=np.float64)
    end = np.asarray(b, dtype=np.float64)
    for name, xy in people_xy.items():
        point = np.asarray(xy, dtype=np.float64)
        for index in range(samples + 1):
            sample = start + (end - start) * (index / samples)
            if float(np.linalg.norm(sample - point)) < clearance:
                return name
    return ""


def segment_blocked(a, b, people_xy: dict | None = None,
                    inflate: float = PLAN_INFLATE_M,
                    person_clearance: float = PERSON_CLEARANCE_M) -> str:
    """Why a straight segment is unusable: an obstacle name, a person, or ``""``."""
    for obstacle in OBSTACLES:
        if obstacle.segment_hits(a, b, inflate):
            return obstacle.name
    if people_xy:
        who = _people_clear(a, b, people_xy, person_clearance)
        if who:
            return f"person:{who}"
    return ""


def _in_hall(xy, margin: float) -> bool:
    return (abs(float(xy[0])) <= FLOOR_HALF[0] - margin
            and abs(float(xy[1])) <= FLOOR_HALF[1] - margin)


def waypoint_candidates(inflate: float = PLAN_INFLATE_M) -> list[np.ndarray]:
    """Inflated corners of every obstacle that lie inside the hall."""
    points: list[np.ndarray] = []
    for obstacle in OBSTACLES:
        for corner in obstacle.corners(inflate):
            if not _in_hall(corner, 0.12):
                continue
            if any(o.distance_to(corner) < inflate - 1e-6 for o in OBSTACLES):
                continue
            points.append(corner)
    return points


@dataclass(frozen=True)
class Route:
    """A planned rejoin path: waypoints, why it bends, and what it avoids."""

    waypoints: tuple[np.ndarray, ...]
    length_m: float
    direct_blocked_by: str
    bends_around: tuple[str, ...]
    feasible: bool

    def as_record(self) -> dict:
        return {
            "waypoints": [[round(float(p[0]), 4), round(float(p[1]), 4)]
                          for p in self.waypoints],
            "length_m": round(self.length_m, 4),
            "direct_blocked_by": self.direct_blocked_by,
            "bends_around": list(self.bends_around),
            "feasible": self.feasible,
            "waypoint_count": len(self.waypoints),
        }


def _attribution(waypoint, inflate: float) -> str:
    """Which obstacle a waypoint belongs to, for explaining a route."""
    best, best_gap = "", float("inf")
    for obstacle in OBSTACLES:
        gap = abs(obstacle.distance_to(waypoint) - inflate)
        if gap < best_gap:
            best, best_gap = obstacle.name, gap
    return best if best_gap < 0.05 else ""


def plan_route(start, goal, *, people_at_time=None, speed: float = 0.209,
               t0: float = 0.0, inflate: float = PLAN_INFLATE_M) -> Route:
    """Shortest obstacle-free polyline from ``start`` to ``goal``.

    ``people_at_time(t)`` returns a ``{name: xy}`` mapping at time ``t``; the
    crowd is tested at the moment the duck would actually reach each segment,
    computed from the measured walking speed.  Passing ``None`` plans against
    the static scenery only, which is what the obstacle-avoidance unit tests do.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    def crowd_at(elapsed: float) -> dict | None:
        return None if people_at_time is None else people_at_time(t0 + elapsed)

    direct_block = segment_blocked(start, goal, crowd_at(
        float(np.linalg.norm(goal - start)) / max(speed, 1e-6) * 0.5))
    if not direct_block:
        length = float(np.linalg.norm(goal - start))
        return Route((start, goal), length, "", (), True)

    nodes = [start, goal] + waypoint_candidates(inflate)
    count = len(nodes)
    # Dijkstra over the visibility graph, with edge feasibility evaluated at the
    # time the duck would traverse that edge.
    best_cost = [float("inf")] * count
    best_cost[0] = 0.0
    previous: list[int | None] = [None] * count
    queue: list[tuple[float, int]] = [(0.0, 0)]
    visited = [False] * count
    while queue:
        cost, index = heapq.heappop(queue)
        if visited[index]:
            continue
        visited[index] = True
        if index == 1:
            break
        for other in range(count):
            if other == index or visited[other]:
                continue
            step = float(np.linalg.norm(nodes[other] - nodes[index]))
            if step < 1e-9:
                continue
            arrival = cost / max(speed, 1e-6)
            if segment_blocked(nodes[index], nodes[other],
                               crowd_at(arrival + 0.5 * step / max(speed, 1e-6)),
                               inflate):
                continue
            if cost + step < best_cost[other]:
                best_cost[other] = cost + step
                previous[other] = index
                heapq.heappush(queue, (cost + step, other))

    if math.isinf(best_cost[1]):
        return Route((start, goal), float(np.linalg.norm(goal - start)),
                     direct_block, (), False)

    path: list[int] = []
    cursor: int | None = 1
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    waypoints = tuple(nodes[i] for i in path)
    bends = tuple(
        name for name in
        (_attribution(nodes[i], inflate) for i in path[1:-1]) if name)
    return Route(waypoints, best_cost[1], direct_block, bends, True)


def route_progress(route: Route, duck_xy) -> tuple[int, float]:
    """Index of the waypoint the duck should aim at, and the range to the goal."""
    duck = np.asarray(duck_xy, dtype=np.float64)
    remaining = float(np.linalg.norm(route.waypoints[-1] - duck))
    for index in range(1, len(route.waypoints)):
        if float(np.linalg.norm(route.waypoints[index] - duck)) > 0.16:
            return index, remaining
    return len(route.waypoints) - 1, remaining


def line_of_sight_available(eye_xy, target_xy) -> tuple[bool, str]:
    """Would a static occluder block this sightline?  Planar, cheap, advisory.

    Used only to decide whether a missing sighting is EXPECTED, so tracking
    during the rejoin can be graded on the steps where seeing was possible.  The
    authoritative measurement is always the MuJoCo ray cast in ``lost_camera``.
    """
    blocker = occluder_between(eye_xy, target_xy)
    return (blocker is None), (blocker or "")
