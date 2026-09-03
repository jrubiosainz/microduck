#!/usr/bin/env python3
"""Ward-relative stations and safety geometry for personal-space behavior."""
from __future__ import annotations
import math
import numpy as np
from pps_cast import PLANNING_HALF_EXTENT_M
from pps_plaza import FIXTURES, clamp_inside
from pps_states import (DUCK_PLANAR_RADIUS, ESCORT_BEHIND_M, ESCORT_LATERAL_M,
                        ESCAPE_MIN_CLEARANCE_M, ESCAPE_RADIUS_M,
                        INTERPOSE_BEARING_TOL_DEG, INTERPOSE_FROM_PERSON_M)


def axes(yaw: float):
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-forward[1], forward[0]])
    return forward, left


def escort_point(ward_xy, ward_yaw: float) -> np.ndarray:
    forward, left = axes(ward_yaw)
    return np.asarray(ward_xy) - ESCORT_BEHIND_M * forward - ESCORT_LATERAL_M * left


def interpose_point(ward_xy, threat_xy) -> np.ndarray:
    ward, threat = np.asarray(ward_xy), np.asarray(threat_xy)
    delta = threat - ward
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        delta, norm = np.array([1.0, 0.0]), 1.0
    return clamp_inside(ward + INTERPOSE_FROM_PERSON_M * delta / norm,
                        DUCK_PLANAR_RADIUS + 0.08)


def bearing_deg(origin, target) -> float:
    d = np.asarray(target) - np.asarray(origin)
    return math.degrees(math.atan2(float(d[1]), float(d[0])))


def angle_delta_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def is_between(ward_xy, duck_xy, threat_xy) -> bool:
    bearing = bearing_deg(ward_xy, threat_xy)
    duck_bearing = bearing_deg(ward_xy, duck_xy)
    ward_range = float(np.linalg.norm(np.asarray(duck_xy)-np.asarray(ward_xy)))
    threat_range = float(np.linalg.norm(np.asarray(threat_xy)-np.asarray(ward_xy)))
    return (abs(angle_delta_deg(duck_bearing, bearing)) <= INTERPOSE_BEARING_TOL_DEG
            and 0.25 < ward_range < threat_range)


def surface_gap(a, b, half_a=DUCK_PLANAR_RADIUS,
                half_b=PLANNING_HALF_EXTENT_M) -> float:
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))) - half_a - half_b


def static_clearance(xy) -> float:
    return min(f.distance_to(xy) - DUCK_PLANAR_RADIUS for f in FIXTURES)


def escape_point(ward_xy, threat_positions: list[np.ndarray], people_xy: dict,
                 samples: int = 24, start=None) -> tuple[np.ndarray, float, list[dict]]:
    ward = np.asarray(ward_xy)
    candidates = []
    for index, angle in enumerate(np.linspace(0, 2*math.pi, samples, endpoint=False)):
        point = clamp_inside(ward + ESCAPE_RADIUS_M*np.array([math.cos(angle), math.sin(angle)]),
                             DUCK_PLANAR_RADIUS + 0.10)
        static = static_clearance(point)
        people = min((surface_gap(point, p) for p in people_xy.values()), default=9.0)
        threats = min((float(np.linalg.norm(point-p)) for p in threat_positions), default=9.0)
        route_gap=9.0
        if start is not None:
            for u in np.linspace(0.0,1.0,17):
                sample=np.asarray(start)+(point-np.asarray(start))*u
                route_gap=min(route_gap,static_clearance(sample),
                              min((surface_gap(sample,p) for p in people_xy.values()),default=9.0))
        score = min(static, people, threats, route_gap)
        candidates.append({"index": index, "point": point, "score": score,
                           "static": static, "people": people,
                           "threats": threats, "route_gap": route_gap})
    best = max(candidates, key=lambda c: c["score"])
    return best["point"], float(best["score"]), candidates


def route_around_ward(start, ward_xy, goal, clearance: float = 0.78,
                      heading: float | None = None) -> list[np.ndarray]:
    """Add a lateral waypoint when the direct segment would cross the ward.

    Interposing on the opposite bearing must go AROUND the protected person,
    never through her. The chosen tangent-side waypoint is the one nearer the
    duck's current side and the cursor consuming it is monotonic.
    """
    start, ward, goal = map(lambda x: np.asarray(x, dtype=np.float64),
                            (start, ward_xy, goal))
    segment = goal-start
    denom = float(segment@segment)
    u = 0.0 if denom < 1e-9 else float(np.clip((ward-start)@segment/denom,0,1))
    nearest = start+u*segment
    if float(np.linalg.norm(nearest-ward)) >= clearance:
        return [goal]
    # Follow an arc around the ward instead of a single lateral waypoint: the
    # chord between two safe points can still cut through the protected disc.
    radius = max(clearance, 0.95)
    route: list[np.ndarray] = []
    radial = start-ward
    radial /= max(float(np.linalg.norm(radial)), 1e-9)
    tangents = [np.array([-radial[1], radial[0]]),
                np.array([radial[1], -radial[0]])]
    if heading is not None:
        forward = np.array([math.cos(heading), math.sin(heading)])
        tangent_dir = max(tangents, key=lambda v: float(v@forward))
        tangent = start + 0.65*tangent_dir
        route.append(clamp_inside(tangent, DUCK_PLANAR_RADIUS+.10))
        arc_start = tangent
    else:
        arc_start = start
    start_angle = math.atan2(float(arc_start[1]-ward[1]), float(arc_start[0]-ward[0]))
    goal_angle = math.atan2(float(goal[1]-ward[1]), float(goal[0]-ward[0]))
    delta = (goal_angle-start_angle+math.pi)%(2*math.pi)-math.pi
    count = max(2, int(math.ceil(abs(delta)/math.radians(35))))
    points = [ward+radius*np.array([math.cos(start_angle+delta*i/count),
                                    math.sin(start_angle+delta*i/count)])
              for i in range(1,count+1)]
    return route + [clamp_inside(p, DUCK_PLANAR_RADIUS+.10) for p in points] + [goal]


def projected_along(start, end, heading: float) -> float:
    forward, _ = axes(heading)
    return float((np.asarray(end)-np.asarray(start)) @ forward)
