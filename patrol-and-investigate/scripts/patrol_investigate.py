#!/usr/bin/env python3
"""Planning the investigation: where to stand to look at something, without
touching it.

ONE JOB: TURN A TARGET INTO A PLACE TO STAND
----------------------------------------------
The duck may not approach an anomaly to arbitrary range.  It must stop inside a
SAFE OBSERVATION STANDOFF - :data:`STANDOFF_MIN_M` to :data:`STANDOFF_MAX_M`
from the target's surface - close enough for the camera to resolve it, far
enough that a person is not crowded and nothing is ever touched.  This module
chooses the point that satisfies that band, and it is the only place a standoff
point is ever produced.

THE CANDIDATES ARE BEARINGS, BECAUSE THE DUCK CANNOT ORBIT
------------------------------------------------------------
Turning in place is MEASURED to be unavailable and a full orbit at 0.128 m/s
would be twenty seconds of walking per angle, so the standoff is a SINGLE point
and the multiple viewing angles come from the head.  The point is therefore
chosen once, from several candidate bearings round the target, and scored on
three things the duck can measure:

* it must be REACHABLE - the straight line from the duck to it must not pass
  through a fixture, and it must not sit inside one;
* it must be OUTSIDE the restricted zone by :data:`ZONE_STANDOFF_M`, because
  approaching an intruder is not a licence to enter the area they are in;
* among the survivors, the one nearest the duck's current position wins, so the
  approach is the short way round rather than a walk to the far side.

THE ZONE RULE IS THE INTERESTING ONE.  The intrusion the duck has to investigate
happens INSIDE a rectangle the duck itself must stay out of, so the standoff for
that target is necessarily on the boundary looking in.  A planner that ignored
the zone would produce a perfectly good observation point that the robot was not
allowed to occupy, and the gate would then have to catch it after the fact.
Pruning it here means the constraint is respected by construction, and the gate
still measures the trunk's real distance every tick as an independent check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from patrol_cast import BY_NAME, planning_radius
from patrol_facility import (
    FIXTURES,
    FLOOR_HALF,
    RESTRICTED_ZONE,
    ZONE_STANDOFF_M,
    static_gap,
)
from patrol_states import (
    DUCK_PLANAR_RADIUS,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
    STANDOFF_TARGET_M,
)

# Candidate bearings round the target, in degrees relative to the direction from
# the target back to the duck.  0 deg is straight back the way the duck came,
# which is the shortest approach; the others let the planner step round a
# fixture or off the edge of the restricted zone without walking all the way
# round.
CANDIDATE_BEARINGS_DEG: tuple[float, ...] = (
    0.0, -22.0, 22.0, -45.0, 45.0, -70.0, 70.0, -100.0, 100.0, 180.0)
# How far a standoff point must clear any static surface.  DERIVED from the
# duck's own conservative planar radius plus a margin, so a point the planner
# offers is one the robot fits at.
STANDOFF_FIXTURE_MARGIN_M = DUCK_PLANAR_RADIUS + 0.10
# How far a standoff point must stay inside the perimeter walls.
STANDOFF_WALL_MARGIN_M = DUCK_PLANAR_RADIUS + 0.12
# Resolution of the reachability check along the approach line.
PATH_SAMPLES = 40


@dataclass
class Standoff:
    """One candidate observation point, scored."""

    target: str
    bearing_deg: float
    xy: tuple[float, float]
    standoff_m: float
    walk_m: float
    ok: bool
    reason: str = ""
    fixture_gap_m: float = 0.0
    zone_gap_m: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.xy, dtype=np.float64)

    def as_record(self) -> dict:
        return {
            "bearing_deg": round(float(self.bearing_deg), 2),
            "xy": [round(float(self.xy[0]), 4), round(float(self.xy[1]), 4)],
            "standoff_m": round(float(self.standoff_m), 4),
            "walk_m": round(float(self.walk_m), 4),
            "ok": bool(self.ok),
            "reason": self.reason,
            "fixture_gap_m": round(float(self.fixture_gap_m), 4),
            "zone_gap_m": round(float(self.zone_gap_m), 4),
        }


@dataclass
class InvestigationPlan:
    """The chosen standoff point and every candidate that was rejected.

    Both are kept, because a plan that recorded only its answer could not
    distinguish a decision from a default.  The HUD draws the rejected
    candidates beside the chosen one for exactly that reason.
    """

    target: str
    chosen: Standoff | None
    candidates: list[Standoff] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.chosen is not None

    @property
    def standoff_xy(self) -> np.ndarray | None:
        return None if self.chosen is None else self.chosen.position

    def as_record(self) -> dict:
        return {
            "target": self.target,
            "chosen": None if self.chosen is None else self.chosen.as_record(),
            "candidates": [c.as_record() for c in self.candidates],
            "rejected": [c.as_record() for c in self.candidates if not c.ok],
            "band_m": [STANDOFF_MIN_M, STANDOFF_MAX_M],
        }


def standoff_from_range(target: str, centre_range_m: float) -> float:
    """Convert a centre-to-centre range into a SURFACE standoff.

    The band is a surface clearance, so the target's planning radius and the
    duck's own are subtracted here once rather than at every call site.  Both
    are PLANNING figures and deliberately generous; the gate measures the real
    geoms with ``ContactProbe`` every tick and is what any safety claim rests
    on.  Over-stating both bodies makes the planned standoff slightly WIDER
    than the measured one, which is the safe direction.
    """
    return float(centre_range_m - planning_radius(target)
                 - DUCK_PLANAR_RADIUS)


def range_for_standoff(target: str, standoff_m: float) -> float:
    """The centre-to-centre range that produces a given surface standoff."""
    return float(standoff_m + planning_radius(target) + DUCK_PLANAR_RADIUS)


def _blocked_by_fixture(a, b) -> str:
    """Name of the first fixture the approach line passes through."""
    start = np.asarray(a, dtype=np.float64)[:2]
    end = np.asarray(b, dtype=np.float64)[:2]
    for fixture in FIXTURES:
        if fixture.segment_hits(start, end, DUCK_PLANAR_RADIUS,
                                samples=PATH_SAMPLES):
            return fixture.name
    return ""


def plan_standoff(target: str, target_xy, duck_xy) -> InvestigationPlan:
    """Choose where to stand to observe ``target`` safely.

    Every candidate is scored and kept.  The chosen one is the reachable,
    zone-legal candidate closest to the duck, which is the short way round.
    """
    target_point = np.asarray(target_xy, dtype=np.float64)[:2]
    duck = np.asarray(duck_xy, dtype=np.float64)[:2]

    back = duck - target_point
    norm = float(np.linalg.norm(back))
    base = (math.atan2(float(back[1]), float(back[0])) if norm > 1e-9 else 0.0)
    radius = range_for_standoff(target, STANDOFF_TARGET_M)

    candidates: list[Standoff] = []
    for bearing in CANDIDATE_BEARINGS_DEG:
        angle = base + math.radians(bearing)
        point = target_point + radius * np.array([math.cos(angle),
                                                  math.sin(angle)])
        walk = float(np.linalg.norm(point - duck))
        _, fixture_gap = static_gap(point)
        # Signed distance from the zone's edge: positive outside.
        zone_gap = -RESTRICTED_ZONE.depth_inside(point)

        entry = Standoff(
            target=target, bearing_deg=bearing,
            xy=(float(point[0]), float(point[1])),
            standoff_m=STANDOFF_TARGET_M, walk_m=walk, ok=True,
            fixture_gap_m=float(fixture_gap), zone_gap_m=float(zone_gap))

        if abs(float(point[0])) > FLOOR_HALF[0] - STANDOFF_WALL_MARGIN_M \
                or abs(float(point[1])) > FLOOR_HALF[1] - STANDOFF_WALL_MARGIN_M:
            entry.ok, entry.reason = False, "outside the facility walls"
        elif fixture_gap < STANDOFF_FIXTURE_MARGIN_M:
            entry.ok = False
            entry.reason = (f"only {fixture_gap:.3f} m from the nearest "
                            f"fixture, below the "
                            f"{STANDOFF_FIXTURE_MARGIN_M:.3f} m the duck fits "
                            "in")
        elif zone_gap < ZONE_STANDOFF_M:
            entry.ok = False
            entry.reason = (f"inside the restricted zone by "
                            f"{-zone_gap:.3f} m; the duck may not enter it "
                            "even to investigate")
        else:
            blocker = _blocked_by_fixture(duck, point)
            if blocker:
                entry.ok = False
                entry.reason = f"the approach line passes through {blocker}"
        candidates.append(entry)

    viable = [c for c in candidates if c.ok]
    chosen = min(viable, key=lambda c: c.walk_m) if viable else None
    return InvestigationPlan(target=target, chosen=chosen,
                             candidates=candidates)


def observation_look_point(target: str, target_xy, angle_deg: float,
                           duck_xy) -> np.ndarray:
    """The world point the head aims at for one observation angle.

    The angles are swept ACROSS the target rather than round it, because the
    duck is standing still: the head yaws off the target's centre by
    ``angle_deg``, sampling its left edge, its middle and its right edge.  That
    is what a pan-tilt camera does from a fixed standoff, and it is why the
    observation is called multi-angle rather than multi-position.
    """
    target_point = np.asarray(target_xy, dtype=np.float64)[:2]
    duck = np.asarray(duck_xy, dtype=np.float64)[:2]
    span = target_point - duck
    distance = float(np.linalg.norm(span))
    if distance < 1e-9:
        bearing = 0.0
    else:
        bearing = math.atan2(float(span[1]), float(span[0]))
    angle = bearing + math.radians(angle_deg)
    point = duck + distance * np.array([math.cos(angle), math.sin(angle)])
    spec = BY_NAME[target]
    height = (spec.origin_z + 0.12 * spec.stature if spec.is_person
              else 0.5 * _object_height(target))
    return np.array([float(point[0]), float(point[1]), height])


def _object_height(name: str) -> float:
    from patrol_cast import OBJECT_HEIGHT_M
    return OBJECT_HEIGHT_M[BY_NAME[name].kind]


def target_look_point(target: str, target_xy) -> np.ndarray:
    """The world point at the target's own centre, at a sensible height."""
    point = np.asarray(target_xy, dtype=np.float64)[:2]
    spec = BY_NAME[target]
    height = (spec.origin_z + 0.12 * spec.stature if spec.is_person
              else 0.5 * _object_height(target))
    return np.array([float(point[0]), float(point[1]), height])
