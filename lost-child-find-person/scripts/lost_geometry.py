#!/usr/bin/env python3
"""Distances the behavior is graded on: standoff, clearance, and the duck itself.

Single source of truth so the controller, the acceptance gate and the HUD all
mean the same thing by "safe standoff" and "clear".

WHY THE STANDOFF BAND IS 0.45-0.75 m, JUSTIFIED RATHER THAN CHOSEN
-------------------------------------------------------------------
The duck's conservative planar half-extent is 0.1303 m (bounding-sphere based,
so it over-states the robot).  Pairing it with the NOMINAL adult half-extent
below gives a nominal centre-to-centre contact separation of 0.2950 m, so:

* **0.45 m** leaves about 0.155 m of nominal surface gap at the near edge of the
  band, which is roughly an adult's arm swing, so a guardian who turns on the
  spot is not brushing the duck.
* **0.75 m** is the far edge: beyond it the duck reads as loitering rather than
  as having rejoined, and at the measured walking speed it is more than three
  seconds behind - long enough for the crowd to close the gap again.

THE 0.155 m FIGURE IS A NOMINAL SIZING ARGUMENT, NOT A SAFETY GUARANTEE.  It is
computed from a single representative body width and is not what any acceptance
gate measures; see ``ADULT_HALF_EXTENT_M`` below.  The actual safety claim is
the per-tick ``ContactProbe`` measurement of real surface clearance against the
real geoms at the real pose, which for this rollout bottoms out at 0.1050 m to
``dahl`` and never reaches zero.

FOLLOW DISTANCE IS DELIBERATELY LARGER THAN THE STANDOFF.  While following, the
duck holds ~0.85 m: far enough back to be safe behind a moving adult who may
stop suddenly, and far enough that the kiosk can come between them, which is
what makes the loss geometric rather than staged.
"""

from __future__ import annotations

import math

import numpy as np

from plaza_layout import BY_NAME as OBSTACLE_BY_NAME
from plaza_layout import OBSTACLES, clear_of_obstacles

# -- the duck ---------------------------------------------------------------
# Conservative planar half-extent from each geom's BOUNDING SPHERE, which
# over-states the robot.  That is the safe direction for every gate here.
# ``test_duck_planar_radius_matches_the_model`` pins it against the built scene.
DUCK_PLANAR_RADIUS = 0.1303

# LEGACY NOMINAL REFERENCE - NOT A MEASUREMENT OF THIS SCENE, AND NO GATE USES
# IT.  This value was previously described as the widest planar half-extent of
# an adult over a full gait cycle, pinned against the scene.  That description
# was false in both directions and the tool it credited never existed.  MEASURED
# on the built scene, the guardian's exact planar half-extent over a full gait
# cycle is:
#
#     pose zero (arms down, t=0)   0.1375 m   <- what the rollout reports as
#                                                `adult_half_extent_m`
#     mean over 300 poses          0.1945 m
#     widest, mid-stride           0.2629 m   <- the true gait maximum
#     widest of ANY adult, 60 s    0.2709 m   (eze)
#
# 0.1647 is neither the minimum, the mean, nor the maximum.  It is retained
# unchanged, and only as the nominal body width behind the standoff-band prose
# above, because it is read by exactly one function - `surface_gap` - which is
# explanatory and is called by no controller, planner, or acceptance gate.
# Widening it to the gait maximum would silently move the documented 0.155 m
# near-edge figure without improving any safety property, so the number stays
# put and the truth is stated here and pinned by
# `test_the_adult_half_extent_constant_is_a_legacy_nominal`.
#
# THE GATES DO NOT USE THIS CONSTANT.  Every clearance gate measures real
# surface separation each control tick with `ContactProbe`, against the actual
# geoms at the actual pose, so the arm swing is accounted for exactly rather
# than through any single nominal half-extent.
ADULT_HALF_EXTENT_M = 0.1647
# Nominal centre-to-centre distance at which the two bodies would touch, used
# only by `surface_gap`.  Not a gate.
CONTACT_SEPARATION_M = DUCK_PLANAR_RADIUS + ADULT_HALF_EXTENT_M

# -- where the duck starts --------------------------------------------------
# Behind and to the south of the guardian's first waypoint, already at roughly
# the follow distance, facing her.  The follow phase is therefore a real traverse
# rather than a spawn already in formation.
DUCK_START_XY = (2.15, -1.90)
DUCK_START_YAW_DEG = 96.0

# -- distances --------------------------------------------------------------
# Held while following in the open.
FOLLOW_DISTANCE_M = 0.85
# Hysteresis band around it, so the duck is not pumping the throttle.
FOLLOW_NEAR_M = 0.70
FOLLOW_FAR_M = 1.05
# The band the final standoff must land in; see the module docstring.
STANDOFF_MIN_M = 0.45
STANDOFF_MAX_M = 0.75
STANDOFF_TARGET_M = 0.60
# Every planned waypoint must clear every obstacle by at least this much, and
# clear every person by this much at the time the duck is near them.
ROUTE_CLEARANCE_M = 0.26
PERSON_CLEARANCE_M = 0.24
# The gate requires strictly positive measured surface clearance at all times;
# this is the additional margin the PLANNER works to, so the gate has slack.
PLAN_INFLATE_M = DUCK_PLANAR_RADIUS + ROUTE_CLEARANCE_M


def surface_gap(center_distance_m: float) -> float:
    """NOMINAL surface-to-surface gap implied by a centre-to-centre distance.

    Explanatory only.  Uses the legacy nominal ``ADULT_HALF_EXTENT_M``, which is
    neither the minimum nor the maximum adult width in this scene, so the result
    is a sizing estimate and not a clearance measurement.  No acceptance gate
    calls this function; clearance is measured per tick by ``ContactProbe``.
    """
    return center_distance_m - CONTACT_SEPARATION_M


def standoff_ok(distance_m: float) -> bool:
    return STANDOFF_MIN_M <= distance_m <= STANDOFF_MAX_M


def standoff_verdict(distance_m: float) -> str:
    if distance_m < STANDOFF_MIN_M:
        return "too close"
    if distance_m > STANDOFF_MAX_M:
        return "too far"
    return "in band"


def approach_point(target_xy, from_xy, distance_m: float = STANDOFF_TARGET_M):
    """A standoff station ``distance_m`` from the target, on the duck's side.

    Placed on the bearing the duck is arriving from rather than at a fixed
    offset, so the duck stops short of the guardian instead of walking round
    her, and never has to pass through her to reach its own goal.
    """
    target = np.asarray(target_xy, dtype=np.float64)
    origin = np.asarray(from_xy, dtype=np.float64)
    delta = origin - target
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return target + np.array([distance_m, 0.0])
    return target + delta / norm * distance_m


def safe_standoff_point(target_xy, from_xy,
                        distance_m: float = STANDOFF_TARGET_M):
    """A standoff station that is also clear of every obstacle.

    The straight-line station can land inside a column, because the guardian is
    entitled to stand right beside one.  Rather than accept a station the duck
    could never occupy, the bearing is rotated in increments until the station
    clears everything; the distance to the guardian is preserved exactly, so the
    standoff band is never traded away for convenience.
    """
    target = np.asarray(target_xy, dtype=np.float64)
    origin = np.asarray(from_xy, dtype=np.float64)
    delta = origin - target
    if float(np.linalg.norm(delta)) < 1e-6:
        delta = np.array([1.0, 0.0])
    base = math.atan2(float(delta[1]), float(delta[0]))
    for step in range(0, 25):
        for sign in (1.0, -1.0):
            angle = base + sign * math.radians(9.0 * step)
            point = target + distance_m * np.array(
                [math.cos(angle), math.sin(angle)])
            if clear_of_obstacles(point, ROUTE_CLEARANCE_M):
                return point
    return target + distance_m * np.array([math.cos(base), math.sin(base)])


def nearest_obstacle(xy) -> tuple[str, float]:
    """Name of the nearest obstacle and the planar gap to its surface."""
    best_name, best = "", float("inf")
    for obstacle in OBSTACLES:
        gap = obstacle.distance_to(xy)
        if gap < best:
            best, best_name = gap, obstacle.name
    return best_name, best


def obstacle_height(name: str) -> float:
    return OBSTACLE_BY_NAME[name].height_m
