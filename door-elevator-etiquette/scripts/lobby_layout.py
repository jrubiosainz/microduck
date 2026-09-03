#!/usr/bin/env python3
"""The concourse, the automatic door, the lift and its cabin — one source of
truth for every surface, aperture and zone in this behavior.

Four consumers read these SAME objects, so a geometry edit moves everything at
once and none of them can drift apart:

* ``tools/build_scene.py`` paints the MuJoCo geometry from these shapes;
* ``etiquette_zones`` derives the threshold, passage and cabin zones from the
  same apertures;
* the acceptance gate measures real surface clearance against the SAME geoms;
* ``lobby_doors`` slides its leaves inside the SAME apertures.

WHY THE LIFT IS A THROUGH-CAR, AND WHY THAT IS NOT A CONVENIENCE
------------------------------------------------------------------
A single-entry cabin forces the robot to turn about 180 deg inside a box a
metre and a half across.  The duck's MEASURED yaw authority is a few degrees
per second, so that manoeuvre is not available to it at all — a behavior built
on one would be a behavior that cannot be walked.  A through-car (doors front
and rear, a real service-lift configuration) lets the whole route be
monotonically forward, which is what makes every leg of it physically
reachable.  The cost is that the cabin is a goods lift rather than a passenger
one, and that is stated in the README rather than hidden.

THE APERTURE IS WIDE ENOUGH FOR TWO, AND THAT IS THE POINT
------------------------------------------------------------
The automatic door's clear width is :data:`DOOR_CLEAR_W` = 0.66 m.  The duck's
MEASURED exact lateral half-width is 0.0710 m and the widest adult's 0.1071 m,
so a duck and a person abreast occupy 0.356 m and would pass with **0.304 m** of
clear air between them - comfortably more than the :data:`ABREAST_MARGIN_M` this
behavior treats as room to pass.  The lift aperture is wider still, at 0.364 m.

That arithmetic is stated here because it is what makes the etiquette gate MEAN
something.  Squeezing the doorway until two bodies physically could not fit
would turn "the duck never went through side by side" into a fact about the wall
rather than about the robot, and the gate would pass no matter what the state
machine did.  The openings are deliberately wide enough to misuse, and the gate
checks per tick that the duck and another body were never inside the same
aperture box at the same time.  ``tools/check_layout.py`` prints the budget so
the non-vacuity is visible rather than asserted.

WHAT OCCLUDES AND WHAT DOES NOT
--------------------------------
An adult's mocap origin sits at ``z = 0.36`` and the camera samples them at
``-0.10, +0.02, +0.16, +0.28, +0.34`` about it, so the topmost sample is at
``z = 0.70`` and the duck's head camera near ``z = 0.19``.  Anything
:data:`OCCLUDING_HEIGHT_M` or taller removes every sample of a body behind it.
``occludes`` is DERIVED from the height, so shortening a body here also stops it
counting as an occluder in the camera bookkeeping, the tests and the metrics at
once.

THE PARTITIONS ARE CUTAWAY HEIGHT, AND THAT IS A DELIBERATE TRADE
-------------------------------------------------------------------
A real lift shaft is full height, and a full-height one here would mean the
overhead wide camera films the lid of a box for the entire cabin sequence - the
part of the behavior the video exists to show.  :data:`PARTITION_H` and
:data:`CABIN_H` are therefore CUTAWAY heights.  Both are above
:data:`OCCLUDING_HEIGHT_M` and far above the 0.70 m topmost sample of an adult,
so the duck's own head camera is occluded by them EXACTLY as it would be by
full-height walls: nothing the ROBOT measures changes.  Only the spectator's
camera sees over them, and the README states it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -- extent -----------------------------------------------------------------
# Sized against the ROUTE and the CLOCK rather than chosen for looks.
#
# The clock is the binding constraint and it was MEASURED, not estimated.
# ``tools/measure_legs.py`` walks the whole route with the real policy and no
# state machine, and the first draft of this building came back at 74.7 s of
# pure walking over 8.65 m - which with the yields, the lift wait and the ride
# on top puts the video past 115 s.  The duck averages 0.109-0.133 m/s depending
# on how much of a leg sits inside an aperture's careful band, so every metre of
# building costs 8-9 s of video.
#
# The building was therefore shortened rather than the behavior: the concourse
# lost 0.4 m of approach and the lift moved 0.26 m west, which buys back time
# without removing a single thing the scenario has to show.
FLOOR_HALF = (4.20, 2.40)
WALL_HALF_Z = 0.72
WALL_T = 0.06

OCCLUDING_HEIGHT_M = 0.90
# Cutaway heights.  See the module docstring: above the occlusion threshold, so
# the head camera cannot see through them, but low enough that the overhead wide
# camera can see into the lift cabin.
PARTITION_H = 1.35
CABIN_H = 1.15

# -- the automatic door in the concourse divider ----------------------------
DOOR_WALL_X = -1.10
DOOR_WALL_HALF_T = 0.07
DOOR_CENTER_Y = 0.00
DOOR_CLEAR_W = 0.66
DOOR_CLEAR_HALF = DOOR_CLEAR_W * 0.5
# Each leaf spans half the opening when closed and retracts into the wall, so a
# fully open door restores the whole clear width and a closed one covers it.
# The travel EQUALS the clear half width, which is what makes the effective gap
# ``2 * travel * fraction`` rather than an independent number to keep in step.
DOOR_LEAF_HALF_W = DOOR_CLEAR_HALF
DOOR_LEAF_HALF_T = 0.025
DOOR_LEAF_TRAVEL = DOOR_CLEAR_HALF

# -- the lift, front (lobby) face -------------------------------------------
# The aperture centre sits at y = +0.10 rather than on the hall axis, which
# together with the rear face's y = -0.35 gives the route through the car a
# gentle S instead of a pair of right angles.  That is a MEASURED constraint,
# not a preference: see ``etiquette_states`` for the yaw authority this robot
# actually has.
LIFT_WALL_X = 1.34
LIFT_WALL_HALF_T = 0.06
LIFT_FRONT_Y = 0.10
LIFT_CLEAR_W = 0.72
LIFT_CLEAR_HALF = LIFT_CLEAR_W * 0.5
LIFT_LEAF_HALF_W = LIFT_CLEAR_HALF
LIFT_LEAF_HALF_T = 0.022
LIFT_LEAF_TRAVEL = LIFT_CLEAR_HALF

# -- the cabin --------------------------------------------------------------
# Interior faces.  A goods lift: wide enough that the duck can cross from the
# front aperture to the side at a corner radius its MEASURED yaw rate supports.
CABIN_X = (1.40, 3.04)
CABIN_Y = (-1.00, 0.95)
CABIN_WALL_HALF_T = 0.06

# -- the lift, rear (target floor) face -------------------------------------
REAR_WALL_X = 3.10
REAR_WALL_HALF_T = 0.06
REAR_Y = -0.35
REAR_CLEAR_W = 0.72
REAR_CLEAR_HALF = REAR_CLEAR_W * 0.5
REAR_LEAF_HALF_W = REAR_CLEAR_HALF
REAR_LEAF_HALF_T = 0.022
REAR_LEAF_TRAVEL = REAR_CLEAR_HALF

# Two bodies abreast must keep at least this much clear air between their
# surfaces for the manoeuvre to count as a pass rather than a squeeze.  Derived
# from the duck's own conservative planar diameter (0.2324 m): anything below
# about one robot width is not room to pass, it is room to collide.  Both
# apertures here EXCEED it by a wide margin, on purpose - see the module
# docstring.
ABREAST_MARGIN_M = 0.26


@dataclass(frozen=True)
class Obstacle:
    """A static axis-aligned box.  ``height_m`` is the full height above floor."""

    name: str
    center: tuple[float, float]
    half: tuple[float, float]
    height_m: float
    material: str = "wallmat"
    label: str = ""

    @property
    def occludes(self) -> bool:
        return self.height_m >= OCCLUDING_HEIGHT_M

    def distance_to(self, xy) -> float:
        """Planar distance from ``xy`` to this surface (negative inside)."""
        point = np.asarray(xy, dtype=np.float64)
        delta = np.abs(point - np.asarray(self.center, dtype=np.float64)) \
            - np.asarray(self.half, dtype=np.float64)
        outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
        inside = float(min(max(delta[0], delta[1]), 0.0))
        return outside + inside

    def segment_hits(self, a, b, inflate: float = 0.0, samples: int = 96) -> bool:
        """Does segment ``a -> b`` come within ``inflate`` of this box?"""
        start = np.asarray(a, dtype=np.float64)
        end = np.asarray(b, dtype=np.float64)
        for index in range(samples + 1):
            point = start + (end - start) * (index / samples)
            if self.distance_to(point) < inflate:
                return True
        return False


def _wall_pair(prefix: str, x: float, half_t: float, center_y: float,
               clear_half: float, height: float, material: str,
               label: str) -> tuple[Obstacle, Obstacle]:
    """The two jamb segments either side of one aperture, spanning the hall.

    Built from the aperture rather than declared, so widening an opening cannot
    leave a jamb behind at the old width.
    """
    south_lo, south_hi = -FLOOR_HALF[1], center_y - clear_half
    north_lo, north_hi = center_y + clear_half, FLOOR_HALF[1]
    return (
        Obstacle(f"{prefix}_s", (x, 0.5 * (south_lo + south_hi)),
                 (half_t, 0.5 * (south_hi - south_lo)), height, material,
                 f"{label} (south jamb)"),
        Obstacle(f"{prefix}_n", (x, 0.5 * (north_lo + north_hi)),
                 (half_t, 0.5 * (north_hi - north_lo)), height, material,
                 f"{label} (north jamb)"),
    )


DOOR_JAMB_S, DOOR_JAMB_N = _wall_pair(
    "wall_door", DOOR_WALL_X, DOOR_WALL_HALF_T, DOOR_CENTER_Y,
    DOOR_CLEAR_HALF, PARTITION_H, "doormat",
    "concourse divider with the automatic door")

LIFT_JAMB_S, LIFT_JAMB_N = _wall_pair(
    "wall_lift", LIFT_WALL_X, LIFT_WALL_HALF_T, LIFT_FRONT_Y,
    LIFT_CLEAR_HALF, PARTITION_H, "shaftmat", "lift shaft front face")

# The rear face closes the cabin, so its jambs span the CABIN rather than the
# hall: a jamb that ran to the perimeter would wall the target-floor corridor
# off from the rest of the scene and hide the arrival.
_REAR_S_LO, _REAR_S_HI = CABIN_Y[0] - CABIN_WALL_HALF_T, REAR_Y - REAR_CLEAR_HALF
_REAR_N_LO, _REAR_N_HI = REAR_Y + REAR_CLEAR_HALF, CABIN_Y[1] + CABIN_WALL_HALF_T
REAR_JAMB_S = Obstacle(
    "wall_rear_s", (REAR_WALL_X, 0.5 * (_REAR_S_LO + _REAR_S_HI)),
    (REAR_WALL_HALF_T, 0.5 * (_REAR_S_HI - _REAR_S_LO)), CABIN_H,
    "shaftmat", "cabin rear face (south jamb)")
REAR_JAMB_N = Obstacle(
    "wall_rear_n", (REAR_WALL_X, 0.5 * (_REAR_N_LO + _REAR_N_HI)),
    (REAR_WALL_HALF_T, 0.5 * (_REAR_N_HI - _REAR_N_LO)), CABIN_H,
    "shaftmat", "cabin rear face (north jamb)")

_CABIN_MID_X = 0.5 * (CABIN_X[0] + REAR_WALL_X + REAR_WALL_HALF_T)
_CABIN_HALF_X = 0.5 * (REAR_WALL_X + REAR_WALL_HALF_T - CABIN_X[0])
CABIN_WALL_S = Obstacle(
    "cabin_wall_s", (_CABIN_MID_X, CABIN_Y[0] - CABIN_WALL_HALF_T),
    (_CABIN_HALF_X, CABIN_WALL_HALF_T), CABIN_H, "cabinmat",
    "cabin south wall")
CABIN_WALL_N = Obstacle(
    "cabin_wall_n", (_CABIN_MID_X, CABIN_Y[1] + CABIN_WALL_HALF_T),
    (_CABIN_HALF_X, CABIN_WALL_HALF_T), CABIN_H, "cabinmat",
    "cabin north wall")

# Low furniture, deliberately below the 0.26 m lowest camera sample on an adult
# so it constrains the duck without ever hiding anybody, and kept well off the
# route.  It gives the per-tick clearance gate something non-vacuous to grade in
# the lobby and in the target-floor corridor.
BENCH_LOBBY = Obstacle("obs_bench_lobby", (0.10, 1.86), (0.46, 0.14), 0.44,
                       "benchmat", "lobby bench")
PLANTER_CORR = Obstacle("obs_planter_corr", (3.72, 0.92), (0.20, 0.28), 0.52,
                        "plantermat", "corridor planter")
CRATE_WEST = Obstacle("obs_crate_west", (-2.60, 1.55), (0.30, 0.22), 0.60,
                      "cratemat", "stacked crates")

STATIC_OBSTACLES: tuple[Obstacle, ...] = (
    DOOR_JAMB_S, DOOR_JAMB_N, LIFT_JAMB_S, LIFT_JAMB_N,
    REAR_JAMB_S, REAR_JAMB_N, CABIN_WALL_S, CABIN_WALL_N,
    BENCH_LOBBY, PLANTER_CORR, CRATE_WEST,
)
OCCLUDERS: tuple[Obstacle, ...] = tuple(
    o for o in STATIC_OBSTACLES if o.occludes)
BY_NAME: dict[str, Obstacle] = {o.name: o for o in STATIC_OBSTACLES}


def wall_gap(xy) -> float:
    """Planar gap from ``xy`` to the nearest perimeter wall's inner face."""
    point = np.asarray(xy, dtype=np.float64)
    return float(min(FLOOR_HALF[0] - abs(float(point[0])),
                     FLOOR_HALF[1] - abs(float(point[1]))))


def nearest_obstacle(xy) -> tuple[str, float]:
    """Name of the nearest static obstacle and the planar gap to its surface."""
    best_name, best = "", float("inf")
    for obstacle in STATIC_OBSTACLES:
        gap = obstacle.distance_to(xy)
        if gap < best:
            best, best_name = gap, obstacle.name
    return best_name, best


def static_gap(xy) -> tuple[str, float]:
    """Gap to the nearest STATIC surface of any kind, obstacle or perimeter."""
    name, gap = nearest_obstacle(xy)
    walls = wall_gap(xy)
    return (name, gap) if gap <= walls else ("wall", walls)


def occluder_between(eye_xy, target_xy, margin: float = 0.0) -> str | None:
    """Name of the first full-height STATIC occluder in a planar sightline.

    Planar-only and deliberately so: every body in :data:`OCCLUDERS` is taller
    than the highest camera sample, so the third dimension cannot rescue a
    blocked sightline.  This is the CHEAP predicate used for the "was line of
    sight even available" bookkeeping; the authoritative visibility measurement
    is always the real MuJoCo ray cast in ``etiquette_camera``.

    The door and lift LEAVES are excluded on purpose.  They move, so a sightline
    they block is a sightline the scenario blocked at that instant, and the ray
    cast is what grades it.

    PEOPLE ARE HANDLED SEPARATELY, BY ``etiquette_sense.los_blocked_by``, and
    that separation is a scar.  A version of this behavior conditioned the
    visibility gate on static geometry alone, which meant that when the guardian
    stood between the duck's head camera and the exiter it was waiting for, the
    duck was held responsible for not seeing through her: the gate read 28.7%
    where the honest figure was near 100%.  A body in the way is exactly as real
    an occluder as a wall, and neither is the robot's fault.
    """
    for obstacle in OCCLUDERS:
        if obstacle.segment_hits(eye_xy, target_xy, margin):
            return obstacle.name
    return None
