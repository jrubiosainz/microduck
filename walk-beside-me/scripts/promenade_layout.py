#!/usr/bin/env python3
"""The promenade: floor, walls, and the bodies that make a SIDE unusable.

Single source of truth for every static surface, shared by three consumers that
must never disagree:

* ``tools/build_scene.py`` paints the MuJoCo geometry from these shapes;
* ``side_choice`` measures each candidate side slot against the SAME shapes;
* the acceptance gate measures real surface clearance against the SAME geoms.

WHY THESE SHAPES EXIST, ONE BY ONE
-----------------------------------
This behavior is about *which side of a person the robot walks on*, so scenery
here is not decoration: each body exists to make one specific side choice have a
right answer and a wrong one.

* ``hedge_s`` — a low planter row running along the south edge of the first
  straight.  The guardian's RIGHT-hand slot sits 0.07 m from it, so the very
  first join has to reject a side rather than pick one by default.  At 0.45 m it
  is well above the duck, and it stands above only the two LOWEST of the five
  camera samples on an adult (which sit at 0.26, 0.38, 0.52, 0.64 and 0.70), so
  it obstructs the robot while a body behind it keeps a majority of its samples
  and stays visible.
* ``kiosk`` — 1.05 m, north of the middle leg.  The guardian's LEFT-hand slot
  runs into its south face, which is what forces the first *switch*: the duck is
  already walking there when the side stops being available.
* ``column_n``, ``column_w`` — full-height cylinders, far enough from every slot
  that they never decide anything.  They are in the scene so that "the sightline
  was clear" is a measurement taken in a hall that contains occluders, rather
  than in an empty box where it could not have been otherwise.
* ``bench_w``, ``planter_ne`` — low furniture off the route, for the same
  reason: the clearance gate must have something to be non-vacuous about.

THE OCCLUSION ARITHMETIC
------------------------
An adult's mocap origin sits at ``z = 0.36`` and the camera samples them at
``-0.10, +0.02, +0.16, +0.28, +0.34`` about that origin, so the topmost sample
is at ``z = 0.70`` and the lowest at ``z = 0.26``.  The duck's head camera sits
near ``z = 0.19``.  Anything ``OCCLUDING_HEIGHT_M`` tall or more therefore
removes every sample of a body behind it; anything below 0.26 m cannot remove
any.  ``hedge_s`` at 0.45 m sits deliberately between those two figures and is
classified by the derived :attr:`Obstacle.occludes` rule rather than by a hand
label, so shortening it in this file also stops it counting as an occluder in
the tests and in the metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# -- promenade extent -------------------------------------------------------
FLOOR_CENTER = (0.00, 0.00)
FLOOR_HALF = (6.20, 3.20)
WALL_HALF_Z = 0.62
WALL_T = 0.06

# Anything at least this tall removes every camera sample of a person, because
# the topmost sample sits at z = 0.70 and the eye at z ~ 0.19.
OCCLUDING_HEIGHT_M = 0.90


@dataclass(frozen=True)
class Obstacle:
    """A static planar obstacle: an axis-aligned box or a circle.

    ``height_m`` is the full height above the floor.  ``occludes`` is derived,
    never declared, so a scene edit that shortens a body also stops it counting
    as an occluder everywhere at once.
    """

    name: str
    kind: str                      # "box" | "circle"
    center: tuple[float, float]
    half: tuple[float, float]      # box: (hx, hy);  circle: (r, r)
    height_m: float
    label: str = ""

    @property
    def radius(self) -> float:
        return float(self.half[0])

    @property
    def occludes(self) -> bool:
        return self.height_m >= OCCLUDING_HEIGHT_M

    def distance_to(self, xy) -> float:
        """Planar distance from ``xy`` to this surface (negative inside)."""
        point = np.asarray(xy, dtype=np.float64)
        center = np.asarray(self.center, dtype=np.float64)
        if self.kind == "circle":
            return float(np.linalg.norm(point - center)) - self.radius
        delta = np.abs(point - center) - np.asarray(self.half, dtype=np.float64)
        outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
        inside = float(min(max(delta[0], delta[1]), 0.0))
        return outside + inside

    def segment_hits(self, a, b, inflate: float = 0.0, samples: int = 64) -> bool:
        """Does segment ``a→b`` come within ``inflate`` of this obstacle?

        Sampled rather than solved in closed form because the two shapes need
        one predicate, and the sample count is fixed at a resolution far finer
        than the smallest obstacle: 64 samples over a segment no longer than the
        12.4 m promenade is a 0.20 m step against a 0.22 m minimum obstacle
        radius.  Endpoints are included.
        """
        start = np.asarray(a, dtype=np.float64)
        end = np.asarray(b, dtype=np.float64)
        for index in range(samples + 1):
            point = start + (end - start) * (index / samples)
            if self.distance_to(point) < inflate:
                return True
        return False


# -- the promenade's furniture ---------------------------------------------
# The planter row that makes the FIRST join a rejection rather than a default.
# It spans the whole south straight and reaches the south wall, so the ENTIRE
# 0.45-0.75 m right-hand band there is unusable, not merely its outer edge — a
# duck that simply tightened its formation could not evade it.
HEDGE_S = Obstacle("hedge_s", "box", (-4.65, -3.02), (0.95, 0.18), 0.45,
                   "planter row along the south edge")
# The body that takes the duck's own side away from it mid-walk.  Its south face
# sits 0.55 m from the guardian's east-straight centreline, which puts every
# lateral offset in the band inside the refusal margin.
KIOSK = Obstacle("kiosk", "box", (-2.20, -1.25), (0.52, 0.48), 1.05,
                 "information kiosk")
COLUMN_N = Obstacle("column_n", "circle", (0.40, 1.85), (0.24, 0.24), 1.35,
                    "column")
COLUMN_W = Obstacle("column_w", "circle", (-3.20, -0.20), (0.22, 0.22), 1.35,
                    "column")
BENCH_W = Obstacle("bench_w", "box", (-4.60, 1.30), (0.45, 0.16), 0.42,
                   "bench")
PLANTER_W = Obstacle("planter_w", "box", (-1.20, 2.60), (0.28, 0.55), 0.50,
                     "planter")

OBSTACLES: tuple[Obstacle, ...] = (
    HEDGE_S, KIOSK, COLUMN_N, COLUMN_W, BENCH_W, PLANTER_W)
OCCLUDERS: tuple[Obstacle, ...] = tuple(o for o in OBSTACLES if o.occludes)
BY_NAME: dict[str, Obstacle] = {o.name: o for o in OBSTACLES}


def nearest_obstacle(xy) -> tuple[str, float]:
    """Name of the nearest obstacle and the planar gap to its surface."""
    best_name, best = "", float("inf")
    for obstacle in OBSTACLES:
        gap = obstacle.distance_to(xy)
        if gap < best:
            best, best_name = gap, obstacle.name
    return best_name, best


def wall_gap(xy) -> float:
    """Planar gap from ``xy`` to the nearest perimeter wall's inner face."""
    point = np.asarray(xy, dtype=np.float64)
    return float(min(FLOOR_HALF[0] - abs(float(point[0])),
                     FLOOR_HALF[1] - abs(float(point[1]))))


def static_gap(xy) -> tuple[str, float]:
    """Gap to the nearest STATIC surface of any kind, obstacle or wall.

    This is the predicate ``side_choice`` grades a candidate side slot with, so
    a slot pressed against the promenade wall is refused for the same reason and
    on the same scale as one pressed against the kiosk.
    """
    name, gap = nearest_obstacle(xy)
    walls = wall_gap(xy)
    return (name, gap) if gap <= walls else ("wall", walls)


def occluder_between(eye_xy, target_xy, margin: float = 0.0) -> str | None:
    """Name of the first full-height occluder standing in a planar sightline.

    Planar-only, and deliberately so: every body in :data:`OCCLUDERS` is taller
    than the highest camera sample, so the third dimension cannot rescue a
    blocked sightline.  This is the CHEAP predicate used for the "was line of
    sight even available" bookkeeping.  The authoritative visibility measurement
    is always the real MuJoCo ray cast in ``beside_camera``.
    """
    for obstacle in OCCLUDERS:
        if obstacle.segment_hits(eye_xy, target_xy, margin):
            return obstacle.name
    return None


def clear_of_obstacles(xy, clearance: float) -> bool:
    """Is ``xy`` at least ``clearance`` from every obstacle and inside the hall?"""
    point = np.asarray(xy, dtype=np.float64)
    if abs(float(point[0])) > FLOOR_HALF[0] - clearance:
        return False
    if abs(float(point[1])) > FLOOR_HALF[1] - clearance:
        return False
    return all(o.distance_to(point) >= clearance for o in OBSTACLES)


def segment_blocked_by(a, b, inflate: float) -> str | None:
    """First obstacle a straight walk from ``a`` to ``b`` would come too near."""
    for obstacle in OBSTACLES:
        if obstacle.segment_hits(a, b, inflate):
            return obstacle.name
    return None


def hall_diagonal() -> float:
    return float(math.hypot(2.0 * FLOOR_HALF[0], 2.0 * FLOOR_HALF[1]))
