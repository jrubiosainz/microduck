#!/usr/bin/env python3
"""The concourse itself: floor, walls, occluders, and what blocks what.

Single source of truth for every static surface, shared by three consumers that
must never disagree:

* ``tools/build_scene.py`` paints the MuJoCo geometry from these shapes;
* ``route_planner`` inflates the SAME shapes to plan a collision-safe rejoin;
* the acceptance gate measures real surface clearance against the SAME geoms.

WHY THE OCCLUDERS ARE SHAPED THE WAY THEY ARE
----------------------------------------------
An occluder that only sometimes hides a person is decoration.  This behavior
needs a body that geometrically and *durably* removes the guardian from the
duck's head camera, so every occluder here is taller than the tallest camera
sample point on a person.

An adult's mocap origin sits at ``z = 0.36`` and the camera samples them at
``-0.06, +0.06, +0.20, +0.30`` about that origin, so the topmost sample is at
``z = 0.66``.  The duck's head camera sits near ``z = 0.19``.  A sightline
between those two heights never rises above 0.66, so **an occluder 0.90 m tall
or more blocks every sample at any range** and the only question left is
whether it is laterally in the way — which is pure planar geometry and is what
:meth:`Obstacle.segment_hits` answers.

THE BENCH IS DELIBERATELY LOW
-----------------------------
``bench`` is 0.24 m tall: taller than the duck, so it is a genuine obstacle the
rejoin route has to go round, but far below the 0.30 m lowest camera sample, so
it never occludes anybody.  That separation is the point.  It lets the scene
demonstrate "the route avoided an obstacle" without contaminating "the camera
could see the target", which would otherwise be the same event.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# -- hall extent ------------------------------------------------------------
FLOOR_CENTER = (0.00, 0.00)
FLOOR_HALF = (3.30, 2.30)
WALL_HALF_Z = 0.62
WALL_T = 0.06

# Anything at least this tall removes every camera sample of a person, because
# the topmost sample sits at z = 0.66 and the eye at z ~ 0.19.
OCCLUDING_HEIGHT_M = 0.90


@dataclass(frozen=True)
class Obstacle:
    """A static planar obstacle: an axis-aligned box or a circle.

    ``height_m`` is the full height above the floor.  ``occludes`` is derived,
    never declared, so a scene edit that shortens a body also stops it counting
    as an occluder in the tests and in the HUD.
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
        """Planar distance from ``xy`` to this obstacle's surface (negative inside)."""
        point = np.asarray(xy, dtype=np.float64)
        center = np.asarray(self.center, dtype=np.float64)
        if self.kind == "circle":
            return float(np.linalg.norm(point - center)) - self.radius
        delta = np.abs(point - center) - np.asarray(self.half, dtype=np.float64)
        outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
        inside = float(min(max(delta[0], delta[1]), 0.0))
        return outside + inside

    def corners(self, inflate: float) -> list[np.ndarray]:
        """Waypoint candidates around this obstacle, inflated by ``inflate``.

        A box gives its four expanded corners.  A circle gives eight points on
        the expanded circle, which is the discrete stand-in for the tangent
        points a continuous planner would use.
        """
        center = np.asarray(self.center, dtype=np.float64)
        if self.kind == "circle":
            r = self.radius + inflate
            return [center + r * np.array([math.cos(a), math.sin(a)])
                    for a in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)]
        hx, hy = self.half[0] + inflate, self.half[1] + inflate
        return [center + np.array([sx * hx, sy * hy])
                for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]

    def segment_hits(self, a, b, inflate: float = 0.0, samples: int = 64) -> bool:
        """Does segment ``a→b`` come within ``inflate`` of this obstacle?

        Sampled rather than solved in closed form because the two shapes need
        one predicate, and the sample count is fixed at a resolution far finer
        than the smallest obstacle: 64 samples over a segment no longer than the
        6.6 m hall is a 0.10 m step against a 0.22 m minimum obstacle radius, so
        a segment cannot slip through unnoticed.  Endpoints are included.
        """
        start = np.asarray(a, dtype=np.float64)
        end = np.asarray(b, dtype=np.float64)
        for index in range(samples + 1):
            point = start + (end - start) * (index / samples)
            if self.distance_to(point) < inflate:
                return True
        return False


# -- the concourse's furniture ---------------------------------------------
# The kiosk is the behavior's principal occluder: the guardian rounds its
# north-east corner and walks away along the north face while the duck is still
# on the east leg, and 1.10 m of solid kiosk stands in the sightline.
KIOSK = Obstacle("kiosk", "box", (0.45, 0.20), (0.55, 0.46), 1.10,
                 "information kiosk")
# The second cycle's occluder, on the far side of the hall.
COLUMN_W = Obstacle("column_w", "circle", (-1.35, 0.45), (0.24, 0.24), 1.30,
                    "column")
COLUMN_S = Obstacle("column_s", "circle", (-0.60, -1.35), (0.22, 0.22), 1.30,
                    "column")
CRATES = Obstacle("crates", "box", (-2.30, -0.55), (0.36, 0.30), 0.96,
                  "stacked crates")
PANEL_N = Obstacle("panel_n", "box", (-1.95, 1.62), (0.44, 0.06), 1.04,
                   "sign panel")
# LOW: a route obstacle that is not an occluder.  See the module docstring.
BENCH = Obstacle("bench", "box", (-2.70, -1.90), (0.34, 0.17), 0.24, "bench")

OBSTACLES: tuple[Obstacle, ...] = (
    KIOSK, COLUMN_W, COLUMN_S, CRATES, PANEL_N, BENCH)
OCCLUDERS: tuple[Obstacle, ...] = tuple(o for o in OBSTACLES if o.occludes)
BY_NAME: dict[str, Obstacle] = {o.name: o for o in OBSTACLES}


def occluder_between(eye_xy, target_xy, margin: float = 0.0) -> str | None:
    """Name of the first occluder standing in a planar sightline, if any.

    Planar-only, and deliberately so: every occluder is taller than the highest
    camera sample, so the third dimension cannot rescue a blocked sightline.
    This is the CHEAP predicate used for planning and for the "was line of sight
    even available" bookkeeping.  The authoritative visibility measurement is
    always the real MuJoCo ray cast in ``lost_camera``.
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
