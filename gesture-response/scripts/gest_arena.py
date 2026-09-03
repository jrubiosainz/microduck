#!/usr/bin/env python3
"""The training area: its extent, its fixtures, and the two painted marks.

One source of truth for every static surface and every named place.  Five
consumers read these SAME objects, so a geometry edit moves everything at once:
``tools/build_scene.py`` paints the MuJoCo geometry, ``gest_control`` prunes
nothing but reads the marks, the acceptance gate measures real surface clearance
against the SAME geoms, the HUD plan view draws them where they are, and
``gest_actors`` routes the distracting adults around them.

WHY THE AREA IS OPEN, AND WHAT THE FEW FIXTURES ARE FOR
---------------------------------------------------------
The scenario is an open training area, so the middle of the floor is deliberately
empty: every metre the duck walks is a metre it chose to walk, not a metre a
corridor forced on it.  The fixtures sit around the edges and exist for two
measurable reasons rather than for decoration:

* **Two of them are real occluders.**  ``obs_rack_e`` and ``obs_post_n`` stand
  0.75 m and 1.20 m tall against a head camera at about 0.20 m, so a distracting
  adult who walks behind one is genuinely hidden and :func:`occluder_between`
  returns a real name on the real run.  The visibility gate is therefore
  conditioned on something that actually happens.
* **The rest bound the working area** so the turns have a floor to happen on and
  the clearance gate has surfaces to grade against.

THE TWO MARKS ARE PAINTED ON THE FLOOR
----------------------------------------
The instructor's mark and the duck's start pad are real world geometry rather
than HUD annotation, because a viewer must be able to see that the instructor
stood still on her mark for the whole session while the duck moved around her.

THE GEOMETRY THE TURNS NEED
-----------------------------
This robot **cannot turn on the spot** - MEASURED at ``vx = 0`` across the whole
command range it manages about a degree per second - so every commanded turn is
a WALKED ARC of a MEASURED radius.  That is why the instructor's mark sits
:data:`INSTRUCTOR_MARK` metres from the duck's start rather than closer: the arc
of a 90 deg turn taken from the safe observation standoff must not carry the
robot into the person it is being trained by, and
``tools/check_layout.py`` checks exactly that against the MEASURED per-sign
turning radii.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -- extent -----------------------------------------------------------------
# Sized against the turn arcs rather than chosen: the duck must be able to take
# a 90 deg left arc and a 90 deg right arc from the observation standoff, plus a
# reverse leg, without leaving the floor or reaching a wall.
FLOOR_HALF = (3.00, 2.40)
WALL_HALF_Z = 0.90
WALL_T = 0.06

# Anything this tall removes a body behind it from the head camera, whose
# optical centre sits at about 0.20 m.
OCCLUDING_HEIGHT_M = 0.50

# -- the two marks -----------------------------------------------------------
# The instructor stands here for the whole session and does not move.  Standing
# still is deliberate: a moving instructor would let "the duck closed the range"
# be partly a fact about her walking toward it.
INSTRUCTOR_MARK = (0.00, 1.30)
# She faces due south, straight down the training area, for the whole run.  A
# FIXED facing is what makes the two pointing gestures mirror images of each
# other - a pure lateral abduction of one arm or the other - which is the most
# readable pose a camera can be offered and the least ambiguous one to measure.
INSTRUCTOR_FACING_DEG = -90.0

# Where the duck starts, and which way it faces.
DUCK_START = (0.00, -1.60)
DUCK_START_YAW_DEG = 90.0
MARK_HALF = (0.26, 0.26)


@dataclass(frozen=True)
class Fixture:
    """One static body.  ``kind`` is ``"box"`` or ``"cylinder"``."""

    name: str
    kind: str
    center: tuple[float, float]
    half: tuple[float, float]
    height_m: float
    material: str
    label: str = ""

    @property
    def radius(self) -> float:
        return float(self.half[0])

    @property
    def occludes(self) -> bool:
        return self.height_m >= OCCLUDING_HEIGHT_M

    def distance_to(self, xy) -> float:
        """Planar distance from ``xy`` to this surface.  Negative inside."""
        point = np.asarray(xy, dtype=np.float64)[:2]
        center = np.asarray(self.center, dtype=np.float64)
        if self.kind == "cylinder":
            return float(np.linalg.norm(point - center)) - self.radius
        delta = np.abs(point - center) - np.asarray(self.half, dtype=np.float64)
        outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
        inside = float(min(max(delta[0], delta[1]), 0.0))
        return outside + inside

    def segment_hits(self, a, b, inflate: float = 0.0, samples: int = 48) -> bool:
        """Does the segment ``a -> b`` come within ``inflate`` of this body?"""
        start = np.asarray(a, dtype=np.float64)[:2]
        end = np.asarray(b, dtype=np.float64)[:2]
        for index in range(samples + 1):
            point = start + (end - start) * (index / samples)
            if self.distance_to(point) < inflate:
                return True
        return False


FIXTURES: tuple[Fixture, ...] = (
    # THE TWO OCCLUDERS.  Tall enough to genuinely hide a person from a head
    # camera at 0.20 m, and placed on the routes the distracting adults walk, so
    # the occlusion predicate fires on the real run instead of being decorative.
    Fixture("obs_rack_e", "box", (2.12, 0.45), (0.30, 0.58), 0.75,
            "rackmat", "equipment rack, east edge: a real occluder"),
    Fixture("obs_post_n", "cylinder", (-1.44, 1.98), (0.10, 0.10), 1.20,
            "columnmat", "roof post, north-west: a real occluder"),
    # The rest bound the area without screening anything.
    Fixture("obs_rack_w", "box", (-2.30, -0.35), (0.26, 0.52), 0.46,
            "rackmat", "low kit rack, west edge"),
    Fixture("obs_bench_s", "box", (1.62, -1.92), (0.55, 0.18), 0.42,
            "benchmat", "spectator bench, south-east"),
    Fixture("obs_cone_e", "cylinder", (1.18, 0.06), (0.09, 0.09), 0.32,
            "conemat", "training cone, east of the working line"),
    Fixture("obs_cone_w", "cylinder", (-1.18, 0.06), (0.09, 0.09), 0.32,
            "conemat", "training cone, west of the working line"),
)

OCCLUDERS: tuple[Fixture, ...] = tuple(f for f in FIXTURES if f.occludes)
BY_FIXTURE: dict[str, Fixture] = {f.name: f for f in FIXTURES}


def instructor_position() -> np.ndarray:
    return np.asarray(INSTRUCTOR_MARK, dtype=np.float64)


def wall_gap(xy) -> float:
    """Planar gap from ``xy`` to the nearest perimeter wall's inner face."""
    point = np.asarray(xy, dtype=np.float64)[:2]
    return float(min(FLOOR_HALF[0] - abs(float(point[0])),
                     FLOOR_HALF[1] - abs(float(point[1]))))


def nearest_fixture(xy) -> tuple[str, float]:
    best_name, best = "", float("inf")
    for fixture in FIXTURES:
        gap = fixture.distance_to(xy)
        if gap < best:
            best, best_name = gap, fixture.name
    return best_name, best


def static_gap(xy) -> tuple[str, float]:
    """Gap to the nearest STATIC surface of any kind, fixture or perimeter."""
    name, gap = nearest_fixture(xy)
    walls = wall_gap(xy)
    return (name, gap) if gap <= walls else ("wall", walls)


def occluder_between(eye_xy, target_xy, margin: float = 0.0) -> str | None:
    """Name of the first full-height STATIC occluder in a planar sightline."""
    for fixture in OCCLUDERS:
        if fixture.segment_hits(eye_xy, target_xy, margin):
            return fixture.name
    return None


def inside_area(xy, margin: float = 0.0) -> bool:
    """Is a planar point inside the working floor, with ``margin`` to spare?"""
    point = np.asarray(xy, dtype=np.float64)[:2]
    return bool(abs(float(point[0])) <= FLOOR_HALF[0] - margin
                and abs(float(point[1])) <= FLOOR_HALF[1] - margin)
