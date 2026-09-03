#!/usr/bin/env python3
"""The concourse: floor, walls, the bodies that shape the route, and the
three semantic destinations one of which will be requested.

Single source of truth for every static surface and every destination, shared by
four consumers that must never disagree:

* ``tools/build_scene.py`` paints the MuJoCo geometry from these shapes;
* ``guide_planner`` searches free space against the SAME shapes, inflated;
* the acceptance gate measures real surface clearance against the SAME geoms;
* ``guide_machine`` resolves the requested destination against the SAME registry.

WHY EACH BODY EXISTS, ONE BY ONE
---------------------------------
This behavior is about *leading somebody along a route they could not have taken
in a straight line*, so scenery here is not decoration: each body exists to make
one specific bend inevitable, or to make one specific sightline breakable.

* ``partition_c`` — a full-height partition running from the north wall down to
  ``y = -1.23``.  Its north end leaves 0.17 m of gap, which no inflated path can
  use, so the concourse is SEALED on that side and the route has to round its
  south end.  It is also the body that breaks the sightline to a follower who
  has not yet rounded the corner, which is what makes the second lag episode a
  genuine LOSS of the person rather than a large number.
* ``hall_screen`` — a full-height screen running from the south wall UP to
  ``y = +1.06``, offset east of the partition.  It seals the opposite side, so
  after rounding the partition to the south the route must climb back north.
  Those two bodies together are what make the route a zigzag rather than a
  dogleg, and neither of them can be squeezed past: both gaps are measured
  against the planner's own inflation in ``tools/check_layout.py``.
* ``column_e`` — a full-height column standing beside the final leg to the
  lifts, close enough to the straight line that the last approach has to bend
  around it.
* ``column_w`` — a full-height column beside the first leg.  It bends the
  opening leg and gives the scene a second real occluder, so "the sightline was
  clear" is a measurement taken in a hall that contains occluders.
* ``crate_ne``, ``bench_s``, ``planter_w`` — low furniture off the route.  They
  are below the lowest camera sample on an adult, so they can constrain the duck
  without ever hiding anybody, and they give the clearance gate something to be
  non-vacuous about.

THE OCCLUSION ARITHMETIC
------------------------
An adult's mocap origin sits at ``z = 0.36`` and the camera samples them at
``-0.10, +0.02, +0.16, +0.28, +0.34`` about that origin, so the topmost sample
is at ``z = 0.70`` and the lowest at ``z = 0.26``.  The duck's head camera sits
near ``z = 0.19``.  Anything :data:`OCCLUDING_HEIGHT_M` tall or more therefore
removes every sample of a body behind it; anything below 0.26 m cannot remove
any.  ``occludes`` is DERIVED from the height rather than declared, so shortening
a body in this file also stops it counting as an occluder in the planner, the
tests and the metrics at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# -- concourse extent -------------------------------------------------------
# SIZED AGAINST THE CLOCK AND AGAINST THE CROWD, not chosen for looks.
#
# The requested route has to be long enough to carry three real bends and two
# lag episodes, and short enough that the whole scenario fits a 70-90 s video at
# the duck's MEASURED 0.150 m/s lead pace.  A 4.60 x 2.90 m hall produced an
# 11.34 m route — 76 s of walking before a single wait — so the hall was scaled
# down until the route came out near 9 m.
FLOOR_CENTER = (0.00, 0.00)
FLOOR_HALF = (3.60, 2.30)
WALL_HALF_Z = 0.62
WALL_T = 0.06

# Anything at least this tall removes every camera sample of a person, because
# the topmost sample sits at z = 0.70 and the eye at z ~ 0.19.
OCCLUDING_HEIGHT_M = 0.90


@dataclass(frozen=True)
class Obstacle:
    """A static planar obstacle: an axis-aligned box or a circle.

    ``height_m`` is the full height above the floor.  ``occludes`` is derived,
    never declared.
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

    def segment_hits(self, a, b, inflate: float = 0.0, samples: int = 96) -> bool:
        """Does segment ``a→b`` come within ``inflate`` of this obstacle?

        Sampled rather than solved in closed form because the two shapes need
        one predicate, and the sample count is fixed at a resolution far finer
        than the smallest obstacle: 96 samples over a segment no longer than the
        9.2 m concourse is a 0.10 m step against a 0.20 m minimum half-extent.
        Endpoints are included.
        """
        start = np.asarray(a, dtype=np.float64)
        end = np.asarray(b, dtype=np.float64)
        for index in range(samples + 1):
            point = start + (end - start) * (index / samples)
            if self.distance_to(point) < inflate:
                return True
        return False


# -- the concourse's furniture ---------------------------------------------
# EVERY PASSAGE WIDTH HERE IS A MEASUREMENT, AND THE MEASUREMENT THAT MATTERS
# IS NOT THE ONE THAT LOOKS OBVIOUS.
#
# The obvious reading of "sealed at one end" is that the passage is the gap
# between a barrier's free end and the far wall.  There are three such gaps that
# matter, not one, because ``partition_c`` is sealed at the NORTH wall and
# ``hall_screen`` at the SOUTH: a route must pass SOUTH of the partition, climb
# the corridor BETWEEN them, and then pass NORTH of the screen.
#
# Each of those three has to satisfy two opposing constraints at once:
#
#   * narrow enough that the hall genuinely shapes the route;
#   * wider than ``2 * STATIC_INFLATE_M + 2 * CROWD_INFLATE_M`` = 1.70 m, or a
#     SINGLE adult standing in it closes the only way through and the planner
#     correctly reports the concourse sealed.
#
# The lower bound was measured twice, the hard way.  Barriers whose free ends
# left 1.50 m gaps are geometrically fine for a 0.26 m robot and were sealed by
# any one of the six adults; a coordinate-descent sweep over per-actor start
# offsets found ZERO feasible planning instants, which is the correct answer to
# a badly sized hall rather than a scheduling problem to be tuned away.  All
# three passages are now 2.15 m or wider, which one centred adult narrows to
# 1.05 m but cannot close.
#
# Runs from the north wall down to y = -0.15, sealed at the top with 0.08 m to
# spare, leaving a 2.15 m passage to the south.
PARTITION_C = Obstacle("partition_c", "box", (-1.15, 1.115), (0.15, 1.265), 2.05,
                       "full-height partition, sealed at the north wall")
# Runs from the south wall up to y = +0.15.  Sealed at the bottom, so after
# rounding the partition the route must climb the 2.30 m corridor between the
# two bodies and cross back over the screen's north end.  The two openings are
# on opposite sides, which is what makes the route a zigzag rather than a
# dogleg.
HALL_SCREEN = Obstacle("hall_screen", "box", (1.15, -1.115), (0.15, 1.265), 2.05,
                       "full-height screen, sealed at the south wall")
COLUMN_E = Obstacle("column_e", "circle", (2.55, 0.72), (0.18, 0.18), 2.35,
                    "column beside the final leg")
COLUMN_W = Obstacle("column_w", "circle", (-2.45, 0.15), (0.17, 0.17), 2.35,
                    "column beside the opening leg")
# LOW FURNITURE, and its placement is a measurement too.  All three are below
# the 0.26 m lowest camera sample on an adult, so they constrain the duck
# without ever hiding anybody, and they give the per-tick clearance gate
# something to be non-vacuous about.  They are deliberately kept OUT of every
# corridor: ``crate_ne`` at (-2.75, 1.85) walled the HELPDESK corner off
# entirely, and at (-1.15, -1.95) it sealed the southern corridor that carries
# the opening leg.  A hall in which one crate can silently reduce three
# destination candidates to two is not a hall with three candidates.
CRATE_NE = Obstacle("crate_ne", "box", (1.85, 2.08), (0.30, 0.22), 0.62,
                    "stacked crates")
BENCH_S = Obstacle("bench_s", "box", (-0.15, -2.08), (0.44, 0.13), 0.44,
                   "bench")
PLANTER_W = Obstacle("planter_w", "box", (3.22, 0.95), (0.18, 0.30), 0.52,
                     "planter")

OBSTACLES: tuple[Obstacle, ...] = (
    PARTITION_C, HALL_SCREEN, COLUMN_E, COLUMN_W, CRATE_NE, BENCH_S, PLANTER_W)
OCCLUDERS: tuple[Obstacle, ...] = tuple(o for o in OBSTACLES if o.occludes)
BY_NAME: dict[str, Obstacle] = {o.name: o for o in OBSTACLES}


@dataclass(frozen=True)
class Destination:
    """One place a person could ask to be taken to.

    ``key`` is what a request names.  Resolving a request is a lookup in
    :data:`DESTINATIONS` and nothing else, so a run in which the duck walked to
    the wrong place is a run in which the lookup returned the wrong entry — which
    the acceptance gate can state exactly.
    """

    key: str
    label: str
    xy: tuple[float, float]
    # Where the duck should STAND to have arrived: offset from the fixture, so
    # the duck stops in front of it rather than inside it.
    stand_xy: tuple[float, float]
    color: tuple[float, float, float]

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.xy, dtype=np.float64)

    @property
    def stand(self) -> np.ndarray:
        return np.asarray(self.stand_xy, dtype=np.float64)

    @property
    def rgba(self) -> str:
        r, g, b = self.color
        return f"{r:.3f} {g:.3f} {b:.3f} 1"


# WHICH KEY SITS WHERE IS A SCENARIO DECISION, MEASURED RATHER THAN ASSIGNED.
# The duck starts in the south-west.  From there the planner produces:
#
#   * north-east corner -> 7.01 m, 2 bends  (round the partition, then straight)
#   * south-east corner -> 7.76 m, 4 bends  (round the partition, climb the
#                                            corridor, cross the screen's north
#                                            end, then drop back south)
#   * north-west corner -> 3.90 m, 3 bends  (round the crates and the column)
#
# ``LIFTS`` is therefore the SOUTH-EAST corner: it is the request whose route
# the hall genuinely shapes, and the two candidates the duck must not walk to
# are a shorter route in the same direction and a much shorter one behind it.
# A guide that ignored the request and went to the nearest fixture, or to the
# furthest, would fail on a different destination each time.
DESTINATIONS: tuple[Destination, ...] = (
    Destination("LIFTS", "the lifts", (3.32, -1.98), (2.88, -1.62),
                (0.243, 0.545, 0.882)),
    Destination("CAFE", "the cafe", (3.32, 1.98), (2.88, 1.62),
                (0.905, 0.612, 0.212)),
    Destination("HELPDESK", "the help desk", (-3.32, 1.02), (-2.86, 0.72),
                (0.365, 0.784, 0.463)),
)
DESTINATION_BY_KEY: dict[str, Destination] = {d.key: d for d in DESTINATIONS}
DESTINATION_KEYS: tuple[str, ...] = tuple(d.key for d in DESTINATIONS)


def resolve_destination(key: str) -> Destination:
    """The destination a request names, or a loud failure.

    No fuzzy matching and no default.  A guide that silently walked somewhere
    plausible when it did not understand the request would be worse than one
    that refused, and the gate could not tell the two apart.
    """
    if key not in DESTINATION_BY_KEY:
        raise KeyError(
            f"unknown destination {key!r}; this concourse offers "
            f"{DESTINATION_KEYS}")
    return DESTINATION_BY_KEY[key]


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
    """Gap to the nearest STATIC surface of any kind, obstacle or wall."""
    name, gap = nearest_obstacle(xy)
    walls = wall_gap(xy)
    return (name, gap) if gap <= walls else ("wall", walls)


def occluder_between(eye_xy, target_xy, margin: float = 0.0) -> str | None:
    """Name of the first full-height occluder standing in a planar sightline.

    Planar-only, and deliberately so: every body in :data:`OCCLUDERS` is taller
    than the highest camera sample, so the third dimension cannot rescue a
    blocked sightline.  This is the CHEAP predicate used for the "was line of
    sight even available" bookkeeping.  The authoritative visibility measurement
    is always the real MuJoCo ray cast in ``guide_camera``.
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
