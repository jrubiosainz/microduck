#!/usr/bin/env python3
"""The depot floor: its extent, its static obstacles, and the goal the duck
must actually reach.

One source of truth for every static surface in this behavior.  Four consumers
read these SAME objects, so a geometry edit moves everything at once:

* ``tools/build_scene.py`` paints the MuJoCo geometry from these shapes;
* ``slalom_plan`` prunes candidate corridors against them;
* the acceptance gate measures real surface clearance against the SAME geoms;
* the plan view in the HUD draws them where they are.

WHY THE COURSE IS BROAD AND THE OBSTACLES SIT OFF THE CENTRELINE
------------------------------------------------------------------
This behavior is about choosing between two ways round a MOVING body, so the
floor has to offer both.  A corridor narrow enough to force one answer would
make every "it chose the safe side" claim a fact about the walls.  The hall is
therefore 10.0 x 5.7 m with the static crates, pallets and cones set back to
|y| >= 0.68, which leaves the duck a lane it can genuinely leave on either
hand.

BUT THE STATICS ARE NOT DECORATION, AND ONE OF THEM IS LOAD BEARING.
``obs_cone_mid`` sits at ``y = +0.86`` with a 0.13 m radius, so its south edge
is at ``y = +0.73``.  The planner offers three widths on each hand (0.32, 0.46
and 0.60 m); at the third encounter the 0.60 m and 0.46 m LEFT corridors both
come inside the static margin of that cone and are PRUNED, and the duck threads
the 0.32 m one between the cone and the pedestrian crossing in front of it.
That is what makes the static check non-vacuous on the real run rather than only
under a test mutation.

THE GOAL IS A PLACE, NOT A FLAG IN THE METRICS
------------------------------------------------
:data:`GOAL_XY` is the centre of a painted band on the floor, flanked by two
pylons and backed by a beacon post.  The duck has arrived when its own trunk is
inside that band - a claim about where the body ended up.  The beacon is tall
enough to be sampled by the head camera, so "it could see where it was going" is
measured through the same camera the PiP renders from rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -- extent -----------------------------------------------------------------
# Sized against the CLOCK.  The duck's measured cruise is about 0.13 m/s, so
# every metre of course costs 8 s of video; 7.8 m of net travel plus the lateral
# weaving and one wait lands the rollout near 85 s, inside the 75-100 s target.
FLOOR_HALF = (5.00, 2.85)
WALL_HALF_Z = 0.72
WALL_T = 0.06

# Anything this tall or taller removes a body behind it from the head camera.
OCCLUDING_HEIGHT_M = 0.90

# -- where the journey starts and ends ---------------------------------------
# The duck cannot turn on the spot (MEASURED: at most ~1.6 deg/s at vx=0), so it
# starts already pointing down the course.
DUCK_START_XY = (-4.05, 0.00)
DUCK_START_YAW_DEG = 0.0

GOAL_XY = (3.72, 0.00)
# Half-extent of the painted arrival band.  The duck has REACHED THE GOAL when
# its trunk origin is inside this box, which is a statement about the floor.
GOAL_BAND_HALF = (0.30, 0.55)
# The beacon post behind the band, and the two pylons flanking it.
GOAL_BEACON_XY = (4.12, 0.00)
GOAL_BEACON_R = 0.055
GOAL_BEACON_H = 0.62
GOAL_PYLON_DY = 0.66
GOAL_PYLON_R = 0.075
GOAL_PYLON_H = 0.34
# Camera sample heights on the beacon, for the visibility measurement.
GOAL_SAMPLE_Z: tuple[float, ...] = (0.10, 0.22, 0.34, 0.46, 0.56)

# -- the nominal lane --------------------------------------------------------
# The straight line from the start to the goal.  It is the duck's plan whenever
# nothing is in the way, and the reference every lateral offset is measured
# from.
LANE_Y = 0.00
# Half-width of the band a moving body has to enter before it counts as being
# IN THE WAY at all.  Derived from the duck's conservative planar radius plus a
# body's planning half-extent, so anybody who could not physically conflict is
# never called a threat.
LANE_HALF_W = 0.46


@dataclass(frozen=True)
class Obstacle:
    """One static body.  ``kind`` is ``"box"`` or ``"cylinder"``.

    ``half`` is the planar half-extent for a box and ``(radius, radius)`` for a
    cylinder, so :meth:`distance_to` is exact for both without a special case at
    every call site.
    """

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

    def segment_hits(self, a, b, inflate: float = 0.0, samples: int = 64) -> bool:
        """Does the segment ``a -> b`` come within ``inflate`` of this body?"""
        start = np.asarray(a, dtype=np.float64)[:2]
        end = np.asarray(b, dtype=np.float64)[:2]
        for index in range(samples + 1):
            point = start + (end - start) * (index / samples)
            if self.distance_to(point) < inflate:
                return True
        return False


# Seven static bodies: three crate stacks, a pallet, two traffic cones and a
# stacked pallet at the east end.  Set back from the lane so both hands are
# genuinely available, with ONE exception documented in the module docstring.
STATIC_OBSTACLES: tuple[Obstacle, ...] = (
    Obstacle("obs_crate_nw", "box", (-3.05, 1.32), (0.34, 0.26), 0.58,
             "cratemat", "stacked crates, north-west"),
    Obstacle("obs_cone_sw", "cylinder", (-2.40, -1.12), (0.13, 0.13), 0.42,
             "conemat", "traffic cone, south-west"),
    Obstacle("obs_crate_w", "box", (-1.05, 1.24), (0.30, 0.24), 0.50,
             "cratemat", "crate stack, west"),
    Obstacle("obs_pallet_s", "box", (0.30, -1.18), (0.40, 0.28), 0.26,
             "palletmat", "loaded pallet, south"),
    # THE ONE THAT MATTERS.  South edge at y = +0.73; it prunes the two wider
    # LEFT corridors at the third encounter.  See the module docstring.
    Obstacle("obs_cone_mid", "cylinder", (0.55, 0.86), (0.13, 0.13), 0.42,
             "conemat", "traffic cone beside the mid-course crossing"),
    Obstacle("obs_crate_e", "box", (2.05, -1.22), (0.32, 0.26), 0.62,
             "cratemat", "crate stack, east"),
    Obstacle("obs_cone_ne", "cylinder", (3.05, 1.02), (0.13, 0.13), 0.42,
             "conemat", "traffic cone by the arrival band"),
)

OCCLUDERS: tuple[Obstacle, ...] = tuple(
    o for o in STATIC_OBSTACLES if o.occludes)
BY_NAME: dict[str, Obstacle] = {o.name: o for o in STATIC_OBSTACLES}


def wall_gap(xy) -> float:
    """Planar gap from ``xy`` to the nearest perimeter wall's inner face."""
    point = np.asarray(xy, dtype=np.float64)[:2]
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


def goal_contains(xy, radius: float = 0.0) -> bool:
    """Is the duck's footprint centre inside the painted arrival band?

    ``radius`` shrinks the band, so passing with a positive radius asks the
    stronger question: is the whole footprint inside it.
    """
    point = np.asarray(xy, dtype=np.float64)[:2]
    return bool(
        abs(float(point[0]) - GOAL_XY[0]) <= GOAL_BAND_HALF[0] - radius
        and abs(float(point[1]) - GOAL_XY[1]) <= GOAL_BAND_HALF[1] - radius)


def goal_remaining_m(xy) -> float:
    """Along-course distance still to walk to the band's centre.  Never < 0."""
    return max(0.0, GOAL_XY[0] - float(np.asarray(xy, dtype=np.float64)[0]))


def goal_sample_points() -> list[np.ndarray]:
    """The world points the head camera tests when checking it can see the goal."""
    return [np.array([GOAL_BEACON_XY[0], GOAL_BEACON_XY[1], z])
            for z in GOAL_SAMPLE_Z]


def occluder_between(eye_xy, target_xy, margin: float = 0.0) -> str | None:
    """Name of the first full-height STATIC occluder in a planar sightline.

    Nothing on this course reaches :data:`OCCLUDING_HEIGHT_M`, so this returns
    ``None`` for every query and the tuple :data:`OCCLUDERS` is empty.  That is
    a PROPERTY OF THE SCENE and it is kept as a live computation rather than
    deleted, because the same predicate has to keep working if a taller body is
    ever added - and because a reader deserves to see that the crates really are
    too low to hide anybody rather than having to infer it.
    """
    for obstacle in OCCLUDERS:
        if obstacle.segment_hits(eye_xy, target_xy, margin):
            return obstacle.name
    return None
