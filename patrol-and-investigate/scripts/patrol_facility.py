#!/usr/bin/env python3
"""The facility: its extent, its fixtures, the restricted zone, and the five
named checkpoints the patrol visits in order.

One source of truth for every static surface and every named place.  Five
consumers read these SAME objects, so a geometry edit moves everything at once:
``tools/build_scene.py`` paints the MuJoCo geometry, ``patrol_investigate``
prunes standoff points against them, the acceptance gate measures real surface
clearance against the SAME geoms, the HUD plan view draws them where they are,
and ``patrol_detect`` tests zone membership against the SAME rectangle the scene
paints and the stanchions mark.

WHY THE CIRCUIT IS A HEXAGON ROUND A CENTRAL ISLAND
----------------------------------------------------
This robot **cannot turn on the spot** - MEASURED at ``vx = 0`` across the whole
command range it manages about a degree per second - so a patrol route made of
right-angled corridors would be a route it could not walk.  A hexagonal circuit
round a central rack turns 60 deg at each checkpoint, which the duck carves at
its MEASURED yaw ceiling while walking, and it is what a perimeter patrol round
central shelving physically looks like.

THE CENTRAL RACK IS AN OCCLUDER, AND THAT IS THE POINT
-------------------------------------------------------
``obs_rack_core`` is 0.72 m tall against a head camera at about 0.20 m, so it
genuinely hides what is behind it.  Unlike the sibling behaviors, whose
occlusion predicate could never fire, :func:`occluder_between` returns a real
name on the real run - so the visibility gate is conditioned on a line of sight
that is sometimes actually absent.

THE RESTRICTED ZONE IS A PLACE, NOT A FLAG
--------------------------------------------
:data:`RESTRICTED_ZONE` is a painted rectangle with four stanchions at its
corners.  A person is *in* it when their MEASURED position is inside that
rectangle - the same rectangle the scene paints, the HUD draws and the duck's
detector tests.  The duck must never enter it, which is a claim about where its
trunk went, measured every control tick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -- extent -----------------------------------------------------------------
# Sized against two constraints at once.  The circuit has to be small enough
# that one loop fits the clock at the MEASURED 0.128 m/s cruise, and the
# anomalies have to sit far enough OUTSIDE it that approaching one is a real
# walk rather than a body already standing at its own observation distance.
# ``tools/check_layout.py`` solves both numerically and fails on either.
FLOOR_HALF = (2.95, 2.20)
WALL_HALF_Z = 0.90
WALL_T = 0.06

# Anything this tall removes a body behind it from the head camera, whose
# optical centre sits at about 0.20 m.
OCCLUDING_HEIGHT_M = 0.50

# -- the patrol circuit ------------------------------------------------------
# Circumradius, which for a hexagon is also the side length.  DERIVED from the
# clock: six 0.86 m legs is 5.16 m of route, about 40 s of walking at the
# MEASURED 0.129 m/s cruise, which leaves room in a ~135 s video for five
# checkpoint scans and two full investigations.
LOOP_RADIUS_M = 0.86
DUCK_START_YAW_DEG = 30.0


@dataclass(frozen=True)
class Checkpoint:
    """One named place on the patrol circuit."""

    name: str
    xy: tuple[float, float]
    watch_deg: float
    label: str

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.xy, dtype=np.float64)


def _vertex(degrees: float) -> tuple[float, float]:
    angle = np.radians(degrees)
    return (round(float(LOOP_RADIUS_M * np.cos(angle)), 4),
            round(float(LOOP_RADIUS_M * np.sin(angle)), 4))


# The guard post: a checkpoint in every sense except that it is not one of the
# five numbered ones.  The duck leaves it, visits five in order, returns to it.
HOME = Checkpoint("guard-post", _vertex(-90.0), -90.0,
                  "the guard post the patrol starts and ends at")

# THE FIVE CHECKPOINTS, IN THE ORDER THEY MUST BE VISITED.  The order is a
# REQUIREMENT, and the gate compares the sequence the duck recorded against this
# tuple rather than against a counter.
#
# ``watch_deg`` is the world bearing each post's scan sweeps about: what that
# post OVERLOOKS.  It is facility configuration - where a guard standing there
# would look - and it is deliberately not simply radial, because a real post
# overlooks a bay rather than a compass direction.  It aims the CAMERA; it never
# decides anything, and every detection still has to pass the real frustum and
# the real occlusion ray cast.
CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint("dock-gate", _vertex(-30.0), -50.0,
               "loading dock gate: overlooks the dock apron to the south"),
    Checkpoint("east-aisle", _vertex(30.0), 20.0,
               "east aisle head: overlooks the open north-east bay"),
    Checkpoint("north-bay", _vertex(90.0), 90.0,
               "north bay, the top of the circuit"),
    Checkpoint("server-door", _vertex(150.0), 150.0,
               "server room door: faces the restricted annex"),
    Checkpoint("west-stair", _vertex(210.0), 210.0,
               "west stairwell, the last checkpoint before home"),
)

CHECKPOINT_NAMES: tuple[str, ...] = tuple(c.name for c in CHECKPOINTS)
BY_CHECKPOINT: dict[str, Checkpoint] = {c.name: c for c in CHECKPOINTS}
CIRCUIT: tuple[Checkpoint, ...] = CHECKPOINTS + (HOME,)
HOME_PAD_HALF = (0.30, 0.30)


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


@dataclass(frozen=True)
class Zone:
    """The marked rectangle only authorised staff may stand in."""

    name: str
    center: tuple[float, float]
    half: tuple[float, float]
    label: str

    def contains(self, xy, margin: float = 0.0) -> bool:
        point = np.asarray(xy, dtype=np.float64)[:2]
        return bool(
            abs(float(point[0]) - self.center[0]) <= self.half[0] + margin
            and abs(float(point[1]) - self.center[1]) <= self.half[1] + margin)

    def depth_inside(self, xy) -> float:
        """How far inside the rectangle ``xy`` is.  Negative when outside.

        The rule margin behind the intrusion classifier's confidence proxy:
        somebody well inside a restricted area is a stronger call than somebody
        clipping its edge, and this makes that a number rather than a feeling.
        """
        point = np.asarray(xy, dtype=np.float64)[:2]
        return float(min(
            self.half[0] - abs(float(point[0]) - self.center[0]),
            self.half[1] - abs(float(point[1]) - self.center[1])))

    def corners(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (self.center[0] + sx * self.half[0],
             self.center[1] + sy * self.half[1])
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0))


RESTRICTED_ZONE = Zone("server-annex", (-2.10, 1.15), (0.44, 0.44),
                       "restricted server annex: authorised staff only")

# The duck must keep this far outside the marked rectangle.  DERIVED from its
# own conservative planar radius plus a margin, so "it never entered the
# restricted zone" has room in it rather than being a boundary case.
ZONE_STANDOFF_M = 0.16


def _zone_posts() -> tuple[Fixture, ...]:
    """A stanchion at each corner of the restricted zone.

    Generated FROM the zone rather than listed beside it, so the posts a viewer
    sees and the rectangle the detector tests can never drift apart.
    """
    tags = ("sw", "nw", "se", "ne")
    return tuple(
        Fixture(f"obs_zone_post_{tag}", "cylinder", corner, (0.05, 0.05), 0.62,
                "postmat", f"restricted-zone stanchion, {tag}")
        for tag, corner in zip(tags, RESTRICTED_ZONE.corners()))


FIXTURES: tuple[Fixture, ...] = (
    Fixture("obs_rack_core", "box", (0.00, 0.00), (0.30, 0.17), 0.72,
            "rackmat", "central racking island the circuit runs round"),
    Fixture("obs_shelf_ne", "box", (1.95, 1.35), (0.40, 0.20), 0.80,
            "rackmat", "shelf stack, north-east bay"),
    Fixture("obs_shelf_sw", "box", (-1.85, -1.25), (0.38, 0.22), 0.80,
            "rackmat", "shelf stack, south-west"),
    # The designated stow area, and the trolley stands ON it.  Deliberately
    # BELOW the occluding height, so it screens nothing: its only job is to make
    # an object standing on it legitimate.
    Fixture("obs_pallet_s", "box", (1.15, -1.87), (0.32, 0.20), 0.22,
            "palletmat", "designated stow pallet, south-east"),
    Fixture("obs_column_e", "cylinder", (2.45, -0.55), (0.12, 0.12), 1.30,
            "columnmat", "structural column, east"),
    Fixture("obs_column_w", "cylinder", (-2.40, -0.60), (0.12, 0.12), 1.30,
            "columnmat", "structural column, west"),
    Fixture("obs_bin_n", "cylinder", (0.95, 1.92), (0.15, 0.15), 0.52,
            "binmat", "recycling bin, north-east wall"),
) + _zone_posts()

# Objects standing on one of these are STOWED, and therefore not suspicious
# however long they are left alone.  Named rather than inferred from height, so
# the rule can be read off the facility instead of guessed from geometry.
STOW_AREAS: tuple[str, ...] = ("obs_pallet_s",)
# How far beyond a stow area's footprint an object still counts as on it.
STOW_MARGIN_M = 0.22

OCCLUDERS: tuple[Fixture, ...] = tuple(f for f in FIXTURES if f.occludes)
BY_FIXTURE: dict[str, Fixture] = {f.name: f for f in FIXTURES}


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


def stowed_on(xy) -> str:
    """Name of the designated stow area ``xy`` stands on, or ``""``."""
    for name in STOW_AREAS:
        if BY_FIXTURE[name].distance_to(xy) <= STOW_MARGIN_M:
            return name
    return ""


def home_contains(xy, radius: float = 0.0) -> bool:
    """Is the duck's footprint centre inside the painted guard-post pad?"""
    point = np.asarray(xy, dtype=np.float64)[:2]
    return bool(
        abs(float(point[0]) - HOME.xy[0]) <= HOME_PAD_HALF[0] - radius
        and abs(float(point[1]) - HOME.xy[1]) <= HOME_PAD_HALF[1] - radius)


def occluder_between(eye_xy, target_xy, margin: float = 0.0) -> str | None:
    """Name of the first full-height STATIC occluder in a planar sightline."""
    for fixture in OCCLUDERS:
        if fixture.segment_hits(eye_xy, target_xy, margin):
            return fixture.name
    return None
