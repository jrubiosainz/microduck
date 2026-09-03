#!/usr/bin/env python3
"""The plaza: its extent, its fixtures, and the places the behavior names.

One source of truth for every static surface.  Six consumers read these SAME
objects, so a geometry edit moves everything at once: ``tools/build_scene.py``
paints the MuJoCo geometry, ``pps_script`` routes the ward and the moving adults
around them, the acceptance gate measures real surface clearance against the
SAME geoms, the HUD plan view draws them where they are, ``pps_geometry``
refuses an interpose or escape point that would put the duck inside one, and
``pps_sense`` uses them for line-of-sight.

WHY THE MIDDLE OF THE PLAZA IS OPEN
-------------------------------------
This behavior is about a robot CHOOSING where to stand relative to two moving
people.  A cluttered floor would do that choosing for it: every position would
be the only one available rather than the one the duck picked.  So the fixtures
sit around the edges and exist for three measurable reasons rather than for
decoration:

* **Two of them are real occluders.**  ``obs_kiosk_e`` and ``obs_lamp_nw`` stand
  1.30 m and 1.60 m tall against a head camera at about 0.20 m, so a person who
  walks behind one is genuinely hidden and :func:`occluder_between` returns a
  real name on the real run.  The visibility gates are therefore conditioned on
  something that actually happens, and the "line of sight existed" exclusion is
  a measurement rather than a formality.
* **They bound the working area** so the duck's walked arcs have a floor to
  happen on and the clearance gate has surfaces to grade against.
* **``obs_planter_s`` sits where an escape gap could otherwise be chosen**, so
  the gap search has to reject at least one candidate direction on measured
  static clearance rather than only on the two threats.

THE GEOMETRY THE INTERPOSE MANOEUVRE NEEDS
--------------------------------------------
This robot **cannot turn on the spot** - MEASURED at ``vx = 0`` across the whole
command range it manages about a degree per second - so every change of heading
is a WALKED ARC of a MEASURED radius near 0.42 m.  Getting onto the bearing
between two people is therefore a real path problem, not a pivot, and the plaza
is sized so that the arc out of any escort slot to any interpose point fits
inside the floor.  ``tools/check_layout.py`` checks exactly that against the
ward's own hold positions and the MEASURED per-sign turning radii.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -- extent -----------------------------------------------------------------
# Sized against the encounter geometry rather than chosen: an intruder starts
# ALERT_START_M out from a ward hold position, and the ward's route carries her
# across most of the y extent, so the floor must hold both plus the duck's arcs.
FLOOR_HALF = (3.60, 3.00)
WALL_HALF_Z = 1.00
WALL_T = 0.06

# Anything this tall removes a body behind it from the head camera, whose
# optical centre sits at about 0.20 m.
OCCLUDING_HEIGHT_M = 0.55

# Where the duck starts, and which way it faces.  Behind and to the right of the
# ward's own start, so joining the escort slot is a manoeuvre the duck has to
# perform rather than a pose it is handed.
DUCK_START = (0.62, -2.42)
DUCK_START_YAW_DEG = 100.0
MARK_HALF = (0.28, 0.28)


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
    # THE TWO OCCLUDERS.  Tall enough to genuinely hide an adult from a head
    # camera at 0.20 m, and placed on the routes the moving adults walk, so the
    # occlusion predicate fires on the real run instead of being decorative.
    Fixture("obs_kiosk_e", "box", (3.30, 0.72), (0.16, 0.38), 1.30,
            "kioskmat", "newspaper kiosk, east edge: a real occluder"),
    Fixture("obs_lamp_nw", "cylinder", (-1.92, 1.86), (0.09, 0.09), 1.60,
            "columnmat", "lamp column, north-west: a real occluder"),
    # The planter sits inside the arc of one plausible escape gap, so the gap
    # search must reject a direction on MEASURED static clearance rather than
    # only on the two people it is escaping.
    Fixture("obs_planter_s", "cylinder", (-1.06, -1.42), (0.34, 0.34), 0.44,
            "plantmat", "raised planter, south-west: a rejected escape gap"),
    # The rest bound the plaza without screening anything.
    Fixture("obs_bench_w", "box", (-2.72, 0.10), (0.24, 0.70), 0.44,
            "benchmat", "bench, west edge"),
    Fixture("obs_bench_ne", "box", (2.52, 2.55), (0.46, 0.18), 0.42,
            "benchmat", "bench, north-east"),
    Fixture("obs_bollard_se", "cylinder", (2.24, -1.92), (0.10, 0.10), 0.48,
            "bollardmat", "bollard, south-east"),
    Fixture("obs_bollard_sw", "cylinder", (-2.30, -2.28), (0.10, 0.10), 0.48,
            "bollardmat", "bollard, south-west"),
)

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


def occluder_between(eye_xy, target_xy, margin: float = 0.0) -> str | None:
    """Name of the first full-height STATIC occluder in a planar sightline."""
    for fixture in OCCLUDERS:
        if fixture.segment_hits(eye_xy, target_xy, margin):
            return fixture.name
    return None


def inside_area(xy, margin: float = 0.0) -> bool:
    """Is a planar point inside the plaza floor, with ``margin`` to spare?"""
    point = np.asarray(xy, dtype=np.float64)[:2]
    return bool(abs(float(point[0])) <= FLOOR_HALF[0] - margin
                and abs(float(point[1])) <= FLOOR_HALF[1] - margin)


def clamp_inside(xy, margin: float = 0.0) -> np.ndarray:
    """The nearest point to ``xy`` that is inside the plaza with ``margin``.

    Used when a PLANNED point - an interpose station or an escape gap - would
    otherwise sit outside the floor.  Clamping rather than refusing keeps the
    duck's target reachable; the machine's own gates then decide whether the
    clamped point is still a usable station, so a clamp can never smuggle in a
    position that fails the between-ness or clearance tests.
    """
    point = np.asarray(xy, dtype=np.float64)[:2].copy()
    point[0] = float(np.clip(point[0], -(FLOOR_HALF[0] - margin),
                             FLOOR_HALF[0] - margin))
    point[1] = float(np.clip(point[1], -(FLOOR_HALF[1] - margin),
                             FLOOR_HALF[1] - margin))
    return point
