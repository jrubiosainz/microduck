#!/usr/bin/env python3
"""The cast: the guardian the duck walks beside, and the people who make it hard.

WHAT AN "ACTOR" IS HERE, STATED PLAINLY
----------------------------------------
Every person in this scene is a **non-colliding mocap body** posed analytically
each control tick.  They add no degrees of freedom to the robot's floating base,
they cannot push it, and they cannot be pushed by it, so a step the duck takes
is never the result of somebody nudging it.  Their IDENTITY comes from the
simulator rather than from pixels, which makes them **semantic proxies** for
perceived pedestrians and not the output of an RGB detector.  The camera
geometry that decides whether the duck can SEE them, on the other hand, is real:
frustum containment plus a MuJoCo ray cast through actual scene geometry.

WHY EACH PERSON EXISTS
----------------------
* ``nadia`` — the guardian.  The whole behavior is defined relative to her pose,
  so she is the only body whose position enters the controller.
* ``tomas`` — an oncoming pedestrian who walks *down the lane the duck is using*
  on the north leg.  He is what makes the second side decision a measurement:
  the duck's own side becomes unusable because a moving body is predicted to
  occupy it, not because a wall is there.
* ``iris`` — an oncoming pedestrian who passes on the FAR side of the guardian
  early on.  She exists to prove the blockage test is not "switch whenever
  anybody is nearby": she comes close, the duck measures her predicted distance
  from its own lane, and the lane stays open.  A behavior that switched for her
  would fail the gate that counts side decisions against their measured cause.
* ``rafa``, ``lena`` — background walkers crossing the promenade on their own
  routes.  They are what makes "the duck kept positive clearance to every
  person" a claim about a populated space rather than about an empty one.

STATURE IS GEOMETRY, NOT A LABEL
---------------------------------
``stature`` scales every geom and every camera sample point, so a shorter adult
genuinely has a lower head and the camera's topmost sample of them is genuinely
nearer the ground.  Nothing in this behavior reads a person's appearance — there
is no re-identification here — but the geometry has to be honest anyway, because
it is what the occlusion ray casts hit.
"""

from __future__ import annotations

from dataclasses import dataclass

# Nominal standing height of an adult on this promenade, before the stature
# factor.  Used only to report a height; no gate consumes it.
BASE_HEIGHT_M = 1.72
# Mocap origin height, and the camera's sample points relative to it, both
# scaled by a person's stature.  Five samples: knees, waist, chest, head, crown.
BASE_ORIGIN_Z = 0.36
BASE_SAMPLE_DZ: tuple[float, ...] = (-0.10, 0.02, 0.16, 0.28, 0.34)


@dataclass(frozen=True)
class Person:
    """One adult: appearance, size, and the role they play in the scenario."""

    name: str
    shirt: tuple[float, float, float]
    stature: float
    role: str                  # "guardian" | "oncoming" | "background"
    label: str = ""

    @property
    def height_m(self) -> float:
        return BASE_HEIGHT_M * self.stature

    @property
    def origin_z(self) -> float:
        return BASE_ORIGIN_Z * self.stature

    @property
    def sample_dz(self) -> tuple[float, ...]:
        return tuple(dz * self.stature for dz in BASE_SAMPLE_DZ)

    @property
    def rgba(self) -> str:
        r, g, b = self.shirt
        return f"{r:.3f} {g:.3f} {b:.3f} 1"


GUARDIAN = Person(
    "nadia", (0.145, 0.615, 0.595), 1.000, "guardian",
    "guardian - the person the duck walks beside")

ONCOMING: tuple[Person, ...] = (
    Person("tomas", (0.880, 0.430, 0.170), 1.030, "oncoming",
           "oncoming - walks down the duck's own lane on the north leg"),
    Person("iris", (0.630, 0.200, 0.270), 0.955, "oncoming",
           "oncoming - passes on the guardian's FAR side; must NOT cause a switch"),
)

BACKGROUND: tuple[Person, ...] = (
    Person("rafa", (0.200, 0.255, 0.575), 1.015, "background"),
    Person("lena", (0.905, 0.780, 0.250), 0.945, "background"),
)

PEOPLE: tuple[Person, ...] = (GUARDIAN, *ONCOMING, *BACKGROUND)
BY_NAME: dict[str, Person] = {p.name: p for p in PEOPLE}
ALL_NAMES: tuple[str, ...] = tuple(p.name for p in PEOPLE)
OTHER_NAMES: tuple[str, ...] = tuple(
    p.name for p in PEOPLE if p.name != GUARDIAN.name)
ONCOMING_NAMES: tuple[str, ...] = tuple(p.name for p in ONCOMING)
BACKGROUND_NAMES: tuple[str, ...] = tuple(p.name for p in BACKGROUND)

# Planar half-extent used ONLY to inflate a predicted pedestrian position when
# deciding whether a lane will be usable.  It is a PLANNING figure, deliberately
# generous, and it is not what any clearance gate measures: clearance is
# measured every control tick by ``ContactProbe`` against the real geoms at the
# real pose, which accounts for arm swing exactly.
PLANNING_HALF_EXTENT_M = 0.27
