#!/usr/bin/env python3
"""The cast: the guardian the duck follows, and the seven adults around them.

WHAT AN "ACTOR" IS HERE, STATED PLAINLY
----------------------------------------
Every person in this scene is a **non-colliding mocap body** posed analytically
each control tick.  They add no degrees of freedom to the robot's floating base,
they cannot push it and they cannot be pushed by it, so a step the duck takes is
never the result of somebody nudging it.  Their IDENTITY comes from the
simulator rather than from pixels, which makes them **semantic proxies** for
perceived pedestrians, not the output of an RGB detector.  The camera geometry
that decides whether the duck can SEE them is real: frustum containment plus a
MuJoCo ray cast through actual scene geometry.

WHY EACH PERSON EXISTS
-----------------------
* ``nadia`` — the guardian.  She is the only body whose ORDER matters: the duck
  must never overtake her, must enter both apertures behind her, and must leave
  the cabin after her.  Her walk is scripted, and every claim about the duck's
  position relative to her is measured per tick against her measured pose.
* ``tomas``, ``leila`` — the two people who come OUT through the automatic door
  while the duck is approaching it.  They are the yield the whole first half of
  the behavior is about, and they exit in opposite lanes of the same 0.66 m
  opening, so a duck that tried to enter alongside them would be inside the
  aperture with somebody else.
* ``priya``, ``marek``, ``odile`` — the three occupants riding down who step OUT
  of the lift when it opens at the lobby.  Three rather than the required two,
  so the gate's ">= 2 occupants exited before the duck entered" is satisfied
  with margin and the last-occupant-clear measurement has a real tail.
* ``sami``, ``vera`` — background adults crossing the concourse and the lobby.
  They are what makes "the duck kept positive clearance to every person" a claim
  about a populated building rather than an empty one, and one of them crosses
  the sightline to the guardian so the visibility measurement is taken in a
  space where bodies genuinely pass through the line of sight.

STATURE IS GEOMETRY, NOT A LABEL
---------------------------------
``stature`` scales every geom and every camera sample point, so a shorter adult
genuinely has a lower head and the camera's topmost sample of them is genuinely
nearer the ground.  Nothing here reads a person's appearance — there is no
re-identification in this behavior — but the geometry has to be honest anyway,
because it is what the occlusion ray casts hit.
"""

from __future__ import annotations

from dataclasses import dataclass

# Nominal standing height of an adult, before the stature factor.  Used only to
# report a height; no gate consumes it.
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
    role: str          # "guardian" | "door_exiter" | "occupant" | "background"
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
    "nadia", (0.878, 0.286, 0.478), 0.985, "guardian",
    "the guardian: the duck follows her through both doorways and the cabin")

DOOR_EXITERS: tuple[Person, ...] = (
    Person("tomas", (0.180, 0.545, 0.855), 1.030, "door_exiter",
           "comes out through the automatic door in the south lane"),
    Person("leila", (0.945, 0.560, 0.180), 0.975, "door_exiter",
           "comes out through the automatic door in the north lane"),
)

OCCUPANTS: tuple[Person, ...] = (
    Person("priya", (0.365, 0.784, 0.463), 0.992, "occupant",
           "rides down and steps out first when the lift opens"),
    Person("marek", (0.545, 0.435, 0.855), 1.045, "occupant",
           "steps out second"),
    Person("odile", (0.290, 0.760, 0.745), 0.968, "occupant",
           "steps out last; the duck may not move until she is clear"),
)

BACKGROUND: tuple[Person, ...] = (
    Person("sami", (0.855, 0.812, 0.290), 1.010, "background",
           "crosses the west concourse throughout"),
    Person("vera", (0.380, 0.690, 0.900), 0.960, "background",
           "crosses the lobby behind the lift queue"),
)

PEOPLE: tuple[Person, ...] = (GUARDIAN, *DOOR_EXITERS, *OCCUPANTS, *BACKGROUND)
BY_NAME: dict[str, Person] = {p.name: p for p in PEOPLE}
ALL_NAMES: tuple[str, ...] = tuple(p.name for p in PEOPLE)
OTHER_NAMES: tuple[str, ...] = tuple(
    p.name for p in PEOPLE if p.name != GUARDIAN.name)
DOOR_EXITER_NAMES: tuple[str, ...] = tuple(p.name for p in DOOR_EXITERS)
OCCUPANT_NAMES: tuple[str, ...] = tuple(p.name for p in OCCUPANTS)
BACKGROUND_NAMES: tuple[str, ...] = tuple(p.name for p in BACKGROUND)

# Planar half-extent used ONLY to inflate a predicted pedestrian position in the
# choreography checks.  It is a PLANNING figure, deliberately generous, and it is
# NOT what any clearance gate measures: clearance is measured every control tick
# by ``ContactProbe`` against the real geoms at the real pose, which accounts for
# arm swing exactly.
PLANNING_HALF_EXTENT_M = 0.28
