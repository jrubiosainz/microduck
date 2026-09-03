#!/usr/bin/env python3
"""The cast: the person being guided, and the six adults who populate the hall.

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

The destination request is a **semantic simulator event** for the same reason.
There is no speech recognition here: at a scripted instant a request naming one
of three destination keys is delivered to the machine, and the machine resolves
it by exact lookup.  What the gate grades is that the duck led to the destination
the request NAMED, out of three that existed, and that it never silently
substituted a different one.

WHY EACH PERSON EXISTS
----------------------
* ``mara`` — the person who asks to be led and then follows.  She is the only
  body whose pose enters the duck's monitoring logic, and she is the only actor
  in this lab whose motion is defined by the DUCK's own accumulated path rather
  than by a route of her own: a follower who walked a pre-drawn line would prove
  nothing about whether the duck led her anywhere.  See ``guide_follower``.
* ``noor``, ``pablo`` — adults crossing the concourse near the middle of the
  route.  Their swept tubes at planning time are what make the crowd term in the
  planner non-vacuous: the planner is required to report cells it refused
  BECAUSE of them, not merely to report that it avoided the walls.
* ``ivan`` — walks the length of the far aisle.  He is the adult the duck comes
  closest to, so the per-tick clearance gate has something real to grade.
* ``sena``, ``omar``, ``tessa`` — background walkers on their own loops.  They
  are what makes "the duck kept positive clearance to every person" a claim about
  a populated hall rather than about an empty one, and what makes the follower's
  visibility percentage a measurement taken in a hall where other bodies pass
  through the sightline.

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
    role: str                  # "follower" | "crowd" | "background"
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


FOLLOWER = Person(
    "mara", (0.878, 0.286, 0.478), 0.985, "follower",
    "the person who asked to be led, and who then follows the duck")

CROWD: tuple[Person, ...] = (
    Person("noor", (0.180, 0.545, 0.855), 1.020, "crowd",
           "crosses the middle of the hall; blocks planner cells at plan time"),
    Person("pablo", (0.945, 0.560, 0.180), 1.045, "crowd",
           "crosses the middle of the hall; blocks planner cells at plan time"),
    Person("ivan", (0.365, 0.784, 0.463), 1.005, "crowd",
           "walks the far aisle; the adult the duck comes closest to"),
)

BACKGROUND: tuple[Person, ...] = (
    Person("sena", (0.855, 0.812, 0.290), 0.955, "background"),
    Person("omar", (0.545, 0.435, 0.855), 1.030, "background"),
    Person("tessa", (0.290, 0.760, 0.745), 0.968, "background"),
)

PEOPLE: tuple[Person, ...] = (FOLLOWER, *CROWD, *BACKGROUND)
BY_NAME: dict[str, Person] = {p.name: p for p in PEOPLE}
ALL_NAMES: tuple[str, ...] = tuple(p.name for p in PEOPLE)
OTHER_NAMES: tuple[str, ...] = tuple(
    p.name for p in PEOPLE if p.name != FOLLOWER.name)
CROWD_NAMES: tuple[str, ...] = tuple(p.name for p in CROWD)
BACKGROUND_NAMES: tuple[str, ...] = tuple(p.name for p in BACKGROUND)

# Planar half-extent used ONLY to inflate a predicted pedestrian position when
# the planner refuses a cell.  It is a PLANNING figure, deliberately generous,
# and it is not what any clearance gate measures: clearance is measured every
# control tick by ``ContactProbe`` against the real geoms at the real pose,
# which accounts for arm swing exactly.
PLANNING_HALF_EXTENT_M = 0.28
