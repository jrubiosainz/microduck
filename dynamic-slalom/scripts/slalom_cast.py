#!/usr/bin/env python3
"""The traffic: three pedestrians, two rolling carts and two carried boxes.

WHAT AN "ACTOR" IS HERE, STATED PLAINLY
----------------------------------------
Every moving body in this scene is a **non-colliding kinematic mocap proxy**
posed analytically each control tick.  They add no degrees of freedom to the
robot's floating base, they cannot push it and they cannot be pushed by it, so a
step the duck takes is never the result of somebody nudging it.  Their IDENTITY
comes from the simulator rather than from pixels, which makes them **semantic
proxies** for perceived road users, not the output of an RGB detector.  The
camera geometry that decides whether the duck can SEE them is real: frustum
containment plus a MuJoCo ray cast through actual scene geometry.

NOBODY HERE REACTS TO THE DUCK, AND THAT IS THE WHOLE POINT
------------------------------------------------------------
Each actor walks a declared route at a declared speed and never stops, slows,
swerves or waits because a robot appeared.  Traffic that yielded would make
every slalom claim vacuous: "the duck predicted the gap and took it" would be
true of a duck that walked in a straight line with its eyes shut.  The
choreography lives in this module; the duck never reads it.  It measures every
body's range with the same contact probe it measures everything with, their
positions and velocities through the same per-tick world state, and their
visibility through the real head camera.

WHAT EACH ACTOR IS FOR
-----------------------
Five encounters are staged, and the SIDE the traffic comes from alternates so
the duck cannot pass everything on one hand and still look correct.

THE GEOMETRY THAT DECIDES EACH SIDE, STATED ONCE
--------------------------------------------------
A NORTHBOUND body is in the south before it crosses and in the north after, so
a duck that lets it pass and goes behind it uses the SOUTH - a RIGHT pass.  A
SOUTHBOUND body is the mirror, and behind it is the NORTH - a LEFT pass.  The
planner is never told this; it falls out of scoring predicted occupancy, which
is why the resulting sequence is evidence rather than a setting.

* ``mara``   E1, NORTHBOUND, on foot            -> RIGHT.
* ``tobin``  E2, SOUTHBOUND, pushing a cart     -> LEFT.
* ``ines``   E3, NORTHBOUND, carrying a box     -> RIGHT.
* ``dev`` + ``karl``  E4, BOTH SIDES AT ONCE.  The slow southbound cart is still
  north of the lane while the fast northbound walker is still south of it, so
  neither corridor is predicted safe and the duck WAITS.  ``dev`` crosses first
  and vacates the north, so it resolves LEFT.
* ``noor``   E5, NORTHBOUND, carrying a box     -> RIGHT.

The resulting pass sides are RIGHT, LEFT, RIGHT, LEFT, RIGHT.  Alternation is
therefore a property of the SCENARIO's geometry and the duck's own prediction,
not a rule the planner was told to obey.

STATURE AND LOAD ARE GEOMETRY, NOT LABELS
------------------------------------------
``stature`` scales every geom and every camera sample point, so a shorter adult
genuinely has a lower head.  A cart is a real box on real wheels ahead of its
pusher and a carried box is a real box in front of its carrier's chest, so both
genuinely widen the body the duck has to clear - and the per-tick clearance
probe measures against those geoms rather than against a nominal radius.
"""

from __future__ import annotations

from dataclasses import dataclass

# Nominal standing height of an adult, before the stature factor.  Reported
# only; no gate consumes it.
BASE_HEIGHT_M = 1.72
# Mocap origin height, and the camera's sample points relative to it, both
# scaled by stature.  Five samples: knees, waist, chest, head, crown.
BASE_ORIGIN_Z = 0.36
BASE_SAMPLE_DZ: tuple[float, ...] = (-0.10, 0.02, 0.16, 0.28, 0.34)


@dataclass(frozen=True)
class Actor:
    """One moving body: appearance, size, load, and its role in the scenario."""

    name: str
    shirt: tuple[float, float, float]
    stature: float
    kind: str          # "pedestrian" | "cart" | "box"
    encounter: str     # "E1".."E5", or "" for background
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

    @property
    def carries_cart(self) -> bool:
        return self.kind == "cart"

    @property
    def carries_box(self) -> bool:
        return self.kind == "box"


# The five staged encounters, in the order the duck meets them.  ``side`` is
# where the body comes FROM; the pass side the duck should choose is the
# opposite one, because that is the corridor the body VACATES.
ENCOUNTER_ORDER: tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5")
ENCOUNTER_FROM: dict[str, str] = {
    "E1": "south", "E2": "north", "E3": "south", "E4": "both", "E5": "south",
}
# What the SCENARIO is built to produce.  The planner is never told this; the
# acceptance gate compares the duck's measured decisions against it, which is
# only meaningful because the two are computed in different places.
EXPECTED_PASS_SIDES: tuple[str, ...] = ("right", "left", "right", "left", "right")


ACTORS: tuple[Actor, ...] = (
    Actor("mara", (0.878, 0.286, 0.478), 0.985, "pedestrian", "E1",
          "crosses north-bound from the south side at the first encounter"),
    Actor("tobin", (0.180, 0.545, 0.855), 1.030, "cart", "E2",
          "pushes a rolling cart south-bound across the second encounter"),
    Actor("ines", (0.945, 0.560, 0.180), 0.975, "box", "E3",
          "carries a box north-bound across the third encounter"),
    Actor("dev", (0.545, 0.435, 0.855), 0.992, "cart", "E4",
          "pushes a slow cart south-bound into the fourth encounter"),
    Actor("karl", (0.365, 0.784, 0.463), 1.045, "pedestrian", "E4",
          "walks north-bound into the SAME encounter, fast, so both corridors "
          "are occupied at once and the duck must wait"),
    Actor("noor", (0.290, 0.760, 0.745), 0.968, "box", "E5",
          "carries a box north-bound across the last encounter"),
    Actor("pilar", (0.855, 0.812, 0.290), 1.010, "pedestrian", "",
          "background: walks the length of the depot's north side throughout"),
)

BY_NAME: dict[str, Actor] = {a.name: a for a in ACTORS}
ALL_NAMES: tuple[str, ...] = tuple(a.name for a in ACTORS)
CROSSING_NAMES: tuple[str, ...] = tuple(
    a.name for a in ACTORS if a.encounter)
BACKGROUND_NAMES: tuple[str, ...] = tuple(
    a.name for a in ACTORS if not a.encounter)
BY_ENCOUNTER: dict[str, tuple[str, ...]] = {
    key: tuple(a.name for a in ACTORS if a.encounter == key)
    for key in ENCOUNTER_ORDER
}

# -- load geometry ----------------------------------------------------------
# A cart: a box on two wheels, pushed ahead of its owner.  Offsets are in the
# actor's own body frame, +x being the direction of travel.
CART_AHEAD_M = 0.30
CART_HALF = (0.17, 0.22, 0.11)
CART_Z = 0.20
CART_WHEEL_R = 0.055
CART_WHEEL_DY = 0.20
# A carried box: held in front of the chest, so it widens the carrier and sits
# at exactly the height the duck's camera samples.
BOX_AHEAD_M = 0.15
BOX_HALF = (0.10, 0.15, 0.12)
BOX_DZ = 0.10

# Planar half-extent used ONLY to inflate a PREDICTED actor position in the
# planner.  It is a PLANNING figure, deliberately generous, and it is NOT what
# any clearance gate measures: clearance is measured every control tick by
# ``ContactProbe`` against the real geoms at the real pose, which accounts for
# arm swing, cart and box exactly.
PLANNING_HALF_EXTENT_M = 0.26
# The extra planning half-extent a loaded body carries, because a cart sticks
# out 0.30 m ahead of its pusher and a box 0.15 m.
LOADED_EXTRA_M = {"pedestrian": 0.00, "box": 0.10, "cart": 0.22}


def planning_radius(name: str) -> float:
    """The radius the PLANNER inflates one body to.  Never a gate's figure."""
    actor = BY_NAME[name]
    return PLANNING_HALF_EXTENT_M + LOADED_EXTRA_M[actor.kind]
