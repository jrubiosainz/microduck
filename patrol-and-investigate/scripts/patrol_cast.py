#!/usr/bin/env python3
"""The population: staff who work the floor, the intruder, and the objects.

WHAT AN "ACTOR" IS HERE, STATED PLAINLY
----------------------------------------
Every moving body is a **non-colliding kinematic mocap proxy** posed
analytically each control tick.  They add no degrees of freedom to the robot's
floating base, they cannot push it and cannot be pushed by it, so a step the
duck takes is never the result of somebody nudging it.  Their IDENTITY comes
from the simulator rather than from pixels, which makes them **semantic proxies**
for perceived people and objects, not the output of an RGB detector.  The camera
geometry that decides whether the duck can SEE them is real: frustum containment
plus a MuJoCo ray cast through actual scene geometry.

NOBODY REACTS TO THE DUCK, AND THAT IS THE POINT
--------------------------------------------------
Each body walks a declared route at a declared speed and never stops, swerves or
waits because a robot appeared.  Staff that got out of the way would make every
patrol claim vacuous.  The choreography lives in ``patrol_actors``; the duck
never reads it.

THE THREE THINGS THE DUCK HAS TO TELL APART
---------------------------------------------
This behavior is a CLASSIFICATION problem wearing a patrol's clothes, and the
whole scenario exists to make the classification non-trivial.  Three objects or
people become newly present during the patrol, at three different places, and
the duck must reach three different verdicts:

* ``crate``  - a hard case with a **plain wooden crate** left in the open
  north-east bay, well away from the designated stow pallet, with nobody within
  reach of it.  It stays put.  VERDICT: **suspicious unattended object.**
* ``visitor`` - a person in a **high-visibility vest** who walks into the marked
  restricted annex and stays there.  Nobody else does.  VERDICT: **unauthorised
  person in a restricted zone.**
* ``trolley`` - a **cleaning trolley** that appears near the south wall, looks
  exactly as new and exactly as unexpected as the crate, and is standing ON the
  designated stow pallet with a member of staff beside it.  VERDICT:
  **benign - dismissed.**

The trolley is the distractor, and it is deliberately the case that a
threshold-only detector gets WRONG.  It appears at a similar time, at a similar
range, and it is a similar size.  What separates it is two rules the duck can
check from its own measurements: it stands in a designated stow area, and a
person is with it.

THE CLASSIFICATION RULES, AND WHAT THEY ARE NOT
-------------------------------------------------
The verdict is a decision over MEASURED geometric features - where the thing is,
how long it has been still, whether anybody is with it, whether it is inside a
marked rectangle.  It is a **semantic proxy** for object and behaviour
recognition, not an RGB classifier, and the confidence attached to it is a
RULE-MARGIN proxy - how far the evidence sits past each rule's own threshold -
rather than a learned probability.  Both are labelled as such everywhere they
surface.
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
class Body:
    """One body in the facility: a person or an object.

    ``kind`` drives geometry and colour; ``role`` drives what the scenario
    expects the duck to conclude, and NOTHING the duck can read.  The gate
    compares the duck's verdict against ``role``, which is only meaningful
    because the two are computed in different places.
    """

    name: str
    kind: str          # "staff" | "intruder" | "object" | "trolley"
    shirt: tuple[float, float, float]
    stature: float
    role: str          # "background" | "suspicious" | "intrusion" | "benign"
    label: str = ""

    @property
    def is_person(self) -> bool:
        return self.kind in ("staff", "intruder")

    @property
    def height_m(self) -> float:
        return BASE_HEIGHT_M * self.stature

    @property
    def origin_z(self) -> float:
        return BASE_ORIGIN_Z * self.stature if self.is_person else 0.0

    @property
    def sample_dz(self) -> tuple[float, ...]:
        if self.is_person:
            return tuple(dz * self.stature for dz in BASE_SAMPLE_DZ)
        # An object is sampled over its own box: base, low, middle, high, top.
        height = OBJECT_HEIGHT_M[self.kind]
        return (0.02, 0.25 * height, 0.5 * height, 0.75 * height,
                height - 0.02)

    @property
    def rgba(self) -> str:
        r, g, b = self.shirt
        return f"{r:.3f} {g:.3f} {b:.3f} 1"


# Object geometry, by kind.  Real boxes with real extents, so the clearance
# probe measures against the thing the viewer sees.
OBJECT_HEIGHT_M = {"object": 0.34, "trolley": 0.40}
OBJECT_HALF = {"object": (0.15, 0.13), "trolley": (0.17, 0.14)}
TROLLEY_WHEEL_R = 0.045
TROLLEY_WHEEL_DY = 0.11

# THE THREE ANOMALY ROLES, and what the scenario expects for each.  The
# detector is never told this; the acceptance gate compares the duck's recorded
# verdicts against it.
EXPECTED_VERDICTS: dict[str, str] = {
    "crate": "suspicious",
    "visitor": "intrusion",
    "trolley": "benign",
}
# The order the duck must meet them in, which follows from where they are on the
# circuit rather than from a schedule the duck reads.
ANOMALY_ORDER: tuple[str, ...] = ("crate", "visitor", "trolley")
# The two that must produce a full investigation, and the one that must not.
INVESTIGATED: tuple[str, ...] = ("crate", "visitor")
DISMISSED: tuple[str, ...] = ("trolley",)


BODIES: tuple[Body, ...] = (
    # -- staff, who belong here ------------------------------------------
    Body("rosa", "staff", (0.180, 0.545, 0.855), 1.005, "background",
         "floor staff: works the east and north aisles throughout"),
    Body("emil", "staff", (0.365, 0.784, 0.463), 1.030, "background",
         "floor staff: crosses the south of the facility, and is the person "
         "standing with the cleaning trolley that makes it benign"),
    Body("nadia", "staff", (0.945, 0.560, 0.180), 0.975, "background",
         "floor staff: walks the west side, never enters the restricted zone"),

    # -- the intruder ------------------------------------------------------
    # A HIGH-VISIBILITY VEST, deliberately: this person does not look like a
    # burglar.  What makes them an intrusion is WHERE THEY GO, which is the only
    # thing the duck can measure, and dressing them innocuously is what stops
    # the scenario from smuggling the answer in as a colour.
    Body("visitor", "intruder", (0.980, 0.760, 0.120), 0.992, "intrusion",
         "walks into the marked restricted annex and stays there: the "
         "unauthorised person in a restricted zone"),

    # -- the objects --------------------------------------------------------
    Body("crate", "object", (0.545, 0.400, 0.235), 1.0, "suspicious",
         "a plain crate left in the open north-east bay, away from any "
         "designated stow area, with nobody near it: the suspicious "
         "unattended object"),
    Body("trolley", "trolley", (0.620, 0.640, 0.680), 1.0, "benign",
         "a cleaning trolley left ON the designated stow pallet with a "
         "member of staff beside it: the benign distractor"),
)

BY_NAME: dict[str, Body] = {b.name: b for b in BODIES}
ALL_NAMES: tuple[str, ...] = tuple(b.name for b in BODIES)
PERSON_NAMES: tuple[str, ...] = tuple(b.name for b in BODIES if b.is_person)
OBJECT_NAMES: tuple[str, ...] = tuple(b.name for b in BODIES if not b.is_person)
STAFF_NAMES: tuple[str, ...] = tuple(
    b.name for b in BODIES if b.kind == "staff")
ANOMALY_NAMES: tuple[str, ...] = tuple(
    b.name for b in BODIES if b.role != "background")

# Planar half-extent used ONLY to inflate a body when planning a standoff point.
# It is a PLANNING figure, deliberately generous, and it is NOT what any
# clearance gate measures: clearance is measured every control tick by
# ``ContactProbe`` against the real geoms at the real pose.
PLANNING_HALF_EXTENT_M = {
    "staff": 0.26, "intruder": 0.26, "object": 0.20, "trolley": 0.24,
}


def planning_radius(name: str) -> float:
    """The radius a standoff plan inflates one body to.  Never a gate's figure."""
    return PLANNING_HALF_EXTENT_M[BY_NAME[name].kind]


def role_of(name: str) -> str:
    """What the SCENARIO says this body is.  The duck never reads this."""
    return BY_NAME[name].role
