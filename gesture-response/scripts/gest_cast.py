#!/usr/bin/env python3
"""The people in the training area: one instructor, several distracting adults.

WHAT AN "ACTOR" IS HERE, STATED PLAINLY
----------------------------------------
Every person is a **non-colliding kinematic mocap proxy** whose base pose is
written analytically each control tick, and whose ARMS ARE REAL ARTICULATED
BODIES driven through real hinge joints.  That split is the whole point of this
behavior: the thing the duck has to read - an arm held in a particular pose - is
a genuine kinematic chain in the MuJoCo model, so the shoulder, elbow and hand
the camera ray-casts against are the same shoulder, elbow and hand the
classifier measures.  Nothing about the gesture is drawn on top.

The people add no degrees of freedom to the robot's floating base, cannot push
it and cannot be pushed by it, so a step the duck takes is never the result of
somebody nudging it.

IDENTITY IS A SEMANTIC PROXY, AND SAYING SO IS NOT A DISCLAIMER
----------------------------------------------------------------
Which body is the instructor comes from the simulator (MuJoCo body id) rather
than from pixels.  The CAMERA GEOMETRY that decides whether the duck can see
that body - frustum containment plus a real occlusion ray cast - is real, and
the ARM POSE it reads is real world geometry.  So "the duck ignored a gesture
from the wrong person" is a claim about an identity proxy, while "the duck could
actually read the arm" is a claim about physics and optics.  Both are labelled
wherever they surface.

THE DISTRACTORS MUST BE READABLE, OR THE WRONG-PERSON GATE IS VACUOUS
----------------------------------------------------------------------
Three distracting adults perform full, well-formed, camera-visible gestures
during the run - a COME, a STOP and a WAVE, the same templates the instructor
uses.  The acceptance gate does not merely require zero wrong-person
acceptances; it requires that each distractor gesture was **visible, fully
readable and sustained past the confirm window** while it was being ignored.  A
robot that ignored a gesture it could not see would pass a weaker test than the
one this behavior claims to pass.
"""

from __future__ import annotations

from dataclasses import dataclass

# Nominal standing height of an adult, before the per-person stature factor.
BASE_HEIGHT_M = 1.72
# Mocap origin height, scaled by stature.  The person is built upward from it.
BASE_ORIGIN_Z = 0.36
# The five body sample points the camera tests for "is this person visible":
# knees, waist, chest, head, crown.  Scaled by stature.
BASE_SAMPLE_DZ: tuple[float, ...] = (-0.10, 0.02, 0.16, 0.28, 0.34)

# -- the arm, which is the instrument this whole behavior reads --------------
# Segment lengths as a fraction of stature.  The upper arm runs from the
# shoulder pivot to the elbow, the forearm from the elbow to the hand centre.
# ``ARM_SPAN`` is what a fully straight arm measures from the shoulder, and it
# is the normaliser the pose classifier divides by, so "the arm is extended" is
# a ratio rather than a length that would change with stature.
UPPER_ARM_L = 0.200
FOREARM_L = 0.185
HAND_R = 0.036
ARM_SPAN = UPPER_ARM_L + FOREARM_L
SHOULDER_DY = 0.088
SHOULDER_DZ = 0.150


@dataclass(frozen=True)
class Person:
    """One adult in the training area.

    ``role`` is what the SCENARIO says this person is - ``"instructor"`` or
    ``"distractor"``.  The duck never reads it: the acquisition layer resolves
    the instructor through the camera by body identity, and the acceptance gate
    compares that against this field, which is only meaningful because the two
    are computed in different places.
    """

    name: str
    shirt: tuple[float, float, float]
    stature: float
    role: str
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
    def arm_span(self) -> float:
        return ARM_SPAN * self.stature

    @property
    def rgba(self) -> str:
        r, g, b = self.shirt
        return f"{r:.3f} {g:.3f} {b:.3f} 1"


PEOPLE: tuple[Person, ...] = (
    # -- the instructor ----------------------------------------------------
    # A PLAIN DARK-BLUE TRAINING TOP, deliberately.  The instructor is not the
    # brightest person in the area and does not wear anything the duck could
    # key on as "the one in charge"; two distractors are dressed far more
    # conspicuously.  Identity is resolved by the acquisition state machine
    # against a body identity proxy, not by being the most obvious person in
    # frame, and dressing the instructor down is what stops the scenario from
    # smuggling the answer in as a colour.
    Person("mira", (0.145, 0.235, 0.420), 1.010, "instructor",
           "the instructor: stands at the training mark and gives the whole "
           "command sequence"),

    # -- the distracting adults ---------------------------------------------
    # Each of these performs a full, well-formed, camera-visible gesture from
    # the SAME vocabulary the instructor uses.  They are the reason the
    # wrong-person gate is a test rather than a formality.
    Person("teo", (0.945, 0.560, 0.180), 1.035, "distractor",
           "crosses the area behind the instructor and throws a full COME "
           "beckon at the duck"),
    Person("ines", (0.180, 0.700, 0.520), 0.980, "distractor",
           "works the north side and holds a full open-palm STOP"),
    Person("bruno", (0.860, 0.240, 0.300), 1.045, "distractor",
           "walks the south edge and gives a full two-arc WAVE"),
    Person("saskia", (0.620, 0.420, 0.880), 0.965, "distractor",
           "walks the east side throughout with her arms down: the person who "
           "never gestures at all"),
)

BY_NAME: dict[str, Person] = {p.name: p for p in PEOPLE}
ALL_NAMES: tuple[str, ...] = tuple(p.name for p in PEOPLE)
INSTRUCTOR: str = next(p.name for p in PEOPLE if p.role == "instructor")
DISTRACTORS: tuple[str, ...] = tuple(
    p.name for p in PEOPLE if p.role == "distractor")

# Planar half-extent a person is inflated to when a standoff point is PLANNED.
# A planning figure, deliberately generous because an arm can be out sideways.
# It is NOT what any clearance gate measures: clearance is measured every
# control tick by ``ContactProbe`` against the real geoms at the real pose.
PLANNING_HALF_EXTENT_M = 0.30


def role_of(name: str) -> str:
    """What the SCENARIO says this person is.  The duck never reads this."""
    return BY_NAME[name].role


def is_instructor(name: str) -> bool:
    return BY_NAME[name].role == "instructor"
