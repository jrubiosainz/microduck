#!/usr/bin/env python3
"""The people in the plaza: one protected person, and the adults who cross it.

WHAT AN "ACTOR" IS HERE, STATED PLAINLY
----------------------------------------
Every person is a **non-colliding kinematic mocap proxy** whose pose is written
analytically each control tick.  They add no degrees of freedom to the robot's
floating base, cannot push it and cannot be pushed by it, so a step the duck
takes is never the result of somebody nudging it - and, just as importantly,
**the duck cannot stop an intrusion by body-blocking it.**  That asymmetry is
deliberate and it is the honest form of this behavior's central claim: a duck
that "blocked" a person by colliding with them would be demonstrating the
contact solver, not a protective policy.  What is measured instead is whether
the robot **put itself on the bearing between the two, with positive clearance
to both, before the intruder arrived.**

IDENTITY IS A SEMANTIC PROXY, AND SAYING SO IS NOT A DISCLAIMER
----------------------------------------------------------------
Which body is the protected person comes from the simulator (MuJoCo body id)
rather than from pixels.  The CAMERA GEOMETRY that decides whether the duck can
see that body - frustum containment plus a real occlusion ray cast - is real.
So "the duck kept its own person, continuously, and never confused her with one
of seven strangers" is a claim about an identity proxy, while "the duck could
actually see her" is a claim about physics and optics.  Both are labelled
wherever they surface.

WHY SEVEN INTRUDERS AND NOT FOUR
----------------------------------
The scenario requires at least six moving adults approaching from different
bearings, at least four genuine intrusion cycles, one false alarm that must be
dismissed, and one simultaneous two-person squeeze.  Those are not the same
people: an adult who is used for the false alarm must NOT go on to intrude, or
the dismissal becomes a delayed detection.  Seven is the smallest cast that
gives four distinct intrusion cycles from alternating bearings, one clean
near-pass, and a squeeze whose two participants both arrive inside the buffer
within the same window.
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


@dataclass(frozen=True)
class Person:
    """One adult in the plaza.

    ``role`` is what the SCENARIO says this person is - ``"ward"`` or
    ``"intruder"``.  The duck never reads it: the acquisition layer resolves the
    protected person through the camera by body identity, and the acceptance
    gate compares that against this field, which is only meaningful because the
    two are computed in different places.
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
    def rgba(self) -> str:
        r, g, b = self.shirt
        return f"{r:.3f} {g:.3f} {b:.3f} 1"


PEOPLE: tuple[Person, ...] = (
    # -- the protected person ----------------------------------------------
    # A PLAIN GREY COAT, deliberately.  The ward is not the brightest person in
    # the plaza and wears nothing the duck could key on as "the one to protect";
    # four of the intruders are dressed far more conspicuously.  Identity is
    # resolved by the acquisition state machine against a body identity proxy,
    # not by being the most obvious person in frame, and dressing the ward down
    # is what stops the scenario from smuggling the answer in as a colour.
    Person("aina", (0.470, 0.490, 0.530), 1.000, "ward",
           "the protected person: walks her own line across the plaza and "
           "never reacts to the duck"),

    # -- the intruders, in the order they first matter ------------------------
    Person("dario", (0.960, 0.520, 0.140), 1.040, "intruder",
           "cycle 1: closes from the EAST at a walking pace"),
    Person("noor", (0.180, 0.720, 0.560), 0.975, "intruder",
           "cycle 2: closes from the WEST, faster"),
    Person("piet", (0.880, 0.230, 0.290), 1.050, "intruder",
           "the FALSE ALARM: passes ahead on a near-miss line and never "
           "enters the buffer"),
    Person("yara", (0.560, 0.400, 0.900), 0.965, "intruder",
           "cycle 3: closes from the NORTH-EAST"),
    Person("kwame", (0.930, 0.800, 0.220), 1.055, "intruder",
           "cycle 4 and the SQUEEZE: closes from the SOUTH-WEST, then returns "
           "as one half of the simultaneous pinch"),
    Person("liesl", (0.240, 0.560, 0.940), 0.985, "intruder",
           "the other half of the SQUEEZE: arrives from the opposite side "
           "within the same window"),
    Person("tomas", (0.760, 0.360, 0.720), 1.020, "intruder",
           "crosses the plaza throughout on a line that never threatens: the "
           "person who is simply there"),
)

BY_NAME: dict[str, Person] = {p.name: p for p in PEOPLE}
ALL_NAMES: tuple[str, ...] = tuple(p.name for p in PEOPLE)
WARD: str = next(p.name for p in PEOPLE if p.role == "ward")
INTRUDERS: tuple[str, ...] = tuple(
    p.name for p in PEOPLE if p.role == "intruder")

# Planar half-extent a person is inflated to when a standoff point is PLANNED.
# A planning figure, deliberately generous because an adult's stride and swinging
# arms sweep wider than their torso.  It is NOT what any clearance gate measures:
# clearance is measured every control tick by ``ContactProbe`` against the real
# geoms at the real pose.
PLANNING_HALF_EXTENT_M = 0.30


def role_of(name: str) -> str:
    """What the SCENARIO says this person is.  The duck never reads this."""
    return BY_NAME[name].role


def is_ward(name: str) -> bool:
    return BY_NAME[name].role == "ward"
