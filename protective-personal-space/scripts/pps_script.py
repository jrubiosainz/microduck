#!/usr/bin/env python3
"""THE SCENARIO: where every person walks, and when.

**Nothing in this module is readable by any decision layer.**
``tests/test_rollout_and_hygiene.py`` parses the import graph with ``ast`` and
fails if ``pps_machine``, ``pps_threat``, ``pps_geometry``, ``pps_control``,
``pps_sense`` or ``pps_camera`` ever imports this module or :mod:`pps_actors`.
The duck measures every person's position through the same per-tick world state
its contact probe uses, and sees them through the real head camera.

WHY THE WARD STOPS SO OFTEN
-----------------------------
The protected person holds still through most of each encounter.  That is not
a convenience: it is what makes the duck's station-keeping a claim about the
duck.  If she were walking while the duck interposed, "the robot got between
them" would be partly a fact about her drifting into the gap, and the range
changes the retreat gate is graded on would be partly hers.  She walks between
encounters, so the escort is a real moving formation, and stands during them,
so every geometric claim is the robot's own doing.

THE ONE PLACE SHE MOVES *AT* THE DUCK
---------------------------------------
Her route contains a deliberate dogleg - she turns out of her line, walks back
toward the slot the duck occupies, and then rejoins.  She is a scripted actor
and does not react to the robot, so the dogleg is aimed at the SLOT rather than
at the duck: the slot is a fixed offset from her own pose, so the point she
walks to is a property of her path alone.  The duck is there because it has
been holding that slot, and the retreat gate is graded on the MEASURED range
between the two bodies, so nothing about the manoeuvre depends on her having
aimed well.

WHY SEVEN ADULTS AND NOT SIX
------------------------------
Six moving adults is the floor the scenario asks for.  Seven is what it takes
for the roles to stay separate: an adult used for the false alarm must not go
on to intrude, or the dismissal becomes a delayed detection, and the two halves
of the squeeze must be people who are not already committed to a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from pps_route import Route

# The ward's own corner radius.  Far tighter than the strangers' 0.40 m because
# her dogleg is a real about-turn, which is what a person does when they change
# their mind and walk back toward their companion.  ``pps_route._build`` raises
# rather than silently leaving a hard vertex, so this number is checked by the
# geometry rather than assumed.
WARD_RADIUS = 0.12

SESSION_S = 190.0


# -- the protected person ----------------------------------------------------
# Corners: north up her line, a dogleg east-south-east back toward the escort
# slot, then a rejoin and on north again.  Hold windows are the encounters.
WARD_ROUTE = Route(
    name="aina",
    corners=(
        (0.62, -1.90),
        (0.55, -0.60),
        (0.34, 0.32),
        (0.86, -0.02),   # the dogleg: back toward the slot the duck holds
        (0.74, 0.96),
        (0.52, 1.72),
    ),
    speed=0.070,
    start_t=0.0,
    radius=WARD_RADIUS,
    hold_windows=(
        (14.0, 32.0),    # encounter 1
        (40.0, 58.0),    # encounter 2
        (62.0, 76.0),    # encounter 3
        (96.0, 110.0),   # encounter 4
        (131.0, 157.0),  # the squeeze
    ),
)


# -- the seven adults who cross the plaza ------------------------------------
# Each route is a filleted polyline at constant speed.  The ones that intrude
# carry a hold window at their closest point, because a person who walks up to
# somebody and then away has to stop somewhere in between - and it is what makes
# the intrusion last long enough to be a manoeuvre rather than a graze.
INTRUDER_ROUTES: dict[str, Route] = {
    # CYCLE 1 - from the EAST, at a walking pace.
    "dario": Route(
        name="dario",
        corners=((3.35, -0.35), (1.72, -1.02), (3.10, -2.25)),
        speed=0.115, start_t=10.0,
        hold_windows=((24.0, 31.0),),
    ),
    # CYCLE 2 - from the WEST, faster.
    "noor": Route(
        name="noor",
        corners=((-2.95, 0.30), (-0.68, -0.30), (-2.60, -1.60)),
        speed=0.135, start_t=36.0,
        hold_windows=((54.0, 61.0),),
    ),
    # THE FALSE ALARM - a straight near pass that never enters the buffer.
    # No hold: he does not stop, he simply goes past, which is exactly what
    # makes him the thing a jumpy protector would over-react to.
    "piet": Route(
        name="piet",
        corners=((2.90, 1.55), (-1.10, 2.10)),
        speed=0.160, start_t=22.0,
    ),
    # CYCLE 3 - from the NORTH-EAST.
    "yara": Route(
        name="yara",
        corners=((2.95, 2.40), (1.22, 0.72), (2.95, -0.60)),
        speed=0.105, start_t=44.0,
        hold_windows=((66.0, 72.0),),
    ),
    # CYCLE 4 - from the SOUTH-WEST.
    "kwame": Route(
        name="kwame",
        corners=((-2.85, -2.15), (-0.16, -0.02), (-2.40, 0.95)),
        speed=0.098, start_t=64.0,
        hold_windows=((93.0, 100.0),),
    ),
    # THE SQUEEZE, half A - arrives from the NORTH.
    "liesl": Route(
        name="liesl",
        corners=((1.85, 2.86), (0.45, 2.04), (-2.10, 2.55)),
        speed=0.125, start_t=115.0,
        hold_windows=((133.0, 153.0),),
    ),
    # THE SQUEEZE, half B - crosses the whole plaza first, on the far side,
    # and only turns in at the end.  He is the person who is simply there for
    # most of the session, which is what a populated plaza actually looks like.
    "tomas": Route(
        name="tomas",
        corners=((-3.10, -2.60), (-2.80, -0.20), (-2.55, 2.30),
                 (2.30, 2.55), (2.90, 0.20), (1.21, 0.09)),
        speed=0.118, start_t=29.0,
        hold_windows=((147.0, 153.0),),
    ),
}

ROUTES: dict[str, Route] = {"aina": WARD_ROUTE, **INTRUDER_ROUTES}


@dataclass(frozen=True)
class Encounter:
    """One episode the SCENARIO intends, for cross-checking only.

    The duck never reads this.  It exists so the metrics can put what the
    scenario meant next to what the duck actually concluded, and so a test can
    fail if the two disagree - which is a far stronger check than either alone.

    ``kind`` is one of ``intrusion``, ``false_alarm``, ``ward_approach`` or
    ``squeeze``.  ``bearing_deg`` is the world bearing from the ward to the
    person at their closest point, and it is what the alternating-bearings gate
    is cross-checked against.
    """

    kind: str
    people: tuple[str, ...]
    from_s: float
    to_s: float
    bearing_deg: float
    note: str = ""


# What the scenario intends, in order.  Every one of these is graded against a
# quantity the duck measured for itself; none of them is an input to the duck.
ENCOUNTERS: tuple[Encounter, ...] = (
    Encounter("intrusion", ("dario",), 12.0, 34.0, -4.9,
              "cycle 1, from the east"),
    Encounter("false_alarm", ("piet",), 28.0, 40.0, 100.0,
              "a near pass that never enters the buffer"),
    Encounter("intrusion", ("noor",), 40.0, 58.0, 176.7,
              "cycle 2, from the west"),
    Encounter("intrusion", ("yara",), 60.0, 76.0, 46.1,
              "cycle 3, from the north-east"),
    Encounter("ward_approach", ("aina",), 77.0, 92.0, 0.0,
              "she leaves her line and walks back at the escort slot"),
    Encounter("intrusion", ("kwame",), 96.0, 112.0, 214.9,
              "cycle 4, from the south-west"),
    Encounter("squeeze", ("liesl", "tomas"), 137.0, 164.0, 0.0,
              "two people converging from nearly opposite bearings"),
)

# The kinds of episode the duck is REQUIRED to produce, in order.  Compared
# against what the machine actually logged; a run that produced them in a
# different order fails rather than passing on a count.
EXPECTED_EPISODES: tuple[str, ...] = (
    "intrusion", "intrusion", "intrusion", "ward_approach", "intrusion",
    "squeeze",
)


def session_end_s() -> float:
    return SESSION_S


def route_of(name: str) -> Route:
    return ROUTES[name]


def encounters_of(kind: str) -> tuple[Encounter, ...]:
    return tuple(e for e in ENCOUNTERS if e.kind == kind)
