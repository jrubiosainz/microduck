#!/usr/bin/env python3
"""The session: what each person does, and when.  The duck never reads this.

WHAT IS SCRIPTED, STATED PLAINLY
---------------------------------
The instructor stands on her mark facing down the training area and performs a
declared sequence of gestures at declared times.  Four distracting adults walk
declared routes; three of them perform full, well-formed gestures from the SAME
vocabulary at declared times.  **Nobody reacts to the duck.**  Nobody steps out
of the way, nobody waits, and the instructor does not adjust her timing to what
the robot is doing.

WHY THE DISTRACTOR GESTURES ARE FULL-QUALITY AND DELIBERATELY TIMED
---------------------------------------------------------------------
A distractor who gestured badly, or off-camera, would let the duck pass the
wrong-person gate for the wrong reason.  Each distractor gesture is therefore
placed so that ``tools/check_layout.py`` can verify - on the REAL rollout - that
it was inside the duck's camera frustum, fully readable, and sustained past the
confirm window while it was being ignored.  Two of them fire while the duck is
in OBSERVE, watching the instructor: that is the hardest case, because the
gesture is in frame at the moment the duck is actively looking for one.

THE AMBIGUOUS GESTURE IS THE INSTRUCTOR'S, ON PURPOSE
-------------------------------------------------------
A partial gesture from a distractor would be rejected twice over - wrong person
AND unrecognisable - and would prove neither. The instructor gives it, from the
right place, fully visible and sustained, so the ONLY reason it can be refused
is that no template's measured margin cleared the bar.

THE TIMES ARE SOLVED AGAINST THE DUCK'S OWN MEASURED PROGRESS
---------------------------------------------------------------
Each instructor gesture must begin while the duck is READY and looking at her,
and the duck's schedule is a consequence of its MEASURED walking speed, its
confirm windows and how long each physical action takes.  ``tools/tune_timing.py``
runs the REAL rollout and reports when the duck returns to READY; the times below
are read off that measurement rather than guessed from a cruise speed.
"""

from __future__ import annotations

from dataclasses import dataclass

from gest_arm import (
    BACK_UP,
    COME,
    PARTIAL,
    POINT_LEFT_ARM,
    POINT_RIGHT_ARM,
    STOP,
    WAVE,
)
from gest_route import Route

# Corner radius for the scripted walking routes.  Small, because the training
# area is compact and ``gest_route._build`` raises rather than silently leaving
# a hard vertex when a cutback does not fit.
ACTOR_CORNER_RADIUS = 0.28


@dataclass(frozen=True)
class Cue:
    """One gesture performed by one person, from ``at_s`` for ``span_s``.

    ``expect`` is what the SCENARIO says should come of it - ``"accept"``,
    ``"reject_partial"`` or ``"reject_person"``.  The duck never reads it; the
    acceptance gate compares the duck's own log against it, which is only
    meaningful because the two are computed in different places.
    """

    person: str
    gesture: str
    at_s: float
    span_s: float
    expect: str
    label: str = ""

    @property
    def ends_s(self) -> float:
        return self.at_s + self.span_s


# -- THE INSTRUCTOR'S SEQUENCE ------------------------------------------------
# Six commands in the required order, plus one ambiguous partial that must be
# refused.  Each span is long enough for the arm to travel up (0.70 s), be held
# past the confirm window, and travel back down - so "sustained" is a property
# of the animation as well as of the gate.
#
# THE PARTIAL SITS BETWEEN TWO REAL COMMANDS, not at the end, so a rejection
# cannot be confused with the session finishing.  It is given while the duck is
# READY and watching, from the same mark, with the same visibility as every
# accepted gesture.
INSTRUCTOR_CUES: tuple[Cue, ...] = (
    Cue("mira", COME, 6.0, 6.4, "accept",
        "COME/HERE: the duck must close the range to the safe standoff"),
    Cue("mira", STOP, 17.4, 5.4, "accept",
        "STOP with an open palm, given WHILE THE DUCK IS STILL WALKING"),
    Cue("mira", POINT_RIGHT_ARM, 28.0, 5.6, "accept",
        "points to her own right, which is the duck's LEFT"),
    Cue("mira", PARTIAL, 40.0, 5.2, "reject_partial",
        "a half-lifted bent arm held still: the ambiguous gesture that must "
        "be refused on its own measured margin"),
    Cue("mira", POINT_LEFT_ARM, 49.6, 5.6, "accept",
        "points to her own left, which is the duck's RIGHT"),
    Cue("mira", BACK_UP, 61.2, 5.8, "accept",
        "both arms pushing away: the duck must reverse"),
    Cue("mira", WAVE, 73.0, 6.2, "accept",
        "goodbye: the duck acknowledges and stands down"),
)

# -- THE DISTRACTORS' GESTURES -------------------------------------------------
# Full-quality gestures from the same vocabulary, deliberately placed in the
# duck's field of view.  ``teo`` and ``ines`` fire while the duck is mid-episode
# on an instructor command, which is the hardest case to ignore.
DISTRACTOR_CUES: tuple[Cue, ...] = (
    Cue("teo", COME, 12.6, 5.0, "reject_person",
        "a full COME beckon from behind the instructor, while the duck is "
        "executing her COME"),
    Cue("ines", STOP, 31.4, 5.0, "reject_person",
        "a full open-palm STOP from the north side, while the duck is turning"),
    Cue("bruno", WAVE, 54.0, 5.0, "reject_person",
        "a full two-arc WAVE from the south edge, while the duck is turning "
        "the other way"),
)

CUES: tuple[Cue, ...] = tuple(
    sorted(INSTRUCTOR_CUES + DISTRACTOR_CUES, key=lambda c: c.at_s))

# THE ANIMATION NAME IS NOT THE TEMPLATE NAME, AND CONFLATING THEM WAS A REAL
# BUG.  ``gest_arm`` names the two pointing animations ``POINT_L_ARM`` and
# ``POINT_R_ARM`` (which ARM is raised); ``gest_gesture`` names the templates
# they must satisfy ``POINT_LEFT_ARM`` and ``POINT_RIGHT_ARM``.  Feeding an
# animation name straight into :func:`command_for` returned ``""`` for exactly
# those two, so ``EXPECTED_COMMANDS`` silently became
# ``('COME', 'STOP', '', '', 'BACK_UP', 'WAVE')`` - an acceptance gate that
# asked for two empty commands and would have been satisfied by a duck that
# executed neither turn.  The scenario is the one place that legitimately knows
# both sides, so the bridge lives here and is asserted below.
TEMPLATE_FOR_GESTURE: dict[str, str] = {
    COME: "COME",
    STOP: "STOP",
    POINT_LEFT_ARM: "POINT_LEFT_ARM",
    POINT_RIGHT_ARM: "POINT_RIGHT_ARM",
    BACK_UP: "BACK_UP",
    WAVE: "WAVE",
}

# The exact order of COMMANDS the duck must accept, derived from the cues rather
# than restated, so the two can never drift apart.
from gest_gesture import command_for  # noqa: E402


def command_for_gesture(gesture: str) -> str:
    """The command an ANIMATION name must ultimately produce, via its template."""
    return command_for(TEMPLATE_FOR_GESTURE.get(gesture, ""))


EXPECTED_COMMANDS: tuple[str, ...] = tuple(
    command_for_gesture(cue.gesture) for cue in INSTRUCTOR_CUES
    if cue.expect == "accept")
if not all(EXPECTED_COMMANDS):
    raise RuntimeError(
        f"a scripted instructor gesture maps to no command: {EXPECTED_COMMANDS}")
# Every cue the duck must NOT accept, for whatever reason.
REJECTED_CUES: tuple[Cue, ...] = tuple(
    cue for cue in CUES if cue.expect != "accept")

# -- THE WALKING ROUTES --------------------------------------------------------
# Every distractor keeps moving for the whole session, and each route is solved
# against the fixtures by ``tools/check_layout.py`` rather than drawn.  Two of
# them pass BEHIND a real occluder, which is what makes the occlusion predicate
# fire on the real run.
ROUTES: dict[str, Route] = {
    # Teo crosses behind the instructor, west to east, passing behind the roof
    # post on his way.  He is BEHIND her from the duck's viewpoint, so his COME
    # is in the same part of the frame as hers - the hardest place to put a
    # distraction.
    "teo": Route(
        "teo",
        ((-2.30, 1.95), (-1.05, 2.05), (0.30, 2.02), (1.45, 1.86),
         (2.35, 1.55)),
        0.112, start_t=0.5, radius=ACTOR_CORNER_RADIUS,
        hold_windows=((11.6, 18.2),)),

    # Ines works the north-east, holds her STOP, then carries on east behind
    # the equipment rack.
    "ines": Route(
        "ines",
        ((1.62, 1.72), (1.30, 1.05), (1.55, 0.30), (2.20, -0.35),
         (2.42, -1.20)),
        0.098, start_t=0.8, radius=ACTOR_CORNER_RADIUS,
        hold_windows=((30.4, 37.0),)),

    # Bruno walks the south edge, between the duck and the bench, and waves
    # from there.
    "bruno": Route(
        "bruno",
        ((-2.45, -1.05), (-1.95, -1.85), (-0.95, -2.10), (0.35, -2.12),
         (1.05, -1.60)),
        0.104, start_t=1.2, radius=ACTOR_CORNER_RADIUS,
        hold_windows=((53.0, 59.8),)),

    # Saskia never gestures at all.  She walks the east side for the whole run
    # with her arms down, which is what makes "a person in frame is not a
    # command" a case the run actually contains.
    "saskia": Route(
        "saskia",
        ((2.55, -1.85), (2.62, -0.70), (2.30, 0.95), (1.90, 1.80),
         (0.95, 2.15), (-0.60, 2.18), (-1.90, 1.55), (-2.35, 0.40),
         (-2.45, -0.85)),
        0.086, start_t=0.4, radius=ACTOR_CORNER_RADIUS),
}


def cues_for(person: str) -> tuple[Cue, ...]:
    return tuple(cue for cue in CUES if cue.person == person)


def active_cue(person: str, t: float) -> Cue | None:
    """The cue this person is performing at ``t``, if any."""
    for cue in cues_for(person):
        if cue.at_s <= t < cue.ends_s:
            return cue
    return None


def session_end_s() -> float:
    return max(cue.ends_s for cue in CUES)
