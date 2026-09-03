#!/usr/bin/env python3
"""Palette, state colours and plain-English captions for the PPS overlay.

The drawing PRIMITIVES - fonts, panels, bars, ellipsising, wrapping - are shared
with the sibling behaviors and live in :mod:`hud_style`.  What is specific to
this behavior is the colour code and the vocabulary, and both are here so a
reader can see the whole visual language in one file.

THE COLOUR CODE IS THE ARGUMENT
--------------------------------
This behavior repeatedly answers three questions: **who am I protecting**,
**who is coming at her**, and **where did I decide to stand**.  The strongest
colours therefore belong to those three answers and to nothing else:

* ``WARD`` cyan       - the protected person.  Exactly one body ever wears it.
* ``THREAT`` red      - the person currently being acted on.
* ``SECOND`` amber    - the second half of a squeeze, which is the only time
  two people are being reasoned about at once.
* ``WATCHED`` slate   - a person being predicted but NOT acted on.  Deliberately
  dull, because "seen and correctly ignored" is half of what this behavior
  claims and it should read as calm rather than as an alert.
* ``STATION`` green   - the geometric station the duck chose: the escort slot,
  the interpose point, the escape gap.
* ``ZERO`` violet     - the command register at an EXACT zero.  It has its own
  colour because HOLD_BUFFER standing perfectly still on the line between two
  people is the single hardest claim here to believe from a description.

WHY THE STATE COLOURS GROUP RATHER THAN CYCLE
-----------------------------------------------
Thirteen states with thirteen unrelated hues is a legend, not a picture.  They
are grouped instead, so the 190 s timeline reads as a shape before it reads as
labels: calm escort/monitor blues, a red-through-violet protective arc for the
intrusion cycle, warm oranges for yielding to the protected person herself, and
yellow for the squeeze branch.
"""

from __future__ import annotations

# -- the three roles ---------------------------------------------------------
WARD = (86, 214, 226)
THREAT = (240, 96, 84)
SECOND = (246, 178, 72)
WATCHED = (126, 140, 160)
STATION = (86, 214, 138)
ZERO = (176, 148, 246)

# -- neutrals, shared with the sibling palette so panels look the same -------
PANEL = (16, 18, 24)
INK = (232, 236, 244)
DIM = (150, 158, 172)
GOOD = (78, 210, 130)
BAD = (238, 96, 78)
WARN = (246, 190, 86)
ACCENT = (96, 176, 246)
GRID = (58, 66, 82)
OUTLINE = (70, 78, 94)
TRAIL = (128, 108, 196)
BUFFER = (64, 156, 196)
HEADING = (92, 200, 250)

# -- the thirteen states -----------------------------------------------------
# Grouped by what the duck is DOING, not by state index.  See the module
# docstring: the timeline has to be readable as a shape.
STATE_COLORS: dict[str, tuple[int, int, int]] = {
    # calm: forming and holding the escort
    "ESCORT": (96, 176, 246),
    "MONITOR": (72, 128, 190),
    "RETURN_ESCORT": (110, 196, 210),
    # the protective arc against one stranger
    "PREDICT_INTRUSION": (246, 190, 86),
    "INTERPOSE": (240, 96, 84),
    "HOLD_BUFFER": (176, 148, 246),
    "THREAT_CLEAR": (86, 214, 138),
    # yielding to the protected person herself
    "PERSON_APPROACH": (250, 140, 70),
    "RETREAT": (232, 108, 40),
    "RECOVER": (150, 200, 150),
    # the simultaneous squeeze
    "MULTI_THREAT": (250, 224, 92),
    "ESCAPE_GAP": (214, 196, 60),
    # done
    "DONE": (150, 158, 172),
}

# One line of plain English per state.  A viewer who has never read the README
# should still be able to say what the robot is doing and why.
STATE_CAPTION: dict[str, str] = {
    "ESCORT": "joining the escort slot beside and behind Aina",
    "MONITOR": "escorting: watching everyone, acting on nobody",
    "PREDICT_INTRUSION": "predicted an intrusion; command cut to an exact zero",
    "INTERPOSE": "walking to the station BETWEEN Aina and the intruder",
    "HOLD_BUFFER": "on station: standing still on the line between them",
    "THREAT_CLEAR": "the intruder left the buffer; holding before standing down",
    "RETURN_ESCORT": "returning to the escort slot",
    "PERSON_APPROACH": "Aina is walking at the duck: yielding, not blocking",
    "RETREAT": "reversing along its own heading to give her room",
    "RECOVER": "recovering the escort slot after the yield",
    "MULTI_THREAT": "two people converging at once: a station covers neither",
    "ESCAPE_GAP": "leaving the pinch through the measured safe gap",
    "DONE": "session complete, escort restored, standing down",
}

# The five phases the timeline legend groups the thirteen states into, so the
# strip can be read before any individual label is.
PHASE_LEGEND: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("escort", (96, 176, 246)),
    ("predict/interpose", (240, 96, 84)),
    ("hold (exact zero)", (176, 148, 246)),
    ("yield to Aina", (232, 108, 40)),
    ("squeeze/escape", (250, 224, 92)),
)


def state_color(state: str) -> tuple[int, int, int]:
    return STATE_COLORS.get(state, DIM)


def state_caption(state: str) -> str:
    return STATE_CAPTION.get(state, "")


def person_color(name: str, ward: str, selected: str | None,
                 secondary: str | None) -> tuple[int, int, int]:
    """Colour a person by their CURRENT role in the duck's own reasoning.

    Not by cast identity: a person is amber because the duck is treating them
    as the second half of a squeeze right now, not because the scenario cast
    them that way.  That distinction is the point of the plan view.
    """
    if name == ward:
        return WARD
    if selected is not None and name == selected:
        return THREAT
    if secondary is not None and name == secondary:
        return SECOND
    return WATCHED
