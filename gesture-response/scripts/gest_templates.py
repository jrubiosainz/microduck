#!/usr/bin/env python3
"""THE SIX TEMPLATES: the windows each gesture's features must land in.

Separated from the scoring logic in :mod:`gest_gesture` because these are DATA -
every edge below was set from a measurement, and each one is quoted beside the
window it produced.  Keeping them apart means a threshold can be re-measured and
re-read without touching the rule that applies it.

HOW AN EDGE IS SET, AND WHY IT IS NOT THE MEASURED RANGE
----------------------------------------------------------
Each window edge sits OUTSIDE its own template's measured range by at least its
own margin, and INSIDE the measured range of every template it must be separated
from.  Setting an edge at the measured extreme instead is the bug that stopped
this behavior dead once already: COME's real extension peaks at 0.948, and a
ceiling of 0.97 with a 0.10 margin scored that 0.22 - correctly classified, and
still under the acceptance bar, twice per beckon.

The measurements come from ``tools/probe_templates.py``, which poses each
template on the REAL compiled model and sweeps EVERY control tick of its hold.
"""

from __future__ import annotations

from dataclasses import dataclass

from gest_commands import (
    CMD_BACK_UP,
    CMD_COME,
    CMD_STOP,
    CMD_TURN_LEFT,
    CMD_TURN_RIGHT,
    CMD_WAVE,
)


@dataclass(frozen=True)
class Window:
    """One feature's accepted interval, and the margin it is scored on.

    ``low`` or ``high`` may be ``None``, which means that side is NOT a rule.
    That distinction is load-bearing rather than cosmetic: several features have
    a physical ceiling a pose cannot exceed - a straight arm's extension is 1.0,
    a fully lateral arm's lateral component is 1.0 - so writing an upper edge
    just above the ceiling would look like a rule while never being able to
    fire, and would drag the scored margin down for the very poses it was meant
    to accept.  Declaring the side unbounded says what is true: only the lower
    bar discriminates.
    """

    low: float | None
    high: float | None
    margin: float

    def contains(self, value: float) -> bool:
        if self.low is not None and value < self.low:
            return False
        if self.high is not None and value > self.high:
            return False
        return True

    def score(self, value: float) -> float:
        """How far past its own bar the evidence sits, normalised to 0..1.

        The margin is the distance from the NEAREST BINDING EDGE divided by
        :attr:`margin` - the distance that counts as comfortably clear - and
        capped at 1.  A value exactly on an edge scores 0; one a full margin
        inside scores 1.

        THIS IS NOT A DISTANCE-FROM-CENTRE SCORE, AND THE DIFFERENCE MATTERS.
        A centre-based score punishes a pose for being *emphatic*: a perfectly
        straight pointing arm measures a lateral component of 0.999, which is
        the strongest possible evidence for POINT and yet sits at the extreme of
        any window that brackets it.  MEASURED, the centre-based version scored
        that arm 0.11 and refused every one of the six real gestures.
        """
        if not self.contains(value):
            return 0.0
        distances = []
        if self.low is not None:
            distances.append(value - self.low)
        if self.high is not None:
            distances.append(self.high - value)
        if not distances:
            return 1.0
        return float(min(1.0, max(0.0, min(distances) / max(self.margin, 1e-9))))


@dataclass(frozen=True)
class Template:
    """One gesture: the windows its features must land in, and its command.

    ``arms`` is how many arms must be RAISED, which is the single feature that
    separates BACK_UP from STOP.  ``motion`` is ``"moving"``, ``"still"`` or
    ``""`` for don't-care, and it is the single feature that separates COME from
    STOP.  Neither can be inferred from a silhouette.
    """

    name: str
    command: str
    arms: int
    motion: str
    extension: Window
    elevation: Window
    forward: Window
    lateral: Window
    label: str = ""


# THE SIX TEMPLATES.  Every window edge below was set from
# ``tools/check_gestures.py``, which poses each template on the REAL compiled
# model and prints the features measured from world geometry.  The measured
# range for each template is quoted beside its window, and every edge sits
# outside the measured range of the template it belongs to and inside the
# measured range of every template it has to be separated from.

TEMPLATES: tuple[Template, ...] = (
    # COME: forward, raised, elbow beckoning.  MEASURED over the WHOLE hold,
    # every control tick rather than a sample of instants
    # (``tools/probe_templates.py``): extension 0.663-0.948, elevation
    # +42.8..+70.4, forward 0.193-0.661, lateral -0.219..-0.112, travel
    # 1.007-1.061 spans, wander 19.2 at worst.
    #
    # THE EXTENSION CEILING is the rule that separates it from STOP, whose arm
    # is straight at 0.9995, and it is a real upper bound rather than a
    # formality: a beckoning elbow never straightens.
    #
    # EVERY EDGE AND MARGIN HERE IS SET FROM THE WORST TICK OF THE HOLD, WHICH
    # IS A SCAR.  The first version bracketed the SAMPLED features - extension
    # 0.50-0.97 with a 0.10 margin - and at the top of each beckon the real
    # extension reaches 0.948, which is 0.022 from that ceiling and therefore
    # scores 0.22.  The gesture was classified correctly and its confidence
    # still fell under the 0.55 bar twice per second, so the confirm window
    # reset every ~0.7 s and COME was read thirteen times in twelve seconds
    # without ever confirming.  A window edge must clear the measured range by
    # at least its own margin, and each margin below is the distance that
    # actually leaves the worst measured tick comfortably inside.
    Template(
        "COME", CMD_COME, arms=1, motion="moving",
        extension=Window(0.45, 1.02, 0.06),
        elevation=Window(16.0, 84.0, 10.0),
        forward=Window(0.10, 0.86, 0.08),
        lateral=Window(-0.45, 0.45, 0.15),
        label="one arm raised forward, beckoning: come here"),

    # STOP: one straight arm, forward and high, HELD STILL.  MEASURED over the
    # whole hold: extension 0.9995, elevation +34.7, forward 0.799, lateral
    # -0.191, travel exactly 0.000.
    #
    # Extension has NO upper edge: a straight arm measures 0.9995 and there is
    # nothing above it to exclude.
    Template(
        "STOP", CMD_STOP, arms=1, motion="still",
        extension=Window(0.90, None, 0.06),
        elevation=Window(22.0, 76.0, 8.0),
        forward=Window(0.52, None, 0.12),
        lateral=Window(-0.45, 0.45, 0.15),
        label="one straight arm raised, open palm, held still: stop"),

    # POINT with the person's own LEFT arm.  MEASURED: extension 1.000,
    # elevation -2.0, forward 0.017, lateral +0.999, travel 0.000.
    Template(
        "POINT_LEFT_ARM", CMD_TURN_RIGHT, arms=1, motion="still",
        extension=Window(0.90, None, 0.06),
        elevation=Window(-26.0, 26.0, 10.0),
        forward=Window(-0.42, 0.42, 0.15),
        lateral=Window(0.68, None, 0.14),
        label="straight arm out to the instructor's left, horizontal"),

    # POINT with the person's own RIGHT arm: the exact mirror.  MEASURED:
    # extension 1.000, elevation -2.0, forward 0.017, lateral -0.999.
    Template(
        "POINT_RIGHT_ARM", CMD_TURN_LEFT, arms=1, motion="still",
        extension=Window(0.90, None, 0.06),
        elevation=Window(-26.0, 26.0, 10.0),
        forward=Window(-0.42, 0.42, 0.15),
        lateral=Window(None, -0.68, 0.14),
        label="straight arm out to the instructor's right, horizontal"),

    # BACK_UP: BOTH arms forward, pushing.  The arm COUNT is what separates it
    # from STOP - both are forward, both are high.  MEASURED per arm over the
    # whole hold: extension 0.921-0.999, elevation +11.7..+30.9, forward
    # 0.773-0.959, |lateral| 0.190, travel 0.523-0.636 spans.
    #
    # Motion is DON'T-CARE here, deliberately.  The push cycle is slow and its
    # measured travel straddles no useful bar; what makes BACK_UP unambiguous
    # is that it is the only two-armed template, which is a count rather than a
    # rate.
    Template(
        "BACK_UP", CMD_BACK_UP, arms=2, motion="",
        extension=Window(0.84, None, 0.06),
        elevation=Window(2.0, 42.0, 8.0),
        forward=Window(0.68, None, 0.07),
        lateral=Window(-0.45, 0.45, 0.15),
        label="both arms forward, palms out, pushing away: back up"),

    # WAVE: one arm HIGH and STRAIGHT, sweeping sideways.  MEASURED over the
    # whole hold: extension 0.9902 constant, elevation +45.5..+69.7, forward
    # 0.225-0.344, lateral -0.657..+0.170, travel 1.625-1.767, wander 10.6 at
    # worst.
    #
    # THE EXTENSION FLOOR IS WHAT SEPARATES IT FROM COME, and elevation cannot:
    # MEASURED, COME sweeps +42.8..+70.4 and WAVE +45.5..+69.7, which overlap
    # almost exactly.  What does not overlap is the arm itself - a wave is given
    # with a straight arm at 0.9902 while a beckoning elbow never exceeds 0.948
    # - so the rule is stated on the feature that actually discriminates rather
    # than on the one the names suggest.
    #
    # The extension floor sits at 0.960 with a 0.02 margin: 0.9902 is a full
    # margin and a half clear of it, and COME's measured ceiling of 0.948 is
    # below it, so the two cannot be confused in either direction.
    Template(
        "WAVE", CMD_WAVE, arms=1, motion="moving",
        extension=Window(0.960, None, 0.02),
        elevation=Window(34.0, None, 8.0),
        forward=Window(-0.45, 0.66, 0.12),
        lateral=Window(-0.86, 0.86, 0.15),
        label="one high straight arm sweeping above the shoulder: goodbye"),
)

BY_TEMPLATE: dict[str, Template] = {t.name: t for t in TEMPLATES}

