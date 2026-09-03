#!/usr/bin/env python3
"""The gestures as ARM POSES: joint targets over time, and nothing else.

This module is the ANIMATION side of the behavior.  It answers one question -
"where should this person's six arm joints be at this instant" - and it answers
it for the instructor and for a distracting adult in exactly the same way, from
the same templates.  That symmetry is deliberate: a distractor whose COME was
animated more sloppily than the instructor's would make the wrong-person gate
pass for the wrong reason.

THE DUCK NEVER IMPORTS THIS FILE
---------------------------------
``tests/test_rollout_and_hygiene.py`` parses the import graph with ``ast`` and
fails if any decision module reaches it.  The perception side
(:mod:`gest_pose`) measures the arm from WORLD GEOMETRY - the real positions of
the shoulder, elbow and hand bodies MuJoCo computed - so a sign error here shows
up as a gesture that does not classify, never as a gesture that classifies
wrongly.

THE JOINT CONVENTION, STATED ONCE
-----------------------------------
Each arm is three nested bodies, so every rotation is unambiguous:

    shoulder (hinge y, ``flex``) -> upper arm (hinge x, ``abd``)
                                 -> forearm (hinge y, ``elbow``)

In a person's own frame ``+x`` is forward, ``+y`` is their left, ``+z`` is up,
and a relaxed arm hangs along ``-z``.  Rotating ``(0, 0, -1)`` about ``+y`` by
``flex`` gives ``(-sin flex, 0, -cos flex)``, so **negative flex raises the arm
FORWARD** and ``flex = -90 deg`` is a horizontal forward reach.  Rotating about
``+x`` by ``abd`` gives ``(0, sin abd, -cos abd)``, so **positive abd raises the
arm to the person's LEFT** and ``abd = -90 deg`` puts the right arm straight out
to the person's right.

THE ANGLES BELOW WERE TUNED AGAINST MEASURED FEATURES, NOT AGAINST INTENT
---------------------------------------------------------------------------
``tools/check_gestures.py`` poses each template on the real compiled model and
prints the features :mod:`gest_pose` measures from world geometry, plus the
classifier's answer.  Every constant here is the value that made the MEASURED
features land inside its template's window with margin - so "the arm is
horizontal" is 1.5 deg of measured elevation rather than a joint angle that
ought to have produced it.
"""

from __future__ import annotations

import math

# -- gesture names ----------------------------------------------------------
REST = "REST"
COME = "COME"
STOP = "STOP"
POINT_LEFT_ARM = "POINT_L_ARM"     # the person's LEFT arm goes out
POINT_RIGHT_ARM = "POINT_R_ARM"    # the person's RIGHT arm goes out
BACK_UP = "BACK_UP"
WAVE = "WAVE"
PARTIAL = "PARTIAL"

ANIMATED: tuple[str, ...] = (
    COME, STOP, POINT_LEFT_ARM, POINT_RIGHT_ARM, BACK_UP, WAVE, PARTIAL)

# -- envelope ---------------------------------------------------------------
# How long the arm takes to travel from rest into the pose, and back.  Slow
# enough to read in a 50 fps video, fast enough that the sustained hold is the
# majority of every gesture window.  The confirm gate MEASURES the hold, so a
# longer raise does not buy a shorter confirm.
RAISE_S = 0.70
LOWER_S = 0.70
# Below this envelope value the arm is treated as travelling rather than posed.
# Nothing reads it except the metrics, which report the pose-held duration
# separately from the whole gesture window.
POSED_ENVELOPE = 0.92

# -- the idle sway ----------------------------------------------------------
# A person standing with their arms down still moves.  Small enough that no
# template can fire on it - MEASURED extension stays above 0.93 with elevation
# below -70 deg, which is outside every window - and large enough that a body at
# rest does not read as a mannequin.
IDLE_RATE_HZ = 0.23
IDLE_FLEX_DEG = 5.0
IDLE_ABD_DEG = 2.5
IDLE_ELBOW_DEG = 4.0

# -- oscillation rates ------------------------------------------------------
BECKON_HZ = 1.15
PUSH_HZ = 0.95
WAVE_HZ = 1.05


def _deg(value: float) -> float:
    return math.radians(value)


def rest_pose(phase: float) -> dict[str, float]:
    """Arms down, with the small idle sway."""
    sway = math.sin(2.0 * math.pi * IDLE_RATE_HZ * phase)
    counter = math.sin(2.0 * math.pi * IDLE_RATE_HZ * phase + 1.9)
    return {
        "l_flex": _deg(IDLE_FLEX_DEG * sway),
        "l_abd": _deg(6.0 + IDLE_ABD_DEG * counter),
        "l_elbow": _deg(-8.0 + IDLE_ELBOW_DEG * sway),
        "r_flex": _deg(IDLE_FLEX_DEG * counter),
        "r_abd": _deg(-6.0 - IDLE_ABD_DEG * sway),
        "r_elbow": _deg(-8.0 + IDLE_ELBOW_DEG * counter),
    }


def _pose_for(gesture: str, held_s: float) -> dict[str, float]:
    """The fully-posed joint angles for one gesture at ``held_s`` into its hold.

    ``held_s`` drives the oscillation of the three gestures whose signature IS
    an oscillation.  The other four are static poses, which is itself a measured
    signature: :mod:`gest_pose` reports hand motion, and STOP requires it to be
    small while COME requires it to be large.
    """
    if gesture == COME:
        # Forward and up, elbow beckoning: the HAND cycles toward and away from
        # the shoulder, so the signature is an EXTENSION oscillation.
        beckon = math.sin(2.0 * math.pi * BECKON_HZ * held_s)
        return {
            "l_flex": _deg(-4.0), "l_abd": _deg(7.0), "l_elbow": _deg(-12.0),
            "r_flex": _deg(-116.0), "r_abd": _deg(-14.0),
            "r_elbow": _deg(-67.0 + 30.0 * beckon),
        }
    if gesture == STOP:
        # One straight arm, raised forward and up, held perfectly still.  The
        # open palm is a real geom on the hand, so it reads as a palm in the PiP.
        return {
            "l_flex": _deg(-3.0), "l_abd": _deg(6.0), "l_elbow": _deg(-10.0),
            "r_flex": _deg(-124.0), "r_abd": _deg(-11.0), "r_elbow": _deg(-3.0),
        }
    if gesture == POINT_RIGHT_ARM:
        # The RIGHT arm straight out to the person's own right, horizontal.
        return {
            "l_flex": _deg(-3.0), "l_abd": _deg(6.0), "l_elbow": _deg(-10.0),
            "r_flex": _deg(0.0), "r_abd": _deg(-88.0), "r_elbow": _deg(-2.0),
        }
    if gesture == POINT_LEFT_ARM:
        # The LEFT arm straight out to the person's own left, horizontal: the
        # exact mirror of the pose above, which is what makes the two pointing
        # commands a physical mirror rather than two labels.
        return {
            "l_flex": _deg(0.0), "l_abd": _deg(88.0), "l_elbow": _deg(-2.0),
            "r_flex": _deg(-3.0), "r_abd": _deg(-6.0), "r_elbow": _deg(-10.0),
        }
    if gesture == BACK_UP:
        # BOTH arms forward, palms out, pushing away.  Two arms is the feature
        # that separates it from STOP, and it is measured on both arms rather
        # than assumed from the template's name.
        push = math.sin(2.0 * math.pi * PUSH_HZ * held_s)
        elbow = _deg(-26.0 + 20.0 * push)
        return {
            "l_flex": _deg(-99.0), "l_abd": _deg(11.0), "l_elbow": elbow,
            "r_flex": _deg(-99.0), "r_abd": _deg(-11.0), "r_elbow": elbow,
        }
    if gesture == WAVE:
        # One arm HIGH, sweeping sideways above the shoulder.  The signature is
        # a LATERAL oscillation at high elevation, which no other template has.
        swing = math.sin(2.0 * math.pi * WAVE_HZ * held_s)
        return {
            "l_flex": _deg(-4.0), "l_abd": _deg(7.0), "l_elbow": _deg(-12.0),
            "r_flex": _deg(-152.0), "r_abd": _deg(-16.0 + 26.0 * swing),
            "r_elbow": _deg(-16.0),
        }
    if gesture == PARTIAL:
        # THE AMBIGUOUS ONE.  A half-lifted arm with a bent elbow, held still:
        # too bent for STOP, too still for COME, one arm short of BACK_UP, too
        # low for WAVE and nowhere near horizontal enough to point.  It is
        # rejected because no template's MEASURED margin clears the bar, not
        # because it is named PARTIAL.
        return {
            "l_flex": _deg(-4.0), "l_abd": _deg(7.0), "l_elbow": _deg(-12.0),
            "r_flex": _deg(-103.0), "r_abd": _deg(-13.0), "r_elbow": _deg(-78.0),
        }
    raise ValueError(f"unknown gesture template {gesture!r}")


def envelope(elapsed_s: float, span_s: float) -> float:
    """How far into the pose the arm is: 0 at rest, 1 fully posed."""
    if elapsed_s <= 0.0 or elapsed_s >= span_s:
        return 0.0
    if elapsed_s < RAISE_S:
        return elapsed_s / RAISE_S
    remaining = span_s - elapsed_s
    if remaining < LOWER_S:
        return max(remaining / LOWER_S, 0.0)
    return 1.0


def arm_targets(gesture: str, elapsed_s: float, span_s: float,
                phase: float) -> tuple[dict[str, float], float]:
    """Joint angles this instant, and the envelope value they were blended at.

    The pose is a linear blend from the idle sway into the template, so the arm
    genuinely TRAVELS through the intermediate angles.  That matters: the confirm
    gate must not fire while the arm is still on its way up, and it does not,
    because the features measured mid-travel do not match any template's window.
    """
    idle = rest_pose(phase)
    if gesture == REST or gesture not in ANIMATED:
        return idle, 0.0
    weight = envelope(elapsed_s, span_s)
    if weight <= 0.0:
        return idle, 0.0
    held = max(elapsed_s - RAISE_S, 0.0)
    posed = _pose_for(gesture, held)
    return ({key: (1.0 - weight) * idle[key] + weight * posed[key]
             for key in idle}, weight)


JOINT_KEYS: tuple[str, ...] = (
    "l_flex", "l_abd", "l_elbow", "r_flex", "r_abd", "r_elbow")
