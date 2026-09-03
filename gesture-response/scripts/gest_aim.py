#!/usr/bin/env python3
"""Where the head looks, and what the world markers show.

Two small modules' worth of concern kept together because both answer the same
question from the same state: what is the duck attending to right now.

THE AIM IS DERIVED FROM THE STATE, NEVER FROM THE SCHEDULE
------------------------------------------------------------
A state with no subject produces a SEARCH sweep; every other state aims at the
locked instructor.  That is what makes "the camera was on her throughout the
command" a consequence of the machine rather than a separate script that could
drift out of step with it.

WHERE THE HEAD AIMS ON A PERSON MATTERS, AND IT IS NOT THE TORSO
------------------------------------------------------------------
The gesture gate requires all three keypoints of the gesturing arm to be in
frame, and a raised hand sits well above the chest.  Aiming at the torso centre
therefore puts the hand near the top edge of the frustum at exactly the range a
gesture is read from.  The aim point is lifted to :data:`GESTURE_AIM_Z`, which
is between the chest and the raised hand, so both the body and the arm sit
inside the frame at once - and the ARM READABILITY percentage in the metrics is
what shows that choice was right.
"""

from __future__ import annotations

import numpy as np

from gest_cast import BY_NAME

# Height above a person's mocap origin the head aims at while reading a gesture.
# Between the chest sample and a raised hand, so both are inside the frustum at
# gesture range.  Scaled by the person's own stature, like every other body
# offset, so it does not silently key on height.
GESTURE_AIM_DZ = 0.30
# What the head aims at during a search sweep, in metres above the floor.
SEARCH_AIM_Z = 0.42


class Aim:
    """What the head should do this tick."""

    __slots__ = ("subject", "look_at", "searching", "kind")

    def __init__(self, subject: str, look_at, searching: bool, kind: str):
        self.subject = subject
        self.look_at = look_at
        self.searching = searching
        self.kind = kind


def gesture_aim_point(name: str, position) -> np.ndarray:
    """The world point the head aims at to read this person's gestures."""
    person = BY_NAME[name]
    planar = np.asarray(position, dtype=np.float64)[:2]
    return np.array([planar[0], planar[1],
                     person.origin_z + GESTURE_AIM_DZ * person.stature])


def select(state: str, *, locked: str, positions: dict) -> Aim:
    """Choose the head's subject and aim point from the machine's state alone.

    READY with no lock is the only state that searches.  Every other state has
    the instructor as its subject, INCLUDING the execute states: a robot that
    stopped watching the person the moment it started obeying them would be
    following an instruction rather than attending to a trainer, and the
    visibility gate is conditioned on exactly that.
    """
    if locked and locked in positions:
        return Aim(locked, gesture_aim_point(locked, positions[locked]),
                   False, "instructor")
    return Aim("", None, True, "search")
