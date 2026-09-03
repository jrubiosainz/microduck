#!/usr/bin/env python3
"""Who or what the head camera looks at in each state, and why.

Kept out of the camera module so the acquisition policy can be unit-tested with
no MuJoCo at all: this is pure state-to-subject logic.

THE GAZE POLICY IS THE DECISION MADE VISIBLE
----------------------------------------------
* While walking with nothing in the way, the duck looks at **the goal**.  That
  is what makes "it was going somewhere" visible in the PiP rather than only in
  the metrics, and it is the same beacon the goal-visibility gate samples.
* From the moment a threat is predicted until the pass is over, it looks at
  **the body it is negotiating with** — through THREAT, the choose states, WAIT
  and PASS.  A robot that decided to wait for a cart while looking somewhere
  else would be guessing, however correct the outcome looked.
* On REPLAN and at the GOAL it looks at **the goal** again, which is also how
  the video shows the duck re-acquiring its destination after each encounter.

There is no gesture and no scan pattern.  Adding one would be shipping
unmeasured behavior; the absence is recorded in
:func:`SlalomCamera.set_gesture` rather than left silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slalom_cast import BY_NAME
from slalom_course import GOAL_BEACON_XY

# The states in which the head is on the negotiated body rather than the goal.
#
# PASS IS DELIBERATELY NOT IN THIS LIST, AND THAT IS A MEASURED FIX.  While the
# duck is executing a pass the body is abeam or behind it, so tracking it swings
# the head 90-150 deg off the direction of travel - and the goal, which the duck
# is still walking toward, leaves the frame entirely.  With PASS included the
# goal was visible in only 54.9 % of the steps where line of sight to it existed,
# because PASS is the single longest state in the run.  A person crossing a yard
# looks at somebody until they have committed to going round them, then looks
# back at where they are going; so does this.
THREAT_GAZE_STATES = ("THREAT", "CHOOSE_LEFT", "CHOOSE_RIGHT", "WAIT")


@dataclass
class Aim:
    """This tick's gaze target and pursuit target."""

    subject: str            # tracked body name, or "" when looking at the goal
    look_at: np.ndarray | None
    target_xy: np.ndarray | None
    kind: str               # "goal" | "corridor" | "none"


def gaze_subject(state: str, threat: str) -> str:
    """The body the head tracks, or "" for the goal.

    Returns the goal whenever there is no threat to watch, so a state that
    should be watching somebody but has nobody cannot silently keep the previous
    subject.
    """
    if state in THREAT_GAZE_STATES and threat:
        return threat
    return ""


def look_point(subject: str, actors) -> np.ndarray:
    """The world point the head aims at.

    A body is aimed at slightly above its mocap origin, scaled by its own
    stature so a shorter body is genuinely looked at lower.  With no subject the
    aim point is the goal beacon, at the height the visibility gate samples.
    """
    if not subject:
        return np.array([GOAL_BEACON_XY[0], GOAL_BEACON_XY[1], 0.34])
    position = np.asarray(actors[subject].pos, dtype=np.float64)
    return np.array([float(position[0]), float(position[1]),
                     BY_NAME[subject].origin_z
                     + 0.12 * BY_NAME[subject].stature])


def role_of(subject: str) -> str:
    """The cast role of a gaze subject, or ``"goal"`` when looking at the band."""
    if not subject:
        return "goal"
    return BY_NAME[subject].kind


def select(state: str, *, duck_xy, threat: str, actors, corridor_target=None,
           goal_target=None) -> Aim:
    """The gaze subject and the pursuit target for this tick.

    The pursuit target comes from the PLANNER's chosen corridor when there is
    one and from the goal otherwise.  A zero-command state gets ``None``, which
    is how its zero is structural rather than remembered.
    """
    from slalom_states import ZERO_COMMAND_STATES

    subject = gaze_subject(state, threat)
    look_at = look_point(subject, actors)

    if state in ZERO_COMMAND_STATES:
        return Aim(subject=subject, look_at=look_at, target_xy=None,
                   kind="none")
    if corridor_target is not None:
        return Aim(subject=subject, look_at=look_at,
                   target_xy=np.asarray(corridor_target, dtype=np.float64),
                   kind="corridor")
    if goal_target is not None:
        return Aim(subject=subject, look_at=look_at,
                   target_xy=np.asarray(goal_target, dtype=np.float64),
                   kind="goal")
    return Aim(subject=subject, look_at=look_at, target_xy=None, kind="none")
