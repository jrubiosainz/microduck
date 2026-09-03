#!/usr/bin/env python3
"""Where the head looks in each state, and where the feet are told to go.

Kept out of the camera module so the acquisition policy can be unit-tested with
no MuJoCo at all: this is pure state-to-target logic.

THE GAZE POLICY IS THE BEHAVIOR MADE VISIBLE
----------------------------------------------
* While WALKING a patrol leg, the duck looks **where it is going** - at the
  checkpoint ahead.  That is what a guard robot walking a route does, and it
  keeps the PiP legible as a first-person view of the patrol.
* At a CHECKPOINT_STOP it looks along the checkpoint's own **watch bearing**,
  which is what that post overlooks.
* During a SCAN it **sweeps** that bearing, because turning in place is MEASURED
  to be unavailable and the head is the only thing that can move.
* From DETECT onward it looks at **the thing it is investigating**, through the
  approach, every observation angle and the classification.  A robot that
  approached an unidentified object while looking elsewhere would be a robot
  that had stopped investigating.
* While RETURNING it looks **back at its route**, which is how the video shows
  the duck re-acquiring the patrol rather than wandering back.

There is no gesture and no idle animation.  Adding one would be shipping
unmeasured behavior; the absence is recorded here rather than left silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from patrol_investigate import observation_look_point, target_look_point
from patrol_states import ZERO_COMMAND_STATES

# The states in which the head is on the investigated body rather than on the
# route.  CLASSIFY is included: the verdict is delivered while still looking at
# the thing it is about, which is what makes the PiP legible at the moment the
# HUD prints the classification.
TARGET_GAZE_STATES = ("DETECT", "INVESTIGATE_PLAN", "APPROACH", "OBSERVE",
                      "CLASSIFY")
# The states in which the head sweeps rather than tracks.
SCAN_GAZE_STATES = ("SCAN",)
# Height the duck looks at when it is looking along its route rather than at a
# body: roughly a standing person's chest, so the PiP frames the facility rather
# than the floor.
ROUTE_LOOK_Z = 0.30


@dataclass
class Aim:
    """This tick's gaze target and pursuit target."""

    subject: str            # the body being watched, or "" for the route
    look_at: np.ndarray | None
    target_xy: np.ndarray | None
    kind: str               # "route" | "standoff" | "resume" | "none"
    scanning: bool = False


def select(state: str, *, duck_xy, subject: str, positions, plan,
           standoff_xy=None, observe_angle_deg: float = 0.0,
           watch_deg: float = 0.0) -> Aim:
    """The gaze subject and the pursuit target for this tick.

    A zero-command state gets ``target_xy=None``, which is how its zero is
    structural rather than remembered: the controller has nothing to aim at, so
    it cannot emit a walking command even if it wanted to.
    """
    duck = np.asarray(duck_xy, dtype=np.float64)[:2]
    walking = state not in ZERO_COMMAND_STATES

    # -- where the head looks -------------------------------------------
    if state in SCAN_GAZE_STATES:
        # The sweep point is produced by the camera itself, because it is
        # rate-limited by the same measured head rate that limits tracking.
        return Aim(subject="", look_at=None, target_xy=None, kind="none",
                   scanning=True)

    if state in TARGET_GAZE_STATES and subject and subject in positions:
        if state == "OBSERVE":
            look_at = observation_look_point(
                subject, positions[subject], observe_angle_deg, duck)
        else:
            look_at = target_look_point(subject, positions[subject])
        target_xy = (np.asarray(standoff_xy, dtype=np.float64)
                     if walking and standoff_xy is not None else None)
        return Aim(subject=subject, look_at=look_at, target_xy=target_xy,
                   kind="standoff" if target_xy is not None else "none")

    if state in ("RETURN_TO_PATROL", "RESUME"):
        resume = plan.resume_xy
        point = resume if resume is not None else plan.target_xy
        look_at = np.array([float(point[0]), float(point[1]), ROUTE_LOOK_Z])
        return Aim(subject="", look_at=look_at,
                   target_xy=(np.asarray(point, dtype=np.float64)
                              if walking else None),
                   kind="resume")

    if state == "CHECKPOINT_STOP":
        # Already looking along the watch bearing, so the sweep starts from the
        # direction the post overlooks rather than from the walking heading.
        import math
        look_at = np.array([duck[0] + 2.2 * math.cos(math.radians(watch_deg)),
                            duck[1] + 2.2 * math.sin(math.radians(watch_deg)),
                            ROUTE_LOOK_Z])
        return Aim(subject="", look_at=look_at, target_xy=None, kind="none")

    # PATROL, CLEAR, HOME, DONE: look where the route goes.
    goal = plan.target_xy
    look_at = np.array([float(goal[0]), float(goal[1]), ROUTE_LOOK_Z])
    return Aim(subject="", look_at=look_at,
               target_xy=(np.asarray(goal, dtype=np.float64) if walking
                          else None),
               kind="route" if walking else "none")


def role_of(subject: str) -> str:
    """The cast role of a gaze subject, or ``"route"`` when following the route."""
    if not subject:
        return "route"
    from patrol_cast import BY_NAME
    return BY_NAME[subject].kind
