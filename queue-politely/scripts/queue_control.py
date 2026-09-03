#!/usr/bin/env python3
"""The path-following controller: pure pursuit in arc-length space.

THE FOLLOWING LAW, AND WHY IT IS NOT A WAYPOINT CHASE
------------------------------------------------------
The duck tracks the queue PATH, not a point on it.  Each tick it projects its
own trunk onto the path, looks a fixed arc length ahead, and steers at that
lookahead point.  Chasing the target station directly would cut the corner of
the fold, because the straight line to a station on the far side of the bend
leaves the lane entirely.  Pursuing a point that is always ON the path is what
makes the duck go round the bend instead of across it, and ``cross_track`` is
graded every tick to prove it.

TWO THINGS ARE ENFORCED HERE RATHER THAN HOPED FOR
---------------------------------------------------
* **Zero means exactly zero.**  Every stationary state returns
  ``(0.0, 0.0, 0.0)`` with no filter tail, because the acceptance gate tests for
  EXACT zero and a decaying command is still a command.
* **No sub-onset commands, ever.**  Forward gait onset on this scene was
  MEASURED as a cliff between ``vx=0.20`` (0.010 m in 6 s - no gait at all) and
  ``vx=0.22`` (0.409 m).  A command in between looks like motion in the HUD and
  produces none on the floor, so the controller emits either a walking command
  or exact zero.

Yaw authority is asymmetric and was measured per sign; see ``queue_constants``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from policy_runtime import wrap_angle
from queue_constants import (
    CROSS_SETPOINT_LIMIT_DEG,
    KP_CROSS,
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    LOOKAHEAD_M,
    LOOKAHEAD_MIN_M,
    STATIONARY_STATES,
    VX_ADVANCE,
    VX_APPROACH,
    VX_SETTLE,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
)
from queue_path import PATH


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _bendiness(arc: float, probe: float = 0.30) -> float:
    """How sharply the path turns near ``arc``: 0 on a straight, 1 on the fold.

    Measured from the path itself rather than from a hardcoded arc range, so it
    stays correct if the queue's shape is ever changed.  Used by the acceptance
    gate to restrict the corner-cutting measure to the part of the path that
    actually has a corner.
    """
    low = max(arc - probe, 0.0)
    high = min(arc + probe, PATH.length)
    if high - low < 1e-6:
        return 0.0
    turn = abs(wrap_angle(
        PATH.away_heading_at(high) - PATH.away_heading_at(low)))
    # A full 180 deg fold spans about 1.95 m of arc, so a 0.60 m window over it
    # turns roughly 55 deg.  Normalise on that, and saturate.
    return float(min(turn / math.radians(50.0), 1.0))


@dataclass
class QueueController:
    """Produce ``(vx, vy, wz)`` from the state, the duck's pose and the target.

    Every moving state is a pure-pursuit follow of the queue path toward an
    arc-length setpoint; every stationary state is exactly zero.
    """

    ctrl_hz: float = 50.0
    command: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))

    def reset(self) -> None:
        self.command[:] = 0.0

    def raw_command(self, state: str, duck_xy, duck_yaw: float, *,
                    duck_arc: float, target_arc: float | None,
                    approach_target=None) -> tuple[float, float, float]:
        if state in STATIONARY_STATES:
            return (0.0, 0.0, 0.0)

        if state == "APPROACH":
            return self._approach(duck_xy, duck_yaw, approach_target)

        if state in ("JOIN", "ADVANCE"):
            return self._follow_path(duck_xy, duck_yaw, duck_arc, target_arc)

        return (0.0, 0.0, 0.0)

    # -- primitives ------------------------------------------------------
    def _approach(self, duck_xy, duck_yaw: float,
                  target) -> tuple[float, float, float]:
        """Walk in from outside the lane, straight at the entry point.

        The approach is NOT path-following: the duck is off the path entirely,
        and pursuing a path point from outside the lane would curve it in
        through a rope.  It walks at the entry station and the path law takes
        over once it is on the lane.
        """
        if target is None:
            return (0.0, 0.0, 0.0)
        delta = np.asarray(target, dtype=np.float64) - np.asarray(
            duck_xy, dtype=np.float64)
        distance = float(np.linalg.norm(delta))
        if distance <= 0.10:
            return (0.0, 0.0, 0.0)
        desired = math.atan2(float(delta[1]), float(delta[0]))
        vx = VX_APPROACH if distance > 0.55 else VX_ADVANCE
        return (vx, 0.0, self._yaw_to(desired, duck_yaw))

    def _follow_path(self, duck_xy, duck_yaw: float, duck_arc: float,
                     target_arc: float | None) -> tuple[float, float, float]:
        """Pure pursuit along the queue path, toward a DECREASING arc setpoint.

        The lookahead point is always ON the path, which is what takes the duck
        round the fold instead of across it.  It is also clamped never to pass
        the target, so the duck cannot overshoot its standoff and cannot
        overtake the person in front.
        """
        if target_arc is None:
            return (0.0, 0.0, 0.0)
        remaining = duck_arc - target_arc
        if remaining <= 0.02:
            return (0.0, 0.0, 0.0)

        # Speed: full advance until close, then the slow settle command.  Both
        # are above the measured gait onset; nothing between 0 and onset is
        # ever emitted.
        vx = VX_ADVANCE if remaining > 0.26 else VX_SETTLE

        # SHORTEN THE LOOKAHEAD ONLY AS THE TARGET IS APPROACHED, never for
        # curvature: see the LOOKAHEAD_M note for the measurement that rejected
        # curvature scaling.
        lookahead = min(max(LOOKAHEAD_M, LOOKAHEAD_MIN_M), max(remaining, 0.0))
        aim_arc = max(duck_arc - lookahead, target_arc)
        aim = PATH.point_at(aim_arc)
        delta = aim - np.asarray(duck_xy, dtype=np.float64)
        if float(np.linalg.norm(delta)) < 1e-6:
            desired = PATH.travel_heading_at(duck_arc)
        else:
            desired = math.atan2(float(delta[1]), float(delta[0]))

        # Fold the cross-track error into the heading setpoint, so a duck that
        # has drifted off the lane aims back onto it rather than parallel to it.
        _, cross, _ = PATH.project(duck_xy)
        correction = clamp(-KP_CROSS * cross,
                           -math.radians(CROSS_SETPOINT_LIMIT_DEG),
                           math.radians(CROSS_SETPOINT_LIMIT_DEG))
        return (vx, 0.0, self._yaw_to(desired + correction, duck_yaw))

    def _yaw_to(self, desired_yaw: float, duck_yaw: float) -> float:
        """Closed-loop yaw command, with independently measured signs.

        The two directions are NOT symmetric on this policy - at vx=0.34,
        wz=-0.18 turns at R=1.119 m while wz=+0.18 turns at R=3.689 m - so each
        sign carries its own gain, ceiling and dead band.  A command below the
        dead band is emitted as exact zero rather than as a small number that
        does nothing.
        """
        error = wrap_angle(desired_yaw - duck_yaw)
        if error >= 0.0:
            wz = clamp(KP_YAW_LEFT * error, 0.0, WZ_MAX_LEFT)
            return 0.0 if wz < WZ_MIN_LEFT else wz
        wz = -clamp(KP_YAW_RIGHT * abs(error), 0.0, WZ_MAX_RIGHT)
        return 0.0 if abs(wz) < WZ_MIN_RIGHT else wz

    def update(self, state: str, duck_xy, duck_yaw: float, *,
               duck_arc: float, target_arc: float | None,
               approach_target=None) -> np.ndarray:
        target = self.raw_command(
            state, duck_xy, duck_yaw, duck_arc=duck_arc,
            target_arc=target_arc, approach_target=approach_target)
        # Applied directly.  A low-pass filter here would spend its first ticks
        # BELOW the measured gait onset, which is not a gentle start - it is no
        # motion at all, followed by a jump.
        self.command[:] = np.asarray(target, dtype=np.float32)
        return self.command.copy()
