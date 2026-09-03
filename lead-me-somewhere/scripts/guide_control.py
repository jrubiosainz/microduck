#!/usr/bin/env python3
"""The controller: pure pursuit along the planned route, and an exact zero
everywhere else.

TWO THINGS ARE ENFORCED HERE RATHER THAN HOPED FOR
---------------------------------------------------
* **No sub-onset commands, ever.**  Forward gait onset on this scene was
  MEASURED as a cliff between ``vx = 0.20`` (0.010 m in 6 s — no gait at all)
  and ``vx = 0.22`` (0.409 m).  A command in between appears in the metrics and
  produces nothing on the floor.  **This is the measurement that makes waiting a
  STATE rather than a speed:** a guide cannot ease off for a lagging follower,
  because there is no command between zero and a walk.  So it walks, or it holds
  exactly zero.

* **No ``vy``, ever.**  MEASURED on this model: ``vy = ±0.22`` at ``vx = 0``
  produces under 4 mm of lateral motion — no gait at all — while ``vy = -0.28``
  produces 0.255 m sideways together with 51 deg of unwanted yaw.  Lateral
  commands on this policy are a yaw disturbance wearing a strafe's clothes.  The
  duck therefore reaches every point by pointing at it and walking.

THE YAW AXIS IS ASYMMETRIC AND BIASED
--------------------------------------
MEASURED at ``vx = 0.34`` over 3 s: ``wz = -0.10`` gives -6.3 deg/s while
``wz = +0.10`` gives 0.0 deg/s — the policy's own right bias swallows a small
left command completely.  Each sign therefore carries its own gain, ceiling and
dead band, and the left dead band sits above the bias.

WHY THE PURSUIT POINT IS AN ARC LENGTH AND NOT A WAYPOINT
-----------------------------------------------------------
Driving at the next waypoint makes the duck cut every corner: it aims at the
vertex, reaches the fillet, and turns late.  Driving at a point a fixed arc
length ahead of the duck's own PROJECTION onto the route keeps the pursuit point
on the path at all times, so the duck tracks the bends instead of the vertices —
which is what makes "the route it walked has the bends the planner produced" a
statement about the same curve.

THE TURN-IN-PLACE CASE WAS MEASURED AND REMOVED
------------------------------------------------
The first draft of this controller had a ``spin_to`` that turned the body toward
the follower at a spin command copied from a sibling behavior.  The sweep on THIS
scene killed it: at ``vx = 0`` the entire command range produces 0.5-1.6 deg/s,
so squaring up to somebody 130 deg behind would take 80 seconds.  There is no
turn-in-place command in this file at all, and ``guide_states.SPIN_BEST_RATE_DPS``
records the measurement so the absence stays a finding rather than an oversight.

A second draft replaced it with a bounded walking ARC, which worked but meant
the duck was still moving in the state that claims it stopped.  That is gone
too: the follower now walks a little to one side of the guide's line rather than
in its footprints, which keeps her inside the head's MEASURED +/-170 deg reach
without any manoeuvre.  Every monitoring state is therefore a literal
``(0, 0, 0)``.

FACING THE DESTINATION IS ALSO SOLVED BY GEOMETRY, NOT BY TURNING
------------------------------------------------------------------
For the same reason, the duck cannot pirouette to face the lifts once it has
stopped in front of them.  The PLANNER solves it instead: the final approach
waypoint is placed so that the duck's own walking heading as it reaches the
standing point already points at the fixture.  Arriving facing the right way is
then a property of the route rather than a manoeuvre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from guide_states import (
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    VX_LEAD,
    VX_REJOIN,
    VX_SETTLE,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
    ZERO_COMMAND_STATES,
)
from policy_runtime import wrap_angle

# How far off the route's centreline the duck may be before it is treated as
# rejoining rather than following.  Beyond this it uses the faster command, so a
# duck pushed wide by a bend closes back rather than drifting alongside.
REJOIN_ERROR_M = 0.34
# Within this distance of the end of the route the duck eases in, so it stops in
# the arrival band rather than walking through it.
SETTLE_REMAINING_M = 0.55


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


@dataclass
class GuideController:
    """Produce ``(vx, vy, wz)`` from the state, the duck's pose and its target."""

    ctrl_hz: float = 50.0
    command: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))

    def reset(self) -> None:
        self.command[:] = 0.0

    # -- yaw ---------------------------------------------------------------
    def yaw_to(self, desired_yaw: float, duck_yaw: float) -> float:
        """Closed-loop yaw while WALKING, with independently measured signs."""
        error = wrap_angle(desired_yaw - duck_yaw)
        if error >= 0.0:
            wz = clamp(KP_YAW_LEFT * error, 0.0, WZ_MAX_LEFT)
            return 0.0 if wz < WZ_MIN_LEFT else wz
        wz = -clamp(KP_YAW_RIGHT * abs(error), 0.0, WZ_MAX_RIGHT)
        return 0.0 if abs(wz) < WZ_MIN_RIGHT else wz

    def spin_to(self, desired_yaw: float, duck_yaw: float) -> float:
        """There is no turn in place on this model.  Always exactly zero.

        Kept as a named function, returning a measured constant, so that the
        finding is discoverable from the controller rather than only from a
        comment: ``tools/sweep_commands.py --what spin`` produced 0.5-1.6 deg/s
        across the whole command range at ``vx = 0``, which is not a turn.
        ``test_the_controller_never_spins`` pins this against the state machine.
        """
        return 0.0

    # -- the command ------------------------------------------------------
    def raw_command(self, state: str, duck_xy, duck_yaw: float, *,
                    target_xy=None, look_at_yaw: float | None = None,
                    cross_track_m: float = 0.0,
                    route_remaining_m: float = 1e9
                    ) -> tuple[float, float, float]:
        """The command for this tick, before it is stored.

        Separated from :meth:`update` so the tests can assert every property on
        hand-built inputs without instantiating anything or touching MuJoCo.

        The zero-command states return a literal ``(0, 0, 0)`` — not a small
        number, not a decayed one.  The acceptance gate checks that literally,
        per tick, and the MEASURED 10 s zero-command drift of 0.0014 m is what
        makes it a claim about the floor.
        """
        if state in ZERO_COMMAND_STATES:
            return (0.0, 0.0, 0.0)

        if state == "CHECK_FOLLOWER":
            # Stopped, watching her with the head.  A literal zero: see the
            # module docstring for the two drafts this replaced.
            return (0.0, 0.0, 0.0)

        if state == "ARRIVE":
            # Still closing on the standing point, or already there.  Facing the
            # destination is achieved by the route's final heading, not by a
            # turn, so arriving is just the last of the walk.
            if route_remaining_m > 0.0 and target_xy is not None:
                delta = np.asarray(target_xy, dtype=np.float64) - np.asarray(
                    duck_xy, dtype=np.float64)
                if float(np.linalg.norm(delta)) > 0.10:
                    desired = math.atan2(float(delta[1]), float(delta[0]))
                    return (VX_SETTLE, 0.0, self.yaw_to(desired, duck_yaw))
            return (0.0, 0.0, 0.0)

        # LEAD / RESUME: pure pursuit along the route.
        if target_xy is None:
            return (0.0, 0.0, 0.0)
        delta = np.asarray(target_xy, dtype=np.float64) - np.asarray(
            duck_xy, dtype=np.float64)
        if float(np.linalg.norm(delta)) <= 0.06:
            return (0.0, 0.0, 0.0)
        desired = math.atan2(float(delta[1]), float(delta[0]))
        wz = self.yaw_to(desired, duck_yaw)
        if route_remaining_m <= SETTLE_REMAINING_M:
            return (VX_SETTLE, 0.0, wz)
        if abs(cross_track_m) >= REJOIN_ERROR_M:
            return (VX_REJOIN, 0.0, wz)
        return (VX_LEAD, 0.0, wz)

    def update(self, state: str, duck_xy, duck_yaw: float, *,
               target_xy=None, look_at_yaw: float | None = None,
               cross_track_m: float = 0.0,
               route_remaining_m: float = 1e9) -> np.ndarray:
        target = self.raw_command(
            state, duck_xy, duck_yaw, target_xy=target_xy,
            look_at_yaw=look_at_yaw, cross_track_m=cross_track_m,
            route_remaining_m=route_remaining_m)
        # Applied directly.  A low-pass filter here would spend its first ticks
        # BELOW the measured gait onset, which is not a gentle start — it is no
        # motion at all, followed by a jump.  It would also make the exact-zero
        # claim false for several ticks after every stop.
        self.command[:] = np.asarray(target, dtype=np.float32)
        return self.command.copy()
