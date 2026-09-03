#!/usr/bin/env python3
"""The controller: aim at the corridor's own line, and an exact zero everywhere
else.

THREE THINGS ARE ENFORCED HERE RATHER THAN HOPED FOR
------------------------------------------------------
* **No sub-onset commands, ever.**  Forward gait onset on this scene was
  MEASURED as a cliff between ``vx = 0.22`` (0.009 m in 6 s — no gait at all)
  and ``vx = 0.24`` (0.524 m).  A command in between appears in the metrics and
  produces nothing on the floor.  **This is the measurement that makes WAITING a
  STATE rather than a speed:** the duck cannot creep past a moving cart, because
  there is no command between zero and a walk.  So it walks, or it holds exactly
  zero.

* **No ``vy``, ever.**  Lateral commands on this policy are a yaw disturbance
  wearing a strafe's clothes.  MEASURED: a 0.34 m sidestep costs 0.64 m of
  course and 5.8 s, executed as turn-out / run / turn-back.  Every corridor
  change in this behavior is therefore a real TURNING PATH, which is why the
  gate can require lateral displacement and a curved path at the same time.

* **The target comes from the PLANNER's chosen corridor**, not from a fixed
  route.  ``slalom_control`` never invents a direction: it is handed the
  corridor line the planner scored and closes the loop onto it.  A state with no
  corridor has no target at all, which is how every zero-command state gets its
  zero structurally rather than by the controller remembering to return one.

THE YAW AXIS IS ASYMMETRIC AND BIASED
--------------------------------------
MEASURED at ``vx = 0.34`` over 3 s: ``wz = -0.10`` gives -6.7 deg/s while
``wz = +0.10`` gives +1.0 deg/s — the policy's own right bias swallows a small
left command almost entirely.  Each sign therefore carries its own gain, ceiling
and dead band, and the left dead band sits above the bias.  MEASURED at
``wz = 0``: 6 s of straight walking drifts -11.4 deg, which is why the heading
loop is closed even when walking a straight lane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from policy_runtime import wrap_angle
from slalom_states import (
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    VX_CAREFUL,
    VX_SETTLE,
    VX_WALK,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
    ZERO_COMMAND_STATES,
)

# Within this distance of the goal the duck eases in, so it stops inside the
# band rather than walking through it.  DERIVED from the MEASURED 0.0088 m coast
# after a stop plus the settle command's own 0.087 m/s: easing from 0.30 m out
# gives about 3.4 s of slow walking, which lands the trunk inside the band's
# 0.30 m half-depth without overshooting it.
SETTLE_REMAINING_M = 0.30
# The duck has ARRIVED when it is this near the band centre.  Smaller than the
# band's own half-extent, so arriving means being properly inside it.
GOAL_ARRIVED_M = 0.26
# Cross-track error below which a corridor counts as reached.  MEASURED against
# the lateral budget: the duck converges to about 0.08 m of a corridor line
# before its own pursuit dead zone stops correcting.
ON_CORRIDOR_M = 0.12


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


@dataclass
class Interlock:
    """This tick's raw reason the duck may not advance.

    Built by the rollout from measured RANGE to the nearest body, never from the
    planner, so it is an INDEPENDENT check rather than an echo of one.  The
    planner reasons about PREDICTED clearance over a horizon; this reasons about
    the distance right now.  A mistake in either alone cannot produce a duck
    walking into somebody.
    """

    blocked: bool = False
    reason: str = ""
    body: str = ""


@dataclass
class SlalomController:
    """Produce ``(vx, vy, wz)`` from the state, the duck's pose and its target."""

    ctrl_hz: float = 50.0
    command: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))
    interlock_holds: int = 0

    def reset(self) -> None:
        self.command[:] = 0.0
        self.interlock_holds = 0

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

        Kept as a named function returning a MEASURED constant so the finding is
        discoverable from the controller rather than only from a comment:
        ``tools/sweep_commands.py --what spin`` produced 0.5-1.4 deg/s across the
        whole command range at ``vx = 0``, which is not a turn.  It is also why
        every pass in this behavior is a turning path rather than a pivot.
        """
        return 0.0

    # -- the command ------------------------------------------------------
    def raw_command(self, state: str, duck_xy, duck_yaw: float, *,
                    target_xy=None, remaining_m: float = 1e9,
                    careful: bool = False,
                    interlock: Interlock | None = None
                    ) -> tuple[float, float, float]:
        """The command for this tick, before it is stored.

        Separated from :meth:`update` so the tests can assert every property on
        hand-built inputs without instantiating anything or touching MuJoCo.

        The zero-command states return a literal ``(0, 0, 0)`` — not a small
        number, not a decayed one.  The acceptance gate checks that literally,
        per tick, and the MEASURED 10 s zero-command drift of 0.0006 m is what
        makes it a claim about the floor.
        """
        if state in ZERO_COMMAND_STATES:
            return (0.0, 0.0, 0.0)

        # THE INDEPENDENT REFUSAL.  Checked before the target is even consulted,
        # so no arithmetic below can talk its way past it.
        if interlock is not None and interlock.blocked:
            return (0.0, 0.0, 0.0)

        if target_xy is None:
            return (0.0, 0.0, 0.0)

        delta = np.asarray(target_xy, dtype=np.float64) - np.asarray(
            duck_xy, dtype=np.float64)
        if float(np.linalg.norm(delta)) <= 0.05:
            return (0.0, 0.0, 0.0)
        desired = math.atan2(float(delta[1]), float(delta[0]))
        wz = self.yaw_to(desired, duck_yaw)

        if remaining_m <= SETTLE_REMAINING_M:
            return (VX_SETTLE, 0.0, wz)
        if careful:
            return (VX_CAREFUL, 0.0, wz)
        return (VX_WALK, 0.0, wz)

    def update(self, state: str, duck_xy, duck_yaw: float, *,
               target_xy=None, remaining_m: float = 1e9,
               careful: bool = False,
               interlock: Interlock | None = None) -> np.ndarray:
        if interlock is not None and interlock.blocked \
                and state not in ZERO_COMMAND_STATES:
            self.interlock_holds += 1
        target = self.raw_command(
            state, duck_xy, duck_yaw, target_xy=target_xy,
            remaining_m=remaining_m, careful=careful, interlock=interlock)
        # Applied directly.  A low-pass filter here would spend its first ticks
        # BELOW the MEASURED gait onset, which is not a gentle start — it is no
        # motion at all, followed by a jump.  It would also make the exact-zero
        # claim false for several ticks after every stop.
        self.command[:] = np.asarray(target, dtype=np.float32)
        return self.command.copy()
