#!/usr/bin/env python3
"""The controller: aim at the route, and an exact zero everywhere else.

THREE THINGS ARE ENFORCED HERE RATHER THAN HOPED FOR
------------------------------------------------------
* **No sub-onset commands, ever.**  Forward gait onset on this scene was
  MEASURED as a cliff between ``vx = 0.22`` (0.009 m in 6 s - no gait at all)
  and ``vx = 0.24`` (0.508 m).  A command in between appears in the metrics and
  produces nothing on the floor.  **This is the measurement that makes a
  CHECKPOINT STOP a STATE rather than a speed:** the duck cannot creep past its
  own checkpoint looking attentive, because there is no command between zero and
  a walk.  So it walks, or it holds exactly zero.

* **No ``vy``, ever.**  Lateral commands on this policy are a yaw disturbance
  wearing a strafe's clothes, so every change of direction is a real turning
  path.  The gate requires ``max |vy| == 0.0`` over every control tick, measured
  per tick rather than read off this source.

* **The target comes from the MACHINE's state**, never from a schedule.  A state
  with no target has no target at all, which is how every zero-command state
  gets its zero structurally rather than by the controller remembering to return
  one.

THE YAW AXIS IS ASYMMETRIC AND BIASED RIGHT
---------------------------------------------
MEASURED at ``vx = 0.34`` over 3 s: ``wz = -0.10`` gives -8.7 deg/s while
``wz = +0.10`` gives only +0.7 deg/s - the policy's own right bias very nearly
swallows a small left command.  Each sign therefore carries its own gain,
ceiling and dead band, and the left dead band sits above the bias.  MEASURED at
``wz = 0``: 6 s of straight walking at ``vx = 0.34`` drifts **-16.7 deg**, which
is why the heading loop is closed even to walk a straight leg - and why a
counter-clockwise circuit, whose every corner is a LEFT turn into the weak sign,
is a real test of the controller rather than of the policy's own drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from policy_runtime import wrap_angle
from patrol_states import (
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    SETTLE_REMAINING_M,
    VX_APPROACH,
    VX_PATROL,
    VX_SETTLE,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
    ZERO_COMMAND_STATES,
)

# Below this distance to the target the duck is close enough that chasing the
# bearing to it produces only yaw chatter, so the command goes to zero and the
# machine's own arrival test takes over.
ARRIVED_M = 0.05


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


@dataclass
class Interlock:
    """This tick's raw reason the duck may not advance.

    Built by the rollout from MEASURED range to the nearest body and from the
    duck's own distance to the restricted zone - never from the detector or the
    standoff planner - so it is an INDEPENDENT check rather than an echo of one.

    Two different mistakes are therefore needed to produce a duck that walks
    into somebody or into the marked zone: the planner would have to offer a bad
    standoff AND this would have to fail to refuse it.
    """

    blocked: bool = False
    reason: str = ""
    body: str = ""


@dataclass
class PatrolController:
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
        ``tools/sweep_commands.py --what spin`` produced 0.5-1.6 deg/s across
        the whole command range at ``vx = 0``, which is not a turn.  It is why
        every checkpoint scan and every observation is done with the HEAD.
        """
        return 0.0

    # -- the command ------------------------------------------------------
    def raw_command(self, state: str, duck_xy, duck_yaw: float, *,
                    target_xy=None, remaining_m: float = 1e9,
                    approach: bool = False,
                    interlock: Interlock | None = None
                    ) -> tuple[float, float, float]:
        """The command for this tick, before it is stored.

        Separated from :meth:`update` so the tests can assert every property on
        hand-built inputs without instantiating anything or touching MuJoCo.

        The zero-command states return a literal ``(0, 0, 0)`` - not a small
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
        if float(np.linalg.norm(delta)) <= ARRIVED_M:
            return (0.0, 0.0, 0.0)
        desired = math.atan2(float(delta[1]), float(delta[0]))
        wz = self.yaw_to(desired, duck_yaw)

        if remaining_m <= SETTLE_REMAINING_M:
            return (VX_SETTLE, 0.0, wz)
        if approach:
            return (VX_APPROACH, 0.0, wz)
        return (VX_PATROL, 0.0, wz)

    def update(self, state: str, duck_xy, duck_yaw: float, *,
               target_xy=None, remaining_m: float = 1e9,
               approach: bool = False,
               interlock: Interlock | None = None) -> np.ndarray:
        if interlock is not None and interlock.blocked \
                and state not in ZERO_COMMAND_STATES:
            self.interlock_holds += 1
        target = self.raw_command(
            state, duck_xy, duck_yaw, target_xy=target_xy,
            remaining_m=remaining_m, approach=approach, interlock=interlock)
        # Applied directly.  A low-pass filter here would spend its first ticks
        # BELOW the MEASURED gait onset, which is not a gentle start - it is no
        # motion at all, followed by a jump.  It would also make the exact-zero
        # claim false for several ticks after every checkpoint stop.
        self.command[:] = np.asarray(target, dtype=np.float32)
        return self.command.copy()
