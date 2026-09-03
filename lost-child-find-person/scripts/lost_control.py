#!/usr/bin/env python3
"""The controller: exact zero while lost, pure pursuit while moving.

TWO THINGS ARE ENFORCED HERE RATHER THAN HOPED FOR
---------------------------------------------------
* **Zero means exactly zero.**  Every state in ``STATIONARY_STATES`` returns
  ``(0.0, 0.0, 0.0)`` with no filter tail, because the acceptance gate tests for
  EXACT zero and a decaying command is still a command.  This is the behavior's
  central safety claim — the duck never moves while it does not know where its
  guardian is — and it is implemented as the only thing the controller CAN do in
  those states, not as a rule it remembers to apply.

* **No sub-onset commands, ever.**  Forward gait onset on this scene was
  MEASURED as a cliff between ``vx = 0.20`` (0.010 m in 6 s — no gait at all)
  and ``vx = 0.22`` (0.409 m).  A command in between appears in the HUD and
  produces nothing on the floor, so the controller emits either a walking
  command or exact zero.

WHY THERE IS NO ``vy``
----------------------
MEASURED on this model: ``vy = ±0.18`` produces 0.002 m of lateral motion — no
gait at all — and ``vy = -0.28`` produces 0.184 m sideways together with 33 deg
of unwanted yaw.  Lateral commands on this policy are a yaw disturbance wearing
a strafe's clothes, so the controller never emits one.

THE YAW AXIS IS ASYMMETRIC AND BIASED
--------------------------------------
Straight-line runs at ``wz = 0`` drift about -6 deg over 6 s, and MEASURED at
``vx = 0.30``: ``wz = -0.16`` gives -9.0 deg/s while ``wz = +0.16`` gives only
+4.0 deg/s.  Each sign therefore carries its own gain, ceiling and dead band,
and the left dead band sits above the bias so a small left command cannot be
swallowed by it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lost_constants import (
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    STATIONARY_STATES,
    VX_CLOSE,
    VX_FOLLOW,
    VX_SETTLE,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
)
from lost_geometry import FOLLOW_FAR_M, FOLLOW_NEAR_M
from policy_runtime import wrap_angle


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


@dataclass
class LostController:
    """Produce ``(vx, vy, wz)`` from the state, the duck's pose and its target."""

    ctrl_hz: float = 50.0
    command: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))
    # Hysteresis latch for the follow distance, so the duck is not pumping the
    # throttle on and off around a single threshold.
    _closing: bool = False

    def reset(self) -> None:
        self.command[:] = 0.0
        self._closing = False

    def raw_command(self, state: str, duck_xy, duck_yaw: float, *,
                    target_xy=None, range_m: float | None = None,
                    settle: bool = False) -> tuple[float, float, float]:
        """The command for this tick, before it is stored.

        Separated from :meth:`update` so the acceptance tests can assert the
        exact-zero property on hand-built inputs without instantiating anything.
        """
        if state in STATIONARY_STATES:
            return (0.0, 0.0, 0.0)
        if state == "FOLLOW":
            return self._follow(duck_xy, duck_yaw, target_xy, range_m)
        if state == "REJOIN":
            return self._march(duck_xy, duck_yaw, target_xy, settle)
        return (0.0, 0.0, 0.0)

    # -- primitives ------------------------------------------------------
    def _follow(self, duck_xy, duck_yaw: float, target_xy,
                range_m: float | None) -> tuple[float, float, float]:
        """Hold the follow distance behind the guardian, with hysteresis.

        Inside the near threshold the duck stops entirely rather than creeping,
        because there is no command between zero and the measured gait onset
        that would let it creep.  It is a walk-or-stand policy by physical
        necessity, and the hysteresis is what stops that becoming a stutter.
        """
        if target_xy is None or range_m is None:
            return (0.0, 0.0, 0.0)
        if range_m <= FOLLOW_NEAR_M:
            self._closing = False
        elif range_m >= FOLLOW_FAR_M:
            self._closing = True
        if not self._closing:
            # Standing off, but still aiming the body at her, so the head does
            # not have to carry the whole tracking burden.
            return (0.0, 0.0, 0.0)

        delta = np.asarray(target_xy, dtype=np.float64) - np.asarray(
            duck_xy, dtype=np.float64)
        desired = math.atan2(float(delta[1]), float(delta[0]))
        return (VX_FOLLOW, 0.0, self._yaw_to(desired, duck_yaw))

    def _march(self, duck_xy, duck_yaw: float, target_xy,
               settle: bool) -> tuple[float, float, float]:
        """Walk at the current route waypoint; slow into the standoff."""
        if target_xy is None:
            return (0.0, 0.0, 0.0)
        delta = np.asarray(target_xy, dtype=np.float64) - np.asarray(
            duck_xy, dtype=np.float64)
        distance = float(np.linalg.norm(delta))
        if distance <= 0.06:
            return (0.0, 0.0, 0.0)
        desired = math.atan2(float(delta[1]), float(delta[0]))
        vx = VX_SETTLE if settle else VX_CLOSE
        return (vx, 0.0, self._yaw_to(desired, duck_yaw))

    def _yaw_to(self, desired_yaw: float, duck_yaw: float) -> float:
        """Closed-loop yaw, with independently measured signs and dead bands."""
        error = wrap_angle(desired_yaw - duck_yaw)
        if error >= 0.0:
            wz = clamp(KP_YAW_LEFT * error, 0.0, WZ_MAX_LEFT)
            return 0.0 if wz < WZ_MIN_LEFT else wz
        wz = -clamp(KP_YAW_RIGHT * abs(error), 0.0, WZ_MAX_RIGHT)
        return 0.0 if abs(wz) < WZ_MIN_RIGHT else wz

    def update(self, state: str, duck_xy, duck_yaw: float, *,
               target_xy=None, range_m: float | None = None,
               settle: bool = False) -> np.ndarray:
        target = self.raw_command(
            state, duck_xy, duck_yaw, target_xy=target_xy, range_m=range_m,
            settle=settle)
        # Applied directly.  A low-pass filter here would spend its first ticks
        # BELOW the measured gait onset, which is not a gentle start — it is no
        # motion at all, followed by a jump.
        self.command[:] = np.asarray(target, dtype=np.float32)
        return self.command.copy()
