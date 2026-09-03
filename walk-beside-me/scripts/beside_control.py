#!/usr/bin/env python3
"""The controller: pure pursuit of a slot, with speed chosen by longitudinal error.

TWO THINGS ARE ENFORCED HERE RATHER THAN HOPED FOR
---------------------------------------------------
* **No sub-onset commands, ever.**  Forward gait onset on this scene was
  MEASURED as a cliff between ``vx = 0.20`` (0.010 m in 6 s — no gait at all)
  and ``vx = 0.22`` (0.409 m).  A command in between appears in the metrics and
  produces nothing on the floor, so the controller emits either a walking
  command or exact zero.  Because station-keeping cannot be trimmed
  continuously, holding a slot is a walk-or-stand policy with hysteresis; the
  hysteresis is what stops that becoming a stutter.

* **No ``vy``, ever.**  MEASURED on this model: ``vy = ±0.22`` at ``vx = 0``
  produces under 4 mm of lateral motion — no gait at all — while ``vy = -0.28``
  produces 0.255 m sideways together with 51 deg of unwanted yaw.  Lateral
  commands on this policy are a yaw disturbance wearing a strafe's clothes.
  **This is the measurement that makes the behavior what it is:** the duck
  cannot simply slide across to the other side, so changing sides has to be a
  path flown behind the guardian, which is exactly what the state machine does.

THE YAW AXIS IS ASYMMETRIC AND BIASED
--------------------------------------
MEASURED at ``vx = 0.34`` over 3 s: ``wz = -0.10`` gives -6.3 deg/s while
``wz = +0.10`` gives 0.0 deg/s — the policy's own right bias swallows a small
left command completely.  Each sign therefore carries its own gain, ceiling and
dead band, and the left dead band sits above the bias.

WHY SPEED IS CHOSEN FROM THE LEAD POINT, NOT FROM THE PRESENT SLOT
-------------------------------------------------------------------
The slot moves with the guardian, and on the outside of a bend it moves FASTER
than she does.  A controller that drove at the slot's present position would be
permanently late through every corner, and the lateral error would grow exactly
where the behavior is most interesting.  Speed is therefore selected from the
distance to a LEAD point — where the slot will be in :data:`LEAD_TIME_S` — while
the longitudinal error retains one job and one only: stopping the duck when it
has run ahead of its station, which is the single error that walking faster
cannot fix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from beside_constants import (
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    VX_CLOSE,
    VX_CRUISE,
    VX_SETTLE,
    VX_SPRINT,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
)
from beside_geometry import BESIDE_LONG_TARGET_M
from policy_runtime import wrap_angle

# Longitudinal error bands, in metres, measured in the guardian's frame.
# Positive means the duck is AHEAD of its station and must ease off.
LONG_AHEAD_STOP_M = 0.26
LONG_AHEAD_EASE_M = 0.08
# Hysteresis on the stop/go latch, so a duck sitting exactly on the threshold
# does not toggle every tick.
LONG_RESUME_M = 0.10

# Pursuit distances to the LEAD point, in metres.  Speed while holding station
# is selected from how far the duck is from where its slot is ABOUT to be, not
# from where the slot is now: on the outside of a bend the slot accelerates away
# and a controller watching only the present slot is permanently late.
PURSUIT_SPRINT_M = 0.62
PURSUIT_CLOSE_M = 0.30
PURSUIT_SETTLE_M = 0.14

# How far ahead the station is predicted when holding formation.  MEASURED
# against the geometry rather than picked: at the guardian's 0.130 m/s a 1.4 s
# lead is 0.18 m of travel, which is about the longitudinal error the gait-onset
# cliff leaves uncorrectable, so the lead cancels the lag the quantised speed
# ladder would otherwise accumulate.
LEAD_TIME_S = 1.4


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


@dataclass
class BesideController:
    """Produce ``(vx, vy, wz)`` from the state, the duck's pose and its target."""

    ctrl_hz: float = 50.0
    command: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))
    # Latched when the duck has run ahead of its station and must let her catch
    # up.  Cleared only once the error has fallen well back, which is the
    # hysteresis that keeps a walk-or-stand policy from stuttering.
    _holding: bool = False

    def reset(self) -> None:
        self.command[:] = 0.0
        self._holding = False

    # -- speed selection --------------------------------------------------
    def speed_for(self, longitudinal_error: float, pursuit_m: float, *,
                  urgent: bool = False) -> float:
        """Forward speed while holding station beside her.

        ``longitudinal_error`` is ``longitudinal - BESIDE_LONG_TARGET_M``:
        positive when the duck is ahead of where it should be, and it is the ONLY
        thing that can stop the duck, because running ahead of somebody is the
        one error that walking faster cannot fix.  ``pursuit_m`` is the distance
        to the LEAD point — where the slot will be — and it is what selects
        between the walking speeds, so a duck on the outside of a bend speeds up
        instead of quietly falling behind.

        Returns exactly zero when the duck must let her walk on, because there is
        no command between zero and the MEASURED gait onset that would let it
        creep.
        """
        if self._holding:
            if longitudinal_error <= LONG_RESUME_M:
                self._holding = False
        elif longitudinal_error >= LONG_AHEAD_STOP_M:
            self._holding = True
        if self._holding:
            return 0.0
        if urgent or pursuit_m >= PURSUIT_SPRINT_M:
            return VX_SPRINT
        if pursuit_m >= PURSUIT_CLOSE_M:
            return VX_CLOSE
        if pursuit_m <= PURSUIT_SETTLE_M or longitudinal_error >= LONG_AHEAD_EASE_M:
            return VX_SETTLE
        return VX_CRUISE

    def _yaw_to(self, desired_yaw: float, duck_yaw: float) -> float:
        """Closed-loop yaw, with independently measured signs and dead bands."""
        error = wrap_angle(desired_yaw - duck_yaw)
        if error >= 0.0:
            wz = clamp(KP_YAW_LEFT * error, 0.0, WZ_MAX_LEFT)
            return 0.0 if wz < WZ_MIN_LEFT else wz
        wz = -clamp(KP_YAW_RIGHT * abs(error), 0.0, WZ_MAX_RIGHT)
        return 0.0 if abs(wz) < WZ_MIN_RIGHT else wz

    # -- the command ------------------------------------------------------
    def raw_command(self, state: str, duck_xy, duck_yaw: float, *,
                    target_xy=None, longitudinal: float | None = None,
                    urgent: bool = False, settle: bool = False
                    ) -> tuple[float, float, float]:
        """The command for this tick, before it is stored.

        Separated from :meth:`update` so the tests can assert every property on
        hand-built inputs without instantiating anything or touching MuJoCo.
        """
        if state == "DONE" or target_xy is None:
            return (0.0, 0.0, 0.0)

        delta = np.asarray(target_xy, dtype=np.float64) - np.asarray(
            duck_xy, dtype=np.float64)
        distance = float(np.linalg.norm(delta))
        desired = math.atan2(float(delta[1]), float(delta[0]))
        wz = self._yaw_to(desired, duck_yaw)

        # While falling back or crossing, the target is a point astern of her
        # and the duck drives at it directly: the longitudinal servo would fight
        # the manoeuvre it is trying to perform.
        if state in ("FALL_BACK", "CROSS_BEHIND"):
            if distance <= 0.08:
                return (0.0, 0.0, wz)
            return (VX_CLOSE if state == "CROSS_BEHIND" else VX_SETTLE,
                    0.0, wz)

        if state in ("ACQUIRE", "JOIN_SIDE", "JOIN_OTHER_SIDE"):
            # Closing into a slot: drive at it, fast while far, easing in.
            if distance <= 0.08:
                return (0.0, 0.0, wz)
            speed = VX_SPRINT if distance > 0.90 else (
                VX_CLOSE if distance > 0.35 else VX_SETTLE)
            return (speed, 0.0, wz)

        # BESIDE_LEFT / BESIDE_RIGHT / SIDE_BLOCKED: hold station.
        if longitudinal is None:
            return (0.0, 0.0, wz)
        error = longitudinal - BESIDE_LONG_TARGET_M
        return (self.speed_for(error, distance, urgent=urgent), 0.0, wz)

    def update(self, state: str, duck_xy, duck_yaw: float, *,
               target_xy=None, longitudinal: float | None = None,
               urgent: bool = False, settle: bool = False) -> np.ndarray:
        target = self.raw_command(
            state, duck_xy, duck_yaw, target_xy=target_xy,
            longitudinal=longitudinal, urgent=urgent, settle=settle)
        # Applied directly.  A low-pass filter here would spend its first ticks
        # BELOW the measured gait onset, which is not a gentle start — it is no
        # motion at all, followed by a jump.
        self.command[:] = np.asarray(target, dtype=np.float32)
        return self.command.copy()
