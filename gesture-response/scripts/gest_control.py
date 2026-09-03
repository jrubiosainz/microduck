#!/usr/bin/env python3
"""The controller: turn a state into a command, and an exact zero everywhere else.

FOUR THINGS ARE ENFORCED HERE RATHER THAN HOPED FOR
-----------------------------------------------------
* **No sub-gait commands, ever.**  Forward gait onset on this scene was MEASURED
  as a cliff between ``vx = 0.22`` (0.009 m in 6 s - no gait at all) and
  ``vx = 0.24`` (0.522 m); reverse onset sits separately at ``vx = -0.32``,
  since ``-0.30`` produces 4 mm in six seconds.  A command between those bounds
  appears in the metrics and produces nothing on the floor.  So the duck walks,
  reverses, or holds exactly zero - there is no decorative middle, and
  :func:`is_sub_gait` exists so the gate can check that per tick rather than
  read this source.

* **No ``vy``, ever.**  Lateral commands on this policy are a yaw disturbance
  wearing a strafe's clothes, so every change of direction is a real turning
  path.  The gate requires ``max |vy| == 0.0`` over every control tick.

* **A STOP is an immediate literal zero.**  No ramp, no filter, no decay.  The
  tick the state becomes ``EXECUTE_STOP`` is the tick the command register
  reads ``(0, 0, 0)``, which is what makes "it interrupted the motion promptly"
  a one-tick claim the gate measures by tick index.

* **The target comes from the MACHINE's state**, never from a schedule.  A state
  with no target has no target at all, which is how every zero-command state
  gets its zero structurally rather than by the controller remembering to
  return one.

THE TURNS ARE CLOSED ON MEASURED YAW ERROR, PER SIGN
------------------------------------------------------
A commanded turn is NOT an open-loop hold of ``wz`` for a computed duration.
MEASURED, the two signs deliver +21.1 and -21.7 deg/s at the turn speed, which
is close but not equal, and either can be perturbed by the gait.  The controller
therefore closes on the REMAINING yaw error - target minus the trunk yaw
actually turned so far - with its own gain and dead band per sign, and the
machine exits on the measured delta.  That is what makes LEFT and RIGHT
comparable opposite manoeuvres rather than two timed open-loop holds that happen
to be labelled.

THE REVERSE CLOSES A HEADING LOOP FOR A MEASURED REASON
---------------------------------------------------------
MEASURED at the reverse onset: 6 s of open-loop reverse drifts **-50 deg** of
yaw.  A reverse leg run open-loop therefore curls hard, and the displacement
along the intended heading falls well short of the path walked.  The reverse
command closes on the heading the duck held when the command was accepted, so
"it went backwards" is displacement along that heading rather than distance
travelled in some direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from policy_runtime import wrap_angle
from gest_states import (
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    VX_APPROACH,
    VX_BACK_UP,
    VX_ONSET,
    VX_REVERSE_ONSET,
    VX_SETTLE,
    VX_TURN,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
    ZERO_COMMAND_STATES,
)

# Below this distance to the target the duck is close enough that chasing the
# bearing produces only yaw chatter, so the command goes to zero and the
# machine's own arrival test takes over.
ARRIVED_M = 0.05
# Within this of the standoff the approach eases to the settle command, so the
# duck stops INSIDE the band rather than walking through it.  DERIVED from the
# MEASURED 0.0086 m coast plus the settle command's own 0.087 m/s.
SETTLE_REMAINING_M = 0.26
# The yaw error below which a turn stops commanding yaw.  Well inside the
# machine's own 8 deg exit tolerance, so the controller is never fighting for a
# precision the exit does not ask for.
TURN_SETTLED_DEG = 3.0


# The commanded twist is stored in a float32 register, because that is what the
# policy observation is built from.  A float64 onset constant therefore CANNOT
# be compared directly against a command that has been through that register:
# ``float(np.float32(-0.32))`` is ``-0.3199999928474426``, which is strictly
# GREATER than ``-0.32``.  MEASURED consequence: the reverse leg emitted exactly
# the measured onset command and the gate counted all 230 of its ticks as
# "sub-gait", reporting that the duck had logged a reverse it could not perform
# - while the same run measured 0.363 m of real backward displacement.
#
# The tolerance is one float32 epsilon at this magnitude, which is about 3e-8;
# 1e-6 is comfortably above it and far below the gap between any onset and the
# next command the behavior emits (the nearest is 0.02 away).  So it cannot
# excuse a genuinely sub-gait command, only a round-tripped exact one.
GAIT_ONSET_EPS = 1e-6


def is_sub_gait(vx: float) -> bool:
    """Is this a forward/reverse command that MEASURABLY produces no motion?

    Exactly zero is fine - that is a hold.  Anything strictly between zero and
    an onset is not: it logs an intention and moves the robot a few millimetres.
    Exposed as a function so the acceptance gate can test every emitted command
    against it, per tick, rather than trusting this module's own branches.

    Compared with :data:`GAIT_ONSET_EPS` of slack so a command that IS the
    measured onset is not failed by its own float32 round trip - see the
    constant's comment for the measurement that forced this.
    """
    value = float(vx)
    if value == 0.0:
        return False
    if value > 0.0:
        return value < VX_ONSET - GAIT_ONSET_EPS
    return value > VX_REVERSE_ONSET + GAIT_ONSET_EPS


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


@dataclass
class Interlock:
    """This tick's raw reason the duck may not advance.

    Built by the rollout from MEASURED surface clearance to the nearest person
    and from the duck's own distance to the area boundary - never from the
    detector or the machine - so it is an INDEPENDENT check rather than an echo
    of one.  Two different mistakes are therefore needed to produce a duck that
    walks into somebody: the executor would have to drive at them AND this would
    have to fail to refuse.
    """

    blocked: bool = False
    reason: str = ""
    body: str = ""


@dataclass
class GestureController:
    """Produce ``(vx, vy, wz)`` from the state and the duck's own measurements."""

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
        every commanded turn here is a walked ARC and every look is a HEAD
        movement.
        """
        return 0.0

    def turn_command(self, direction: str, turned_deg: float,
                     target_deg: float) -> tuple[float, float, float]:
        """The command for a named turn, closed on the MEASURED yaw so far.

        ``turned_deg`` is the trunk yaw delta the duck has ACTUALLY accumulated
        since the command was accepted, and the error is what remains of the
        target.  Each sign carries its own gain and dead band, measured
        independently, because they are not the same axis in reverse.
        """
        wanted = target_deg if direction == "left" else -target_deg
        remaining = wanted - turned_deg
        if abs(remaining) <= TURN_SETTLED_DEG:
            # The turn is done in yaw; the machine's own exit will fire.  Keep
            # walking straight rather than dropping to zero mid-arc, which would
            # end the manoeuvre with a jolt.
            return (VX_TURN, 0.0, 0.0)
        error = math.radians(remaining)
        if direction == "left":
            wz = clamp(KP_YAW_LEFT * error, 0.0, WZ_MAX_LEFT)
            wz = 0.0 if wz < WZ_MIN_LEFT else wz
        else:
            wz = -clamp(KP_YAW_RIGHT * abs(error), 0.0, WZ_MAX_RIGHT)
            wz = 0.0 if abs(wz) < WZ_MIN_RIGHT else wz
        return (VX_TURN, 0.0, wz)

    def reverse_command(self, duck_yaw: float, reference_yaw: float
                        ) -> tuple[float, float, float]:
        """The command for a reverse leg, closed on the pre-action heading.

        The yaw term is INVERTED relative to a forward leg, and that is the
        subtle part: walking backwards, a positive ``wz`` still rotates the
        trunk the same way, but the correction needed to hold a heading while
        travelling in the opposite direction has the opposite sense.  It is
        applied here as a heading-hold on the reference yaw, so the sign follows
        from the error rather than from an assumption about which way a reverse
        curls - and the MEASURED -50 deg/6 s open-loop drift is exactly what it
        has to cancel.
        """
        error = wrap_angle(reference_yaw - duck_yaw)
        if error >= 0.0:
            wz = clamp(KP_YAW_LEFT * error, 0.0, WZ_MAX_LEFT)
            wz = 0.0 if wz < WZ_MIN_LEFT else wz
        else:
            wz = -clamp(KP_YAW_RIGHT * abs(error), 0.0, WZ_MAX_RIGHT)
            wz = 0.0 if abs(wz) < WZ_MIN_RIGHT else wz
        return (VX_BACK_UP, 0.0, wz)

    # -- the command ------------------------------------------------------
    def raw_command(self, state: str, duck_xy, duck_yaw: float, *,
                    target_xy=None, remaining_m: float = 1e9,
                    turned_deg: float = 0.0,
                    turn_target_deg: float = 0.0,
                    reference_yaw: float = 0.0,
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

        # THE INDEPENDENT REFUSAL.  Checked before anything else, so no
        # arithmetic below can talk its way past it.
        if interlock is not None and interlock.blocked:
            return (0.0, 0.0, 0.0)

        if state == "EXECUTE_TURN_LEFT":
            return self.turn_command("left", turned_deg, turn_target_deg)
        if state == "EXECUTE_TURN_RIGHT":
            return self.turn_command("right", turned_deg, turn_target_deg)
        if state == "EXECUTE_BACK_UP":
            return self.reverse_command(duck_yaw, reference_yaw)

        if state == "EXECUTE_APPROACH":
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
            return (VX_APPROACH, 0.0, wz)

        return (0.0, 0.0, 0.0)

    def update(self, state: str, duck_xy, duck_yaw: float, *,
               target_xy=None, remaining_m: float = 1e9,
               turned_deg: float = 0.0, turn_target_deg: float = 0.0,
               reference_yaw: float = 0.0,
               interlock: Interlock | None = None) -> np.ndarray:
        if interlock is not None and interlock.blocked \
                and state not in ZERO_COMMAND_STATES:
            self.interlock_holds += 1
        target = self.raw_command(
            state, duck_xy, duck_yaw, target_xy=target_xy,
            remaining_m=remaining_m, turned_deg=turned_deg,
            turn_target_deg=turn_target_deg, reference_yaw=reference_yaw,
            interlock=interlock)
        # Applied directly.  A low-pass filter here would spend its first ticks
        # BELOW the MEASURED gait onset, which is not a gentle start - it is no
        # motion at all, followed by a jump.  It would also make the STOP claim
        # false for several ticks after every interruption, which is the one
        # claim this behavior most needs to be literally true.
        self.command[:] = np.asarray(target, dtype=np.float32)
        return self.command.copy()
