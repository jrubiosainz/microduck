#!/usr/bin/env python3
"""The controller: pure pursuit along the route, clamped to the authorised leg,
and an exact zero everywhere else.

THREE THINGS ARE ENFORCED HERE RATHER THAN HOPED FOR
------------------------------------------------------
* **No sub-onset commands, ever.**  Forward gait onset on this scene was
  MEASURED as a cliff between ``vx = 0.22`` (0.009 m in 6 s - no gait at all)
  and ``vx = 0.24`` (0.525 m).  A command in between appears in the metrics and
  produces nothing on the floor.  **This is the measurement that makes yielding
  a STATE rather than a speed:** a robot cannot creep towards a doorway while
  somebody comes out, because there is no command between zero and a walk.  So
  it walks, or it holds exactly zero.

* **No ``vy``, ever.**  Lateral commands on this policy are a yaw disturbance
  wearing a strafe's clothes.  The duck reaches every point by pointing at it
  and walking, which is also why the cabin is a through-car.

* **The target is CLAMPED to the authorised leg.**  Each state may walk exactly
  one leg of the route (:data:`etiquette_path.STATE_LEG`), and the pursuit point
  is clipped to that leg's end.  Stopping at a holding point is therefore
  STRUCTURAL: there is no target beyond it for the duck to chase, so a state
  machine bug cannot produce a robot that walks through a door it has not been
  released for.  The zero-command states have no leg at all.

THE APERTURE INTERLOCK IS A SECOND, INDEPENDENT REFUSAL
---------------------------------------------------------
:meth:`EtiquetteController.raw_command` refuses to advance whenever the duck is
about to enter an aperture that another body is already inside, or that is not
open far enough to pass.  This duplicates what the state machine already
guarantees, and the duplication is deliberate: the two are computed from
different quantities (the machine from confirmed clear-for windows, the
controller from this tick's raw occupancy), so a mistake in either one alone
cannot produce a side-by-side pass or a walk through a closing door.  A test
mutates the machine to release early and requires the controller to hold.

THE YAW AXIS IS ASYMMETRIC AND BIASED
--------------------------------------
MEASURED at ``vx = 0.34`` over 3 s: ``wz = -0.10`` gives -7.9 deg/s while
``wz = +0.10`` gives +0.9 deg/s - the policy's own right bias swallows a small
left command almost entirely.  Each sign therefore carries its own gain, ceiling
and dead band, and the left dead band sits above the bias.  MEASURED at
``wz = 0``: 6 s of straight walking drifts -13.0 deg, which is why the heading
loop is closed even on a straight leg through a 0.66 m opening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from etiquette_states import (
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
from policy_runtime import wrap_angle

# Within this distance of the leg's end the duck eases in, so it stops at the
# holding point rather than walking through it.  Derived from the MEASURED
# 0.0106 m coast after a stop plus the settle command's own 0.088 m/s: a robot
# that begins easing 0.22 m out arrives with about 2.5 s of slow walking, which
# is enough to land inside the arrival tolerance without overshooting it.  An
# earlier draft used 0.40 m, which cost 2 s per leg across five legs for no
# measurable improvement in where the duck stopped.
SETTLE_REMAINING_M = 0.22
# The duck has reached a holding point when it is this near it.  MEASURED
# against the first full rollout: with the pursuit point clamped to the leg's
# end, the duck settles between 0.02 and 0.05 m short of the arc length and then
# stops, because the remaining pursuit vector falls under the controller's own
# 0.05 m dead zone.  A leg-completion test written as ``remaining <= 0`` therefore
# NEVER fires, and the first run spent 30 s of FOLLOW_THROUGH and 40 s of
# APPROACH_LIFT sitting still at the holding point waiting for a ceiling.  The
# tolerance is what turns arriving into an event.
LEG_ARRIVED_M = 0.08
# And the same claim in world space, for the holding points the machine tests
# by position rather than by arc length.
HOLD_RADIUS_M = 0.20


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


@dataclass
class Interlock:
    """This tick's raw reasons the duck may not advance into an aperture.

    Built by the rollout from measured occupancy and measured door fractions,
    never from the state machine, so it is an INDEPENDENT check rather than an
    echo of one.
    """

    blocked: bool = False
    reason: str = ""
    aperture: str = ""


@dataclass
class EtiquetteController:
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

        Kept as a named function returning a measured constant so the finding is
        discoverable from the controller rather than only from a comment:
        ``tools/sweep_commands.py --what spin`` produced 0.5-1.6 deg/s across the
        whole command range at ``vx = 0``, which is not a turn.  It is also why
        the lift is a through-car.  ``test_the_controller_never_spins`` pins this
        against the state machine.
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
        # BELOW the measured gait onset, which is not a gentle start - it is no
        # motion at all, followed by a jump.  It would also make the exact-zero
        # claim false for several ticks after every stop.
        self.command[:] = np.asarray(target, dtype=np.float32)
        return self.command.copy()
