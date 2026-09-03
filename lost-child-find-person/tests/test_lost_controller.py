#!/usr/bin/env python3
"""The controller: exact zero while lost, and no command the floor ignores.

Pure logic — the controller never consults anything but its arguments — so every
rule here is exercised on hand-built poses with no MuJoCo anywhere.

THE CENTRAL SAFETY CLAIM, TESTED AS A STRUCTURAL PROPERTY
----------------------------------------------------------
"The duck never moves while it does not know where its guardian is" is not
checked after the fact here.  It is asserted as a property of
``LostController.raw_command``: for every state in ``STATIONARY_STATES``, and
for every pose and target that could be handed to it, the command is exactly
``(0.0, 0.0, 0.0)`` — bit-exact, not approximately, because a decaying command
is still a command.

The state machine that chooses those states is graded in ``test_lost_machine``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lost_constants import (  # noqa: E402
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    MOVING_STATES,
    STATIONARY_STATES,
    VX_CLOSE,
    VX_FOLLOW,
    VX_ONSET,
    VX_SETTLE,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
)
from lost_control import LostController  # noqa: E402
from lost_geometry import FOLLOW_FAR_M, FOLLOW_NEAR_M  # noqa: E402


# ---------------------------------------------------- the controller: ZERO
def test_every_stationary_state_returns_bit_exact_zero():
    """Not approximately zero: the gate tests for EXACT zero."""
    controller = LostController()
    for state in STATIONARY_STATES:
        for yaw in (-2.9, 0.0, 1.7):
            for target in (None, (3.0, 3.0), (-2.0, 0.5)):
                command = controller.raw_command(
                    state, (0.4, -0.2), yaw, target_xy=target, range_m=4.0,
                    settle=False)
                assert command == (0.0, 0.0, 0.0), state


def test_an_unknown_state_is_also_exactly_zero():
    """Fail safe: a state the controller does not recognise must not walk."""
    controller = LostController()
    assert controller.raw_command("NOT_A_STATE", (0.0, 0.0), 0.0,
                                  target_xy=(2.0, 2.0)) == (0.0, 0.0, 0.0)


def test_the_stored_command_is_exactly_zero_in_a_stationary_state():
    controller = LostController()
    controller.update("FOLLOW", (0.0, 0.0), 0.0, target_xy=(3.0, 0.0),
                      range_m=2.0)
    assert float(np.max(np.abs(controller.command))) > 0.0
    stored = controller.update("SEARCH_SWEEP", (0.0, 0.0), 0.0)
    assert float(np.max(np.abs(stored))) == 0.0
    assert list(stored) == [0.0, 0.0, 0.0]


def test_there_is_no_filter_tail_after_a_walking_command():
    """A decaying command is still a command; zero must be immediate."""
    controller = LostController()
    for _ in range(5):
        controller.update("REJOIN", (0.0, 0.0), 0.0, target_xy=(3.0, 0.0))
    for _ in range(3):
        command = controller.update("LOST", (0.0, 0.0), 0.0)
        assert float(np.max(np.abs(command))) == 0.0


# --------------------------------------------------- the controller: motion
def test_the_controller_never_emits_a_lateral_command():
    """MEASURED: vy is a yaw disturbance wearing a strafe's clothes."""
    controller = LostController()
    for state in MOVING_STATES:
        for yaw in (-1.4, 0.0, 2.2):
            command = controller.raw_command(
                state, (0.0, 0.0), yaw, target_xy=(2.0, 1.5), range_m=2.5)
            assert command[1] == 0.0


def test_no_forward_command_ever_lands_in_the_dead_band():
    """MEASURED cliff: vx=0.20 gives no gait, vx=0.22 walks."""
    controller = LostController()
    for state in MOVING_STATES:
        for target in ((2.0, 0.0), (-2.0, 1.0), (0.5, -3.0)):
            for yaw in (-3.0, -0.5, 0.0, 1.1, 3.0):
                vx = controller.raw_command(
                    state, (0.0, 0.0), yaw, target_xy=target, range_m=2.5,
                    settle=True)[0]
                assert vx == 0.0 or vx >= VX_ONSET


def test_the_follow_speeds_are_the_measured_ones():
    assert VX_ONSET == 0.22
    assert VX_FOLLOW == 0.42
    assert VX_CLOSE == 0.46
    assert VX_SETTLE == 0.28
    assert VX_SETTLE > VX_ONSET


def test_the_follow_distance_has_hysteresis_rather_than_a_single_threshold():
    """Otherwise the duck pumps the throttle around one number."""
    controller = LostController()
    assert controller.raw_command("FOLLOW", (0.0, 0.0), 0.0,
                                  target_xy=(2.0, 0.0),
                                  range_m=FOLLOW_NEAR_M - 0.01) == (0.0, 0.0, 0.0)
    far = controller.raw_command("FOLLOW", (0.0, 0.0), 0.0,
                                 target_xy=(2.0, 0.0),
                                 range_m=FOLLOW_FAR_M + 0.01)
    assert far[0] == VX_FOLLOW
    # Still closing inside the band, because the latch is set.
    middle = controller.raw_command("FOLLOW", (0.0, 0.0), 0.0,
                                    target_xy=(2.0, 0.0),
                                    range_m=0.5 * (FOLLOW_NEAR_M + FOLLOW_FAR_M))
    assert middle[0] == VX_FOLLOW


def test_following_without_a_target_is_exactly_zero():
    controller = LostController()
    assert controller.raw_command("FOLLOW", (0.0, 0.0), 0.0, target_xy=None,
                                  range_m=None) == (0.0, 0.0, 0.0)


def test_the_march_stops_when_the_waypoint_is_underfoot():
    controller = LostController()
    assert controller.raw_command("REJOIN", (1.0, 1.0), 0.0,
                                  target_xy=(1.02, 1.0)) == (0.0, 0.0, 0.0)


def test_the_settle_flag_slows_the_last_stretch_into_the_standoff():
    controller = LostController()
    fast = controller.raw_command("REJOIN", (0.0, 0.0), 0.0,
                                  target_xy=(2.0, 0.0), settle=False)
    slow = controller.raw_command("REJOIN", (0.0, 0.0), 0.0,
                                  target_xy=(2.0, 0.0), settle=True)
    assert fast[0] == VX_CLOSE
    assert slow[0] == VX_SETTLE
    assert slow[0] < fast[0]


# ------------------------------------------------ the controller: yaw signs
def test_the_yaw_axis_carries_its_measured_asymmetry():
    """Right is the stronger sense and is helped by the policy's own bias."""
    assert KP_YAW_LEFT > KP_YAW_RIGHT
    assert WZ_MIN_LEFT > WZ_MIN_RIGHT
    assert WZ_MAX_LEFT == WZ_MAX_RIGHT == 0.55


def test_a_left_turn_command_clears_the_bias_dead_band():
    controller = LostController()
    wz = controller.raw_command("REJOIN", (0.0, 0.0), 0.0,
                                target_xy=(0.0, 2.0))[2]
    assert wz >= WZ_MIN_LEFT


def test_a_right_turn_command_is_negative_and_clears_its_own_dead_band():
    controller = LostController()
    wz = controller.raw_command("REJOIN", (0.0, 0.0), 0.0,
                                target_xy=(0.0, -2.0))[2]
    assert wz <= -WZ_MIN_RIGHT


def test_a_tiny_heading_error_produces_no_yaw_rather_than_a_swallowed_one():
    controller = LostController()
    for error in (0.01, -0.01, 0.03, -0.03):
        target = (math.cos(error) * 3.0, math.sin(error) * 3.0)
        assert controller.raw_command("REJOIN", (0.0, 0.0), 0.0,
                                      target_xy=target)[2] == 0.0


def test_the_yaw_command_never_exceeds_its_measured_ceiling():
    controller = LostController()
    for bearing in np.linspace(-math.pi, math.pi, 73):
        target = (3.0 * math.cos(bearing), 3.0 * math.sin(bearing))
        wz = controller.raw_command("REJOIN", (0.0, 0.0), 0.0,
                                    target_xy=target)[2]
        assert -WZ_MAX_RIGHT <= wz <= WZ_MAX_LEFT


def test_the_yaw_sign_always_points_toward_the_target():
    """A sign error here would make the duck circle away from the waypoint."""
    controller = LostController()
    for bearing in (0.9, 1.8, 2.6, -0.9, -1.8, -2.6):
        target = (3.0 * math.cos(bearing), 3.0 * math.sin(bearing))
        wz = controller.raw_command("REJOIN", (0.0, 0.0), 0.0,
                                    target_xy=target)[2]
        assert math.copysign(1.0, wz) == math.copysign(1.0, bearing)


def test_resetting_the_controller_clears_the_command_and_the_latch():
    controller = LostController()
    controller.update("FOLLOW", (0.0, 0.0), 0.0, target_xy=(3.0, 0.0),
                      range_m=2.0)
    controller.reset()
    assert list(controller.command) == [0.0, 0.0, 0.0]
    assert controller._closing is False
