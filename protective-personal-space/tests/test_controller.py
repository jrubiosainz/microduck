#!/usr/bin/env python3
"""The controller: exact zeros, no strafe, gait onset, and the reverse leg.

THE MEASUREMENT THIS BEHAVIOR TURNS ON
----------------------------------------
Forward gait onset on this scene is a CLIFF, not a ramp: 0.22 produces 9 mm in
six seconds and 0.24 produces 0.52 m.  So there is no such thing as a small
command, and three claims follow that are checked literally here rather than
described:

* a hold is an EXACT zero, because a sub-onset command would stand perfectly
  still and log motion - the appearance of station-keeping with none of the
  physics;
* every reposition is a real walk, since 5 cm sideways is not available;
* the policy cannot strafe, so ``vy`` is exactly zero in every branch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pps_control import PpsController, clamp, wrap
from pps_states import (KP_YAW_LEFT, KP_YAW_RIGHT, SPEED_AT_ESCORT,
                        SPEED_AT_REPOSITION, SPEED_AT_RETREAT, SPEED_AT_SETTLE,
                        SPIN_BEST_RATE_DPS, STATES, VX_ESCORT, VX_ONSET,
                        VX_REPOSITION, VX_RETREAT, VX_REVERSE_ONSET, VX_SETTLE,
                        WALKING_STATES, WZ_MAX_LEFT, WZ_MAX_RIGHT, WZ_MIN_LEFT,
                        WZ_MIN_RIGHT, ZERO_COMMAND_STATES, retreat_seconds,
                        turn_rate_dps, wrap_deg)

ALL_STATES = list(STATES)
# Distance below which the controller stops driving forward at all.
ARRIVED_M = 0.08
# Distance below which a reposition eases into the station at onset speed.
SETTLE_M = 0.28


@pytest.fixture()
def controller():
    return PpsController()


# -- the exact zero ----------------------------------------------------------
@pytest.mark.parametrize("state", list(ZERO_COMMAND_STATES))
def test_declared_hold_states_command_a_literal_zero(controller, state):
    """Checked as ``== 0.0``, not as "small".

    HOLD_BUFFER is the one that matters most: it is entered from a walk, and
    the whole point of holding a buffer is that the duck STANDS on the line
    between the two people rather than shuffling on it.
    """
    command = controller.raw(state, (0.0, 0.0), 0.0, target_xy=(5.0, 5.0))
    assert command == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("state", list(ZERO_COMMAND_STATES))
def test_a_hold_ignores_every_input_that_could_move_it(controller, state):
    """No target, no settle flag and no heading can lift a declared zero."""
    for kwargs in ({}, {"target_xy": (9.0, -9.0)},
                   {"target_xy": (0.1, 0.1), "settle": True},
                   {"retreat_heading": 2.0}):
        assert controller.raw(state, (0.0, 0.0), 1.1, **kwargs) == (0.0, 0.0,
                                                                    0.0)


def test_done_is_zero_even_though_it_is_not_a_walking_state(controller):
    assert controller.raw("DONE", (0.0, 0.0), 0.0,
                          target_xy=(3.0, 3.0)) == (0.0, 0.0, 0.0)


def test_the_hold_states_and_the_walking_states_do_not_overlap():
    assert set(ZERO_COMMAND_STATES).isdisjoint(WALKING_STATES)
    assert set(ZERO_COMMAND_STATES) | set(WALKING_STATES) | {"DONE"} == \
        set(STATES)


@pytest.mark.parametrize("state", ALL_STATES)
def test_a_walking_state_without_a_target_still_stands_still(controller,
                                                             state):
    """No station means nothing to walk to; guessing a direction is worse."""
    if state == "RETREAT":
        pytest.skip("retreat is driven by a heading, not a target")
    assert controller.raw(state, (0.0, 0.0), 0.0, target_xy=None) == (0.0, 0.0,
                                                                      0.0)


def test_an_unknown_state_is_treated_as_stand_still(controller):
    assert controller.raw("CHARGE_INTRUDER", (0.0, 0.0), 0.0,
                          target_xy=(5.0, 0.0)) == (0.0, 0.0, 0.0)


# -- no strafe ---------------------------------------------------------------
@pytest.mark.parametrize("state", ALL_STATES)
@pytest.mark.parametrize("target", [(2.0, 0.0), (0.0, 2.0), (-1.5, 1.5),
                                    (0.05, 0.05)])
def test_the_lateral_command_is_always_exactly_zero(controller, state, target):
    """The stock policy cannot strafe, so a nonzero ``vy`` would be a lie."""
    command = controller.raw(state, (0.0, 0.0), 0.4, target_xy=target,
                             retreat_heading=0.0)
    assert command[1] == 0.0


def test_the_update_path_also_never_strafes(controller):
    for state in ALL_STATES:
        command = controller.update(state, (0.0, 0.0), 0.0,
                                    target_xy=(1.0, 1.0),
                                    retreat_heading=0.0)
        assert float(command[1]) == 0.0


# -- gait onset --------------------------------------------------------------
@pytest.mark.parametrize("state", [s for s in WALKING_STATES
                                   if s != "RETREAT"])
@pytest.mark.parametrize("distance", [0.09, 0.3, 1.0, 3.0])
def test_every_forward_command_is_at_or_above_gait_onset(controller, state,
                                                         distance):
    """There is no command between zero and 0.087 m/s, so none is emitted."""
    vx = controller.raw(state, (0.0, 0.0), 0.0, target_xy=(distance, 0.0))[0]
    assert vx == 0.0 or vx >= VX_ONSET


@pytest.mark.parametrize("state", [s for s in WALKING_STATES
                                   if s != "RETREAT"])
def test_the_controller_stops_driving_once_it_has_arrived(controller, state):
    assert controller.raw(state, (0.0, 0.0), 0.0,
                          target_xy=(ARRIVED_M, 0.0))[0] == 0.0
    assert controller.raw(state, (0.0, 0.0), 0.0,
                          target_xy=(ARRIVED_M + 0.001, 0.0))[0] >= VX_ONSET


def test_the_escort_walks_at_onset_speed_at_every_distance(controller):
    """She walks at 0.070 m/s; the duck closes the gap in short walks.

    There is no command that could match her pace continuously, so the escort
    command is the onset itself and the formation is held in bursts.
    """
    for distance in (0.1, 0.29, 0.31, 1.0, 2.5):
        assert controller.raw("ESCORT", (0.0, 0.0), 0.0,
                              target_xy=(distance, 0.0))[0] == VX_ESCORT
    assert VX_ESCORT == VX_ONSET
    assert SPEED_AT_ESCORT > 0.070, "faster than the person being escorted"


@pytest.mark.parametrize("state", ["INTERPOSE", "ESCAPE_GAP", "MONITOR",
                                   "RETURN_ESCORT", "RECOVER"])
def test_a_reposition_races_at_the_faster_command(controller, state):
    """An interpose is a race against a walking adult, and the gate says so."""
    assert controller.raw(state, (0.0, 0.0), 0.0,
                          target_xy=(2.0, 0.0))[0] == VX_REPOSITION
    assert SPEED_AT_REPOSITION > SPEED_AT_ESCORT


@pytest.mark.parametrize("state", ["INTERPOSE", "ESCAPE_GAP", "RETURN_ESCORT"])
def test_a_reposition_eases_into_the_station_over_the_last_centimetres(
        controller, state):
    assert controller.raw(state, (0.0, 0.0), 0.0,
                          target_xy=(SETTLE_M - 0.001, 0.0))[0] == VX_SETTLE
    assert controller.raw(state, (0.0, 0.0), 0.0,
                          target_xy=(SETTLE_M, 0.0))[0] == VX_REPOSITION
    assert VX_SETTLE == VX_ONSET


def test_the_settle_flag_slows_a_far_target_too(controller):
    """The rollout raises it near a station, so it must not depend on distance."""
    assert controller.raw("INTERPOSE", (0.0, 0.0), 0.0, target_xy=(3.0, 0.0),
                          settle=True)[0] == VX_SETTLE


def test_the_measured_speeds_are_ordered_the_way_the_commands_are():
    assert SPEED_AT_SETTLE == SPEED_AT_ESCORT < SPEED_AT_REPOSITION
    assert VX_SETTLE == VX_ESCORT < VX_REPOSITION


# -- reverse -----------------------------------------------------------------
def test_the_retreat_is_a_genuine_reverse_gait(controller):
    """Turning to walk away would carry the duck CLOSER to her first.

    So the yield is a real reverse leg, at the slowest reverse command that
    exists: -0.30 produces four millimetres in six seconds.
    """
    assert controller.raw("RETREAT", (0.0, 0.0), 0.0,
                          retreat_heading=0.0)[0] == VX_RETREAT
    assert VX_RETREAT == VX_REVERSE_ONSET < 0.0
    assert SPEED_AT_RETREAT > 0.0


def test_the_retreat_ignores_any_station_it_is_handed(controller):
    """It is driven by the heading the yield began on, not by a target."""
    with_target = controller.raw("RETREAT", (0.0, 0.0), 0.0,
                                 target_xy=(9.0, 9.0), retreat_heading=0.0)
    without = controller.raw("RETREAT", (0.0, 0.0), 0.0, retreat_heading=0.0)
    assert with_target == without == (VX_RETREAT, 0.0, 0.0)


def test_the_retreat_closes_a_heading_loop_on_its_starting_heading(controller):
    """The reverse gait carries a large open-loop yaw drift, so it is corrected."""
    assert controller.raw("RETREAT", (0.0, 0.0), 0.0,
                          retreat_heading=1.0)[2] > 0.0
    assert controller.raw("RETREAT", (0.0, 0.0), 0.0,
                          retreat_heading=-1.0)[2] < 0.0
    assert controller.raw("RETREAT", (0.0, 0.0), 0.7,
                          retreat_heading=0.7)[2] == 0.0


def test_without_a_heading_the_retreat_holds_the_current_one(controller):
    assert controller.raw("RETREAT", (0.0, 0.0), 0.9)[2] == 0.0


def test_retreat_duration_follows_from_the_measured_reverse_speed():
    assert retreat_seconds(0.34) == pytest.approx(0.34 / SPEED_AT_RETREAT)
    assert 2.5 < retreat_seconds() < 3.5
    assert retreat_seconds(-0.34) == retreat_seconds(0.34)


# -- yaw ---------------------------------------------------------------------
@pytest.mark.parametrize("error", [0.2, 0.5, 1.0, 2.5])
def test_a_left_error_turns_left_and_a_right_error_turns_right(controller,
                                                               error):
    """Sign convention, stated once and checked both ways."""
    assert controller.yaw_command(error, 0.0) > 0.0
    assert controller.yaw_command(-error, 0.0) < 0.0


@pytest.mark.parametrize("error", [0.2, 0.4, 0.9, 2.0])
def test_the_yaw_command_is_clamped_to_the_measured_authority(controller,
                                                              error):
    assert controller.yaw_command(error, 0.0) <= WZ_MAX_LEFT
    assert controller.yaw_command(-error, 0.0) >= -WZ_MAX_RIGHT


@pytest.mark.parametrize("error,gain", [(0.2, KP_YAW_LEFT),
                                        (0.3, KP_YAW_LEFT)])
def test_a_small_left_error_is_proportional_to_its_own_gain(controller, error,
                                                            gain):
    """Each sign is tuned independently, because the two are not symmetric."""
    expected = min(gain * error, WZ_MAX_LEFT)
    assert controller.yaw_command(error, 0.0) == pytest.approx(expected)


@pytest.mark.parametrize("error", [0.2, 0.3, 0.5])
def test_a_small_right_error_uses_the_right_hand_gain(controller, error):
    expected = min(KP_YAW_RIGHT * error, WZ_MAX_RIGHT)
    assert controller.yaw_command(-error, 0.0) == pytest.approx(-expected)


def test_the_two_yaw_gains_are_tuned_separately():
    """Measured trunk-yaw deltas per sign, not one gain trusted for both."""
    assert KP_YAW_LEFT != KP_YAW_RIGHT
    assert WZ_MIN_LEFT != WZ_MIN_RIGHT


@pytest.mark.parametrize("sign,floor,gain", [(1.0, WZ_MIN_LEFT, KP_YAW_LEFT),
                                             (-1.0, WZ_MIN_RIGHT,
                                              KP_YAW_RIGHT)])
def test_a_yaw_command_below_the_measured_floor_is_dropped_to_zero(
        controller, sign, floor, gain):
    """A command the robot cannot act on is not a small turn, it is noise."""
    just_under = sign * (floor / gain) * 0.98
    just_over = sign * (floor / gain) * 1.05
    assert controller.yaw_command(just_under, 0.0) == 0.0
    assert abs(controller.yaw_command(just_over, 0.0)) >= floor


def test_a_zero_error_commands_no_turn(controller):
    assert controller.yaw_command(0.0, 0.0) == 0.0
    assert controller.yaw_command(1.3, 1.3) == 0.0


def test_yaw_error_takes_the_short_way_round(controller):
    """Facing +170 deg and asked for -160 deg is a 30 deg LEFT turn, not 330 right.

    Sized above the measured turn floor on purpose: a 2 deg wrapped error is
    correctly dropped by the deadband, which would make the wrap untestable
    through the command.  The half-open wrap itself is checked separately.
    """
    desired, current = math.radians(-160.0), math.radians(170.0)
    assert wrap(desired - current) == pytest.approx(math.radians(30.0),
                                                    abs=1e-9)
    command = controller.yaw_command(desired, current)
    assert command > 0.0, "the short way round is to the left"
    assert command <= WZ_MAX_LEFT


def test_a_wrapped_error_inside_the_deadband_is_still_dropped(controller):
    """Wrapping must not smuggle a command the robot cannot act on back in."""
    desired, current = math.radians(-179.0), math.radians(179.0)
    assert wrap(desired - current) == pytest.approx(math.radians(2.0),
                                                    abs=1e-9)
    assert controller.yaw_command(desired, current) == 0.0


@pytest.mark.parametrize("angle,expected", [(0.0, 0.0), (math.pi, -math.pi),
                                            (3 * math.pi, -math.pi),
                                            (-3 * math.pi, -math.pi)])
def test_wrap_maps_into_the_half_open_interval(angle, expected):
    assert wrap(angle) == pytest.approx(expected)
    assert -math.pi <= wrap(angle) < math.pi


@pytest.mark.parametrize("degrees", [0.0, 179.0, 181.0, 540.0, -181.0])
def test_wrap_deg_agrees_with_the_radian_version(degrees):
    assert wrap_deg(degrees) == pytest.approx(
        math.degrees(wrap(math.radians(degrees))), abs=1e-9)


@pytest.mark.parametrize("value,low,high,expected", [(5.0, 0.0, 1.0, 1.0),
                                                     (-5.0, 0.0, 1.0, 0.0),
                                                     (0.5, 0.0, 1.0, 0.5)])
def test_clamp_bounds_both_ways(value, low, high, expected):
    assert clamp(value, low, high) == expected


def test_the_duck_aims_at_a_point_because_it_cannot_pivot():
    """Turning in place was MEASURED to be unavailable: about a degree a second.

    So a station is reached by a walked arc and the controller closes on a
    POINT rather than on a heading.
    """
    assert SPIN_BEST_RATE_DPS < 2.0
    for direction in ("left", "right"):
        assert turn_rate_dps(direction) > 10.0 * SPIN_BEST_RATE_DPS


@pytest.mark.parametrize("bearing", [0.0, 60.0, 130.0, -45.0, -170.0])
def test_the_controller_steers_toward_the_station_from_any_heading(controller,
                                                                   bearing):
    angle = math.radians(bearing)
    target = (2.0 * math.cos(angle), 2.0 * math.sin(angle))
    _, _, yaw = controller.raw("INTERPOSE", (0.0, 0.0), 0.0, target_xy=target)
    if abs(bearing) < 6.0:
        assert yaw == 0.0
    else:
        assert yaw * bearing > 0.0, "turns the short way toward the station"


# -- the command vector ------------------------------------------------------
def test_update_stores_a_three_vector_and_returns_a_copy(controller):
    command = controller.update("INTERPOSE", (0.0, 0.0), 0.0,
                                target_xy=(2.0, 0.0))
    assert command.shape == (3,)
    assert command.dtype == np.float32
    assert command is not controller.command
    command[0] = 99.0
    assert controller.command[0] != 99.0


def test_reset_clears_the_stored_command(controller):
    controller.update("INTERPOSE", (0.0, 0.0), 0.0, target_xy=(2.0, 0.0))
    assert float(np.max(np.abs(controller.command))) > 0.0
    controller.reset()
    assert float(np.max(np.abs(controller.command))) == 0.0


def test_the_controller_is_stateless_between_calls(controller):
    """A station change must not be coloured by the previous command."""
    first = controller.raw("INTERPOSE", (0.0, 0.0), 0.0, target_xy=(3.0, 0.0))
    controller.update("ESCORT", (0.0, 0.0), 0.0, target_xy=(0.5, 0.5))
    assert controller.raw("INTERPOSE", (0.0, 0.0), 0.0,
                          target_xy=(3.0, 0.0)) == first


def test_the_named_failures_are_declared_so_a_run_would_fail_loudly():
    from pps_states import FORBIDDEN_STATES
    assert FORBIDDEN_STATES == ("CHARGE_INTRUDER", "BLOCK_BOTH",
                                "CONTACT_THREAT")
    assert set(FORBIDDEN_STATES).isdisjoint(STATES)
