#!/usr/bin/env python3
"""The controller: no sub-onset commands, no ``vy``, and an asymmetric yaw axis.

Two properties are enforced in the code rather than hoped for, and both are
consequences of measurements taken on THIS scene with THIS model:

* the forward gait onset is a cliff, so the controller emits either a walking
  command or exact zero — never a value in the dead band that would appear in
  the metrics and produce nothing on the floor;
* lateral commands are a yaw disturbance wearing a strafe's clothes, so the
  controller never emits ``vy`` at all.  That is why changing sides has to be a
  path, and it is the single measurement the whole behavior rests on.

``raw_command`` is deliberately separated from ``update`` so every property here
runs on hand-built inputs with no MuJoCo.  The exhaustive sweep at the end is
what makes "never" a real word: it covers every state, a grid of geometries and
both yaw signs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from beside_constants import (  # noqa: E402
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    MOVING_STATES,
    STATES,
    VX_CLOSE,
    VX_CRUISE,
    VX_ONSET,
    VX_SETTLE,
    VX_SPRINT,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
)
from beside_control import (  # noqa: E402
    LEAD_TIME_S,
    LONG_AHEAD_EASE_M,
    LONG_AHEAD_STOP_M,
    LONG_RESUME_M,
    PURSUIT_CLOSE_M,
    PURSUIT_SETTLE_M,
    PURSUIT_SPRINT_M,
    BesideController,
    clamp,
)
from beside_geometry import BESIDE_LONG_TARGET_M  # noqa: E402

WALKING_SPEEDS = (VX_SETTLE, VX_CRUISE, VX_CLOSE, VX_SPRINT)


def controller() -> BesideController:
    return BesideController(ctrl_hz=50.0)


# -- the gait-onset cliff -----------------------------------------------------

def test_every_commanded_speed_is_zero_or_above_the_measured_onset():
    """A command between zero and the onset appears in the metrics and produces
    no motion on the floor, so the ladder must not contain one."""
    for speed in WALKING_SPEEDS:
        assert speed >= VX_ONSET, f"{speed} is below the measured gait onset"
    assert VX_SETTLE < VX_CRUISE < VX_CLOSE < VX_SPRINT


def test_the_speed_ladder_never_lands_in_the_dead_band():
    control = controller()
    for pursuit in np.arange(0.0, 1.6, 0.01):
        for error in np.arange(-0.8, 0.8, 0.02):
            control.reset()
            speed = control.speed_for(float(error), float(pursuit))
            assert speed == 0.0 or speed >= VX_ONSET, (
                f"pursuit {pursuit:.2f} error {error:.2f} -> {speed}")


def test_the_speed_ladder_selects_on_the_lead_distance():
    control = controller()
    assert control.speed_for(0.0, PURSUIT_SPRINT_M) == VX_SPRINT
    assert control.speed_for(0.0, PURSUIT_SPRINT_M - 0.01) == VX_CLOSE
    assert control.speed_for(0.0, PURSUIT_CLOSE_M) == VX_CLOSE
    assert control.speed_for(0.0, PURSUIT_CLOSE_M - 0.01) == VX_CRUISE
    assert control.speed_for(0.0, PURSUIT_SETTLE_M) == VX_SETTLE


def test_urgent_overrides_the_ladder_to_the_fastest_command():
    control = controller()
    assert control.speed_for(0.0, 0.01, urgent=True) == VX_SPRINT


def test_easing_applies_when_the_duck_has_edged_ahead_of_its_station():
    control = controller()
    mid = (PURSUIT_CLOSE_M + PURSUIT_SETTLE_M) * 0.5
    assert control.speed_for(0.0, mid) == VX_CRUISE
    control.reset()
    assert control.speed_for(LONG_AHEAD_EASE_M, mid) == VX_SETTLE


# -- the stop/go hysteresis ---------------------------------------------------

def test_running_ahead_of_the_station_stops_the_duck_completely():
    """Walking faster cannot fix being in front of somebody."""
    control = controller()
    assert control.speed_for(LONG_AHEAD_STOP_M, 2.0) == 0.0
    assert control._holding


def test_the_hold_latch_releases_only_once_the_error_has_fallen_well_back():
    control = controller()
    control.speed_for(LONG_AHEAD_STOP_M, 2.0)
    assert control.speed_for(LONG_AHEAD_STOP_M - 0.01, 2.0) == 0.0, (
        "releasing at the same threshold it latched on is a stutter")
    assert control.speed_for(LONG_RESUME_M + 0.01, 2.0) == 0.0
    assert control.speed_for(LONG_RESUME_M, 2.0) > 0.0
    assert not control._holding


def test_the_hysteresis_band_is_a_real_band():
    assert LONG_RESUME_M < LONG_AHEAD_STOP_M
    assert LONG_AHEAD_EASE_M < LONG_AHEAD_STOP_M


def test_a_duck_sitting_exactly_on_the_threshold_does_not_toggle_every_tick():
    control = controller()
    commands = [control.speed_for(LONG_AHEAD_STOP_M, 0.5) for _ in range(40)]
    assert set(commands) == {0.0}, "no toggling once latched"


def test_reset_clears_both_the_latch_and_the_stored_command():
    control = controller()
    control.speed_for(LONG_AHEAD_STOP_M, 2.0)
    control.command[:] = (0.42, 0.0, -0.3)
    control.reset()
    assert not control._holding
    assert np.allclose(control.command, 0.0)


# -- the yaw axis -------------------------------------------------------------

def test_the_yaw_axis_carries_its_own_gain_and_dead_band_per_sign():
    """The policy's own right bias swallows a small left command, so the two
    signs are not mirror images of each other."""
    assert KP_YAW_LEFT > KP_YAW_RIGHT
    assert WZ_MIN_LEFT > WZ_MIN_RIGHT, (
        "the left dead band must sit ABOVE the bias that swallows it")


def test_a_yaw_command_is_either_zero_or_outside_its_own_dead_band():
    control = controller()
    for error_deg in np.arange(-180.0, 180.0, 0.25):
        wz = control._yaw_to(math.radians(float(error_deg)), 0.0)
        if wz == 0.0:
            continue
        if wz > 0.0:
            assert wz >= WZ_MIN_LEFT, f"{error_deg} deg -> {wz}"
            assert wz <= WZ_MAX_LEFT
        else:
            assert abs(wz) >= WZ_MIN_RIGHT, f"{error_deg} deg -> {wz}"
            assert abs(wz) <= WZ_MAX_RIGHT


def test_the_yaw_sign_follows_the_shorter_way_round():
    control = controller()
    assert control._yaw_to(math.radians(40.0), 0.0) > 0.0
    assert control._yaw_to(math.radians(-40.0), 0.0) < 0.0
    # Wrapping: a target at +170 deg from a duck at -170 deg is 20 deg RIGHT.
    wz = control._yaw_to(math.radians(170.0), math.radians(-170.0))
    assert wz < 0.0


def test_a_tiny_yaw_error_produces_exactly_zero_rather_than_a_swallowed_command():
    control = controller()
    assert control._yaw_to(math.radians(1.0), 0.0) == 0.0
    assert control._yaw_to(math.radians(-1.0), 0.0) == 0.0


def test_clamp_is_inclusive_at_both_ends():
    assert clamp(-5.0, -1.0, 1.0) == -1.0
    assert clamp(5.0, -1.0, 1.0) == 1.0
    assert clamp(0.25, -1.0, 1.0) == 0.25


# -- the command, per state ---------------------------------------------------

def test_the_controller_never_emits_vy_in_any_state_or_geometry():
    """THE measurement the behavior rests on.

    An exhaustive sweep, because a single ``vy`` command anywhere would make the
    state machine's CROSS_BEHIND state pointless and would move the duck
    sideways by turning it.
    """
    control = controller()
    checked = 0
    for state in STATES:
        for dx in (-1.4, -0.3, -0.05, 0.0, 0.05, 0.3, 1.4):
            for dy in (-1.4, -0.3, 0.0, 0.3, 1.4):
                for duck_yaw_deg in (-179.0, -90.0, 0.0, 47.0, 179.0):
                    for longitudinal in (None, -1.2, -0.12, 0.0, 0.4):
                        for urgent in (False, True):
                            control.reset()
                            vx, vy, wz = control.raw_command(
                                state, (0.0, 0.0),
                                math.radians(duck_yaw_deg),
                                target_xy=(dx, dy),
                                longitudinal=longitudinal, urgent=urgent)
                            assert vy == 0.0, (
                                f"{state} emitted vy={vy} at "
                                f"target=({dx},{dy})")
                            assert vx == 0.0 or vx >= VX_ONSET
                            checked += 1
    assert checked > 5000, "the sweep must actually be exhaustive"


def test_no_target_means_no_motion_at_all():
    control = controller()
    for state in STATES:
        assert control.raw_command(state, (0.0, 0.0), 0.0,
                                   target_xy=None) == (0.0, 0.0, 0.0)


def test_the_done_state_commands_exact_zero_even_with_a_target():
    control = controller()
    assert control.raw_command("DONE", (0.0, 0.0), 0.0,
                               target_xy=(5.0, 5.0)) == (0.0, 0.0, 0.0)


def test_arriving_within_the_tolerance_stops_the_forward_command_but_keeps_yaw():
    control = controller()
    for state in ("ACQUIRE", "JOIN_SIDE", "JOIN_OTHER_SIDE", "FALL_BACK",
                  "CROSS_BEHIND"):
        vx, vy, wz = control.raw_command(
            state, (0.0, 0.0), 0.0, target_xy=(0.0, 0.07))
        assert vx == 0.0 and vy == 0.0
        assert wz != 0.0, "the duck may still turn onto the target"


def test_closing_into_a_slot_drives_fast_while_far_and_eases_in():
    control = controller()
    for state in ("ACQUIRE", "JOIN_SIDE", "JOIN_OTHER_SIDE"):
        far = control.raw_command(state, (0.0, 0.0), 0.0, target_xy=(1.5, 0.0))
        middle = control.raw_command(state, (0.0, 0.0), 0.0,
                                     target_xy=(0.5, 0.0))
        near = control.raw_command(state, (0.0, 0.0), 0.0,
                                   target_xy=(0.2, 0.0))
        assert far[0] == VX_SPRINT
        assert middle[0] == VX_CLOSE
        assert near[0] == VX_SETTLE
        assert far[0] > middle[0] > near[0]


def test_the_longitudinal_servo_is_suspended_during_the_manoeuvre():
    """It would fight the very manoeuvre it is trying to perform.

    While falling back the duck is deliberately BEHIND its station, which is
    exactly the condition the station-keeping servo exists to correct.
    """
    control = controller()
    for state in ("FALL_BACK", "CROSS_BEHIND"):
        vx, _, _ = control.raw_command(
            state, (0.0, 0.0), 0.0, target_xy=(0.9, 0.0),
            longitudinal=+2.0)
        assert vx > 0.0, f"{state} must keep walking to its rear waypoint"


def test_the_crossing_is_driven_faster_than_the_fall_back():
    """The duck is astern of her during the crossing and must not dawdle."""
    control = controller()
    fall = control.raw_command("FALL_BACK", (0.0, 0.0), 0.0,
                               target_xy=(0.9, 0.0))[0]
    cross = control.raw_command("CROSS_BEHIND", (0.0, 0.0), 0.0,
                                target_xy=(0.9, 0.0))[0]
    assert cross > fall
    assert cross == VX_CLOSE and fall == VX_SETTLE


def test_holding_station_without_a_longitudinal_reading_only_turns():
    control = controller()
    for state in ("BESIDE_LEFT", "BESIDE_RIGHT", "SIDE_BLOCKED"):
        vx, vy, _ = control.raw_command(state, (0.0, 0.0), 0.0,
                                        target_xy=(0.6, 0.6),
                                        longitudinal=None)
        assert (vx, vy) == (0.0, 0.0)


def test_holding_station_measures_its_error_against_the_station_not_zero():
    control = controller()
    # A duck exactly at its station has zero error and walks at cruise.
    vx, _, _ = control.raw_command(
        "BESIDE_LEFT", (0.0, 0.0), 0.0, target_xy=(0.2, 0.0),
        longitudinal=BESIDE_LONG_TARGET_M)
    assert vx == VX_CRUISE
    # A duck at longitudinal zero is AHEAD of its station by |target|.
    control.reset()
    vx, _, _ = control.raw_command(
        "BESIDE_LEFT", (0.0, 0.0), 0.0, target_xy=(0.2, 0.0),
        longitudinal=BESIDE_LONG_TARGET_M + LONG_AHEAD_STOP_M)
    assert vx == 0.0


# -- update() -----------------------------------------------------------------

def test_update_stores_the_command_and_returns_an_independent_copy():
    control = controller()
    returned = control.update("BESIDE_LEFT", (0.0, 0.0), 0.0,
                              target_xy=(0.8, 0.2), longitudinal=-0.12)
    assert returned.dtype == np.float32
    assert np.allclose(returned, control.command)
    returned[:] = 99.0
    assert not np.allclose(returned, control.command), (
        "the caller must not be able to reach back into the controller")


def test_update_applies_the_command_directly_with_no_low_pass_filter():
    """A filter would spend its first ticks BELOW the measured gait onset,
    which is not a gentle start — it is no motion at all, then a jump.

    Checked behaviorally: the first tick out of rest is already the full
    commanded speed, and a step change between two states arrives whole on the
    very next tick rather than being blended towards.
    """
    control = controller()
    first = control.update("JOIN_SIDE", (0.0, 0.0), 0.0, target_xy=(1.5, 0.0))
    assert float(first[0]) == pytest.approx(VX_SPRINT, abs=1e-6)
    # A step down to the settle speed lands immediately and exactly.
    second = control.update("JOIN_SIDE", (0.0, 0.0), 0.0, target_xy=(0.2, 0.0))
    assert float(second[0]) == pytest.approx(VX_SETTLE, abs=1e-6)
    # ... and a step to exact zero is exact, not asymptotic.
    third = control.update("DONE", (0.0, 0.0), 0.0, target_xy=(0.2, 0.0))
    assert float(third[0]) == 0.0


def test_the_command_is_three_wide_and_the_middle_slot_is_always_zero():
    control = controller()
    for state in MOVING_STATES:
        command = control.update(state, (0.0, 0.0), 0.4,
                                 target_xy=(0.7, -0.3), longitudinal=-0.2)
        assert command.shape == (3,)
        assert float(command[1]) == 0.0


def test_the_lead_time_is_positive_and_bounded():
    """Speed is chosen from where the slot WILL be; a zero lead is no lead and
    a huge one aims at a point she has not decided to walk to yet."""
    assert 0.0 < LEAD_TIME_S <= 3.0


def test_the_controller_never_reads_a_route_or_a_schedule():
    source = (REPO / "scripts" / "beside_control.py").read_text()
    for forbidden in ("import mujoco", "ROUTES", "people_at", "Route",
                      "pos_at", "beside_actors"):
        assert forbidden not in source, (
            f"beside_control must not reference {forbidden!r}")
