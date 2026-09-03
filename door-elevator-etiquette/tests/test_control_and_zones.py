#!/usr/bin/env python3
"""The controller, the zones and the aim, on hand-built inputs.

Everything here is pure geometry and pure control law: no MuJoCo, no policy, no
rollout.  These are the claims the acceptance gate rests on, checked where they
are cheap to check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from etiquette_aim import (
    EXITER_STATES,
    GUARDIAN_STATES,
    OCCUPANT_STATES,
    expected_subject_order,
    least_clear,
    look_through_point,
    role_of,
    subject_for,
)
from etiquette_control import (
    HOLD_RADIUS_M,
    LEG_ARRIVED_M,
    SETTLE_REMAINING_M,
    EtiquetteController,
    Interlock,
)
from etiquette_states import (
    DUCK_PLANAR_RADIUS,
    SPIN_BEST_RATE_DPS,
    VX_CAREFUL,
    VX_ONSET,
    VX_SETTLE,
    VX_WALK,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
    ZERO_COMMAND_STATES,
)
from etiquette_zones import (
    CABIN_HOLD_XY,
    CABIN_INTERIOR,
    DOOR_APERTURE,
    DOOR_THRESHOLD,
    LIFT_APERTURE,
    LIFT_PASSAGE,
    WAIT_SIDE_XY,
    cabin_contains,
    cabin_margin_m,
)


class Person:
    """The bare minimum of a PersonState for the pure-logic paths."""

    def __init__(self, x, y, speed=0.0):
        self.pos = np.array([float(x), float(y)])
        self.speed = speed


# -- the controller ----------------------------------------------------------
def test_every_zero_command_state_returns_a_LITERAL_zero():
    controller = EtiquetteController()
    for state in ZERO_COMMAND_STATES:
        command = controller.raw_command(
            state, (0.0, 0.0), 0.0, target_xy=(5.0, 5.0), remaining_m=9.0)
        assert command == (0.0, 0.0, 0.0), state


def test_no_command_ever_falls_between_zero_and_the_measured_gait_onset():
    """The onset is a cliff: a command below it produces no gait at all."""
    for command in (VX_WALK, VX_CAREFUL, VX_SETTLE):
        assert command >= VX_ONSET, command


def test_the_controller_never_emits_a_lateral_command():
    controller = EtiquetteController()
    for state in ("APPROACH_DOOR", "FOLLOW_THROUGH", "FOLLOW_OUT"):
        for yaw in (-2.0, 0.0, 1.5):
            _, vy, _ = controller.raw_command(
                state, (0.0, 0.0), yaw, target_xy=(1.0, 1.0), remaining_m=5.0)
            assert vy == 0.0


def test_the_controller_never_spins():
    """Turn in place is MEASURED at 1.6 deg/s, so it is not available at all."""
    controller = EtiquetteController()
    assert controller.spin_to(3.0, 0.0) == 0.0
    assert controller.spin_to(-3.0, 0.0) == 0.0
    assert SPIN_BEST_RATE_DPS < 2.0


def test_the_interlock_holds_the_duck_whatever_the_state_wants():
    controller = EtiquetteController()
    blocked = Interlock(True, "occupied by tomas", "concourse_door")
    command = controller.raw_command(
        "FOLLOW_THROUGH", (0.0, 0.0), 0.0, target_xy=(2.0, 0.0),
        remaining_m=2.0, interlock=blocked)
    assert command == (0.0, 0.0, 0.0)


def test_the_interlock_is_INDEPENDENT_of_the_state_machine():
    """A machine that released early must still be refused by the controller."""
    controller = EtiquetteController()
    # FOLLOW_GUARDIAN_IN is a walking state; the machine has authorised it.
    free = controller.raw_command(
        "FOLLOW_GUARDIAN_IN", (0.0, 0.0), 0.0, target_xy=(2.0, 0.0),
        remaining_m=2.0, interlock=Interlock(False))
    held = controller.raw_command(
        "FOLLOW_GUARDIAN_IN", (0.0, 0.0), 0.0, target_xy=(2.0, 0.0),
        remaining_m=2.0,
        interlock=Interlock(True, "the guardian is in this aperture",
                            "lift_front"))
    assert free[0] > 0.0
    assert held == (0.0, 0.0, 0.0)


def test_the_yaw_signs_carry_their_own_measured_gains_and_dead_bands():
    controller = EtiquetteController()
    # A small LEFT error is swallowed by the policy's own right bias.
    assert controller.yaw_to(math.radians(4.0), 0.0) == 0.0
    # The same error to the right is not.
    assert controller.yaw_to(math.radians(-8.0), 0.0) < 0.0
    # Both ceilings are respected.
    assert controller.yaw_to(math.radians(170.0), 0.0) <= WZ_MAX_LEFT
    assert abs(controller.yaw_to(math.radians(-170.0), 0.0)) <= WZ_MAX_RIGHT
    assert WZ_MIN_LEFT > WZ_MIN_RIGHT, "the left dead band must clear the bias"


def test_the_duck_eases_in_near_a_holding_point():
    controller = EtiquetteController()
    far = controller.raw_command("APPROACH_DOOR", (0.0, 0.0), 0.0,
                                 target_xy=(3.0, 0.0), remaining_m=3.0)
    near = controller.raw_command("APPROACH_DOOR", (0.0, 0.0), 0.0,
                                  target_xy=(0.3, 0.0),
                                  remaining_m=SETTLE_REMAINING_M - 0.05)
    assert far[0] == VX_WALK
    assert near[0] == VX_SETTLE


def test_the_careful_band_slows_the_duck_through_an_aperture():
    controller = EtiquetteController()
    cruise = controller.raw_command("FOLLOW_THROUGH", (0.0, 0.0), 0.0,
                                    target_xy=(3.0, 0.0), remaining_m=3.0,
                                    careful=False)
    careful = controller.raw_command("FOLLOW_THROUGH", (0.0, 0.0), 0.0,
                                     target_xy=(3.0, 0.0), remaining_m=3.0,
                                     careful=True)
    assert cruise[0] == VX_WALK
    assert careful[0] == VX_CAREFUL
    assert VX_CAREFUL < VX_WALK


def test_the_arrival_tolerance_is_reachable_by_the_pursuit_dead_zone():
    """The leg-arrival tolerance must exceed the controller's own dead zone.

    A leg completion test tighter than the distance at which the controller
    stops steering can never fire - which cost this behavior 25 s of a run
    standing at a holding point waiting for a phase ceiling.
    """
    assert LEG_ARRIVED_M > 0.05
    assert HOLD_RADIUS_M > LEG_ARRIVED_M


# -- the zones ---------------------------------------------------------------
def test_the_door_holding_point_is_outside_the_threshold_band():
    from etiquette_path import door_hold_xy
    assert DOOR_THRESHOLD.depth_into(door_hold_xy(), DUCK_PLANAR_RADIUS) == 0.0


def test_the_lift_holding_point_is_beside_the_doors_not_in_front_of_them():
    assert LIFT_PASSAGE.depth_into(WAIT_SIDE_XY, DUCK_PLANAR_RADIUS) == 0.0
    assert not LIFT_PASSAGE.contains(WAIT_SIDE_XY, DUCK_PLANAR_RADIUS)


def test_the_cabin_holding_point_is_inside_the_cabin_whole_footprint():
    assert cabin_contains(CABIN_HOLD_XY, DUCK_PLANAR_RADIUS)
    assert cabin_margin_m(CABIN_HOLD_XY) > DUCK_PLANAR_RADIUS


def test_the_cabin_holding_point_is_clear_of_the_front_aperture():
    assert LIFT_APERTURE.depth_into(CABIN_HOLD_XY, DUCK_PLANAR_RADIUS) == 0.0


def test_depth_into_is_zero_outside_and_positive_inside():
    centre = DOOR_APERTURE.center()
    assert DOOR_APERTURE.depth_into(centre) > 0.0
    assert DOOR_APERTURE.depth_into((centre[0] + 5.0, centre[1])) == 0.0


def test_a_footprint_penetrates_a_band_its_centre_is_outside_of():
    """The radius has to count, or a robot's nose could hover over a line."""
    edge_x = DOOR_THRESHOLD.x_range[0] - 0.5 * DUCK_PLANAR_RADIUS
    point = (edge_x, DOOR_THRESHOLD.center()[1])
    assert DOOR_THRESHOLD.depth_into(point, 0.0) == 0.0
    assert DOOR_THRESHOLD.depth_into(point, DUCK_PLANAR_RADIUS) > 0.0


def test_the_cabin_interior_is_inset_by_the_duck_radius():
    from lobby_layout import CABIN_X, CABIN_Y
    assert CABIN_INTERIOR.x_range[0] == pytest.approx(
        CABIN_X[0] + DUCK_PLANAR_RADIUS)
    assert CABIN_INTERIOR.y_range[1] == pytest.approx(
        CABIN_Y[1] - DUCK_PLANAR_RADIUS)


# -- the aim -----------------------------------------------------------------
def test_the_subject_is_the_person_the_state_is_waiting_on():
    people = {
        "nadia": Person(0.0, 0.0), "tomas": Person(-2.0, 0.0),
        "leila": Person(-0.5, 0.0), "priya": Person(2.0, 0.0),
        "marek": Person(1.0, 0.0), "odile": Person(2.5, 0.0),
    }
    for state in EXITER_STATES:
        assert subject_for(state, people) in ("tomas", "leila")
    for state in OCCUPANT_STATES:
        assert subject_for(state, people) in ("priya", "marek", "odile")
    for state in GUARDIAN_STATES:
        assert subject_for(state, people) == "nadia"


def test_least_clear_picks_the_one_with_furthest_still_to_go():
    """Not the nearest body - the one whose progress gates the decision."""
    people = {"tomas": Person(-2.4, 0.0), "leila": Person(-0.4, 0.0)}
    # leila is still east of the door plane, so she is the least clear.
    assert least_clear(people, ("tomas", "leila"), "concourse_door",
                       -1.0) == "leila"


def test_least_clear_is_deterministic_on_a_tie():
    people = {"tomas": Person(-1.0, 0.0), "leila": Person(-1.0, 0.0)}
    first = least_clear(people, ("tomas", "leila"), "concourse_door", -1.0)
    again = least_clear(people, ("tomas", "leila"), "concourse_door", -1.0)
    assert first == again == "tomas"


def test_the_head_looks_THROUGH_an_aperture_while_crossing_it():
    for state in ("FOLLOW_THROUGH", "FOLLOW_GUARDIAN_IN", "FOLLOW_OUT"):
        point = look_through_point(state)
        assert point is not None and len(point) == 3
    assert look_through_point("RIDE") is None


def test_the_expected_subject_order_is_a_DECLARATION_not_derived():
    order = expected_subject_order()
    assert order == ("door_exiter", "guardian", "occupant", "guardian")
    assert role_of("nadia") == "guardian"
    assert role_of("tomas") == "door_exiter"
    assert role_of("priya") == "occupant"
    assert role_of("sami") == "background"
