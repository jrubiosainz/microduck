#!/usr/bin/env python3
"""The state machine, on hand-built inputs.  No MuJoCo, no physics, no rollout.

Every transition rule in this behavior is a claim about what the duck does when
it MEASURES something, so each one can be driven directly with a
:class:`etiquette_machine.Sense` and checked.  That is the whole reason the
machine owns no physics.
"""

from __future__ import annotations

import pytest

from etiquette_machine import EtiquetteMachine, Sense
from etiquette_states import (
    CLEAR_CONFIRM_S,
    MIN_OCCUPANTS_EXITED,
    MIN_RIDE_S,
    MIN_WAIT_SIDE_S,
    MIN_YIELD_S,
    STATES,
    ZERO_COMMAND_STATES,
)

HZ = 50.0
DT = 1.0 / HZ


def machine() -> EtiquetteMachine:
    m = EtiquetteMachine(ctrl_hz=HZ)
    m.set_guardian("nadia")
    return m


def drive(m: EtiquetteMachine, sense: Sense, seconds: float,
          t0: float = 0.0) -> float:
    """Feed one unchanging Sense for ``seconds``.  Returns the end time."""
    steps = int(seconds * HZ)
    for index in range(steps):
        m.update(t0 + index * DT, sense)
    return t0 + steps * DT


# -- the doorway ------------------------------------------------------------
def test_approach_holds_until_the_duck_reaches_its_holding_point():
    m = machine()
    t = drive(m, Sense(at_door_threshold=False, leg_arrived=False), 5.0)
    assert m.state == "APPROACH_DOOR"
    m.update(t, Sense(at_door_threshold=True))
    assert m.state == "YIELD_EXITERS"


def test_the_yield_refuses_to_end_while_an_exiter_is_pending():
    m = machine()
    m.update(0.0, Sense(at_door_threshold=True))
    # Everything else is ready; only the exiters are not.
    t = drive(m, Sense(all_exiters_clear=False, door_passable=True,
                       exiters_pending=1), 20.0, 0.02)
    assert m.state == "YIELD_EXITERS", "moved off with somebody still coming out"
    assert t < 30.0


def test_the_yield_refuses_to_end_while_the_door_is_not_passable():
    m = machine()
    m.update(0.0, Sense(at_door_threshold=True))
    drive(m, Sense(all_exiters_clear=True, door_passable=False), 12.0, 0.02)
    assert m.state == "YIELD_EXITERS", "walked at a door that was not open"


def test_the_yield_needs_the_clear_measurement_SUSTAINED():
    """One clear tick is not a green light."""
    m = machine()
    m.update(0.0, Sense(at_door_threshold=True))
    t = drive(m, Sense(all_exiters_clear=False, door_passable=True),
              MIN_YIELD_S + 1.0, 0.02)
    # A single clear tick, then not clear again.
    m.update(t, Sense(all_exiters_clear=True, door_passable=True))
    m.update(t + DT, Sense(all_exiters_clear=False, door_passable=True))
    assert m.state == "YIELD_EXITERS"
    # Now clear continuously for the confirm window.
    drive(m, Sense(all_exiters_clear=True, door_passable=True),
          CLEAR_CONFIRM_S + 0.2, t + 2 * DT)
    assert m.state == "FOLLOW_THROUGH"


def test_the_yield_lasts_at_least_its_minimum_even_if_everything_is_clear():
    m = machine()
    m.update(0.0, Sense(at_door_threshold=True))
    clear = Sense(all_exiters_clear=True, door_passable=True)
    drive(m, clear, MIN_YIELD_S - 0.4, 0.02)
    assert m.state == "YIELD_EXITERS", "yielded for less than the minimum"
    drive(m, clear, 1.0, MIN_YIELD_S)
    assert m.state == "FOLLOW_THROUGH"


def test_the_yield_is_recorded_with_what_was_measured_at_the_stop():
    m = machine()
    m.update(0.0, Sense(at_door_threshold=True, exiters_pending=2,
                        exiters_in_aperture=1, door_open_fraction=0.8))
    drive(m, Sense(all_exiters_clear=True, door_passable=True), 4.0, 0.02)
    assert len(m.yields) == 1
    entry = m.yields[0]
    assert entry["exiters_pending_at_stop"] == 2
    assert entry["exiters_in_aperture_at_stop"] == 1
    assert entry["duration_s"] >= MIN_YIELD_S


# -- the lift ---------------------------------------------------------------
def _to_wait_side(m: EtiquetteMachine) -> float:
    m.update(0.0, Sense(at_door_threshold=True))
    t = drive(m, Sense(all_exiters_clear=True, door_passable=True), 4.0, 0.02)
    m.update(t, Sense(leg_arrived=True))
    assert m.state == "APPROACH_LIFT"
    m.update(t + DT, Sense(at_lift_hold=True))
    assert m.state == "WAIT_SIDE"
    return t + 2 * DT


def test_wait_side_holds_until_the_doors_actually_begin_to_open():
    m = machine()
    t = _to_wait_side(m)
    drive(m, Sense(lift_open_fraction=0.0), 20.0, t)
    assert m.state == "WAIT_SIDE", "left the holding spot before the doors moved"


def test_wait_side_holds_for_its_minimum_even_if_the_doors_open_at_once():
    m = machine()
    t = _to_wait_side(m)
    drive(m, Sense(lift_open_fraction=0.9), MIN_WAIT_SIDE_S - 0.4, t)
    assert m.state == "WAIT_SIDE"


def test_the_duck_does_not_board_until_enough_occupants_have_EXITED():
    m = machine()
    t = _to_wait_side(m)
    t = drive(m, Sense(lift_open_fraction=0.9), MIN_WAIT_SIDE_S + 0.2, t)
    assert m.state == "DOORS_OPEN"
    m.update(t, Sense(lift_passable=True, lift_open_fraction=1.0))
    assert m.state == "LET_OCCUPANTS_EXIT"
    # Everybody is clear, but nobody actually came out.
    drive(m, Sense(all_occupants_clear=True, lift_passable=True,
                   occupants_exited=MIN_OCCUPANTS_EXITED - 1), 12.0, t + DT)
    assert m.state == "LET_OCCUPANTS_EXIT", "boarded before anybody got out"


def test_the_duck_does_not_board_while_an_occupant_is_still_in_the_way():
    m = machine()
    t = _to_wait_side(m)
    t = drive(m, Sense(lift_open_fraction=0.9), MIN_WAIT_SIDE_S + 0.2, t)
    m.update(t, Sense(lift_passable=True))
    drive(m, Sense(all_occupants_clear=False, lift_passable=True,
                   occupants_exited=3), 12.0, t + DT)
    assert m.state == "LET_OCCUPANTS_EXIT", "moved with somebody still exiting"


def _to_ride(m: EtiquetteMachine) -> float:
    t = _to_wait_side(m)
    t = drive(m, Sense(lift_open_fraction=0.9), MIN_WAIT_SIDE_S + 0.2, t)
    m.update(t, Sense(lift_passable=True))
    t = drive(m, Sense(all_occupants_clear=True, lift_passable=True,
                       occupants_exited=3), CLEAR_CONFIRM_S + 0.2, t + DT)
    assert m.state == "FOLLOW_GUARDIAN_IN"
    m.update(t, Sense(inside_cabin=True, guardian_inside_cabin=True))
    assert m.state == "POSITION_INSIDE"
    m.update(t + DT, Sense(at_cabin_hold=True))
    assert m.state == "RIDE"
    return t + 2 * DT


def test_the_ride_cannot_be_shortened_by_the_machine():
    m = machine()
    t = _to_ride(m)
    drive(m, Sense(rear_open_fraction=0.0), MIN_RIDE_S + 5.0, t)
    assert m.state == "RIDE", "left the lift before the rear doors opened"


def test_the_ride_lasts_at_least_its_minimum_even_if_the_doors_open_early():
    m = machine()
    t = _to_ride(m)
    drive(m, Sense(rear_open_fraction=1.0), MIN_RIDE_S - 0.5, t)
    assert m.state == "RIDE"


def test_the_guardian_leaves_first_is_a_TRANSITION_not_a_caption():
    m = machine()
    t = _to_ride(m)
    t = drive(m, Sense(rear_open_fraction=1.0), MIN_RIDE_S + 0.2, t)
    assert m.state == "DOORS_OPEN_TARGET"
    # The doors are open and passable, but she has not stepped out.
    drive(m, Sense(rear_open_fraction=1.0, rear_passable=True,
                   guardian_through_rear=False), 8.0, t)
    assert m.state == "DOORS_OPEN_TARGET", "left the cabin before she did"
    m.update(t + 8.0, Sense(rear_passable=True, guardian_through_rear=True))
    assert m.state == "FOLLOW_OUT"


# -- structure ---------------------------------------------------------------
def test_the_guardian_cannot_be_reassigned():
    m = machine()
    with pytest.raises(ValueError, match="follows one person"):
        m.set_guardian("tomas")


def test_every_declared_state_has_a_handler():
    m = machine()
    for state in STATES:
        assert hasattr(m, f"_{state.lower()}_state"), state


def test_done_is_terminal():
    m = EtiquetteMachine(ctrl_hz=HZ)
    m.state = "DONE"
    drive(m, Sense(leg_arrived=True, route_remaining_m=0.0), 5.0)
    assert m.state == "DONE"


def test_the_zero_command_states_are_the_ones_that_claim_a_standstill():
    """Every state that waits on somebody must be a zero-command state.

    Pinned as a list rather than as a rule so that adding a waiting state and
    forgetting to make it still is a test failure rather than a quiet
    regression.
    """
    for state in ("YIELD_EXITERS", "WAIT_SIDE", "DOORS_OPEN",
                  "LET_OCCUPANTS_EXIT", "RIDE", "DOORS_OPEN_TARGET"):
        assert state in ZERO_COMMAND_STATES, state


def test_a_ceiling_MOVES_the_machine_rather_than_only_logging():
    """A ceiling that does not transition is not a ceiling."""
    from etiquette_states import YIELD_MAX_S
    m = machine()
    m.update(0.0, Sense(at_door_threshold=True))
    drive(m, Sense(all_exiters_clear=False, door_passable=False),
          YIELD_MAX_S + 1.0, 0.02)
    assert m.state == "FOLLOW_THROUGH"
    assert any("YIELD_EXITERS" in entry for entry in m.timeouts)
    assert m.yields and m.yields[0].get("ceiling_reached") is True
