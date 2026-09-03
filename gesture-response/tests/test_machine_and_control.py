#!/usr/bin/env python3
"""The state machine and the controller, on hand-built inputs.

No MuJoCo anywhere in this file.  Every transition rule and every command
property is asserted on a :class:`Sense` built by hand, which is what lets the
decision layers be tested exhaustively rather than through whichever states one
rollout happened to visit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gest_control import (
    ARRIVED_M,
    GAIT_ONSET_EPS,
    GestureController,
    Interlock,
    is_sub_gait,
)
from gest_episode import Sense
from gest_machine import GestureMachine
from gest_states import (
    ACK_S,
    BACK_UP_TARGET_M,
    BACK_UP_TOLERANCE_M,
    CONFIRM_MAX_S,
    EXECUTE_STATES,
    GOODBYE_S,
    INTERRUPT_COMMAND,
    READY_MAX_S,
    STATE_FOR_COMMAND,
    STATES,
    STOP_HOLD_S,
    TURN_TARGET_DEG,
    TURN_TOLERANCE_DEG,
    VX_APPROACH,
    VX_BACK_UP,
    VX_ONSET,
    VX_REVERSE_ONSET,
    VX_TURN,
    WALKING_STATES,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    ZERO_COMMAND_STATES,
)

DT = 0.02


def confirmed(command: str, held: float = 0.9, confidence: float = 1.0) -> dict:
    return {"command": command, "template": command, "held_s": held,
            "match_fraction": 1.0, "readable_fraction": 1.0,
            "confidence": confidence, "rule": "test", "features": {}}


def sense(**kwargs) -> Sense:
    base = dict(locked="mira", acquisition_state="locked",
                instructor_visible=True, arm_readable=True,
                instructor_range_m=1.5, in_gesture_range=True)
    base.update(kwargs)
    return Sense(**base)


def run_to(machine: GestureMachine, state: str, t0: float = 0.0) -> float:
    """Drive the machine into ``state`` through its real transitions."""
    t = t0
    machine.update(t, sense())
    t += DT
    command = next(c for c, s in STATE_FOR_COMMAND.items() if s == state)
    machine.update(t, sense(candidate_command=command))
    t += DT
    machine.update(t, sense(candidate_command=command,
                            confirmed=confirmed(command)))
    assert machine.state == state, f"expected {state}, got {machine.state}"
    return t + DT


# -- the state table -----------------------------------------------------------
def test_every_state_has_a_handler():
    machine = GestureMachine(ctrl_hz=50.0)
    for state in STATES:
        assert hasattr(machine, f"_{state.lower()}_state"), (
            f"{state} has no handler; the machine would raise on entering it")


def test_every_command_maps_to_a_real_state():
    for command, state in STATE_FOR_COMMAND.items():
        assert state in STATES, f"{command} maps to unknown state {state}"


def test_walking_states_are_not_zero_command_states():
    """A state cannot both be walking and be required to hold an exact zero."""
    overlap = set(WALKING_STATES) & set(ZERO_COMMAND_STATES)
    assert not overlap, f"contradictory states: {overlap}"


def test_execute_stop_is_a_zero_command_state():
    """The strongest claim in the behavior, asserted structurally."""
    assert "EXECUTE_STOP" in ZERO_COMMAND_STATES
    assert "EXECUTE_STOP" not in WALKING_STATES


# -- entering an action ---------------------------------------------------------
def test_no_action_is_entered_without_a_confirmation():
    """A candidate alone must never execute; only a completed confirm may."""
    machine = GestureMachine(ctrl_hz=50.0)
    machine.update(0.0, sense())
    assert machine.state == "OBSERVE"
    for index in range(1, 200):
        machine.update(index * DT, sense(candidate_command="COME"))
        assert machine.state in ("OBSERVE", "CONFIRM"), (
            f"entered {machine.state} on a candidate with no confirmation")


def test_confirm_returns_to_observe_when_the_reading_stops_holding():
    """This is exactly what the ambiguous partial gesture must produce."""
    machine = GestureMachine(ctrl_hz=50.0)
    machine.update(0.0, sense())
    machine.update(DT, sense(candidate_command="COME"))
    assert machine.state == "CONFIRM"
    machine.update(2 * DT, sense(candidate_command=""))
    assert machine.state == "OBSERVE"
    assert not machine.episodes


@pytest.mark.parametrize("command,state", sorted(STATE_FOR_COMMAND.items()))
def test_each_confirmed_command_enters_its_own_state(command, state):
    machine = GestureMachine(ctrl_hz=50.0)
    machine.update(0.0, sense())
    machine.update(DT, sense(candidate_command=command))
    machine.update(2 * DT, sense(candidate_command=command,
                                 confirmed=confirmed(command)))
    assert machine.state == state


def test_a_confirmed_command_with_no_action_does_not_execute():
    machine = GestureMachine(ctrl_hz=50.0)
    machine.update(0.0, sense())
    machine.update(DT, sense(candidate_command="NONSENSE"))
    machine.update(2 * DT, sense(candidate_command="NONSENSE",
                                 confirmed=confirmed("NONSENSE")))
    assert machine.state == "OBSERVE"


# -- the interrupt ---------------------------------------------------------------
def test_stop_interrupts_a_walking_state():
    """A STOP that could not interrupt would be a formality, not a stop."""
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "EXECUTE_APPROACH")
    machine.update(t, sense(confirmed=confirmed(INTERRUPT_COMMAND)))
    assert machine.state == "EXECUTE_STOP"
    assert machine.interrupts and machine.interrupts[0]["interrupted"] == "COME"


def test_the_interrupted_episode_is_closed_and_labelled():
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "EXECUTE_APPROACH")
    machine.update(t, sense(confirmed=confirmed(INTERRUPT_COMMAND)))
    assert machine.episodes[-1].command == "COME"
    assert machine.episodes[-1].interrupted_by == INTERRUPT_COMMAND


@pytest.mark.parametrize("command", ["COME", "TURN_LEFT", "BACK_UP", "WAVE"])
def test_only_the_interrupt_command_may_cut_in(command):
    """Any other confirmed command must NOT redirect a manoeuvre under way."""
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "EXECUTE_TURN_LEFT")
    machine.update(t, sense(confirmed=confirmed(command), yaw_delta_deg=10.0))
    assert machine.state == "EXECUTE_TURN_LEFT", (
        f"{command} redirected a manoeuvre already under way")


def test_stop_does_not_interrupt_a_non_walking_state():
    machine = GestureMachine(ctrl_hz=50.0)
    machine.update(0.0, sense())
    machine.update(DT, sense(confirmed=confirmed(INTERRUPT_COMMAND)))
    assert not machine.interrupts


# -- exits are on MEASURED quantities ----------------------------------------------
def test_approach_exits_only_on_measured_clearance():
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "EXECUTE_APPROACH")
    machine.update(t, sense(in_standoff_band=False))
    assert machine.state == "EXECUTE_APPROACH"
    machine.update(t + DT, sense(in_standoff_band=True))
    assert machine.state == "ACK"


def test_stop_exits_only_after_a_measured_hold():
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "EXECUTE_STOP")
    machine.update(t, sense(stop_hold_s=STOP_HOLD_S - 0.5))
    assert machine.state == "EXECUTE_STOP"
    machine.update(t + DT, sense(stop_hold_s=STOP_HOLD_S))
    assert machine.state == "ACK"


@pytest.mark.parametrize("state,sign", [
    ("EXECUTE_TURN_LEFT", +1.0), ("EXECUTE_TURN_RIGHT", -1.0)])
def test_a_turn_requires_the_right_SIGN_not_just_the_magnitude(state, sign):
    """A left turn that drifted right must fail rather than pass on size."""
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, state)
    machine.update(t, sense(yaw_delta_deg=-sign * TURN_TARGET_DEG))
    assert machine.state == state, (
        "a turn in the WRONG direction satisfied its own exit test")
    machine.update(t + DT, sense(yaw_delta_deg=sign * TURN_TARGET_DEG))
    assert machine.state == "ACK"


def test_a_turn_accepts_within_its_tolerance():
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "EXECUTE_TURN_LEFT")
    machine.update(t, sense(
        yaw_delta_deg=TURN_TARGET_DEG - TURN_TOLERANCE_DEG + 0.1))
    assert machine.state == "ACK"


def test_back_up_is_graded_on_heading_projected_displacement():
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "EXECUTE_BACK_UP")
    machine.update(t, sense(back_along_heading_m=0.0))
    assert machine.state == "EXECUTE_BACK_UP"
    machine.update(t + DT, sense(
        back_along_heading_m=BACK_UP_TARGET_M - BACK_UP_TOLERANCE_M))
    assert machine.state == "ACK"


def test_back_up_rejects_forward_travel():
    """Travelling FORWARD must never satisfy a reverse."""
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "EXECUTE_BACK_UP")
    machine.update(t, sense(back_along_heading_m=-1.0))
    assert machine.state == "EXECUTE_BACK_UP"


# -- ceilings MOVE the machine ------------------------------------------------------
def test_every_ceiling_transitions_rather_than_only_logging():
    machine = GestureMachine(ctrl_hz=50.0)
    machine.update(0.0, sense(locked="", instructor_visible=False))
    assert machine.state == "READY"
    machine.update(READY_MAX_S, sense(locked="", instructor_visible=False))
    assert machine.state == "DONE", "the READY ceiling did not move the machine"
    assert machine.timeouts


def test_confirm_ceiling_returns_to_observe():
    machine = GestureMachine(ctrl_hz=50.0)
    machine.update(0.0, sense())
    machine.update(DT, sense(candidate_command="COME"))
    # One tick PAST the ceiling rather than exactly on it: the elapsed time is
    # a float subtraction, so sitting on the boundary tests the arithmetic
    # rather than the rule.
    machine.update(DT + CONFIRM_MAX_S + DT, sense(candidate_command="COME"))
    assert machine.state == "OBSERVE"
    assert not machine.episodes, "a ceiling must not execute anything"


# -- the controller ------------------------------------------------------------------
@pytest.mark.parametrize("state", sorted(ZERO_COMMAND_STATES))
def test_zero_states_emit_a_literal_zero(state):
    controller = GestureController(ctrl_hz=50.0)
    command = controller.raw_command(state, np.zeros(2), 0.0,
                                     target_xy=np.array([5.0, 5.0]),
                                     remaining_m=5.0)
    assert command == (0.0, 0.0, 0.0), (
        f"{state} emitted {command}; it must be a literal zero, not a small one")


@pytest.mark.parametrize("state", sorted(EXECUTE_STATES))
def test_the_interlock_overrides_every_execute_state(state):
    controller = GestureController(ctrl_hz=50.0)
    blocked = Interlock(True, "somebody in the way", "teo")
    command = controller.raw_command(
        state, np.zeros(2), 0.0, target_xy=np.array([5.0, 0.0]),
        remaining_m=5.0, turned_deg=0.0, turn_target_deg=TURN_TARGET_DEG,
        interlock=blocked)
    assert command == (0.0, 0.0, 0.0), (
        f"{state} drove through a blocked interlock with {command}")


def test_no_command_the_controller_can_emit_is_sub_gait():
    """Swept over the whole input space the controller is actually given."""
    controller = GestureController(ctrl_hz=50.0)
    for state in EXECUTE_STATES:
        for turned in np.linspace(-90.0, 90.0, 25):
            for remaining in (0.0, 0.1, 0.3, 1.0, 3.0):
                for yaw in np.linspace(-math.pi, math.pi, 9):
                    command = controller.raw_command(
                        state, np.zeros(2), float(yaw),
                        target_xy=np.array([2.0, 0.0]),
                        remaining_m=float(remaining),
                        turned_deg=float(turned),
                        turn_target_deg=TURN_TARGET_DEG,
                        reference_yaw=0.0)
                    assert not is_sub_gait(command[0]), (
                        f"{state} emitted vx={command[0]} between zero and an "
                        "onset")


def test_no_command_the_controller_can_emit_has_lateral():
    controller = GestureController(ctrl_hz=50.0)
    for state in EXECUTE_STATES:
        for yaw in np.linspace(-math.pi, math.pi, 17):
            command = controller.raw_command(
                state, np.zeros(2), float(yaw),
                target_xy=np.array([1.0, 1.0]), remaining_m=1.0,
                turned_deg=10.0, turn_target_deg=TURN_TARGET_DEG,
                reference_yaw=0.0)
            assert command[1] == 0.0, f"{state} emitted vy={command[1]}"


def test_sub_gait_recognises_the_real_dead_zone():
    assert is_sub_gait(0.10) and is_sub_gait(VX_ONSET - 0.01)
    assert is_sub_gait(-0.10) and is_sub_gait(VX_REVERSE_ONSET + 0.01)
    assert not is_sub_gait(0.0)
    assert not is_sub_gait(VX_ONSET) and not is_sub_gait(VX_APPROACH)
    assert not is_sub_gait(VX_REVERSE_ONSET) and not is_sub_gait(VX_BACK_UP)


def test_sub_gait_tolerates_the_float32_round_trip():
    """THE REGRESSION THIS PINS.

    The command register is float32, so an exactly-onset command comes back as
    ``-0.3199999928474426`` - strictly greater than ``-0.32``.  MEASURED, that
    made the gate count all 230 ticks of a real reverse leg as sub-gait, while
    the same run measured 0.363 m of genuine backward displacement.
    """
    round_tripped = float(np.float32(VX_REVERSE_ONSET))
    assert round_tripped > VX_REVERSE_ONSET, (
        "this test is meaningless unless float32 really does round the other way")
    assert not is_sub_gait(round_tripped)
    assert not is_sub_gait(float(np.float32(VX_ONSET)))


def test_the_tolerance_cannot_excuse_a_real_sub_gait_command():
    """The slack must be far smaller than the gap to any command emitted."""
    assert GAIT_ONSET_EPS < 0.001
    assert is_sub_gait(VX_REVERSE_ONSET + 0.02)
    assert is_sub_gait(VX_ONSET - 0.02)


def test_turn_commands_carry_the_right_yaw_sign():
    controller = GestureController(ctrl_hz=50.0)
    left = controller.turn_command("left", 0.0, TURN_TARGET_DEG)
    right = controller.turn_command("right", 0.0, TURN_TARGET_DEG)
    assert left[2] > 0.0 and right[2] < 0.0, (
        f"named turns are not opposite: left {left[2]}, right {right[2]}")
    assert left[0] == right[0] == VX_TURN


def test_turn_yaw_is_clamped_to_the_measured_ceiling():
    controller = GestureController(ctrl_hz=50.0)
    left = controller.turn_command("left", -180.0, TURN_TARGET_DEG)
    right = controller.turn_command("right", 180.0, TURN_TARGET_DEG)
    assert left[2] <= WZ_MAX_LEFT + 1e-9
    assert abs(right[2]) <= WZ_MAX_RIGHT + 1e-9


def test_there_is_no_turn_in_place():
    """MEASURED to be unavailable, so the function returns a constant zero."""
    controller = GestureController(ctrl_hz=50.0)
    for desired in np.linspace(-math.pi, math.pi, 9):
        assert controller.spin_to(float(desired), 0.0) == 0.0


def test_reverse_closes_a_heading_loop():
    """Open-loop reverse drifts -50 deg in 6 s, so the sign must follow error."""
    controller = GestureController(ctrl_hz=50.0)
    drifted_left = controller.reverse_command(0.0, math.radians(20.0))
    drifted_right = controller.reverse_command(0.0, math.radians(-20.0))
    assert drifted_left[2] > 0.0 > drifted_right[2]
    assert drifted_left[0] == VX_BACK_UP


def test_approach_stops_commanding_when_it_has_arrived():
    controller = GestureController(ctrl_hz=50.0)
    command = controller.raw_command(
        "EXECUTE_APPROACH", np.zeros(2), 0.0,
        target_xy=np.array([ARRIVED_M * 0.5, 0.0]), remaining_m=0.0)
    assert command == (0.0, 0.0, 0.0)


def test_approach_without_a_target_is_a_zero():
    controller = GestureController(ctrl_hz=50.0)
    assert controller.raw_command(
        "EXECUTE_APPROACH", np.zeros(2), 0.0, target_xy=None) == (0.0, 0.0, 0.0)


def test_ack_and_goodbye_hold_for_their_declared_times():
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "EXECUTE_STOP")
    machine.update(t, sense(stop_hold_s=STOP_HOLD_S))
    assert machine.state == "ACK"
    ack_entered = machine.state_since
    machine.update(ack_entered + ACK_S - DT, sense())
    assert machine.state == "ACK"
    machine.update(ack_entered + ACK_S, sense())
    assert machine.state == "READY"


def test_goodbye_ends_the_session():
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "GOODBYE")
    machine.update(t + GOODBYE_S, sense())
    assert machine.state == "DONE"
    assert machine.finished


def test_done_is_terminal():
    machine = GestureMachine(ctrl_hz=50.0)
    t = run_to(machine, "GOODBYE")
    machine.update(t + GOODBYE_S, sense())
    for index in range(50):
        machine.update(t + GOODBYE_S + index * DT,
                       sense(candidate_command="COME",
                             confirmed=confirmed("COME")))
        assert machine.state == "DONE"
