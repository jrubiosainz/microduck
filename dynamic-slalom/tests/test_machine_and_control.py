#!/usr/bin/env python3
"""The state machine and the controller, on hand-built inputs.

No MuJoCo, no physics, no scenario.  Every transition rule and every command
property is asserted directly, which is what makes them cheap enough to run on
every edit and precise enough to name the rule that broke.
"""

from __future__ import annotations

import numpy as np
import pytest

from slalom_control import (
    GOAL_ARRIVED_M,
    ON_CORRIDOR_M,
    SETTLE_REMAINING_M,
    Interlock,
    SlalomController,
)
from slalom_encounter import Sense
from slalom_machine import SlalomMachine
from slalom_states import (
    COMMIT_CONFIRM_S,
    GOAL_SETTLED_MPS,
    MIN_GOAL_S,
    MIN_REPLAN_S,
    MIN_WAIT_S,
    PASS_CLEAR_M,
    RESOLVED_IGNORE_S,
    STATES,
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

DT = 1.0 / 50.0


def advance(machine: SlalomMachine, sense: Sense, seconds: float,
            t0: float = 0.0) -> float:
    """Run the machine for ``seconds`` on a constant sense.  Returns the clock."""
    t = t0
    for _ in range(int(seconds / DT)):
        t += DT
        machine.update(t, sense)
    return t


def clear_sense(**kw) -> Sense:
    """A sense with nothing in the way and plenty of course left.

    ``lateral_error_m`` defaults to a value ABOVE ``ON_CORRIDOR_M`` so that a
    CHOOSE state does not instantly complete into PASS.  A duck that has just
    committed to a corridor is by definition not on it yet; defaulting the error
    to zero would make every CHOOSE test measure the wrong transition.
    """
    base = dict(goal_remaining_m=6.0, at_goal=False, threat="",
                encounter_resolved=True, decision_side="",
                measured_speed_mps=0.13,
                lateral_error_m=ON_CORRIDOR_M + 0.10)
    base.update(kw)
    return Sense(**base)


# -- the machine's shape -------------------------------------------------------
def test_every_declared_state_has_a_handler():
    machine = SlalomMachine()
    for state in STATES:
        assert hasattr(machine, f"_{state.lower()}_state"), state


def test_the_machine_starts_in_plan_and_leaves_it_promptly():
    machine = SlalomMachine()
    assert machine.state == "PLAN"
    advance(machine, clear_sense(), 0.5)
    assert machine.state == "ADVANCE"


# -- threat handling -----------------------------------------------------------
def test_a_threat_moves_the_machine_out_of_advance():
    machine = SlalomMachine()
    advance(machine, clear_sense(), 0.5)
    machine.update(1.0, clear_sense(threat="cart", encounter_resolved=False))
    assert machine.state == "THREAT"


def test_a_commit_needs_the_same_side_sustained():
    """One favourable tick is not a green light."""
    machine = SlalomMachine()
    t = advance(machine, clear_sense(), 0.5)
    machine.update(t + DT, clear_sense(threat="c", encounter_resolved=False))
    assert machine.state == "THREAT"
    # Alternating answers must never accumulate confirmation.
    for index in range(40):
        side = "left" if index % 2 else "right"
        t += DT
        machine.update(t, clear_sense(threat="c", encounter_resolved=False,
                                      decision_side=side,
                                      chosen_clearance_m=0.5))
    assert machine.state == "THREAT", "committed on a flapping decision"
    # A steady answer commits after the confirmation window.
    t = advance(machine, clear_sense(threat="c", encounter_resolved=False,
                                     decision_side="left",
                                     chosen_clearance_m=0.5),
                COMMIT_CONFIRM_S + 0.1, t)
    assert machine.state == "CHOOSE_LEFT"


def test_a_choose_state_completes_when_the_duck_reaches_the_corridor():
    """CHOOSE ends on a MEASURED lateral error, not on a timer."""
    machine = SlalomMachine()
    machine.state = "CHOOSE_LEFT"
    on_the_way = clear_sense(threat="c", encounter_resolved=False,
                             lateral_error_m=ON_CORRIDOR_M + 0.2)
    advance(machine, on_the_way, 1.0)
    assert machine.state == "CHOOSE_LEFT"
    arrived = clear_sense(threat="c", encounter_resolved=False,
                          lateral_error_m=ON_CORRIDOR_M * 0.5)
    machine.update(2.0, arrived)
    assert machine.state == "PASS"


def test_neither_side_safe_enters_wait_and_records_the_rejection():
    machine = SlalomMachine()
    t = advance(machine, clear_sense(), 0.5)
    t = advance(machine, clear_sense(threat="c", encounter_resolved=False,
                                     decision_side="wait",
                                     rejected_side="left",
                                     rejected_clearance_m=-0.05),
                COMMIT_CONFIRM_S + 0.2, t)
    assert machine.state == "WAIT"
    # The wait is only logged when it ends, but its opening record exists.
    assert machine._wait["threat"] == "c"
    assert machine._wait["rejected_side"] == "left"


def test_a_wait_cannot_be_shorter_than_the_minimum():
    machine = SlalomMachine()
    t = advance(machine, clear_sense(), 0.5)
    t = advance(machine, clear_sense(threat="c", encounter_resolved=False,
                                     decision_side="wait"),
                COMMIT_CONFIRM_S + 0.2, t)
    assert machine.state == "WAIT"
    safe = clear_sense(threat="c", encounter_resolved=False,
                       decision_side="left", chosen_clearance_m=0.5)
    t = advance(machine, safe, MIN_WAIT_S - 0.3, t)
    assert machine.state == "WAIT", "resolved before the minimum wait elapsed"
    t = advance(machine, safe, 0.5, t)
    assert machine.state == "CHOOSE_LEFT"


def test_the_wait_duration_is_carried_into_the_pass_record():
    machine = SlalomMachine()
    t = advance(machine, clear_sense(), 0.5)
    t = advance(machine, clear_sense(threat="c", encounter_resolved=False,
                                     decision_side="wait"),
                COMMIT_CONFIRM_S + 0.2, t)
    t = advance(machine, clear_sense(threat="c", encounter_resolved=False,
                                     decision_side="right",
                                     chosen_clearance_m=0.4),
                MIN_WAIT_S + 0.4, t)
    assert machine.state == "CHOOSE_RIGHT"
    assert machine._pass.waited_s >= MIN_WAIT_S


# -- passing -------------------------------------------------------------------
def test_a_pass_ends_only_when_the_encounter_is_resolved():
    """REGRESSION: ending on ``threat`` going empty split one crossing in two.

    The threat disappears precisely BECAUSE the sidestep worked, while the body
    has not crossed yet.
    """
    from slalom_encounter import PassRecord
    machine = SlalomMachine()
    machine.state = "PASS"
    machine._encounter_body = "cart"
    machine._pass = PassRecord(index=0, threat="cart", side="left",
                               began_at_s=0.0, chosen_clearance_m=0.4,
                               rejected_side="right",
                               rejected_clearance_m=0.1)
    # Threat gone, but the encounter is NOT resolved: the pass must continue.
    advance(machine, clear_sense(threat="", encounter_resolved=False), 1.0)
    assert machine.state == "PASS"
    advance(machine, clear_sense(threat="", encounter_resolved=True), 0.1, 1.0)
    assert machine.state == "REPLAN"
    assert machine.pass_sides == ["left"]


def test_a_resolved_body_is_ignored_as_a_threat_for_a_while():
    """REGRESSION: one crossing was re-detected as a fresh encounter."""
    machine = SlalomMachine()
    machine.state = "ADVANCE"
    machine._resolved_threat = "cart"
    machine._resolved_at = 10.0
    machine.update(10.5, clear_sense(threat="cart", encounter_resolved=False))
    assert machine.state == "ADVANCE", "re-opened an encounter just resolved"
    # Once the window has passed, the same body may legitimately return.
    machine.update(10.0 + RESOLVED_IGNORE_S + 0.1,
                   clear_sense(threat="cart", encounter_resolved=False))
    assert machine.state == "THREAT"


def test_a_replan_is_recorded_after_every_pass():
    machine = SlalomMachine()
    machine.state = "REPLAN"
    advance(machine, clear_sense(), MIN_REPLAN_S + 0.2)
    assert machine.state == "ADVANCE"
    assert len(machine.replans) == 1


# -- arriving -------------------------------------------------------------------
def test_the_goal_dwell_starts_when_the_body_actually_stops():
    """REGRESSION: timing the dwell from ENTRY counted the gait's coast as drift."""
    machine = SlalomMachine()
    machine.state = "GOAL"
    still_moving = clear_sense(at_goal=True, measured_speed_mps=0.12)
    t = advance(machine, still_moving, MIN_GOAL_S + 0.5)
    assert machine.state == "GOAL", "counted the dwell while still walking"
    stopped = clear_sense(at_goal=True,
                          measured_speed_mps=GOAL_SETTLED_MPS * 0.5)
    t = advance(machine, stopped, MIN_GOAL_S + 0.2, t)
    assert machine.state == "DONE"


def test_arriving_beats_a_late_threat():
    machine = SlalomMachine()
    machine.state = "ADVANCE"
    machine.update(1.0, clear_sense(at_goal=True, threat="late",
                                    encounter_resolved=False))
    assert machine.state == "GOAL"


# -- alternation ----------------------------------------------------------------
def test_alternation_is_reported_not_enforced():
    machine = SlalomMachine()
    machine.pass_sides = ["right", "left", "right"]
    assert machine.alternating()
    machine.pass_sides = ["right", "right"]
    assert not machine.alternating()


# -- the controller --------------------------------------------------------------
def test_zero_command_states_return_a_literal_zero():
    controller = SlalomController()
    for state in ZERO_COMMAND_STATES:
        command = controller.raw_command(state, (0.0, 0.0), 0.0,
                                         target_xy=(10.0, 0.0))
        assert command == (0.0, 0.0, 0.0), state


def test_there_is_never_a_lateral_command():
    """The policy has no strafe; a vy term would be a yaw disturbance."""
    controller = SlalomController()
    for target in ((1.0, 1.0), (1.0, -1.0), (0.0, 2.0)):
        assert controller.raw_command("PASS", (0.0, 0.0), 0.0,
                                      target_xy=target)[1] == 0.0


def test_no_command_ever_falls_between_zero_and_the_gait_onset():
    """MEASURED: vx=0.22 produces 0.009 m in 6 s.  There is nothing in between."""
    controller = SlalomController()
    for state in ("ADVANCE", "PASS", "CHOOSE_LEFT", "REPLAN"):
        for remaining in (0.05, 0.2, 1.0, 9.0):
            for careful in (False, True):
                vx = controller.raw_command(
                    state, (0.0, 0.0), 0.0, target_xy=(5.0, 0.0),
                    remaining_m=remaining, careful=careful)[0]
                assert vx == 0.0 or vx >= VX_ONSET, (state, remaining, vx)


def test_the_settle_command_is_used_only_near_the_goal():
    controller = SlalomController()
    far = controller.raw_command("ADVANCE", (0.0, 0.0), 0.0,
                                 target_xy=(5.0, 0.0), remaining_m=5.0)
    near = controller.raw_command("ADVANCE", (0.0, 0.0), 0.0,
                                  target_xy=(5.0, 0.0),
                                  remaining_m=SETTLE_REMAINING_M - 0.01)
    assert far[0] == VX_WALK
    assert near[0] == VX_SETTLE


def test_the_careful_command_is_slower_but_still_a_walk():
    controller = SlalomController()
    careful = controller.raw_command("PASS", (0.0, 0.0), 0.0,
                                     target_xy=(5.0, 0.0), careful=True)
    assert careful[0] == VX_CAREFUL
    assert VX_ONSET <= VX_CAREFUL < VX_WALK


def test_the_interlock_refuses_before_the_target_is_consulted():
    controller = SlalomController()
    blocked = Interlock(True, "somebody ahead", "cart")
    assert controller.raw_command("PASS", (0.0, 0.0), 0.0,
                                  target_xy=(5.0, 0.0),
                                  interlock=blocked) == (0.0, 0.0, 0.0)


def test_the_yaw_signs_are_independent_and_respect_their_dead_bands():
    """MEASURED: wz=+0.10 gives +1.0 deg/s, wz=-0.10 gives -6.7 deg/s."""
    controller = SlalomController()
    # A large left error produces a positive command at the left ceiling.
    assert controller.yaw_to(1.5, 0.0) == pytest.approx(WZ_MAX_LEFT)
    # A large right error produces a negative one at the right ceiling.
    assert controller.yaw_to(-1.5, 0.0) == pytest.approx(-WZ_MAX_RIGHT)
    # Tiny errors fall inside each sign's own dead band.
    assert controller.yaw_to(0.001, 0.0) == 0.0
    assert controller.yaw_to(-0.001, 0.0) == 0.0
    # The left dead band sits ABOVE the right one, because of the policy's bias.
    assert WZ_MIN_LEFT > WZ_MIN_RIGHT


def test_there_is_no_turn_in_place():
    """MEASURED: the whole command range at vx=0 yields at most 1.4 deg/s."""
    controller = SlalomController()
    for desired in (-3.0, -1.0, 0.0, 1.0, 3.0):
        assert controller.spin_to(desired, 0.0) == 0.0


def test_reaching_the_target_stops_the_command():
    controller = SlalomController()
    assert controller.raw_command("ADVANCE", (0.0, 0.0), 0.0,
                                  target_xy=(0.01, 0.0)) == (0.0, 0.0, 0.0)


def test_a_missing_target_is_a_structural_zero():
    controller = SlalomController()
    assert controller.raw_command("ADVANCE", (0.0, 0.0), 0.0,
                                  target_xy=None) == (0.0, 0.0, 0.0)
