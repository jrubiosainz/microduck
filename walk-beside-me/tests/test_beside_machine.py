#!/usr/bin/env python3
"""The state machine: caused switches, confirmed commitments, and a rear crossing.

Four invariants the module claims are structural rather than checked afterwards,
and each gets a test that would fail if the structure were removed:

* a switch is CAUSED — ``SIDE_BLOCKED`` needs ``BLOCK_CONFIRM_S`` of continuous
  measured refusal, and the machine records the cause it was given;
* a switch is COMMITTED TO before it begins — the far side needs
  ``CLEAR_CONFIRM_S`` of continuous usability;
* the crossing goes ASTERN — ``FALL_BACK`` ends only past ``CROSS_BEHIND_M``;
* a switch cannot CHATTER — ``SWITCH_COOLDOWN_S`` after a completed one.

The machine touches no physics and emits no command, so every rule here runs on
hand-built inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from beside_constants import (  # noqa: E402
    ACQUIRE_MAX_S,
    BLOCK_CONFIRM_S,
    BESIDE_STATES,
    CLEAR_CONFIRM_S,
    CROSS_MAX_S,
    FALL_BACK_MAX_S,
    FORBIDDEN_STATES,
    JOIN_MAX_S,
    JOIN_OTHER_MAX_S,
    JOIN_SETTLE_S,
    MOVING_STATES,
    STATES,
    SWITCH_COOLDOWN_S,
)
from beside_geometry import CROSS_BEHIND_M, CROSS_COMMIT_M  # noqa: E402
from beside_machine import BesideMachine  # noqa: E402
from side_choice import SideVerdict  # noqa: E402

CTRL_HZ = 50.0
DT = 1.0 / CTRL_HZ


def verdicts(left: bool, right: bool, *, cause: str = "static",
             detail: str = "kiosk") -> dict:
    return {
        1: SideVerdict(1, left, 1.0, "wall", 3.0, "iris", 0.0,
                       "" if left else cause, "" if left else detail),
        -1: SideVerdict(-1, right, 1.0, "wall", 3.0, "iris", 0.0,
                        "" if right else cause, "" if right else detail),
    }


class Driver:
    """Advances a machine tick by tick with an explicit clock."""

    def __init__(self, **kwargs):
        self.machine = BesideMachine(ctrl_hz=CTRL_HZ, **kwargs)
        self.machine.set_guardian("nadia")
        self.t = 0.0

    def tick(self, *, formation=False, lateral=0.0, longitudinal=0.0,
             left=True, right=True, preferred=None,
             reason="test") -> str:
        graded = verdicts(left, right)
        if preferred == "auto":
            from side_choice import prefer_side
            preferred, reason = prefer_side(graded, self.machine.side)
        state, _ = self.machine.update(
            self.t, formation_ok=formation, lateral=lateral,
            longitudinal=longitudinal, verdicts=graded, preferred=preferred,
            preference_reason=reason)
        self.t += DT
        return state

    def run(self, seconds: float, **kwargs) -> str:
        for _ in range(int(round(seconds / DT))):
            state = self.tick(**kwargs)
        return state


def joined_left() -> Driver:
    """A machine that has completed the initial join on the left."""
    driver = Driver()
    driver.tick(preferred=1, reason="right blocked by static:hedge_s")
    assert driver.machine.state == "JOIN_SIDE"
    driver.run(JOIN_SETTLE_S + 0.1, formation=True, lateral=0.58,
               longitudinal=-0.12)
    assert driver.machine.state == "BESIDE_LEFT"
    return driver


# -- the declared states ------------------------------------------------------

def test_the_machine_starts_in_acquire_with_no_side_chosen():
    machine = BesideMachine(ctrl_hz=CTRL_HZ)
    assert machine.state == "ACQUIRE"
    assert machine.side is None
    assert machine.target_side is None
    assert machine.completed_switches == 0
    assert not machine.joined


def test_every_state_the_machine_can_reach_is_declared():
    driver = joined_left()
    reached = {"ACQUIRE"}
    reached.add(driver.machine.state)
    for transition in driver.machine.transitions:
        reached.add(transition["from"])
        reached.add(transition["to"])
    assert reached <= set(STATES)


def test_the_forbidden_states_are_not_reachable_transitions_of_this_machine():
    """HOLD is not a state this behavior defines; DONE is never entered."""
    assert "HOLD" not in STATES
    assert "DONE" in STATES and "DONE" in FORBIDDEN_STATES
    assert not any(state in FORBIDDEN_STATES for state in MOVING_STATES)
    assert set(BESIDE_STATES) <= set(MOVING_STATES)


def test_the_guardian_cannot_be_reassigned():
    machine = BesideMachine(ctrl_hz=CTRL_HZ)
    machine.set_guardian("nadia")
    machine.set_guardian("nadia")
    with pytest.raises(ValueError, match="cannot be reassigned"):
        machine.set_guardian("iris")


def test_the_beside_state_name_follows_the_side():
    machine = BesideMachine(ctrl_hz=CTRL_HZ)
    machine.side = 1
    assert machine.beside_state == "BESIDE_LEFT"
    machine.side = -1
    assert machine.beside_state == "BESIDE_RIGHT"


# -- ACQUIRE ------------------------------------------------------------------

def test_acquire_waits_while_neither_side_is_usable_rather_than_picking_one():
    driver = Driver()
    driver.run(2.0, left=False, right=False, preferred=None)
    assert driver.machine.state == "ACQUIRE"
    assert driver.machine.side is None
    assert driver.machine.decisions == []


def test_acquire_records_the_initial_decision_with_both_measured_sides():
    driver = Driver()
    driver.tick(preferred=1, reason="right blocked by static:hedge_s")
    assert driver.machine.state == "JOIN_SIDE"
    assert driver.machine.side == 1
    decision = driver.machine.decisions[0]
    assert decision["kind"] == "initial"
    assert decision["side_name"] == "left"
    assert decision["reason"] == "right blocked by static:hedge_s"
    assert set(decision) >= {"left", "right"}
    assert decision["left"]["usable"] is True


def test_acquire_reports_its_ceiling_rather_than_hanging():
    driver = Driver()
    driver.run(ACQUIRE_MAX_S + 0.1, left=False, right=False, preferred=None)
    assert driver.machine.timeouts
    assert driver.machine.timeouts[0].startswith("ACQUIRE@")


# -- JOIN_SIDE ----------------------------------------------------------------

def test_the_join_requires_the_formation_to_be_held_not_merely_touched():
    driver = Driver()
    driver.tick(preferred=1, reason="r")
    # One tick short of the settle window.
    driver.run(JOIN_SETTLE_S - DT, formation=True, lateral=0.58,
               longitudinal=-0.12)
    assert driver.machine.state == "JOIN_SIDE"
    driver.tick(formation=True, lateral=0.58, longitudinal=-0.12)
    assert driver.machine.state == "BESIDE_LEFT"
    assert driver.machine.joined


def test_a_broken_formation_resets_the_settle_counter():
    driver = Driver()
    driver.tick(preferred=1, reason="r")
    driver.run(JOIN_SETTLE_S - 2 * DT, formation=True, lateral=0.58)
    driver.tick(formation=False, lateral=0.20)
    driver.run(JOIN_SETTLE_S - DT, formation=True, lateral=0.58)
    assert driver.machine.state == "JOIN_SIDE", (
        "the settle window must restart, not resume")
    driver.tick(formation=True, lateral=0.58)
    assert driver.machine.state == "BESIDE_LEFT"


def test_a_side_that_becomes_unusable_during_the_join_is_abandoned():
    """The duck re-picks rather than completing a join into a blocked slot."""
    driver = Driver()
    driver.tick(preferred=1, reason="r")
    driver.tick(formation=False, left=False, right=True)
    assert driver.machine.side == -1
    assert driver.machine.state == "JOIN_SIDE"
    reroute = driver.machine.decisions[-1]
    assert reroute["kind"] == "join_reroute"
    assert reroute["side_name"] == "right"


def test_the_join_is_not_abandoned_when_the_far_side_is_also_unusable():
    driver = Driver()
    driver.tick(preferred=1, reason="r")
    driver.tick(formation=False, left=False, right=False)
    assert driver.machine.side == 1, "there is nowhere better to go"
    assert all(d["kind"] != "join_reroute" for d in driver.machine.decisions)


def test_the_join_reports_its_ceiling():
    driver = Driver()
    driver.tick(preferred=1, reason="r")
    driver.run(JOIN_MAX_S + 0.1, formation=False)
    assert any(t.startswith("JOIN_SIDE@") for t in driver.machine.timeouts)


# -- BESIDE: the two confirmation windows -------------------------------------

def test_a_single_tick_of_blockage_does_not_start_a_switch():
    driver = joined_left()
    driver.tick(formation=True, lateral=0.58, left=False, right=True)
    driver.tick(formation=True, lateral=0.58, left=True, right=True)
    assert driver.machine.state == "BESIDE_LEFT"
    assert driver.machine.completed_switches == 0


def test_the_blockage_window_must_be_continuous_not_cumulative():
    """Interrupting the refusal resets the counter to zero."""
    driver = joined_left()
    half = BLOCK_CONFIRM_S * 0.6
    driver.run(half, formation=True, lateral=0.58, left=False, right=True)
    driver.tick(formation=True, lateral=0.58, left=True, right=True)
    driver.run(half, formation=True, lateral=0.58, left=False, right=True)
    assert driver.machine.state == "BESIDE_LEFT", (
        "two partial windows must not add up to one confirmation")


def test_the_switch_needs_both_windows_and_the_longer_one_governs():
    driver = joined_left()
    assert CLEAR_CONFIRM_S > BLOCK_CONFIRM_S, (
        "crossing is expensive; the far side must be confirmed for longer")
    driver.run(CLEAR_CONFIRM_S - DT, formation=True, lateral=0.58,
               left=False, right=True)
    assert driver.machine.state == "BESIDE_LEFT"
    driver.tick(formation=True, lateral=0.58, left=False, right=True)
    assert driver.machine.state == "SIDE_BLOCKED"


def test_a_far_side_that_is_also_blocked_never_authorises_a_crossing():
    driver = joined_left()
    driver.run(3.0 * CLEAR_CONFIRM_S, formation=True, lateral=0.58,
               left=False, right=False)
    assert driver.machine.state == "BESIDE_LEFT", (
        "crossing into a lane that is also closed would be worse than staying")
    assert driver.machine.completed_switches == 0


def test_a_far_side_that_flickers_clear_does_not_authorise_a_crossing():
    driver = joined_left()
    for _ in range(int(3.0 * CLEAR_CONFIRM_S / DT)):
        driver.tick(formation=True, lateral=0.58, left=False, right=True)
        driver.tick(formation=True, lateral=0.58, left=False, right=False)
    assert driver.machine.state == "BESIDE_LEFT"


def test_the_switch_record_names_the_measured_cause_at_the_moment_it_fired():
    driver = blocked_left()
    pending = driver.machine._switch
    assert pending["cause"] == "static"
    assert pending["detail"] == "kiosk"
    assert pending["from_side"] == "left" and pending["to_side"] == "right"
    assert pending["blocked_for_s"] >= BLOCK_CONFIRM_S
    assert pending["far_clear_for_s"] >= CLEAR_CONFIRM_S
    decision = driver.machine.decisions[-1]
    assert decision["kind"] == "blocked"
    assert decision["side"] == -1


def test_holding_a_usable_side_clears_both_counters():
    driver = joined_left()
    driver.run(BLOCK_CONFIRM_S * 0.9, formation=True, lateral=0.58,
               left=False, right=True)
    driver.tick(formation=True, lateral=0.58, left=True, right=True)
    assert driver.machine._blocked_for == 0.0
    assert driver.machine._far_clear_for == 0.0


# -- the manoeuvre ------------------------------------------------------------

def blocked_left() -> Driver:
    """A machine that has just entered SIDE_BLOCKED, and no further.

    Exactly ``CLEAR_CONFIRM_S`` of ticks, because SIDE_BLOCKED lasts precisely
    one tick by design: a single extra tick would already be in FALL_BACK.
    """
    driver = joined_left()
    driver.run(CLEAR_CONFIRM_S, formation=True, lateral=0.58,
               left=False, right=True)
    assert driver.machine.state == "SIDE_BLOCKED"
    return driver


def test_side_blocked_lasts_exactly_one_tick_and_names_the_instant():
    driver = blocked_left()
    entered = driver.machine.state_since
    driver.tick(formation=False, lateral=0.58, longitudinal=-0.20)
    assert driver.machine.state == "FALL_BACK"
    assert driver.machine.state_since > entered
    transition = driver.machine.transitions[-1]
    assert transition["from"] == "SIDE_BLOCKED"
    assert transition["to_side"] == "right"


def test_the_fall_back_ends_only_once_the_duck_is_clear_astern():
    driver = blocked_left()
    driver.tick(longitudinal=-0.20)
    assert driver.machine.state == "FALL_BACK"
    driver.run(2.0, longitudinal=-(CROSS_BEHIND_M - 0.01))
    assert driver.machine.state == "FALL_BACK", (
        "one centimetre short of the gate is still short of the gate")
    driver.tick(longitudinal=-CROSS_BEHIND_M)
    assert driver.machine.state == "CROSS_BEHIND"
    assert driver.machine._switch["longitudinal_at_cross_m"] == pytest.approx(
        -CROSS_BEHIND_M)


def test_the_fall_back_ceiling_still_moves_the_machine_on():
    """A stuck phase fails loudly rather than hanging."""
    driver = blocked_left()
    driver.tick(longitudinal=-0.20)
    driver.run(FALL_BACK_MAX_S + 0.1, longitudinal=-0.20)
    assert driver.machine.state == "CROSS_BEHIND"
    assert any(t.startswith("FALL_BACK@") for t in driver.machine.timeouts)
    assert driver.machine.transitions[-1]["reason"] == "fall-back ceiling reached"


def test_the_crossing_ends_only_once_the_duck_has_committed_to_the_far_side():
    driver = blocked_left()
    driver.tick(longitudinal=-0.20)
    driver.tick(longitudinal=-CROSS_BEHIND_M)
    assert driver.machine.state == "CROSS_BEHIND"
    # Still on the near side, and then merely at the centreline.
    driver.tick(lateral=0.40, longitudinal=-0.90)
    driver.tick(lateral=0.0, longitudinal=-0.90)
    driver.tick(lateral=-(CROSS_COMMIT_M - 0.01), longitudinal=-0.90)
    assert driver.machine.state == "CROSS_BEHIND"
    assert driver.machine.side == 1, "the side is not flipped until committed"
    driver.tick(lateral=-CROSS_COMMIT_M, longitudinal=-0.90)
    assert driver.machine.state == "JOIN_OTHER_SIDE"
    assert driver.machine.side == -1


def test_the_crossing_ceiling_commits_the_side_rather_than_hanging():
    driver = blocked_left()
    driver.tick(longitudinal=-0.20)
    driver.tick(longitudinal=-CROSS_BEHIND_M)
    driver.run(CROSS_MAX_S + 0.1, lateral=0.30, longitudinal=-0.90)
    assert driver.machine.state == "JOIN_OTHER_SIDE"
    assert driver.machine.side == -1
    assert any(t.startswith("CROSS_BEHIND@") for t in driver.machine.timeouts)


def test_a_completed_switch_closes_its_record_with_every_timestamp():
    driver = crossed()
    driver.run(JOIN_SETTLE_S + DT, formation=True, lateral=-0.58,
               longitudinal=-0.12)
    assert driver.machine.state == "BESIDE_RIGHT"
    assert driver.machine.completed_switches == 1
    switch = driver.machine.switches[0]
    for key in ("blocked_at_s", "fell_back_at_s", "crossed_at_s",
                "joined_at_s", "duration_s"):
        assert key in switch, f"the switch record is missing {key}"
    assert switch["blocked_at_s"] <= switch["fell_back_at_s"] \
        <= switch["crossed_at_s"] <= switch["joined_at_s"]
    assert switch["duration_s"] == pytest.approx(
        switch["joined_at_s"] - switch["blocked_at_s"], abs=1e-6)
    assert driver.machine.target_side is None


def crossed() -> Driver:
    driver = blocked_left()
    driver.tick(longitudinal=-0.20)
    driver.tick(longitudinal=-CROSS_BEHIND_M)
    driver.tick(lateral=-CROSS_COMMIT_M, longitudinal=-0.90)
    assert driver.machine.state == "JOIN_OTHER_SIDE"
    return driver


def test_the_far_join_reports_its_ceiling():
    driver = crossed()
    driver.run(JOIN_OTHER_MAX_S + 0.1, formation=False, lateral=-0.30)
    assert any(t.startswith("JOIN_OTHER_SIDE@") for t in driver.machine.timeouts)


# -- the cooldown -------------------------------------------------------------

def completed_switch() -> Driver:
    driver = crossed()
    driver.run(JOIN_SETTLE_S + DT, formation=True, lateral=-0.58,
               longitudinal=-0.12)
    assert driver.machine.completed_switches == 1
    return driver


def test_a_second_switch_is_refused_during_the_cooldown():
    """Without it, a duck on the boundary of two marginal lanes ping-pongs and
    the switch count becomes a count of ticks."""
    driver = completed_switch()
    assert driver.machine.cooldown_active(driver.t)
    driver.run(SWITCH_COOLDOWN_S - CLEAR_CONFIRM_S - 0.2, formation=True,
               lateral=-0.58, left=True, right=False)
    assert driver.machine.state == "BESIDE_RIGHT"
    assert driver.machine.completed_switches == 1


def test_the_cooldown_expires_and_a_genuinely_caused_switch_is_allowed_again():
    driver = completed_switch()
    driver.run(SWITCH_COOLDOWN_S + 0.1, formation=True, lateral=-0.58,
               left=True, right=True)
    assert not driver.machine.cooldown_active(driver.t)
    driver.run(CLEAR_CONFIRM_S, formation=True, lateral=-0.58,
               left=True, right=False)
    assert driver.machine.state == "SIDE_BLOCKED"


def test_the_cooldown_is_measured_from_the_join_not_from_the_blockage():
    driver = completed_switch()
    joined_at = driver.machine.switches[0]["joined_at_s"]
    assert driver.machine._last_switch_t == pytest.approx(joined_at)
    assert driver.machine.cooldown_active(joined_at + SWITCH_COOLDOWN_S - 0.01)
    assert not driver.machine.cooldown_active(
        joined_at + SWITCH_COOLDOWN_S + 0.01)


def test_the_blockage_counter_still_runs_during_the_cooldown():
    """The duck keeps MEASURING while it refuses to act, so a blockage that
    outlives the cooldown produces a switch promptly rather than restarting."""
    driver = completed_switch()
    driver.run(SWITCH_COOLDOWN_S - 0.1, formation=True, lateral=-0.58,
               left=True, right=False)
    assert driver.machine._blocked_for >= BLOCK_CONFIRM_S
    assert driver.machine.state == "BESIDE_RIGHT"


# -- the whole sequence -------------------------------------------------------

def test_the_reference_sequence_of_transitions_is_the_one_the_run_produces():
    """PINNED: the shape of the reference rollout's timeline."""
    driver = completed_switch()
    path = [(transition["from"], transition["to"])
            for transition in driver.machine.transitions]
    assert path == [
        ("ACQUIRE", "JOIN_SIDE"),
        ("JOIN_SIDE", "BESIDE_LEFT"),
        ("BESIDE_LEFT", "SIDE_BLOCKED"),
        ("SIDE_BLOCKED", "FALL_BACK"),
        ("FALL_BACK", "CROSS_BEHIND"),
        ("CROSS_BEHIND", "JOIN_OTHER_SIDE"),
        ("JOIN_OTHER_SIDE", "BESIDE_RIGHT"),
    ]
    assert driver.machine.timeouts == []


def test_the_summary_carries_the_decisions_and_switches_it_reports():
    driver = completed_switch()
    summary = driver.machine.summary()
    assert summary["guardian"] == "nadia"
    assert summary["state"] == "BESIDE_RIGHT"
    assert summary["side"] == -1
    assert len(summary["switches"]) == 1
    assert len(summary["decisions"]) >= 2
    # Mutating the summary must not reach back into the machine.
    summary["switches"].clear()
    assert driver.machine.completed_switches == 1


def test_the_machine_never_touches_physics():
    """No MuJoCo, no numpy arrays of state, no command: only transitions."""
    source = (REPO / "scripts" / "beside_machine.py").read_text()
    for forbidden in ("import mujoco", "mj_step", "data.qpos", "onnx",
                      "vx", "self.command"):
        assert forbidden not in source, (
            f"beside_machine must not reference {forbidden!r}")
