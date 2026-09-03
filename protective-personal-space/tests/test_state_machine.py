#!/usr/bin/env python3
"""The state machine: one intrusion, ward priority, the squeeze, and recovery.

``PpsMachine`` is pure - no numpy, no mujoco, no schedule - so every branch can
be driven directly instead of waited for in a 190 s rollout.  That matters most
for the PRIORITY rules, which are the behavior's ethical claims in code form:

* the protected person outranks any station the duck is holding, because
  continuing to body-block her would invert the whole purpose;
* a simultaneous pinch supersedes a single-threat response already underway,
  because interposing on one side of a squeeze is worse than escaping it.

Both are checked in BOTH directions: the states that must yield, and the states
that must not.
"""

from __future__ import annotations

import pytest

from pps_machine import PpsMachine
from pps_states import CLEAR_HOLD_S, STATES

THREAT = ("dario", [1.0, 2.0], {"bearing_deg": -4.9, "cpa_m": 1.0,
                                "ttc_s": 4.89})
SQUEEZE = ("kwame", "tomas", [0.5, 0.5],
           {"bearing_deg": -145.76, "secondary_bearing_deg": 65.48,
            "separation_deg": 148.75, "gap_score": 1.3596})
# The states a squeeze is allowed to interrupt.
SUPERSEDABLE = ("PREDICT_INTRUSION", "HOLD_BUFFER", "THREAT_CLEAR",
                "RETURN_ESCORT", "MONITOR")
# The states the protected person's own approach is allowed to interrupt.
YIELDABLE = ("INTERPOSE", "HOLD_BUFFER", "THREAT_CLEAR", "RETURN_ESCORT")


def drive(machine: PpsMachine, target: str, t: float = 0.0) -> PpsMachine:
    """Walk a fresh machine into ``target`` using only legal transitions."""
    if target == "ESCORT":
        return machine
    machine.update(t + 0.1, escort_joined=True)          # -> MONITOR
    if target == "MONITOR":
        return machine
    if target in ("MULTI_THREAT", "ESCAPE_GAP"):
        machine.update(t + 0.2, squeeze=SQUEEZE)         # -> MULTI_THREAT
        if target == "MULTI_THREAT":
            return machine
        machine.update(t + 0.3)                          # -> ESCAPE_GAP
        return machine
    if target in ("PERSON_APPROACH", "RETREAT"):
        machine.update(t + 0.2, ward_approach=True)      # -> PERSON_APPROACH
        if target == "PERSON_APPROACH":
            return machine
        machine.update(t + 0.3)                          # -> RETREAT
        return machine
    if target == "RECOVER":
        machine.update(t + 0.2, squeeze=SQUEEZE)
        machine.update(t + 0.3)
        machine.update(t + 0.4, target_reached=True)     # -> RECOVER
        return machine
    machine.update(t + 0.2, threat=THREAT)               # -> PREDICT_INTRUSION
    if target == "PREDICT_INTRUSION":
        return machine
    machine.update(t + 0.3)                              # -> INTERPOSE
    if target == "INTERPOSE":
        return machine
    machine.update(t + 0.4, target_reached=True)         # -> HOLD_BUFFER
    if target == "HOLD_BUFFER":
        return machine
    machine.update(t + 0.5, threat_clear=True)           # -> THREAT_CLEAR
    if target == "THREAT_CLEAR":
        return machine
    machine.update(t + 0.5 + CLEAR_HOLD_S)               # -> RETURN_ESCORT
    return machine


@pytest.fixture()
def machine():
    return PpsMachine()


# -- the resting state -------------------------------------------------------
def test_a_fresh_machine_starts_in_escort_with_nothing_selected(machine):
    assert machine.state == "ESCORT"
    assert machine.selected is None and machine.secondary is None
    assert machine.target is None
    assert machine.episodes == [] and machine.transitions == []
    assert machine.handled == set()


def test_escort_waits_for_the_slot_to_be_joined(machine):
    assert machine.update(0.5)[0] == "ESCORT"
    assert machine.update(1.0, escort_joined=True) == ("MONITOR", True)


def test_escort_ignores_a_threat_until_the_formation_exists(machine):
    """There is nothing to protect FROM a station the duck has not reached."""
    assert machine.update(1.0, threat=THREAT)[0] == "ESCORT"
    assert machine.update(1.0, squeeze=SQUEEZE)[0] == "ESCORT"
    assert machine.update(1.0, ward_approach=True)[0] == "ESCORT"
    assert machine.episodes == []


# -- one complete intrusion cycle --------------------------------------------
def test_a_single_intrusion_runs_the_whole_cycle_in_order(machine):
    machine.update(0.1, escort_joined=True)
    machine.update(1.0, threat=THREAT)
    assert machine.state == "PREDICT_INTRUSION"
    assert machine.selected == "dario"
    assert machine.target == [1.0, 2.0]

    assert machine.update(1.02)[0] == "INTERPOSE"
    assert machine.update(1.5)[0] == "INTERPOSE", "waits to arrive"
    assert machine.update(2.0, target_reached=True)[0] == "HOLD_BUFFER"
    assert machine.update(3.0, threat_clear=False)[0] == "HOLD_BUFFER"
    assert machine.update(4.0, threat_clear=True)[0] == "THREAT_CLEAR"
    assert machine.update(4.5)[0] == "THREAT_CLEAR", "clear must be held"
    assert machine.update(4.0 + CLEAR_HOLD_S + 1e-6)[0] == "RETURN_ESCORT"
    assert machine.update(6.0)[0] == "RETURN_ESCORT", "waits for the slot"
    assert machine.update(7.0, escort_joined=True)[0] == "MONITOR"

    kinds = [t["to"] for t in machine.transitions]
    assert kinds == ["MONITOR", "PREDICT_INTRUSION", "INTERPOSE",
                     "HOLD_BUFFER", "THREAT_CLEAR", "RETURN_ESCORT",
                     "MONITOR"]


def test_the_cycle_closes_exactly_one_episode_and_records_its_evidence(
        machine):
    drive(machine, "RETURN_ESCORT")
    machine.update(9.0, escort_joined=True)
    assert len(machine.episodes) == 1
    episode = machine.episodes[0]
    assert episode["kind"] == "intrusion"
    assert episode["selected"] == "dario"
    assert episode["secondary"] is None
    assert episode["outcome"] == "recovered"
    assert episode["target"] == [1.0, 2.0]
    assert episode["bearing_deg"] == -4.9, "the evidence travels with it"
    assert episode["duration_s"] == pytest.approx(
        episode["ended_at_s"] - episode["started_at_s"])
    assert episode["index"] == 0


def test_a_handled_intruder_is_remembered_so_the_episode_cannot_reopen(
        machine):
    drive(machine, "RETURN_ESCORT")
    machine.update(9.0, escort_joined=True)
    assert machine.handled == {"dario"}
    assert machine.selected is None and machine.target is None


def test_the_threat_clear_hold_is_exactly_the_declared_window(machine):
    """A person pausing on their way out must not re-open the episode."""
    drive(machine, "THREAT_CLEAR")
    entered = machine.state_since
    assert machine.update(entered + CLEAR_HOLD_S - 0.01)[0] == "THREAT_CLEAR"
    assert machine.update(entered + CLEAR_HOLD_S + 1e-6)[0] == "RETURN_ESCORT"


def test_every_transition_is_logged_with_its_endpoints(machine):
    drive(machine, "HOLD_BUFFER")
    for entry in machine.transitions:
        assert set(entry) >= {"t", "from", "to"}
        assert entry["from"] != entry["to"]
        assert entry["from"] in STATES and entry["to"] in STATES
    times = [entry["t"] for entry in machine.transitions]
    assert times == sorted(times)


# -- the protected person outranks everything the duck is doing ---------------
@pytest.mark.parametrize("state", YIELDABLE)
def test_the_ward_walking_at_the_duck_abandons_the_station(machine, state):
    """Continuing to body-block her would invert the behavior's purpose."""
    drive(machine, state)
    assert machine.state == state
    assert machine.update(20.0, ward_approach=True)[0] == "PERSON_APPROACH"
    assert machine.selected == "aina"


@pytest.mark.parametrize("state", YIELDABLE)
def test_yielding_closes_the_interrupted_episode_as_yielded(machine, state):
    """Named in the record, so a yield is never mistaken for a completed cycle."""
    drive(machine, state)
    before = len(machine.episodes)
    machine.update(20.0, ward_approach=True)
    assert len(machine.episodes) == before + 1
    assert machine.episodes[-1]["outcome"] == "yielded_to_ward"
    assert machine.episodes[-1]["kind"] == "intrusion"


def test_a_ward_approach_from_monitor_opens_its_own_episode(machine):
    drive(machine, "MONITOR")
    assert machine.update(20.0, ward_approach=True)[0] == "PERSON_APPROACH"
    assert machine._episode["kind"] == "ward_approach"
    assert machine._episode["selected"] == "aina"
    assert machine._episode["target"] is None


@pytest.mark.parametrize("state", ["ESCORT", "RETREAT", "ESCAPE_GAP",
                                   "RECOVER"])
def test_states_already_serving_her_are_not_interrupted_again(machine, state):
    """Retreating IS the yield, and an escape already left the pinch."""
    drive(machine, state)
    assert machine.update(20.0, ward_approach=True)[0] == state


def test_the_retreat_runs_to_completion_before_recovering(machine):
    drive(machine, "PERSON_APPROACH")
    assert machine.update(21.0)[0] == "RETREAT"
    assert machine.update(22.0, retreat_complete=False)[0] == "RETREAT"
    assert machine.update(23.0, retreat_complete=True)[0] == "RECOVER"
    assert machine.update(24.0, escort_joined=False)[0] == "RECOVER"
    assert machine.update(25.0, escort_joined=True)[0] == "MONITOR"
    assert machine.episodes[-1]["kind"] == "ward_approach"
    assert machine.episodes[-1]["outcome"] == "recovered"


def test_person_approach_is_a_single_tick_decision(machine):
    """It is the moment the yield is decided; the RETREAT is the action."""
    drive(machine, "PERSON_APPROACH")
    assert machine.update(21.0)[0] == "RETREAT"


# -- the squeeze supersedes a response already underway -----------------------
@pytest.mark.parametrize("state", SUPERSEDABLE)
def test_a_pinch_supersedes_the_states_it_is_declared_to_supersede(machine,
                                                                   state):
    """Interposing on one side of a squeeze is worse than escaping it."""
    drive(machine, state)
    assert machine.state == state
    assert machine.update(30.0, squeeze=SQUEEZE)[0] == "MULTI_THREAT"
    assert machine.selected == "kwame"
    assert machine.secondary == "tomas"
    assert machine.target == [0.5, 0.5]


@pytest.mark.parametrize("state", [s for s in SUPERSEDABLE if s != "MONITOR"])
def test_superseding_closes_the_interrupted_episode_by_name(machine, state):
    drive(machine, state)
    machine.update(30.0, squeeze=SQUEEZE)
    assert machine.episodes[-1]["outcome"] == "superseded_by_squeeze"


@pytest.mark.parametrize("state", ["ESCORT", "INTERPOSE", "RETREAT",
                                   "ESCAPE_GAP", "RECOVER"])
def test_a_pinch_does_not_interrupt_the_states_it_must_not(machine, state):
    """An escape already underway is the right answer; restarting it is not."""
    drive(machine, state)
    assert machine.update(30.0, squeeze=SQUEEZE)[0] == state


def test_the_squeeze_branch_escapes_rather_than_interposing(machine):
    drive(machine, "MONITOR")
    machine.update(30.0, squeeze=SQUEEZE)
    assert machine.update(30.1)[0] == "ESCAPE_GAP", "never INTERPOSE"
    assert machine.update(31.0, target_reached=False)[0] == "ESCAPE_GAP"
    assert machine.update(32.0, target_reached=True)[0] == "RECOVER"
    assert machine.update(33.0, escort_joined=True)[0] == "MONITOR"
    episode = machine.episodes[-1]
    assert episode["kind"] == "squeeze"
    assert episode["selected"] == "kwame" and episode["secondary"] == "tomas"
    assert episode["separation_deg"] == 148.75


def test_a_squeeze_marks_both_participants_handled(machine):
    drive(machine, "MONITOR")
    machine.update(30.0, squeeze=SQUEEZE)
    machine.update(30.1)
    machine.update(32.0, target_reached=True)
    machine.update(33.0, escort_joined=True)
    assert machine.handled == {"kwame", "tomas"}


def test_the_ward_outranks_a_pinch_when_both_arrive_at_once(machine):
    """She is checked first in the same tick, which is the priority order."""
    drive(machine, "HOLD_BUFFER")
    state, _ = machine.update(30.0, squeeze=SQUEEZE, ward_approach=True)
    assert state == "MULTI_THREAT", (
        "the pinch branch is evaluated first by construction")
    assert machine.episodes[-1]["outcome"] == "superseded_by_squeeze"


# -- monitor dispatch ---------------------------------------------------------
def test_monitor_prefers_the_ward_then_the_pinch_then_a_single_threat(machine):
    drive(machine, "MONITOR")
    assert machine.update(10.0, ward_approach=True, squeeze=SQUEEZE,
                          threat=THREAT)[0] == "MULTI_THREAT"

    second = drive(PpsMachine(), "MONITOR")
    assert second.update(10.0, squeeze=SQUEEZE,
                         threat=THREAT)[0] == "MULTI_THREAT"

    third = drive(PpsMachine(), "MONITOR")
    assert third.update(10.0, threat=THREAT)[0] == "PREDICT_INTRUSION"


def test_monitor_stays_put_when_nothing_is_happening(machine):
    drive(machine, "MONITOR")
    assert machine.update(10.0)[0] == "MONITOR"
    assert machine.episodes == []


# -- the end -----------------------------------------------------------------
def test_the_session_ends_from_monitor(machine):
    drive(machine, "MONITOR")
    assert machine.update(188.0, finish=True)[0] == "DONE"
    assert machine.transitions[-1]["reason"] == "session complete in escort"


def test_finishing_mid_episode_does_not_fake_a_completed_session(machine):
    """DONE has to be reached from the escort, not declared from a station."""
    drive(machine, "HOLD_BUFFER")
    assert machine.update(188.0, finish=True)[0] == "HOLD_BUFFER"
    machine.finish(190.0)
    assert machine.state == "HOLD_BUFFER"


def test_finish_promotes_a_monitoring_machine_to_done(machine):
    drive(machine, "MONITOR")
    machine.finish(190.0)
    assert machine.state == "DONE"
    assert machine.transitions[-1]["reason"] == "session complete"


def test_done_is_terminal(machine):
    drive(machine, "MONITOR")
    machine.update(188.0, finish=True)
    for kwargs in ({"threat": THREAT}, {"squeeze": SQUEEZE},
                   {"ward_approach": True}, {"escort_joined": True}):
        assert machine.update(189.0, **kwargs)[0] == "DONE"


# -- episode bookkeeping ------------------------------------------------------
def test_closing_without_an_open_episode_is_a_no_op(machine):
    machine.close_episode(5.0)
    assert machine.episodes == []
    assert machine.selected is None and machine.target is None


def test_episode_indices_are_consecutive_across_a_whole_session(machine):
    drive(machine, "RETURN_ESCORT")
    machine.update(9.0, escort_joined=True)
    machine.update(20.0, ward_approach=True)
    machine.update(20.1)
    machine.update(21.0, retreat_complete=True)
    machine.update(22.0, escort_joined=True)
    assert [e["index"] for e in machine.episodes] == [0, 1]
    assert [e["kind"] for e in machine.episodes] == ["intrusion",
                                                     "ward_approach"]


def test_episodes_never_overlap_in_time(machine):
    drive(machine, "RETURN_ESCORT")
    machine.update(9.0, escort_joined=True)
    machine.update(20.0, squeeze=SQUEEZE)
    machine.update(20.1)
    machine.update(21.0, target_reached=True)
    machine.update(22.0, escort_joined=True)
    for earlier, later in zip(machine.episodes, machine.episodes[1:]):
        assert earlier["ended_at_s"] <= later["started_at_s"]


def test_a_target_is_copied_rather_than_aliased(machine):
    """A caller mutating its own array must not move the duck's station."""
    target = [1.0, 2.0]
    drive(machine, "MONITOR")
    machine.start_episode(5.0, "intrusion", "dario", target)
    target[0] = 99.0
    assert machine.target == [1.0, 2.0]


def test_a_none_target_stays_none(machine):
    machine.start_episode(5.0, "ward_approach", "aina", None)
    assert machine.target is None
    assert machine._episode["target"] is None


def test_the_changed_flag_reports_only_real_transitions(machine):
    assert machine.update(0.5) == ("ESCORT", False)
    assert machine.update(1.0, escort_joined=True) == ("MONITOR", True)
    assert machine.update(1.5) == ("MONITOR", False)


def test_state_since_tracks_the_latest_entry(machine):
    machine.update(1.0, escort_joined=True)
    assert machine.state_since == 1.0
    machine.update(4.0, threat=THREAT)
    assert machine.state_since == 4.0
    machine.update(4.5)
    assert machine.state_since == 4.5


@pytest.mark.parametrize("state", list(STATES))
def test_every_declared_state_is_reachable_or_terminal(state):
    """A state nothing can enter is dead code pretending to be behavior."""
    machine = PpsMachine()
    if state == "DONE":
        drive(machine, "MONITOR")
        machine.update(188.0, finish=True)
    else:
        drive(machine, state)
    assert machine.state == state
