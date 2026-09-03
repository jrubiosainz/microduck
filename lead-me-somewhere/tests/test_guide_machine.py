"""The state machine: every transition rule, on hand-built inputs, no MuJoCo.

The machine is deliberately physics-free so that "a wait is caused, never
scheduled" and "a resume is justified, never assumed" can be tested directly
rather than inferred from a rollout.
"""

from __future__ import annotations

import pytest

from guide_layout import DESTINATION_KEYS
from guide_machine import GuideMachine
from guide_states import (
    ACK_SECONDS,
    CATCHUP_DISTANCE_M,
    CHECK_CONFIRM_S,
    FORBIDDEN_STATES,
    INDICATE_SECONDS,
    LAG_CONFIRM_S,
    LAG_COOLDOWN_S,
    LAG_DISTANCE_M,
    LOST_CONFIRM_S,
    PLAN_DWELL_S,
    RESUME_CONFIRM_S,
    STATES,
    ZERO_COMMAND_STATES,
)

DT = 0.02


class _Plan:
    """The smallest thing ``note_plan`` accepts."""
    length_m = 7.8
    bends = [{"hand": "left"}, {"hand": "right"}, {"hand": "right"}]


def machine() -> GuideMachine:
    m = GuideMachine(ctrl_hz=50.0)
    m.set_follower("mara")
    return m


def tick(m: GuideMachine, t: float, *, distance=0.6, visible=True,
         los=True, remaining=5.0, facing=False):
    return m.update(t, distance_m=distance, visible=visible,
                    los_available=los, route_remaining_m=remaining,
                    facing_ok=facing)


def lead_from(t0: float = 0.0) -> tuple[GuideMachine, float]:
    """Drive a machine to LEAD and return it with the current time."""
    m = machine()
    t = t0
    m.receive(t, "LIFTS", DESTINATION_KEYS)
    while m.state != "PLAN":
        t += DT
        tick(m, t)
    m.note_plan(t, _Plan())
    while m.state != "LEAD":
        t += DT
        tick(m, t)
    return m, t


# -- the request ------------------------------------------------------------

def test_the_declared_states_and_the_forbidden_ones_are_disjoint():
    assert not set(STATES) & set(FORBIDDEN_STATES)
    assert set(ZERO_COMMAND_STATES) <= set(STATES)


def test_a_request_is_resolved_by_exact_lookup():
    m = machine()
    m.receive(1.6, "LIFTS", DESTINATION_KEYS)
    assert m.destination.key == "LIFTS"
    assert m.requested_key == "LIFTS"
    assert m.request_t == 1.6
    assert tuple(m.candidates) == DESTINATION_KEYS


def test_an_unknown_request_raises():
    m = machine()
    with pytest.raises(KeyError):
        m.receive(1.6, "PLATFORM_9", DESTINATION_KEYS)


def test_a_second_different_request_is_refused_mid_route():
    """A guide that silently re-targeted would arrive somewhere nobody asked
    for while every log said it was obeying."""
    m = machine()
    m.receive(1.6, "LIFTS", DESTINATION_KEYS)
    with pytest.raises(ValueError) as excinfo:
        m.receive(9.0, "CAFE", DESTINATION_KEYS)
    assert "re-target" in str(excinfo.value)
    assert m.destination.key == "LIFTS"


def test_repeating_the_same_request_is_harmless():
    m = machine()
    m.receive(1.6, "LIFTS", DESTINATION_KEYS)
    m.receive(2.0, "LIFTS", DESTINATION_KEYS)
    assert m.request_t == 1.6


def test_the_follower_cannot_be_reassigned():
    m = machine()
    with pytest.raises(ValueError):
        m.set_follower("noor")


def test_the_duck_acknowledges_before_it_plans():
    m = machine()
    m.receive(1.6, "LIFTS", DESTINATION_KEYS)
    t = 1.6
    while t < 1.6 + ACK_SECONDS - DT:
        t += DT
        tick(m, t)
        assert m.state == "RECEIVE_DESTINATION"
    t += 2 * DT
    tick(m, t)
    assert m.state == "PLAN"


def test_plan_is_a_state_the_run_spends_time_in():
    """A state entered and left inside one tick is a state that did not
    happen: nothing could be seen and nothing could be graded."""
    m, _ = lead_from()
    plan_entry = next(x for x in m.transitions if x["to"] == "PLAN")
    lead_entry = next(x for x in m.transitions if x["to"] == "LEAD")
    assert lead_entry["t"] - plan_entry["t"] >= PLAN_DWELL_S - 1e-6


# -- detection --------------------------------------------------------------

def test_a_lag_must_be_sustained_before_it_counts():
    """One tick of a swinging arm is not an episode."""
    m, t = lead_from()
    far = LAG_DISTANCE_M + 0.3
    for _ in range(int(LAG_CONFIRM_S / DT) - 1):
        t += DT
        tick(m, t, distance=far)
        assert m.state == "LEAD", (
            "the machine acted before the lag window had elapsed")
    t += DT
    tick(m, t, distance=far)
    assert m.state == "CHECK_FOLLOWER"


def test_an_intermittent_lag_never_accumulates():
    m, t = lead_from()
    for _ in range(400):
        t += DT
        tick(m, t, distance=LAG_DISTANCE_M + 0.3)
        t += DT
        tick(m, t, distance=0.5)
    assert m.state == "LEAD"
    assert m.completed_episodes == 0


def test_a_loss_of_sight_is_its_own_cause():
    """The second episode in the scenario is a LOSS, not a distance event, so
    the detector has to fire on visibility alone."""
    m, t = lead_from()
    for _ in range(int(LOST_CONFIRM_S / DT) + 2):
        t += DT
        tick(m, t, distance=0.5, visible=False)
    assert m.state == "CHECK_FOLLOWER"
    assert m._episode["cause"] == "loss"


def test_the_lag_threshold_is_what_actually_gates_detection():
    """A mutation that widened the threshold must stop the episode firing."""
    m, t = lead_from()
    just_inside = LAG_DISTANCE_M - 0.01
    for _ in range(int(LAG_CONFIRM_S / DT) * 3):
        t += DT
        tick(m, t, distance=just_inside)
    assert m.state == "LEAD", "a distance under the threshold triggered a lag"


def test_arrival_wins_over_a_fresh_lag():
    """A duck at the destination should stop, not open an episode about
    somebody who is about to arrive too."""
    m, t = lead_from()
    t += DT
    tick(m, t, distance=LAG_DISTANCE_M + 1.0, remaining=0.0)
    assert m.state == "ARRIVE"


# -- waiting and resuming ---------------------------------------------------

def reach_wait(m: GuideMachine, t: float) -> float:
    far = LAG_DISTANCE_M + 0.3
    for _ in range(int(LAG_CONFIRM_S / DT) + 2):
        t += DT
        tick(m, t, distance=far)
    assert m.state == "CHECK_FOLLOWER"
    for _ in range(int(CHECK_CONFIRM_S / DT) + 2):
        t += DT
        tick(m, t, distance=far)
        m.confirm_check(t, looking_back=True, distance_m=far, visible=True,
                        bearing_ok=True)
    assert m.state == "WAIT_FOR_PERSON"
    return t


def test_the_check_needs_sustained_visual_contact():
    m, t = lead_from()
    far = LAG_DISTANCE_M + 0.3
    for _ in range(int(LAG_CONFIRM_S / DT) + 2):
        t += DT
        tick(m, t, distance=far)
    for _ in range(200):
        t += DT
        tick(m, t, distance=far, visible=False)
        m.confirm_check(t, looking_back=True, distance_m=far, visible=False,
                        bearing_ok=True)
        if m.state != "CHECK_FOLLOWER":
            break
    assert m.state == "CHECK_FOLLOWER" or "CHECK_FOLLOWER" in str(m.timeouts)


def test_a_resume_needs_BOTH_distance_and_visibility():
    """Distance alone would let the duck set off while she was behind a
    partition; visibility alone while she was still three metres back."""
    m, t = lead_from()
    t = reach_wait(m, t)

    # Close but unseen: no resume.
    for _ in range(int(RESUME_CONFIRM_S / DT) * 3):
        t += DT
        tick(m, t, distance=0.5, visible=False)
    assert m.state == "WAIT_FOR_PERSON"

    # Seen but far: still no resume.
    for _ in range(int(RESUME_CONFIRM_S / DT) * 3):
        t += DT
        tick(m, t, distance=CATCHUP_DISTANCE_M + 0.4, visible=True)
    assert m.state == "WAIT_FOR_PERSON"

    # Both, sustained: resume.
    for _ in range(int(RESUME_CONFIRM_S / DT) + 2):
        t += DT
        tick(m, t, distance=0.5, visible=True)
    assert m.state == "RESUME"


def test_a_resume_must_be_sustained_not_a_single_good_frame():
    m, t = lead_from()
    t = reach_wait(m, t)
    for _ in range(int(RESUME_CONFIRM_S / DT) - 3):
        t += DT
        tick(m, t, distance=0.5, visible=True)
        assert m.state == "WAIT_FOR_PERSON"
    t += DT
    tick(m, t, distance=3.0, visible=False)   # one bad frame resets it
    for _ in range(int(RESUME_CONFIRM_S / DT) - 3):
        t += DT
        tick(m, t, distance=0.5, visible=True)
    assert m.state == "WAIT_FOR_PERSON"


def test_the_episode_records_what_justified_the_resume():
    m, t = lead_from()
    t = reach_wait(m, t)
    for _ in range(int(RESUME_CONFIRM_S / DT) + 2):
        t += DT
        tick(m, t, distance=0.5, visible=True)
    assert m.completed_episodes == 1
    episode = m.episodes[0]
    assert episode["distance_at_resume_m"] <= CATCHUP_DISTANCE_M
    assert episode["visible_at_resume"] is True
    assert episode["recovered_for_s"] >= RESUME_CONFIRM_S
    assert episode["distance_at_detect_m"] > LAG_DISTANCE_M
    assert episode["wait_duration_s"] > 0.0


def test_an_episode_cannot_chatter():
    """Without a cooldown the episode count is a count of ticks."""
    m, t = lead_from()
    t = reach_wait(m, t)
    for _ in range(int(RESUME_CONFIRM_S / DT) + 2):
        t += DT
        tick(m, t, distance=0.5, visible=True)
    assert m.state == "RESUME"
    resumed_at = t
    far = LAG_DISTANCE_M + 0.5
    while t < resumed_at + LAG_COOLDOWN_S - 0.1:
        t += DT
        tick(m, t, distance=far)
    assert m.completed_episodes == 1
    assert m.state in ("RESUME", "LEAD")


# -- arrival ----------------------------------------------------------------

def test_the_duck_does_not_indicate_until_it_faces_the_destination():
    """A guide that announced arrival facing away from what it led somebody to
    has not arrived in any useful sense."""
    m, t = lead_from()
    t += DT
    tick(m, t, remaining=0.0)
    assert m.state == "ARRIVE"
    for _ in range(100):
        t += DT
        tick(m, t, remaining=0.0, facing=False)
    assert m.state == "ARRIVE"
    t += DT
    tick(m, t, remaining=0.0, facing=True)
    assert m.state == "INDICATE"


def test_the_indication_lasts_its_declared_duration():
    m, t = lead_from()
    t += DT
    tick(m, t, remaining=0.0, facing=True)
    t += DT
    tick(m, t, remaining=0.0, facing=True)
    assert m.state == "INDICATE"
    started = t
    while t < started + INDICATE_SECONDS - DT:
        t += DT
        tick(m, t, remaining=0.0, facing=True)
        assert m.state == "INDICATE"
    t += 2 * DT
    tick(m, t, remaining=0.0, facing=True)
    assert m.state == "DONE"
    assert m.arrival["destination"] == "LIFTS"


def test_done_is_terminal():
    m, t = lead_from()
    t += DT
    tick(m, t, remaining=0.0, facing=True)
    for _ in range(int((INDICATE_SECONDS + 2.0) / DT)):
        t += DT
        tick(m, t, remaining=0.0, facing=True)
    assert m.state == "DONE"
    for _ in range(200):
        t += DT
        tick(m, t, distance=9.0, visible=False)
    assert m.state == "DONE"


# -- ceilings move the machine ----------------------------------------------

def test_the_check_ceiling_transitions_rather_than_only_logging():
    """A ceiling that does not move the machine is not a ceiling.  An earlier
    draft only appended to ``timeouts`` and the run spent 71 s of 95 s stuck in
    CHECK_FOLLOWER emitting zero."""
    m, t = lead_from()
    far = LAG_DISTANCE_M + 0.3
    for _ in range(int(LAG_CONFIRM_S / DT) + 2):
        t += DT
        tick(m, t, distance=far)
    assert m.state == "CHECK_FOLLOWER"
    for _ in range(int(40.0 / DT)):
        t += DT
        tick(m, t, distance=far, visible=False)
        m.confirm_check(t, looking_back=False, distance_m=far, visible=False,
                        bearing_ok=False)
        if m.state != "CHECK_FOLLOWER":
            break
    assert m.state == "WAIT_FOR_PERSON"
    assert m._episode.get("squaring_up_incomplete") is True


def test_no_transition_invents_an_undeclared_state():
    m, t = lead_from()
    t = reach_wait(m, t)
    for _ in range(int(RESUME_CONFIRM_S / DT) + 2):
        t += DT
        tick(m, t, distance=0.5, visible=True)
    for _ in range(int(20.0 / DT)):
        t += DT
        tick(m, t, remaining=0.0, facing=True)
    visited = {x["to"] for x in m.transitions} | {x["from"] for x
                                                  in m.transitions}
    assert visited <= set(STATES)
    assert not visited & set(FORBIDDEN_STATES)
