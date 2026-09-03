#!/usr/bin/env python3
"""The state machine: losing somebody, refusing look-alikes, and rejoining.

The machine never touches physics and never emits a command, so every
transition rule is exercised on hand-built inputs with no MuJoCo anywhere.

THREE TRANSITIONS CARRY THE BEHAVIOR AND ARE PINNED INDIVIDUALLY: a loss is
declared only after SUSTAINED invisibility, so one stride of somebody crossing
the sightline cannot become one; a look-alike can never reach REACQUIRED however
long it is confirmed, because the name is checked at the transition and not only
in the scorer; and the guardian is NOT exempt from the confirmation duration, so
it is the same gate for everybody.

The controller that turns those states into commands is graded in
``test_lost_controller``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lost_cast import GUARDIAN, WEIGHTS  # noqa: E402
from lost_constants import (  # noqa: E402
    LOSS_CONFIRM_S,
    MOVING_STATES,
    REACQUIRE_CONFIRM_S,
    REJECT_HOLD_S,
    REJOIN_MAX_S,
    SEARCH_MAX_S,
    STATES,
    STATIONARY_STATES,
)
from lost_identity import IdentityTracker, Sighting  # noqa: E402
from lost_machine import STOP_S, LostMachine  # noqa: E402

DT = 0.02


def _machine() -> LostMachine:
    machine = LostMachine(ctrl_hz=50.0)
    machine.set_guardian(GUARDIAN.name)
    return machine


def _tracker() -> IdentityTracker:
    return IdentityTracker(reference=GUARDIAN.descriptor(),
                           guardian=GUARDIAN.name, dt=DT)


def _sighting(name: str, score: float, t: float = 0.0,
              verdict: str = "candidate") -> Sighting:
    return Sighting(name, t, score, {}, tuple(sorted(WEIGHTS)), True, 1.2, 4.0,
                    verdict, "synthetic")


def _drive(machine, steps, *, t0=0.0, visible=False, confirmed=0.0,
           candidate=None, reached=False, tracker=None):
    """Advance the machine ``steps`` ticks with fixed inputs; return the states."""
    tracker = tracker or _tracker()
    seen = []
    for step in range(steps):
        state, _ = machine.update(
            t0 + step * DT, guardian_visible=visible,
            guardian_confirmed_s=confirmed, best_candidate=candidate,
            reached_goal=reached, tracker=tracker)
        seen.append(state)
    return seen


# ------------------------------------------------------------ the state set
def test_every_state_is_either_stationary_or_moving_and_never_both():
    """Only FOLLOW and REJOIN may move; every searching state is stationary."""
    assert set(STATIONARY_STATES) | set(MOVING_STATES) == set(STATES)
    assert set(STATIONARY_STATES) & set(MOVING_STATES) == set()
    assert set(MOVING_STATES) == {"FOLLOW", "REJOIN"}
    for state in ("LOST", "STOP", "SEARCH_SWEEP", "CANDIDATE", "REJECT",
                  "REACQUIRED"):
        assert state in STATIONARY_STATES


# ------------------------------------------------------- guardian immutable
def test_the_guardian_identity_is_set_once_and_is_then_immutable():
    """Identity continuity IS the behavior; a silent swap would void every gate."""
    machine = _machine()
    assert machine.guardian == GUARDIAN.name
    machine.set_guardian(GUARDIAN.name)      # idempotent re-assert is fine
    with pytest.raises(ValueError, match="cannot be"):
        machine.set_guardian("mira")
    assert machine.guardian == GUARDIAN.name


# ------------------------------------------------------------- the loss gate
def test_a_brief_glimpse_of_invisibility_is_not_a_loss():
    """A single stride of somebody crossing the sightline must not declare LOST."""
    machine = _machine()
    _drive(machine, int(LOSS_CONFIRM_S / DT) - 1, visible=False)
    assert machine.state == "FOLLOW"


def test_loss_is_declared_only_after_the_sustained_window():
    machine = _machine()
    states = _drive(machine, int(LOSS_CONFIRM_S / DT) + 2, visible=False)
    assert "LOST" in states
    assert states.index("LOST") == int(LOSS_CONFIRM_S / DT) - 1


def test_the_invisibility_clock_resets_on_any_sighting():
    machine = _machine()
    _drive(machine, int(LOSS_CONFIRM_S / DT) - 1, visible=False)
    _drive(machine, 1, visible=True)
    assert machine._invisible_for == 0.0
    _drive(machine, int(LOSS_CONFIRM_S / DT) - 1, visible=False)
    assert machine.state == "FOLLOW"


def test_the_loss_confirm_window_is_the_documented_value():
    assert LOSS_CONFIRM_S == 0.60
    assert REACQUIRE_CONFIRM_S == 0.90
    assert REACQUIRE_CONFIRM_S > LOSS_CONFIRM_S


def test_the_recorded_loss_names_how_long_she_was_invisible():
    machine = _machine()
    _drive(machine, int(LOSS_CONFIRM_S / DT) + 1, visible=False)
    transition = next(t for t in machine.transitions if t["to"] == "LOST")
    assert "not visible for" in transition["reason"]


# ------------------------------------------------------ the search sequence
def test_lost_lasts_one_tick_and_hands_over_to_a_deliberate_halt():
    machine = _machine()
    _drive(machine, int(LOSS_CONFIRM_S / DT) + 2, visible=False)
    assert machine.state == "STOP"


def test_the_halt_runs_its_full_duration_before_the_sweep_begins():
    machine = _machine()
    _drive(machine, int(LOSS_CONFIRM_S / DT) + 2, visible=False)
    t0 = machine.state_since
    _drive(machine, int(STOP_S / DT) - 2, t0=t0 + DT, visible=False)
    assert machine.state == "STOP"
    _drive(machine, 3, t0=t0 + STOP_S, visible=False)
    assert machine.state == "SEARCH_SWEEP"


def test_a_candidate_entering_the_frame_interrupts_the_sweep():
    machine = _machine()
    machine.state, machine.state_since = "SEARCH_SWEEP", 0.0
    _drive(machine, 1, t0=1.0, candidate=_sighting("mira", 0.86, 1.0))
    assert machine.state == "CANDIDATE"
    assert machine.candidate == "mira"


def test_a_look_alike_that_cannot_confirm_is_refused_after_the_window():
    machine = _machine()
    machine.state, machine.state_since = "SEARCH_SWEEP", 0.0
    _drive(machine, 1, t0=1.0, candidate=_sighting("mira", 0.86, 1.0))
    _drive(machine, int(REACQUIRE_CONFIRM_S / DT) + 2, t0=1.0 + DT,
           candidate=_sighting("mira", 0.86, 1.0))
    assert machine.state == "REJECT"


def test_a_candidate_that_leaves_the_frame_is_refused_immediately():
    machine = _machine()
    machine.state, machine.state_since = "SEARCH_SWEEP", 0.0
    _drive(machine, 1, t0=1.0, candidate=_sighting("sofia", 0.82, 1.0))
    _drive(machine, 1, t0=1.02, candidate=None)
    assert machine.state == "REJECT"


def test_the_refusal_is_held_long_enough_to_be_a_visible_decision():
    machine = _machine()
    machine.state, machine.state_since = "REJECT", 0.0
    machine.candidate = "mira"
    _drive(machine, int(REJECT_HOLD_S / DT) - 2, t0=DT)
    assert machine.state == "REJECT"
    _drive(machine, 3, t0=REJECT_HOLD_S)
    assert machine.state == "SEARCH_SWEEP"
    assert machine.candidate is None


def test_the_guardian_slipping_out_before_confirming_is_not_a_refusal():
    """A failed confirmation of her own guardian must not count as a rejection."""
    machine = _machine()
    machine.state, machine.state_since = "SEARCH_SWEEP", 0.0
    _drive(machine, 1, t0=1.0, candidate=_sighting(GUARDIAN.name, 1.0, 1.0))
    assert machine.candidate == GUARDIAN.name
    _drive(machine, 1, t0=1.02, candidate=None)
    assert machine.state == "SEARCH_SWEEP"
    assert machine.rejections == []


def test_the_guardian_is_promoted_only_after_the_confirm_duration():
    machine = _machine()
    machine.state, machine.state_since = "SEARCH_SWEEP", 0.0
    _drive(machine, 1, t0=1.0, candidate=_sighting(GUARDIAN.name, 1.0, 1.0))
    _drive(machine, 5, t0=1.02, candidate=_sighting(GUARDIAN.name, 1.0),
           confirmed=REACQUIRE_CONFIRM_S - 0.1)
    assert machine.state == "CANDIDATE"
    _drive(machine, 1, t0=1.2, candidate=_sighting(GUARDIAN.name, 1.0),
           confirmed=REACQUIRE_CONFIRM_S)
    assert machine.state == "REACQUIRED"


def test_a_look_alike_can_never_reach_reacquired_however_long_it_is_confirmed():
    """The name check is applied at the transition, not only in the scorer."""
    machine = _machine()
    machine.state, machine.state_since = "SEARCH_SWEEP", 0.0
    _drive(machine, 1, t0=1.0, candidate=_sighting("mira", 0.99, 1.0))
    _drive(machine, 60, t0=1.02, candidate=_sighting("mira", 0.99),
           confirmed=10.0)
    assert machine.state != "REACQUIRED"
    assert machine.state in ("REJECT", "SEARCH_SWEEP")


def test_reacquired_lasts_one_tick_and_hands_over_to_the_rejoin():
    machine = _machine()
    machine.state, machine.state_since = "REACQUIRED", 0.0
    _drive(machine, 1, t0=DT, confirmed=REACQUIRE_CONFIRM_S)
    assert machine.state == "REJOIN"


# ---------------------------------------------------------------- the rejoin
def test_arriving_at_the_standoff_closes_the_cycle_and_resumes_following():
    machine = _machine()
    machine._cycle = {"index": 0, "lost_at_s": 10.0, "rejections": []}
    machine.state, machine.state_since = "REJOIN", 10.0
    _drive(machine, 1, t0=20.0, reached=True)
    assert machine.state == "FOLLOW"
    assert len(machine.cycles) == 1
    assert machine.cycles[0]["outcome"] == "rejoined"


def test_the_rejoin_ceiling_closes_the_cycle_as_a_timeout():
    machine = _machine()
    machine._cycle = {"index": 0, "lost_at_s": 0.0, "rejections": []}
    machine.state, machine.state_since = "REJOIN", 0.0
    _drive(machine, 1, t0=REJOIN_MAX_S + DT, reached=False)
    assert machine.state == "FOLLOW"
    assert machine.timeouts and machine.timeouts[0].startswith("REJOIN@")
    assert machine.cycles[0]["outcome"] == "timeout"


def test_the_search_ceiling_is_recorded_rather_than_hanging():
    machine = _machine()
    machine.state, machine.state_since = "SEARCH_SWEEP", 0.0
    _drive(machine, 1, t0=SEARCH_MAX_S + DT, candidate=None)
    assert machine.timeouts and machine.timeouts[0].startswith("SEARCH_SWEEP@")


def test_the_invisibility_clock_is_cleared_when_following_resumes():
    """Otherwise a stale count would re-declare a loss the instant it resumed."""
    machine = _machine()
    machine._invisible_for = 0.58
    machine._cycle = {"index": 0, "lost_at_s": 1.0, "rejections": []}
    machine.state, machine.state_since = "REJOIN", 1.0
    _drive(machine, 1, t0=5.0, reached=True)
    assert machine._invisible_for == 0.0


def test_a_refusal_is_attached_to_the_cycle_it_happened_in():
    machine = _machine()
    machine._cycle = {"index": 0, "lost_at_s": 5.0, "rejections": []}
    machine.note_rejection({"name": "mira", "score": 0.86})
    assert machine._cycle["rejections"] == [{"name": "mira", "score": 0.86}]
    assert machine.rejections == [{"name": "mira", "score": 0.86}]


def test_finishing_from_follow_parks_the_run_at_a_safe_standoff():
    machine = _machine()
    machine.finish(60.0)
    assert machine.state == "SAFE"


def test_finishing_mid_rejoin_closes_that_cycle_rather_than_dropping_it():
    machine = _machine()
    machine._cycle = {"index": 0, "lost_at_s": 40.0, "rejections": []}
    machine.state, machine.state_since = "REJOIN", 40.0
    machine.finish(60.0)
    assert machine.state == "SAFE"
    assert machine.cycles[0]["outcome"] == "ended in rejoin"


def test_the_summary_reports_the_guardian_and_the_cycle_count():
    machine = _machine()
    machine._cycle = {"index": 0, "lost_at_s": 1.0, "rejections": []}
    machine.state, machine.state_since = "REJOIN", 1.0
    _drive(machine, 1, t0=5.0, reached=True)
    summary = machine.summary()
    assert summary["guardian"] == GUARDIAN.name
    assert summary["cycle_count"] == 1
    assert summary["timeouts"] == []


