#!/usr/bin/env python3
"""The identity tracker: confirmation as a duration, and refusals that stick.

Pure logic on hand-built sightings.  What is graded here is the bookkeeping that
turns per-frame scores into a decision:

* a lock is a CONTINUOUS accept-grade duration, never a single frame;
* an accept of somebody who is not the guardian is recorded the instant it
  happens, before any duration logic, so a wrong lock cannot escape the counter
  by being brief;
* a refused body stays refused for a cooldown, so the refusal count counts
  PEOPLE rather than sweep ticks.

The per-sighting scoring rules those decisions consume are graded in
``test_identity_logic``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lost_cast import BY_NAME, GUARDIAN, WEIGHTS  # noqa: E402
from lost_constants import CANDIDATE_MIN_S, REJECT_COOLDOWN_S  # noqa: E402
from lost_identity import IdentityTracker, Sighting, evaluate  # noqa: E402

REFERENCE = GUARDIAN.descriptor()


def _entry(person, *, visible=True, readable=None, off_axis=4.0, range_m=1.2):
    """A camera record for ``person`` with a chosen readable-feature set."""
    readable = tuple(sorted(WEIGHTS if readable is None else readable))
    observed = {}
    if "shirt" in readable:
        observed["shirt"] = person.shirt
    if "stature" in readable:
        observed["stature"] = person.height_m
    if "cap" in readable:
        observed["cap"] = person.cap
    if "satchel" in readable:
        observed["satchel"] = person.satchel
    return {"visible": visible, "readable": list(readable),
            "observed": observed, "off_axis_deg": off_axis, "range_m": range_m}


# --------------------------------------------------- the confirmation gate
def _tracker(dt: float = 0.02) -> IdentityTracker:
    return IdentityTracker(reference=REFERENCE, guardian=GUARDIAN.name, dt=dt)


def _accept(name: str, t: float = 0.0) -> Sighting:
    return evaluate(name, t, REFERENCE, _entry(BY_NAME[name]))


def test_confirmation_is_a_duration_and_not_an_instant():
    """One accept-grade frame must not be enough to reacquire."""
    tracker = _tracker()
    held = tracker.confirm(_accept(GUARDIAN.name))
    assert held == pytest.approx(0.02)
    assert held < 0.90


def test_the_confirm_clock_reaches_the_documented_reacquire_window():
    from lost_constants import REACQUIRE_CONFIRM_S
    tracker = _tracker()
    held = 0.0
    for step in range(45):
        held = tracker.confirm(_accept(GUARDIAN.name, step * 0.02))
    assert held == pytest.approx(REACQUIRE_CONFIRM_S, abs=1e-9)
    assert REACQUIRE_CONFIRM_S == 0.90


def test_a_single_non_accept_frame_resets_the_confirm_clock():
    """The duration must be CONTINUOUS, not cumulative."""
    tracker = _tracker()
    for step in range(30):
        tracker.confirm(_accept(GUARDIAN.name, step * 0.02))
    assert tracker.confirm_time > 0.5
    tracker.confirm(evaluate(GUARDIAN.name, 1.0, REFERENCE,
                             _entry(GUARDIAN, visible=False)))
    assert tracker.confirm_time == 0.0


def test_the_confirm_clock_resets_when_the_accepting_body_changes():
    tracker = _tracker()
    for step in range(20):
        tracker.confirm(_accept(GUARDIAN.name, step * 0.02))
    before = tracker.confirm_time
    other = Sighting("mira", 1.0, 0.95, {}, tuple(WEIGHTS), True, 1.0, 2.0,
                     "accept", "forced")
    assert tracker.confirm(other) == pytest.approx(0.02)
    assert before > 0.02


def test_an_accept_of_the_wrong_person_is_recorded_immediately():
    """Recorded before any duration logic, so a brief one still counts."""
    tracker = _tracker()
    forged = Sighting("mira", 1.0, 0.95, {}, tuple(WEIGHTS), True, 1.0, 2.0,
                      "accept", "forged")
    tracker.confirm(forged)
    assert len(tracker.wrong_accepts) == 1
    assert tracker.wrong_accepts[0]["name"] == "mira"


def test_confirming_the_guardian_never_records_a_wrong_accept():
    tracker = _tracker()
    for step in range(60):
        tracker.confirm(_accept(GUARDIAN.name, step * 0.02))
    assert tracker.wrong_accepts == []


# --------------------------------------------------------------- cooldowns
def test_a_refused_candidate_is_not_re_scored_during_the_cooldown():
    """Otherwise the refusal count counts ticks rather than people."""
    tracker = _tracker()
    tracker.reject(_accept("mira", 10.0), 10.0)
    assert tracker.on_cooldown("mira", 10.0 + REJECT_COOLDOWN_S - 0.1)
    assert not tracker.on_cooldown("mira", 10.0 + REJECT_COOLDOWN_S + 0.1)
    assert not tracker.on_cooldown("sofia", 10.0)


def test_distinct_rejected_counts_people_not_ticks():
    tracker = _tracker()
    for t in (10.0, 11.0, 12.0):
        tracker.reject(_accept("mira", t), t)
    tracker.reject(_accept("sofia", 20.0), 20.0)
    assert tracker.distinct_rejected() == ("mira", "sofia")
    assert len(tracker.rejections) == 4


def test_a_body_clipped_by_one_sweep_tick_never_becomes_a_candidate():
    """CANDIDATE_MIN_S of continuous visibility is required first."""
    tracker = _tracker()
    tracker.note_visible("mira", True)
    assert not tracker.ready_to_evaluate("mira")
    for _ in range(int(CANDIDATE_MIN_S / 0.02)):
        tracker.note_visible("mira", True)
    assert tracker.ready_to_evaluate("mira")


def test_a_gap_in_visibility_resets_the_visible_clock():
    tracker = _tracker()
    for _ in range(20):
        tracker.note_visible("mira", True)
    assert tracker.seen_time["mira"] > CANDIDATE_MIN_S
    tracker.note_visible("mira", False)
    assert tracker.seen_time["mira"] == 0.0


def test_the_guardian_is_not_exempt_from_the_visible_time_requirement():
    """A shortcut for her would make the confirmation gate untestable."""
    tracker = _tracker()
    people = {GUARDIAN.name: _entry(GUARDIAN)}
    assert tracker.best_candidate(1.0, people) is None
    for _ in range(int(CANDIDATE_MIN_S / 0.02)):
        tracker.best_candidate(1.0, people)
    best = tracker.best_candidate(1.0, people)
    assert best is not None and best.name == GUARDIAN.name


def test_the_strongest_evaluable_sighting_wins_the_tick():
    tracker = _tracker()
    people = {GUARDIAN.name: _entry(GUARDIAN),
              "mira": _entry(BY_NAME["mira"]),
              "sofia": _entry(BY_NAME["sofia"])}
    for _ in range(int(CANDIDATE_MIN_S / 0.02) + 1):
        best = tracker.best_candidate(1.0, people)
    assert best is not None and best.name == GUARDIAN.name


def test_a_body_on_cooldown_is_skipped_even_when_it_is_the_strongest():
    tracker = _tracker()
    people = {"mira": _entry(BY_NAME["mira"])}
    for _ in range(int(CANDIDATE_MIN_S / 0.02) + 1):
        tracker.best_candidate(1.0, people)
    tracker.reject(_accept("mira", 1.0), 1.0)
    assert tracker.best_candidate(1.5, people) is None
