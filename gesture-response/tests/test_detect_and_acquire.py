#!/usr/bin/env python3
"""The detector's two gates, on hand-built percepts.  No MuJoCo in this file.

The rollout tests grade what the detector concluded on one real run.  These
grade the RULES, on inputs chosen to isolate one at a time - including the cases
that run did not happen to contain.

THE INTERRUPT WINDOW GETS THE MOST ATTENTION HERE, because it is the one place
the behavior deliberately keeps reading while carrying out a command, and a
mistake there would let the duck be re-tasked mid-manoeuvre by anybody.
"""

from __future__ import annotations

import numpy as np
import pytest

from gest_acquire import Acquisition, HandTrack
from gest_detect import GestureDetector
from gest_gesture import MOTION_WINDOW_S
from gest_states import ACQUIRE_CONFIRM_S, CONFIRM_S, INTERRUPT_COMMAND

DT = 0.02
INSTRUCTOR = "mira"
STRANGER = "teo"


def seen(*names, arm=True):
    """A camera view in which each named person is visible and readable."""
    return {name: {"visible": True, "present": True,
                   "arm_readable": {"l": arm, "r": arm}} for name in names}


def unseen(*names):
    return {name: {"visible": False, "present": True,
                   "arm_readable": {"l": False, "r": False}} for name in names}


# -- acquisition: WHO -------------------------------------------------------
def test_lock_requires_the_measured_dwell():
    """A graze must not lock; only sustained visibility may."""
    acquisition = Acquisition(wanted=INSTRUCTOR)
    ticks = int(ACQUIRE_CONFIRM_S / DT)
    for index in range(ticks - 1):
        acquisition.feed(index * DT, DT, seen(INSTRUCTOR))
        assert acquisition.locked == "", "locked before the dwell completed"
    acquisition.feed(ticks * DT, DT, seen(INSTRUCTOR))
    assert acquisition.locked == INSTRUCTOR
    assert acquisition.state == "locked"


def test_visibility_lost_resets_the_dwell():
    acquisition = Acquisition(wanted=INSTRUCTOR)
    for index in range(int(ACQUIRE_CONFIRM_S / DT) - 1):
        acquisition.feed(index * DT, DT, seen(INSTRUCTOR))
    acquisition.feed(99 * DT, DT, unseen(INSTRUCTOR))
    assert acquisition.state == "search"
    assert acquisition.visible_s == 0.0


def test_only_the_requested_identity_can_lock():
    """A stranger in frame for the whole session must never become subject."""
    acquisition = Acquisition(wanted=INSTRUCTOR)
    for index in range(400):
        acquisition.feed(index * DT, DT, seen(STRANGER))
    assert acquisition.locked == ""
    assert STRANGER in acquisition.seen, (
        "the stranger must be RECORDED as seen, or 'it chose among several' "
        "is not a claim about anything")


def test_the_lock_is_permanent_once_made():
    acquisition = Acquisition(wanted=INSTRUCTOR)
    for index in range(int(ACQUIRE_CONFIRM_S / DT) + 2):
        acquisition.feed(index * DT, DT, seen(INSTRUCTOR))
    assert acquisition.locked == INSTRUCTOR
    for index in range(200):
        acquisition.feed((100 + index) * DT, DT, seen(STRANGER))
    assert acquisition.locked == INSTRUCTOR, "the lock moved to somebody else"


# -- the hand history -------------------------------------------------------
def test_the_motion_window_must_fill_before_it_is_trusted():
    """A partial window under-reports path and would make motion look still."""
    track = HandTrack(dt=DT)
    for index in range(4):
        track.push(np.array([0.0, 0.0, float(index)]),
                   np.array([0.0, 0.0, 0.0]))
    _, _, full = track.features(0.4)
    assert not full
    for index in range(int(MOTION_WINDOW_S / DT) + 2):
        track.push(np.array([0.0, 0.0, float(index)]),
                   np.array([0.0, 0.0, 0.0]))
    _, _, full = track.features(0.4)
    assert full


def test_wander_separates_an_oscillation_from_a_one_way_raise():
    """The measurement the whole COME confirmation turns on."""
    capacity = int(MOTION_WINDOW_S / DT) + 1
    rising = HandTrack(dt=DT)
    for index in range(capacity):
        rising.push(np.array([0.0, 0.0, 0.01 * index]), np.zeros(3))
    _, raise_wander, _ = rising.features(0.4)

    swinging = HandTrack(dt=DT)
    for index in range(capacity):
        offset = 0.05 * (1 if (index // 8) % 2 == 0 else -1)
        swinging.push(np.array([0.0, 0.0, offset]), np.zeros(3))
    _, swing_wander, _ = swinging.features(0.4)

    assert raise_wander < swing_wander, (
        f"a one-way raise ({raise_wander:.2f}) must wander LESS than an "
        f"oscillation ({swing_wander:.2f})")
    assert raise_wander == pytest.approx(1.0, abs=0.15), (
        "a monotonic raise should measure about 1.0")


def test_the_window_clears_on_resume():
    """A window spanning a suspension would blend two different gestures."""
    detector = GestureDetector(DT, INSTRUCTOR)
    track = detector._track_for(INSTRUCTOR)
    for index in range(20):
        track.push(np.array([0.0, 0.0, float(index)]), np.zeros(3))
    detector.suspend()
    detector.resume()
    _, _, full = detector.tracks[INSTRUCTOR].features(0.4)
    assert not full, "the hand history survived a suspension"


# -- the interrupt window ---------------------------------------------------
def test_a_plain_suspension_reads_nothing():
    detector = GestureDetector(DT, INSTRUCTOR)
    detector.suspend()
    assert detector.suspended
    assert not detector.interrupt_only


def test_an_interrupt_only_suspension_is_not_a_full_one():
    """While WALKING the detector stays live - for one command only."""
    detector = GestureDetector(DT, INSTRUCTOR)
    detector.suspend(interrupt_only=True)
    assert detector.interrupt_only
    assert not detector.suspended, (
        "an interrupt-only suspension must NOT shut the detector down, or a "
        "STOP could never be given to a moving robot")


def test_resume_clears_both_kinds_of_suspension():
    detector = GestureDetector(DT, INSTRUCTOR)
    detector.suspend(interrupt_only=True)
    detector.resume()
    assert not detector.suspended and not detector.interrupt_only


def test_the_interrupt_command_is_the_stop():
    """Pinned, because the machine and the detector must agree on it."""
    assert INTERRUPT_COMMAND == "STOP"


def test_a_full_suspension_accumulates_nothing():
    """The candidate is dropped and no new one may start."""
    detector = GestureDetector(DT, INSTRUCTOR)
    detector.acquisition.state = "locked"
    detector.acquisition.locked = INSTRUCTOR
    detector.suspend()
    for index in range(int(CONFIRM_S / DT) * 2):
        detector.feed(index * DT, visibility=seen(INSTRUCTOR),
                      keypoints={}, yaws={}, ranges={INSTRUCTOR: 1.5})
    assert detector.confirmed(9.9) is None
    assert detector.candidate is None


# -- what the confirm gate requires -----------------------------------------
def test_confirmation_requires_the_full_window():
    detector = GestureDetector(DT, INSTRUCTOR)
    from gest_acquire import Candidate

    detector.candidate = Candidate(command="COME", template="COME",
                                   began_at_s=0.0)
    short = int(CONFIRM_S / DT) - 2
    detector.candidate.ticks = short
    detector.candidate.matching = short
    detector.candidate.readable_ticks = short
    assert detector.confirmed(1.0) is None, "confirmed before the window closed"

    full = int(CONFIRM_S / DT) + 1
    detector.candidate.ticks = full
    detector.candidate.matching = full
    detector.candidate.readable_ticks = full
    assert detector.confirmed(1.0) is not None


def test_confirmation_requires_every_counted_tick_to_be_readable():
    """An unreadable arm cannot be made up for by a long hold."""
    from gest_acquire import Candidate

    detector = GestureDetector(DT, INSTRUCTOR)
    full = int(CONFIRM_S / DT) + 1
    detector.candidate = Candidate(command="COME", template="COME",
                                   began_at_s=0.0)
    detector.candidate.ticks = full
    detector.candidate.matching = full
    detector.candidate.readable_ticks = full - 1
    assert detector.confirmed(1.0) is None


def test_confirmation_requires_the_matching_fraction():
    from gest_acquire import Candidate
    from gest_states import CONFIRM_MIN_FRACTION

    detector = GestureDetector(DT, INSTRUCTOR)
    full = int(CONFIRM_S / DT) + 40
    matching = int(full * (CONFIRM_MIN_FRACTION - 0.15))
    detector.candidate = Candidate(command="COME", template="COME",
                                   began_at_s=0.0)
    detector.candidate.ticks = full
    detector.candidate.matching = matching
    detector.candidate.readable_ticks = matching
    assert detector.confirmed(1.0) is None


def test_accepting_clears_the_window():
    """So one sustained gesture cannot be executed twice."""
    from gest_acquire import Candidate

    detector = GestureDetector(DT, INSTRUCTOR)
    full = int(CONFIRM_S / DT) + 1
    detector.candidate = Candidate(command="COME", template="COME",
                                   began_at_s=0.0)
    detector.candidate.ticks = full
    detector.candidate.matching = full
    detector.candidate.readable_ticks = full
    record = detector.confirmed(1.0)
    detector.accept(record)
    assert detector.candidate is None
    assert detector.accepted_commands == ["COME"]
