#!/usr/bin/env python3
"""Is this side usable?  The measurement, its causes, and the two pinned decisions.

``side_choice`` is the module the behavior turns on, and it is pure, so every
claim here is checked on hand-built inputs with no MuJoCo and no rollout.  The
last section pins the TWO real decisions the 86 s reference run makes — the
initial refusal of the hedged right side, and the kiosk blockage that causes the
switch — against the actual actor routes, so a scenery or route edit that
quietly removes a decision fails here rather than in a 30 s rollout.

The control case gets its own test: ``iris`` passes closer to the duck's lane
than anybody else and must NOT make it unusable.  A behavior that switched
whenever a pedestrian came near would pass every other test in this file.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from beside_actors import ROUTES, people_at  # noqa: E402
from beside_cast import ONCOMING_NAMES, PLANNING_HALF_EXTENT_M  # noqa: E402
from beside_geometry import (  # noqa: E402
    BESIDE_TARGET_M,
    SIDE_LOOKAHEAD_S,
    SIDE_PERSON_MARGIN_M,
    SIDE_STATIC_MARGIN_M,
    slot_point,
)
from promenade_layout import (  # noqa: E402
    HEDGE_S,
    KIOSK,
    OBSTACLES,
    static_gap,
)
from side_choice import (  # noqa: E402
    Track,
    bearing_of,
    closing_speed,
    evaluate_both,
    evaluate_side,
    prediction_error,
    prefer_side,
    tracks_from_states,
)

STILL = (0.0, 0.0)


def _verdicts(left_usable: bool, right_usable: bool, *,
              left_static: float = 1.0, right_static: float = 1.0):
    """Both sides graded, built directly rather than measured."""
    from side_choice import SideVerdict

    return {
        1: SideVerdict(1, left_usable, left_static, "wall", 3.0, "iris", 0.0,
                       "" if left_usable else "static",
                       "" if left_usable else "kiosk"),
        -1: SideVerdict(-1, right_usable, right_static, "wall", 3.0, "iris",
                        0.0, "" if right_usable else "static",
                        "" if right_usable else "hedge_s"),
    }


# -- the empty case -----------------------------------------------------------

def test_an_empty_lane_far_from_everything_is_usable():
    verdict = evaluate_side((0.0, 0.0), 0.0, STILL, 1, [])
    assert verdict.usable
    assert verdict.cause == ""
    assert verdict.detail == ""
    assert verdict.person_gap_m == float("inf")


def test_the_verdict_record_rounds_without_losing_the_cause():
    verdict = evaluate_side((0.0, 0.0), 0.0, STILL, 1, [])
    record = verdict.as_record()
    assert record["side"] == 1
    assert record["usable"] is True
    assert set(record) >= {"side", "usable", "static_gap_m", "static_name",
                           "person_gap_m", "person_name", "person_dt_s",
                           "cause", "detail"}


# -- static refusal -----------------------------------------------------------

def test_a_slot_pressed_against_a_wall_is_refused_like_one_against_a_kiosk():
    """The wall and the furniture are graded on the same scale, which is what
    stops a duck escaping a refusal by walking into the perimeter."""
    from promenade_layout import FLOOR_HALF

    # Facing +x at the north edge: her LEFT slot is hard against wall_n.
    guardian = (0.0, FLOOR_HALF[1] - BESIDE_TARGET_M + 0.05)
    verdict = evaluate_side(guardian, 0.0, STILL, 1, [])
    assert not verdict.usable
    assert verdict.cause == "static"
    assert verdict.static_name == "wall"


def test_the_static_refusal_fires_exactly_at_the_margin():
    """Usability flips at the margin, and nowhere else.

    Built by construction: walk her north in 2 mm steps so her left slot closes
    on the kiosk's south face, and require that ``usable`` is true for every
    sample whose measured gap is at or above the margin and false for every
    sample below it.  That is a stronger statement than "it refused somewhere".
    """
    samples = []
    for step in range(400):
        y = KIOSK.center[1] - KIOSK.half[1] - BESIDE_TARGET_M - 0.40 \
            + step * 0.002
        verdict = evaluate_side((KIOSK.center[0], y), 0.0, STILL, 1, [])
        samples.append((verdict.static_gap_m, verdict.usable))

    assert any(usable for _, usable in samples)
    assert any(not usable for _, usable in samples)
    for gap, usable in samples:
        assert usable == (gap >= SIDE_STATIC_MARGIN_M), (
            f"gap {gap:.4f} m against a {SIDE_STATIC_MARGIN_M} m margin "
            f"reported usable={usable}")


def test_a_zero_length_lookahead_still_grades_the_present_slot():
    """The sweep must not be the only thing that can refuse a side."""
    guardian = (HEDGE_S.center[0], HEDGE_S.center[1] + HEDGE_S.half[1] + 0.10)
    verdict = evaluate_side(guardian, 0.0, STILL, -1, [], lookahead_s=0.0,
                            samples=1)
    assert not verdict.usable
    assert verdict.cause == "static"


# -- the swept lane -----------------------------------------------------------

def test_the_lane_is_swept_along_her_motion_not_only_sampled_where_she_stands():
    """The kiosk refusal must happen BEFORE the duck is level with it.

    Place her far enough south of the kiosk that her present left slot is clear,
    but walking north at a speed that carries the slot into it inside the
    lookahead.  A chooser that graded only the present slot would call it
    usable.
    """
    start = (KIOSK.center[0],
             KIOSK.center[1] - KIOSK.half[1] - BESIDE_TARGET_M - 1.10)
    still = evaluate_side(start, 0.0, STILL, 1, [])
    assert still.usable, "the present slot must genuinely be clear"

    velocity = (0.0, 1.10 / SIDE_LOOKAHEAD_S)
    moving = evaluate_side(start, 0.0, velocity, 1, [])
    assert not moving.usable
    assert moving.cause == "static"
    assert moving.detail == "kiosk"


def test_the_sweep_reports_the_worst_sample_not_the_last_one():
    verdict = evaluate_side(
        (KIOSK.center[0],
         KIOSK.center[1] - KIOSK.half[1] - BESIDE_TARGET_M - 1.10),
        0.0, (0.0, 1.10 / SIDE_LOOKAHEAD_S), 1, [])
    gaps = [sample["static_gap_m"] for sample in verdict.samples]
    assert verdict.static_gap_m == pytest.approx(min(gaps), abs=1e-4)
    assert gaps[0] > gaps[-1], "she is walking INTO it, so the gap must shrink"


# -- predicted people ---------------------------------------------------------

def test_a_pedestrian_predicted_into_the_lane_blocks_it_before_arriving():
    """The track's PRESENT position is clear; its predicted one is not."""
    guardian = (0.0, 0.0)
    slot = slot_point(guardian, 0.0, 1)
    approach = 2.4
    track = Track("tomas", np.array([slot[0], slot[1] + approach]),
                  np.array([0.0, -approach / SIDE_LOOKAHEAD_S]))
    assert float(np.linalg.norm(track.pos - slot)) > SIDE_PERSON_MARGIN_M

    verdict = evaluate_side(guardian, 0.0, STILL, 1, [track])
    assert not verdict.usable
    assert verdict.cause == "person"
    assert verdict.detail == "tomas"
    assert 0.0 < verdict.person_dt_s <= SIDE_LOOKAHEAD_S


def test_a_pedestrian_who_stays_outside_the_margin_does_not_block_the_lane():
    """The control case, in miniature: near is not the same as in the way."""
    guardian = (0.0, 0.0)
    slot = slot_point(guardian, 0.0, 1)
    offset = SIDE_PERSON_MARGIN_M + 0.13
    track = Track("iris", np.array([slot[0] - 3.0, slot[1] + offset]),
                  np.array([3.0 / SIDE_LOOKAHEAD_S, 0.0]))
    verdict = evaluate_side(guardian, 0.0, STILL, 1, [track])
    assert verdict.usable
    assert verdict.cause == ""
    assert verdict.person_gap_m == pytest.approx(offset, abs=1e-6)


def test_the_person_margin_is_the_exact_threshold():
    guardian = (0.0, 0.0)
    slot = slot_point(guardian, 0.0, 1)
    for delta, expected in ((+1e-4, True), (-1e-3, False)):
        track = Track("tomas",
                      np.array([slot[0], slot[1] + SIDE_PERSON_MARGIN_M + delta]),
                      np.zeros(2))
        assert evaluate_side(guardian, 0.0, STILL, 1, [track]).usable is expected


def test_the_cause_names_the_hazard_that_actually_decided():
    """When both hazards bind, the tighter one relative to ITS OWN margin wins."""
    guardian = (0.0, 0.0)
    slot = slot_point(guardian, 0.0, 1)
    # A person right on top of the slot, and a wall only marginally too close.
    person = Track("tomas", np.array(slot), np.zeros(2))
    verdict = evaluate_side(
        guardian, 0.0, STILL, 1, [person],
        static_margin=1e6)  # force the static test to bind trivially
    assert verdict.cause == "person"
    assert verdict.detail == "tomas"


def test_tracks_from_states_excludes_the_guardian_and_reads_no_route():
    people = people_at(12.0)
    tracks = tracks_from_states(people, "nadia")
    assert "nadia" not in {track.name for track in tracks}
    assert len(tracks) == len(people) - 1
    for track in tracks:
        assert not hasattr(track, "route")
        assert track.pos.shape == (2,) and track.velocity.shape == (2,)


def test_a_track_extrapolates_linearly_and_owns_its_own_arrays():
    pos = np.array([1.0, 2.0])
    track = Track("x", pos.copy(), np.array([0.5, -0.25]))
    assert track.at(0.0) == pytest.approx(pos)
    assert track.at(4.0) == pytest.approx(np.array([3.0, 1.0]))
    pos[:] = 99.0
    assert track.at(0.0)[0] == 1.0, "the track must not alias its source array"


def test_prediction_error_is_zero_for_a_track_that_walks_its_own_line():
    track = Track("x", np.array([0.0, 0.0]), np.array([0.3, 0.0]))
    assert prediction_error(track, (0.9, 0.0), 3.0) == pytest.approx(0.0)
    assert prediction_error(track, (0.9, 0.4), 3.0) == pytest.approx(0.4)


def test_closing_speed_and_bearing_are_measured_in_her_frame():
    track = Track("x", np.array([2.0, 0.0]), np.array([-0.3, 0.0]))
    assert closing_speed(track, (0.12, 0.0)) == pytest.approx(0.42)
    assert bearing_of(track, (0.0, 0.0), 0.0) == pytest.approx(0.0)
    assert bearing_of(track, (0.0, 0.0), math.pi / 2.0) == pytest.approx(-90.0)


# -- preference ---------------------------------------------------------------

def test_both_usable_keeps_the_current_side_rather_than_switching_for_nothing():
    verdicts = _verdicts(True, True, left_static=0.3, right_static=9.0)
    side, reason = prefer_side(verdicts, current_side=1)
    assert side == 1 and "remains usable" in reason
    side, reason = prefer_side(verdicts, current_side=-1)
    assert side == -1 and "remains usable" in reason


def test_with_no_current_side_the_larger_static_clearance_wins_ties_to_the_left():
    assert prefer_side(_verdicts(True, True, left_static=2.0,
                                 right_static=1.0), None)[0] == 1
    assert prefer_side(_verdicts(True, True, left_static=1.0,
                                 right_static=2.0), None)[0] == -1
    # Exactly equal is deterministic, and it goes left.
    assert prefer_side(_verdicts(True, True, left_static=1.0,
                                 right_static=1.0), None)[0] == 1


def test_one_usable_side_is_taken_and_the_reason_names_the_other_hazard():
    side, reason = prefer_side(_verdicts(True, False), None)
    assert side == 1
    assert "right blocked by static:hedge_s" == reason
    side, reason = prefer_side(_verdicts(False, True), None)
    assert side == -1
    assert "left blocked by static:kiosk" == reason


def test_neither_usable_returns_none_rather_than_a_licence_to_stop():
    side, reason = prefer_side(_verdicts(False, False), 1)
    assert side is None
    assert "both blocked" in reason


def test_a_blocked_current_side_is_abandoned_even_when_it_was_preferred():
    side, _ = prefer_side(_verdicts(False, True), current_side=1)
    assert side == -1


def test_evaluate_both_grades_each_side_independently():
    guardian = (HEDGE_S.center[0], HEDGE_S.center[1] + HEDGE_S.half[1] + 0.10)
    verdicts = evaluate_both(guardian, 0.0, STILL, [])
    assert set(verdicts) == {1, -1}
    assert verdicts[1].side == 1 and verdicts[-1].side == -1
    assert verdicts[1].usable != verdicts[-1].usable


# -- the two pinned decisions of the reference run ----------------------------

def _measure(t: float):
    people = people_at(t)
    guardian = people["nadia"]
    tracks = tracks_from_states(people, "nadia")
    return evaluate_both(guardian.pos, guardian.yaw, guardian.velocity, tracks)


def test_the_initial_join_refuses_the_hedged_right_side_at_t_zero():
    """PINNED: the very first decision of the reference run.

    The guardian's right-hand slot on the south straight lies inside
    ``hedge_s``, so the duck cannot default to a side — it measures both and
    refuses one.  A scenery edit that moves the hedge off the lane deletes the
    first side decision entirely, and the run's ``side_decision_count`` gate
    would then be carried by the switch alone.
    """
    verdicts = _measure(0.0)
    assert verdicts[1].usable is True
    assert verdicts[-1].usable is False
    assert verdicts[-1].cause == "static"
    assert verdicts[-1].detail == "hedge_s"
    assert verdicts[-1].static_gap_m < 0.0, (
        "the slot is INSIDE the hedge, not merely close to it")
    side, reason = prefer_side(verdicts, None)
    assert side == 1
    assert reason == "right blocked by static:hedge_s"


def test_the_first_switch_is_caused_by_the_kiosk_taking_the_left_lane():
    """PINNED: the blockage the reference run switches on, at t = 8.86 s."""
    verdicts = _measure(8.86)
    assert verdicts[1].usable is False
    assert verdicts[1].cause == "static"
    assert verdicts[1].detail == "kiosk"
    assert verdicts[-1].usable is True
    side, reason = prefer_side(verdicts, current_side=1)
    assert side == -1
    assert reason == "left blocked by static:kiosk"


def test_the_left_lane_is_continuously_blocked_across_the_confirm_window():
    """A single tick of a swinging arm is not a blockage.

    The machine requires ``BLOCK_CONFIRM_S`` of CONTINUOUS refusal before it
    will commit, so the scenario has to supply one.
    """
    from beside_constants import BLOCK_CONFIRM_S

    start = 8.86 - BLOCK_CONFIRM_S
    samples = [_measure(start + i * 0.02)[1].usable
               for i in range(int(BLOCK_CONFIRM_S / 0.02) + 1)]
    assert not any(samples), (
        "the left lane must be unusable for the whole confirm window")


def test_iris_is_the_control_case_and_never_makes_the_duck_lane_unusable():
    """She comes closer than anybody else and must not cause a switch.

    Graded over the whole run against the side the duck is actually on at each
    instant in the reference rollout: left until the switch, right afterwards.
    """
    switch_t = 8.86
    for step in range(0, 4300, 5):
        t = step * 0.02
        people = people_at(t)
        guardian = people["nadia"]
        side = 1 if t < switch_t else -1
        iris_only = [track for track in tracks_from_states(people, "nadia")
                     if track.name == "iris"]
        verdict = evaluate_side(guardian.pos, guardian.yaw, guardian.velocity,
                                side, iris_only)
        assert verdict.cause != "person" or verdict.detail != "iris", (
            f"iris blocked the duck's own lane at t={t:.2f}s "
            f"(gap {verdict.person_gap_m:.3f} m)")


def test_every_oncoming_walker_is_actually_oncoming_at_some_point():
    """A 'control case' who never comes near proves nothing."""
    closest = {name: float("inf") for name in ONCOMING_NAMES}
    for step in range(0, 4300, 5):
        t = step * 0.02
        people = people_at(t)
        guardian = people["nadia"]
        for name in ONCOMING_NAMES:
            gap = float(np.linalg.norm(people[name].pos - guardian.pos))
            closest[name] = min(closest[name], gap)
    for name, gap in closest.items():
        assert gap < 2.0, f"{name} never came within 2 m of the guardian"


def test_the_planning_half_extent_is_generous_but_not_a_clearance_gate():
    """It inflates a PREDICTED person; clearance is measured on real geoms."""
    assert PLANNING_HALF_EXTENT_M > 0.0
    assert SIDE_PERSON_MARGIN_M > PLANNING_HALF_EXTENT_M


def test_static_gap_agrees_with_the_obstacle_set_it_is_derived_from():
    """The chooser and the scene must never disagree about what is where."""
    for obstacle in OBSTACLES:
        name, gap = static_gap(obstacle.center)
        assert gap <= 0.0, f"the centre of {obstacle.name} is inside it"
        assert name == obstacle.name or name == "wall"


def test_the_guardian_route_actually_walks_past_both_deciding_bodies():
    """A hedge and a kiosk that the route never approaches decide nothing."""
    route = ROUTES["nadia"]
    nearest = {"hedge_s": float("inf"), "kiosk": float("inf")}
    for step in range(0, 4301, 10):
        t = step * 0.02
        position = route.pos_at(t)
        nearest["hedge_s"] = min(nearest["hedge_s"],
                                 HEDGE_S.distance_to(position))
        nearest["kiosk"] = min(nearest["kiosk"], KIOSK.distance_to(position))
    for name, gap in nearest.items():
        assert gap < BESIDE_TARGET_M + SIDE_STATIC_MARGIN_M, (
            f"the guardian never brings a slot near {name} (min {gap:.3f} m)")
