#!/usr/bin/env python3
"""Reading the queue: the path, arc-length ordering, membership and gaps.

No MuJoCo, no physics, no rendering.  Everything runs on hand-built inputs,
which is possible only because the queue reading was kept free of simulator
state.  The load-bearing test here is
``test_both_naive_orderings_name_the_wrong_tail``: if that ever passes
trivially, the scene has stopped posing the problem this behavior solves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import math

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from queue_geometry import (  # noqa: E402
    COUNTER_SIDE_XY,
    DUCK_PLANAR_RADIUS,
    JOIN_LATERAL_BAND_M,
    JOIN_LONGITUDINAL_MAX_M,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
    STANDOFF_TARGET_M,
    classify_gap,
    enumerate_gaps,
    gap_fits_duck,
    in_join_band,
    queue_geometry_summary,
)
from queue_model import (  # noqa: E402
    order_is_correct,
    overtaking_violations,
    read_queue,
    rejected_available_gaps,
    standoff_ok,
)
from queue_control import _bendiness  # noqa: E402
from queue_path import PATH, naive_orders  # noqa: E402
from queue_people import (  # noqa: E402
    ADULT_HALF_EXTENT_M,
    BYSTANDER_NAMES,
    DEPARTURE_TIMES,
    QUEUE,
    QUEUE_NAMES,
    departures,
    max_visible_jump,
    people_at,
)

TRUTH = list(QUEUE_NAMES)
STATIONS = {adult.name: adult.initial_arc for adult in QUEUE}


# ---------------------------------------------------------------- the path
def test_projection_round_trips_on_every_segment():
    """A point ON the path projects back to its own arc length, exactly."""
    for index in range(200):
        s = index * PATH.length / 199.0
        arc, cross, distance = PATH.project(PATH.point_at(s))
        assert arc == pytest.approx(s, abs=2e-3)
        assert cross == pytest.approx(0.0, abs=2e-3)
        assert distance == pytest.approx(0.0, abs=2e-3)


def test_arc_length_is_monotone_along_the_path():
    previous = -1.0
    for index in range(150):
        s = index * PATH.length / 149.0
        arc = PATH.arc_of(PATH.point_at(s))
        assert arc >= previous - 1e-6
        previous = arc


def test_positive_cross_track_is_inside_the_bend():
    """The sign convention the corner-cutting gate depends on.

    Measured, not assumed: displacing a point on the fold TOWARD the arc's
    centre must produce a POSITIVE cross-track.  The acceptance gate grades
    corner cutting on that sign, and an earlier version had it inverted and so
    graded swinging wide as cutting in.
    """
    fold = PATH.segments[1]
    point = PATH.point_at(1.65)
    inward = (fold.center - point) / np.linalg.norm(fold.center - point)
    assert PATH.project(point + inward * 0.10)[1] > 0.05
    assert PATH.project(point - inward * 0.10)[1] < -0.05


def test_travel_heading_is_opposite_the_away_heading():
    for s in (0.2, 1.0, 1.65, 2.4, 3.2, 4.2):
        delta = abs(PATH.travel_heading_at(s) - PATH.away_heading_at(s))
        assert math.degrees(min(delta, 2 * math.pi - delta)) == pytest.approx(
            180.0, abs=1e-6)


def test_the_path_does_not_intersect_itself():
    """Two legs of a hairpin must not overlap, or the queue is ambiguous."""
    polyline = PATH.polyline(0.02)
    worst = min(
        float(np.linalg.norm(polyline[i] - polyline[j]))
        for i in range(len(polyline))
        for j in range(i + 30, len(polyline)))
    assert worst > 2 * DUCK_PLANAR_RADIUS


def test_bendiness_is_zero_on_straights_and_one_on_the_fold():
    assert _bendiness(4.30) == pytest.approx(0.0, abs=1e-9)
    assert _bendiness(3.20) == pytest.approx(0.0, abs=1e-9)
    assert _bendiness(1.65) == pytest.approx(1.0)
    assert _bendiness(1.10) == pytest.approx(1.0)


# ------------------------------------------------------- ordering, the point
def test_arc_length_ordering_is_correct():
    positions = {name: PATH.point_at(arc) for name, arc in STATIONS.items()}
    reading = read_queue(positions)
    assert reading.order == TRUTH
    assert reading.tail == "eriksson"


def test_both_naive_orderings_name_the_wrong_tail():
    """THE CENTRAL CLAIM.  If this ever passes trivially, the scene is wrong.

    Not merely 'a heuristic could fail' - both heuristics DO fail on this exact
    geometry, and they fail differently, naming two different wrong people.
    """
    positions = {name: PATH.point_at(arc) for name, arc in STATIONS.items()}
    naive = naive_orders({n: tuple(p) for n, p in positions.items()})
    assert naive["by_range"][-1] == "dubois"
    assert naive["by_max_minus_x"][-1] == "chandra"
    assert naive["by_range"][-1] != "eriksson"
    assert naive["by_max_minus_x"][-1] != "eriksson"
    assert naive["by_range"] != TRUTH
    assert naive["by_max_minus_x"] != TRUTH


def test_the_duck_join_station_is_nearer_the_counter_than_someone_ahead_of_it():
    """Why a range-sorted reading would misrank the duck itself.

    MEASURED: the join station sits 1.298 m from the counter, while ``dubois``,
    who is two places AHEAD of the duck, sits at 1.307 m.  A reading that sorted
    by distance would therefore place the newly joined duck in front of somebody
    it is genuinely behind.  (``eriksson`` at 1.256 m is nearer still, which is
    the same effect one place further on.)
    """
    join = float(np.linalg.norm(PATH.point_at(
        STATIONS["eriksson"] + STANDOFF_TARGET_M)))
    assert join < float(np.linalg.norm(PATH.point_at(STATIONS["dubois"])))
    # And the true tail is nearer the counter than the person ahead of it,
    # which is what breaks the range heuristic in the first place.
    assert (float(np.linalg.norm(PATH.point_at(STATIONS["eriksson"])))
            < float(np.linalg.norm(PATH.point_at(STATIONS["dubois"]))))


def test_bystanders_are_excluded_by_measured_distance():
    positions = {name: tuple(PATH.point_at(arc))
                 for name, arc in STATIONS.items()}
    people = people_at(0.0)
    for name in BYSTANDER_NAMES:
        positions[name] = tuple(people[name].pos)
    reading = read_queue(positions)
    assert reading.order == TRUTH
    for name in BYSTANDER_NAMES:
        assert name in reading.excluded
        assert reading.excluded[name] > 0.30


def test_a_bystander_close_enough_to_the_path_would_be_included():
    """The exclusion is a measured band, not a hardcoded name list."""
    positions = {name: tuple(PATH.point_at(arc))
                 for name, arc in STATIONS.items()}
    positions["nakamura"] = tuple(PATH.point_at(3.60))
    reading = read_queue(positions)
    assert "nakamura" in reading.order
    assert reading.tail == "nakamura"


# ------------------------------------------------------------------- gaps
def _gaps():
    return enumerate_gaps(TRUTH, STATIONS, ADULT_HALF_EXTENT_M)


def test_exactly_one_gap_is_accepted_and_it_is_behind_the_tail():
    gaps = _gaps()
    accepted = [g for g in gaps if g.accepted]
    assert len(accepted) == 1
    assert accepted[0].name == "behind_tail"
    assert accepted[0].ahead == "eriksson"
    assert accepted[0].behind is None


def test_at_least_two_refused_gaps_were_physically_available():
    """Refusing a gap too narrow to stand in would prove nothing."""
    available = rejected_available_gaps(_gaps())
    assert len(available) >= 2
    names = {gap.name for gap in available}
    assert "beside_counter" in names
    assert "between_dubois_eriksson" in names


def test_the_stragglers_gap_genuinely_fits_the_duck():
    summary = queue_geometry_summary()
    assert summary["straggler_gap_fits_duck"] is True
    assert summary["straggler_gap_surface_slack_m"] > 0.05
    # And the nominal spacing does NOT, so the straggler's hole is special.
    assert summary["nominal_gap_fits_duck"] is False


def test_classify_gap_admits_only_the_place_with_nobody_behind_it():
    for gap in _gaps():
        assert classify_gap(gap) == gap.verdict
        assert (classify_gap(gap) == "join") == (gap.behind is None)


def test_gap_fits_duck_is_monotone_in_separation():
    previous = False
    for separation in (0.30, 0.45, 0.55, 0.70, 0.90, 1.20):
        fits = gap_fits_duck(separation, ADULT_HALF_EXTENT_M)
        assert not (previous and not fits)
        previous = fits


def test_the_counter_side_candidate_is_off_the_queue_path():
    """The tempting place is beside the queue, not a slot in it."""
    assert PATH.project(COUNTER_SIDE_XY)[2] > 0.30


# --------------------------------------------------------------- standoff
def test_standoff_band_accepts_the_target_and_rejects_the_extremes():
    assert standoff_ok(STANDOFF_TARGET_M)
    assert standoff_ok(STANDOFF_MIN_M)
    assert standoff_ok(STANDOFF_MAX_M)
    assert not standoff_ok(STANDOFF_MIN_M - 0.01)
    assert not standoff_ok(STANDOFF_MAX_M + 0.01)


def test_join_band_rejects_beside_and_in_front_of_the_tail():
    tail = STATIONS["eriksson"]
    assert in_join_band(tail + STANDOFF_TARGET_M, 0.0, tail)[0]
    # In front of the tail: negative longitudinal.
    assert not in_join_band(tail - 0.30, 0.0, tail)[0]
    # Beside it, in the lane but level with the tail.
    assert not in_join_band(tail + 0.02, 0.0, tail)[0]
    # Behind but out of the lane.
    assert not in_join_band(
        tail + STANDOFF_TARGET_M, JOIN_LATERAL_BAND_M + 0.05, tail)[0]
    # Far too far back to be queueing.
    assert not in_join_band(
        tail + JOIN_LONGITUDINAL_MAX_M + 0.2, 0.0, tail)[0]


def test_overtaking_is_detected_on_arc_length_not_distance():
    arcs = dict(STATIONS)
    behind_everyone = arcs["eriksson"] + STANDOFF_TARGET_M
    assert overtaking_violations(behind_everyone, arcs, TRUTH) == []
    # A duck that took the straggler's gap is AHEAD of eriksson in the queue,
    # even though it is further from the counter in a straight line.
    cut_in = 0.5 * (arcs["dubois"] + arcs["eriksson"])
    assert overtaking_violations(cut_in, arcs, TRUTH) == ["eriksson"]
