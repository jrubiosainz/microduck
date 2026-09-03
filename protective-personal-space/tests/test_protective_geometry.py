#!/usr/bin/env python3
"""Protective geometry: the slot, the station, between-ness and the escape gap.

Pure planar geometry with no simulator, which is what lets the harder claims -
"the route around the protected person never crosses her", "the escape gap
rejects a direction on static clearance" - be checked exhaustively instead of
sampled from one rollout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pps_cast import PLANNING_HALF_EXTENT_M
from pps_geometry import (angle_delta_deg, axes, bearing_deg, escape_point,
                          escort_point, interpose_point, is_between,
                          projected_along, route_around_ward, static_clearance,
                          surface_gap)
from pps_plaza import BY_FIXTURE, FIXTURES, FLOOR_HALF, inside_area
from pps_states import (DUCK_PLANAR_RADIUS, ESCAPE_MIN_CLEARANCE_M,
                        ESCAPE_RADIUS_M, ESCORT_BEHIND_M, ESCORT_LATERAL_M,
                        INTERPOSE_BEARING_TOL_DEG, INTERPOSE_FROM_PERSON_M)

SLOT_RANGE_M = math.hypot(ESCORT_BEHIND_M, ESCORT_LATERAL_M)
YAWS = [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2, -2.4]


# -- frames ------------------------------------------------------------------
@pytest.mark.parametrize("yaw", YAWS)
def test_axes_are_orthonormal_and_right_handed(yaw):
    forward, left = axes(yaw)
    assert float(np.linalg.norm(forward)) == pytest.approx(1.0)
    assert float(np.linalg.norm(left)) == pytest.approx(1.0)
    assert float(forward @ left) == pytest.approx(0.0, abs=1e-12)
    cross = float(forward[0] * left[1] - forward[1] * left[0])
    assert cross == pytest.approx(1.0), "left must be +90 deg from forward"


# -- the escort slot ---------------------------------------------------------
@pytest.mark.parametrize("yaw", YAWS)
def test_the_slot_sits_behind_and_to_the_wards_right(yaw):
    """A REQUIRED formation: beside and slightly behind, on her right.

    Checked in HER frame rather than the world's, so the claim survives every
    heading she walks.
    """
    slot = escort_point((0.0, 0.0), yaw)
    forward, left = axes(yaw)
    assert float(slot @ forward) == pytest.approx(-ESCORT_BEHIND_M)
    assert float(slot @ left) == pytest.approx(-ESCORT_LATERAL_M)


@pytest.mark.parametrize("yaw", YAWS)
def test_the_slot_keeps_a_constant_distance_from_the_ward(yaw):
    slot = escort_point((0.0, 0.0), yaw)
    assert float(np.linalg.norm(slot)) == pytest.approx(SLOT_RANGE_M)
    assert SLOT_RANGE_M == pytest.approx(0.7632, abs=1e-4)


@pytest.mark.parametrize("ward", [(0.0, 0.0), (1.4, -0.8), (-2.0, 2.2)])
def test_the_slot_translates_with_the_ward(ward):
    at_origin = escort_point((0.0, 0.0), 0.7)
    moved = escort_point(ward, 0.7)
    assert moved == pytest.approx(np.asarray(ward) + at_origin)


def test_the_slot_is_further_out_than_the_join_tolerance():
    """Otherwise standing anywhere near her would count as joined."""
    from pps_states import ESCORT_JOIN_M
    assert SLOT_RANGE_M > 3.0 * ESCORT_JOIN_M


# -- the interpose station ---------------------------------------------------
@pytest.mark.parametrize("bearing", [0.0, 40.0, 90.0, 175.0, -60.0, -140.0])
def test_the_station_sits_on_the_person_to_threat_bearing(bearing):
    angle = math.radians(bearing)
    threat = (3.0 * math.cos(angle), 3.0 * math.sin(angle))
    station = interpose_point((0.0, 0.0), threat)
    assert float(np.linalg.norm(station)) == pytest.approx(
        INTERPOSE_FROM_PERSON_M)
    assert bearing_deg((0.0, 0.0), station) == pytest.approx(bearing, abs=1e-6)


@pytest.mark.parametrize("distance", [1.0, 2.0, 3.4])
def test_the_station_distance_does_not_depend_on_how_far_the_threat_is(
        distance):
    """It is a station relative to the PERSON, not a midpoint."""
    station = interpose_point((0.0, 0.0), (distance, 0.0))
    assert station == pytest.approx(np.array([INTERPOSE_FROM_PERSON_M, 0.0]))


def test_the_station_is_inside_the_buffer_but_does_not_crowd_her():
    from pps_states import BUFFER_M, INTERPOSE_MIN_CLEARANCE_M
    assert INTERPOSE_FROM_PERSON_M < BUFFER_M
    assert INTERPOSE_FROM_PERSON_M > INTERPOSE_MIN_CLEARANCE_M + \
        DUCK_PLANAR_RADIUS


def test_a_degenerate_coincident_threat_gives_a_usable_station():
    """A threat exactly on the ward has no bearing; guessing beats dividing."""
    station = interpose_point((0.0, 0.0), (0.0, 0.0))
    assert np.all(np.isfinite(station))
    assert float(np.linalg.norm(station)) == pytest.approx(
        INTERPOSE_FROM_PERSON_M)


def test_a_station_that_would_leave_the_plaza_is_clamped_inside_it():
    """Clamping keeps the target reachable; the gates then judge the clamp.

    A clamp can never smuggle in a usable position, because between-ness and
    clearance are graded separately on the position actually reached.
    """
    station = interpose_point((3.5, 2.9), (9.0, 9.0))
    assert inside_area(station, DUCK_PLANAR_RADIUS + 0.08 - 1e-9)
    assert abs(station[0]) <= FLOOR_HALF[0]
    assert abs(station[1]) <= FLOOR_HALF[1]


# -- between-ness ------------------------------------------------------------
def test_a_duck_on_the_line_is_between():
    assert is_between((0.0, 0.0), (0.85, 0.0), (3.0, 0.0))


def test_a_duck_behind_the_threat_is_not_between():
    """Standing further out than the intruder is not interposing."""
    assert not is_between((0.0, 0.0), (4.0, 0.0), (3.0, 0.0))


def test_a_duck_on_top_of_the_ward_is_not_between():
    """Under 0.25 m the bearing is noise, and it would be crowding her anyway."""
    assert not is_between((0.0, 0.0), (0.2, 0.0), (3.0, 0.0))
    assert not is_between((0.0, 0.0), (0.0, 0.0), (3.0, 0.0))


@pytest.mark.parametrize("offset_deg,expected", [
    (0.0, True), (20.0, True), (33.9, True), (34.1, False), (60.0, False),
    (-33.9, True), (-34.1, False), (90.0, False), (180.0, False)])
def test_between_ness_holds_exactly_to_the_declared_angular_tolerance(
        offset_deg, expected):
    """34 deg subtends more arc than the duck is wide, so the edge is physical."""
    angle = math.radians(offset_deg)
    duck = (0.85 * math.cos(angle), 0.85 * math.sin(angle))
    assert is_between((0.0, 0.0), duck, (3.0, 0.0)) is expected
    assert INTERPOSE_BEARING_TOL_DEG == 34.0


@pytest.mark.parametrize("bearing", [0.0, 75.0, 150.0, -110.0])
def test_between_ness_is_rotation_invariant(bearing):
    angle = math.radians(bearing)
    rotate = np.array([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])
    ward = np.array([1.3, -0.4])
    duck = ward + rotate @ np.array([0.85, 0.0])
    threat = ward + rotate @ np.array([3.0, 0.0])
    assert is_between(ward, duck, threat)


def test_the_tolerance_is_wider_than_the_duck_is_at_the_station_radius():
    """The derivation the constant claims, checked rather than repeated."""
    arc = 2.0 * INTERPOSE_FROM_PERSON_M * math.sin(
        math.radians(INTERPOSE_BEARING_TOL_DEG) / 2.0)
    assert arc > 2.0 * DUCK_PLANAR_RADIUS


# -- bearings ----------------------------------------------------------------
@pytest.mark.parametrize("target,expected", [((1.0, 0.0), 0.0),
                                             ((0.0, 1.0), 90.0),
                                             ((-1.0, 0.0), 180.0),
                                             ((0.0, -1.0), -90.0),
                                             ((1.0, 1.0), 45.0)])
def test_bearing_matches_the_world_convention(target, expected):
    assert bearing_deg((0.0, 0.0), target) == pytest.approx(expected)


@pytest.mark.parametrize("a,b,expected", [(170.0, -170.0, -20.0),
                                          (-170.0, 170.0, 20.0),
                                          (10.0, 350.0, 20.0),
                                          (0.0, 0.0, 0.0),
                                          (90.0, -90.0, -180.0)])
def test_angle_delta_wraps_to_the_short_way_round(a, b, expected):
    """Half-open ``[-180, 180)``, so an exact reversal reports ``-180``."""
    assert angle_delta_deg(a, b) == pytest.approx(expected)
    assert -180.0 <= angle_delta_deg(a, b) < 180.0


# -- clearances --------------------------------------------------------------
@pytest.mark.parametrize("distance", [0.5, 1.0, 2.5])
def test_surface_gap_subtracts_both_half_extents(distance):
    gap = surface_gap((0.0, 0.0), (distance, 0.0))
    assert gap == pytest.approx(
        distance - DUCK_PLANAR_RADIUS - PLANNING_HALF_EXTENT_M)


def test_surface_gap_is_negative_when_the_planning_discs_overlap():
    assert surface_gap((0.0, 0.0), (0.3, 0.0)) < 0.0


def test_static_clearance_is_negative_inside_a_fixture():
    """The planter is exactly the fixture an escape gap must refuse."""
    planter = BY_FIXTURE["obs_planter_s"]
    assert static_clearance(planter.center) < 0.0
    assert static_clearance((0.0, 0.0)) > 1.0


def test_static_clearance_reports_the_nearest_fixture():
    for fixture in FIXTURES:
        just_outside = (fixture.center[0] + fixture.half[0] + 0.05,
                        fixture.center[1])
        assert static_clearance(just_outside) <= 0.05 - DUCK_PLANAR_RADIUS \
            + 1e-9


@pytest.mark.parametrize("heading", [0.0, math.pi / 2, -1.1])
def test_projected_along_measures_displacement_on_a_heading(heading):
    forward, left = axes(heading)
    start = np.array([0.7, -1.2])
    assert projected_along(start, start + forward * 0.4, heading) == \
        pytest.approx(0.4)
    assert projected_along(start, start - forward * 0.4, heading) == \
        pytest.approx(-0.4)
    assert projected_along(start, start + left * 0.9, heading) == \
        pytest.approx(0.0, abs=1e-12)


def test_the_retreat_is_graded_on_projection_not_path():
    """The reverse gait drifts in yaw, so path length would flatter it.

    A duck that reversed in an arc covers path without going backwards; the
    projection onto the pre-action heading is what makes the retreat a yield.
    """
    heading = 0.0
    start = np.array([0.0, 0.0])
    arced = np.array([-0.10, 0.60])
    path = float(np.linalg.norm(arced - start))
    assert path > 0.34 > -projected_along(start, arced, heading)


# -- the escape gap ----------------------------------------------------------
def test_the_gap_search_returns_one_candidate_per_sampled_direction():
    point, score, candidates = escape_point(
        (0.0, 0.0), [np.array([2.0, 0.0])], {})
    assert len(candidates) == 24
    assert [c["index"] for c in candidates] == list(range(24))
    assert score == max(c["score"] for c in candidates)
    assert any(np.allclose(c["point"], point) for c in candidates)


@pytest.mark.parametrize("samples", [8, 16, 36])
def test_the_sample_count_is_honoured(samples):
    _, _, candidates = escape_point((0.0, 0.0), [np.array([2.0, 0.0])], {},
                                    samples=samples)
    assert len(candidates) == samples


def test_the_gap_is_chosen_at_the_declared_radius_from_the_ward():
    point, _, _ = escape_point((0.0, 0.0), [np.array([2.0, 0.0])], {})
    assert float(np.linalg.norm(point)) == pytest.approx(ESCAPE_RADIUS_M)


def test_the_gap_runs_away_from_both_people_in_a_pinch():
    """Two threats on the x axis, so the safe directions are lateral."""
    point, score, _ = escape_point(
        (0.0, 0.0), [np.array([2.0, 0.0]), np.array([-2.0, 0.0])],
        {"a": np.array([2.0, 0.0]), "b": np.array([-2.0, 0.0])})
    assert abs(float(point[1])) > abs(float(point[0]))
    assert score >= ESCAPE_MIN_CLEARANCE_M


def test_the_score_is_the_worst_of_its_four_measured_components():
    _, _, candidates = escape_point(
        (0.0, 0.0), [np.array([1.5, 0.0])],
        {"a": np.array([1.5, 0.0])}, start=(-2.0, 0.0))
    for candidate in candidates:
        assert candidate["score"] == pytest.approx(
            min(candidate["static"], candidate["people"],
                candidate["threats"], candidate["route_gap"]))


def test_without_a_start_the_route_leg_is_not_scored():
    """The route term only exists when the caller says where the duck is."""
    _, _, candidates = escape_point((0.0, 0.0), [np.array([2.0, 0.0])], {})
    assert all(c["route_gap"] == 9.0 for c in candidates)


def test_with_a_start_the_route_leg_is_scored_and_can_dominate():
    """A gap you cannot walk to without brushing somebody is not a gap.

    The leg from the duck to the candidate is sampled, so a direction whose
    APPROACH passes through a person scores on that rather than on its endpoint.
    """
    _, _, candidates = escape_point(
        (0.0, 0.0), [np.array([2.0, 0.0])], {"a": np.array([2.0, 0.0])},
        start=(-2.5, 0.0))
    assert all(c["route_gap"] < 9.0 for c in candidates)
    assert any(c["route_gap"] == c["score"] for c in candidates)


def test_the_planter_direction_is_rejected_on_measured_static_clearance():
    """The fixture exists precisely so the search must refuse a direction.

    Standing the ward next to the planter makes one arc of candidate gaps sit
    on it, and the chosen gap has to be measurably clear of it.
    """
    ward = (-1.06 + 0.95, -1.42)
    point, score, candidates = escape_point(ward, [np.array([3.0, 3.0])], {})
    worst = min(candidates, key=lambda c: c["static"])
    assert worst["static"] < static_clearance(point)
    assert static_clearance(point) > 0.0


def test_every_candidate_gap_stays_inside_the_plaza():
    for ward in [(0.0, 0.0), (3.2, 2.6), (-3.3, -2.7)]:
        _, _, candidates = escape_point(ward, [np.array([0.0, 0.0])], {})
        for candidate in candidates:
            assert inside_area(candidate["point"],
                               DUCK_PLANAR_RADIUS + 0.10 - 1e-9)


def test_no_people_and_no_threats_still_returns_a_finite_gap():
    point, score, _ = escape_point((0.0, 0.0), [], {})
    assert np.all(np.isfinite(point))
    assert score < 9.0, "static clearance always bounds the score"


# -- routing around the protected person -------------------------------------
def test_a_clear_segment_is_left_alone():
    """No detour when the direct line already misses her."""
    route = route_around_ward((0.0, -2.0), (0.0, 0.0), (0.5, -1.5))
    assert len(route) == 1
    assert route[0] == pytest.approx(np.array([0.5, -1.5]))


def test_a_segment_through_the_ward_is_replaced_by_an_arc_around_her():
    """Interposing on the opposite bearing must go AROUND her, never through.

    A single lateral waypoint is not enough: the CHORD between two safe points
    can still cut the protected disc, which is why the detour is an arc.
    """
    route = route_around_ward((-2.0, 0.0), (0.0, 0.0), (2.0, 0.0))
    assert len(route) > 1
    assert route[-1] == pytest.approx(np.array([2.0, 0.0]))
    for waypoint in route[:-1]:
        assert float(np.linalg.norm(waypoint)) >= 0.94


def test_every_leg_of_the_detour_clears_the_protected_disc():
    """Sampled along each leg, because a waypoint test alone misses the chord."""
    start = np.array([-2.0, 0.0])
    ward = np.zeros(2)
    route = route_around_ward(start, ward, np.array([2.0, 0.0]))
    cursor = start
    for waypoint in route[:-1]:
        for u in np.linspace(0.0, 1.0, 33):
            sample = cursor + (waypoint - cursor) * u
            assert float(np.linalg.norm(sample - ward)) > 0.55
        cursor = waypoint


@pytest.mark.parametrize("heading,side", [(math.pi / 2, 1.0),
                                          (-math.pi / 2, -1.0)])
def test_the_detour_leaves_on_the_side_the_duck_is_already_facing(heading,
                                                                  side):
    """The duck cannot turn in place, so a detour to the wrong hand is a cost."""
    route = route_around_ward((-2.0, 0.0), (0.0, 0.0), (2.0, 0.0),
                              heading=heading)
    assert float(route[0][1]) * side > 0.0


def test_the_detour_ends_where_it_was_asked_to_end():
    for heading in (None, 0.0, math.pi / 2, -2.0):
        route = route_around_ward((-2.0, 0.3), (0.0, 0.0), (1.8, -0.2),
                                  heading=heading)
        assert route[-1] == pytest.approx(np.array([1.8, -0.2]))


def test_a_degenerate_zero_length_segment_does_not_divide_by_zero():
    route = route_around_ward((1.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    assert len(route) >= 1
    assert np.all(np.isfinite(route[-1]))


def test_a_wider_clearance_demands_a_wider_detour():
    tight = route_around_ward((-2.0, 0.0), (0.0, 0.0), (2.0, 0.0),
                              clearance=0.60)
    wide = route_around_ward((-2.0, 0.0), (0.0, 0.0), (2.0, 0.0),
                             clearance=1.30)
    assert min(float(np.linalg.norm(p)) for p in wide[:-1]) > \
        min(float(np.linalg.norm(p)) for p in tight[:-1]) - 1e-9
    assert min(float(np.linalg.norm(p)) for p in wide[:-1]) >= 1.29


def test_every_waypoint_of_a_detour_stays_inside_the_plaza():
    for goal in [(3.4, 2.8), (-3.4, -2.8), (3.4, -2.8)]:
        route = route_around_ward((-1.0, 0.0), (0.0, 0.0), goal)
        for waypoint in route[:-1]:
            assert inside_area(waypoint, DUCK_PLANAR_RADIUS + 0.10 - 1e-9)
