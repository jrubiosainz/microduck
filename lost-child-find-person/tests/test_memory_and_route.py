#!/usr/bin/env python3
"""The world-space trail and the rejoin route planner.

Pure numpy: no MuJoCo, no policy, no rendering.  What is graded here is the
memory the duck keeps across a loss, and the route it plans back to its
guardian around real inflated geometry.

THE REGRESSION THIS MODULE EXISTS TO KEEP FIXED
------------------------------------------------
``route_progress`` is a STATELESS waypoint selector: it returns the first
waypoint farther than a tolerance.  Once the duck walks past a corner, that
corner's distance grows again, so the stateless rule re-selects it and the duck
turns round and chases a point it already visited — which is how an earlier
rejoin looped until it hit the 30 s ceiling.  The rollout therefore carries a
MONOTONIC cursor instead.  ``test_the_stateless_selector_re_chases_a_passed_corner``
pins the defect and ``test_the_monotonic_cursor_never_goes_backwards`` pins the
fix, so neither can be quietly reverted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lost_geometry import PLAN_INFLATE_M  # noqa: E402
from lost_memory import (  # noqa: E402
    TRAIL_MAX,
    TRAIL_STEP_M,
    GuardianTrail,
    line_of_sight_available,
    plan_route,
    route_progress,
    segment_blocked,
    waypoint_candidates,
)
from plaza_layout import OBSTACLES, clear_of_obstacles  # noqa: E402

# A start/goal pair whose straight line runs through the kiosk: the scenario's
# own geometry, and the case the rejoin actually has to solve.
KIOSK_START = (1.90, -0.60)
KIOSK_GOAL = (-0.60, 0.90)


# ---------------------------------------------------------------- the trail
def test_the_trail_records_at_a_fixed_spatial_interval_not_every_tick():
    """A time-sampled trail bunches wherever she slowed down and says nothing."""
    trail = GuardianTrail()
    for step in range(50):
        trail.observe(step * 0.02, np.array([step * 0.01, 0.0]))
    spacings = [float(np.linalg.norm(b - a))
                for a, b in zip(trail.points, trail.points[1:])]
    assert spacings
    assert all(space >= TRAIL_STEP_M - 1e-6 for space in spacings)


def test_the_trail_is_bounded_and_keeps_the_most_recent_footprints():
    trail = GuardianTrail()
    for step in range(400):
        trail.observe(step * 0.1, np.array([step * 0.05, 0.0]))
    assert len(trail.points) == TRAIL_MAX
    assert float(trail.points[-1][0]) == pytest.approx(
        float(trail.last_seen_xy[0]), abs=TRAIL_STEP_M)


def test_the_trail_survives_the_loss_and_is_not_derived_from_her_live_pose():
    """The whole point: after she disappears the duck still has where she was."""
    trail = GuardianTrail()
    for step in range(30):
        trail.observe(step * 0.2, np.array([2.0 - 0.1 * step, 0.5]))
    last = trail.last_seen_xy.copy()
    last_t = trail.last_seen_t
    # She keeps walking, unobserved.  Nothing is recorded.
    assert trail.age(last_t + 6.0) == pytest.approx(6.0)
    assert np.array_equal(trail.last_seen_xy, last)


def test_an_empty_trail_reports_infinite_age_and_no_extrapolation():
    trail = GuardianTrail()
    assert math.isinf(trail.age(10.0))
    assert trail.extrapolated(10.0) is None
    assert trail.length_m() == 0.0
    assert trail.as_record()["last_seen_xy"] is None


def test_the_heading_is_taken_from_consecutive_sightings():
    trail = GuardianTrail()
    trail.observe(0.0, np.array([0.0, 0.0]))
    trail.observe(1.0, np.array([1.0, 0.0]))
    assert trail.last_seen_heading == pytest.approx(0.0, abs=1e-9)
    trail.observe(2.0, np.array([1.0, 1.0]))
    assert trail.last_seen_heading == pytest.approx(math.pi / 2, abs=1e-9)


def test_the_extrapolation_is_capped_so_it_stops_growing():
    """A linear guess about a walking person stops meaning anything after a few s."""
    trail = GuardianTrail()
    trail.observe(0.0, np.array([0.0, 0.0]))
    trail.observe(1.0, np.array([1.0, 0.0]))
    near = trail.extrapolated(5.0, speed=0.175, max_s=4.0)
    far = trail.extrapolated(60.0, speed=0.175, max_s=4.0)
    assert np.allclose(near, far)
    assert float(np.linalg.norm(far - trail.last_seen_xy)) == \
        pytest.approx(0.175 * 4.0, abs=1e-9)


def test_the_trail_record_is_json_shaped_and_reports_its_length():
    trail = GuardianTrail()
    for step in range(20):
        trail.observe(step * 0.5, np.array([step * 0.25, 0.0]))
    record = trail.as_record()
    assert len(record["points"]) == len(record["times"])
    assert record["length_m"] == pytest.approx(trail.length_m())
    assert record["last_seen_heading_deg"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------ route planning
def test_a_clear_straight_line_is_planned_as_a_straight_line():
    route = plan_route((2.60, -2.00), (2.60, 1.80))
    assert route.feasible is True
    assert len(route.waypoints) == 2
    assert route.direct_blocked_by == ""
    assert route.bends_around == ()


def test_a_route_through_the_kiosk_is_refused_and_bent_around_it():
    route = plan_route(KIOSK_START, KIOSK_GOAL)
    assert route.direct_blocked_by == "kiosk"
    assert route.feasible is True
    assert len(route.waypoints) > 2


def test_every_segment_of_a_planned_route_is_actually_clear():
    """The claim is about the whole polyline, not only about the endpoints."""
    route = plan_route(KIOSK_START, KIOSK_GOAL)
    for a, b in zip(route.waypoints, route.waypoints[1:]):
        assert segment_blocked(a, b) == "", f"{a} -> {b}"


def test_a_planned_route_is_longer_than_the_straight_line_it_replaced():
    route = plan_route(KIOSK_START, KIOSK_GOAL)
    straight = float(np.linalg.norm(
        np.array(KIOSK_GOAL) - np.array(KIOSK_START)))
    assert route.length_m > straight
    assert route.length_m == pytest.approx(
        sum(float(np.linalg.norm(b - a))
            for a, b in zip(route.waypoints, route.waypoints[1:])), abs=1e-6)


def test_the_route_starts_at_the_duck_and_ends_at_the_goal():
    route = plan_route(KIOSK_START, KIOSK_GOAL)
    assert route.waypoints[0] == pytest.approx(np.array(KIOSK_START))
    assert route.waypoints[-1] == pytest.approx(np.array(KIOSK_GOAL))


def test_every_waypoint_candidate_clears_every_obstacle_and_stays_in_the_hall():
    candidates = waypoint_candidates()
    assert len(candidates) > 8
    for point in candidates:
        assert clear_of_obstacles(point, 0.0)
        for obstacle in OBSTACLES:
            assert obstacle.distance_to(point) >= PLAN_INFLATE_M - 1e-6


def test_an_impossible_goal_is_reported_infeasible_rather_than_faked():
    """A goal inside an inflated obstacle has no honest route; say so."""
    route = plan_route((-2.30, -1.60), (-2.30, 0.60))
    assert route.direct_blocked_by == "crates"
    assert route.feasible is False


def test_a_person_standing_in_the_way_blocks_a_segment_by_name():
    who = segment_blocked((0.00, -1.90), (2.60, -1.90),
                          {"dahl": np.array([1.30, -1.90])})
    assert who == "person:dahl"


def test_a_person_off_to_one_side_does_not_block_the_segment():
    assert segment_blocked((0.00, -1.90), (2.60, -1.90),
                           {"dahl": np.array([1.30, -0.90])}) == ""


def test_the_crowd_is_tested_at_the_time_the_duck_would_arrive():
    """A static snapshot routes the duck through where somebody is about to be."""
    def walking(t):
        return {"dahl": np.array([2.00, -2.00 + 0.20 * t])}

    blocked = plan_route((2.00, -2.20), (2.00, 1.60),
                         people_at_time=walking, speed=0.209, t0=0.0)
    assert blocked.direct_blocked_by.startswith("person:")
    # The same geometry with nobody in it is a straight line.
    assert plan_route((2.00, -2.20), (2.00, 1.60)).direct_blocked_by == ""


def test_planning_with_no_crowd_argument_grades_the_scenery_only():
    route = plan_route(KIOSK_START, KIOSK_GOAL, people_at_time=None)
    assert route.feasible is True
    assert route.direct_blocked_by == "kiosk"


def test_the_route_record_is_json_shaped_and_self_describing():
    record = plan_route(KIOSK_START, KIOSK_GOAL).as_record()
    assert record["waypoint_count"] == len(record["waypoints"])
    assert record["feasible"] is True
    assert record["direct_blocked_by"] == "kiosk"
    assert record["length_m"] > 0.0


def test_line_of_sight_availability_is_advisory_and_names_its_blocker():
    ok, blocker = line_of_sight_available((2.00, -0.60), (-0.60, 1.00))
    assert ok is False and blocker == "kiosk"
    ok, blocker = line_of_sight_available((2.60, -2.00), (2.60, 1.80))
    assert ok is True and blocker == ""


# --------------------------------------------- the waypoint cursor regression
def test_the_stateless_selector_re_chases_a_passed_corner():
    """REGRESSION.  This is the defect the rollout's monotonic cursor exists for.

    ``route_progress`` returns the FIRST waypoint farther than its tolerance.
    Standing at the goal, waypoint 1 is far away again — because the duck walked
    past it — so the stateless rule sends the duck back to a corner it already
    visited, and the rejoin loops until it hits its ceiling.
    """
    route = plan_route(KIOSK_START, KIOSK_GOAL)
    at_goal = np.array(KIOSK_GOAL)
    index, remaining = route_progress(route, at_goal)
    assert remaining < 0.05                     # the duck IS at the goal
    assert index == 1                           # ...and is sent back to corner 1
    passed = route.waypoints[1]
    assert float(np.linalg.norm(passed - at_goal)) > 1.5


def _monotonic_cursor(route, positions, tolerance: float = 0.18):
    """The rollout's advance rule, replayed on a hand-built path."""
    last = len(route.waypoints) - 1
    cursor = 1 if last >= 1 else 0
    seen = []
    for position in positions:
        while (cursor < last
               and float(np.linalg.norm(
                   route.waypoints[cursor] - np.asarray(position))) <= tolerance):
            cursor += 1
        seen.append(cursor)
    return seen


def test_the_monotonic_cursor_never_goes_backwards():
    route = plan_route(KIOSK_START, KIOSK_GOAL)
    # Walk the polyline, then sit at the goal for a while.
    path = []
    for a, b in zip(route.waypoints, route.waypoints[1:]):
        for step in range(40):
            path.append(a + (b - a) * (step / 39.0))
    path.extend([np.array(KIOSK_GOAL)] * 40)

    seen = _monotonic_cursor(route, path)
    assert seen == sorted(seen)
    assert seen[0] == 1
    assert seen[-1] == len(route.waypoints) - 1


def test_the_monotonic_cursor_reaches_the_last_waypoint_and_stays_there():
    route = plan_route(KIOSK_START, KIOSK_GOAL)
    last = len(route.waypoints) - 1
    seen = _monotonic_cursor(route, list(route.waypoints) + [
        np.array(KIOSK_GOAL)] * 20)
    assert seen[-1] == last
    assert max(seen) == last


def test_the_monotonic_cursor_and_the_stateless_selector_disagree_at_the_goal():
    """The disagreement IS the bug; keeping it visible keeps the fix honest."""
    route = plan_route(KIOSK_START, KIOSK_GOAL)
    at_goal = [np.array(KIOSK_GOAL)]
    walked = _monotonic_cursor(route, list(route.waypoints) + at_goal)[-1]
    stateless = route_progress(route, np.array(KIOSK_GOAL))[0]
    assert walked == len(route.waypoints) - 1
    assert stateless < walked
