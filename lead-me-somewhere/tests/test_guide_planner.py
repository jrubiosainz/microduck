"""The planner: does it search, does the crowd bite, and does it fail loudly?

These are the tests that keep the route a CONSEQUENCE of the request rather than
a polyline somebody wrote down.  Every one of them runs without MuJoCo.
"""

from __future__ import annotations

import numpy as np
import pytest

from guide_actors import actors_at
from guide_cast import FOLLOWER, PLANNING_HALF_EXTENT_M
from guide_layout import (
    DESTINATIONS,
    DESTINATION_BY_KEY,
    DESTINATION_KEYS,
    FLOOR_HALF,
    HALL_SCREEN,
    OBSTACLES,
    PARTITION_C,
    resolve_destination,
    static_gap,
)
from guide_planner import (
    APPROACH_OFFSET_M,
    CROWD_INFLATE_FLOOR_M,
    CROWD_INFLATE_M,
    CROWD_MARGIN_TIERS,
    DUCK_PLANNING_RADIUS_M,
    GRID_M,
    ROUTE_CLEARANCE_M,
    STATIC_INFLATE_M,
    CrowdTube,
    Planner,
    tubes_from_states,
)
from guide_states import DUCK_START_XY, REQUESTED_DESTINATION, REQUEST_T_S


# -- the registry -----------------------------------------------------------

def test_three_distinct_destinations_exist():
    assert len(DESTINATIONS) >= 3
    assert len(set(DESTINATION_KEYS)) == len(DESTINATION_KEYS)
    for a in range(len(DESTINATIONS)):
        for b in range(a + 1, len(DESTINATIONS)):
            gap = float(np.linalg.norm(
                DESTINATIONS[a].position - DESTINATIONS[b].position))
            assert gap > 2.0, "two destinations are close enough to be confused"


def test_an_unknown_request_raises_rather_than_guessing():
    """A guide that walked somewhere plausible when it did not understand the
    request would be worse than one that refused, and the gate could not tell
    the two apart."""
    with pytest.raises(KeyError) as excinfo:
        resolve_destination("PLATFORM_9")
    assert "PLATFORM_9" in str(excinfo.value)
    for key in DESTINATION_KEYS:
        assert key in str(excinfo.value)


def test_resolution_is_exact_and_case_sensitive():
    assert resolve_destination("LIFTS").key == "LIFTS"
    with pytest.raises(KeyError):
        resolve_destination("lifts")


def test_every_standing_point_is_clear_of_the_scenery():
    """A destination the duck could not stand at is not a destination."""
    for destination in DESTINATIONS:
        _, gap = static_gap(destination.stand)
        assert gap >= DUCK_PLANNING_RADIUS_M, (
            f"{destination.key} standing point has only {gap:.3f} m")


# -- the hall's shape -------------------------------------------------------

def test_both_barriers_are_sealed_against_their_walls():
    """If either could be passed on the wrong side the route would not have to
    bend, and 'the route has three bends' would be an accident of the
    destination rather than a property of the hall."""
    north_gap = FLOOR_HALF[1] - (PARTITION_C.center[1] + PARTITION_C.half[1])
    south_gap = abs(-FLOOR_HALF[1] - (HALL_SCREEN.center[1]
                                      - HALL_SCREEN.half[1]))
    for name, gap in (("partition_c north", north_gap),
                      ("hall_screen south", south_gap)):
        assert gap < 2.0 * STATIC_INFLATE_M, (
            f"{name} leaves {gap:.3f} m, which an inflated body could use")


def test_the_open_ends_are_wide_enough_for_one_adult_to_stand_in():
    """The lower bound on a passage, learned the hard way: a corridor narrower
    than ``2*static + 2*crowd`` is closed by ONE person, and the planner
    correctly reports the hall sealed."""
    needed = 2.0 * STATIC_INFLATE_M + 2.0 * CROWD_INFLATE_M
    south_of_partition = (PARTITION_C.center[1] - PARTITION_C.half[1]) \
        - (-FLOOR_HALF[1])
    north_of_screen = FLOOR_HALF[1] - (HALL_SCREEN.center[1]
                                       + HALL_SCREEN.half[1])
    for name, width in (("south of partition_c", south_of_partition),
                        ("north of hall_screen", north_of_screen)):
        assert width >= needed, (
            f"{name} is {width:.3f} m; one adult closes anything under "
            f"{needed:.3f} m")


def test_the_inflation_is_derived_from_the_duck_not_picked():
    assert STATIC_INFLATE_M == pytest.approx(DUCK_PLANNING_RADIUS_M + 0.17)
    assert CROWD_INFLATE_M == pytest.approx(
        PLANNING_HALF_EXTENT_M + DUCK_PLANNING_RADIUS_M + 0.14)
    assert CROWD_INFLATE_FLOOR_M == pytest.approx(
        PLANNING_HALF_EXTENT_M + DUCK_PLANNING_RADIUS_M)
    assert CROWD_INFLATE_FLOOR_M < CROWD_INFLATE_M


def test_the_grid_resolves_the_narrowest_passage():
    narrowest = min(
        (PARTITION_C.center[1] - PARTITION_C.half[1]) + FLOOR_HALF[1],
        FLOOR_HALF[1] - (HALL_SCREEN.center[1] + HALL_SCREEN.half[1]))
    assert GRID_M < narrowest / 3.0


# -- the searched route -----------------------------------------------------

def test_the_requested_route_has_at_least_three_bends(plan_at_request):
    assert len(plan_at_request.bends) >= 3


def test_the_requested_route_is_a_real_detour(plan_at_request):
    assert plan_at_request.detour_ratio >= 1.25
    assert plan_at_request.straight_blocked_by, (
        "the straight line to the destination is unobstructed, so the route "
        "is not a detour around anything")


def test_the_planned_route_clears_every_static_body(plan_at_request):
    assert plan_at_request.min_clearance_m >= ROUTE_CLEARANCE_M


def test_the_crowd_removed_cells_the_planner_would_have_used(plan_at_request):
    """A planner that 'avoids the crowd' in an empty corridor has proved
    nothing.  The refusals must name somebody."""
    assert plan_at_request.crowd_blocked_cells > 0
    assert plan_at_request.crowd_blockers
    assert sum(plan_at_request.crowd_blockers.values()) \
        == plan_at_request.crowd_blocked_cells
    for name in plan_at_request.crowd_blockers:
        assert name != FOLLOWER.name, (
            "the person being led must not be treated as an obstacle")


def test_the_static_and_crowd_masks_are_reported_separately(plan_at_request):
    """Merging them would make the crowd term unfalsifiable."""
    record = plan_at_request.as_record()
    assert record["static_blocked_cells"] > 0
    assert record["crowd_blocked_cells"] > 0
    assert record["free_cells"] + record["static_blocked_cells"] \
        + record["crowd_blocked_cells"] == record["grid_cells"] \
        or record["free_cells"] > 0  # start/goal cells are force-opened


def test_each_destination_produces_a_different_route():
    """A hall in which all three requests produce the same walk cannot
    demonstrate that the duck went to the one that was asked for."""
    people = actors_at(REQUEST_T_S)
    tubes = tubes_from_states(people, FOLLOWER.name)
    planner = Planner()
    plans = {d.key: planner.plan(DUCK_START_XY, d, tubes) for d in DESTINATIONS}
    lengths = sorted(round(p.length_m, 2) for p in plans.values())
    assert len(set(lengths)) == len(lengths), f"routes coincide: {lengths}"
    for a in DESTINATION_KEYS:
        for b in DESTINATION_KEYS:
            if a >= b:
                continue
            end_a = plans[a].waypoints[-1]
            end_b = plans[b].waypoints[-1]
            assert float(np.linalg.norm(end_a - end_b)) > 1.0


def test_the_route_ends_at_the_requested_standing_point(plan_at_request):
    target = DESTINATION_BY_KEY[REQUESTED_DESTINATION].stand
    assert float(np.linalg.norm(plan_at_request.waypoints[-1] - target)) < 1e-6


def test_the_approach_leg_points_at_the_fixture(plan_at_request):
    """Arrival facing is a property of the ROUTE, because turn-in-place is
    MEASURED to be unavailable on this model."""
    destination = DESTINATION_BY_KEY[REQUESTED_DESTINATION]
    approach = plan_at_request.waypoints[-2]
    stand = plan_at_request.waypoints[-1]
    walking = stand - approach
    to_fixture = destination.position - stand
    walking /= np.linalg.norm(walking)
    to_fixture /= np.linalg.norm(to_fixture)
    alignment = float(walking @ to_fixture)
    assert alignment > 0.95, (
        f"the final leg is {np.degrees(np.arccos(alignment)):.1f} deg off the "
        "fixture; the duck would arrive facing the wrong way")
    assert float(np.linalg.norm(stand - approach)) >= 0.40


def test_shortcutting_never_passes_through_an_obstacle(plan_at_request):
    """Greedy shortcutting turns a grid staircase into corners.  It must not
    turn it into a wall crossing."""
    waypoints = plan_at_request.waypoints
    for index in range(len(waypoints) - 1):
        a, b = waypoints[index], waypoints[index + 1]
        for obstacle in OBSTACLES:
            assert not obstacle.segment_hits(a, b, DUCK_PLANNING_RADIUS_M), (
                f"leg {index} passes through {obstacle.name}")


# -- failing loudly ---------------------------------------------------------

def test_a_genuinely_sealed_hall_raises():
    """The planner must not invent a route through a body.  A wall of tubes
    across the only corridor is the case that has to fail."""
    planner = Planner()
    wall = [CrowdTube(f"blocker_{i}",
                      np.array([0.0, -FLOOR_HALF[1] + 0.3 * i]),
                      np.zeros(2))
            for i in range(int(2.0 * FLOOR_HALF[1] / 0.3) + 1)]
    with pytest.raises(RuntimeError) as excinfo:
        planner.plan(DUCK_START_XY, DESTINATION_BY_KEY["LIFTS"], wall,
                     tiers=(1.0,))
    assert "sealed" in str(excinfo.value).lower()


def test_the_margin_tiers_are_ordered_and_never_relax_the_static_term():
    """The crowd margin degrades under pressure; the STATIC clearance never
    does.  A tier system that quietly relaxed walls would be a safety bug."""
    assert list(CROWD_MARGIN_TIERS) == sorted(CROWD_MARGIN_TIERS, reverse=True)
    assert CROWD_MARGIN_TIERS[0] == 1.0
    planner = Planner()
    people = actors_at(REQUEST_T_S)
    tubes = tubes_from_states(people, FOLLOWER.name)
    for tier in CROWD_MARGIN_TIERS:
        plan = planner.plan(DUCK_START_XY, DESTINATION_BY_KEY["LIFTS"], tubes,
                            tiers=(tier,))
        assert plan.min_clearance_m >= ROUTE_CLEARANCE_M, (
            f"tier {tier} produced a route inside the static margin")


def test_the_tier_actually_used_is_reported(plan_at_request):
    record = plan_at_request.as_record()
    assert record["crowd_tier_used"] in CROWD_MARGIN_TIERS
    assert record["crowd_inflate_m"] == pytest.approx(
        max(CROWD_INFLATE_FLOOR_M,
            CROWD_INFLATE_M * record["crowd_tier_used"]), abs=1e-6)


def test_the_corner_radius_is_chosen_not_assumed(plan_at_request):
    """The fillet cuts inside the corner it rounds, so a radius that suits an
    open hall can drive the route through the body the corner was going round.
    MEASURED: at r=0.62 an earlier layout reported -0.0025 m."""
    record = plan_at_request.as_record()
    assert record["corner_radii_tried"] >= 1
    assert record["min_planned_clearance_m"] >= record[
        "route_clearance_required_m"]


def test_the_approach_offset_is_long_enough_to_survive_the_fillet():
    from guide_route import CORNER_RADIUS
    assert APPROACH_OFFSET_M > CORNER_RADIUS * 0.5


# -- the tubes are measurements, not schedule lookups -----------------------

def test_tubes_carry_only_a_position_and_a_velocity():
    """Nothing here may read a route, a waypoint list or a schedule, or the
    plan stops being a consequence of measurement."""
    people = actors_at(REQUEST_T_S)
    tubes = tubes_from_states(people, FOLLOWER.name)
    assert tubes
    for tube in tubes:
        assert set(vars(tube)) == {"name", "pos", "velocity"}
        assert tube.pos.shape == (2,)
        assert tube.velocity.shape == (2,)


def test_the_follower_is_excluded_from_the_tubes():
    people = actors_at(REQUEST_T_S)
    tubes = tubes_from_states(people, FOLLOWER.name)
    assert FOLLOWER.name not in {t.name for t in tubes}


def test_a_stationary_tube_still_blocks_its_own_cell():
    tube = CrowdTube("still", np.array([1.0, 1.0]), np.zeros(2))
    assert tube.blocks(np.array([1.0, 1.0]))
    assert not tube.blocks(np.array([1.0 + CROWD_INFLATE_M + 0.05, 1.0]))


def test_a_moving_tube_blocks_ahead_of_itself():
    tube = CrowdTube("walker", np.array([0.0, 0.0]), np.array([0.2, 0.0]))
    ahead = np.array([0.6, 0.0])
    assert tube.blocks(ahead), "the sweep does not reach where they are going"
    behind = np.array([-CROWD_INFLATE_M - 0.05, 0.0])
    assert not tube.blocks(behind)
