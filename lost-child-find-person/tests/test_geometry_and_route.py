#!/usr/bin/env python3
"""The concourse's own geometry: occluders, obstacles and the standoff band.

Pure numpy and dataclasses: no MuJoCo, no policy, no rendering.  What is graded
here is the layer everything else is built on — that an occluder is DERIVED from
its height rather than declared, that a segment test cannot slip through a
solid, and that the standoff band leaves real surface gap rather than being a
round number somebody liked.

The trail and the route planner that consume this layer are graded in
``test_memory_and_route``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lost_geometry import (  # noqa: E402
    ADULT_HALF_EXTENT_M,
    CONTACT_SEPARATION_M,
    DUCK_PLANAR_RADIUS,
    DUCK_START_XY,
    FOLLOW_DISTANCE_M,
    FOLLOW_FAR_M,
    FOLLOW_NEAR_M,
    PERSON_CLEARANCE_M,
    PLAN_INFLATE_M,
    ROUTE_CLEARANCE_M,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
    STANDOFF_TARGET_M,
    approach_point,
    nearest_obstacle,
    obstacle_height,
    safe_standoff_point,
    standoff_ok,
    standoff_verdict,
    surface_gap,
)
from plaza_layout import (  # noqa: E402
    BENCH,
    BY_NAME,
    FLOOR_HALF,
    KIOSK,
    OBSTACLES,
    OCCLUDERS,
    OCCLUDING_HEIGHT_M,
    clear_of_obstacles,
    occluder_between,
)

# ------------------------------------------------------------- the concourse
def test_every_occluder_is_taller_than_the_highest_camera_sample():
    """z=0.66 is the crown sample; the eye is near z=0.19, so 0.90 m suffices."""
    assert OCCLUDING_HEIGHT_M >= 0.90
    for obstacle in OCCLUDERS:
        assert obstacle.height_m >= OCCLUDING_HEIGHT_M, obstacle.name


def test_occlusion_is_derived_from_height_and_never_declared():
    """A scene edit that shortens a body must stop it counting as an occluder."""
    for obstacle in OBSTACLES:
        assert obstacle.occludes == (obstacle.height_m >= OCCLUDING_HEIGHT_M)


def test_the_bench_is_a_route_obstacle_that_is_deliberately_not_an_occluder():
    """It separates 'the route avoided something' from 'the camera was blocked'."""
    assert BENCH in OBSTACLES
    assert BENCH not in OCCLUDERS
    assert BENCH.occludes is False
    assert BENCH.height_m < 0.30


def test_the_kiosk_is_the_principal_occluder_and_is_a_real_solid():
    assert KIOSK.occludes is True
    assert KIOSK.height_m == 1.10
    assert KIOSK.kind == "box"
    assert min(KIOSK.half) >= 0.40


def test_obstacle_names_are_unique_and_indexed():
    assert len(BY_NAME) == len(OBSTACLES)
    for obstacle in OBSTACLES:
        assert BY_NAME[obstacle.name] is obstacle
        assert obstacle_height(obstacle.name) == obstacle.height_m


def test_distance_to_a_box_is_zero_on_its_face_and_negative_inside():
    face = (KIOSK.center[0] + KIOSK.half[0], KIOSK.center[1])
    assert KIOSK.distance_to(face) == pytest.approx(0.0, abs=1e-9)
    assert KIOSK.distance_to(KIOSK.center) < 0.0
    outside = (KIOSK.center[0] + KIOSK.half[0] + 0.5, KIOSK.center[1])
    assert KIOSK.distance_to(outside) == pytest.approx(0.5)


def test_distance_to_a_circle_is_the_radial_gap():
    column = BY_NAME["column_w"]
    assert column.kind == "circle"
    point = (column.center[0] + column.radius + 0.25, column.center[1])
    assert column.distance_to(point) == pytest.approx(0.25)


def test_a_segment_through_an_obstacle_is_detected_from_either_direction():
    a = (KIOSK.center[0] - 2.0, KIOSK.center[1])
    b = (KIOSK.center[0] + 2.0, KIOSK.center[1])
    assert KIOSK.segment_hits(a, b) is True
    assert KIOSK.segment_hits(b, a) is True


def test_a_segment_that_merely_passes_nearby_is_not_a_hit():
    offset = KIOSK.center[1] + KIOSK.half[1] + 0.40
    assert KIOSK.segment_hits((-2.0, offset), (2.0, offset)) is False
    # ...but it is a hit once inflated past the gap.
    assert KIOSK.segment_hits((-2.0, offset), (2.0, offset), 0.50) is True


def test_segment_endpoints_are_included_in_the_hit_test():
    inside = KIOSK.center
    assert KIOSK.segment_hits(inside, (3.0, 2.0)) is True
    assert KIOSK.segment_hits((3.0, 2.0), inside) is True


def test_a_box_offers_four_inflated_corners_and_a_circle_eight_points():
    assert len(KIOSK.corners(0.3)) == 4
    assert len(BY_NAME["column_w"].corners(0.3)) == 8
    for corner in KIOSK.corners(0.3):
        assert KIOSK.distance_to(corner) >= 0.3 - 1e-9


def test_the_sightline_predicate_names_the_body_in_the_way():
    assert occluder_between((2.0, -0.60), (-0.60, 1.00)) == "kiosk"
    assert occluder_between((2.60, -2.00), (2.60, 2.00)) is None


def test_the_low_bench_never_blocks_a_sightline():
    """Planar-only occlusion must consult OCCLUDERS, not every obstacle."""
    a = (BENCH.center[0] - 1.0, BENCH.center[1])
    b = (BENCH.center[0] + 1.0, BENCH.center[1])
    assert BENCH.segment_hits(a, b) is True
    assert occluder_between(a, b) is None


def test_clear_of_obstacles_rejects_points_outside_the_hall():
    assert clear_of_obstacles((0.0, -2.10), 0.10) is True
    assert clear_of_obstacles((FLOOR_HALF[0] + 0.1, 0.0), 0.10) is False
    assert clear_of_obstacles((0.0, FLOOR_HALF[1] + 0.1), 0.10) is False
    assert clear_of_obstacles(KIOSK.center, 0.10) is False


def test_the_duck_starts_behind_the_guardian_and_inside_the_hall():
    assert clear_of_obstacles(DUCK_START_XY, ROUTE_CLEARANCE_M) is True


def test_nearest_obstacle_names_the_closest_surface():
    """Just east of the kiosk's east face, and nothing else is nearer."""
    probe = (KIOSK.center[0] + KIOSK.half[0] + 0.30, KIOSK.center[1])
    name, gap = nearest_obstacle(probe)
    assert name == "kiosk"
    assert gap == pytest.approx(0.30)
    assert all(BY_NAME[o.name].distance_to(probe) >= gap for o in OBSTACLES)


# ------------------------------------------------------------ the standoff
def test_the_standoff_band_leaves_real_surface_gap_at_its_near_edge():
    """The NOMINAL sizing argument behind the band's near edge.

    0.155 m is a nominal figure computed from the legacy nominal body width
    ``ADULT_HALF_EXTENT_M`` (see its note in ``lost_geometry``), not a measured
    clearance and not an acceptance gate.  It states only that the near edge of
    the band was sized with roughly an arm's swing in mind.  The actual safety
    property is measured per tick by ``ContactProbe`` against the real geoms.
    """
    assert CONTACT_SEPARATION_M == pytest.approx(
        DUCK_PLANAR_RADIUS + ADULT_HALF_EXTENT_M)
    assert surface_gap(STANDOFF_MIN_M) == pytest.approx(0.155, abs=1e-3)
    assert surface_gap(STANDOFF_MIN_M) > 0.10


def test_the_standoff_band_is_two_sided_and_contains_its_target():
    assert STANDOFF_MIN_M < STANDOFF_TARGET_M < STANDOFF_MAX_M
    assert standoff_ok(STANDOFF_TARGET_M)
    assert not standoff_ok(STANDOFF_MIN_M - 0.01)
    assert not standoff_ok(STANDOFF_MAX_M + 0.01)
    assert standoff_verdict(0.30) == "too close"
    assert standoff_verdict(STANDOFF_TARGET_M) == "in band"
    assert standoff_verdict(1.20) == "too far"


def test_the_follow_distance_is_larger_than_the_standoff_band():
    """So the kiosk can come between them: the loss is geometric, not staged."""
    assert FOLLOW_DISTANCE_M > STANDOFF_MAX_M
    assert FOLLOW_NEAR_M < FOLLOW_DISTANCE_M < FOLLOW_FAR_M


def test_the_approach_point_stops_short_on_the_ducks_own_bearing():
    """The duck must never have to walk through her to reach its own goal."""
    target = np.array([0.0, 0.0])
    duck = np.array([2.0, 0.0])
    point = approach_point(target, duck, 0.60)
    assert float(np.linalg.norm(point - target)) == pytest.approx(0.60)
    assert point[0] > 0.0            # on the duck's side


def test_the_approach_point_is_defined_even_when_the_duck_is_on_top_of_her():
    point = approach_point((1.0, 1.0), (1.0, 1.0), 0.60)
    assert float(np.linalg.norm(np.asarray(point) - np.array([1.0, 1.0]))) == \
        pytest.approx(0.60)


def test_a_standoff_station_inside_a_column_is_rotated_out_of_it():
    """She is entitled to stand beside a column; the duck still needs a station."""
    column = BY_NAME["column_w"]
    guardian = (column.center[0] + 0.30, column.center[1])
    duck = (column.center[0] - 2.0, column.center[1])
    naive = approach_point(guardian, duck, STANDOFF_TARGET_M)
    assert clear_of_obstacles(naive, ROUTE_CLEARANCE_M) is False

    station = safe_standoff_point(guardian, duck, STANDOFF_TARGET_M)
    assert clear_of_obstacles(station, ROUTE_CLEARANCE_M) is True


def test_rotating_the_station_never_trades_away_the_standoff_distance():
    """The band is the claim; convenience must not buy its way out of it."""
    column = BY_NAME["column_w"]
    guardian = np.array([column.center[0] + 0.30, column.center[1]])
    station = safe_standoff_point(guardian, (column.center[0] - 2.0,
                                             column.center[1]),
                                  STANDOFF_TARGET_M)
    assert float(np.linalg.norm(station - guardian)) == \
        pytest.approx(STANDOFF_TARGET_M, abs=1e-9)
    assert standoff_ok(float(np.linalg.norm(station - guardian)))


def test_an_unobstructed_station_is_left_exactly_where_it_was():
    guardian = (0.0, -1.80)
    station = safe_standoff_point(guardian, (2.0, -1.80), STANDOFF_TARGET_M)
    expected = approach_point(guardian, (2.0, -1.80), STANDOFF_TARGET_M)
    assert station == pytest.approx(expected)


def test_the_planner_inflation_exceeds_the_gate_clearance():
    """So the acceptance gate has slack rather than sitting on the planner's edge."""
    assert PLAN_INFLATE_M == pytest.approx(DUCK_PLANAR_RADIUS + ROUTE_CLEARANCE_M)
    assert PLAN_INFLATE_M > ROUTE_CLEARANCE_M
    assert PERSON_CLEARANCE_M > 0.0


