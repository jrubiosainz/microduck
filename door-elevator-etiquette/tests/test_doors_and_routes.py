#!/usr/bin/env python3
"""The doors, the routes and the scene: geometry, continuity and the model.

The door tests matter because the open fraction is the only thing the rest of
the behavior sees of a door, and the "no movement through a closed door" gate is
graded on it.  The route tests matter because a hard vertex in a walker's path is
a teleport of its body axis, and this behavior found three of them the hard way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from etiquette_actors import (
    ACTOR_CORNER_RADIUS,
    ROUTES,
    max_heading_step,
    people_at,
)
from etiquette_cast import ALL_NAMES, GUARDIAN, PEOPLE
from etiquette_path import (
    aperture_crossings,
    build_route,
    careful_bands,
    in_careful_band,
    leg_bounds,
    route_bend_report,
)
from etiquette_route import Route
from etiquette_states import (
    DUCK_PLANAR_RADIUS,
    MIN_LEFT_TURN_RADIUS_M,
    MIN_RIGHT_TURN_RADIUS_M,
)
from lobby_doors import (
    APERTURE_NAMES,
    DOOR_PASSABLE_FRACTION,
    DOOR_RAMP_S,
    DOOR_SCHEDULE,
    doors_at,
    open_fraction,
)
from lobby_layout import OCCLUDERS, OCCLUDING_HEIGHT_M, STATIC_OBSTACLES


# -- the doors ---------------------------------------------------------------
def test_a_shut_door_has_exactly_zero_clear_gap():
    for name in APERTURE_NAMES:
        door = doors_at(0.0)[name]
        if door.fraction == 0.0:
            assert door.effective_gap_m == 0.0
            assert door.closed
            assert not door.passable


def test_a_fully_open_door_restores_the_whole_clear_width():
    for name, open_at, _ in DOOR_SCHEDULE:
        door = doors_at(open_at + DOOR_RAMP_S + 0.5)[name]
        assert door.fraction == pytest.approx(1.0, abs=1e-6)
        assert door.effective_gap_m == pytest.approx(door.clear_w, abs=1e-9)


def test_the_effective_gap_is_DERIVED_from_the_fraction():
    """A schedule edit must not be able to leave a stale width behind."""
    for name in APERTURE_NAMES:
        for t in (0.0, 6.0, 20.0, 50.0, 80.0, 100.0):
            door = doors_at(t)[name]
            assert door.effective_gap_m == pytest.approx(
                2.0 * door.travel * door.fraction)


def test_the_open_fraction_is_continuous_across_every_edge():
    """A door that snapped open in one tick would be a teleport."""
    for name in APERTURE_NAMES:
        previous = open_fraction(name, 0.0)
        for step in range(1, 11001):
            t = step * 0.01
            value = open_fraction(name, t)
            assert abs(value - previous) < 0.02, (name, t)
            previous = value


def test_the_fraction_stays_inside_zero_and_one():
    for name in APERTURE_NAMES:
        for step in range(0, 11001, 7):
            value = open_fraction(name, step * 0.01)
            assert 0.0 <= value <= 1.0


def test_a_leaf_covers_its_half_of_the_opening_when_shut():
    door = doors_at(0.0)["lift_front"]
    south, north = door.leaf_offsets()
    assert south == pytest.approx(-0.5 * door.clear_w)
    assert north == pytest.approx(+0.5 * door.clear_w)


def test_the_passable_threshold_leaves_room_for_the_duck():
    """A door at the passable fraction must clear the robot's own width."""
    for name in APERTURE_NAMES:
        door = doors_at(0.0)[name]
        gap = 2.0 * door.travel * DOOR_PASSABLE_FRACTION
        assert gap > 2.0 * DUCK_PLANAR_RADIUS, name


# -- the duck's route --------------------------------------------------------
def test_every_bend_fits_the_measured_turning_circle_for_its_own_sign():
    for bend in route_bend_report(build_route()):
        needed = (MIN_LEFT_TURN_RADIUS_M if bend["hand"] == "left"
                  else MIN_RIGHT_TURN_RADIUS_M)
        assert bend["radius_m"] >= needed, bend
        assert bend["walkable"]


def test_the_route_crosses_the_middle_of_every_aperture():
    for crossing in aperture_crossings(build_route()):
        assert crossing["crossed"], crossing
        assert crossing["margin_m"] > DUCK_PLANAR_RADIUS, crossing


def test_the_legs_are_monotonic_and_end_at_the_route_end():
    route = build_route()
    bounds = leg_bounds(route)
    assert bounds == sorted(bounds)
    assert bounds[-1] == pytest.approx(route.length, abs=0.02)


def test_the_careful_bands_cover_every_aperture_crossing():
    route = build_route()
    bands = careful_bands(route)
    assert len(bands) == len(APERTURE_NAMES)
    for crossing in aperture_crossings(route):
        assert in_careful_band(bands, crossing["arc_s_m"]), crossing


# -- the route primitive -----------------------------------------------------
def test_a_corner_that_cannot_be_filleted_RAISES_rather_than_going_silent():
    """A hard vertex turns its walker through a whole corner in one tick.

    This behavior shipped three of them before the exception existed, and the
    symptom surfaced far away as a 51 deg single-tick heading change with
    nothing in the route to point at.
    """
    with pytest.raises(ValueError, match="hard vertex"):
        Route("tight", ((0.0, 0.0), (0.3, 0.0), (0.3, 0.6)), 0.1, radius=0.5)


def test_the_exception_names_a_radius_that_actually_fits():
    try:
        Route("tight", ((0.0, 0.0), (0.3, 0.0), (0.3, 0.6)), 0.1, radius=0.5)
    except ValueError as error:
        suggested = float(str(error).split("below ")[1].split(" m")[0])
    Route("tight", ((0.0, 0.0), (0.3, 0.0), (0.3, 0.6)), 0.1,
          radius=suggested * 0.95)


def test_position_and_tangent_are_continuous_across_every_piece():
    for route in list(ROUTES.values()) + [build_route()]:
        previous_point, previous_tangent = route.pose_at_arc(0.0)
        for step in range(1, 2001):
            s = route.length * step / 2000
            point, tangent = route.pose_at_arc(s)
            assert float(np.linalg.norm(point - previous_point)) < 0.02
            turn = abs(math.degrees(math.atan2(
                float(previous_tangent[0] * tangent[1]
                      - previous_tangent[1] * tangent[0]),
                float(previous_tangent @ tangent))))
            assert turn < 3.0, (route.name, s, turn)
            previous_point, previous_tangent = point, tangent


def test_a_hold_window_stops_the_walker_and_resumes_it_continuously():
    route = Route("held", ((0.0, 0.0), (2.0, 0.0)), 0.1,
                  hold_windows=((5.0, 9.0),))
    before = route.pos_at(4.99)
    during = route.pos_at(7.0)
    after = route.pos_at(9.01)
    assert route.speed_at(7.0) == 0.0
    assert float(np.linalg.norm(during - before)) < 0.01
    assert float(np.linalg.norm(after - during)) < 0.01
    # And it really was held: arc at 12 s is 3 s of walking short of unheld.
    unheld = Route("free", ((0.0, 0.0), (2.0, 0.0)), 0.1)
    assert route.arc_at(12.0) == pytest.approx(unheld.arc_at(12.0) - 0.4)


def test_finish_time_accounts_for_the_holds():
    route = Route("held", ((0.0, 0.0), (2.0, 0.0)), 0.1,
                  hold_windows=((5.0, 9.0),))
    assert route.finish_t() == pytest.approx(4.0 + 2.0 / 0.1)


# -- the scripted people -----------------------------------------------------
def test_no_scripted_person_turns_faster_than_a_walking_person_could():
    worst, who, when = max_heading_step(110.0)
    assert worst <= 6.0, f"{who} turned {worst:.1f} deg in one tick at {when}s"


def test_every_actor_route_uses_the_actor_corner_radius():
    for name, route in ROUTES.items():
        assert route.radius == ACTOR_CORNER_RADIUS, name


def test_the_guardian_is_the_only_person_with_hold_windows():
    for name, route in ROUTES.items():
        if name == GUARDIAN.name:
            assert route.hold_windows
        else:
            assert not route.hold_windows, name


def test_the_cast_has_at_least_five_adults_besides_the_guardian():
    assert len(PEOPLE) - 1 >= 5


def test_stature_scales_the_geometry_rather_than_labelling_it():
    for person in PEOPLE:
        assert person.origin_z == pytest.approx(0.36 * person.stature)
        assert person.sample_dz[-1] == pytest.approx(0.34 * person.stature)


def test_everybody_is_somewhere_definite_at_every_instant():
    for t in (0.0, 20.0, 60.0, 109.0):
        people = people_at(t)
        assert set(people) == set(ALL_NAMES)
        for state in people.values():
            assert np.all(np.isfinite(state.pos))


# -- what occludes -----------------------------------------------------------
def test_occlusion_is_derived_from_height_not_declared():
    for obstacle in STATIC_OBSTACLES:
        assert obstacle.occludes == (obstacle.height_m >= OCCLUDING_HEIGHT_M)
    assert OCCLUDERS, "no occluders at all makes the LOS bookkeeping vacuous"


def test_the_cutaway_partitions_still_occlude_the_head_camera():
    """They are shortened for the spectator camera, not for the robot's."""
    from lobby_layout import CABIN_H, PARTITION_H
    assert PARTITION_H >= OCCLUDING_HEIGHT_M
    assert CABIN_H >= OCCLUDING_HEIGHT_M
    # And far above the topmost camera sample on an adult (0.36 + 0.34).
    assert min(PARTITION_H, CABIN_H) > 0.70
