#!/usr/bin/env python3
"""The cast, the plaza and the encounter schedule, pinned as they stand.

Nothing here runs physics.  These are the declarations the whole behavior is
built on, and each test states the SCENARIO REQUIREMENT it protects rather than
restating the constant, so a change that quietly weakens the scenario - one
fewer intruder, a false alarm who also intrudes, a ward dressed as the brightest
person in the plaza - fails with the reason attached.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pps_cast import (ALL_NAMES, BASE_HEIGHT_M, BASE_ORIGIN_Z, BASE_SAMPLE_DZ,
                      BY_NAME, INTRUDERS, PEOPLE, PLANNING_HALF_EXTENT_M,
                      WARD, is_ward, role_of)
from pps_plaza import (BY_FIXTURE, DUCK_START, DUCK_START_YAW_DEG, FIXTURES,
                       FLOOR_HALF, OCCLUDERS, OCCLUDING_HEIGHT_M, clamp_inside,
                       inside_area, nearest_fixture, occluder_between,
                       static_gap, wall_gap)
from pps_script import (ENCOUNTERS, EXPECTED_EPISODES, INTRUDER_ROUTES, ROUTES,
                        SESSION_S, WARD_RADIUS, WARD_ROUTE, encounters_of,
                        route_of, session_end_s)
from pps_states import DUCK_PLANAR_RADIUS

# The cast, in the order the behavior declares it.  Pinned so a reordering -
# which would silently change the gait phase offsets in ``pose_bodies`` and the
# tie-break order in ``predict_all`` - fails here rather than in a video.
EXPECTED_CAST = ("aina", "dario", "noor", "piet", "yara", "kwame", "liesl",
                 "tomas")


# -- who is in the plaza -----------------------------------------------------
def test_cast_order_is_pinned():
    assert ALL_NAMES == EXPECTED_CAST
    assert tuple(p.name for p in PEOPLE) == EXPECTED_CAST


def test_exactly_one_ward_and_seven_intruders():
    """Seven is the smallest cast that keeps the roles separate.

    Four distinct intrusion cycles, one false alarm who must NOT also intrude,
    two squeeze participants not already committed to a cycle, and one person
    who is simply crossing.  Six would force one person into two roles and turn
    the dismissal into a delayed detection.
    """
    wards = [p for p in PEOPLE if p.role == "ward"]
    assert len(wards) == 1
    assert wards[0].name == WARD == "aina"
    assert len(INTRUDERS) == 7
    assert set(INTRUDERS) | {WARD} == set(ALL_NAMES)


def test_no_duplicate_names():
    assert len(set(ALL_NAMES)) == len(ALL_NAMES)


@pytest.mark.parametrize("name", EXPECTED_CAST)
def test_role_lookup_agrees_with_the_dataclass(name):
    assert role_of(name) == BY_NAME[name].role
    assert is_ward(name) == (BY_NAME[name].role == "ward")
    assert BY_NAME[name].role in ("ward", "intruder")


def test_only_the_ward_answers_is_ward():
    assert [n for n in ALL_NAMES if is_ward(n)] == [WARD]


def test_the_ward_is_not_the_most_conspicuous_person():
    """A grey coat, deliberately.

    Identity is resolved by the acquisition layer against a body-identity proxy.
    If the ward were the brightest person in frame, "the duck kept its own
    person" would be smuggling the answer in as a colour.  Saturation is the
    honest measure of conspicuousness: her shirt is nearly neutral and at least
    four intruders are far more saturated.
    """
    def saturation(shirt):
        return max(shirt) - min(shirt)

    ward_saturation = saturation(BY_NAME[WARD].shirt)
    assert ward_saturation < 0.10, ward_saturation
    louder = [n for n in INTRUDERS if saturation(BY_NAME[n].shirt) > 0.40]
    assert len(louder) >= 4, louder


@pytest.mark.parametrize("name", EXPECTED_CAST)
def test_every_person_is_a_plausible_adult(name):
    person = BY_NAME[name]
    assert 0.95 <= person.stature <= 1.06
    assert 1.60 <= person.height_m <= 1.85
    assert person.height_m == pytest.approx(BASE_HEIGHT_M * person.stature)
    assert person.origin_z == pytest.approx(BASE_ORIGIN_Z * person.stature)


@pytest.mark.parametrize("name", EXPECTED_CAST)
def test_sample_points_scale_with_stature_and_stay_ordered(name):
    """Five body samples, knees to crown, scaled by the person's own stature."""
    person = BY_NAME[name]
    assert len(person.sample_dz) == len(BASE_SAMPLE_DZ) == 5
    assert list(person.sample_dz) == sorted(person.sample_dz)
    for scaled, base in zip(person.sample_dz, BASE_SAMPLE_DZ):
        assert scaled == pytest.approx(base * person.stature)


@pytest.mark.parametrize("name", EXPECTED_CAST)
def test_rgba_renders_the_declared_shirt(name):
    person = BY_NAME[name]
    channels = [float(v) for v in person.rgba.split()]
    assert channels[:3] == pytest.approx(list(person.shirt), abs=5e-4)
    assert channels[3] == 1.0


def test_every_person_carries_a_stated_role_note():
    """Each label says which scenario role the person plays.

    A cast member with no stated purpose is a person the scenario cannot
    account for, and this behavior's whole claim is that the roles are separate.
    """
    for person in PEOPLE:
        assert len(person.label) > 20, person.name


def test_planning_half_extent_is_generous_but_not_a_measurement():
    """A planning figure only.  Clearance is measured against real geoms."""
    assert PLANNING_HALF_EXTENT_M == 0.30
    assert PLANNING_HALF_EXTENT_M > DUCK_PLANAR_RADIUS


# -- the plaza ---------------------------------------------------------------
def test_floor_is_big_enough_for_the_declared_routes():
    """Every scripted corner has to be ON the floor, or a walker leaves it."""
    for name, route in ROUTES.items():
        for x, y in route.corners:
            assert abs(x) <= FLOOR_HALF[0], (name, x)
            assert abs(y) <= FLOOR_HALF[1], (name, y)


def test_exactly_two_fixtures_are_real_occluders():
    """The visibility exclusion is conditioned on something that happens.

    Two fixtures stand above head-camera height so a person behind one is
    genuinely hidden; the rest bound the plaza without screening anything.
    """
    assert len(FIXTURES) == 7
    assert tuple(f.name for f in OCCLUDERS) == ("obs_kiosk_e", "obs_lamp_nw")
    for fixture in OCCLUDERS:
        assert fixture.height_m >= OCCLUDING_HEIGHT_M
    for fixture in FIXTURES:
        if fixture not in OCCLUDERS:
            assert fixture.height_m < OCCLUDING_HEIGHT_M, fixture.name


def test_the_middle_of_the_plaza_is_open():
    """A cluttered floor would choose the duck's station for it.

    Every fixture sits away from the centre, so an interpose position is the one
    the duck picked rather than the only one available.
    """
    for fixture in FIXTURES:
        assert float(np.linalg.norm(np.asarray(fixture.center))) >= 1.0, (
            fixture.name)


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f.name for f in FIXTURES])
def test_fixture_distance_is_negative_inside_and_zero_on_the_surface(fixture):
    assert fixture.distance_to(fixture.center) < 0.0
    if fixture.kind == "cylinder":
        edge = (fixture.center[0] + fixture.radius, fixture.center[1])
        assert fixture.distance_to(edge) == pytest.approx(0.0, abs=1e-9)
    else:
        edge = (fixture.center[0] + fixture.half[0], fixture.center[1])
        assert fixture.distance_to(edge) == pytest.approx(0.0, abs=1e-9)
    far = (fixture.center[0] + 5.0, fixture.center[1])
    assert fixture.distance_to(far) > 4.0


def test_segment_hits_finds_a_crossing_and_misses_a_clear_line():
    kiosk = BY_FIXTURE["obs_kiosk_e"]
    assert kiosk.segment_hits((2.5, 0.72), (4.0, 0.72))
    assert not kiosk.segment_hits((0.0, -2.0), (0.0, 2.0))


def test_occluder_between_names_only_a_full_height_fixture():
    """The planter is 0.44 m: it is in the way and it still does not occlude."""
    assert occluder_between((2.5, 0.72), (4.0, 0.72)) == "obs_kiosk_e"
    assert occluder_between((-1.92, 0.5), (-1.92, 3.0)) == "obs_lamp_nw"
    assert occluder_between((-2.5, -1.42), (0.5, -1.42)) is None
    assert occluder_between((0.0, -2.0), (0.0, 2.0)) is None


def test_wall_gap_and_inside_area_agree_at_the_boundary():
    assert wall_gap((0.0, 0.0)) == pytest.approx(min(FLOOR_HALF))
    assert wall_gap((FLOOR_HALF[0], 0.0)) == pytest.approx(0.0)
    assert wall_gap((FLOOR_HALF[0] + 0.5, 0.0)) < 0.0
    assert inside_area((0.0, 0.0))
    assert inside_area((FLOOR_HALF[0] - 0.01, 0.0))
    assert not inside_area((FLOOR_HALF[0] + 0.01, 0.0))
    assert not inside_area((FLOOR_HALF[0] - 0.05, 0.0), margin=0.10)


def test_static_gap_prefers_whichever_surface_is_nearer():
    """Near a wall it reports the wall; near a fixture it names the fixture."""
    name, gap = static_gap((FLOOR_HALF[0] - 0.05, 0.0))
    assert name == "wall" and gap == pytest.approx(0.05)
    name, gap = static_gap((-1.06, -1.42))
    assert name == "obs_planter_s" and gap < 0.0
    assert nearest_fixture((-1.06, -1.42))[0] == "obs_planter_s"


@pytest.mark.parametrize("point", [(9.0, 0.0), (0.0, -9.0), (-9.0, 9.0)])
def test_clamp_inside_returns_a_point_the_area_accepts(point):
    margin = DUCK_PLANAR_RADIUS + 0.10
    clamped = clamp_inside(point, margin)
    assert inside_area(clamped, margin - 1e-9)
    assert clamped.shape == (2,)


def test_clamp_inside_leaves_an_interior_point_untouched():
    inside = np.array([0.4, -0.3])
    assert clamp_inside(inside, 0.2) == pytest.approx(inside)


def test_duck_starts_off_the_slot_so_joining_is_a_manoeuvre():
    """Behind and right of the ward's own start.

    If the duck began in the escort slot, "the escort was established" would be
    the initial condition rather than something the controller achieved.
    """
    from pps_geometry import escort_point
    start = np.asarray(DUCK_START)
    ward_start = WARD_ROUTE.pos_at(0.0)
    slot = escort_point(ward_start, WARD_ROUTE.yaw_at(0.0))
    assert float(np.linalg.norm(start - slot)) > 0.5
    assert inside_area(start, DUCK_PLANAR_RADIUS)
    assert DUCK_START_YAW_DEG == 100.0


# -- the encounter schedule --------------------------------------------------
def test_the_scenario_asks_for_what_the_behavior_claims():
    kinds = [e.kind for e in ENCOUNTERS]
    assert kinds.count("intrusion") >= 4
    assert kinds.count("false_alarm") == 1
    assert kinds.count("squeeze") == 1
    assert kinds.count("ward_approach") == 1
    assert len(ENCOUNTERS) == 7


def test_encounters_are_chronological_and_bounded():
    for earlier, later in zip(ENCOUNTERS, ENCOUNTERS[1:]):
        assert earlier.from_s <= later.from_s, (earlier.kind, later.kind)
    for encounter in ENCOUNTERS:
        assert encounter.from_s < encounter.to_s <= SESSION_S


def test_the_false_alarm_never_goes_on_to_intrude():
    """The separation the cast size exists for.

    An adult used for the false alarm who later intruded would turn the
    dismissal into a delayed detection, which is the opposite claim.
    """
    false_alarm = encounters_of("false_alarm")
    assert len(false_alarm) == 1
    piet = set(false_alarm[0].people)
    assert piet == {"piet"}
    for encounter in ENCOUNTERS:
        if encounter.kind != "false_alarm":
            assert piet.isdisjoint(encounter.people), encounter.kind


def test_the_four_intrusion_cycles_are_four_different_people():
    people = [e.people[0] for e in encounters_of("intrusion")]
    assert len(people) == 4
    assert len(set(people)) == 4
    assert set(people) == {"dario", "noor", "yara", "kwame"}


def test_intrusion_bearings_alternate_sides_in_the_schedule():
    """Alternating bearings is what makes four cycles four MANOEUVRES.

    Four intrusions from the same side would be one station held four times.
    """
    sides = [1 if math.cos(math.radians(e.bearing_deg)) >= 0 else -1
             for e in encounters_of("intrusion")]
    assert all(a != b for a, b in zip(sides, sides[1:])), sides


def test_the_squeeze_has_two_people_from_nearly_opposite_sides():
    squeeze = encounters_of("squeeze")
    assert len(squeeze) == 1
    assert len(squeeze[0].people) == 2
    assert set(squeeze[0].people) == {"liesl", "tomas"}


def test_expected_episodes_matches_the_encounter_kinds_it_grades():
    """The required response sequence, minus the one that must NOT respond."""
    assert EXPECTED_EPISODES == ("intrusion", "intrusion", "intrusion",
                                 "ward_approach", "intrusion", "squeeze")
    assert "false_alarm" not in EXPECTED_EPISODES
    responded = [e.kind for e in ENCOUNTERS if e.kind != "false_alarm"]
    assert sorted(EXPECTED_EPISODES) == sorted(responded)


def test_encounters_of_filters_and_returns_a_tuple():
    assert isinstance(encounters_of("intrusion"), tuple)
    assert encounters_of("nonexistent") == ()
    for kind in ("intrusion", "false_alarm", "squeeze", "ward_approach"):
        assert all(e.kind == kind for e in encounters_of(kind))


def test_session_helpers_agree_with_the_declared_length():
    assert session_end_s() == SESSION_S == 190.0
    for name in ALL_NAMES:
        assert route_of(name) is ROUTES[name]


def test_every_person_has_a_route_and_every_route_a_person():
    assert set(ROUTES) == set(ALL_NAMES)
    assert set(INTRUDER_ROUTES) == set(INTRUDERS)
    assert WARD not in INTRUDER_ROUTES
    for name, route in ROUTES.items():
        assert route.name == name


def test_the_wards_corner_radius_is_tighter_than_the_strangers():
    """Her dogleg is a real about-turn; theirs are walking-pace bends."""
    from pps_route import CORNER_RADIUS
    assert WARD_RADIUS == 0.12 < CORNER_RADIUS == 0.40
    assert WARD_ROUTE.radius == WARD_RADIUS
    for route in INTRUDER_ROUTES.values():
        assert route.radius == CORNER_RADIUS
