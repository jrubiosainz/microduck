#!/usr/bin/env python3
"""Route geometry: the properties a fillet buys and a polyline does not.

A cornered polyline turns its walker through a whole corner in ONE control
tick, which is a teleport of the body axis.  ``pps_route`` replaces every
interior corner with a tangent circular arc and parameterizes by arc length, so
position AND heading are continuous everywhere and speed is constant except
during an explicit delay or hold.

Every property below is checked on the SHIPPED routes rather than on a toy,
because a synthetic route proves the construction and the shipped ones prove the
choreography actually uses it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pps_actors import (bodies_at, max_heading_step, moving_fraction,
                        route_records)
from pps_cast import ALL_NAMES, BY_NAME, WARD
from pps_route import CORNER_RADIUS, Route, _build, _unit
from pps_script import ROUTES, SESSION_S, WARD_ROUTE

# The bends actually built into each shipped route, pinned.  A corner that
# stopped being filleted, or a route that gained one, changes these counts.
EXPECTED_BENDS = {"aina": 4, "dario": 1, "noor": 1, "piet": 0, "yara": 1,
                  "kwame": 1, "liesl": 1, "tomas": 4}
# Total arc length of each shipped route, in metres.
EXPECTED_LENGTH_M = {"aina": 4.0917, "dario": 3.1415, "noor": 3.8226,
                     "piet": 4.0376, "yara": 4.3468, "kwame": 5.3604,
                     "liesl": 4.2094, "tomas": 13.4196}
NAMES = list(ALL_NAMES)


# -- the construction --------------------------------------------------------
def test_unit_refuses_a_zero_length_direction():
    """A zero-length segment has no direction, and guessing one hides a bug."""
    with pytest.raises(ValueError):
        _unit(np.zeros(2))
    assert _unit(np.array([3.0, 4.0])) == pytest.approx([0.6, 0.8])


def test_a_corner_that_cannot_be_filleted_raises_and_names_itself():
    """Skipping would leave a hard vertex and surface far away as a teleport.

    The message has to name the corner and the radius that WOULD fit, so the
    fix happens where the geometry is rather than where the symptom shows up.
    """
    with pytest.raises(ValueError) as excinfo:
        Route(name="impossible", corners=((0.0, 0.0), (0.1, 0.0), (0.1, 0.5)),
              speed=0.1, radius=0.40)
    message = str(excinfo.value)
    assert "impossible" in message
    assert "cutback" in message
    assert "hard vertex" in message
    assert "Use a radius below" in message


def test_a_route_needs_at_least_two_corners():
    with pytest.raises(ValueError):
        Route(name="stub", corners=((0.0, 0.0),), speed=0.1)


def test_a_hold_window_that_ends_before_it_starts_is_refused():
    with pytest.raises(ValueError):
        Route(name="backwards", corners=((0.0, 0.0), (1.0, 0.0)), speed=0.1,
              hold_windows=((5.0, 3.0),))


def test_a_nearly_straight_corner_is_skipped_rather_than_filleted():
    """There is no turn to fillet, so no arc is built and none is expected."""
    straight = Route(name="straight",
                     corners=((0.0, 0.0), (1.0, 0.0), (2.0, 0.001)),
                     speed=0.1)
    assert straight.corner_report() == []
    assert straight.length == pytest.approx(2.0, abs=1e-3)


def test_build_returns_line_and_arc_pieces_in_arc_length_order():
    pieces = _build([np.array([0.0, 0.0]), np.array([2.0, 0.0]),
                     np.array([2.0, 2.0])], 0.40, "L")
    assert [p.kind for p in pieces] == ["line", "arc", "line"]
    cursor = 0.0
    for piece in pieces:
        assert piece.start_s == pytest.approx(cursor)
        cursor += piece.length


# -- the shipped routes ------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_shipped_route_length_is_pinned(name):
    assert ROUTES[name].length == pytest.approx(EXPECTED_LENGTH_M[name],
                                                abs=1e-4)


@pytest.mark.parametrize("name", NAMES)
def test_shipped_bend_count_is_pinned(name):
    report = ROUTES[name].corner_report()
    assert len(report) == EXPECTED_BENDS[name]
    for bend in report:
        assert bend["hand"] in ("left", "right")
        assert bend["hand"] == ("left" if bend["turn_deg"] > 0 else "right")
        assert bend["radius_m"] == pytest.approx(ROUTES[name].radius)
        assert bend["arc_len_m"] > 0.0
        assert bend["end_s_m"] > bend["start_s_m"]


@pytest.mark.parametrize("name", NAMES)
def test_total_length_equals_the_sum_of_its_pieces(name):
    route = ROUTES[name]
    assert route.length == pytest.approx(sum(p.length for p in route.pieces))


@pytest.mark.parametrize("name", NAMES)
def test_position_is_continuous_across_every_piece_boundary(name):
    """The property a fillet buys: no teleport at a corner."""
    route = ROUTES[name]
    for piece in route.pieces[1:]:
        before = route.pose_at_arc(piece.start_s - 1e-6)[0]
        after = route.pose_at_arc(piece.start_s + 1e-6)[0]
        assert float(np.linalg.norm(after - before)) < 1e-4, piece.kind


@pytest.mark.parametrize("name", NAMES)
def test_tangent_is_continuous_across_every_piece_boundary(name):
    """And the one a cornered polyline does NOT: no heading discontinuity."""
    route = ROUTES[name]
    for piece in route.pieces[1:]:
        before = route.pose_at_arc(piece.start_s - 1e-6)[1]
        after = route.pose_at_arc(piece.start_s + 1e-6)[1]
        assert float(before @ after) > 0.999, piece.kind


@pytest.mark.parametrize("name", NAMES)
def test_the_tangent_is_a_unit_vector_everywhere(name):
    route = ROUTES[name]
    for s in np.linspace(0.0, route.length, 97):
        tangent = route.pose_at_arc(float(s))[1]
        assert float(np.linalg.norm(tangent)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_curvature_is_bounded_by_the_declared_corner_radius(name):
    """Speed times curvature is the yaw rate the walker must physically turn at."""
    route = ROUTES[name]
    for piece in route.pieces:
        if piece.kind == "arc":
            assert piece.radius == pytest.approx(route.radius)


@pytest.mark.parametrize("name", NAMES)
def test_arc_length_is_clipped_to_the_route_at_both_ends(name):
    route = ROUTES[name]
    assert route.arc_at(-10.0) == 0.0
    assert route.arc_at(route.start_t - 1e-6) == pytest.approx(0.0)
    assert route.arc_at(1e6) == pytest.approx(route.length)


@pytest.mark.parametrize("name", NAMES)
def test_a_walker_stands_still_before_departure(name):
    route = ROUTES[name]
    if route.start_t <= 0.0:
        pytest.skip("departs immediately")
    early = route.pos_at(route.start_t - 1.0)
    assert early == pytest.approx(route.pos_at(0.0))
    assert route.speed_at(route.start_t - 1.0) == 0.0
    assert not route.moving(route.start_t - 1.0)


@pytest.mark.parametrize("name", NAMES)
def test_a_walker_holds_at_the_far_end_rather_than_wrapping(name):
    route = ROUTES[name]
    finish = route.finish_t()
    assert route.pos_at(finish + 5.0) == pytest.approx(route.pos_at(finish),
                                                       abs=1e-6)
    assert route.speed_at(finish + 5.0) == 0.0


@pytest.mark.parametrize(
    "name", [n for n in NAMES if ROUTES[n].hold_windows])
def test_position_is_continuous_across_both_edges_of_every_hold(name):
    """A hold freezes arc length; it must not move the body.

    Arc length is ``speed * (elapsed - held)`` in closed form, so a hold is a
    pure function of ``t`` with no accumulated state - and both edges of every
    window are therefore continuous rather than a step.
    """
    route = ROUTES[name]
    for start, end in route.hold_windows:
        for edge in (start, end):
            before = route.pos_at(edge - 1e-4)
            after = route.pos_at(edge + 1e-4)
            assert float(np.linalg.norm(after - before)) < 1e-4, (name, edge)


@pytest.mark.parametrize(
    "name", [n for n in NAMES if ROUTES[n].hold_windows])
def test_arc_length_does_not_advance_during_a_hold(name):
    route = ROUTES[name]
    for start, end in route.hold_windows:
        assert route.arc_at(end - 1e-3) == pytest.approx(
            route.arc_at(start + 1e-3), abs=1e-3)
        assert route.speed_at((start + end) * 0.5) == 0.0
        assert not route.moving((start + end) * 0.5)


@pytest.mark.parametrize("name", NAMES)
def test_speed_is_constant_whenever_the_walker_is_actually_walking(name):
    """No smootherstep: a walker is either at its declared pace or stopped."""
    route = ROUTES[name]
    observed = {route.speed_at(float(t))
                for t in np.arange(0.0, SESSION_S, 0.25)}
    assert observed <= {0.0, route.speed}, sorted(observed)
    assert route.speed in observed


@pytest.mark.parametrize("name", NAMES)
def test_finish_time_accounts_for_every_hold(name):
    route = ROUTES[name]
    held = sum(end - start for start, end in route.hold_windows)
    assert route.finish_t() == pytest.approx(
        route.start_t + held + route.length / route.speed)


@pytest.mark.parametrize("name", NAMES)
def test_measured_ground_speed_matches_the_declared_speed(name):
    """Differentiated from real positions, not read back from the field."""
    route = ROUTES[name]
    dt = 0.02
    for t in np.arange(route.start_t + 0.5, route.finish_t() - 0.5, 3.7):
        t = float(t)
        if not route.moving(t) or not route.moving(t + dt):
            continue
        step = float(np.linalg.norm(route.pos_at(t + dt) - route.pos_at(t)))
        assert step / dt == pytest.approx(route.speed, rel=0.02)


@pytest.mark.parametrize("name", NAMES)
def test_yaw_agrees_with_the_direction_the_walker_is_travelling(name):
    """Heading is the path tangent, so it has to match measured motion."""
    route = ROUTES[name]
    dt = 0.05
    for t in np.arange(route.start_t + 0.4, route.finish_t() - 0.6, 4.3):
        t = float(t)
        if not (route.moving(t) and route.moving(t + dt)):
            continue
        step = route.pos_at(t + dt) - route.pos_at(t)
        travelled = math.atan2(float(step[1]), float(step[0]))
        error = abs(math.degrees(
            math.atan2(math.sin(route.yaw_at(t) - travelled),
                       math.cos(route.yaw_at(t) - travelled))))
        assert error < 4.0, (name, t, error)


def test_the_wards_dogleg_is_a_real_about_turn():
    """She turns out of her line, walks back at the slot, and rejoins.

    Two bends of more than 120 deg with opposite hands is what an about-turn
    is; a route that lost the dogleg would keep the corner count and lose this.
    """
    big = [b for b in WARD_ROUTE.corner_report() if abs(b["turn_deg"]) > 120.0]
    assert len(big) == 2, WARD_ROUTE.corner_report()
    assert {b["hand"] for b in big} == {"left", "right"}


def test_the_ward_holds_through_every_encounter():
    """Her stillness is what makes the duck's station-keeping the duck's.

    Five hold windows, one per encounter, and she walks between them - so the
    escort is a moving formation and the geometry during encounters is not
    partly her drifting into the gap.
    """
    assert len(WARD_ROUTE.hold_windows) == 5
    for start, end in WARD_ROUTE.hold_windows:
        assert end - start >= 7.0
    for (_, end), (later, _) in zip(WARD_ROUTE.hold_windows,
                                    WARD_ROUTE.hold_windows[1:]):
        assert later > end, "hold windows must not overlap"


def test_the_false_alarm_never_stops():
    """He walks past and keeps going, which is what makes him a false alarm.

    An adult who stopped near the ward would be an intrusion whatever the
    schedule called him, so the absence of a hold window is load-bearing.
    """
    piet = ROUTES["piet"]
    assert piet.hold_windows == ()
    assert piet.corner_report() == [], "a straight near pass, with no bend"
    assert piet.speed == max(r.speed for r in ROUTES.values())


def test_only_the_intruders_who_stop_carry_a_hold_window():
    """A person who walks up to somebody and away has to stop in between."""
    holding = {n for n in ALL_NAMES if ROUTES[n].hold_windows}
    assert holding == {"aina", "dario", "noor", "yara", "kwame", "liesl",
                       "tomas"}
    assert "piet" not in holding


# -- the world the actors present each tick ----------------------------------
def test_no_scripted_person_ever_teleports_their_heading():
    """The whole reason the routes are filleted, measured over the session.

    A cornered polyline turns a walker through a whole corner in one control
    tick.  The largest single-tick heading change any walking person makes is
    well under a degree, which is a walk rather than a snap.
    """
    worst, name, when = max_heading_step(SESSION_S)
    assert worst < 1.0, f"{name} turned {worst:.2f} deg in one tick at {when}s"


def test_everybody_spends_real_time_walking_and_real_time_stopped():
    """A plaza of statues is not a populated plaza, and neither is a treadmill."""
    fractions = moving_fraction(SESSION_S)
    assert set(fractions) == set(ALL_NAMES)
    for name, fraction in fractions.items():
        assert 0.10 <= fraction <= 0.85, (name, fraction)


def test_the_person_who_is_simply_there_walks_the_most():
    """Tomas crosses the whole plaza before turning in for the squeeze."""
    fractions = moving_fraction(SESSION_S)
    assert max(fractions, key=fractions.get) == "tomas"


@pytest.mark.parametrize("t", [0.0, 12.0, 45.5, 83.0, 104.0, 140.0, 190.0])
def test_bodies_at_reports_every_person_in_cast_order(t):
    states = bodies_at(t)
    assert list(states) == list(ALL_NAMES)
    for name, state in states.items():
        assert state.name == name
        assert state.pos.shape == (2,)
        assert state.velocity.shape == (2,)
        assert np.all(np.isfinite(state.pos))
        assert state.present == (t >= ROUTES[name].start_t)


@pytest.mark.parametrize("t", [5.0, 30.0, 70.0, 120.0, 165.0])
def test_velocity_agrees_with_speed_and_heading(t):
    """Velocity is heading times speed, and exactly zero when stopped."""
    for name, state in bodies_at(t).items():
        assert float(np.linalg.norm(state.velocity)) == pytest.approx(
            state.speed, abs=1e-9)
        if state.speed > 0.0:
            expected = np.array([math.cos(state.yaw), math.sin(state.yaw)])
            assert state.velocity == pytest.approx(expected * state.speed)
        else:
            assert state.velocity == pytest.approx(np.zeros(2))


def test_a_person_who_has_not_departed_is_marked_absent():
    """``predict_all`` skips absent people, so this flag gates prediction."""
    states = bodies_at(0.0)
    assert not states["liesl"].present, "liesl departs at 115 s"
    assert states["aina"].present, "the ward is there from the start"
    assert bodies_at(120.0)["liesl"].present


def test_route_records_publish_every_route_with_its_role():
    records = route_records()
    assert [r["name"] for r in records] == list(ALL_NAMES)
    for record in records:
        assert record["role"] == BY_NAME[record["name"]].role
        assert record["length_m"] == pytest.approx(
            EXPECTED_LENGTH_M[record["name"]], abs=1e-4)
        assert len(record["bends"]) == EXPECTED_BENDS[record["name"]]
        assert record["corner_radius_m"] == ROUTES[record["name"]].radius
        assert record["finish_t_s"] == pytest.approx(
            ROUTES[record["name"]].finish_t(), abs=1e-3)


def test_the_ward_is_the_slowest_person_in_the_plaza():
    """Every stranger overtakes her, which is what makes them approaches."""
    assert ROUTES[WARD].speed == min(r.speed for r in ROUTES.values())
    for name in ALL_NAMES:
        if name != WARD:
            assert ROUTES[name].speed > ROUTES[WARD].speed, name
