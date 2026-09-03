#!/usr/bin/env python3
"""The route: constant speed, continuous heading, and the three pinned bends.

The filleted arc-length route replaces the smootherstep walker the sibling
behaviors use, for two measured reasons: a smootherstep actor STOPS at every
waypoint, and it turns through a whole corner in one control tick.  A formation
keeper graded against such an actor is being graded on the actor's stutter and
on a heading step no robot could follow.

The construction is pure geometry with no MuJoCo, no time-stepping and no state,
so every property is tested directly: total length, positional continuity,
TANGENT continuity across every boundary (that is what the fillet buys), bounded
curvature, and constant speed except during an explicit delay or terminal hold.

The last section pins the guardian's own route — three bends, two lefts and a
right — because a formation keeper that only ever turns one way has not been
tested on the sign its yaw controller is weakest on.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from beside_actors import (  # noqa: E402
    GUARDIAN_CORNERS,
    GUARDIAN_SPEED,
    ROUTES,
    max_heading_step,
    max_visible_jump,
    moving_fraction,
    people_at,
)
from beside_cast import ALL_NAMES  # noqa: E402
from beside_constants import VX_CRUISE, VX_SPRINT  # noqa: E402
from beside_route import CORNER_RADIUS, Route  # noqa: E402

SQUARE = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0))


# -- construction -------------------------------------------------------------

def test_a_route_needs_at_least_two_corners():
    with pytest.raises(ValueError, match="at least two corners"):
        Route("x", ((0.0, 0.0),), 0.2)


def test_a_straight_route_has_exactly_its_polyline_length():
    route = Route("x", ((0.0, 0.0), (3.0, 4.0)), 0.2)
    assert route.length == pytest.approx(5.0)
    assert route.corner_report() == []


def test_the_length_is_the_sum_of_its_pieces():
    route = Route("x", SQUARE, 0.2)
    assert route.length == pytest.approx(sum(p.length for p in route.pieces))


def test_a_fillet_is_shorter_than_the_corner_it_replaces():
    """Cutting the corner must actually cut it."""
    filleted = Route("x", SQUARE, 0.2)
    polyline = 4.0 + 4.0
    assert filleted.length < polyline
    # ... and the shortfall is bounded by the cutback geometry.
    cutback = CORNER_RADIUS * math.tan(math.radians(90.0) / 2.0)
    assert polyline - filleted.length < 2.0 * cutback


def test_a_nearly_straight_corner_is_left_as_a_plain_vertex():
    route = Route("x", ((0.0, 0.0), (4.0, 0.0), (8.0, 0.02)), 0.2)
    assert route.corner_report() == []


def test_a_corner_without_room_for_its_own_cutback_is_reported_as_missing():
    """Not silently accepted: it simply does not appear in the report."""
    tight = Route("x", ((0.0, 0.0), (0.30, 0.0), (0.30, 0.30)), 0.2,
                  radius=CORNER_RADIUS)
    assert tight.corner_report() == []
    roomy = Route("x", ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)), 0.2,
                  radius=CORNER_RADIUS)
    assert len(roomy.corner_report()) == 1


def test_a_route_that_collapses_to_nothing_is_refused():
    with pytest.raises(ValueError):
        Route("x", ((1.0, 1.0), (1.0, 1.0)), 0.2)


# -- continuity ---------------------------------------------------------------

def _sampled(route: Route, count: int = 4000):
    arcs = np.linspace(0.0, route.length, count)
    poses = [route.pose_at_arc(float(s)) for s in arcs]
    return arcs, [p for p, _ in poses], [t for _, t in poses]


def test_position_is_continuous_across_every_piece_boundary():
    route = Route("x", GUARDIAN_CORNERS, GUARDIAN_SPEED)
    arcs, points, _ = _sampled(route)
    step = float(arcs[1] - arcs[0])
    for index in range(1, len(points)):
        jump = float(np.linalg.norm(points[index] - points[index - 1]))
        assert jump <= step * 1.02 + 1e-9, (
            f"positional discontinuity of {jump:.5f} m at s={arcs[index]:.3f}")


def test_the_tangent_is_continuous_across_every_boundary():
    """This is exactly what the fillet buys, and what a cornered polyline
    cannot offer: the heading never steps."""
    route = Route("x", GUARDIAN_CORNERS, GUARDIAN_SPEED)
    arcs, _, tangents = _sampled(route)
    step = float(arcs[1] - arcs[0])
    limit = math.degrees(step / CORNER_RADIUS) * 1.5
    for index in range(1, len(tangents)):
        previous, current = tangents[index - 1], tangents[index]
        turn = abs(math.degrees(math.atan2(
            float(previous[0] * current[1] - previous[1] * current[0]),
            float(previous @ current))))
        assert turn <= limit, (
            f"heading stepped {turn:.3f} deg at s={arcs[index]:.3f}")


def test_every_tangent_is_a_unit_vector():
    route = Route("x", GUARDIAN_CORNERS, GUARDIAN_SPEED)
    _, _, tangents = _sampled(route, 600)
    for tangent in tangents:
        assert float(np.linalg.norm(tangent)) == pytest.approx(1.0, abs=1e-9)


def test_curvature_is_bounded_by_one_over_the_corner_radius():
    """Which is what makes 'the duck followed the bends' measurable rather than
    a hope: the turn rate the formation must track has a known ceiling."""
    route = Route("x", GUARDIAN_CORNERS, GUARDIAN_SPEED)
    arcs, _, tangents = _sampled(route)
    step = float(arcs[1] - arcs[0])
    worst = 0.0
    for index in range(1, len(tangents)):
        previous, current = tangents[index - 1], tangents[index]
        turn = abs(math.atan2(
            float(previous[0] * current[1] - previous[1] * current[0]),
            float(previous @ current)))
        worst = max(worst, turn / step)
    assert worst <= 1.0 / CORNER_RADIUS + 1e-6


def test_the_arc_parameterization_starts_and_ends_on_the_end_corners():
    route = Route("x", GUARDIAN_CORNERS, GUARDIAN_SPEED)
    assert route.pose_at_arc(0.0)[0] == pytest.approx(
        np.asarray(GUARDIAN_CORNERS[0]))
    assert route.pose_at_arc(route.length)[0] == pytest.approx(
        np.asarray(GUARDIAN_CORNERS[-1]))


def test_arc_length_is_clipped_rather_than_extrapolated():
    route = Route("x", GUARDIAN_CORNERS, GUARDIAN_SPEED)
    assert route.pose_at_arc(-5.0)[0] == pytest.approx(
        route.pose_at_arc(0.0)[0])
    assert route.pose_at_arc(route.length + 5.0)[0] == pytest.approx(
        route.pose_at_arc(route.length)[0])


# -- speed --------------------------------------------------------------------

def test_the_actor_walks_at_constant_speed_between_departure_and_arrival():
    route = Route("x", GUARDIAN_CORNERS, GUARDIAN_SPEED)
    dt = 0.02
    for index in range(1, int(route.finish_t() / dt)):
        t = index * dt
        if t <= dt or t >= route.finish_t() - dt:
            continue
        travelled = float(np.linalg.norm(
            route.pos_at(t) - route.pos_at(t - dt)))
        assert travelled == pytest.approx(GUARDIAN_SPEED * dt, rel=0.02)


def test_a_start_delay_is_a_real_standstill_and_the_end_is_a_real_hold():
    route = Route("x", ((0.0, 0.0), (2.0, 0.0)), 0.2, start_t=3.0)
    assert not route.moving(0.0)
    assert not route.moving(2.99)
    assert route.moving(3.5)
    assert route.pos_at(0.0) == pytest.approx(route.pos_at(2.99))
    finish = route.finish_t()
    assert not route.moving(finish + 0.5)
    assert route.pos_at(finish) == pytest.approx(route.pos_at(finish + 10.0))


def test_speed_at_reports_zero_exactly_when_moving_is_false():
    route = Route("x", ((0.0, 0.0), (2.0, 0.0)), 0.2, start_t=1.0)
    for t in np.arange(0.0, route.finish_t() + 2.0, 0.05):
        assert (route.speed_at(float(t)) > 0.0) == route.moving(float(t))


def test_yaw_at_is_the_path_tangent():
    route = Route("x", GUARDIAN_CORNERS, GUARDIAN_SPEED)
    for t in (2.0, 20.0, 40.0, 70.0):
        tangent = route.tangent_at(t)
        assert route.yaw_at(t) == pytest.approx(
            math.atan2(float(tangent[1]), float(tangent[0])))


# -- the actors ---------------------------------------------------------------

def test_no_actor_ever_teleports():
    jump, name, t = max_visible_jump(86.0)
    fastest = max(route.speed for route in ROUTES.values())
    assert jump <= fastest * 0.02 * 1.05, (
        f"{name} jumped {jump:.4f} m in one tick at t={t:.2f}s")


def test_no_actor_who_can_affect_a_decision_ever_steps_its_heading():
    """The property the smootherstep walker does NOT have.

    Graded on the guardian and the two oncoming walkers, because those are the
    bodies whose motion can reach a candidate slot and therefore enter a side
    decision.  ``rafa`` is excluded and held to a separate, explicit claim
    below: his loop contains a 168-deg hairpin whose fillet does not fit, which
    the route builder drops rather than silently accepting.
    """
    dt = 0.02
    fastest = max(ROUTES[name].speed for name in ("nadia", "tomas", "iris"))
    limit = math.degrees(fastest * dt / CORNER_RADIUS) * 1.6

    previous = {name: state.yaw for name, state in people_at(0.0).items()}
    for index in range(1, int(86.0 / dt) + 1):
        t = index * dt
        for name, state in people_at(t).items():
            if name in ("rafa", "lena"):
                continue
            delta = abs(math.degrees(math.atan2(
                math.sin(state.yaw - previous[name]),
                math.cos(state.yaw - previous[name]))))
            assert delta <= limit, (
                f"{name} turned {delta:.3f} deg in one tick at t={t:.2f}s, "
                f"limit {limit:.3f} deg")
            previous[name] = state.yaw


def test_the_one_dropped_hairpin_belongs_to_a_walker_who_decides_nothing():
    """``rafa``'s loop doubles back through 168 deg.

    A 0.90 m fillet there would need an 8.6 m cutback and the legs are 1.8 m,
    so ``_build`` leaves it as a plain vertex and his heading DOES step once,
    at t = 23.80 s.  That is acceptable only because he can never enter a side
    decision, and this test is what keeps that conditional honest: it measures
    his closest approach to either candidate slot over the whole rollout and
    requires it to stay far outside the pedestrian margin.  Move his route near
    the guardian and this fails, as it should.
    """
    from beside_geometry import SIDE_PERSON_MARGIN_M, slot_point

    assert len(ROUTES["rafa"].corner_report()) == 2, (
        "rafa has three interior corners and one is dropped for want of room")

    closest = float("inf")
    for index in range(0, int(86.0 / 0.02) + 1, 2):
        t = index * 0.02
        people = people_at(t)
        guardian = people["nadia"]
        for side in (1, -1):
            slot = slot_point(guardian.pos, guardian.yaw, side)
            closest = min(closest, float(np.linalg.norm(
                people["rafa"].pos - slot)))
    assert closest > 4.0 * SIDE_PERSON_MARGIN_M, (
        f"rafa came within {closest:.3f} m of a candidate slot; his dropped "
        "fillet can now reach a side decision and the route needs a real bend")


def test_max_heading_step_reports_the_worst_offender_and_when():
    """The helper the two tests above rely on actually finds the worst tick."""
    step, name, t = max_heading_step(86.0)
    assert name == "rafa", "the dropped hairpin is the worst step in the scene"
    assert step > 90.0
    assert 23.0 < t < 24.5


def test_the_guardian_walks_for_the_whole_rollout_rather_than_stopping():
    """A companion graded against somebody who stops is graded on her stutter."""
    fractions = moving_fraction(86.0)
    assert fractions["nadia"] == pytest.approx(1.0, abs=0.02)


def test_every_actor_has_a_route_and_every_route_has_an_actor():
    assert set(ROUTES) == set(ALL_NAMES)


def test_people_at_returns_a_velocity_consistent_with_its_own_yaw_and_speed():
    for t in (0.0, 5.0, 33.0, 70.0):
        for name, state in people_at(t).items():
            expected = np.array([math.cos(state.yaw), math.sin(state.yaw)]) \
                * state.speed
            assert state.velocity == pytest.approx(expected, abs=1e-12)


def test_a_stationary_actor_reports_a_zero_velocity_not_a_stale_heading():
    route = ROUTES["nadia"]
    late = route.finish_t() + 5.0
    state = people_at(late)["nadia"]
    assert state.speed == 0.0
    assert state.velocity == pytest.approx(np.zeros(2))


# -- the guardian's route, pinned ---------------------------------------------

def test_the_guardian_route_has_exactly_three_bends_two_left_and_one_right():
    """PINNED.  A formation keeper that only ever turns one way has not been
    tested on the sign its yaw controller is weakest on."""
    bends = ROUTES["nadia"].corner_report()
    assert len(bends) == 3
    assert [bend["hand"] for bend in bends] == ["left", "right", "left"]
    assert {bend["hand"] for bend in bends} == {"left", "right"}


def test_the_pinned_bend_angles_and_windows_match_the_reference_run():
    """PINNED against the 86 s metrics: the bend gate grades records inside
    these windows, so a route edit that moves them invalidates that gate."""
    bends = ROUTES["nadia"].corner_report()
    expected = [
        (42.879, "left", 35.14, 40.848),
        (-42.879, "right", 51.049, 56.757),
        (81.254, "left", 66.286, 77.103),
    ]
    for bend, (turn, hand, start, end) in zip(bends, expected):
        assert bend["turn_deg"] == pytest.approx(turn, abs=1e-3)
        assert bend["hand"] == hand
        assert bend["start_t_s"] == pytest.approx(start, abs=1e-2)
        assert bend["end_t_s"] == pytest.approx(end, abs=1e-2)
        assert bend["radius_m"] == CORNER_RADIUS


def test_every_bend_lasts_long_enough_for_the_formation_gate_to_grade_it():
    """The bend gate needs at least 1.0 s of BESIDE inside each window."""
    for bend in ROUTES["nadia"].corner_report():
        assert bend["end_t_s"] - bend["start_t_s"] >= 1.0


def test_the_bends_do_not_overlap_and_are_ordered_in_time():
    bends = ROUTES["nadia"].corner_report()
    for previous, current in zip(bends, bends[1:]):
        assert previous["end_t_s"] < current["start_t_s"]


def test_the_route_is_finished_no_earlier_than_the_rollout_it_is_graded_over():
    """A guardian who arrives and stops mid-rollout would leave the duck holding
    station beside somebody standing still, which is not this behavior."""
    assert ROUTES["nadia"].finish_t() >= 86.0


def test_the_guardian_walks_slower_than_the_duck_can_and_faster_than_it_crawls():
    """She must be catchable and also worth keeping up with."""
    from beside_constants import SPEED_AT_CRUISE, SPEED_AT_SPRINT

    assert SPEED_AT_CRUISE < GUARDIAN_SPEED + 0.02
    assert GUARDIAN_SPEED < SPEED_AT_SPRINT, (
        "the duck must be able to recover station after the crossing")
    assert VX_CRUISE < VX_SPRINT


def test_the_route_record_reports_everything_the_metrics_quote():
    record = ROUTES["nadia"].as_record()
    assert record["name"] == "nadia"
    assert record["length_m"] == pytest.approx(10.9566, abs=1e-3)
    assert record["speed_mps"] == GUARDIAN_SPEED
    assert record["corner_radius_m"] == CORNER_RADIUS
    assert len(record["bends"]) == 3
    assert len(record["corners"]) == len(GUARDIAN_CORNERS)
