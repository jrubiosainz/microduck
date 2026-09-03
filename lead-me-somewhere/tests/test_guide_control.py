"""The controller, the route tracker and the follower: pure logic, no MuJoCo.

The three modules that decide what the duck commands, where it is on its own
route, and how the person behind it moves.  All three are testable without
building a scene, and all three carry a measurement that would be easy to get
wrong silently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from guide_control import (
    REJOIN_ERROR_M,
    SETTLE_REMAINING_M,
    GuideController,
)
from guide_follower import (
    CATCHUP_GAP_M,
    CATCHUP_SPEED_MPS,
    FOLLOW_OFFSET_M,
    FOLLOW_SPEED_MPS,
    MIN_TRAIL_GAP_M,
    STALLS,
    TANGENT_SMOOTH_M,
    Follower,
    Trail,
    stall_factor,
    stall_windows,
)
from guide_route import Route
from guide_states import (
    SPIN_BEST_RATE_DPS,
    VX_LEAD,
    VX_ONSET,
    VX_REJOIN,
    VX_SETTLE,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
    ZERO_COMMAND_STATES,
    ZERO_PATH_10S_M,
)
from guide_tracker import RouteTracker


def controller() -> GuideController:
    return GuideController(ctrl_hz=50.0)


# -- the exact-zero claim ---------------------------------------------------

def test_every_zero_command_state_returns_a_literal_zero():
    """Not a small number, not a decayed one.  The gate checks it literally."""
    c = controller()
    for state in ZERO_COMMAND_STATES:
        for kwargs in ({}, {"target_xy": (5.0, 5.0)},
                       {"look_at_yaw": 2.0}, {"route_remaining_m": 0.0}):
            command = c.raw_command(state, (0.0, 0.0), 0.0, **kwargs)
            assert command == (0.0, 0.0, 0.0), (
                f"{state} emitted {command} with {kwargs}")


def test_the_command_is_applied_without_filtering():
    """A low-pass filter would spend its first ticks BELOW the measured gait
    onset — no motion at all, followed by a jump — and would make the
    exact-zero claim false for several ticks after every stop."""
    c = controller()
    c.update("LEAD", (0.0, 0.0), 0.0, target_xy=(5.0, 0.0))
    assert c.command[0] == pytest.approx(VX_LEAD)
    c.update("WAIT_FOR_PERSON", (0.0, 0.0), 0.0)
    assert list(c.command) == [0.0, 0.0, 0.0]


def test_the_controller_never_spins():
    """Turn-in-place is MEASURED at 1.6 deg/s on this model, so it is not a
    manoeuvre.  ``spin_to`` exists only to make the finding discoverable."""
    c = controller()
    assert SPIN_BEST_RATE_DPS < 2.0
    for desired in np.linspace(-math.pi, math.pi, 25):
        assert c.spin_to(float(desired), 0.0) == 0.0


def test_no_command_ever_lands_in_the_dead_band():
    """A command between zero and the MEASURED onset appears in the metrics and
    produces nothing on the floor."""
    c = controller()
    samples = []
    for state in ("LEAD", "RESUME", "ARRIVE", "CHECK_FOLLOWER",
                  "WAIT_FOR_PERSON"):
        for remaining in (0.05, 0.4, 2.0, 9.0):
            for cross in (0.0, 0.2, 0.6):
                for yaw in np.linspace(-math.pi, math.pi, 9):
                    samples.append(c.raw_command(
                        state, (0.0, 0.0), float(yaw),
                        target_xy=(2.0, 1.0), look_at_yaw=1.0,
                        cross_track_m=cross, route_remaining_m=remaining))
    for vx, vy, wz in samples:
        assert vx == 0.0 or vx >= VX_ONSET, f"sub-onset forward command {vx}"
        assert vy == 0.0, "this behavior never emits vy"
        assert wz == 0.0 or abs(wz) >= min(WZ_MIN_LEFT, WZ_MIN_RIGHT)


def test_vy_is_never_emitted():
    """MEASURED: vy=+/-0.22 at vx=0 moves under 4 mm, and vy=-0.28 produces
    0.255 m sideways with 51 deg of unwanted yaw."""
    c = controller()
    for state in ("LEAD", "RESUME", "ARRIVE", "CHECK_FOLLOWER"):
        assert c.raw_command(state, (0.0, 0.0), 0.0,
                             target_xy=(1.0, 1.0), look_at_yaw=3.0)[1] == 0.0


# -- speed selection --------------------------------------------------------

def test_the_duck_eases_in_near_the_end_of_the_route():
    c = controller()
    fast = c.raw_command("LEAD", (0.0, 0.0), 0.0, target_xy=(3.0, 0.0),
                         route_remaining_m=3.0)
    slow = c.raw_command("LEAD", (0.0, 0.0), 0.0, target_xy=(3.0, 0.0),
                         route_remaining_m=SETTLE_REMAINING_M - 0.05)
    assert fast[0] == pytest.approx(VX_LEAD)
    assert slow[0] == pytest.approx(VX_SETTLE)
    assert slow[0] < fast[0]


def test_a_duck_pushed_off_the_line_closes_back_faster():
    c = controller()
    on_line = c.raw_command("LEAD", (0.0, 0.0), 0.0, target_xy=(3.0, 0.0),
                            cross_track_m=0.0, route_remaining_m=4.0)
    off_line = c.raw_command("LEAD", (0.0, 0.0), 0.0, target_xy=(3.0, 0.0),
                             cross_track_m=REJOIN_ERROR_M + 0.05,
                             route_remaining_m=4.0)
    assert on_line[0] == pytest.approx(VX_LEAD)
    assert off_line[0] == pytest.approx(VX_REJOIN)
    assert off_line[0] > on_line[0]


def test_the_yaw_gains_compensate_the_measured_bias():
    """MEASURED at vx=0.34: wz=-0.10 gives -7.3 deg/s but wz=+0.10 gives only
    +0.7 deg/s.  The policy's own right bias nearly swallows a small left
    command, so the LEFT gain is the larger one — for the same heading error the
    controller must push harder to the left to get the same rotation.

    Note which way round this is: it is the COMMAND that is asymmetric, not the
    angle at which each sign wakes up.  A first draft of this test asserted the
    opposite and passed for the wrong reason.
    """
    from guide_states import KP_YAW_LEFT, KP_YAW_RIGHT
    c = controller()
    assert KP_YAW_LEFT > KP_YAW_RIGHT
    assert WZ_MIN_LEFT > WZ_MIN_RIGHT
    error = math.radians(20.0)
    left = c.yaw_to(error, 0.0)
    right = c.yaw_to(-error, 0.0)
    assert left > 0.0 and right < 0.0
    assert abs(left) > abs(right), (
        "the left command is not stronger, so the measured right bias is "
        "uncompensated")


def test_both_yaw_signs_have_a_dead_band():
    """A command below the dead band appears in the metrics and produces no
    rotation, which is the same trap as the forward gait-onset cliff."""
    c = controller()
    tiny = math.radians(1.0)
    assert c.yaw_to(tiny, 0.0) == 0.0
    assert c.yaw_to(-tiny, 0.0) == 0.0


def test_the_yaw_controller_turns_the_short_way():
    c = controller()
    assert c.yaw_to(math.radians(170.0), math.radians(-170.0)) < 0.0
    assert c.yaw_to(math.radians(-170.0), math.radians(170.0)) > 0.0


def test_a_reached_target_stops_the_duck():
    c = controller()
    assert c.raw_command("LEAD", (0.0, 0.0), 0.0,
                         target_xy=(0.01, 0.0))[0] == 0.0


# -- the route tracker ------------------------------------------------------

def straight() -> Route:
    return Route("straight", ((0.0, 0.0), (4.0, 0.0)), 1.0)


def hairpin() -> Route:
    """A route that comes back near itself, which is the case the monotonic
    cursor exists for."""
    return Route("hairpin", ((0.0, 0.0), (3.0, 0.0), (3.0, 0.6), (0.0, 0.6)),
                 1.0, radius=0.25)


def test_the_cursor_never_moves_backwards():
    """``lost-child-find-person`` learned this the hard way: a stateless
    nearest-point selector re-acquires an already-passed part of the route and
    loops for ever."""
    tracker = RouteTracker(straight())
    tracker.project((2.0, 0.0))
    advanced = tracker.arc_s
    assert advanced > 0.0
    tracker.project((0.0, 0.0))
    assert tracker.arc_s >= advanced


def test_a_route_that_doubles_back_cannot_pull_the_cursor_back():
    tracker = RouteTracker(hairpin())
    for x in np.linspace(0.0, 3.0, 40):
        tracker.project((float(x), 0.0))
    at_corner = tracker.arc_s
    # A point on the RETURN leg is physically near the outbound leg.
    tracker.project((1.5, 0.6))
    assert tracker.arc_s >= at_corner


def test_remaining_distance_is_arc_length_not_euclidean():
    """On a route that bends back, the straight-line distance to the end is
    small while metres of path remain; using it would make the duck ease in
    halfway across the hall."""
    route = hairpin()
    tracker = RouteTracker(route)
    tracker.project((0.05, 0.0))
    euclidean = float(np.linalg.norm(
        np.array(route.corners[-1]) - np.array([0.05, 0.0])))
    assert tracker.remaining_m > euclidean


def test_the_pursuit_point_is_on_the_route_and_ahead():
    route = straight()
    tracker = RouteTracker(route)
    tracker.project((1.0, 0.0))
    point = tracker.pursuit_point()
    assert point[0] > 1.0
    assert abs(float(point[1])) < 1e-9


def test_the_pursuit_point_is_clamped_to_the_end():
    """The cursor advances through a BOUNDED window each tick, so reaching the
    end takes repeated projections — which is exactly what stops a route that
    doubles back from being re-acquired at the wrong place."""
    route = straight()
    tracker = RouteTracker(route)
    for x in np.linspace(0.0, 4.0, 200):
        tracker.project((float(x), 0.0))
    assert tracker.remaining_m == pytest.approx(0.0, abs=0.05)
    point = tracker.pursuit_point()
    assert float(np.linalg.norm(
        point - np.array(route.corners[-1]))) < 0.06


def test_cross_track_is_measured():
    tracker = RouteTracker(straight())
    tracker.project((1.0, 0.25))
    assert tracker.cross_track_m == pytest.approx(0.25, abs=0.02)


# -- the follower -----------------------------------------------------------

def test_she_walks_the_ducks_trail_and_cannot_lead_it():
    """A follower with a route of her own would arrive whether or not the duck
    ever moved, and every 'she followed me' claim would be a coincidence."""
    follower = Follower((-1.0, 0.0), (0.0, 0.0))
    for step in range(400):
        follower.push_duck((step * 0.01, 0.0))
        follower.update(step * 0.02, 0.02, moving=True)
        assert follower.trail_gap_m >= MIN_TRAIL_GAP_M - 1e-9, (
            "she got closer than the arc-length clamp allows")
    assert follower.walked_m > 0.5


def test_she_cannot_move_before_the_duck_sets_off():
    follower = Follower((-1.0, 0.0), (0.0, 0.0))
    start = follower.pos.copy()
    for step in range(200):
        follower.push_duck((0.0, 0.0))
        follower.update(step * 0.02, 0.02, moving=False)
    assert float(np.linalg.norm(follower.pos - start)) == 0.0
    assert follower.walked_m == 0.0


def test_a_stopped_duck_lets_her_close_the_gap():
    """'Waiting worked' has to be a measurement, not a hope."""
    follower = Follower((-1.0, 0.0), (0.0, 0.0))
    for step in range(600):
        follower.push_duck((step * 0.004, 0.0))
        follower.update(step * 0.02, 0.02, moving=True)
    gap_before = follower.trail_gap_m
    assert gap_before > MIN_TRAIL_GAP_M + 0.3, (
        "she never fell behind, so there is nothing to close")
    duck_at = (599 * 0.004, 0.0)
    for step in range(600, 1800):
        follower.push_duck(duck_at)          # the duck has stopped
        follower.update(step * 0.02, 0.02, moving=True)
    assert follower.trail_gap_m < gap_before
    assert follower.trail_gap_m == pytest.approx(MIN_TRAIL_GAP_M, abs=0.02), (
        "she did not close all the way to the arc-length clamp")


def test_she_hurries_when_she_has_fallen_a_long_way_behind():
    assert CATCHUP_SPEED_MPS > FOLLOW_SPEED_MPS
    assert CATCHUP_GAP_M > MIN_TRAIL_GAP_M


def test_the_stall_ramps_are_continuous():
    """A person who goes from 0.16 m/s to 0 in one control tick is a teleport
    with extra steps, and the lag detector would be grading the script."""
    previous, _ = stall_factor(0.0)
    for index in range(1, 5000):
        t = index * 0.02
        factor, _ = stall_factor(t)
        assert abs(factor - previous) < 0.05, (
            f"speed factor jumped {previous} -> {factor} at t={t}")
        previous = factor


def test_the_stalls_are_declared_but_the_machine_cannot_see_them():
    """The comparison between declared stalls and detected episodes is the
    whole point; conflating them would make the detection unfalsifiable.

    Checked by IMPORT rather than by text search: the machine's docstring
    legitimately explains where the stalls live, and a substring test on the
    whole file fails on the explanation instead of on the code.
    """
    import ast

    import guide_machine

    tree = ast.parse(open(guide_machine.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "guide_follower" not in imported, (
        "the machine imports the module that declares the stall schedule")
    assert "guide_actors" not in imported

    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "STALLS" not in names
    assert "stall_factor" not in names

    windows = stall_windows()
    assert len(windows) >= 2
    assert any(w["speed_factor"] == 0.0 for w in windows), (
        "no stall is a full stop")


def test_the_two_stalls_are_different_kinds_of_event():
    assert len(STALLS) >= 2
    factors = [s[2] for s in STALLS]
    assert min(factors) == 0.0
    assert max(factors) > 0.0, (
        "both stalls are full stops; one should be a slow crawl so the "
        "detector is exercised on distance as well as on a standstill")


# -- the follow offset, and the bug it fixed --------------------------------

def test_the_follow_offset_keeps_her_inside_the_heads_reach():
    """MEASURED: with a zero offset she sits at 173-180 deg astern, outside the
    head's +/-170 deg range, and the camera cannot see her however well the
    tracker aims.  This is the arithmetic that fixed it."""
    assert FOLLOW_OFFSET_M > 0.0
    bearing = 180.0 - math.degrees(math.atan2(FOLLOW_OFFSET_M,
                                              MIN_TRAIL_GAP_M))
    assert bearing <= 160.0, (
        f"at the minimum trail gap she sits at {bearing:.1f} deg, which leaves "
        "no margin against the 170 deg head limit")


def test_the_trail_tangent_is_smoothed_before_the_offset_is_applied():
    """MEASURED: the raw per-segment tangent of a 50 Hz trail swings with the
    gait, and an offset along it turned a 10 m walk into 163 m of path."""
    assert TANGENT_SMOOTH_M > 0.0
    # A trail with realistic per-tick jitter: 2.6 mm steps with lateral noise.
    rng = np.random.default_rng(7)
    points = [np.array([0.0, 0.0])]
    for index in range(1, 900):
        points.append(np.array([index * 0.0026,
                                float(rng.normal(0.0, 0.0008))]))
    trail = Trail(points[:2])
    for point in points[2:]:
        trail.append(point)

    def path_length(smooth: float) -> float:
        total, previous = 0.0, None
        for step in range(600):
            s = step * 0.0035
            position, tangent = trail.pose_at(s, smooth)
            normal = np.array([-tangent[1], tangent[0]])
            here = position + normal * FOLLOW_OFFSET_M
            if previous is not None:
                total += float(np.linalg.norm(here - previous))
            previous = here
        return total

    raw = path_length(0.0)
    smoothed = path_length(TANGENT_SMOOTH_M)
    assert smoothed < raw / 3.0, (
        f"smoothing barely helped: {smoothed:.2f} m against {raw:.2f} m raw")


def test_zero_command_drift_is_small_enough_to_call_it_stopped():
    """The measurement WAIT_FOR_PERSON rests on."""
    assert ZERO_PATH_10S_M < 0.01
