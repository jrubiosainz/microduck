#!/usr/bin/env python3
"""Pure logic tests: conflict prediction, the state machine, the controller.

No MuJoCo, no ONNX, no rendering.  Everything here runs in milliseconds and
tests the DECISION layer, which is where a crossing behavior is right or wrong.

Every meaningful gate has a MUTATION COUNTEREXAMPLE: a deliberately broken
variant that the same assertion must reject.  A test that only ever sees the
correct implementation cannot tell whether it is testing anything, and an
assertion comparing a constant to itself is worse than no assertion at all.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from conflict import (  # noqa: E402
    CROSS_DURATION_PESSIMISM,
    CROSS_SPEED_FAST_FACTOR,
    CROSS_SPEED_MPS,
    GAP_CONFIRM_S,
    ONSET_DEAD_TIME_S,
    PREDICT_HORIZON_S,
    SAFETY_MARGIN_S,
    STATES,
    STATIONARY_STATES,
    VX_MIN_EFFECTIVE,
    Interval,
    crossing_duration,
    duck_lane_intervals,
    evaluate_gap,
    vehicle_corridor_interval,
)
from guardian_model import (  # noqa: E402
    APPROACH_RELEASE_X,
    LOOK_PHASES,
    SECTOR_CONFIRM_S,
    GuardianController,
    GuardianMachine,
)
from street import (  # noqa: E402
    CROSS_GOAL_X,
    CURB_STOP_X,
    DUCK_PLANAR_RADIUS,
    LANE_DIRECTION,
    LANE_SIDE,
    ROAD_HALF_WIDTH,
    SAFE_ZONE_SPAN,
    WAIT_LINE_X,
    duck_span,
    encroaches_wait_line,
    in_lane,
    in_road,
    in_safe_zone,
)
from traffic import (  # noqa: E402
    HALF_LENGTH,
    HALF_WIDTH,
    KIND,
    LOOP_HALF_Y,
    SCHEDULE,
    VEHICLE_NAMES,
    crossing_arrivals,
    max_visible_jump,
    min_vehicle_separation,
    traffic_at,
)

DT = 1.0 / 50.0


@dataclass
class FakeVehicle:
    """Minimal stand-in for a VehicleState, so gap logic needs no simulator."""

    lane: str
    pos: np.ndarray
    vel: np.ndarray
    half_length: float = 0.229


def _vehicle(lane: str, y: float, vy: float, half_length: float = 0.229):
    x = -0.275 if lane == "near" else 0.275
    return FakeVehicle(lane=lane, pos=np.array([x, y]),
                       vel=np.array([0.0, vy]), half_length=half_length)


# ===================================================================
# street geometry
# ===================================================================

def test_lane_sides_match_travel_directions():
    """Near lane traffic comes from the left; far lane from the right.

    This is the fact that makes LOOK_LEFT → LOOK_RIGHT → LOOK_LEFT_AGAIN the
    correct order.  If the scene's travel directions and the scan's sector
    mapping ever disagree, the duck looks the wrong way and the whole behavior
    is a lie told convincingly.
    """
    # near lane travels −Y, so vehicles come from +Y, which is the duck's left
    assert LANE_DIRECTION["near"] < 0.0
    assert LANE_SIDE["near"] == "left"
    assert LANE_DIRECTION["far"] > 0.0
    assert LANE_SIDE["far"] == "right"


def test_scan_order_checks_the_first_lane_entered_first_and_last():
    """The scan must open and close on the lane the duck steps into first."""
    order = [sector for _, _, sector in LOOK_PHASES]
    assert order == ["left", "right", "left"]
    # and "left" must be the sector belonging to the NEAR lane
    assert LANE_SIDE["near"] == order[0] == order[-1]


def test_footprint_occupancy_uses_the_whole_duck_not_its_centre():
    """A trunk centre just outside a lane can still have the body inside it."""
    edge = -ROAD_HALF_WIDTH          # near edge of the road
    just_outside = edge - 0.5 * DUCK_PLANAR_RADIUS
    assert not (just_outside > edge)             # centre is outside
    assert in_road(just_outside)                 # body is not
    low, high = duck_span(just_outside)
    assert high > edge

    # MUTATION: a centre-point test would call this clear.
    def centre_only_in_road(x):
        return -ROAD_HALF_WIDTH < x < ROAD_HALF_WIDTH

    assert not centre_only_in_road(just_outside)
    assert in_road(just_outside) != centre_only_in_road(just_outside)


def test_curb_stop_keeps_the_whole_duck_behind_the_wait_line():
    assert not encroaches_wait_line(CURB_STOP_X)
    assert duck_span(CURB_STOP_X)[1] < -WAIT_LINE_X
    # and the release point is behind the stop, never past it
    assert APPROACH_RELEASE_X < CURB_STOP_X


def test_cross_goal_puts_the_whole_duck_clear_of_the_road():
    assert in_safe_zone(CROSS_GOAL_X)
    assert duck_span(CROSS_GOAL_X)[0] > ROAD_HALF_WIDTH
    assert SAFE_ZONE_SPAN[0] <= CROSS_GOAL_X <= SAFE_ZONE_SPAN[1]


def test_lane_membership_is_exclusive_at_the_centre_line_only_by_overlap():
    assert in_lane(-0.40, "near") and not in_lane(-0.40, "far")
    assert in_lane(0.40, "far") and not in_lane(0.40, "near")
    # straddling the centre line means BOTH, which is the honest answer
    assert in_lane(0.0, "near") and in_lane(0.0, "far")


# ===================================================================
# interval algebra
# ===================================================================

def test_interval_gap_is_signed_and_symmetric():
    a, b = Interval(0.0, 2.0), Interval(5.0, 7.0)
    assert a.gap_to(b) == pytest.approx(3.0)
    assert b.gap_to(a) == pytest.approx(3.0)


def test_overlapping_intervals_report_negative_overlap_duration():
    a, b = Interval(0.0, 4.0), Interval(3.0, 9.0)
    assert a.gap_to(b) == pytest.approx(-1.0)


def test_empty_interval_never_constrains():
    assert Interval(1.0, 0.0).empty
    assert Interval(0.0, 5.0).gap_to(Interval(1.0, 0.0)) == float("inf")


# ===================================================================
# the conflict predictor
# ===================================================================

def test_predicted_lane_window_contains_the_measured_occupancy():
    """The prediction must BRACKET the measurement on both sides.

    MEASURED (tools/measure_crossing.py, closed loop from x=-0.95):
        near lane occupied [1.16, 3.72] s,  far lane occupied [2.88, 5.50] s.

    This is the single most important property in the behavior: an optimistic
    prediction authorises a crossing the duck cannot complete in time.
    """
    measured = {"near": (1.16, 3.72), "far": (2.88, 5.50)}
    windows = duck_lane_intervals(-0.95)
    for lane, (enter, exit_) in measured.items():
        assert windows[lane].start < enter, (
            f"{lane}: predicted entry {windows[lane].start:.2f} is LATER than "
            f"measured {enter:.2f} — the duck is in the lane before the "
            "prediction says so")
        assert windows[lane].end > exit_, (
            f"{lane}: predicted exit {windows[lane].end:.2f} is EARLIER than "
            f"measured {exit_:.2f}")


def test_uniform_pessimism_would_fail_the_bracket():
    """Mutation: the bug this design replaced.

    Dividing the nominal speed by a single pessimism factor stretches BOTH ends
    later, so the predicted entry slips past the measured one.  The measured
    near-lane entry is 1.16 s; the uniform form predicts 1.38 s.
    """
    speed = CROSS_SPEED_MPS / CROSS_DURATION_PESSIMISM
    naive_entry = (-ROAD_HALF_WIDTH - DUCK_PLANAR_RADIUS - (-0.95)) / speed
    assert naive_entry > 1.16, "the historical bug should reproduce here"
    assert duck_lane_intervals(-0.95)["near"].start < 1.16


def test_fast_factor_and_dead_time_push_the_window_outward():
    wide = duck_lane_intervals(-0.95)
    tight = duck_lane_intervals(-0.95, fast_factor=1.0, dead_time=0.0,
                                pessimism=1.0)
    assert wide.__class__ is dict
    assert wide["near"].start < tight["near"].start
    assert wide["near"].end > tight["near"].end


def test_vehicle_window_brackets_its_own_body_length():
    """A longer vehicle occupies the corridor for longer at the same speed."""
    short = vehicle_corridor_interval(-5.0, 1.0, 0.10)
    long = vehicle_corridor_interval(-5.0, 1.0, 0.40)
    assert (long.end - long.start) > (short.end - short.start)


def test_receding_vehicle_has_an_empty_window():
    """A vehicle that has already passed cannot come back inside the horizon."""
    assert vehicle_corridor_interval(3.0, 1.5, 0.229).empty


def test_stationary_vehicle_on_the_crossing_blocks_forever():
    window = vehicle_corridor_interval(0.0, 0.0, 0.229)
    assert not window.empty
    assert window.end == pytest.approx(PREDICT_HORIZON_S)


def test_far_lane_vehicle_is_judged_against_the_far_lane_window():
    """Per-lane judging is what makes the decision honest AND harder.

    A far-lane vehicle arriving at t=1 s is harmless — the duck is not there
    yet — while the same vehicle arriving when the duck is in the far lane is
    not.  A predictor that treated the road as one block would get the first
    case wrong in the unsafe direction and the second right by luck.
    """
    early = evaluate_gap({"v": _vehicle("far", -1.2, 1.6)}, (CURB_STOP_X, 0.0))
    late = evaluate_gap({"v": _vehicle("far", -7.2, 1.6)}, (CURB_STOP_X, 0.0))
    # the early one passes before the duck ever reaches the far lane
    assert early.conflicts[0].vehicle_window.end < \
        duck_lane_intervals(CURB_STOP_X)["far"].start
    assert early.safe
    assert not late.safe


def test_unsafe_gap_names_the_limiting_vehicle():
    traffic = {
        "close": _vehicle("near", 1.4, -1.3),
        "distant": _vehicle("far", -24.0, 1.0),
    }
    decision = evaluate_gap(traffic, (CURB_STOP_X, 0.0))
    assert not decision.safe
    assert decision.limiting_vehicle == "close"
    assert decision.blocking[0].name == "close"


def test_empty_road_is_safe_and_a_crowded_one_is_not():
    assert evaluate_gap({}, (CURB_STOP_X, 0.0)).safe
    crowded = {name: _vehicle("near" if i % 2 else "far",
                              -2.0 - 0.4 * i, 1.2 * (1 if i % 2 == 0 else -1))
               for i, name in enumerate(VEHICLE_NAMES)}
    assert not evaluate_gap(crowded, (CURB_STOP_X, 0.0)).safe


def test_margin_requirement_is_not_vacuous():
    """A gap that merely avoids collision must still be rejected.

    The vehicle here clears the duck's near-lane window by 0.70 s: no overlap
    at all, so a pure collision test accepts it.  The margin requirement
    rejects it.  Mutation: with ``margin=0`` the same geometry IS accepted, so
    the test proves the margin does work rather than restating "no overlap".
    """
    traffic = {"v": _vehicle("near", 6.9, -1.0)}
    strict = evaluate_gap(traffic, (CURB_STOP_X, 0.0))
    loose = evaluate_gap(traffic, (CURB_STOP_X, 0.0), margin=0.0)
    assert 0.0 < strict.worst_margin_s < SAFETY_MARGIN_S, (
        "fixture must sit in the band that separates 'no collision' from "
        f"'safe'; got {strict.worst_margin_s:.3f}")
    assert not strict.safe
    assert loose.safe


def test_crossing_duration_grows_with_distance_and_includes_dead_time():
    near = crossing_duration(-0.5)
    far = crossing_duration(-1.5)
    assert far > near
    assert crossing_duration(0.0, goal_x=0.0) == pytest.approx(
        ONSET_DEAD_TIME_S)


def test_zero_or_negative_crossing_speed_is_refused():
    with pytest.raises(ValueError):
        duck_lane_intervals(-0.95, speed=0.0)


# ===================================================================
# the state machine
# ===================================================================

def _run_machine(*, ticks: int, sector_visible=True, decision=None,
                 trunk_x=lambda t, state: CURB_STOP_X, machine=None):
    machine = machine or GuardianMachine(ctrl_hz=50.0)
    seen = []
    for index in range(ticks):
        t = index * DT
        x = trunk_x(t, machine.state)
        visible = (sector_visible(t, machine.state)
                   if callable(sector_visible) else sector_visible)
        verdict = (decision(t, machine.state) if callable(decision)
                   else decision)
        state, _ = machine.update(t, trunk_x=x, sector_visible=visible,
                                  decision=verdict)
        seen.append(state)
    return machine, seen


class _Safe:
    safe = True
    worst_margin_s = 4.2
    limiting_vehicle = "van"
    crossing_duration_s = 7.5

    def as_record(self):
        return {"safe": True, "worst_margin_s": 4.2, "limiting_vehicle": "van",
                "crossing_duration_s": 7.5, "start_x": CURB_STOP_X,
                "blocking": []}


class _Unsafe:
    safe = False
    worst_margin_s = -0.8
    limiting_vehicle = "hatch"
    crossing_duration_s = 7.5

    def as_record(self):
        return {"safe": False, "worst_margin_s": -0.8,
                "limiting_vehicle": "hatch", "crossing_duration_s": 7.5,
                "start_x": CURB_STOP_X,
                "blocking": [{"vehicle": "hatch", "lane": "near",
                              "margin_s": -0.8,
                              "vehicle_window_s": [1.0, 2.2],
                              "duck_window_s": [0.8, 5.3],
                              "range_m": 2.0, "speed_mps": 1.25}]}


def test_full_sequence_reaches_safe_in_order():
    def x_of(t, state):
        if state == "APPROACH_CURB":
            return -2.05 + 0.30 * t
        if state == "CROSSING":
            return CURB_STOP_X + 0.30 * t
        return CURB_STOP_X

    machine, seen = _run_machine(
        ticks=3000, decision=_Safe(), trunk_x=x_of)
    ordered = [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]]
    assert ordered == list(STATES)


def test_a_look_phase_cannot_complete_without_seeing_its_sector():
    """The dwell time alone must not advance the scan.

    This is the gate that makes "the duck looked" mean something.  With the
    sector never visible, the phase must sit until its ceiling and record
    ``sector_confirmed=False`` — never silently pass.
    """
    machine, seen = _run_machine(ticks=400, sector_visible=False)
    assert "LOOK_LEFT" in seen
    left = [e for e in machine.scan_log if e["phase"] == "LOOK_LEFT"]
    if left:
        assert left[0]["sector_confirmed"] is False
        assert machine.timeouts
    else:
        assert machine.state == "LOOK_LEFT"


def test_look_phase_needs_continuous_visibility_not_a_flicker():
    """A sector glimpsed for one tick at a time must not satisfy the gate."""
    def flicker(t, state):
        return int(t / DT) % 2 == 0        # visible every other tick

    machine, _ = _run_machine(ticks=int(6.0 / DT), sector_visible=flicker)
    # Confirmation resets on every miss, so it can never reach SECTOR_CONFIRM_S.
    assert machine.state == "LOOK_LEFT"
    assert SECTOR_CONFIRM_S > DT


def test_unsafe_gap_is_rejected_and_recorded_with_its_limiting_vehicle():
    machine = GuardianMachine(ctrl_hz=50.0)
    machine.state = "WAIT_FOR_GAP"
    _run_machine(ticks=200, decision=_Unsafe(), machine=machine)
    assert machine.state == "WAIT_FOR_GAP"
    assert len(machine.rejected_gaps) == 1
    entry = machine.rejected_gaps[0]
    assert entry["limiting_vehicle"] == "hatch"
    assert entry["worst_margin_s"] < 0.0
    assert entry["blocking"][0]["vehicle"] == "hatch"


def test_repeated_rejection_of_one_vehicle_counts_once():
    """Collapsing by vehicle keeps the acceptance gate meaningful.

    Counting per tick would make ``rejected_unsafe_gap >= 1`` satisfiable by a
    single frame of arithmetic, so a run that never genuinely waited would
    pass.
    """
    machine = GuardianMachine(ctrl_hz=50.0)
    machine.state = "WAIT_FOR_GAP"
    _run_machine(ticks=500, decision=_Unsafe(), machine=machine)
    assert len(machine.rejected_gaps) == 1
    assert machine.rejected_gaps[0]["ticks"] > 100


def test_commitment_requires_the_gap_to_hold_not_one_lucky_tick():
    """One safe tick amid unsafe ones must not launch the duck."""
    def mostly_unsafe(t, state):
        return _Safe() if int(t / DT) % 40 == 0 else _Unsafe()

    machine = GuardianMachine(ctrl_hz=50.0)
    machine.state = "WAIT_FOR_GAP"
    _run_machine(ticks=1000, decision=mostly_unsafe, machine=machine)
    assert machine.state == "WAIT_FOR_GAP"
    assert GAP_CONFIRM_S > DT


def test_commitment_records_the_margin_that_authorised_it():
    machine = GuardianMachine(ctrl_hz=50.0)
    machine.state = "WAIT_FOR_GAP"
    _run_machine(ticks=200, decision=_Safe(), machine=machine)
    assert machine.state == "CROSSING"
    assert machine.commit["worst_margin_s"] == pytest.approx(4.2)
    assert machine.commit["limiting_vehicle"] == "van"


def test_crossing_is_never_re_decided_even_when_the_gap_turns_unsafe():
    """Stopping in a live lane is the worst response to a surprise.

    The commitment already covered the whole crossing, so the machine must
    ignore a later unsafe verdict rather than freezing mid-road.
    """
    machine = GuardianMachine(ctrl_hz=50.0)
    machine.state = "CROSSING"
    machine.commit = {"committed_at_s": 0.0}
    _run_machine(ticks=100, decision=_Unsafe(), machine=machine,
                 trunk_x=lambda t, s: 0.0)
    assert machine.state == "CROSSING"


def test_crossing_ends_only_at_the_goal_not_at_the_zone_edge():
    """Clipping the safe zone's near edge is not arriving.

    At the edge the duck's trailing surface is still only 0.12 m from the road.
    """
    edge = SAFE_ZONE_SPAN[0] + 0.001
    machine = GuardianMachine(ctrl_hz=50.0)
    machine.state = "CROSSING"
    machine.commit = {}
    _run_machine(ticks=20, machine=machine, trunk_x=lambda t, s: edge)
    assert machine.state == "CROSSING"
    assert in_safe_zone(edge)          # the weaker test would have passed
    _run_machine(ticks=20, machine=machine,
                 trunk_x=lambda t, s: CROSS_GOAL_X)
    assert machine.state == "SAFE"


def test_safe_is_absorbing():
    machine = GuardianMachine(ctrl_hz=50.0)
    machine.state = "SAFE"
    _run_machine(ticks=500, decision=_Unsafe(), machine=machine,
                 trunk_x=lambda t, s: CROSS_GOAL_X)
    assert machine.state == "SAFE"


# ===================================================================
# the controller
# ===================================================================

def test_every_stationary_state_commands_exactly_zero():
    controller = GuardianController(ctrl_hz=50.0)
    for state in STATIONARY_STATES:
        command = controller.update(state, CURB_STOP_X, 0.3, 0.2)
        assert list(command) == [0.0, 0.0, 0.0], state


def test_stationary_command_has_no_filter_tail():
    """Zero must be reached on the FIRST tick, not decayed toward.

    A decaying command is still a command, and the acceptance gate tests for
    exact zero, so a low-pass filter here would be a real defect.
    """
    controller = GuardianController(ctrl_hz=50.0)
    controller.update("CROSSING", 0.0, 0.0, 0.0)
    assert float(np.linalg.norm(controller.command)) > 0.0
    command = controller.update("WAIT_FOR_GAP", 0.0, 0.0, 0.0)
    assert float(np.linalg.norm(command)) == 0.0


def test_moving_states_never_emit_a_sub_onset_command():
    """A vx between zero and the measured onset produces NO motion at all."""
    controller = GuardianController(ctrl_hz=50.0)
    for state, x in (("APPROACH_CURB", -2.0), ("CROSSING", 0.0)):
        for yaw in np.linspace(-0.6, 0.6, 13):
            for y in (-0.3, 0.0, 0.3):
                vx = controller.raw_command(state, x, float(yaw), y)[0]
                assert vx == 0.0 or vx >= VX_MIN_EFFECTIVE


def test_heading_hold_steers_back_toward_the_zebra_centreline():
    """Drifted left of centre, the correction must point right, and vice versa."""
    controller = GuardianController(ctrl_hz=50.0)
    left_of_centre = controller.raw_command("CROSSING", 0.0, 0.0, +0.30)[2]
    right_of_centre = controller.raw_command("CROSSING", 0.0, 0.0, -0.30)[2]
    assert left_of_centre < 0.0, "drifted +y, must steer −wz (right)"
    assert right_of_centre > 0.0, "drifted −y, must steer +wz (left)"


def test_turn_gains_are_independent_per_sign():
    """The measured yaw authority is 3-5x stronger to the right at vx=0.58.

    Mirroring one gain onto the other side would make every left correction a
    violent over-correction, so the two must NOT be symmetric.
    """
    controller = GuardianController(ctrl_hz=50.0)
    left = controller.raw_command("CROSSING", 0.0, -0.35, 0.0)[2]
    right = controller.raw_command("CROSSING", 0.0, +0.35, 0.0)[2]
    assert left > 0.0 and right < 0.0
    assert abs(left) != pytest.approx(abs(right)), (
        "symmetric gains would ignore the measured asymmetry")


def test_approach_releases_its_command_at_the_release_point():
    controller = GuardianController(ctrl_hz=50.0)
    assert controller.raw_command(
        "APPROACH_CURB", APPROACH_RELEASE_X - 0.01, 0.0, 0.0)[0] > 0.0
    assert controller.raw_command(
        "APPROACH_CURB", APPROACH_RELEASE_X + 0.01, 0.0, 0.0) == (0.0, 0.0, 0.0)


# ===================================================================
# the traffic schedule
# ===================================================================

def test_both_travel_directions_are_used():
    directions = {v.direction for v in SCHEDULE}
    assert directions == {-1.0, 1.0}


def test_at_least_three_independently_moving_road_users():
    assert len(SCHEDULE) >= 3
    speeds = {v.speed for v in SCHEDULE}
    assert len(speeds) >= 3, "identical speeds would not be independent"


def test_nobody_freezes_at_any_point_in_the_rollout():
    for t in np.linspace(0.0, 50.0, 400):
        for state in traffic_at(float(t)).values():
            assert abs(float(state.vel[1])) > 0.1


def test_no_vehicle_teleports_anywhere_it_could_be_seen():
    """The loop wrap is a discontinuity; it must happen far off-screen.

    The honest claim is not "there is no jump" but "no jump happens where it
    could be seen or where it could affect a prediction".
    """
    jump, name, t = max_visible_jump(50.0, visible_half_y=12.0)
    fastest = max(v.speed for v in SCHEDULE)
    assert jump <= fastest * 0.02 + 1e-6, (
        f"{name} jumped {jump:.3f} m at t={t:.2f} s inside the visible band")


def test_a_wrapped_vehicle_is_beyond_the_predictor_horizon():
    """A vehicle that has just wrapped must not be able to alter a decision.

    A wrap moves a vehicle from one end of the loop to the other in one tick.
    If the far end were inside the predictor's reach, that jump would appear as
    a road user materialising inside the horizon and could flip a gap decision
    between two consecutive ticks.

    The requirement is therefore on the SLOWEST vehicle, not the fastest: the
    binding case is the one that takes longest to become relevant, and
    ``LOOP_HALF_Y`` must exceed the distance the FASTEST vehicle covers in a
    full horizon so that no wrap can ever land inside it.
    """
    fastest = max(v.speed for v in SCHEDULE)
    assert LOOP_HALF_Y > fastest * PREDICT_HORIZON_S, (
        f"a vehicle wrapping to |y|={LOOP_HALF_Y} m at {fastest} m/s reaches "
        f"the crossing in {LOOP_HALF_Y / fastest:.1f} s, inside the "
        f"{PREDICT_HORIZON_S} s horizon")
    for vehicle in SCHEDULE:
        assert LOOP_HALF_Y / vehicle.speed > PREDICT_HORIZON_S


def test_same_lane_vehicles_never_drive_through_each_other():
    gap, first, second = min_vehicle_separation(50.0)
    assert gap > 0.0, f"{first} and {second} overlap by {-gap:.3f} m"


def test_the_crossing_is_genuinely_blocked_before_it_is_clear():
    """The gap the duck takes must be one the schedule produced.

    Several vehicles must cross early, from both lanes, and the wide gap must
    come later — otherwise "waited for a safe gap" is unearned.
    """
    arrivals = crossing_arrivals(46.0)
    early = [a for a in arrivals if a["enter_s"] < 25.0]
    assert len(early) >= 3
    assert {a["lane"] for a in early} == {"near", "far"}
    gaps = [(arrivals[i + 1]["enter_s"] - arrivals[i]["exit_s"])
            for i in range(len(arrivals) - 1)]
    assert max(gaps) > 10.0, "no gap wide enough to cross ever opens"


def test_vehicles_are_several_times_faster_than_the_duck():
    """This is what makes the gap decision non-trivial rather than decorative."""
    for vehicle in SCHEDULE:
        assert vehicle.speed > 2.5 * CROSS_SPEED_MPS


def test_traffic_is_deterministic():
    first = traffic_at(13.37)
    second = traffic_at(13.37)
    for name in VEHICLE_NAMES:
        assert np.allclose(first[name].pos, second[name].pos)


def test_every_scheduled_vehicle_exists_in_the_scene_generator():
    """The schedule, the geometry tables and the scene must agree exactly.

    Adding a vehicle to the schedule without adding it to the generator gives a
    KeyError deep inside the render — AFTER the whole rollout has run, which is
    the most expensive possible moment to find out.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(
        0, str(_Path(__file__).resolve().parents[1] / "tools"))
    from build_scene import VEHICLES as BUILD_VEHICLES

    built = {name for name, _kind, _body, _trim in BUILD_VEHICLES}
    scheduled = {v.name for v in SCHEDULE}
    assert scheduled == built
    assert set(VEHICLE_NAMES) == scheduled
    for name in scheduled:
        assert name in HALF_LENGTH
        assert name in HALF_WIDTH
        assert name in KIND


def test_the_overlay_has_a_colour_for_every_vehicle():
    """Presentation tables must be derived, never restated by hand."""
    from video_overlay import VEHICLE_RGB

    for name in VEHICLE_NAMES:
        assert name in VEHICLE_RGB
        assert len(VEHICLE_RGB[name]) == 3


def test_measured_constants_are_self_consistent():
    """Guard rails on the constants the whole decision rests on."""
    assert CROSS_SPEED_FAST_FACTOR > 1.0
    assert CROSS_DURATION_PESSIMISM > 1.0
    assert ONSET_DEAD_TIME_S > 0.0
    assert SAFETY_MARGIN_S > 0.0
    assert VX_MIN_EFFECTIVE > 0.0
    assert PREDICT_HORIZON_S > crossing_duration(CURB_STOP_X, CROSS_GOAL_X)
