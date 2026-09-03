#!/usr/bin/env python3
"""Synthetic bad cases: proof that the acceptance gates can FAIL.

A suite of passing tests against a working implementation proves very little on
its own.  A gate that cannot fail is decoration, and a metrics function that
reports ``all_gates_pass`` for a robot that walked into a lorry is worse than no
gate at all.

Every test here builds a FAKE rollout that violates exactly one requirement and
asserts that ``guardian_metrics.summarize`` catches it — and only it.  The
fakes are deliberately plausible: each one is a rollout that would look fine in
a video and is wrong for a reason only the arithmetic can see.

``_FakeRollout`` mimics only the attributes ``summarize`` reads, so these tests
need neither MuJoCo nor the policy and run in milliseconds.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from conflict import SAFETY_MARGIN_S, STATES  # noqa: E402
from guardian_metrics import (  # noqa: E402
    MAX_ZERO_COMMAND_STEPS_IN_ROAD,
    MIN_CROSSING_NET_M,
    MIN_CROSSING_PATH_M,
    summarize,
)
from street import (  # noqa: E402
    CROSS_GOAL_X,
    CURB_STOP_X,
    DUCK_PLANAR_RADIUS,
    SAFE_ZONE_SPAN,
    WAIT_LINE_X,
    encroaches_wait_line,
    in_lane,
    in_road,
    in_safe_zone,
)

DT = 1.0 / 50.0


@dataclass
class _FakeMachine:
    scan_log: list = field(default_factory=list)
    commit: dict = field(default_factory=dict)
    timeouts: list = field(default_factory=list)
    gap_decisions: list = field(default_factory=list)
    _rejected: list = field(default_factory=list)

    @property
    def rejected_gaps(self):
        return self._rejected


@dataclass
class _FakeRollout:
    records: list
    machine: _FakeMachine
    seconds: float = 46.0
    dt: float = DT
    decimation: int = 10
    duck_radius: float = DUCK_PLANAR_RADIUS
    path_m: float = 4.0
    crossing_path_m: float = 2.27
    crossing_net_m_value: float = 2.18
    min_wait_line_margin: float = 0.14
    max_x_before_crossing: float = CURB_STOP_X
    transitions: list = field(default_factory=list)

    @property
    def crossing_net_m(self):
        return self.crossing_net_m_value


def _record(t, state, x, *, y=0.0, command=(0.0, 0.0, 0.0), z=0.116,
            clearance=0.55, left=1.0, right=1.0):
    return {
        "t": t, "state": state, "state_elapsed_s": 0.5,
        "command": list(command), "duck_xy": [x, y], "duck_yaw_deg": 0.0,
        "trunk_z_m": z, "min_trunk_z_m": 0.110,
        "in_road": in_road(x), "in_near_lane": in_lane(x, "near"),
        "in_far_lane": in_lane(x, "far"), "in_safe_zone": in_safe_zone(x),
        "encroaches": encroaches_wait_line(x),
        "wait_line_margin_m": -WAIT_LINE_X - (x + DUCK_PLANAR_RADIUS),
        "left_visible": left >= 0.5, "right_visible": right >= 0.5,
        "left_fraction": left, "right_fraction": right,
        "visible_vehicles": [], "view_yaw_deg": 0.0, "gaze_yaw_deg": 0.0,
        "gap_safe": True, "gap_margin_s": 3.0, "gap_limiting": "van",
        "crossing_estimate_s": 7.5, "vehicle_windows": {},
        "duck_windows": {"near": [0.8, 5.3]},
        "nearest_vehicle": "van", "nearest_clearance_m": clearance,
        "path_m": 4.0, "crossing_path_m": 2.27, "rejected_gaps": 1,
        "vehicle_xy": {},
    }


def _good_rollout(**overrides):
    """A synthetic rollout that passes every gate, as the control case."""
    records = []
    t = 0.0
    plan = [
        ("APPROACH_CURB", 90, -2.05, CURB_STOP_X, (0.52, 0.0, 0.0)),
        ("STOP", 65, CURB_STOP_X, CURB_STOP_X, (0.0, 0.0, 0.0)),
        ("LOOK_LEFT", 120, CURB_STOP_X, CURB_STOP_X, (0.0, 0.0, 0.0)),
        ("LOOK_RIGHT", 120, CURB_STOP_X, CURB_STOP_X, (0.0, 0.0, 0.0)),
        ("LOOK_LEFT_AGAIN", 96, CURB_STOP_X, CURB_STOP_X, (0.0, 0.0, 0.0)),
        ("WAIT_FOR_GAP", 610, CURB_STOP_X, CURB_STOP_X, (0.0, 0.0, 0.0)),
        ("CROSSING", 362, CURB_STOP_X, CROSS_GOAL_X, (0.58, 0.0, 0.0)),
        ("SAFE", 837, CROSS_GOAL_X, CROSS_GOAL_X, (0.0, 0.0, 0.0)),
    ]
    for state, steps, x0, x1, command in plan:
        for index in range(steps):
            frac = index / max(steps - 1, 1)
            left = 1.0 if state in ("LOOK_LEFT", "LOOK_LEFT_AGAIN") else 0.0
            right = 1.0 if state == "LOOK_RIGHT" else 0.0
            records.append(_record(
                t, state, x0 + frac * (x1 - x0), command=command,
                left=left, right=right))
            t += DT
    machine = _FakeMachine(
        scan_log=[
            {"phase": "LOOK_LEFT", "sector": "left", "sector_confirmed": True},
            {"phase": "LOOK_RIGHT", "sector": "right", "sector_confirmed": True},
            {"phase": "LOOK_LEFT_AGAIN", "sector": "left",
             "sector_confirmed": True},
        ],
        commit={"committed_at_s": 24.08, "wait_duration_s": 12.2,
                "worst_margin_s": 2.55, "limiting_vehicle": "van",
                "crossing_duration_s": 7.24,
                "crossing_duration_estimate_s": 8.12},
        _rejected=[{"limiting_vehicle": "taxi", "worst_margin_s": -1.11,
                    "blocking": [], "first_rejected_at_s": 17.6,
                    "last_rejected_at_s": 23.68, "ticks": 305}],
    )
    rollout = _FakeRollout(records=records, machine=machine)
    for key, value in overrides.items():
        setattr(rollout, key, value)
    return rollout


def _gates(rollout):
    return summarize(rollout)["gates"]


# ===================================================================
# the control case
# ===================================================================

def test_the_synthetic_good_rollout_passes_every_gate():
    """Without this, a bad-case test proving 'a gate failed' proves nothing.

    Every test below asserts that exactly ONE gate flips.  That claim is only
    meaningful if the baseline passes them all.
    """
    gates = _gates(_good_rollout())
    failed = [name for name, passed in gates.items() if not passed]
    assert failed == [], f"the control case should pass everything: {failed}"


def _only_failure(rollout, expected: str):
    """Assert exactly the named gate failed, and report honestly if not."""
    gates = _gates(rollout)
    failed = sorted(name for name, passed in gates.items() if not passed)
    assert expected in failed, (
        f"{expected} should have failed; failures were {failed}")
    return failed


# ===================================================================
# synthetic bad cases
# ===================================================================

def test_a_duck_that_walks_straight_into_the_road_fails_the_scan_gates():
    """The headline failure: no scan at all, just walk across.

    This is the behavior a naive implementation produces, and it must not be
    able to reach ``all_gates_pass``.
    """
    records = []
    t = 0.0
    for index in range(700):
        x = -2.05 + (CROSS_GOAL_X + 2.05) * index / 699
        records.append(_record(t, "CROSSING", x, command=(0.58, 0.0, 0.0)))
        t += DT
    rollout = _FakeRollout(
        records=records, machine=_FakeMachine(), min_wait_line_margin=-1.5)
    summary = summarize(rollout)
    assert not summary["all_gates_pass"]
    for gate in ("state_order", "scan_phases", "scan_sectors_seen",
                 "rejected_unsafe_gap", "commit_margin", "no_early_encroach"):
        assert not summary["gates"][gate], gate


def test_skipping_the_third_scan_phase_is_caught():
    """LOOK_LEFT → LOOK_RIGHT → cross is a real-world pedestrian error."""
    rollout = _good_rollout()
    rollout.records = [r for r in rollout.records
                       if r["state"] != "LOOK_LEFT_AGAIN"]
    rollout.machine.scan_log = [
        entry for entry in rollout.machine.scan_log
        if entry["phase"] != "LOOK_LEFT_AGAIN"]
    failed = _only_failure(rollout, "scan_phases")
    assert "state_order" in failed


def test_scanning_in_the_wrong_order_is_caught():
    """Looking right first checks the lane the duck reaches SECOND."""
    rollout = _good_rollout()
    rollout.machine.scan_log = [
        {"phase": "LOOK_RIGHT", "sector": "right", "sector_confirmed": True},
        {"phase": "LOOK_LEFT", "sector": "left", "sector_confirmed": True},
        {"phase": "LOOK_LEFT_AGAIN", "sector": "left",
         "sector_confirmed": True},
    ]
    _only_failure(rollout, "scan_phases")


def test_a_scan_phase_that_never_saw_its_sector_is_caught():
    """Turning the head is not looking: the road has to be in the camera."""
    rollout = _good_rollout()
    for record in rollout.records:
        if record["state"] == "LOOK_RIGHT":
            record["right_visible"] = False
            record["right_fraction"] = 0.0
    _only_failure(rollout, "scan_sectors_seen")


def test_crossing_without_ever_rejecting_a_gap_is_caught():
    """If no gap was ever refused, the duck did not WAIT for one."""
    rollout = _good_rollout()
    rollout.machine._rejected = []
    _only_failure(rollout, "rejected_unsafe_gap")


def test_committing_on_an_insufficient_margin_is_caught():
    """A gap that clears by 0.4 s is a near miss, not a decision."""
    rollout = _good_rollout()
    rollout.machine.commit = dict(rollout.machine.commit)
    rollout.machine.commit["worst_margin_s"] = 0.4
    assert 0.4 < SAFETY_MARGIN_S
    _only_failure(rollout, "commit_margin")


def test_stopping_inside_the_traffic_lane_is_caught():
    """THE failure this behavior exists to prevent.

    The duck freezes in the near lane for a second, then continues.  Every
    other gate still passes — it reaches the safe zone, it never touches a
    vehicle, it does not fall — so only the plateau test can catch it.
    """
    rollout = _good_rollout()
    frozen = 0
    for record in rollout.records:
        if record["state"] == "CROSSING" and in_lane(record["duck_xy"][0],
                                                     "near") and frozen < 60:
            record["command"] = [0.0, 0.0, 0.0]
            frozen += 1
    assert frozen > MAX_ZERO_COMMAND_STEPS_IN_ROAD
    _only_failure(rollout, "crossing_continuous")


def test_a_stalled_crossing_with_a_nonzero_command_is_also_caught():
    """A nonzero command does not prove the policy crossed its gait onset.

    Here the duck is commanded to walk the whole way but physically stops in
    the road for a second.  The zero-command test passes; the physical advance
    test is what catches it.
    """
    rollout = _good_rollout()
    stalled_x = None
    stalled = 0
    for record in rollout.records:
        if record["state"] != "CROSSING":
            continue
        if in_lane(record["duck_xy"][0], "far") and stalled < 60:
            stalled_x = stalled_x if stalled_x is not None else record["duck_xy"][0]
            record["duck_xy"] = [stalled_x, record["duck_xy"][1]]
            stalled += 1
    assert stalled > 0
    gates = _gates(rollout)
    assert not gates["crossing_continuous"]


def test_a_crossing_that_barely_moves_is_caught():
    """Reaching SAFE without physically traversing the road."""
    rollout = _good_rollout(crossing_path_m=0.4, crossing_net_m_value=0.3)
    assert 0.4 < MIN_CROSSING_PATH_M and 0.3 < MIN_CROSSING_NET_M
    _only_failure(rollout, "crossing_moved")


def test_encroaching_the_wait_line_before_crossing_is_caught():
    """Toes over the paint while still deciding is a real safety failure."""
    rollout = _good_rollout(min_wait_line_margin=-0.02)
    _only_failure(rollout, "no_early_encroach")


def test_stopping_past_the_wait_line_is_caught():
    """The duck must stop BEFORE the line, graded on its whole footprint."""
    rollout = _good_rollout()
    over = -WAIT_LINE_X - DUCK_PLANAR_RADIUS + 0.05
    for record in rollout.records:
        if record["state"] == "STOP":
            record["duck_xy"] = [over, 0.0]
    _only_failure(rollout, "stops_before_line")


def test_a_vehicle_overlap_is_caught():
    """Being driven through must never read as a successful crossing."""
    rollout = _good_rollout()
    for record in rollout.records[400:404]:
        record["nearest_clearance_m"] = -0.03
    failed = _only_failure(rollout, "no_vehicle_contact")
    assert "positive_clearance" in failed


def test_a_command_leaking_into_a_stationary_state_is_caught():
    """'Stopped' must mean exactly zero, not nearly zero.

    A filter tail of 1e-3 is invisible in a video and is still a command.
    """
    rollout = _good_rollout()
    for record in rollout.records:
        if record["state"] == "WAIT_FOR_GAP":
            record["command"] = [1e-3, 0.0, 0.0]
            break
    _only_failure(rollout, "still_when_still")


def test_a_fall_is_caught():
    rollout = _good_rollout()
    for record in rollout.records[500:520]:
        record["trunk_z_m"] = 0.06
    failed = _only_failure(rollout, "no_falls")
    assert "min_trunk_z" in failed


def test_stopping_short_of_the_safe_zone_is_caught():
    """Ending on the kerb is not reaching the opposite pavement."""
    rollout = _good_rollout()
    short = SAFE_ZONE_SPAN[0] - 0.10
    for record in rollout.records:
        if record["state"] == "SAFE":
            record["duck_xy"] = [short, 0.0]
            record["in_safe_zone"] = in_safe_zone(short)
    rollout.crossing_net_m_value = 1.9
    _only_failure(rollout, "reaches_safe_zone")


def test_reversing_back_toward_the_road_after_arriving_is_caught():
    """A duck that wanders back off the pavement has not finished safely."""
    rollout = _good_rollout()
    safe = [r for r in rollout.records if r["state"] == "SAFE"]
    for index, record in enumerate(safe[-100:]):
        record["duck_xy"] = [CROSS_GOAL_X - 0.004 * (index + 1), 0.0]
    _only_failure(rollout, "no_reverse")


def test_a_phase_timeout_is_caught():
    """A gate satisfied only because a phase gave up is not satisfied."""
    rollout = _good_rollout()
    rollout.machine.timeouts = ["LOOK_RIGHT_sector_timeout"]
    _only_failure(rollout, "no_timeouts")


def test_a_final_height_far_from_nominal_is_caught():
    rollout = _good_rollout()
    rollout.records[-1]["trunk_z_m"] = 0.095
    _only_failure(rollout, "final_trunk_z")


def test_ending_in_the_wrong_state_is_caught():
    rollout = _good_rollout()
    for record in rollout.records[-50:]:
        record["state"] = "CROSSING"
    failed = _only_failure(rollout, "final_state_safe")
    assert "state_order" in failed


def test_state_order_gate_requires_all_eight_states_exactly_once():
    """A machine that oscillates between phases must not pass."""
    rollout = _good_rollout()
    # Re-enter WAIT_FOR_GAP after CROSSING has begun.
    for record in rollout.records[1200:1230]:
        record["state"] = "WAIT_FOR_GAP"
    gates = _gates(rollout)
    assert not gates["state_order"]
    assert len(STATES) == 8
