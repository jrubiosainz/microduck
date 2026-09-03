#!/usr/bin/env python3
"""Every gate is MUTATED once and must reject the mutated run.

A gate that cannot fail is not a gate, it is a sentence.  Each test here takes
the summary of a REAL passing run, breaks exactly one thing in it, and requires
the named gate to go red — and every other gate to be untouched by that same
mutation, so a gate that fails for the wrong reason is caught too.

This is the file that would have caught the original ``the command carried no
lateral term`` gate, which compared a constant to itself and could never fail.
"""

from __future__ import annotations

import copy

import pytest

from slalom_metrics import gates


def run_gates(summary: dict) -> dict[str, bool]:
    return {label: ok for label, ok, _ in gates(summary)}


def assert_only_these_broke(baseline: dict, mutated: dict,
                            expected: str | list[str]) -> None:
    """The mutation must break the named gate(s), and nothing else may improve."""
    wanted = {expected} if isinstance(expected, str) else set(expected)
    broke = {label for label, ok in mutated.items()
             if baseline.get(label) and not ok}
    assert wanted <= broke, (
        f"expected {sorted(wanted)} to fail; actually failed {sorted(broke)}")
    fixed = {label for label, ok in mutated.items()
             if not baseline.get(label, True) and ok}
    assert not fixed, f"a mutation repaired {sorted(fixed)}"


@pytest.fixture(scope="module")
def baseline(summary):
    result = run_gates(summary)
    assert all(result.values()), (
        "the reference run must pass every gate before mutations mean "
        f"anything; failing: {[k for k, v in result.items() if not v]}")
    return result


def find(labels: dict, needle: str) -> str:
    """The single gate label containing ``needle``."""
    matches = [label for label in labels if needle in label]
    assert len(matches) == 1, f"{needle!r} matched {matches}"
    return matches[0]


# -- the journey ----------------------------------------------------------------
def test_a_short_walk_fails_the_path_gate(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["path_m"] = 1.0
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "physically walked at least"))


def test_no_net_progress_fails_the_progress_gate(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["net_m"] = 0.4
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "net progress toward"))


def test_never_reaching_the_band_fails_the_goal_gate(summary, baseline):
    """The behavior is about ARRIVING, not merely about avoiding contact."""
    broken = copy.deepcopy(summary)
    broken["reached_goal_at_s"] = None
    broken["goal_seconds"] = 0.0
    broken["min_goal_distance_m"] = 2.4
    result = run_gates(broken)
    assert not result[find(baseline, "REACHED THE GOAL BAND")]
    assert not result[find(baseline, "ended the run inside the band")]


def test_stopping_just_outside_the_band_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["min_goal_distance_m"] = 0.55
    result = run_gates(broken)
    assert not result[find(baseline, "ended the run inside the band")]


# -- the encounters --------------------------------------------------------------
def test_too_few_encounters_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["passes"] = broken["passes"][:2]
    result = run_gates(broken)
    assert not result[find(baseline, "dynamic crossing encounters")]


def test_passing_everything_on_one_hand_fails_the_alternation_gate(
        summary, baseline):
    broken = copy.deepcopy(summary)
    broken["pass_sides"] = ["right"] * len(broken["pass_sides"])
    broken["alternating"] = False
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "ALTERNATED"))


def test_never_waiting_fails_the_wait_gate(summary, baseline):
    """A duck that never refused both sides has not shown the hard case."""
    broken = copy.deepcopy(summary)
    broken["waits"] = []
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "WAIT because NEITHER"))


def test_a_zero_length_wait_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["waits"] = [dict(w) for w in broken["waits"]]
    broken["waits"][0]["duration_s"] = 0.0
    result = run_gates(broken)
    assert not result[find(baseline, "WAIT because NEITHER")]


def test_committing_below_the_planner_bar_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["passes"] = [dict(p) for p in broken["passes"]]
    broken["passes"][0]["chosen_clearance_m"] = 0.02
    result = run_gates(broken)
    assert not result[find(baseline, "positive predicted clearance")]


def test_choosing_the_worse_side_fails(summary, baseline):
    """The rejected corridor must never have scored better than the chosen one."""
    broken = copy.deepcopy(summary)
    broken["passes"] = [dict(p) for p in broken["passes"]]
    chosen = broken["passes"][0]["chosen_clearance_m"]
    broken["passes"][0]["rejected_clearance_m"] = chosen + 0.20
    result = run_gates(broken)
    assert not result[find(baseline, "REJECTED side was never better")]


def test_not_replanning_after_a_pass_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["replans_after_pass"] = 0
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "REPLANNED after every pass"))


# -- the predictions ---------------------------------------------------------------
def test_an_optimistic_prediction_fails_the_bracketing_gate(summary, baseline):
    """The planner must UNDER-promise; over-promising is the dangerous sign."""
    broken = copy.deepcopy(summary)
    broken["prediction_bracketing"] = [
        dict(b) for b in broken["prediction_bracketing"]]
    broken["prediction_bracketing"][0]["conservative"] = False
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "conservatively BRACKETED"))


def test_a_negative_measured_approach_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["prediction_bracketing"] = [
        dict(b) for b in broken["prediction_bracketing"]]
    broken["prediction_bracketing"][0]["measured_positive"] = False
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "measured closest approach during"))


# -- the turning path ----------------------------------------------------------------
def test_a_one_sided_lane_offset_fails_the_lateral_gate(summary, baseline):
    """Going only left is not a slalom, however far left it went."""
    broken = copy.deepcopy(summary)
    broken["turning_path"] = dict(broken["turning_path"])
    broken["turning_path"]["max_right_offset_m"] = 0.01
    broken["turning_path"]["lateral_span_m"] = 0.20
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "REAL lateral displacement"))


def test_a_straight_line_walk_fails_the_turning_gate(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["turning_path"] = dict(broken["turning_path"])
    broken["turning_path"]["excess_over_net_m"] = 0.01
    broken["turning_path"]["yaw_travel_deg"] = 4.0
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "TURNING path"))


def test_a_lateral_command_fails_the_no_strafe_gate(summary, baseline):
    """THE GATE THAT USED TO COMPARE A CONSTANT TO ITSELF.

    An earlier version asserted ``all(x == 0.0 for x in [0.0])`` and could never
    fail whatever the robot did.  It now reads a MEASURED per-tick maximum, so
    this mutation can break it.
    """
    broken = copy.deepcopy(summary)
    broken["max_abs_vy_command"] = 0.18
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "no lateral term"))


# -- safety -----------------------------------------------------------------------------
def test_touching_a_body_fails_the_clearance_gate(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["min_body_clearance_m"] = -0.01
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "clearance to every moving body"))


def test_clipping_a_crate_fails_the_static_gate(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["min_scenery_clearance_m"] = -0.02
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "NO PATH THROUGH ANY STATIC BODY"))


def test_any_contact_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["contact_steps"] = 1
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "zero contacts"))


# -- stillness ----------------------------------------------------------------------------
def test_a_nonzero_command_in_a_zero_state_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["zero_command_violations"] = [
        {"t": 12.0, "state": "WAIT", "command": [0.24, 0.0, 0.0]}]
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "EXACTLY zero in every"))


def test_drifting_while_waiting_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["zero_episodes"] = [dict(e) for e in broken["zero_episodes"]]
    broken["zero_episodes"][0]["path_m"] = 0.40
    broken["zero_episodes"][0]["net_m"] = 0.30
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "real standstill on the floor"))


def test_a_stall_outside_wait_fails(summary, baseline):
    """A duck stopped in ADVANCE has stalled, however good its per-state max."""
    broken = copy.deepcopy(summary)
    broken["longest_illegal_zero_run"] = 400
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "ZERO-COMMAND PLATEAU"))


# -- visibility -----------------------------------------------------------------------------
def test_not_watching_the_body_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["monitor_visible_fraction_with_los"] = 0.10
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "negotiated body was visible"))


def test_not_seeing_the_goal_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["goal_visible_fraction_with_los"] = 0.10
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "THE GOAL was visible"))


# -- the scenario ------------------------------------------------------------------------------
def test_an_empty_course_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["obstacle_and_actor_count"] = 3
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "obstacles and actors populate"))


def test_static_traffic_fails(summary, baseline):
    """Traffic that does not move makes every avoidance claim vacuous."""
    broken = copy.deepcopy(summary)
    broken["moving_actors"] = 1
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "actors were genuinely moving"))


def test_a_teleporting_body_fails_the_continuity_gate(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["max_actor_heading_step_deg"] = 51.0
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "heading is continuous"))


def test_traffic_that_never_got_in_the_way_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["max_bodies_in_lane"] = 0
    broken["lane_occupied_seconds"] = 0.0
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "traffic actually got in the way"))


# -- the states and the contract ---------------------------------------------------------------
def test_a_missing_state_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["states_visited"] = [s for s in broken["states_visited"]
                                if s != "WAIT"]
    result = run_gates(broken)
    assert not result[find(baseline, "every declared state was visited")]


def test_a_forbidden_state_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["forbidden_state_steps"] = {"BARGE_THROUGH": 3,
                                       "FREEZE_FOREVER": 0}
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "BARGE_THROUGH"))


def test_a_phase_ceiling_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["timeouts"] = ["PASS@60.00s"]
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "no phase hit its ceiling"))


def test_a_fall_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["fallen_steps"] = 2
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "zero falls"))


def test_the_wrong_sensor_fails_the_contract_gate(summary, baseline):
    """A different quantity in the base_ang_vel slot invalidates everything."""
    broken = copy.deepcopy(summary)
    broken["gyro_sensor"] = "imu_lin_acc"
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "exact imu_ang_vel"))


def test_a_wrong_observation_width_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["observation_dim"] = 60
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "exact imu_ang_vel"))


def test_a_different_policy_fails(summary, baseline):
    broken = copy.deepcopy(summary)
    broken["policy_sha256"] = "0" * 64
    assert_only_these_broke(baseline, run_gates(broken),
                            find(baseline, "byte-identical stock walking"))
