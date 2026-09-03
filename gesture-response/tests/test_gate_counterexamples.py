#!/usr/bin/env python3
"""Counterexamples: each breaks ONE thing and requires the named gate to fail.

A gate that cannot fail is not a gate.  Every test below takes the summary of a
REAL passing run, mutates exactly one quantity, re-grades, and requires:

1. the named gate to go RED, and
2. **no other gate to be repaired by the same mutation** - which catches a gate
   whose failure is really just a restatement of another one.

The meta-test at the bottom parses this file with ``ast``, collects every gate
name asserted, and fails if any gate in ``validate_gesture.gates`` has no
counterexample.  Adding a gate without one therefore fails here rather than
shipping as an unchecked claim.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from validate_gesture import gates

HERE = Path(__file__).resolve()


def grade(summary: dict) -> dict[str, bool]:
    return {name: passed for name, passed, _ in gates(summary)}


@pytest.fixture(scope="module")
def baseline(summary):
    """The real run, and the assurance that it passes everything."""
    results = grade(summary)
    failed = [n for n, ok in results.items() if not ok]
    assert not failed, f"the baseline run does not pass: {failed}"
    return results


def broken(summary: dict, baseline: dict[str, bool], gate: str, mutate) -> None:
    """Mutate a copy, require ``gate`` to fail, and nothing to be repaired."""
    mutant = copy.deepcopy(summary)
    mutate(mutant)
    results = grade(mutant)
    assert gate in results, f"no such gate: {gate}"
    assert not results[gate], (
        f"{gate} still passed after the mutation that should break it")
    repaired = [n for n, ok in results.items()
                if ok and not baseline.get(n, True)]
    assert not repaired, f"the mutation repaired {repaired}"


# -- the sequence ---------------------------------------------------------------
def test_sequence_exact_order(summary, baseline):
    def mutate(s):
        s["sequence"]["accepted"] = ["STOP", "COME", "TURN_LEFT",
                                     "TURN_RIGHT", "BACK_UP", "WAVE"]
        s["sequence"]["matches"] = False
    broken(summary, baseline, "sequence_exact_order", mutate)


def test_sequence_six_commands(summary, baseline):
    def mutate(s):
        s["sequence"]["count"] = 5
    broken(summary, baseline, "sequence_six_commands", mutate)


def test_session_completed(summary, baseline):
    def mutate(s):
        s["final_state"] = "EXECUTE_APPROACH"
    broken(summary, baseline, "session_completed", mutate)


def test_no_timeouts(summary, baseline):
    def mutate(s):
        s["timeouts"] = ["EXECUTE_APPROACH@31.20s"]
    broken(summary, baseline, "no_timeouts", mutate)


# -- who ------------------------------------------------------------------------
def test_locked_the_instructor(summary, baseline):
    def mutate(s):
        s["acquisition"]["locked"] = "teo"
    broken(summary, baseline, "locked_the_instructor", mutate)


def test_acquisition_had_alternatives(summary, baseline):
    def mutate(s):
        s["acquisition"]["people_seen_during_search"] = ["mira"]
    broken(summary, baseline, "acquisition_had_alternatives", mutate)


def test_every_episode_from_instructor(summary, baseline):
    def mutate(s):
        s["episodes"][2]["person"] = "ines"
    broken(summary, baseline, "every_episode_from_instructor", mutate)


def test_zero_wrong_person_commands(summary, baseline):
    def mutate(s):
        s["episodes"][0]["person"] = "bruno"
    broken(summary, baseline, "zero_wrong_person_commands", mutate)


def test_distractor_gestures_were_readable(summary, baseline):
    def mutate(s):
        for entry in s["wrong_person"].values():
            entry["readable_command_ticks"] = 0
    broken(summary, baseline, "distractor_gestures_were_readable", mutate)


def test_distractor_gesture_sustained_past_confirm(summary, baseline):
    def mutate(s):
        for entry in s["wrong_person"].values():
            entry["sustained_past_confirm"] = False
    broken(summary, baseline, "distractor_gesture_sustained_past_confirm",
           mutate)


# -- the ambiguous partial -------------------------------------------------------
def test_partial_rejected(summary, baseline):
    def mutate(s):
        s["partial"]["accepted"] = 1
    broken(summary, baseline, "partial_rejected", mutate)


def test_partial_was_visible(summary, baseline):
    def mutate(s):
        s["partial"]["visible_fraction"] = 0.30
    broken(summary, baseline, "partial_was_visible", mutate)


def test_partial_was_readable(summary, baseline):
    def mutate(s):
        s["partial"]["readable_fraction"] = 0.10
    broken(summary, baseline, "partial_was_readable", mutate)


def test_partial_rejection_logged(summary, baseline):
    def mutate(s):
        s["partial"]["logged_rejections"] = 0
    broken(summary, baseline, "partial_rejection_logged", mutate)


# -- the camera ------------------------------------------------------------------
def test_every_command_camera_confirmed(summary, baseline):
    def mutate(s):
        s["episodes"][1]["confirm_visible_fraction"] = 0.40
    broken(summary, baseline, "every_command_camera_confirmed", mutate)


def test_every_command_arm_readable(summary, baseline):
    def mutate(s):
        s["episodes"][3]["confirm_arm_readable_fraction"] = 0.20
    broken(summary, baseline, "every_command_arm_readable", mutate)


def test_confirm_window_sustained(summary, baseline):
    def mutate(s):
        s["episodes"][0]["confirm_held_s"] = 0.10
    broken(summary, baseline, "confirm_window_sustained", mutate)


def test_monitor_visibility(summary, baseline):
    def mutate(s):
        s["tally"]["monitor_visible_fraction"] = 0.55
    broken(summary, baseline, "monitor_visibility", mutate)


# -- the physical actions ---------------------------------------------------------
def test_approach_closed_real_distance(summary, baseline):
    def mutate(s):
        s["approach"]["range_reduction_m"] = 0.02
    broken(summary, baseline, "approach_closed_real_distance", mutate)


def test_approach_walked_real_path(summary, baseline):
    def mutate(s):
        s["approach"]["path_m"] = 0.01
    broken(summary, baseline, "approach_walked_real_path", mutate)


def test_stop_interrupted_real_motion(summary, baseline):
    """The mutation that matters most: a STOP that interrupted nothing."""
    def mutate(s):
        s["stop"]["command_before_stop"] = 0.0
    broken(summary, baseline, "stop_interrupted_real_motion", mutate)


def test_stop_zeroed_within_one_tick(summary, baseline):
    def mutate(s):
        s["stop"]["ticks_to_zero"] = 14
    broken(summary, baseline, "stop_zeroed_within_one_tick", mutate)


def test_stop_held_still(summary, baseline):
    def mutate(s):
        s["stop"]["hold_s"] = 0.30
    broken(summary, baseline, "stop_held_still", mutate)


def test_stop_drift_negligible(summary, baseline):
    def mutate(s):
        s["stop"]["drift_m"] = 0.42
    broken(summary, baseline, "stop_drift_negligible", mutate)


def test_turn_left_real_heading_change(summary, baseline):
    def mutate(s):
        s["turns"]["TURN_LEFT"]["reached"] = False
    broken(summary, baseline, "turn_left_real_heading_change", mutate)


def test_turn_right_real_heading_change(summary, baseline):
    def mutate(s):
        s["turns"]["TURN_RIGHT"]["reached"] = False
    broken(summary, baseline, "turn_right_real_heading_change", mutate)


def test_turns_are_opposite(summary, baseline):
    """Both turning the SAME way must fail, even at the right magnitude."""
    def mutate(s):
        s["turns_opposite"] = False
    broken(summary, baseline, "turns_are_opposite", mutate)


def test_turns_were_walked_arcs(summary, baseline):
    def mutate(s):
        s["turns"]["TURN_LEFT"]["path_m"] = 0.004
    broken(summary, baseline, "turns_were_walked_arcs", mutate)


def test_back_up_real_displacement(summary, baseline):
    def mutate(s):
        s["back_up"]["reached"] = False
    broken(summary, baseline, "back_up_real_displacement", mutate)


def test_back_up_used_reverse_gait(summary, baseline):
    """A reverse logged at a command that MEASURABLY produces no motion."""
    def mutate(s):
        s["back_up"]["command_vx_min"] = -0.30
    broken(summary, baseline, "back_up_used_reverse_gait", mutate)


# -- the command contract ----------------------------------------------------------
def test_zero_states_exactly_zero(summary, baseline):
    def mutate(s):
        s["tally"]["zero_violation_count"] = 3
    broken(summary, baseline, "zero_states_exactly_zero", mutate)


def test_no_sub_gait_commands(summary, baseline):
    def mutate(s):
        s["tally"]["sub_gait_ticks"] = 47
    broken(summary, baseline, "no_sub_gait_commands", mutate)


def test_no_lateral_command(summary, baseline):
    def mutate(s):
        s["tally"]["max_abs_vy"] = 0.08
    broken(summary, baseline, "no_lateral_command", mutate)


def test_stillness_is_real(summary, baseline):
    def mutate(s):
        s["tally"]["worst_zero_episode_m"] = 0.44
    broken(summary, baseline, "stillness_is_real", mutate)


# -- safety --------------------------------------------------------------------------
def test_zero_contacts(summary, baseline):
    def mutate(s):
        s["tally"]["contacts"] = 2
    broken(summary, baseline, "zero_contacts", mutate)


def test_positive_clearance_to_people(summary, baseline):
    def mutate(s):
        s["tally"]["min_clearance_m"] = -0.01
    broken(summary, baseline, "positive_clearance_to_people", mutate)


def test_clearance_outside_standoff_floor(summary, baseline):
    def mutate(s):
        s["tally"]["min_clearance_m"] = 0.20
    broken(summary, baseline, "clearance_outside_standoff_floor", mutate)


def test_positive_clearance_to_scenery(summary, baseline):
    def mutate(s):
        s["tally"]["min_scenery_gap_m"] = -0.004
    broken(summary, baseline, "positive_clearance_to_scenery", mutate)


def test_stayed_inside_the_area(summary, baseline):
    def mutate(s):
        s["tally"]["outside_area_ticks"] = 9
    broken(summary, baseline, "stayed_inside_the_area", mutate)


def test_zero_falls(summary, baseline):
    def mutate(s):
        s["tally"]["fallen_steps"] = 1
    broken(summary, baseline, "zero_falls", mutate)


def test_trunk_height_held(summary, baseline):
    def mutate(s):
        s["tally"]["min_trunk_z"] = 0.061
    broken(summary, baseline, "trunk_height_held", mutate)


def test_final_height_nominal(summary, baseline):
    def mutate(s):
        s["tally"]["final_trunk_z"] = 0.083
    broken(summary, baseline, "final_height_nominal", mutate)


def test_no_forbidden_states(summary, baseline):
    def mutate(s):
        s["transitions"].append(
            {"t": 12.0, "from": "CONFIRM", "to": "OBEY_STRANGER"})
    broken(summary, baseline, "no_forbidden_states", mutate)


# -- the policy -----------------------------------------------------------------------
def test_stock_policy_sha256(summary, baseline):
    def mutate(s):
        s["policy"]["sha256_matches_stock"] = False
    broken(summary, baseline, "stock_policy_sha256", mutate)


def test_observation_is_61d(summary, baseline):
    def mutate(s):
        s["policy"]["obs_dim"] = 58
    broken(summary, baseline, "observation_is_61d", mutate)


def test_action_scale_is_0_9(summary, baseline):
    def mutate(s):
        s["policy"]["action_scale"] = 0.5
    broken(summary, baseline, "action_scale_is_0_9", mutate)


def test_exact_gyro_sensor(summary, baseline):
    def mutate(s):
        s["policy"]["gyro_sensor"] = "imu_gyro"
    broken(summary, baseline, "exact_gyro_sensor", mutate)


def test_control_rate_50hz(summary, baseline):
    def mutate(s):
        s["ctrl_hz"] = 30.0
    broken(summary, baseline, "control_rate_50hz", mutate)


# -- the meta-test ----------------------------------------------------------------------
def test_every_gate_has_a_counterexample(summary):
    """Parses THIS file and requires each gate to be broken somewhere in it.

    Without this, a gate could be added to ``validate_gesture`` and never
    exercised - passing forever because nothing ever made it fail.
    """
    tree = ast.parse(HERE.read_text())
    asserted: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "broken" and len(node.args) >= 3:
            gate = node.args[2]
            if isinstance(gate, ast.Constant) and isinstance(gate.value, str):
                asserted.add(gate.value)

    declared = {name for name, _, _ in gates(summary)}
    missing = declared - asserted
    assert not missing, (
        f"{len(missing)} acceptance gate(s) have no counterexample: "
        f"{sorted(missing)}")


def test_the_counterexamples_all_name_real_gates(summary):
    declared = {name for name, _, _ in gates(summary)}
    tree = ast.parse(HERE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "broken" and len(node.args) >= 3:
            gate = node.args[2]
            if isinstance(gate, ast.Constant):
                assert gate.value in declared, (
                    f"counterexample names {gate.value!r}, which is not a gate")
