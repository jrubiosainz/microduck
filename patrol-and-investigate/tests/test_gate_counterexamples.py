#!/usr/bin/env python3
"""Gate counterexamples: each takes the summary of a REAL passing run, breaks
exactly ONE thing, and requires the named gate to go red.

A GATE THAT CANNOT FAIL IS NOT A GATE.  It is very easy to write an acceptance
check that passes because of how it is phrased rather than because of what the
robot did - a gate comparing a constant to itself, a gate reading the
implementation it is grading, a gate whose evidence string is the only thing
that changes.  Every gate in ``patrol_metrics`` is therefore mutated here.

TWO PROPERTIES ARE REQUIRED OF EACH MUTATION, not one:

* the NAMED gate must go red - the mutation is detected;
* NO OTHER GATE may be REPAIRED by the same mutation - a mutation that turns one
  gate red while turning another green is measuring something confused.

The mutations are applied to a deep copy, so the shared summary fixture is never
damaged for a later test.
"""

from __future__ import annotations

import copy

import pytest

from patrol_metrics import gates


def _passing(summary):
    passed = {label for label, ok, _ in gates(summary) if ok}
    assert passed, "the baseline summary must pass at least one gate"
    return passed


def assert_gate_fails(summary, mutate, needle: str):
    """Mutate one thing; require the gate NAMED BY ``needle`` to go red.

    Also requires that no gate which was failing becomes passing, so a mutation
    cannot quietly repair something while breaking the thing under test.
    """
    before = {label: ok for label, ok, _ in gates(summary)}
    assert all(before.values()), \
        f"baseline must pass every gate; failing: " \
        f"{[k for k, v in before.items() if not v]}"

    broken = copy.deepcopy(summary)
    mutate(broken)
    after = {label: ok for label, ok, _ in gates(broken)}

    matched = [label for label in after if needle in label]
    assert matched, f"no gate matched {needle!r}; have {list(after)}"
    assert any(not after[label] for label in matched), (
        f"the mutation did not break any gate matching {needle!r}: "
        f"{[(m, after[m]) for m in matched]}")

    repaired = [label for label in after
                if after[label] and not before.get(label, True)]
    assert not repaired, f"the mutation REPAIRED {repaired}"


# -- A: the patrol was a patrol -------------------------------------------------
def test_a_reordered_checkpoint_sequence_is_caught(summary):
    def mutate(s):
        s["checkpoint_visited_order"] = ["north-bay", "dock-gate", "east-aisle",
                                         "server-door", "west-stair"]
        s["checkpoint_in_declared_order"] = False
    assert_gate_fails(summary, mutate, "DECLARED ORDER")


def test_a_missing_checkpoint_is_caught(summary):
    def mutate(s):
        s["checkpoint_visited_order"] = s["checkpoint_visited_order"][:4]
        s["checkpoint_count"] = 4
        s["checkpoint_all_visited"] = False
        s["checkpoint_in_declared_order"] = False
    assert_gate_fails(summary, mutate, "DECLARED ORDER")


def test_a_repeated_checkpoint_is_caught(summary):
    def mutate(s):
        s["checkpoint_no_repeats"] = False
    assert_gate_fails(summary, mutate, "DECLARED ORDER")


def test_stopping_short_of_a_checkpoint_is_caught(summary):
    def mutate(s):
        s["visits"][0]["arrival_error_m"] = 0.85
    assert_gate_fails(summary, mutate, "stopped ON each checkpoint")


def test_never_getting_home_is_caught(summary):
    def mutate(s):
        s["reached_home_at_s"] = None
    assert_gate_fails(summary, mutate, "FULL LOOP")


def test_stopping_far_from_the_guard_post_is_caught(summary):
    def mutate(s):
        s["min_home_distance_m"] = 1.4
    assert_gate_fails(summary, mutate, "FULL LOOP")


def test_not_walking_far_enough_is_caught(summary):
    def mutate(s):
        s["path_m"] = 2.0
    assert_gate_fails(summary, mutate, "physically walked")


# -- B: the stops and the scans ---------------------------------------------------
def test_a_checkpoint_stop_that_was_too_short_is_caught(summary):
    def mutate(s):
        s["visits"][2]["stopped_s"] = 0.1
        s["scan_min_stopped_s"] = 0.1
    assert_gate_fails(summary, mutate, "REAL STOP")


def test_drifting_during_a_checkpoint_stop_is_caught(summary):
    def mutate(s):
        s["scan_max_still_path_m"] = 0.9
    assert_gate_fails(summary, mutate, "REAL STOP")


def test_a_scan_that_barely_moved_the_head_is_caught(summary):
    def mutate(s):
        s["scan_completed_scan_arcs_deg"] = [4.0, 4.0, 4.0]
        s["scan_min_completed_scan_arc_deg"] = 4.0
    assert_gate_fails(summary, mutate, "COMPLETED scan swept")


def test_a_scan_that_resolved_nobody_at_all_is_caught(summary):
    def mutate(s):
        for scan in s["scan_scans"]:
            scan["bodies_seen"] = []
    assert_gate_fails(summary, mutate, "resolved real bodies")


# -- C: the anomalies ---------------------------------------------------------------
def test_a_wrong_verdict_is_caught(summary):
    def mutate(s):
        s["verdict_verdict_by_target"]["crate"] = "benign"
        s["verdict_all_correct"] = False
    assert_gate_fails(summary, mutate, "EVERY verdict was correct")


def test_investigating_the_benign_distractor_is_caught(summary):
    """The distractor must be DISMISSED, not chased."""
    def mutate(s):
        s["verdict_investigated"] = ["crate", "visitor", "trolley"]
        s["verdict_dismissed"] = []
    assert_gate_fails(summary, mutate, "DISTINCT anomalies were investigated")


def test_failing_to_dismiss_the_distractor_is_caught(summary):
    def mutate(s):
        s["verdict_dismissed"] = []
    assert_gate_fails(summary, mutate, "EXPLICITLY DISMISSED")


def test_a_dismissal_with_no_recorded_rule_is_caught(summary):
    """Silence is not a dismissal."""
    def mutate(s):
        for verdict in s["verdict_verdicts"]:
            if verdict["verdict"] == "benign":
                verdict["rule"] = ""
    assert_gate_fails(summary, mutate, "EXPLICITLY DISMISSED")


def test_detecting_something_outside_the_camera_gate_is_caught(summary):
    """THE GATE THAT MAKES DETECTION A PERCEPTION CLAIM."""
    def mutate(s):
        s["verdict_camera_gate_ticks"]["crate"] = 1
    assert_gate_fails(summary, mutate, "INSIDE the camera gate")


def test_only_one_anomaly_investigated_is_caught(summary):
    def mutate(s):
        s["verdict_investigated"] = ["crate"]
    assert_gate_fails(summary, mutate, "DISTINCT anomalies were investigated")


# -- D: the investigations were physical -----------------------------------------------
def test_an_approach_that_did_not_close_the_range_is_caught(summary):
    """A body already standing at its own observation distance is not an
    approach."""
    def mutate(s):
        s["standoff_investigations"][0]["range_reduction_m"] = 0.02
    assert_gate_fails(summary, mutate, "reduced the range")


def test_an_approach_with_no_walking_in_it_is_caught(summary):
    def mutate(s):
        s["standoff_investigations"][0]["approach_path_m"] = 0.01
    assert_gate_fails(summary, mutate, "REAL WALK")


def test_stopping_too_close_to_the_target_is_caught(summary):
    def mutate(s):
        s["standoff_investigations"][0]["min_clearance_m"] = 0.12
    assert_gate_fails(summary, mutate, "safe observation standoff band")


def test_stopping_too_far_to_have_observed_anything_is_caught(summary):
    def mutate(s):
        s["standoff_investigations"][0]["min_clearance_m"] = 2.4
    assert_gate_fails(summary, mutate, "safe observation standoff band")


def test_holding_fewer_than_the_declared_angles_is_caught(summary):
    def mutate(s):
        entry = s["standoff_investigations"][0]
        entry["angles_held"] = 1
        entry["observations"] = entry["observations"][:1]
    assert_gate_fails(summary, mutate, "declared viewing angles")


def test_an_observation_angle_with_the_target_out_of_frame_is_caught(summary):
    def mutate(s):
        s["standoff_investigations"][0]["observations"][1][
            "visible_fraction"] = 0.05
    assert_gate_fails(summary, mutate, "declared viewing angles")


def test_a_nonzero_command_during_the_observation_is_caught(summary):
    def mutate(s):
        s["state_command_max"]["OBSERVE"] = 0.34
    assert_gate_fails(summary, mutate, "EXACTLY zero throughout every "
                                       "observation")


# -- E: the route memory -------------------------------------------------------------
def test_resuming_toward_the_wrong_checkpoint_is_caught(summary):
    """THE GATE THE WHOLE BEHAVIOR TURNS ON."""
    def mutate(s):
        s["memory_all_preserved"] = False
        s["memory_interruptions"][0]["route_preserved"] = False
        s["memory_interruptions"][0]["resumed_target_name"] = "west-stair"
    assert_gate_fails(summary, mutate, "original checkpoint was PRESERVED")


def test_never_interrupting_the_patrol_is_caught(summary):
    def mutate(s):
        s["memory_count"] = 0
        s["memory_interruptions"] = []
    assert_gate_fails(summary, mutate, "INTERRUPTED")


def test_not_returning_to_the_interruption_point_is_caught(summary):
    def mutate(s):
        s["memory_max_return_error_m"] = 1.9
    assert_gate_fails(summary, mutate, "physically RETURNED")


# -- F: nothing touched or entered -------------------------------------------------
def test_entering_the_restricted_zone_is_caught(summary):
    def mutate(s):
        s["min_zone_gap_m"] = -0.04
        s["zone_breach_steps"] = 12
    assert_gate_fails(summary, mutate, "NEVER ENTERED THE RESTRICTED ZONE")


def test_a_single_breach_tick_is_caught(summary):
    """A rule is not satisfied 'mostly'."""
    def mutate(s):
        s["zone_breach_steps"] = 1
    assert_gate_fails(summary, mutate, "NEVER ENTERED THE RESTRICTED ZONE")


def test_touching_a_body_is_caught(summary):
    def mutate(s):
        s["min_body_clearance_m"] = -0.01
    assert_gate_fails(summary, mutate, "NO CONTACT")


def test_walking_through_a_fixture_is_caught(summary):
    def mutate(s):
        s["min_scenery_clearance_m"] = -0.02
    assert_gate_fails(summary, mutate, "every fixture and wall")


def test_a_contact_step_is_caught(summary):
    def mutate(s):
        s["contact_steps"] = 3
    assert_gate_fails(summary, mutate, "zero contacts")


# -- G: stillness and no decorative commands -----------------------------------------
def test_a_nonzero_command_in_a_zero_state_is_caught(summary):
    def mutate(s):
        s["zero_command_violations"] = [
            {"t": 12.0, "state": "SCAN", "command": [0.22, 0.0, 0.0]}]
    assert_gate_fails(summary, mutate, "EXACTLY zero in every zero-command")


def test_a_sub_onset_command_hidden_in_a_zero_state_is_caught(summary):
    """MEASURED: vx=0.22 moves 0.009 m in 6 s.  It would look like care and
    produce nothing, so the gate is on the EXACT zero, not on the motion."""
    def mutate(s):
        s["state_command_max"]["SCAN"] = 0.22
    assert_gate_fails(summary, mutate, "EXACTLY zero in every zero-command")


def test_drifting_during_a_zero_command_episode_is_caught(summary):
    def mutate(s):
        s["zero_episodes"][0]["path_m"] = 0.9
    assert_gate_fails(summary, mutate, "real standstill")


def test_a_stall_outside_the_permitted_states_is_caught(summary):
    def mutate(s):
        s["longest_illegal_zero_run"] = 900
    assert_gate_fails(summary, mutate, "ZERO-COMMAND PLATEAU")


def test_a_lateral_command_is_caught(summary):
    """There is no strafe on this policy; a vy term is a yaw disturbance."""
    def mutate(s):
        s["max_abs_vy_command"] = 0.05
    assert_gate_fails(summary, mutate, "NO DECORATIVE COMMANDS")


# -- H: it could see -------------------------------------------------------------------
def test_a_camera_that_was_not_active_is_caught(summary):
    def mutate(s):
        s["camera_active_fraction"] = 0.42
    assert_gate_fails(summary, mutate, "camera was ACTIVE")


def test_losing_the_investigated_body_is_caught(summary):
    def mutate(s):
        s["monitor_visible_fraction_with_los"] = 0.31
    assert_gate_fails(summary, mutate, "visible in >=")


# -- I: the facility is real ------------------------------------------------------------
def test_an_empty_facility_is_caught(summary):
    def mutate(s):
        s["body_count"] = 2
    assert_gate_fails(summary, mutate, "populated")


def test_a_facility_where_nobody_moves_is_caught(summary):
    def mutate(s):
        s["moving_bodies"] = 0
    assert_gate_fails(summary, mutate, "genuinely moving")


def test_a_teleporting_actor_is_caught(summary):
    def mutate(s):
        s["max_actor_heading_step_deg"] = 51.0
    assert_gate_fails(summary, mutate, "heading is continuous")


def test_a_second_person_in_the_zone_is_caught(summary):
    """If staff wandered into the restricted area, the intrusion call would be
    about the area rather than about the person."""
    def mutate(s):
        s["zone_occupancy_s"]["rosa"] = 14.0
    assert_gate_fails(summary, mutate, "exactly ONE person entered")


def test_naming_the_wrong_person_as_the_intruder_is_caught(summary):
    def mutate(s):
        s["verdict_verdict_by_target"]["visitor"] = "benign"
    assert_gate_fails(summary, mutate, "exactly ONE person entered")


# -- J: the states --------------------------------------------------------------------
def test_a_state_that_never_ran_is_caught(summary):
    def mutate(s):
        s["states_visited"] = [x for x in s["states_visited"]
                               if x != "OBSERVE"]
    assert_gate_fails(summary, mutate, "every declared state was visited")


def test_an_undeclared_state_is_caught(summary):
    def mutate(s):
        s["states_visited"] = sorted(s["states_visited"] + ["IMPROVISE"])
    assert_gate_fails(summary, mutate, "state is one this behavior declares")


def test_a_forbidden_state_is_caught(summary):
    def mutate(s):
        s["forbidden_state_steps"]["ABANDON_PATROL"] = 7
    assert_gate_fails(summary, mutate, "ABANDON_PATROL")


def test_a_phase_hitting_its_ceiling_is_caught(summary):
    def mutate(s):
        s["timeouts"] = ["APPROACH@61.20s"]
    assert_gate_fails(summary, mutate, "ceiling")


# -- K: locomotion health ----------------------------------------------------------------
def test_a_fall_is_caught(summary):
    def mutate(s):
        s["fallen_steps"] = 5
    assert_gate_fails(summary, mutate, "zero falls")


def test_a_collapsed_trunk_is_caught(summary):
    def mutate(s):
        s["min_trunk_z_m"] = 0.04
    assert_gate_fails(summary, mutate, "trunk never below")


def test_finishing_slumped_is_caught(summary):
    def mutate(s):
        s["final_trunk_z_m"] = 0.06
    assert_gate_fails(summary, mutate, "final trunk height")


# -- L: the physical contract ---------------------------------------------------------------
def test_a_different_sensor_is_caught(summary):
    def mutate(s):
        s["gyro_sensor"] = "imu_acc"
    assert_gate_fails(summary, mutate, "imu_ang_vel")


def test_a_different_observation_width_is_caught(summary):
    def mutate(s):
        s["observation_dim"] = 48
    assert_gate_fails(summary, mutate, "imu_ang_vel")


def test_a_different_action_scale_is_caught(summary):
    def mutate(s):
        s["action_scale"] = 0.5
    assert_gate_fails(summary, mutate, "imu_ang_vel")


def test_a_modified_policy_is_caught(summary):
    def mutate(s):
        s["policy_sha256"] = "0" * 64
    assert_gate_fails(summary, mutate, "stock walking policy")


# -- the suite's own integrity ------------------------------------------------------------
def test_the_baseline_run_passes_every_gate(summary):
    failing = [label for label, ok, _ in gates(summary) if not ok]
    assert not failing, failing


def test_every_gate_has_a_counterexample_in_this_file(summary):
    """A gate nobody mutated is a gate nobody proved can fail.

    The needles are read out of THIS FILE with ``ast`` - every string literal
    passed as the third argument to :func:`assert_gate_fails` - and every gate
    label must be matched by at least one of them.  Adding a gate without a
    counterexample therefore fails here rather than shipping, and the check
    cannot rot into a hand-kept list that drifts from the mutations actually
    written.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(__file__).read_text())
    needles: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (getattr(node.func, "id", None)
                or getattr(node.func, "attr", None))
        if name != "assert_gate_fails" or len(node.args) < 3:
            continue
        needle = node.args[2]
        if isinstance(needle, ast.Constant) and isinstance(needle.value, str):
            needles.append(needle.value)

    assert len(needles) >= 40, f"only {len(needles)} mutations found"

    labels = [label for label, _, _ in gates(summary)]
    uncovered = [label for label in labels
                 if not any(needle in label for needle in needles)]
    assert not uncovered, f"gates with no counterexample: {uncovered}"
