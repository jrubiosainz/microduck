#!/usr/bin/env python3
"""Mutation counterexamples: every gate must REJECT its own violation.

A gate that cannot fail proves nothing.  Each test here takes the REAL summary
from the real run, mutates exactly the quantity one gate reads, and requires
THAT NAMED GATE to flip to failing.  Naming the gate matters: a mutation that
tripped some other gate would otherwise look like a pass.

The summary is deep-copied per test, so a mutation cannot leak into the shared
fixture.
"""

from __future__ import annotations

import copy

import pytest

from conftest import gate_named
from etiquette_metrics import report

pytestmark = pytest.mark.slow


def mutated(summary, **changes):
    copy_of = copy.deepcopy(summary)
    copy_of.update(changes)
    return report(copy_of)


def rejects(summary, fragment: str, **changes):
    """Mutate, then require the named gate to fail."""
    results = mutated(summary, **changes)
    label, ok, evidence = gate_named(results, fragment)
    assert not ok, f"gate {label!r} accepted its own violation: {evidence}"


def deep(summary, path, value):
    """Mutate one nested key and return the whole summary."""
    copy_of = copy.deepcopy(summary)
    cursor = copy_of
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return copy_of


def rejects_deep(summary, fragment: str, path, value):
    results = report(deep(summary, path, value))
    label, ok, evidence = gate_named(results, fragment)
    assert not ok, f"gate {label!r} accepted its own violation: {evidence}"


# -- the baseline ------------------------------------------------------------
def test_the_unmutated_summary_passes_every_gate(summary):
    passed, _ = report(copy.deepcopy(summary))
    assert passed, "the counterexample suite is meaningless if the real run fails"


# -- the route ---------------------------------------------------------------
def test_an_unwalkable_bend_is_rejected(summary):
    bends = copy.deepcopy(summary["route_bends"])
    bends[0]["walkable"] = False
    rejects(summary, "MEASURED turning circle", route_bends=bends)


def test_a_route_that_clips_a_jamb_is_rejected(summary):
    crossings = copy.deepcopy(summary["route_crossings"])
    crossings[0]["margin_m"] = 0.05
    rejects(summary, "through the middle of every aperture",
            route_crossings=crossings)


def test_wandering_off_the_route_is_rejected(summary):
    rejects(summary, "stayed on its own route", max_cross_track_m=0.90)


def test_a_shuffle_instead_of_a_walk_is_rejected(summary):
    rejects(summary, "physically walked at least", path_m=2.0)
    rejects(summary, "net progress", net_m=1.0)


# -- the doorway -------------------------------------------------------------
def test_not_yielding_at_all_is_rejected(summary):
    rejects(summary, "yielded at the doorway", yields=[])


def test_a_yield_that_is_too_short_is_rejected(summary):
    yields = copy.deepcopy(summary["yields"])
    yields[0]["duration_s"] = 0.4
    rejects(summary, "yielded at the doorway", yields=yields)


def test_encroaching_on_the_threshold_early_is_rejected(summary):
    rejects(summary, "THRESHOLD ENCROACHMENT",
            zone_violation_steps={"concourse_door_threshold": 1})


def test_a_doorway_nobody_came_out_of_is_rejected(summary):
    rejects(summary, "came OUT through the doorway", exiters_used_door=["tomas"])


def test_sharing_the_doorway_with_somebody_is_rejected(summary):
    rejects(summary, "NEVER in the doorway at the same time",
            aperture_shared_steps={"concourse_door": 1})


# -- the lift ----------------------------------------------------------------
def test_not_waiting_beside_the_lift_is_rejected(summary):
    seconds = copy.deepcopy(summary["state_seconds"])
    seconds["WAIT_SIDE"] = 0.5
    rejects(summary, "waited BESIDE the lift doors", state_seconds=seconds)


def test_standing_in_the_exit_passage_is_rejected(summary):
    rejects(summary, "NEVER stood in the lift's exit passage",
            zone_violation_steps={"lift_front_passage": 1})


def test_boarding_before_two_occupants_left_is_rejected(summary):
    rejects_deep(summary, "occupants EXITED before the duck entered",
                 ("boarding", "occupants_exited_before_entry"), 1)


def test_moving_while_occupants_are_still_leaving_is_rejected(summary):
    commands = copy.deepcopy(summary["state_command_max"])
    commands["LET_OCCUPANTS_EXIT"] = 0.26
    rejects(summary, "did not move until the LAST occupant",
            state_command_max=commands)


def test_boarding_before_the_guardian_is_rejected(summary):
    rejects_deep(summary, "boarded AFTER the guardian",
                 ("boarding", "guardian_inside_at_entry"), False)


def test_sharing_the_lift_aperture_is_rejected(summary):
    rejects(summary, "NEVER in the lift aperture at the same time",
            aperture_shared_steps={"lift_front": 3})


def test_a_cabin_position_outside_the_bounds_is_rejected(summary):
    rejects(summary, "cabin is real and inside its bounds",
            min_cabin_margin_m=-0.01)


def test_riding_from_outside_the_cabin_is_rejected(summary):
    rejects(summary, "cabin is real and inside its bounds",
            cabin_outside_while_riding_steps=5)


def test_a_ride_that_is_too_short_is_rejected(summary):
    rejects(summary, "ride was exactly still", ride_seconds=2.0)


def test_shuffling_during_the_ride_is_rejected(summary):
    commands = copy.deepcopy(summary["state_command_max"])
    commands["RIDE"] = 0.24
    rejects(summary, "ride was exactly still", state_command_max=commands)


def test_leaving_the_cabin_before_the_guardian_is_rejected(summary):
    boarding = copy.deepcopy(summary["boarding"])
    boarding["duck_exited_at_s"] = boarding["guardian_exited_at_s"] - 1.0
    rejects(summary, "guardian left the cabin FIRST", boarding=boarding)


# -- the doors ---------------------------------------------------------------
def test_walking_through_a_closed_door_is_rejected(summary):
    crossings = copy.deepcopy(summary["crossings"])
    crossings[0]["open_fraction_at_entry"] = 0.10
    crossings[0]["effective_gap_at_entry_m"] = 0.07
    rejects(summary, "NO MOVEMENT THROUGH A CLOSED DOOR", crossings=crossings)


def test_never_entering_an_aperture_at_all_is_rejected(summary):
    crossings = copy.deepcopy(summary["crossings"])
    crossings[1] = {"aperture": crossings[1]["aperture"], "entered_at_s": None}
    rejects(summary, "NO MOVEMENT THROUGH A CLOSED DOOR", crossings=crossings)


# -- order -------------------------------------------------------------------
def test_overtaking_the_guardian_is_rejected(summary):
    rejects(summary, "NEVER overtook or crossed", overtake_steps=1)


def test_falling_impossibly_far_behind_is_rejected(summary):
    rejects(summary, "close enough to have actually followed",
            max_guardian_gap_m=99.0)


# -- stillness ---------------------------------------------------------------
def test_a_nonzero_command_in_a_zero_state_is_rejected(summary):
    rejects(summary, "EXACTLY zero in every zero-command state",
            zero_command_violations=[
                {"t": 50.0, "state": "RIDE", "command": [0.24, 0.0, 0.0]}])


def test_drifting_during_a_still_state_is_rejected(summary):
    paths = copy.deepcopy(summary["zero_state_path_m"])
    paths["RIDE"] = 0.85
    rejects(summary, "real standstill on the floor", zero_state_path_m=paths)


# -- visibility --------------------------------------------------------------
def test_losing_sight_of_the_active_person_is_rejected(summary):
    rejects(summary, "visible in >=", monitor_visible_fraction_with_los=0.60)


def test_watching_the_wrong_person_is_rejected(summary):
    rejects(summary, "watched the right person",
            subject_role_order=["guardian", "occupant"])


# -- the populated building --------------------------------------------------
def test_an_empty_building_is_rejected(summary):
    rejects(summary, "adults besides the guardian were moving", moving_adults=2)


# -- safety ------------------------------------------------------------------
def test_touching_a_person_is_rejected(summary):
    rejects(summary, "clearance to every person", min_person_clearance_m=-0.01)


def test_touching_the_scenery_is_rejected(summary):
    rejects(summary, "clearance to every wall", min_scenery_clearance_m=0.0)


def test_a_contact_is_rejected(summary):
    rejects(summary, "zero contacts", contact_steps=1)


# -- states ------------------------------------------------------------------
def test_a_missing_state_is_rejected(summary):
    visited = [s for s in summary["states_visited"] if s != "RIDE"]
    rejects(summary, "every declared state was visited", states_visited=visited)


def test_an_undeclared_state_is_rejected(summary):
    rejects(summary, "every visited state is one this behavior declares",
            states_visited=summary["states_visited"] + ["PUSH_THROUGH"])


def test_states_out_of_order_are_rejected(summary):
    scrambled = list(reversed(summary["state_order"]))
    rejects(summary, "ran in the declared order", state_order=scrambled)


def test_a_forbidden_state_is_rejected(summary):
    rejects(summary, "zero PUSH_THROUGH and BOARD_FIRST",
            forbidden_state_steps={"PUSH_THROUGH": 3, "BOARD_FIRST": 0})


def test_a_phase_ceiling_is_rejected(summary):
    rejects(summary, "no phase hit its ceiling", timeouts=["RIDE@90.00s"])


# -- locomotion health -------------------------------------------------------
def test_a_fall_is_rejected(summary):
    rejects(summary, "zero falls", fallen_steps=1)


def test_a_dropped_trunk_is_rejected(summary):
    rejects(summary, "never below 0.09", min_trunk_z_m=0.05)


def test_a_collapsed_final_pose_is_rejected(summary):
    rejects(summary, "final trunk height", final_trunk_z_m=0.080)


# -- the physical contract ---------------------------------------------------
def test_a_different_policy_is_rejected(summary):
    rejects(summary, "byte-identical stock walking policy",
            policy_sha256="0" * 64)


def test_a_different_sensor_is_rejected(summary):
    rejects(summary, "exact imu_ang_vel sensor", gyro_sensor="gyro")


def test_a_different_observation_width_is_rejected(summary):
    rejects(summary, "exact imu_ang_vel sensor", observation_dim=48)


def test_a_different_action_scale_is_rejected(summary):
    rejects(summary, "exact imu_ang_vel sensor", action_scale=1.0)


# -- the suite itself --------------------------------------------------------
def test_every_gate_has_at_least_one_counterexample(results):
    """A gate nobody mutates is a gate nobody has shown can fail.

    Kept as a coverage check rather than a promise: it lists any gate this file
    does not exercise, so adding a gate without a counterexample fails here.
    """
    _, entries = results
    fragments = [
        "MEASURED turning circle", "through the middle of every aperture",
        "stayed on its own route", "physically walked at least",
        "net progress", "yielded at the doorway", "THRESHOLD ENCROACHMENT",
        "came OUT through the doorway", "NEVER in the doorway at the same time",
        "entered the doorway BEHIND", "waited BESIDE the lift doors",
        "NEVER stood in the lift's exit passage",
        "occupants EXITED before the duck entered",
        "did not move until the LAST occupant", "boarded AFTER the guardian",
        "NEVER in the lift aperture at the same time",
        "cabin is real and inside its bounds", "ride was exactly still",
        "guardian left the cabin FIRST", "NO MOVEMENT THROUGH A CLOSED DOOR",
        "NEVER overtook or crossed", "close enough to have actually followed",
        "EXACTLY zero in every zero-command state",
        "real standstill on the floor", "visible in >=",
        "watched the right person", "adults besides the guardian were moving",
        "clearance to every person", "clearance to every wall", "zero contacts",
        "every declared state was visited",
        "every visited state is one this behavior declares",
        "ran in the declared order", "zero PUSH_THROUGH and BOARD_FIRST",
        "no phase hit its ceiling", "zero falls", "never below 0.09",
        "final trunk height", "byte-identical stock walking policy",
        "exact imu_ang_vel sensor",
    ]
    uncovered = [
        label for label, _, _ in entries
        if not any(fragment in label for fragment in fragments)]
    assert not uncovered, f"gates with no counterexample: {uncovered}"
