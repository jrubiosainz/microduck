#!/usr/bin/env python3
"""Invariants of the REAL rollout: what actually happened over 110 s of physics.

These share one session-scoped rollout, because the behavior is deterministic
and a second run would cost minutes to produce byte-identical records.  Marked
slow so the pure-logic suites can be run alone during development.
"""

from __future__ import annotations

import numpy as np
import pytest

from etiquette_states import (
    DUCK_PLANAR_RADIUS,
    STATES,
    WALKING_STATES,
    ZERO_COMMAND_STATES,
)
from etiquette_thresholds import (
    MIN_OCCUPANTS_OUT,
    MIN_OPEN_FRACTION_AT_CROSSING,
    ZERO_STATE_PATH_M,
)

pytestmark = pytest.mark.slow


# -- the gate as a whole -----------------------------------------------------
def test_every_acceptance_gate_passes(results):
    passed, entries = results
    failed = [(label, evidence) for label, ok, evidence in entries if not ok]
    assert passed, "failed gates:\n" + "\n".join(
        f"  {label}: {evidence}" for label, evidence in failed)


def test_the_gate_is_not_trivially_small(results):
    _, entries = results
    assert len(entries) >= 30, f"only {len(entries)} gates"


# -- the states --------------------------------------------------------------
def test_every_declared_state_was_visited(summary):
    assert set(STATES) <= set(summary["states_visited"])


def test_the_states_ran_in_the_declared_order(summary):
    assert summary["state_order"] == list(STATES[1:])


def test_no_phase_hit_its_ceiling(summary):
    assert summary["timeouts"] == []


def test_the_run_finished(rollout):
    assert rollout.machine.state == "DONE"
    assert rollout.machine.boarded


# -- stillness ---------------------------------------------------------------
def test_the_command_was_EXACTLY_zero_in_every_zero_command_state(rollout):
    for record in rollout.records:
        if record["state"] in ZERO_COMMAND_STATES:
            assert record["command"] == [0.0, 0.0, 0.0], record["t"]


def test_every_zero_command_state_was_a_standstill_on_the_floor(summary):
    for state, path in summary["zero_state_path_m"].items():
        assert path <= ZERO_STATE_PATH_M, (state, path)


def test_the_ride_is_the_stillest_claim_in_the_run(summary):
    assert summary["state_command_max"]["RIDE"] == 0.0
    assert summary["ride_seconds"] >= 8.0
    assert summary["state_path_m"]["RIDE"] <= ZERO_STATE_PATH_M


# -- the zones ---------------------------------------------------------------
def test_no_threshold_encroachment_before_the_exiters_cleared(summary):
    assert summary["zone_violation_steps"].get(
        "concourse_door_threshold", 0) == 0


def test_the_duck_never_stood_in_the_lift_exit_passage_before_boarding(summary):
    assert summary["zone_violation_steps"].get("lift_front_passage", 0) == 0


def test_the_zone_bookkeeping_is_not_vacuous(summary):
    """The duck must have been in each zone AT SOME POINT, or the gate is empty."""
    for name in ("concourse_door_threshold", "lift_front_passage",
                 "lift_front_threshold"):
        entry = summary["zone_worst"][name]
        assert entry["steps"] > 0, f"{name} was never entered at all"
        assert entry["worst_m"] > 0.0


# -- the apertures -----------------------------------------------------------
def test_the_duck_never_shared_an_aperture_with_anybody(summary):
    assert summary["aperture_shared_steps"] == {}
    assert summary["aperture_shared_with"] == {}


def test_the_duck_really_did_pass_through_all_three_apertures(summary):
    for name in ("concourse_door", "lift_front", "lift_rear"):
        assert summary["aperture_steps"].get(name, 0) > 50, name


def test_no_movement_through_a_closed_door(summary):
    for crossing in summary["crossings"]:
        assert crossing["entered_at_s"] is not None, crossing
        assert crossing["open_fraction_at_entry"] >= \
            MIN_OPEN_FRACTION_AT_CROSSING, crossing


def test_the_duck_was_never_inside_a_closed_aperture_at_ANY_tick(rollout):
    """Stronger than the crossing record: checked per tick, not at first entry."""
    for record in rollout.records:
        for name, entry in record["aperture_occupancy"].items():
            if entry["duck"]:
                assert record["door_fraction"][name] >= 0.5, (record["t"], name)


# -- order relative to the guardian -----------------------------------------
def test_the_duck_never_overtook_the_guardian(summary):
    assert summary["overtake_steps"] == 0
    assert summary["min_guardian_gap_m"] > 0.0


def test_the_order_measurement_has_a_real_denominator(summary):
    assert summary["guardian_gap_samples"] > 3000


def test_the_duck_boarded_after_her_and_left_after_her(summary):
    boarding = summary["boarding"]
    assert boarding["guardian_inside_at_entry"] is True
    assert boarding["guardian_exited_at_s"] < boarding["duck_exited_at_s"]


# -- the lift ----------------------------------------------------------------
def test_at_least_two_occupants_exited_before_the_duck_entered(summary):
    assert summary["boarding"]["occupants_exited_before_entry"] >= \
        MIN_OCCUPANTS_OUT


def test_all_three_occupants_actually_used_the_lift_aperture(summary):
    assert len(summary["occupants_used_lift"]) == 3


def test_the_cabin_position_is_real_and_inside_the_bounds(summary):
    assert summary["min_cabin_margin_m"] > 0.0
    assert summary["cabin_seconds"] >= 8.0
    assert summary["cabin_outside_while_riding_steps"] == 0


def test_the_duck_was_inside_the_cabin_for_every_RIDE_tick(rollout):
    for record in rollout.records:
        if record["state"] == "RIDE":
            assert record["inside_cabin"], record["t"]
            assert record["cabin_margin_m"] > 0.0


# -- safety ------------------------------------------------------------------
def test_positive_clearance_to_every_person_at_all_times(summary):
    assert summary["min_person_clearance_m"] > 0.0
    for name, value in summary["min_clearance_by_person_m"].items():
        assert value > 0.0, name


def test_positive_clearance_to_every_surface(summary):
    assert summary["min_scenery_clearance_m"] > 0.0


def test_zero_contacts_and_zero_falls(summary):
    assert summary["contact_steps"] == 0
    assert summary["fallen_steps"] == 0


def test_the_trunk_never_dropped_and_finished_near_nominal(summary):
    assert summary["min_trunk_z_m"] >= 0.09
    assert abs(summary["final_trunk_z_m"] - 0.116) <= 0.012


# -- visibility --------------------------------------------------------------
def test_the_active_person_was_visible_wherever_line_of_sight_existed(summary):
    assert summary["monitor_visible_fraction_with_los"] >= 0.95


def test_the_camera_watched_the_right_roles_in_the_right_order(summary):
    assert summary["subject_role_order"] == \
        summary["expected_subject_role_order"]


def test_the_visibility_measurement_has_a_real_denominator(summary):
    assert summary["monitor_los_steps"] > 400


def test_the_guardian_was_watched_for_most_of_the_run(summary):
    entry = summary["subject_visibility"]["nadia"]
    assert entry["steps"] > 2000
    assert entry["fraction_with_los"] >= 0.95


# -- the populated building --------------------------------------------------
def test_at_least_five_other_adults_moved(summary):
    assert summary["moving_adults"] >= 5


def test_every_scripted_person_spends_real_time_walking(summary):
    for name, fraction in summary["actor_moving_fraction"].items():
        assert fraction > 0.10, (name, fraction)


# -- the walking itself ------------------------------------------------------
def test_the_journey_is_a_real_physical_walk(summary):
    assert summary["path_m"] >= 6.0
    assert summary["net_m"] >= 5.5
    assert summary["walk_path_m"] > 5.0


def test_the_duck_stayed_on_its_own_route(summary):
    assert summary["max_cross_track_m"] <= 0.45


def test_the_walking_states_are_the_only_ones_that_moved(rollout):
    for record in rollout.records:
        if record["state"] not in WALKING_STATES:
            assert record["command_vx"] == 0.0, (record["t"], record["state"])


def test_the_policy_is_the_byte_identical_stock_checkpoint(summary):
    from etiquette_thresholds import UPSTREAM_POLICY_SHA
    assert summary["policy_sha256"] == UPSTREAM_POLICY_SHA
    assert summary["observation_dim"] == 61
    assert summary["action_scale"] == 0.9
    assert summary["gyro_sensor"] == "imu_ang_vel"


# -- determinism -------------------------------------------------------------
def test_the_record_stream_is_complete_and_ordered(rollout):
    assert len(rollout.records) == rollout.total_steps
    times = [r["t"] for r in rollout.records]
    assert times == sorted(times)


def test_the_duck_ends_where_the_route_ends(rollout, route):
    final = np.array(rollout.records[-1]["duck_xy"])
    end, _ = route.pose_at_arc(route.length)
    assert float(np.linalg.norm(final - end)) < 0.35
