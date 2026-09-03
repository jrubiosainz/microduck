#!/usr/bin/env python3
"""Synthetic counterexamples: proof that every acceptance gate can FAIL.

A gate that cannot fail is decoration.  Each test here takes the ``baseline``
fixture — which provably passes all thirty gates, asserted first — breaks
exactly one invariant, and requires the corresponding gate to go red.  Where the
invariants are independent, it requires that ONLY that gate goes red, which is
what stops a mutation "proving" a gate by breaking the fixture wholesale.

The fakes are hand-built rather than produced by perturbing a real rollout: a
real rollout that fails one gate usually fails several, which would isolate
nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from beside_metrics import (  # noqa: E402
    MIN_BESIDE_PATH_M,
    MIN_BESIDE_SECONDS,
    MIN_COMPLETED_SWITCHES,
    MIN_CROSS_REAR_MARGIN_M,
    MIN_SIDE_DECISIONS,
    MIN_SWITCH_LATERAL_M,
    MIN_SWITCH_NET_M,
    MIN_SWITCH_PATH_M,
    MIN_VISIBLE_WITH_LOS,
    gates,
    summarize,
)
from conftest import failing, gate_map, only_failure  # noqa: E402


# -- the baseline --------------------------------------------------------------

def test_the_baseline_passes_every_gate(baseline):
    """Everything else in this module depends on this.

    Without a baseline that provably passes, a failing mutation would prove
    nothing: the fixture itself might simply be malformed.
    """
    results = gates(summarize(baseline))
    failed = [label for label, ok, _ in results if not ok]
    assert failed == [], f"the fixture is malformed: {failed}"
    assert len(results) == 30, (
        "the gate count changed; every gate needs a counterexample here")


def test_every_gate_carries_evidence_rather_than_a_bare_verdict(baseline):
    for label, _, evidence in gates(summarize(baseline)):
        assert label.strip(), "a gate with no label explains nothing"
        assert evidence.strip(), f"gate {label!r} reports no evidence"


# -- the initial join ----------------------------------------------------------

def test_a_duck_that_spawns_in_formation_fails_the_join_gate(baseline):
    """The whole point of starting out of both slots."""
    for record in baseline.records:
        record["path_m"] = 0.05
        record["duck_xy"] = [0.0, 0.0]
    only_failure(baseline, "walked into formation")


def test_a_join_that_never_completed_fails_the_join_gate(baseline):
    baseline.machine.joined = False
    only_failure(baseline, "walked into formation")


def test_a_trailing_first_position_fails_the_side_slot_gate(baseline):
    """A duck two metres astern has not joined a SIDE."""
    for record in baseline.records:
        if record["state"] == "BESIDE_LEFT":
            record["lateral_m"] = 0.05
            record["lateral_abs_m"] = 0.05
    assert "real SIDE slot" in " ".join(failing(baseline))


# -- beside time and distance --------------------------------------------------

def test_too_little_time_beside_her_fails_its_gate(baseline):
    baseline.beside_steps = int((MIN_BESIDE_SECONDS - 0.1) / baseline.dt)
    baseline.beside_side_steps = {"BESIDE_RIGHT": baseline.beside_steps}
    baseline.formation_steps = baseline.beside_steps
    only_failure(baseline, f"{MIN_BESIDE_SECONDS:.0f}s spent beside her")


def test_too_little_distance_beside_her_fails_its_gate(baseline):
    """A duck held in formation by a guardian who barely moved."""
    baseline.beside_path_m = MIN_BESIDE_PATH_M - 0.01
    only_failure(baseline, "walked while beside her")


def test_a_lateral_offset_outside_the_band_fails_the_band_gate(baseline):
    baseline.beside_lateral = [0.30, 0.58, 0.72]
    only_failure(baseline, "lateral offset stayed inside")


def test_a_lateral_offset_too_far_out_also_fails_the_band_gate(baseline):
    baseline.beside_lateral = [0.48, 0.58, 1.40]
    only_failure(baseline, "lateral offset stayed inside")


def test_an_unbounded_longitudinal_error_fails_its_gate(baseline):
    baseline.beside_longitudinal = [-2.4, -0.12, 0.05]
    only_failure(baseline, "longitudinal error stayed bounded")


# -- side decisions ------------------------------------------------------------

def test_too_few_side_decisions_fails_its_gate(baseline):
    baseline.machine.decisions = baseline.machine.decisions[:MIN_SIDE_DECISIONS - 1]
    failed = failing(baseline)
    assert any("side decisions were made" in label for label in failed)


def test_a_run_with_no_completed_switch_fails_several_switch_gates(baseline):
    """Deliberately NOT isolated: with no switch there is nothing to grade, and
    every switch gate must go red rather than passing vacuously."""
    baseline.machine.switches = []
    baseline.switch_path = {}
    baseline.switch_start_xy = {}
    baseline.switch_end_xy = {}
    failed = failing(baseline)
    assert any(f"{MIN_COMPLETED_SWITCHES} completed physical side switch"
               in label for label in failed)
    for fragment in ("caused by a MEASURED blockage", "crossed BEHIND her",
                     "astern before crossing", "real path with real net",
                     "clear of her legs"):
        assert any(fragment in label for label in failed), (
            f"the gate about {fragment!r} passed vacuously with no switches")


def test_a_switch_with_no_measured_cause_fails_the_causation_gate(baseline):
    """A switch that merely HAPPENED, with no named hazard behind it."""
    baseline.machine.switches[0]["cause"] = ""
    baseline.machine.switches[0]["detail"] = ""
    only_failure(baseline, "caused by a MEASURED blockage")


def test_a_switch_confirmed_for_zero_seconds_fails_the_causation_gate(baseline):
    """A single tick of a swinging arm is not a blockage."""
    baseline.machine.switches[0]["blocked_for_s"] = 0.0
    only_failure(baseline, "caused by a MEASURED blockage")


def test_a_switch_attributed_to_an_undeclared_hazard_fails(baseline):
    baseline.machine.switches[0]["cause"] = "vibes"
    only_failure(baseline, "caused by a MEASURED blockage")


def test_a_run_where_no_side_was_ever_refused_fails_the_refusal_gate(baseline):
    """Every tick usable on both sides: the duck never had to choose."""
    for record in baseline.records:
        record["verdict_left"]["usable"] = True
        record["verdict_right"]["usable"] = True
    baseline.machine.decisions = [
        {**decision, "kind": "initial", "reason": "both usable; left is fine"}
        for decision in baseline.machine.decisions]
    assert any("unsafe side was refused" in label for label in failing(baseline))


# -- the crossover -------------------------------------------------------------

def test_a_switch_that_cut_across_her_front_fails_the_rear_gate(baseline):
    """The single most antisocial thing this behavior could do."""
    baseline.switch_max_longitudinal[0] = +0.65
    baseline.max_forward_during_switch = 0.65
    only_failure(baseline, "crossed BEHIND her")


def test_a_switch_that_never_dropped_astern_fails_the_margin_gate(baseline):
    baseline.switch_min_longitudinal[0] = -(MIN_CROSS_REAR_MARGIN_M - 0.01)
    only_failure(baseline, "astern before crossing")


def test_a_duck_that_got_ahead_of_her_at_any_tick_fails_the_half_plane_gate(
        baseline):
    """Measured EVERY tick, not only during a crossing."""
    baseline.max_forward_longitudinal = 0.40
    only_failure(baseline, "never got ahead of her at ANY tick")


def test_a_switch_that_was_a_step_sideways_fails_the_real_path_gate(baseline):
    baseline.switch_path[0] = MIN_SWITCH_PATH_M - 0.01
    only_failure(baseline, "real path with real net")


def test_a_switch_with_no_net_displacement_fails_the_real_path_gate(baseline):
    baseline.switch_end_xy[0] = baseline.switch_start_xy[0] + np.array(
        [MIN_SWITCH_NET_M - 0.02, 0.0])
    only_failure(baseline, "real path with real net")


def test_a_switch_that_stayed_on_one_side_fails_the_real_path_gate(baseline):
    """Both ends on the same side is not a switch, whatever it is called."""
    baseline.switch_lateral_end[0] = +0.60
    only_failure(baseline, "real path with real net")


def test_a_switch_with_too_little_lateral_travel_fails(baseline):
    baseline.switch_lateral_start[0] = +0.20
    baseline.switch_lateral_end[0] = -(MIN_SWITCH_LATERAL_M - 0.20 - 0.01)
    only_failure(baseline, "real path with real net")


def test_a_switch_that_brushed_her_legs_fails_the_clearance_gate(baseline):
    baseline.switch_min_clearance[0] = -0.01
    only_failure(baseline, "clear of her legs")


# -- the formation after the switch --------------------------------------------

def test_an_unstable_post_switch_formation_fails_its_gate(baseline):
    """Reaching the far side and immediately losing it proves nothing."""
    kept = [record for record in baseline.records
            if record["state"] != "BESIDE_RIGHT"]
    tail = [record for record in baseline.records
            if record["state"] == "BESIDE_RIGHT"][:20]
    baseline.records = kept + tail
    assert any("opposite-side formation is stable" in label
               for label in failing(baseline))


def test_a_post_switch_formation_out_of_band_fails_its_gate(baseline):
    for record in baseline.records:
        if record["state"] == "BESIDE_RIGHT":
            record["lateral_m"] = -1.30
            record["lateral_abs_m"] = 1.30
    assert any("opposite-side formation is stable" in label
               for label in failing(baseline))


# -- the bends -----------------------------------------------------------------

def test_a_formation_lost_through_the_bends_fails_the_bend_gate(baseline):
    """A duck that only holds station on the straights."""
    for record in baseline.records:
        if record["t"] >= 30.0:
            record["state"] = "JOIN_OTHER_SIDE"
    failed = failing(baseline)
    assert any("route bends" in label for label in failed)


def test_a_formation_that_swings_wide_through_a_bend_fails_the_bend_gate(
        baseline):
    for record in baseline.records:
        if record["t"] >= 30.0 and record["state"] == "BESIDE_RIGHT":
            record["lateral_m"] = -1.50
            record["lateral_abs_m"] = 1.50
    assert any("route bends" in label for label in failing(baseline))


def test_a_route_that_turns_only_one_way_fails_the_both_hands_gate(baseline):
    """A formation keeper that only ever turns one way has not been tested on
    the sign its yaw controller is weakest on."""
    summary = summarize(baseline)
    summary["route"] = {
        **summary["route"],
        "bends": [bend for bend in summary["route"]["bends"]
                  if bend["hand"] == "left"],
    }
    results = {label: ok for label, ok, _ in gates(summary)}
    assert not results["the route contains both a left and a right bend"]


# -- visibility ----------------------------------------------------------------

def test_a_duck_that_lost_sight_of_her_fails_the_visibility_gate(baseline):
    baseline.visible_with_los = int(
        baseline.los_steps * (MIN_VISIBLE_WITH_LOS - 0.02))
    baseline.visible_steps = baseline.visible_with_los
    only_failure(baseline, "visible in >=")


def test_the_visibility_gate_is_conditioned_on_line_of_sight(baseline):
    """Ticks where a column was genuinely in the way must not count against the
    duck, but every other tick must."""
    baseline.los_steps = 4000
    baseline.visible_with_los = 4000
    baseline.visible_steps = 4000
    baseline.blocked_by = {"obs_column_n": 300}
    assert failing(baseline) == set(), (
        "genuinely occluded ticks must not fail the duck")


# -- safety --------------------------------------------------------------------

def test_touching_a_person_fails_the_person_clearance_gate(baseline):
    baseline.min_person_clearance = 0.0
    only_failure(baseline, "clearance to every person")


def test_a_negative_person_clearance_also_fails(baseline):
    baseline.min_person_clearance = -0.02
    only_failure(baseline, "clearance to every person")


def test_touching_the_scenery_fails_the_scenery_clearance_gate(baseline):
    baseline.min_scenery_clearance = -0.001
    only_failure(baseline, "clearance to every obstacle and wall")


def test_a_single_contact_step_fails_the_zero_contacts_gate(baseline):
    baseline.contact_steps = 1
    only_failure(baseline, "zero contacts")


# -- states --------------------------------------------------------------------

def test_a_single_done_step_fails_the_forbidden_state_gate(baseline):
    """A rollout that reaches DONE has stopped early."""
    baseline.state_steps = {**baseline.state_steps, "DONE": 1}
    only_failure(baseline, "zero HOLD and DONE steps")


def test_a_single_hold_step_fails_the_forbidden_state_gate(baseline):
    baseline.state_steps = {**baseline.state_steps, "HOLD": 1}
    failed = failing(baseline)
    assert any("zero HOLD and DONE steps" in label for label in failed)


def test_an_undeclared_state_fails_the_declared_states_gate(baseline):
    baseline.records[10]["state"] = "IMPROVISING"
    assert any("visited state is one this behavior declares" in label
               for label in failing(baseline))


def test_a_phase_that_hit_its_ceiling_fails_the_timeout_gate(baseline):
    baseline.machine.timeouts = ["FALL_BACK@21.40s"]
    only_failure(baseline, "no phase hit its ceiling")


# -- locomotion health ---------------------------------------------------------

def test_a_fall_fails_the_zero_falls_gate(baseline):
    baseline.fallen_steps = 1
    only_failure(baseline, "zero falls")


def test_a_trunk_below_the_floor_height_fails_its_gate(baseline):
    baseline.min_trunk_z = 0.085
    only_failure(baseline, "trunk never below")


def test_a_collapsed_final_pose_fails_the_final_height_gate(baseline):
    baseline.records[-1]["trunk_z"] = 0.098
    only_failure(baseline, "final trunk height")


# -- the physical contract ------------------------------------------------------

def test_a_different_policy_fails_the_byte_identical_gate(baseline):
    baseline.policy_sha256 = "0" * 64
    only_failure(baseline, "byte-identical stock walking policy")


def test_a_different_sensor_or_observation_width_fails_the_contract_gate(
        baseline):
    """The contract gate reads the summary, so it is mutated there."""
    for field, value in (("gyro_sensor", "angular-velocity"),
                         ("observation_dim", 48),
                         ("action_scale", 1.0)):
        summary = summarize(baseline)
        summary[field] = value
        results = {label: ok for label, ok, _ in gates(summary)}
        contract = [label for label in results
                    if "imu_ang_vel sensor" in label][0]
        assert not results[contract], f"{field}={value!r} was accepted"


# -- the gate set itself --------------------------------------------------------

def test_no_two_gates_share_a_label(baseline):
    """Duplicate labels would make ``only_failure`` ambiguous and would hide a
    failure behind a passing namesake."""
    labels = [label for label, _, _ in gates(summarize(baseline))]
    assert len(labels) == len(set(labels))


def test_every_gate_is_exercised_by_at_least_one_counterexample(baseline):
    """The coverage claim, computed rather than asserted by hand.

    Each mutation below is applied to a FRESH summary and the set of gates it
    turns red is collected; the union must be every gate.  A gate no mutation
    can redden is decoration, and this test is what finds it.
    """
    def with_summary(mutate):
        summary = summarize(baseline)
        mutate(summary)
        return {label for label, ok, _ in gates(summary) if not ok}

    mutations = [
        lambda s: s.update(join_path_m=0.0, join_net_m=0.0),
        lambda s: s.update(initial_join_lateral_m=0.02),
        lambda s: s.update(beside_seconds=1.0),
        lambda s: s.update(beside_path_m=0.0),
        lambda s: s.update(beside_lateral_min_m=0.01),
        lambda s: s.update(beside_longitudinal_abs_max_m=9.0),
        lambda s: s.update(side_decisions=[], side_decision_count=0),
        lambda s: s.update(switches=[], completed_switches=0),
        lambda s: s.update(refusal_count=0, blocked_tick_count=0),
        lambda s: s.update(max_forward_longitudinal_m=9.0),
        lambda s: s.update(post_switch_seconds=0.0),
        lambda s: s.update(bends_followed=0),
        lambda s: s.update(route={**s["route"], "bends": []}),
        lambda s: s.update(visible_fraction_with_los=0.0),
        lambda s: s.update(min_person_clearance_m=-1.0),
        lambda s: s.update(min_scenery_clearance_m=-1.0),
        lambda s: s.update(contact_steps=5),
        lambda s: s.update(forbidden_state_steps={"HOLD": 3, "DONE": 0}),
        lambda s: s.update(states_visited=["NONSENSE"]),
        lambda s: s.update(timeouts=["JOIN_SIDE@30s"]),
        lambda s: s.update(fallen_steps=9),
        lambda s: s.update(min_trunk_z_m=0.0),
        lambda s: s.update(final_trunk_z_m=0.0),
        lambda s: s.update(gyro_sensor="wrong"),
        lambda s: s.update(policy_sha256="0" * 64),
    ]

    reddened = set()
    for mutate in mutations:
        reddened |= with_summary(mutate)

    every = {label for label, _, _ in gates(summarize(baseline))}
    missing = every - reddened
    assert not missing, f"these gates cannot be made to fail: {sorted(missing)}"
