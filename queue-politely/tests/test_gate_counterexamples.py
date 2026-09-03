#!/usr/bin/env python3
"""Synthetic counterexamples: proof that every gate can FAIL.

Each test takes the baseline fixture from ``conftest.py`` - which passes every
gate - mutates exactly one thing, runs the REAL ``summarize``, and requires that
specific gate to report False.  ``test_the_baseline_fixture_passes_every_gate``
is what makes the rest meaningful.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from queue_metrics import summarize  # noqa: E402
from queue_people import DEPARTURE_TIMES  # noqa: E402

from conftest import STATIONS, TRUTH  # noqa: E402


def test_the_baseline_fixture_passes_every_gate(baseline):
    """Without this, a counterexample proves nothing."""
    summary = summarize(baseline)
    failed = [name for name, ok in summary["gates"].items() if not ok]
    assert failed == [], f"baseline should pass everything, failed: {failed}"
    assert summary["all_gates_pass"]


def _mutate(baseline, fn):
    rollout = copy.deepcopy(baseline)
    fn(rollout)
    return summarize(rollout)


def test_wrong_order_fails_order_inferred(baseline):
    def mutate(rollout):
        rollout.order_samples[40]["correct"] = False
        rollout.order_samples[40]["inferred"] = ["bianchi", "alvarez"]
    assert not _mutate(baseline, mutate)["gates"]["order_inferred"]


def test_wrong_tail_fails_tail_correct(baseline):
    def mutate(rollout):
        rollout.order_samples[40]["tail_correct"] = False
        rollout.order_samples[40]["tail"] = "dubois"
    assert not _mutate(baseline, mutate)["gates"]["tail_correct"]


def test_following_the_wrong_person_fails_no_wrong_locks(baseline):
    def mutate(rollout):
        for record in rollout.records:
            if record["state"] == "WAIT":
                record["predecessor"] = "chandra"
    assert not _mutate(baseline, mutate)["gates"]["no_wrong_locks"]


def test_a_naive_reading_that_happened_to_be_right_fails_naive_would_fail(baseline):
    """If the scene stops defeating the heuristics, the claim is void."""
    def mutate(rollout):
        rollout.first_reading["naive_tails"]["by_range"] = "eriksson"
    assert not _mutate(baseline, mutate)["gates"]["naive_would_fail"]


def test_including_a_bystander_fails_bystanders_excluded(baseline):
    def mutate(rollout):
        rollout.records[60]["inferred_order"] = TRUTH + ["nakamura"]
    assert not _mutate(baseline, mutate)["gates"]["bystanders_excluded"]


def test_refusing_only_one_available_gap_fails_rejected_enough(baseline):
    def mutate(rollout):
        rollout.first_reading["rejected_available"] = (
            rollout.first_reading["rejected_available"][:1])
    assert not _mutate(baseline, mutate)["gates"]["rejected_enough"]


def test_refusing_only_gaps_that_did_not_fit_fails_rejected_enough(baseline):
    """A refusal only counts if the duck could have taken the place."""
    def mutate(rollout):
        rollout.first_reading["rejected_available"] = []
    assert not _mutate(baseline, mutate)["gates"]["rejected_enough"]


def test_taking_a_cut_in_gap_fails_joined_behind_tail(baseline):
    def mutate(rollout):
        rollout.machine.accepted_gap = {
            "gap": "between_dubois_eriksson", "kind": "cut_in",
            "ahead": "dubois", "behind": "eriksson", "separation_m": 0.90,
            "physically_fits": True, "verdict": "join", "reason": "cut"}
    assert not _mutate(baseline, mutate)["gates"]["joined_behind_tail"]


def test_joining_beside_the_tail_fails_join_band(baseline):
    def mutate(rollout):
        rollout.join_evidence["in_band"] = False
        rollout.join_evidence["lateral_m"] = 0.45
    assert not _mutate(baseline, mutate)["gates"]["join_band"]


def test_getting_in_front_of_somebody_fails_no_overtaking(baseline):
    def mutate(rollout):
        record = rollout.records[70]
        record["duck_arc_m"] = 0.10
        record["person_arc_m"] = dict(STATIONS)
    assert not _mutate(baseline, mutate)["gates"]["no_overtaking"]


def test_the_queue_position_going_backwards_fails_no_overtaking(baseline):
    def mutate(rollout):
        for index, record in enumerate(rollout.records):
            if record["state"] == "WAIT":
                record["predecessors_remaining"] = 1 + (index % 3)
    assert not _mutate(baseline, mutate)["gates"]["no_overtaking"]


def test_two_cycles_fails_wait_advance_cycles(baseline):
    def mutate(rollout):
        advances = [c for c in rollout.machine.cycles
                    if c.get("kind") == "advance"]
        rollout.machine.cycles.remove(advances[-1])
    assert not _mutate(baseline, mutate)["gates"]["wait_advance_cycles"]


def test_a_tiny_command_while_waiting_fails_stationary_command_zero(baseline):
    """Exact zero means exact zero; a decaying command is still a command."""
    def mutate(rollout):
        rollout.stationary_command_max["WAIT"] = 1e-6
    assert not _mutate(baseline, mutate)["gates"]["stationary_command_zero"]


def test_stopping_too_far_back_fails_advances_are_real(baseline):
    def mutate(rollout):
        for record in rollout.records:
            if record["state"] == "ADVANCE" and record["cycle"] == 3:
                record["standoff_m"] = 1.20
    assert not _mutate(baseline, mutate)["gates"]["advances_are_real"]


def test_stopping_too_close_fails_advances_are_real(baseline):
    def mutate(rollout):
        for record in rollout.records:
            if record["state"] == "ADVANCE" and record["cycle"] == 3:
                record["standoff_m"] = 0.20
    assert not _mutate(baseline, mutate)["gates"]["advances_are_real"]


def test_an_advance_with_no_real_motion_fails_advances_are_real(baseline):
    def mutate(rollout):
        rollout.cycle_path[2] = 0.01
    assert not _mutate(baseline, mutate)["gates"]["advances_are_real"]


def test_cutting_the_corner_fails_bend_followed(baseline):
    """Positive cross-track, on the bend, is the corner-cut sense."""
    def mutate(rollout):
        for record in rollout.records:
            if record["state"] == "ADVANCE":
                record["duck_arc_m"] = 1.65      # on the fold
                record["duck_cross_track_m"] = 0.22
    assert not _mutate(baseline, mutate)["gates"]["bend_followed"]


def test_leaving_the_lane_fails_bend_followed(baseline):
    def mutate(rollout):
        for record in rollout.records:
            if record["state"] == "ADVANCE":
                record["duck_cross_track_m"] = -0.34
    assert not _mutate(baseline, mutate)["gates"]["bend_followed"]


def test_losing_sight_of_the_person_ahead_fails_predecessor_visible(baseline):
    def mutate(rollout):
        rollout.cycle_tracking[2] = [True] * 5 + [False] * 5
    assert not _mutate(baseline, mutate)["gates"]["predecessor_visible"]


def test_reaching_the_counter_early_fails_counter_last(baseline):
    def mutate(rollout):
        for record in rollout.records:
            if record["state"] == "AT_COUNTER":
                record["t"] = DEPARTURE_TIMES[-1] - 5.0
    assert not _mutate(baseline, mutate)["gates"]["counter_last"]


def test_touching_a_person_fails_person_clearance_and_no_contacts(baseline):
    def mutate(rollout):
        rollout.min_person_clearance = -0.01
        rollout.records[50]["nearest_clearance_m"] = -0.01
    summary = _mutate(baseline, mutate)
    assert not summary["gates"]["person_clearance"]
    assert not summary["gates"]["no_contacts"]


def test_touching_scenery_fails_scenery_clearance(baseline):
    def mutate(rollout):
        rollout.min_scenery_clearance = -0.02
        rollout.records[50]["scenery_clearance_m"] = -0.02
    summary = _mutate(baseline, mutate)
    assert not summary["gates"]["scenery_clearance"]
    assert not summary["gates"]["no_contacts"]


def test_a_sub_onset_command_fails_no_decorative_commands(baseline):
    def mutate(rollout):
        rollout.records[50]["command"] = [0.12, 0.0, 0.0]
    assert not _mutate(baseline, mutate)["gates"]["no_decorative_commands"]


def test_a_fall_fails_no_falls_and_min_trunk_z(baseline):
    def mutate(rollout):
        rollout.records[50]["trunk_z_m"] = 0.04
    summary = _mutate(baseline, mutate)
    assert not summary["gates"]["no_falls"]
    assert not summary["gates"]["min_trunk_z"]


def test_ending_crouched_fails_final_trunk_z(baseline):
    def mutate(rollout):
        rollout.records[-1]["trunk_z_m"] = 0.098
    assert not _mutate(baseline, mutate)["gates"]["final_trunk_z"]


def test_a_timeout_fails_no_timeouts(baseline):
    def mutate(rollout):
        rollout.machine.timeouts.append("advance_timeout")
    assert not _mutate(baseline, mutate)["gates"]["no_timeouts"]


def test_never_reaching_done_fails_state_order(baseline):
    def mutate(rollout):
        rollout.records = [r for r in rollout.records if r["state"] != "DONE"]
    assert not _mutate(baseline, mutate)["gates"]["state_order"]


def test_skipping_the_evaluation_phase_fails_state_order(baseline):
    def mutate(rollout):
        rollout.records = [r for r in rollout.records
                           if r["state"] != "EVALUATE_GAPS"]
    assert not _mutate(baseline, mutate)["gates"]["state_order"]
