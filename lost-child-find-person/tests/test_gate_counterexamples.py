#!/usr/bin/env python3
"""Synthetic counterexamples: proof that every acceptance gate can FAIL.

Each test takes the baseline fixture from ``conftest.py`` — which passes all 25
gates — mutates exactly one measured quantity, runs the REAL ``summarize`` and
``gates``, and requires that specific gate to report False.

``test_the_baseline_fixture_passes_every_gate`` is what makes the rest
meaningful: a mutation that fails a gate proves nothing unless the unmutated
fixture passed it.

WHERE ISOLATION IS NOT POSSIBLE, IT IS STATED
----------------------------------------------
Two gates read the same list.  ">= 2 distinct false candidates refused" and
"every refused candidate was genuinely camera-visible" both require at least two
refusals, so a fixture with one refusal necessarily fails both.  Those tests
assert their own gate and say so; every other mutation is genuinely one gate.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lost_cast import GUARDIAN  # noqa: E402
from lost_metrics import gates, summarize  # noqa: E402

from conftest import find_gate, gate_map  # noqa: E402


def _mutate(baseline, mutation):
    """Apply ``mutation`` to a deep copy and grade it with the real gate."""
    rollout = copy.deepcopy(baseline)
    mutation(rollout)
    return gates(summarize(rollout))


def _fails(baseline, mutation, fragment: str) -> None:
    label, ok, evidence = find_gate(_mutate(baseline, mutation), fragment)
    assert not ok, f"gate {label!r} should have failed; evidence: {evidence}"


def test_the_baseline_fixture_passes_every_gate(baseline):
    """Without this, no counterexample below proves anything."""
    results = gates(summarize(baseline))
    failed = [label for label, ok, _ in results if not ok]
    assert failed == [], f"baseline must pass everything, failed: {failed}"
    assert len(results) == 25


def test_every_gate_label_is_distinct(baseline):
    """Two gates sharing a label would let one hide the other's failure."""
    labels = [label for label, _, _ in gates(summarize(baseline))]
    assert len(set(labels)) == len(labels)


# ------------------------------------------------------- the follow and loss
def test_a_spawn_already_in_formation_fails_the_follow_path_gate(baseline):
    """A duck that never walked before the loss did not follow anybody."""
    def mutate(rollout):
        for record in rollout.records:
            if record["state"] == "FOLLOW":
                record["path_m"] = 0.05
                record["duck_xy"] = [2.15, -1.90]
    _fails(baseline, mutate, "real follow path before the first loss")


def test_following_somebody_never_seen_fails_the_visibility_gate(baseline):
    def mutate(rollout):
        for index, record in enumerate(rollout.records):
            if record["state"] == "FOLLOW":
                record["guardian_visible"] = index % 4 == 0
    _fails(baseline, mutate, "guardian visible while following")


def test_a_rollout_that_never_declares_a_loss_fails_the_loss_gate(baseline):
    """No LOST record at all means the sustained-invisibility gate is vacuous."""
    def mutate(rollout):
        for record in rollout.records:
            if record["state"] == "LOST":
                record["state"] = "STOP"
        rollout.state_steps.pop("LOST", None)
    _fails(baseline, mutate, "loss declared only after a sustained")


def test_one_moving_tick_while_lost_fails_the_zero_command_gate(baseline):
    """The behavior's central safety claim, broken by a single tick."""
    def mutate(rollout):
        moved = next(r for r in rollout.records if r["state"] == "SEARCH_SWEEP")
        moved["command_peak"] = 0.001
    _fails(baseline, mutate, "zero locomotion command in EVERY stationary state")


def test_a_stationary_state_peak_above_zero_fails_the_same_gate(baseline):
    """The per-state peak is graded too, not only the per-record count."""
    def mutate(rollout):
        rollout.state_command_max["REJECT"] = 0.22
    _fails(baseline, mutate, "zero locomotion command in EVERY stationary state")


# ------------------------------------------------------------------ occlusion
def test_a_brief_occlusion_fails_the_two_second_gate(baseline):
    def mutate(rollout):
        for run in rollout.occlusion_runs:
            run["duration_s"] = 1.5
    _fails(baseline, mutate, "geometric occlusion lasting")


def test_looking_the_wrong_way_is_not_a_geometric_occlusion(baseline):
    """An out-of-frustum run is the duck's own gaze, not a body in the way."""
    def mutate(rollout):
        for run in rollout.occlusion_runs:
            run["blockers"] = {"out_of_frustum": sum(run["blockers"].values())}
    _fails(baseline, mutate, "geometric occlusion lasting")


# ---------------------------------------------------------- false candidates
def test_refusing_only_one_person_fails_the_distinct_refusals_gate(baseline):
    """Also fails the camera-evidence gate, which needs two look-alikes too."""
    def mutate(rollout):
        rollout.identity.rejections = [
            r for r in rollout.identity.rejections if r["name"] == "sofia"]
    _fails(baseline, mutate, "distinct false candidates refused")


def test_refusing_somebody_the_camera_never_saw_fails_the_evidence_gate(baseline):
    """A refusal of a body that was never visible is bookkeeping, not perception."""
    def mutate(rollout):
        rollout.lookalike_seen.pop("sofia")
    _fails(baseline, mutate, "genuinely camera-visible")


def test_a_single_wrong_identity_lock_fails_the_wrong_accept_gate(baseline):
    def mutate(rollout):
        rollout.identity.wrong_accepts.append(
            {"name": "mira", "score": 0.94, "t": 24.0})
    _fails(baseline, mutate, "zero wrong-identity locks")


# ------------------------------------------------------- identity continuity
def test_changing_the_guardian_mid_run_fails_the_continuity_gate(baseline):
    def mutate(rollout):
        rollout.records[1200]["guardian"] = "mira"
    _fails(baseline, mutate, "identity is the SAME guardian throughout")


def test_accepting_a_look_alike_fails_the_acceptance_gate(baseline):
    def mutate(rollout):
        rollout.identity.accepted[0]["name"] = "mira"
    _fails(baseline, mutate, "every acceptance is of the guardian")


def test_never_accepting_anybody_also_fails_the_acceptance_gate(baseline):
    """An empty accept list is not a clean sheet; it means no reacquisition."""
    def mutate(rollout):
        rollout.identity.accepted = []
    _fails(baseline, mutate, "every acceptance is of the guardian")


# --------------------------------------------------------------------- trail
def test_a_discarded_trail_fails_the_memory_gate(baseline):
    def mutate(rollout):
        rollout.trail.points = []
    _fails(baseline, mutate, "world-space last-known trail")


# -------------------------------------------------------------- the rejoins
def test_a_shallow_cycle_fails_the_deep_cycle_gate(baseline):
    """One refusal per cycle is a glance, not an evaluation."""
    def mutate(rollout):
        for cycle in rollout.machine.cycles:
            cycle["rejections"] = cycle["rejections"][:1]
    _fails(baseline, mutate, "deep loss/reacquisition cycle")


def test_a_rejoin_that_barely_moved_fails_the_progress_gate(baseline):
    def mutate(rollout):
        rollout.rejoin_path[1] = 0.05
        rollout.rejoin_end_xy[1] = rollout.rejoin_start_xy[1].copy()
    _fails(baseline, mutate, "every rejoin is real physical progress")


def test_a_rejoin_that_ended_farther_away_fails_the_range_gate(baseline):
    def mutate(rollout):
        rollout.rejoin_end_range[0] = 3.4
    _fails(baseline, mutate, "every rejoin lowered the range")


def test_losing_sight_during_the_rejoin_fails_the_tracking_gate(baseline):
    """Graded only on steps where line of sight existed, so this is real."""
    def mutate(rollout):
        rollout.rejoin_visible_with_los[0] = (
            [True] * 400 + [False] * 205)
    _fails(baseline, mutate, "of REJOIN steps")


def test_an_infeasible_route_fails_the_route_gate(baseline):
    def mutate(rollout):
        rollout.rejoin_routes[0]["feasible"] = False
    _fails(baseline, mutate, "no rejoin route cut through")


# -------------------------------------------------------------------- safety
def test_touching_a_person_fails_the_person_clearance_gate(baseline):
    def mutate(rollout):
        rollout.min_person_clearance = -0.004
    _fails(baseline, mutate, "positive clearance to every person")


def test_pressing_into_the_kiosk_fails_the_scenery_clearance_gate(baseline):
    def mutate(rollout):
        rollout.min_scenery_clearance = 0.0
    _fails(baseline, mutate, "positive clearance to every obstacle")


def test_any_contact_step_fails_the_contact_gate(baseline):
    def mutate(rollout):
        rollout.contact_steps = 1
    _fails(baseline, mutate, "zero contacts")


# -------------------------------------------------------------- the finish
def test_stopping_outside_the_band_fails_the_standoff_gate(baseline):
    def mutate(rollout):
        rollout.records[-1]["guardian_range_m"] = 1.20
    _fails(baseline, mutate, "final distance inside the")


def test_stopping_too_close_also_fails_the_standoff_gate(baseline):
    """The band is two-sided: loitering and crowding both fail."""
    def mutate(rollout):
        rollout.records[-1]["guardian_range_m"] = 0.30
    _fails(baseline, mutate, "final distance inside the")


def test_finishing_without_seeing_her_fails_the_final_visibility_gate(baseline):
    def mutate(rollout):
        rollout.records[-1]["guardian_visible"] = False
    _fails(baseline, mutate, "guardian visible at the final standoff")


# ------------------------------------------------------- locomotion health
def test_a_fall_fails_the_fall_gate(baseline):
    def mutate(rollout):
        rollout.fallen_steps = 1
    _fails(baseline, mutate, "zero falls")


def test_a_low_trunk_fails_the_trunk_height_gate(baseline):
    def mutate(rollout):
        rollout.min_trunk_z = 0.0812
    _fails(baseline, mutate, "trunk never below")


def test_finishing_crouched_fails_the_final_height_gate(baseline):
    def mutate(rollout):
        rollout.records[-1]["trunk_z"] = 0.142
    _fails(baseline, mutate, "final trunk height near the nominal")


def test_a_phase_ceiling_fails_the_timeout_gate(baseline):
    def mutate(rollout):
        rollout.machine.timeouts.append("SEARCH_SWEEP@42.00s")
    _fails(baseline, mutate, "no phase hit its ceiling")


# ------------------------------------------------------ the summary is real
def test_the_summary_reports_the_pinned_physical_constants(baseline):
    """The gate quotes these; a drift here would silently rewrite the claim."""
    summary = summarize(baseline)
    assert summary["observation_dim"] == 61
    assert summary["action_scale"] == 0.9
    assert summary["gyro_sensor"] == "imu_ang_vel"
    assert summary["guardian"] == GUARDIAN.name
    assert summary["standoff_band_m"] == [0.45, 0.75]


def test_the_gate_map_covers_every_documented_family(baseline):
    """A gate silently dropped from ``gates()`` would never be missed."""
    labels = gate_map(gates(summarize(baseline)))
    for fragment in ("follow path", "sustained", "zero locomotion command",
                     "occlusion", "refused", "wrong-identity", "trail",
                     "rejoin", "clearance", "contacts", "standoff",
                     "falls", "ceiling"):
        assert any(fragment in label for label in labels), fragment
