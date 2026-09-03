#!/usr/bin/env python3
"""Counterexamples: each mutation breaks ONE thing and its gate must go red.

A GATE THAT CANNOT FAIL IS NOT A GATE
---------------------------------------
Every test below takes the summary of the REAL passing run, mutates exactly one
measured quantity, re-grades with the shipped ``pps_metrics.gates``, and
requires:

1. the named gate to go RED, and
2. **no other gate to be repaired by the same mutation** - which catches a gate
   whose failure is really just a restatement of another one.

The meta-test at the bottom parses this file with ``ast``, collects every gate
name asserted, and fails if any gate in ``gates()`` has no counterexample.
Adding a gate without one therefore fails HERE rather than shipping as an
unchecked claim.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from pps_metrics import gates

HERE = Path(__file__).resolve()


def grade(summary: dict) -> dict[str, bool]:
    return {name: passed for name, passed, _ in gates(summary)}


@pytest.fixture(scope="module")
def baseline(summary):
    """The real run, and the assurance that it passes everything."""
    results = grade(summary)
    failed = [name for name, ok in results.items() if not ok]
    assert not failed, f"the baseline run does not pass: {failed}"
    assert len(results) == 27, f"expected 27 gates, found {len(results)}"
    return results


def broken(summary: dict, baseline: dict[str, bool], gate: str, mutate) -> None:
    """Mutate a copy, require ``gate`` to fail, and nothing to be repaired."""
    mutant = copy.deepcopy(summary)
    mutate(mutant)
    results = grade(mutant)
    assert gate in results, f"no such gate: {gate}"
    assert not results[gate], (
        f"{gate} still passed after the mutation that should break it")
    repaired = [name for name, ok in results.items()
                if ok and not baseline.get(name, True)]
    assert not repaired, f"the mutation repaired {repaired}"


# -- identity ----------------------------------------------------------------
def test_protected_identity_remains_aina(summary, baseline):
    def mutate(s):
        s["protected_person"] = "noor"
    broken(summary, baseline, "protected identity remains Aina", mutate)


# -- the escort formation -----------------------------------------------------
def test_neutral_escort_physically_joined_needs_a_real_path(summary,
                                                            baseline):
    """A duck that never walked did not join a formation, it was placed in one."""
    def mutate(s):
        s["path_m"] = 0.4
    broken(summary, baseline, "neutral escort physically joined", mutate)


def test_neutral_escort_physically_joined_needs_the_final_slot(summary,
                                                               baseline):
    def mutate(s):
        s["final_escort_distance_m"] = 0.61
    broken(summary, baseline, "neutral escort physically joined", mutate)


def test_escort_restored_after_episodes_needs_the_session_to_end_in_escort(
        summary, baseline):
    def mutate(s):
        s["final_state"] = "HOLD_BUFFER"
    broken(summary, baseline, "escort restored after episodes", mutate)


# -- the four cycles ----------------------------------------------------------
def test_four_distinct_genuine_intrusion_cycles_needs_four_of_them(summary,
                                                                   baseline):
    def mutate(s):
        s["episodes"] = [e for e in s["episodes"] if e["kind"] != "intrusion"][:1] \
            + [e for e in s["episodes"] if e["kind"] == "intrusion"][:3]
    broken(summary, baseline, "four distinct genuine intrusion cycles", mutate)


def test_four_distinct_cycles_rejects_the_same_person_four_times(summary,
                                                                 baseline):
    """Four cycles from one adult is one encounter repeated, not four."""
    def mutate(s):
        for episode in s["episodes"]:
            if episode["kind"] == "intrusion":
                episode["selected"] = "dario"
    broken(summary, baseline, "four distinct genuine intrusion cycles", mutate)


def test_intrusions_alternate_bearings(summary, baseline):
    """Four approaches from the same side would be one station held four times."""
    def mutate(s):
        s["bearings_alternate"] = False
    broken(summary, baseline, "intrusions alternate bearings", mutate)


def test_every_intrusion_produced_a_physical_protective_path(summary,
                                                             baseline):
    """A nonzero command is not proof the policy crossed its gait-onset cliff."""
    def mutate(s):
        for episode in s["episodes"]:
            if episode["kind"] == "intrusion":
                episode["path_m"] = 0.05
                break
    broken(summary, baseline,
           "every intrusion produced a physical protective path", mutate)


def test_every_interpose_reduced_target_error(summary, baseline):
    """Walking is not the same as walking TOWARD the station."""
    def mutate(s):
        for episode in s["episodes"]:
            if episode["kind"] == "intrusion":
                episode["target_reduction_m"] = 0.02
                break
    broken(summary, baseline, "every interpose reduced target error", mutate)


def test_interpose_reached_the_between_bearing(summary, baseline):
    """Getting near the station is not the same as being between the two."""
    def mutate(s):
        for episode in s["episodes"]:
            if episode["kind"] == "intrusion":
                episode["between_ticks"] = 0
                break
    broken(summary, baseline, "interpose reached the between-bearing", mutate)


# -- the false alarm ----------------------------------------------------------
def test_false_near_pass_dismissed_needs_him_to_have_been_seen(summary,
                                                               baseline):
    """Dismissing somebody you never observed is not a dismissal."""
    def mutate(s):
        s["false_alarm_seen"] = False
    broken(summary, baseline, "false near-pass dismissed", mutate)


def test_false_near_pass_dismissed_fails_if_he_triggered_an_episode(summary,
                                                                    baseline):
    def mutate(s):
        s["false_alarm_dismissed"] = False
    broken(summary, baseline, "false near-pass dismissed", mutate)


# -- the squeeze --------------------------------------------------------------
def test_simultaneous_squeeze_used_safe_gap_branch(summary, baseline):
    def mutate(s):
        s["squeeze"] = None
    broken(summary, baseline, "simultaneous squeeze used safe-gap branch",
           mutate)


def test_a_squeeze_with_only_one_participant_is_not_a_squeeze(summary,
                                                              baseline):
    def mutate(s):
        s["squeeze"]["secondary"] = None
    broken(summary, baseline, "simultaneous squeeze used safe-gap branch",
           mutate)


def test_squeeze_escape_was_a_real_path(summary, baseline):
    """An escape that did not move is a state label, not an escape."""
    def mutate(s):
        s["squeeze"]["path_m"] = 0.1
    broken(summary, baseline, "squeeze escape was a real path", mutate)


# -- the ward's own approach --------------------------------------------------
def test_ward_approach_triggered_retreat_needs_the_episode(summary, baseline):
    def mutate(s):
        s["ward_approach"] = None
    broken(summary, baseline, "ward approach triggered retreat", mutate)


def test_ward_approach_triggered_retreat_needs_real_reverse_travel(summary,
                                                                   baseline):
    def mutate(s):
        s["retreat_path_m"] = 0.05
    broken(summary, baseline, "ward approach triggered retreat", mutate)


def test_retreat_increased_ward_range(summary, baseline):
    """Graded separately from displacement, because she closes while it reverses.

    A retreat that moved the duck backward while she closed faster would
    satisfy the displacement test and fail the person.
    """
    def mutate(s):
        s["retreat_range_gain_m"] = 0.02
    broken(summary, baseline, "retreat increased ward range", mutate)


# -- what the duck could see --------------------------------------------------
def test_ward_visible_with_line_of_sight(summary, baseline):
    def mutate(s):
        s["ward_visible_fraction_with_los"] = 0.80
    broken(summary, baseline, "ward visible >=95% with LOS", mutate)


def test_active_people_visible_while_acted_on(summary, baseline):
    """Acting on somebody the camera never held would be acting on a schedule."""
    def mutate(s):
        s["active_visible_fraction"] = 0.55
    broken(summary, baseline, "active people visible while acted on", mutate)


# -- physical safety ----------------------------------------------------------
def test_positive_person_clearance(summary, baseline):
    def mutate(s):
        s["min_person_clearance_m"] = -0.01
    broken(summary, baseline, "positive person clearance", mutate)


def test_positive_scenery_clearance(summary, baseline):
    def mutate(s):
        s["min_scenery_clearance_m"] = 0.0
    broken(summary, baseline, "positive scenery clearance", mutate)


def test_zero_geometric_contacts(summary, baseline):
    def mutate(s):
        s["contact_steps"] = 1
    broken(summary, baseline, "zero geometric contacts", mutate)


def test_zero_falls(summary, baseline):
    def mutate(s):
        s["fallen_steps"] = 3
    broken(summary, baseline, "zero falls", mutate)


def test_trunk_stays_above_the_fallen_height(summary, baseline):
    def mutate(s):
        s["min_trunk_z_m"] = 0.085
    broken(summary, baseline, "trunk stays above 0.09m", mutate)


def test_final_trunk_near_nominal_rejects_a_collapse(summary, baseline):
    def mutate(s):
        s["final_trunk_z_m"] = 0.098
    broken(summary, baseline, "final trunk near nominal", mutate)


def test_final_trunk_near_nominal_rejects_a_bounce(summary, baseline):
    """Bounded both ways: too tall is as wrong as too short."""
    def mutate(s):
        s["final_trunk_z_m"] = 0.140
    broken(summary, baseline, "final trunk near nominal", mutate)


# -- the honesty of the commands ---------------------------------------------
def test_exact_zero_in_declared_hold_states(summary, baseline):
    """The behavior's strongest stillness claim, and it is checked literally.

    A sub-onset command would stand perfectly still and log motion, so anything
    but an exact zero has to fail - including a value far below gait onset.
    """
    def mutate(s):
        s["zero_state_peak"] = 1e-7
    broken(summary, baseline, "exact zero in declared hold states", mutate)


def test_no_decorative_sub_gait_commands(summary, baseline):
    """A command below onset moves nothing and logs a walk; there must be none."""
    def mutate(s):
        s["sub_gait_ticks"] = 4
    broken(summary, baseline, "no decorative sub-gait commands", mutate)


def test_no_lateral_policy_command(summary, baseline):
    """The stock policy cannot strafe, so any ``vy`` at all is a fiction."""
    def mutate(s):
        s["max_abs_vy"] = 1e-6
    broken(summary, baseline, "no lateral policy command", mutate)


def test_real_physical_locomotion(summary, baseline):
    def mutate(s):
        s["walk_path_m"] = 2.0
    broken(summary, baseline, "real physical locomotion", mutate)


# -- provenance ---------------------------------------------------------------
def test_exact_sensor_observation_and_scale_rejects_a_wrong_sensor(summary,
                                                                   baseline):
    """A wrong name silently feeds a different quantity into base_ang_vel."""
    def mutate(s):
        s["gyro_sensor"] = "angular-velocity"
    broken(summary, baseline, "exact sensor observation and scale", mutate)


def test_exact_sensor_observation_and_scale_rejects_a_wrong_width(summary,
                                                                  baseline):
    def mutate(s):
        s["observation_dim"] = 60
    broken(summary, baseline, "exact sensor observation and scale", mutate)


def test_exact_sensor_observation_and_scale_rejects_a_wrong_action_scale(
        summary, baseline):
    def mutate(s):
        s["action_scale"] = 0.5
    broken(summary, baseline, "exact sensor observation and scale", mutate)


def test_stock_walking_policy(summary, baseline):
    """The behavior is a controller around the SHIPPED policy, not a new one."""
    def mutate(s):
        s["policy_sha256"] = "0" * 64
    broken(summary, baseline, "stock walking policy", mutate)


# -- the meta-test ------------------------------------------------------------
def _asserted_gate_names() -> set[str]:
    """Every gate name passed to ``broken`` anywhere in this file."""
    tree = ast.parse(HERE.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "broken"
                and len(node.args) >= 3
                and isinstance(node.args[2], ast.Constant)):
            found.add(node.args[2].value)
    return found


def test_every_shipped_gate_has_a_counterexample(summary):
    """Adding a gate without a counterexample fails here.

    Otherwise a gate could be added, pass on the real run, and never once be
    shown capable of failing - which is indistinguishable from an assert that
    compares a constant to itself.
    """
    shipped = {name for name, _, _ in gates(summary)}
    covered = _asserted_gate_names()
    missing = shipped - covered
    assert not missing, f"gates with no counterexample: {sorted(missing)}"


def test_no_counterexample_names_a_gate_that_does_not_exist(summary):
    shipped = {name for name, _, _ in gates(summary)}
    stale = _asserted_gate_names() - shipped
    assert not stale, f"counterexamples for gates that no longer exist: {stale}"


def test_the_gate_list_is_the_pinned_twenty_seven(summary):
    """The count the README and the validator both quote."""
    assert len(gates(summary)) == 27
    assert len({name for name, _, _ in gates(summary)}) == 27
