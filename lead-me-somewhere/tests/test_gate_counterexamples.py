"""Counterexamples: every gate must FAIL when the thing it protects is broken.

A gate that cannot fail is decoration.  Each test here mutates a summary — the
same dict ``guide_metrics.gates`` reads — to describe a run that should be
rejected, and requires the named gate to reject it.

The summaries are built from the REAL rollout and then damaged, so a mutation
that no gate catches is a genuine hole rather than an artifact of a hand-built
fixture.
"""

from __future__ import annotations

import copy

import pytest

from guide_metrics import UPSTREAM_POLICY_SHA, gates

pytestmark = pytest.mark.slow


def verdicts(summary) -> dict[str, bool]:
    return {label: ok for label, ok, _ in gates(summary)}


def find(summary, needle: str) -> tuple[str, bool]:
    """The single gate whose label contains ``needle``."""
    matches = [(label, ok) for label, ok, _ in gates(summary) if needle in label]
    assert len(matches) == 1, (
        f"{needle!r} matched {len(matches)} gates: {[m[0] for m in matches]}")
    return matches[0]


def broken(summary, **changes):
    mutated = copy.deepcopy(summary)
    mutated.update(changes)
    return mutated


# -- the baseline -----------------------------------------------------------

def test_the_unmutated_summary_passes_every_gate(summary):
    failures = [label for label, ok, _ in gates(summary) if not ok]
    assert not failures, f"the real run fails: {failures}"


# -- the request ------------------------------------------------------------

def test_walking_to_the_wrong_destination_is_caught(summary):
    mutated = broken(summary, resolved_destination="CAFE",
                     resolution_correct=False)
    _, ok = find(mutated, "resolved to the correct one")
    assert not ok


def test_a_single_candidate_makes_the_choice_meaningless(summary):
    mutated = broken(summary, destination_candidates=["LIFTS"],
                     destination_candidate_count=1)
    _, ok = find(mutated, "resolved to the correct one")
    assert not ok


def test_arriving_somewhere_other_than_the_request_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["arrival"]["destination"] = "HELPDESK"
    _, ok = find(mutated, "reached the destination it was ASKED for")
    assert not ok


def test_walking_off_without_acknowledging_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["state_command_max"]["RECEIVE_DESTINATION"] = 0.34
    _, ok = find(mutated, "acknowledged before moving")
    assert not ok


# -- the plan ---------------------------------------------------------------

def test_a_two_bend_route_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["plan"]["bends"] = mutated["plan"]["bends"][:2]
    _, ok = find(mutated, "at least 3 bends")
    assert not ok


def test_a_route_that_grazes_an_obstacle_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["plan"]["min_planned_clearance_m"] = 0.01
    _, ok = find(mutated, "avoids the inflated obstacles")
    assert not ok


def test_a_planner_that_ignored_the_crowd_is_caught(summary):
    """The gate that stops 'avoids the crowd' being an empty-corridor claim."""
    mutated = copy.deepcopy(summary)
    mutated["plan"]["crowd_blocked_cells"] = 0
    mutated["plan"]["crowd_blockers"] = {}
    _, ok = find(mutated, "BECAUSE of the crowd")
    assert not ok


def test_a_straight_line_route_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["plan"]["detour_ratio"] = 1.02
    _, ok = find(mutated, "genuine detour")
    assert not ok


def test_a_detour_around_nothing_is_caught(summary):
    """A long route in an empty hall is not a detour."""
    mutated = copy.deepcopy(summary)
    mutated["plan"]["straight_line_blocked_by"] = ""
    _, ok = find(mutated, "genuine detour")
    assert not ok


# -- leading ----------------------------------------------------------------

def test_a_duck_that_barely_moved_is_caught(summary):
    _, ok = find(broken(summary, lead_path_m=0.4), "physically led at least")
    assert not ok


def test_walking_on_the_spot_is_caught(summary):
    """Path without net progress is a treadmill."""
    _, ok = find(broken(summary, lead_net_m=0.2), "net progress")
    assert not ok


def test_a_follower_who_never_walked_is_caught(summary):
    _, ok = find(broken(summary, follower_walked_m=0.3),
                 "person actually walked")
    assert not ok


def test_a_duck_that_ended_up_following_is_caught(summary):
    """The invariant that separates a guide from a follower."""
    mutated = broken(summary, follower_ahead_steps=12, min_lead_gap_m=-0.3)
    _, ok = find(mutated, "the duck LED")
    assert not ok


def test_a_duck_that_wandered_off_its_route_is_caught(summary):
    _, ok = find(broken(summary, max_cross_track_m=0.9),
                 "stayed on its planned route")
    assert not ok


def test_abandoning_her_for_a_prolonged_interval_is_caught(summary):
    mutated = broken(summary,
                     max_safety_breach_s=summary["safety_max_interval_s"] + 1.0)
    _, ok = find(mutated, "safety maximum for a prolonged interval")
    assert not ok


# -- the episodes -----------------------------------------------------------

def test_a_single_episode_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["episodes"] = mutated["episodes"][:1]
    mutated["episode_count"] = 1
    _, ok = find(mutated, "lag/loss episodes were DETECTED")
    assert not ok


def test_zero_episodes_is_caught(summary):
    mutated = broken(summary, episodes=[], episode_count=0)
    _, ok = find(mutated, "lag/loss episodes were DETECTED")
    assert not ok


def test_missing_one_of_the_declared_stalls_is_caught(summary):
    """The strongest claim: the duck noticed the right things at the right
    times without being able to read the script."""
    mutated = copy.deepcopy(summary)
    mutated["declared_stalls"][1]["detected"] = False
    mutated["declared_stalls"][1]["episode_indices"] = []
    _, ok = find(mutated, "every declared stall produced")
    assert not ok


def test_creeping_while_waiting_is_caught(summary):
    """The behavior's strongest per-tick claim.  A command of 0.24 is the
    MEASURED gait onset, so this is a duck that walked while 'waiting'."""
    mutated = copy.deepcopy(summary)
    mutated["episodes"][0]["max_command_while_waiting"] = 0.24
    _, ok = find(mutated, "EXACTLY zero for every WAITING tick")
    assert not ok


def test_even_a_tiny_nonzero_wait_command_is_caught(summary):
    """Exactly zero means exactly zero, not nearly."""
    mutated = copy.deepcopy(summary)
    mutated["episodes"][0]["max_command_while_waiting"] = 1e-6
    _, ok = find(mutated, "EXACTLY zero for every WAITING tick")
    assert not ok


def test_a_recorded_zero_command_violation_is_caught(summary):
    mutated = broken(summary, zero_command_violations=[
        {"t": 30.0, "state": "WAIT_FOR_PERSON", "command": [0.24, 0.0, 0.0]}])
    _, ok = find(mutated, "EXACTLY zero for every WAITING tick")
    assert not ok


def test_drifting_while_checking_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["episodes"][0]["squaring_up_path_m"] = 0.9
    _, ok = find(mutated, "equally still while CHECKING")
    assert not ok


def test_a_token_wait_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["episodes"][0]["wait_duration_s"] = 0.4
    _, ok = find(mutated, "every wait lasted at least")
    assert not ok


def test_a_wait_that_achieved_nothing_is_caught(summary):
    """If she did not close on the duck, the duck stopped for no reason."""
    mutated = copy.deepcopy(summary)
    mutated["episodes"][0]["follower_closed_m"] = 0.02
    _, ok = find(mutated, "every wait lasted at least")
    assert not ok


def test_resuming_while_she_is_still_far_away_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["episodes"][0]["distance_at_resume_m"] = 3.0
    _, ok = find(mutated, "every resume was justified")
    assert not ok


def test_resuming_while_she_is_out_of_sight_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["episodes"][0]["visible_at_resume"] = False
    _, ok = find(mutated, "every resume was justified")
    assert not ok


def test_resuming_on_a_single_good_frame_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["episodes"][0]["recovered_for_s"] = 0.02
    _, ok = find(mutated, "every resume was justified")
    assert not ok


def test_waiting_inside_a_wall_is_caught(summary):
    mutated = copy.deepcopy(summary)
    mutated["episodes"][0]["waiting_spot_scenery_clearance_m"] = -0.05
    _, ok = find(mutated, "waiting spot kept positive clearance")
    assert not ok


# -- the arrival ------------------------------------------------------------

def test_stopping_too_far_from_the_destination_is_caught(summary):
    _, ok = find(broken(summary, final_destination_distance_m=3.4),
                 "final distance to the destination")
    assert not ok


def test_stopping_on_top_of_the_destination_is_caught(summary):
    _, ok = find(broken(summary, final_destination_distance_m=0.05),
                 "final distance to the destination")
    assert not ok


def test_finishing_facing_away_is_caught(summary):
    _, ok = find(broken(summary, final_facing_error_deg=131.0),
                 "finished facing the destination")
    assert not ok


def test_a_missing_indication_is_caught(summary):
    _, ok = find(broken(summary, indicate_seconds=0.4), "arrival was indicated")
    assert not ok


def test_leaving_her_behind_at_the_destination_is_caught(summary):
    _, ok = find(broken(summary, final_follower_distance_m=4.2),
                 "finished safely nearby")
    assert not ok


# -- visibility, safety, states --------------------------------------------

def test_losing_sight_of_her_while_monitoring_is_caught(summary):
    _, ok = find(broken(summary, monitor_visible_fraction_with_los=0.71),
                 "visible in >=")
    assert not ok


def test_an_empty_hall_is_caught(summary):
    _, ok = find(broken(summary, moving_adults=2), "other adults were moving")
    assert not ok


def test_touching_a_person_is_caught(summary):
    _, ok = find(broken(summary, min_person_clearance_m=-0.01),
                 "clearance to every person")
    assert not ok


def test_touching_the_scenery_is_caught(summary):
    _, ok = find(broken(summary, min_scenery_clearance_m=0.0),
                 "clearance to every obstacle")
    assert not ok


def test_a_contact_step_is_caught(summary):
    _, ok = find(broken(summary, contact_steps=1), "zero contacts")
    assert not ok


def test_skipping_a_declared_state_is_caught(summary):
    mutated = broken(summary, states_visited=[
        s for s in summary["states_visited"] if s != "WAIT_FOR_PERSON"])
    _, ok = find(mutated, "every declared state was visited")
    assert not ok


def test_inventing_a_state_is_caught(summary):
    mutated = broken(summary,
                     states_visited=summary["states_visited"] + ["IMPROVISE"])
    _, ok = find(mutated, "every visited state is one this behavior declares")
    assert not ok


def test_abandoning_her_is_caught(summary):
    mutated = broken(summary,
                     forbidden_state_steps={"ABANDON": 40, "SEARCH": 0})
    _, ok = find(mutated, "zero ABANDON and SEARCH")
    assert not ok


def test_a_timeout_is_caught(summary):
    _, ok = find(broken(summary, timeouts=["LEAD@120.00s"]),
                 "no phase hit its ceiling")
    assert not ok


def test_a_fall_is_caught(summary):
    _, ok = find(broken(summary, fallen_steps=3), "zero falls")
    assert not ok


def test_a_collapsed_trunk_is_caught(summary):
    _, ok = find(broken(summary, min_trunk_z_m=0.061),
                 "trunk never below")
    assert not ok


def test_finishing_slumped_is_caught(summary):
    _, ok = find(broken(summary, final_trunk_z_m=0.094),
                 "final trunk height")
    assert not ok


# -- the physical contract --------------------------------------------------

def test_a_different_gyro_sensor_is_caught(summary):
    _, ok = find(broken(summary, gyro_sensor="gyro"), "exact imu_ang_vel")
    assert not ok


def test_a_different_observation_width_is_caught(summary):
    _, ok = find(broken(summary, observation_dim=48), "exact imu_ang_vel")
    assert not ok


def test_a_different_action_scale_is_caught(summary):
    _, ok = find(broken(summary, action_scale=1.0), "exact imu_ang_vel")
    assert not ok


def test_a_retrained_policy_is_caught(summary):
    _, ok = find(broken(summary, policy_sha256="0" * 64),
                 "byte-identical stock walking policy")
    assert not ok


def test_the_expected_sha_is_the_one_the_run_used(summary):
    assert summary["policy_sha256"] == UPSTREAM_POLICY_SHA


# -- the mutation harness itself --------------------------------------------

def test_no_assert_compares_a_constant_to_itself(summary):
    """A mutation that changes nothing must not appear to prove something."""
    unchanged = copy.deepcopy(summary)
    assert verdicts(unchanged) == verdicts(summary)
    assert all(verdicts(summary).values())
