"""The validated rollout: what actually happened, measured from the records.

These are the ``slow`` tests.  They read the SAME rollout the acceptance gate
grades, at the same duration, so a passing test here describes the artifact
rather than a convenient shorter run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from guide_cast import FOLLOWER, OTHER_NAMES
from guide_layout import DESTINATION_KEYS, OCCLUDERS, occluder_between
from guide_states import (
    FORBIDDEN_STATES,
    LAG_DISTANCE_M,
    MONITOR_STATES,
    SAFETY_MAX_DISTANCE_M,
    SAFETY_MAX_INTERVAL_S,
    STATES,
    ZERO_COMMAND_STATES,
)

pytestmark = pytest.mark.slow


# -- the whole gate ---------------------------------------------------------

def test_every_acceptance_gate_passes(gate_results):
    passed, results = gate_results
    failures = [f"{label}: {evidence}"
                for label, ok, evidence in results if not ok]
    assert passed, "failing gates:\n  " + "\n  ".join(failures)


def test_the_gate_is_not_vacuous(gate_results):
    _, results = gate_results
    assert len(results) >= 30, f"only {len(results)} gates"


# -- the request and the plan ----------------------------------------------

def test_the_duck_walked_to_the_destination_it_was_asked_for(summary):
    assert summary["resolution_correct"]
    assert summary["arrival"]["destination"] == summary["requested_destination"]
    assert len(summary["destination_candidates"]) >= 3
    assert set(summary["destination_candidates"]) == set(DESTINATION_KEYS)


def test_the_route_was_searched_from_the_ducks_own_pose(rollout, summary):
    """The plan's first waypoint must be where the duck actually stood, not a
    constant somebody wrote down."""
    plan = summary["plan"]
    first = np.array(plan["waypoints"][0])
    at_plan = next(r for r in rollout.records if r["state"] == "PLAN")
    assert float(np.linalg.norm(first - np.array(at_plan["duck_xy"]))) < 0.12


def test_the_planned_bends_are_the_bends_that_were_walked(rollout, summary):
    """The duck must track the filleted route, not the raw vertices."""
    assert summary["max_cross_track_m"] <= 0.45
    assert len(summary["plan"]["bends"]) >= 3
    # The trunk yaw must actually swing through comparable turns.
    leading = [r for r in rollout.records if r["state"] in ("LEAD", "RESUME")]
    yaws = np.unwrap(np.radians([r["duck_yaw_deg"] for r in leading]))
    total_turn = float(np.sum(np.abs(np.diff(yaws))))
    planned_turn = sum(abs(b["turn_deg"]) for b in summary["plan"]["bends"])
    assert math.degrees(total_turn) >= 0.6 * planned_turn, (
        f"the duck turned {math.degrees(total_turn):.0f} deg against "
        f"{planned_turn:.0f} deg of planned bends")


# -- leading ---------------------------------------------------------------

def test_the_lead_is_a_real_walk_with_net_progress(summary):
    assert summary["lead_path_m"] >= 3.5
    assert summary["lead_net_m"] >= 3.0
    assert summary["lead_path_m"] > summary["lead_net_m"], (
        "path equals net displacement, so the route did not bend")


def test_the_person_was_led_rather_than_walking_her_own_route(summary):
    assert summary["follower_walked_m"] >= 3.0


def test_the_duck_led_at_every_single_tick(rollout, summary):
    """A guide that ends up following is not a guide.  Measured along the
    shared trail, which is the only place 'ahead' has a meaning."""
    assert summary["follower_ahead_steps"] == 0
    assert summary["min_lead_gap_m"] > 0.0
    for record in rollout.records:
        assert record["follower_trail_gap_m"] > 0.0, (
            f"she was ahead at t={record['t']}")


def test_the_duck_never_left_her_beyond_the_safety_maximum_for_long(summary):
    assert summary["max_safety_breach_s"] < SAFETY_MAX_INTERVAL_S


def test_the_safety_gate_is_not_vacuous(summary):
    """If she never went far enough to matter, the safety claim says nothing."""
    assert summary["max_follower_range_m"] > LAG_DISTANCE_M, (
        "she never lagged at all, so the safety interval is untested")
    assert summary["max_follower_range_m"] < SAFETY_MAX_DISTANCE_M + 1.5


# -- the episodes ----------------------------------------------------------

def test_two_episodes_were_detected_from_measurement(summary):
    assert summary["episode_count"] >= 2
    for episode in summary["episodes"]:
        assert episode["cause"] in ("lag", "loss")
        if episode["cause"] == "lag":
            assert episode["distance_at_detect_m"] > LAG_DISTANCE_M
            assert episode["lagging_for_s"] >= summary["lag_confirm_s"] - 0.05
        else:
            assert episode["unseen_for_s"] >= summary["lost_confirm_s"] - 0.05


def test_each_declared_stall_produced_an_episode(summary):
    """The strongest statement in the behavior: the duck noticed the right
    things at the right times, without being able to see the script."""
    for stall in summary["declared_stalls"]:
        assert stall["detected"], f"missed {stall['label']!r}"
        assert stall["detection_lag_s"] >= 0.0, (
            "detected BEFORE the stall began, which would mean the detector "
            "is reading the schedule")
        assert stall["detection_lag_s"] <= 12.0


def test_the_command_was_exactly_zero_for_every_waiting_tick(rollout):
    for record in rollout.records:
        if record["state"] in ZERO_COMMAND_STATES:
            assert record["command"] == [0.0, 0.0, 0.0], (
                f"{record['state']} at t={record['t']} emitted "
                f"{record['command']}")


def test_no_zero_command_violation_was_recorded(summary):
    assert summary["zero_command_violations"] == []


def test_waiting_really_was_standing_still(summary):
    """The MEASURED 10 s zero-command drift is 0.0057 m of path.  Anything much
    beyond that in a waiting state would mean the duck was creeping."""
    for episode in summary["episodes"]:
        assert episode["max_command_while_waiting"] == 0.0
        assert episode["duck_moved_while_waiting_m"] < 0.05, (
            f"episode {episode['index']} drifted "
            f"{episode['duck_moved_while_waiting_m']} m while 'waiting'")
        assert episode["squaring_up_path_m"] <= summary["check_still_path_m"]


def test_each_wait_achieved_something(summary):
    for episode in summary["episodes"]:
        assert episode["wait_duration_s"] >= 1.5
        assert episode["follower_closed_m"] >= 0.35, (
            "she did not close on the duck, so the wait accomplished nothing")


def test_every_resume_was_justified(summary):
    for episode in summary["episodes"]:
        assert episode["distance_at_resume_m"] <= summary["catchup_threshold_m"]
        assert episode["visible_at_resume"] is True
        assert episode["recovered_for_s"] >= summary["resume_confirm_s"] - 0.05


def test_the_duck_waited_somewhere_safe(summary):
    for episode in summary["episodes"]:
        assert episode["waiting_spot_scenery_clearance_m"] > 0.0
        assert episode["waiting_spot_xy"] is not None


def test_the_episodes_are_separated_in_time(summary):
    """Two episodes 200 ms apart would be one episode counted twice."""
    times = [e["detected_at_s"] for e in summary["episodes"]]
    for a, b in zip(times, times[1:]):
        assert b - a > 10.0, f"episodes at {a}s and {b}s are the same event"


# -- visibility ------------------------------------------------------------

def test_the_follower_was_watched_whenever_it_was_possible(summary):
    assert summary["monitor_visible_fraction_with_los"] >= 0.95


def test_the_los_predicate_is_conservative_and_never_excused_a_tick(rollout,
                                                                     summary):
    """The LOS conditioning must never let the duck off for a tick it was
    responsible for.

    MEASURED FINDING, and it is a limitation worth stating plainly: in this
    scenario the planar occluder test NEVER fires.  The follower walks the
    duck's own trail 0.6-1.6 m astern, and no full-height body in the concourse
    ever falls inside a segment that short, so ``los_available`` is true at
    every one of the ~4750 control steps.

    That makes the LOS-conditioned percentage IDENTICAL to the raw one, which is
    the strict direction: the duck is held responsible for every monitoring tick
    without exception.  The conditioning is kept because it is the honest
    predicate for a hall where an occluder COULD intervene, and because a future
    scene change that put one on the sightline would need it — but it is earning
    nothing here, and the gate must not be read as if it were.
    """
    assert OCCLUDERS, "the hall contains no full-height occluders"
    blocked = [r for r in rollout.records if not r["los_available"]]
    assert not blocked, (
        "the LOS predicate now fires, so this test's premise has changed and "
        "the README's stated limitation is stale")
    assert summary["monitor_los_steps"] == summary["monitor_steps"]
    assert summary["monitor_visible_fraction_with_los"] == pytest.approx(
        summary["monitor_visible_with_los_steps"] / summary["monitor_steps"],
        abs=5e-5), "the summary rounds to 4 decimal places"


def test_the_visibility_gate_is_non_vacuous_because_people_do_block_her(
        rollout, summary):
    """The gate has to be passable but not free.

    The REAL occlusions in this run come from moving adults crossing the
    sightline, not from the scenery: the MuJoCo ray cast hits their torsos.
    Those ticks are NOT excused by the planar predicate — it only knows about
    full-height static bodies — so the duck is graded on them and still clears
    95 %.  A run in which nothing ever blocked her would make the percentage
    meaningless.
    """
    invisible = [r for r in rollout.records if not r["follower_visible"]]
    assert invisible, (
        "she was visible at literally every tick, so the visibility gate is "
        "measuring nothing")
    causes = {r["follower_blocked_by"] for r in invisible
              if r["follower_blocked_by"]}
    assert causes, (
        "every invisible tick was a frustum miss; no body ever occluded her")
    assert summary["blocked_by"], "no blocking cause was recorded at all"
    assert summary["monitor_visible_fraction_with_los"] < 1.0, (
        "the monitoring visibility is a perfect 100 %, which means nothing "
        "ever tested it")


def test_the_monitor_visibility_is_measured_over_real_ticks(summary):
    assert summary["monitor_steps"] > 400
    assert summary["monitor_los_steps"] > 0


# -- arrival ---------------------------------------------------------------

def test_the_duck_stopped_in_the_arrival_band(summary):
    low, high = summary["final_destination_band_m"]
    assert low <= summary["final_destination_distance_m"] <= high


def test_the_duck_finished_facing_what_it_led_her_to(summary):
    assert summary["final_facing_error_deg"] <= summary["face_tolerance_deg"]


def test_the_person_finished_safely_nearby(summary):
    assert summary["final_follower_distance_m"] <= summary["final_person_near_m"]


def test_the_arrival_was_indicated_long_enough_to_read(summary):
    assert summary["indicate_seconds"] >= summary["indicate_required_s"] - 0.05


def test_the_duck_arrived_before_her(summary):
    assert summary["follower_trail_gap_final_m"] > 0.0


# -- the populated hall ----------------------------------------------------

def test_at_least_five_other_adults_were_moving(summary):
    assert summary["moving_adults"] >= 5
    assert len(summary["other_adults"]) >= 5


def test_the_other_adults_really_moved(rollout):
    first, last = rollout.records[0], rollout.records[-1]
    moved = 0
    for name in OTHER_NAMES:
        delta = float(np.linalg.norm(
            np.array(last["person_xy"][name]) - np.array(first["person_xy"][name])))
        if delta > 0.5:
            moved += 1
    assert moved >= 5


def test_the_follower_is_not_one_of_the_five(summary):
    assert FOLLOWER.name not in summary["other_adults"]


# -- safety and locomotion health ------------------------------------------

def test_positive_clearance_to_everybody_and_everything(summary):
    assert summary["min_person_clearance_m"] > 0.0
    assert summary["min_scenery_clearance_m"] > 0.0
    assert summary["contact_steps"] == 0


def test_the_clearance_gate_is_not_vacuous(summary):
    """A minimum clearance equal to the probe's cutoff means nothing ever came
    near, and the gate is measuring an empty room."""
    assert summary["min_person_clearance_m"] < 1.4, (
        "nobody ever came near the duck; the clearance claim is vacuous")
    assert summary["min_scenery_clearance_m"] < 0.9


def test_the_robot_stayed_on_its_feet(summary):
    assert summary["fallen_steps"] == 0
    assert summary["min_trunk_z_m"] >= 0.09
    assert abs(summary["final_trunk_z_m"] - 0.116) <= 0.012


def test_the_physical_contract(summary):
    from guide_metrics import UPSTREAM_POLICY_SHA
    assert summary["gyro_sensor"] == "imu_ang_vel"
    assert summary["observation_dim"] == 61
    assert summary["action_scale"] == 0.9
    assert summary["policy_sha256"] == UPSTREAM_POLICY_SHA


# -- states ----------------------------------------------------------------

def test_every_declared_state_was_visited(summary):
    missing = set(STATES) - set(summary["states_visited"])
    assert not missing, f"never entered: {sorted(missing)}"


def test_no_state_outside_the_declaration_appeared(summary):
    assert set(summary["states_visited"]) <= set(STATES)
    for state in FORBIDDEN_STATES:
        assert summary["forbidden_state_steps"][state] == 0


def test_no_state_was_entered_and_left_within_one_tick(summary):
    """A state with a single tick is a state that did not happen."""
    for state in STATES:
        if state == "ARRIVE":
            # ARRIVE is genuinely instantaneous: the route's own final heading
            # already faces the fixture, so facing_ok is true on arrival.  That
            # is the design (turn-in-place is MEASURED to be unavailable), and
            # it is asserted rather than excused.
            continue
        seconds = summary["state_seconds"].get(state, 0.0)
        assert seconds >= 0.5, f"{state} lasted only {seconds}s"


def test_arrive_is_instantaneous_because_the_route_solved_the_facing(summary):
    assert summary["state_seconds"].get("ARRIVE", 0.0) <= 0.1
    assert summary["final_facing_error_deg"] <= summary["face_tolerance_deg"]


def test_no_phase_hit_its_ceiling(summary):
    assert summary["timeouts"] == []


# -- determinism -----------------------------------------------------------

def test_the_run_is_deterministic(policy_path):
    """Two short rollouts from the same start must agree exactly, or none of
    the measurements above describe a reproducible artifact."""
    from rollout_guide import GuideRollout
    a = GuideRollout(str(policy_path), 6.0)
    a.run()
    b = GuideRollout(str(policy_path), 6.0)
    b.run()
    assert len(a.records) == len(b.records)
    for ra, rb in zip(a.records, b.records):
        assert ra["duck_xy"] == rb["duck_xy"]
        assert ra["command"] == rb["command"]
        assert ra["state"] == rb["state"]
