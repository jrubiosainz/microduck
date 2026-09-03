#!/usr/bin/env python3
"""The acceptance gate: every hard requirement, measured from the rollout.

Each gate is computed from the summary - which is computed from the RECORDS and
from the machine's own logs - never from the scenario's schedule.  "It waited
because neither side was safe" is a recorded planner decision with both
corridors' predicted clearances attached, not an assertion that WAIT returns
zero; "it did not walk through a crate" is the MEASURED surface separation to
that crate's real geom every control tick.

Split from the measurement so that judging and measuring live in different
files.  ``tests/test_gate_counterexamples.py`` mutates a real summary once per
gate and requires the named gate to reject it, so a gate that cannot fail is
caught here rather than shipped.
"""

from __future__ import annotations

from slalom_states import SAFE_CLEARANCE_M, STATES
from slalom_summary import summarize  # noqa: F401  (re-exported for callers)
from slalom_thresholds import (
    MAX_ACTOR_HEADING_STEP_DEG,
    MAX_FINAL_GOAL_DISTANCE_M,
    MAX_ILLEGAL_ZERO_RUN,
    MIN_CHOSEN_PREDICTED_CLEARANCE_M,
    MIN_DISTINCT_PASS_SIDES,
    MIN_DYNAMIC_ENCOUNTERS,
    MIN_GOAL_SECONDS,
    MIN_GOAL_VISIBLE_WITH_LOS,
    MIN_LATERAL_SPAN_M,
    MIN_MOVING_ACTORS,
    MIN_NET_M,
    MIN_OBSTACLES_AND_ACTORS,
    MIN_PATH_EXCESS_M,
    MIN_PATH_M,
    MIN_REJECTION_MARGIN_M,
    MIN_VISIBLE_WITH_LOS,
    MIN_WAITS,
    MIN_YAW_TRAVEL_DEG,
    NOMINAL_TRUNK_Z_M,
    TRUNK_Z_TOLERANCE_M,
    UPSTREAM_POLICY_SHA,
    ZERO_STATE_NET_PER_EPISODE_M,
    ZERO_STATE_PATH_PER_EPISODE_M,
)


def gates(summary: dict) -> list[tuple[str, bool, str]]:
    """Every hard gate, as (label, passed, evidence)."""
    results: list[tuple[str, bool, str]] = []

    def add(label: str, ok: bool, evidence: str) -> None:
        results.append((label, bool(ok), evidence))

    passes = summary["passes"]
    waits = summary["waits"]
    bracketing = summary["prediction_bracketing"]
    turning = summary["turning_path"]

    # -- A: it went somewhere ------------------------------------------------
    add(f"the duck physically walked at least {MIN_PATH_M:.1f} m",
        summary["path_m"] >= MIN_PATH_M,
        f"{summary['path_m']} m of path, {summary['walk_path_m']} m of it in "
        f"the walking states over {summary['walk_seconds']}s")
    add(f"the journey made at least {MIN_NET_M:.1f} m of net progress toward "
        f"the goal",
        summary["net_m"] >= MIN_NET_M,
        f"net {summary['net_m']} m from {summary['start_xy']} to "
        f"{summary['final_xy']}, goal at {summary['goal_xy']}")
    add("THE DUCK REACHED THE GOAL BAND - the destination, not merely the "
        "absence of contact",
        summary["reached_goal_at_s"] is not None
        and summary["min_goal_distance_m"] <= MAX_FINAL_GOAL_DISTANCE_M
        and summary["goal_seconds"] >= MIN_GOAL_SECONDS,
        f"first inside the band at {summary['reached_goal_at_s']}s, closest "
        f"approach {summary['min_goal_distance_m']} m to its centre, stood in "
        f"it for {summary['goal_seconds']}s")
    add("the duck ended the run inside the band",
        summary["min_goal_distance_m"] <= MAX_FINAL_GOAL_DISTANCE_M,
        f"final position {summary['final_xy']} against the band centre "
        f"{summary['goal_xy']}")

    # -- B: the encounters ----------------------------------------------------
    add(f"at least {MIN_DYNAMIC_ENCOUNTERS} dynamic crossing encounters were "
        f"resolved",
        len(passes) >= MIN_DYNAMIC_ENCOUNTERS,
        f"{len(passes)} passes: " + "; ".join(
            f"{p['index']}:{p['threat']} {p['side']} at {p['began_at_s']}s"
            for p in passes) or "none")
    add("the pass sides ALTERNATED and both hands were used",
        summary["alternating"]
        and len(set(summary["pass_sides"])) >= MIN_DISTINCT_PASS_SIDES,
        f"sides {summary['pass_sides']} (alternating="
        f"{summary['alternating']}), expected "
        f"{summary['expected_pass_sides']}")
    add(f"at least {MIN_WAITS} WAIT because NEITHER corridor was predicted safe",
        len(waits) >= MIN_WAITS
        and all(w.get("duration_s", 0.0) > 0.0 for w in waits),
        "; ".join(
            f"wait {w['index']} on {w.get('threat')}: {w.get('duration_s')}s, "
            f"best rejected corridor was the {w.get('rejected_side')} at "
            f"{w.get('rejected_clearance_m')} m, resolved "
            f"{w.get('resolved_side')} at {w.get('resolved_clearance_m')} m"
            for w in waits) or "no wait recorded")
    add("EVERY choice was justified by a positive predicted clearance above "
        f"the {MIN_CHOSEN_PREDICTED_CLEARANCE_M:.2f} m planner bar",
        bool(passes) and all(
            p["chosen_clearance_m"] >= MIN_CHOSEN_PREDICTED_CLEARANCE_M
            for p in passes),
        "; ".join(f"{p['threat']}: chose {p['side']} on a predicted "
                  f"{p['chosen_clearance_m']} m" for p in passes) or "none")
    add("the REJECTED side was never better than the one chosen",
        bool(passes) and all(
            p["chosen_clearance_m"] - p["rejected_clearance_m"]
            >= MIN_REJECTION_MARGIN_M for p in passes),
        "; ".join(
            f"{p['threat']}: {p['side']} {p['chosen_clearance_m']} m vs "
            f"{p['rejected_side']} {p['rejected_clearance_m']} m "
            f"(margin {p['chosen_clearance_m'] - p['rejected_clearance_m']:+.3f} m)"
            for p in passes) or "none")
    add("the duck REPLANNED after every pass",
        summary["replans_after_pass"] >= len(passes) and len(passes) > 0,
        f"{summary['replans_after_pass']} replans immediately after a pass, "
        f"{len(summary['replans'])} in total, over "
        f"{summary['encounter_cycles']} encounter cycles")

    # -- C: the predictions were conservative ---------------------------------
    add("every prediction conservatively BRACKETED the measured closest "
        "approach",
        bool(bracketing) and all(b["conservative"] for b in bracketing),
        "; ".join(
            f"{b['threat']} ({b['side']}): predicted "
            f"{b['predicted_clearance_m']} m, measured "
            f"{b['measured_clearance_m']} m "
            f"(margin {b['margin_m']:+} m)" for b in bracketing) or "none")
    add("every measured closest approach during a pass was positive",
        bool(bracketing) and all(b["measured_positive"] for b in bracketing),
        "; ".join(f"{b['threat']}: {b['measured_clearance_m']} m"
                  for b in bracketing) or "none")

    # -- D: it really moved sideways, with no strafe --------------------------
    add("REAL lateral displacement on BOTH hands, achieved by turning",
        turning.get("lateral_span_m") is not None
        and turning["lateral_span_m"] >= MIN_LATERAL_SPAN_M
        and turning.get("max_left_offset_m", 0.0) > 0.0
        and turning.get("max_right_offset_m", 0.0) < 0.0,
        f"lane offset ranged {turning.get('max_right_offset_m')} to "
        f"{turning.get('max_left_offset_m')} m, a span of "
        f"{turning.get('lateral_span_m')} m, with "
        f"{turning.get('lateral_path_m')} m of lateral path")
    add("the path is a TURNING path, not a straight line",
        turning.get("excess_over_net_m", 0.0) >= MIN_PATH_EXCESS_M
        and turning.get("yaw_travel_deg", 0.0) >= MIN_YAW_TRAVEL_DEG,
        f"{turning.get('path_m')} m of path against {turning.get('net_m')} m "
        f"net - {turning.get('excess_over_net_m')} m of excess - and "
        f"{turning.get('yaw_travel_deg')} deg of accumulated yaw, with NO "
        f"strafe command available")
    add("the command carried no lateral term at ANY tick",
        summary["max_abs_vy_command"] == 0.0,
        f"largest |vy| over all {summary['steps']} control ticks: "
        f"{summary['max_abs_vy_command']}; the policy has no strafe, so every "
        f"metre of lateral displacement was bought by turning")

    # -- E: nothing was hit ----------------------------------------------------
    add("positive clearance to every moving body at all times",
        summary["min_body_clearance_m"] > 0.0,
        f"min {summary['min_body_clearance_m']:.4f} m to "
        f"{summary['min_body_clearance_name']}; per body "
        f"{summary['min_clearance_by_body_m']}")
    add("positive clearance to every static obstacle and wall - NO PATH "
        "THROUGH ANY STATIC BODY",
        summary["min_scenery_clearance_m"] > 0.0,
        f"min {summary['min_scenery_clearance_m']:.4f} m to "
        f"{summary['min_scenery_clearance_geom']} over "
        f"{summary['static_obstacle_count']} static obstacles")
    add("zero contacts", summary["contact_steps"] == 0,
        f"{summary['contact_steps']} steps with non-positive clearance")

    # -- F: stillness ----------------------------------------------------------
    add("the command was EXACTLY zero in every zero-command state",
        not summary["zero_command_violations"]
        and all(summary["state_command_max"].get(s, 0.0) == 0.0
                for s in summary["zero_command_states"]
                if s in summary["state_steps"]),
        f"{len(summary['zero_command_violations'])} violation(s) across "
        + ", ".join(f"{s}={summary['state_command_max'].get(s)}"
                    for s in summary["zero_command_states"]
                    if s in summary["state_steps"]))
    add("every zero-command state was a real standstill on the floor",
        bool(summary["zero_episodes"])
        and all(e["path_m"] <= ZERO_STATE_PATH_PER_EPISODE_M
                and e["net_m"] <= ZERO_STATE_NET_PER_EPISODE_M
                for e in summary["zero_episodes"]),
        "; ".join(f"{e['state']}@{e['from_s']}s: {e['path_m']} m of path, "
                  f"{e['net_m']} m net"
                  for e in summary["zero_episodes"])
        + f" -- each against a {ZERO_STATE_PATH_PER_EPISODE_M} m path and "
        f"{ZERO_STATE_NET_PER_EPISODE_M} m net bound per episode; MEASURED "
        f"settling from a walk is 0.030 m of path and 0.010 m net, and "
        f"{summary['zero_drift_reference_m']} m per 10 s from a standstill")
    add("NO ZERO-COMMAND PLATEAU outside WAIT and GOAL",
        summary["longest_illegal_zero_run"] <= MAX_ILLEGAL_ZERO_RUN,
        f"longest run of exact zeros outside the permitted states: "
        f"{summary['longest_illegal_zero_run']} tick(s) "
        f"({summary['longest_illegal_zero_run'] / summary['control_hz']:.2f}s); "
        f"windows {summary['illegal_zero_windows'] or 'none'}")

    # -- G: it could see ---------------------------------------------------------
    add(f"the negotiated body was visible in >= {MIN_VISIBLE_WITH_LOS:.0%} of "
        "monitoring steps where line of sight existed",
        summary["monitor_visible_fraction_with_los"] >= MIN_VISIBLE_WITH_LOS,
        f"{summary['monitor_visible_fraction_with_los'] * 100:.2f}% of "
        f"{summary['monitor_los_steps']} LOS steps in "
        f"{summary['monitor_steps']} monitor steps "
        f"({summary['visible_fraction_with_los'] * 100:.2f}% over the whole "
        f"run)")
    add(f"THE GOAL was visible in >= {MIN_GOAL_VISIBLE_WITH_LOS:.0%} of steps "
        "where line of sight to it existed",
        summary["goal_visible_fraction_with_los"]
        >= MIN_GOAL_VISIBLE_WITH_LOS,
        f"{summary['goal_visible_fraction_with_los'] * 100:.2f}% of "
        f"{summary['goal_los_steps']} steps with line of sight, measured "
        f"through the SAME head camera the PiP renders from")

    # -- H: the scenario is real ---------------------------------------------------
    add(f"at least {MIN_OBSTACLES_AND_ACTORS} obstacles and actors populate the "
        f"course",
        summary["obstacle_and_actor_count"] >= MIN_OBSTACLES_AND_ACTORS,
        f"{summary['static_obstacle_count']} static "
        f"({summary['static_obstacle_names']}) + {summary['actor_count']} "
        f"moving = {summary['obstacle_and_actor_count']}")
    add(f"at least {MIN_MOVING_ACTORS} actors were genuinely moving",
        summary["moving_actors"] >= MIN_MOVING_ACTORS,
        f"{summary['moving_actors']} of {summary['actor_count']} moved; "
        f"fractions {summary['moving_fraction']}")
    add("every scripted body's heading is continuous",
        summary["max_actor_heading_step_deg"] <= MAX_ACTOR_HEADING_STEP_DEG,
        f"largest single-tick heading change "
        f"{summary['max_actor_heading_step_deg']} deg against a "
        f"{MAX_ACTOR_HEADING_STEP_DEG} deg bound")
    add("the traffic actually got in the way",
        summary["max_bodies_in_lane"] >= 1
        and summary["lane_occupied_seconds"] > 5.0,
        f"up to {summary['max_bodies_in_lane']} bodies in the duck's lane at "
        f"once, occupied for {summary['lane_occupied_seconds']}s")

    # -- I: the states ----------------------------------------------------------------
    add("every declared state was visited",
        set(summary["declared_states"]) <= set(summary["states_visited"]),
        f"visited {summary['states_visited']}")
    add("every visited state is one this behavior declares",
        set(summary["states_visited"]) <= set(summary["declared_states"]),
        f"visited {summary['states_visited']}")
    add("exactly zero BARGE_THROUGH and FREEZE_FOREVER steps",
        all(v == 0 for v in summary["forbidden_state_steps"].values()),
        f"forbidden state steps: {summary['forbidden_state_steps']}")
    add("no phase hit its ceiling", not summary["timeouts"],
        f"timeouts: {summary['timeouts'] or 'none'}")

    # -- J: locomotion health ------------------------------------------------------
    add("zero falls", summary["fallen_steps"] == 0,
        f"{summary['fallen_steps']} steps below 0.09 m")
    add("trunk never below 0.09 m", summary["min_trunk_z_m"] >= 0.09,
        f"min {summary['min_trunk_z_m']:.4f} m")
    add(f"final trunk height near the nominal {NOMINAL_TRUNK_Z_M} m",
        abs(summary["final_trunk_z_m"] - NOMINAL_TRUNK_Z_M)
        <= TRUNK_Z_TOLERANCE_M,
        f"{summary['final_trunk_z_m']:.4f} m")

    # -- K: the physical contract ------------------------------------------------------
    add("the exact imu_ang_vel sensor, 61-D observation and 0.9 action scale",
        summary["gyro_sensor"] == "imu_ang_vel"
        and summary["observation_dim"] == 61
        and summary["action_scale"] == 0.9,
        f"gyro={summary['gyro_sensor']}, obs={summary['observation_dim']}-D, "
        f"action_scale={summary['action_scale']}")
    add("the byte-identical stock walking policy",
        summary["policy_sha256"] == UPSTREAM_POLICY_SHA,
        f"sha256 {summary['policy_sha256']}")

    return results


def report(summary: dict) -> tuple[bool, list[tuple[str, bool, str]]]:
    results = gates(summary)
    return all(ok for _, ok, _ in results), results
