#!/usr/bin/env python3
"""The acceptance gate: every hard requirement, measured from the rollout.

Each gate is computed from the summary - which is computed from the RECORDS and
from the machine's own logs - never from the scenario's schedule.  "It stopped at
the checkpoint and scanned" is a MEASURED stationary time and a MEASURED head
arc, not an assertion that SCAN returns zero; "it never entered the restricted
zone" is the MEASURED signed distance from its trunk to the marked rectangle,
every control tick.

Split from the measurement so that judging and measuring live in different
files.  ``tests/test_gate_counterexamples.py`` mutates a real summary once per
gate and requires the named gate to reject it, so a gate that cannot fail is
caught here rather than shipped.
"""

from __future__ import annotations

from patrol_summary import summarize  # noqa: F401  (re-exported for callers)
from patrol_thresholds import (
    MAX_ACTOR_HEADING_STEP_DEG,
    MAX_CHECKPOINT_ERROR_M,
    MAX_HOME_DISTANCE_M,
    MAX_ILLEGAL_ZERO_RUN,
    MAX_RETURN_ERROR_M,
    MAX_STANDOFF_M,
    MAX_STILL_PATH_M,
    MIN_ANGLE_VISIBLE_FRACTION,
    MIN_APPROACH_PATH_M,
    MIN_BODIES,
    MIN_CAMERA_ACTIVE,
    MIN_CAMERA_GATE_TICKS,
    MIN_CHECKPOINT_STOP_S,
    MIN_FIXTURES,
    MIN_HOME_SECONDS,
    MIN_MOVING_BODIES,
    MIN_PATH_M,
    MIN_RANGE_REDUCTION_M,
    MIN_SCAN_ARC_DEG,
    MIN_STANDOFF_M,
    MIN_VISIBLE_WITH_LOS,
    NOMINAL_TRUNK_Z_M,
    REQUIRED_CHECKPOINTS,
    REQUIRED_DISMISSALS,
    REQUIRED_INVESTIGATIONS,
    REQUIRED_OBSERVE_ANGLES,
    REQUIRED_VERDICTS,
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

    visits = summary["visits"]
    investigations = summary["standoff_investigations"]
    scans = summary["scan_scans"]

    # -- A: the patrol was a patrol -------------------------------------------
    add(f"all {REQUIRED_CHECKPOINTS} checkpoints were visited IN THE DECLARED "
        "ORDER, exactly once each",
        summary["checkpoint_in_declared_order"]
        and summary["checkpoint_all_visited"]
        and summary["checkpoint_no_repeats"]
        and summary["checkpoint_count"] == REQUIRED_CHECKPOINTS,
        f"visited {summary['checkpoint_visited_order']} against the declared "
        f"{summary['checkpoint_declared_order']}")
    add("the duck stopped ON each checkpoint",
        bool(visits) and all(
            v["arrival_error_m"] <= MAX_CHECKPOINT_ERROR_M for v in visits),
        "; ".join(f"{v['name']} {v['arrival_error_m']:.3f} m"
                  for v in visits)
        + f" -- each against a {MAX_CHECKPOINT_ERROR_M} m bar")
    add("THE FULL LOOP WAS COMPLETED and the duck returned home",
        summary["reached_home_at_s"] is not None
        and summary["min_home_distance_m"] <= MAX_HOME_DISTANCE_M
        and summary["home_seconds"] >= MIN_HOME_SECONDS,
        f"first on the guard-post pad at {summary['reached_home_at_s']}s, "
        f"closest {summary['min_home_distance_m']} m to its centre, stood "
        f"there {summary['home_seconds']}s")
    add(f"the duck physically walked at least {MIN_PATH_M:.1f} m",
        summary["path_m"] >= MIN_PATH_M,
        f"{summary['path_m']} m of path over a {summary['circuit_length_m']} m "
        f"circuit plus two investigations, {summary['walk_path_m']} m of it in "
        f"the walking states over {summary['walk_seconds']}s")

    # -- B: each checkpoint was a real stop and a real scan --------------------
    add("every checkpoint was a REAL STOP: exact zero, and stationary on the "
        "floor",
        bool(visits)
        and all(v["stopped_s"] >= MIN_CHECKPOINT_STOP_S for v in visits)
        and summary["scan_max_still_path_m"] <= MAX_STILL_PATH_M,
        f"shortest stop {summary['scan_min_stopped_s']}s, worst path during a "
        f"stop-and-scan {summary['scan_max_still_path_m']} m against a "
        f"{MAX_STILL_PATH_M} m bar; MEASURED zero-command drift is "
        f"{summary['zero_drift_reference_m']} m per 10 s")
    add(f"every COMPLETED scan swept at least {MIN_SCAN_ARC_DEG:.0f} deg of "
        "real head travel",
        bool(summary["scan_completed_scan_arcs_deg"])
        and summary["scan_min_completed_scan_arc_deg"] >= MIN_SCAN_ARC_DEG,
        f"completed sweeps {summary['scan_completed_scan_arcs_deg']} deg; the "
        f"arc is MEASURED from the pose the head actually reached, and a scan "
        f"cut short by a detection legitimately sweeps less")
    add("the camera resolved real bodies during the scans",
        all(v["bodies_seen"] is not None for v in scans)
        and any(v["bodies_seen"] for v in scans),
        "; ".join(f"{v['checkpoint']}: {v['bodies_seen'] or 'nobody'}"
                  for v in scans))

    # -- C: the anomalies -------------------------------------------------------
    add(f"{REQUIRED_VERDICTS} anomalies were classified, and EVERY verdict was "
        "correct",
        summary["verdict_all_correct"]
        and len(summary["verdict_verdicts"]) >= REQUIRED_VERDICTS,
        f"{summary['verdict_verdict_by_target']} against the scenario's "
        f"{summary['verdict_expected']}")
    add(f"{REQUIRED_INVESTIGATIONS} DISTINCT anomalies were investigated: a "
        "suspicious unattended object and an unauthorised person in a "
        "restricted zone",
        len(summary["verdict_investigated"]) >= REQUIRED_INVESTIGATIONS
        and set(summary["verdict_investigated"])
        == set(summary["verdict_expected_investigated"]),
        f"investigated {summary['verdict_investigated']}, expected "
        f"{summary['verdict_expected_investigated']}")
    add(f"the benign distractor was EXPLICITLY DISMISSED, with a recorded rule",
        len(summary["verdict_dismissed"]) >= REQUIRED_DISMISSALS
        and set(summary["verdict_dismissed"])
        == set(summary["verdict_expected_dismissed"])
        and all(v["rule"] for v in summary["verdict_verdicts"]
                if v["verdict"] == "benign"),
        "; ".join(f"{v['target']}: {v['rule']}"
                  for v in summary["verdict_verdicts"]
                  if v["verdict"] == "benign") or "nothing dismissed")
    add("EVERY anomaly was detected only INSIDE the camera gate",
        bool(summary["verdict_camera_gate_ticks"]) and all(
            summary["verdict_camera_gate_ticks"].get(name, 0)
            >= MIN_CAMERA_GATE_TICKS
            for name in summary["verdict_verdict_by_target"]),
        "; ".join(
            f"{name}: {summary['verdict_camera_gate_ticks'].get(name, 0)} "
            f"ticks in frustum, first at "
            f"{summary['verdict_first_in_camera_gate_s'].get(name)}s"
            for name in summary["verdict_verdict_by_target"])
        + f" -- each against {MIN_CAMERA_GATE_TICKS} ticks")

    # -- D: the investigations were physical -------------------------------------
    add(f"BOTH approaches physically reduced the range by at least "
        f"{MIN_RANGE_REDUCTION_M:.2f} m",
        len(investigations) >= REQUIRED_INVESTIGATIONS and all(
            i["range_reduction_m"] >= MIN_RANGE_REDUCTION_M
            for i in investigations),
        "; ".join(f"{i['target']}: {i['approach_start_range_m']:.3f} -> "
                  f"{i['approach_end_range_m']:.3f} m "
                  f"({i['range_reduction_m']:+.3f} m)"
                  for i in investigations) or "none")
    add("each approach was a REAL WALK on the floor",
        bool(investigations) and all(
            i["approach_path_m"] >= MIN_APPROACH_PATH_M
            for i in investigations),
        "; ".join(f"{i['target']}: {i['approach_path_m']:.3f} m of path"
                  for i in investigations)
        + f" -- each against {MIN_APPROACH_PATH_M} m")
    add(f"each approach STOPPED inside the {MIN_STANDOFF_M}-{MAX_STANDOFF_M} m "
        "safe observation standoff band",
        bool(investigations) and all(
            MIN_STANDOFF_M <= i["min_clearance_m"] <= MAX_STANDOFF_M
            for i in investigations),
        "; ".join(f"{i['target']}: closest MEASURED surface clearance "
                  f"{i['min_clearance_m']:.4f} m"
                  for i in investigations) or "none")
    add(f"each observation held all {REQUIRED_OBSERVE_ANGLES} declared viewing "
        "angles, with the target in frame",
        bool(investigations) and all(
            i["angles_held"] >= REQUIRED_OBSERVE_ANGLES
            and all(o["visible_fraction"] >= MIN_ANGLE_VISIBLE_FRACTION
                    for o in i["observations"])
            for i in investigations),
        "; ".join(
            f"{i['target']}: " + ", ".join(
                f"{o['angle_deg']:+.0f}deg {o['held_s']:.1f}s "
                f"{o['visible_fraction'] * 100:.0f}% seen"
                for o in i["observations"])
            for i in investigations) or "none")
    add("the command was EXACTLY zero throughout every observation",
        bool(investigations)
        and summary["state_command_max"].get("OBSERVE", 0.0) == 0.0
        and summary["state_command_max"].get("CLASSIFY", 0.0) == 0.0,
        f"OBSERVE max command {summary['state_command_max'].get('OBSERVE')}, "
        f"CLASSIFY max {summary['state_command_max'].get('CLASSIFY')}")

    # -- E: the route memory ------------------------------------------------------
    add("the patrol was INTERRUPTED and the original checkpoint was PRESERVED",
        summary["memory_count"] >= REQUIRED_INVESTIGATIONS
        and summary["memory_all_preserved"],
        "; ".join(
            f"broke off toward {e['target_name']} at {e['at_s']}s for "
            f"{e['anomaly']}, resumed toward {e['resumed_target_name']}"
            for e in summary["memory_interruptions"]) or "never interrupted")
    add("the duck physically RETURNED to the point it broke off at, before "
        "advancing",
        summary["memory_max_return_error_m"] is not None
        and summary["memory_max_return_error_m"] <= MAX_RETURN_ERROR_M,
        "; ".join(
            f"{e['anomaly']}: came back within {e['return_error_m']:.3f} m of "
            f"{e['resume_xy']}" for e in summary["memory_interruptions"])
        + f" -- each against a {MAX_RETURN_ERROR_M} m bar")

    # -- F: nothing was touched or entered ------------------------------------------
    add("THE DUCK NEVER ENTERED THE RESTRICTED ZONE",
        summary["min_zone_gap_m"] > 0.0 and summary["zone_breach_steps"] == 0,
        f"closest approach to the {summary['restricted_zone']['name']} "
        f"rectangle {summary['min_zone_gap_m']:+.4f} m over "
        f"{summary['steps']} control ticks, {summary['zone_breach_steps']} "
        f"breaches")
    add("positive clearance to every body at all times - NO CONTACT with any "
        "anomaly or person",
        summary["min_body_clearance_m"] > 0.0,
        f"min {summary['min_body_clearance_m']:.4f} m to "
        f"{summary['min_body_clearance_name']}; per body "
        f"{summary['min_clearance_by_body_m']}")
    add("positive clearance to every fixture and wall",
        summary["min_scenery_clearance_m"] > 0.0,
        f"min {summary['min_scenery_clearance_m']:.4f} m to "
        f"{summary['min_scenery_clearance_geom']} over "
        f"{summary['fixture_count']} fixtures")
    add("zero contacts", summary["contact_steps"] == 0,
        f"{summary['contact_steps']} steps with non-positive clearance")

    # -- G: stillness and no decorative commands --------------------------------------
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
        f"worst episode {summary['worst_zero_episode_path_m']} m of path and "
        f"{summary['worst_zero_episode_net_m']} m net over "
        f"{len(summary['zero_episodes'])} episodes, against "
        f"{ZERO_STATE_PATH_PER_EPISODE_M} m and "
        f"{ZERO_STATE_NET_PER_EPISODE_M} m per episode")
    add("NO ZERO-COMMAND PLATEAU outside the states allowed to hold one",
        summary["longest_illegal_zero_run"] <= MAX_ILLEGAL_ZERO_RUN,
        f"longest run of exact zeros outside the permitted states: "
        f"{summary['longest_illegal_zero_run']} tick(s) "
        f"({summary['longest_illegal_zero_run'] / summary['control_hz']:.2f}s); "
        f"windows {summary['illegal_zero_windows'] or 'none'}")
    add("NO DECORATIVE COMMANDS: no lateral term at ANY tick, and no "
        "turn-in-place",
        summary["max_abs_vy_command"] == 0.0,
        f"largest |vy| over all {summary['steps']} control ticks: "
        f"{summary['max_abs_vy_command']}; turning in place was MEASURED at "
        f"{summary['spin_rate_measured_dps']} deg/s and is never commanded, "
        f"which is why every scan is a HEAD sweep")

    # -- H: it could see -----------------------------------------------------------
    add(f"the camera was ACTIVE on a target in >= {MIN_CAMERA_ACTIVE:.0%} of "
        "control ticks",
        summary["camera_active_fraction"] >= MIN_CAMERA_ACTIVE,
        f"{summary['camera_active_fraction'] * 100:.2f}% of "
        f"{summary['steps']} ticks ({summary['camera_active_steps']} active)")
    add(f"the investigated body was visible in >= {MIN_VISIBLE_WITH_LOS:.0%} of "
        "monitoring steps where line of sight existed",
        summary["monitor_visible_fraction_with_los"] >= MIN_VISIBLE_WITH_LOS,
        f"{summary['monitor_visible_fraction_with_los'] * 100:.2f}% of "
        f"{summary['monitor_los_steps']} LOS steps in "
        f"{summary['monitor_steps']} monitor steps "
        f"({summary['visible_fraction_with_los'] * 100:.2f}% over the run)")

    # -- I: the facility is real -------------------------------------------------------
    add(f"the facility is populated: at least {MIN_BODIES} bodies and "
        f"{MIN_FIXTURES} fixtures",
        summary["body_count"] >= MIN_BODIES
        and summary["fixture_count"] >= MIN_FIXTURES,
        f"{summary['body_count']} bodies ({summary['body_names']}) and "
        f"{summary['fixture_count']} fixtures")
    add(f"at least {MIN_MOVING_BODIES} bodies were genuinely moving",
        summary["moving_bodies"] >= MIN_MOVING_BODIES,
        f"{summary['moving_bodies']} of {summary['body_count']} moved; "
        f"fractions {summary['moving_fraction']}")
    add("every scripted person's heading is continuous",
        summary["max_actor_heading_step_deg"] <= MAX_ACTOR_HEADING_STEP_DEG,
        f"largest single-tick heading change "
        f"{summary['max_actor_heading_step_deg']} deg against a "
        f"{MAX_ACTOR_HEADING_STEP_DEG} deg bound")
    add("exactly ONE person entered the restricted zone, and the duck named "
        "that person",
        sorted(k for k, v in summary["zone_occupancy_s"].items() if v > 0.0)
        == ["visitor"]
        and summary["verdict_verdict_by_target"].get("visitor") == "intrusion",
        f"seconds inside the rectangle: "
        f"{ {k: v for k, v in summary['zone_occupancy_s'].items() if v > 0} }; "
        f"the duck classified "
        f"{summary['verdict_verdict_by_target'].get('visitor')}")

    # -- J: the states -----------------------------------------------------------------
    add("every declared state was visited",
        set(summary["declared_states"]) <= set(summary["states_visited"]),
        f"visited {summary['states_visited']}")
    add("every visited state is one this behavior declares",
        set(summary["states_visited"]) <= set(summary["declared_states"]),
        f"visited {summary['states_visited']}")
    add("exactly zero ABANDON_PATROL and CONTACT_TARGET steps",
        all(v == 0 for v in summary["forbidden_state_steps"].values()),
        f"forbidden state steps: {summary['forbidden_state_steps']}")
    add("no phase hit its ceiling", not summary["timeouts"],
        f"timeouts: {summary['timeouts'] or 'none'}")

    # -- K: locomotion health ------------------------------------------------------------
    add("zero falls", summary["fallen_steps"] == 0,
        f"{summary['fallen_steps']} steps below 0.09 m")
    add("trunk never below 0.09 m", summary["min_trunk_z_m"] >= 0.09,
        f"min {summary['min_trunk_z_m']:.4f} m")
    add(f"final trunk height near the nominal {NOMINAL_TRUNK_Z_M} m",
        abs(summary["final_trunk_z_m"] - NOMINAL_TRUNK_Z_M)
        <= TRUNK_Z_TOLERANCE_M,
        f"{summary['final_trunk_z_m']:.4f} m")

    # -- L: the physical contract ----------------------------------------------------------
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
