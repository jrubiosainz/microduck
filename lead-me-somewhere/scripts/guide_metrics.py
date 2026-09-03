#!/usr/bin/env python3
"""The acceptance gate: every hard requirement, measured from the rollout.

Each gate is computed from the summary — which is computed from the RECORDS and
from the machine's own logs — never from the scenario's schedule.  "The duck
detected that she was lagging" is a count of episodes the machine opened on
measured distance and measured visibility, not a lookup of when
``guide_follower`` decided she would stop; "the duck waited" is the maximum
command it emitted while waiting, not an assertion that WAIT_FOR_PERSON returns
zero.

Split from the measurement so that judging and measuring live in different
files.  ``tests/test_gate_counterexamples.py`` mutates a real summary once per
gate and requires the named gate to reject it, so a gate that cannot fail is
caught here rather than shipped.
"""

from __future__ import annotations

from guide_states import (
    FACE_TOLERANCE_DEG,
    INDICATE_SECONDS,
)
from guide_summary import summarize  # noqa: F401  (re-exported for callers)
from guide_thresholds import (
    CHECK_STILL_PATH_M,
    FINAL_DISTANCE_BAND_M,
    MIN_BENDS,
    MIN_CLOSED_DISTANCE_M,
    MIN_CROWD_BLOCKED_CELLS,
    MIN_DESTINATION_CANDIDATES,
    MIN_EPISODES,
    MIN_FOLLOWER_WALKED_M,
    MIN_LEAD_NET_M,
    MIN_LEAD_PATH_M,
    MIN_MOVING_ADULTS,
    MIN_VISIBLE_WITH_LOS,
    MIN_WAIT_SECONDS,
    UPSTREAM_POLICY_SHA,
)



def gates(summary: dict) -> list[tuple[str, bool, str]]:
    """Every hard gate, as (label, passed, evidence)."""
    results: list[tuple[str, bool, str]] = []

    def add(label: str, ok: bool, evidence: str) -> None:
        results.append((label, bool(ok), evidence))

    plan = summary["plan"] or {}
    episodes = summary["episodes"]

    # -- the request ---------------------------------------------------------
    add(f"the request was resolved to the correct one of "
        f">= {MIN_DESTINATION_CANDIDATES} candidates",
        summary["resolution_correct"]
        and summary["destination_candidate_count"] >= MIN_DESTINATION_CANDIDATES,
        f"requested {summary['requested_destination']!r} at "
        f"{summary['request_t_s']}s, resolved to "
        f"{summary['resolved_destination']!r} out of "
        f"{summary['destination_candidates']}")
    add("the duck acknowledged before moving, at exactly zero command",
        summary["state_command_max"].get("RECEIVE_DESTINATION", 1.0) == 0.0
        and summary["state_seconds"].get("RECEIVE_DESTINATION", 0.0) > 0.0,
        f"RECEIVE_DESTINATION for "
        f"{summary['state_seconds'].get('RECEIVE_DESTINATION')}s at max command "
        f"{summary['state_command_max'].get('RECEIVE_DESTINATION')}")

    # -- the plan ------------------------------------------------------------
    add(f"the planned route has at least {MIN_BENDS} bends",
        len(plan.get("bends", [])) >= MIN_BENDS,
        f"{plan.get('bend_count')} bends: "
        + ", ".join(f"{b['hand']} {b['turn_deg']:+.1f} deg"
                    for b in plan.get("bends", [])))
    add("the planned route avoids the inflated obstacles",
        plan.get("min_planned_clearance_m", -1.0)
        >= plan.get("route_clearance_required_m", 0.0),
        f"min planned clearance {plan.get('min_planned_clearance_m')} m at "
        f"{plan.get('min_planned_clearance_at')}, required "
        f"{plan.get('route_clearance_required_m')} m "
        f"(corner radius {plan.get('corner_radius_m')} m, "
        f"{plan.get('corner_radii_tried')} tried)")
    add("the planner refused cells BECAUSE of the crowd, not only the walls",
        plan.get("crowd_blocked_cells", 0) >= MIN_CROWD_BLOCKED_CELLS,
        f"{plan.get('crowd_blocked_cells')} crowd-blocked cells "
        f"{plan.get('crowd_blockers')} against "
        f"{plan.get('static_blocked_cells')} static-blocked")
    add("the route is a genuine detour, not the straight line",
        plan.get("detour_ratio", 0.0) >= 1.25
        and bool(plan.get("straight_line_blocked_by")),
        f"{plan.get('length_m')} m against a {plan.get('straight_line_m')} m "
        f"straight line (x{plan.get('detour_ratio')}), which "
        f"{plan.get('straight_line_blocked_by')} blocks")

    # -- the lead ------------------------------------------------------------
    add(f"the duck physically led at least {MIN_LEAD_PATH_M:.1f} m of path",
        summary["lead_path_m"] >= MIN_LEAD_PATH_M,
        f"{summary['lead_path_m']} m walked in LEAD/RESUME over "
        f"{summary['lead_seconds']}s")
    add(f"the lead made at least {MIN_LEAD_NET_M:.1f} m of net progress",
        summary["lead_net_m"] >= MIN_LEAD_NET_M,
        f"net {summary['lead_net_m']} m between the first and last leading tick")
    add(f"the person actually walked at least {MIN_FOLLOWER_WALKED_M:.1f} m "
        "behind the duck",
        summary["follower_walked_m"] >= MIN_FOLLOWER_WALKED_M,
        f"she walked {summary['follower_walked_m']} m along the duck's own "
        f"trail")
    add("the duck LED: the follower was never ahead of it on the path",
        summary["follower_ahead_steps"] == 0
        and summary["min_lead_gap_m"] > 0.0,
        f"{summary['follower_ahead_steps']} step(s) with her ahead; minimum "
        f"trail gap {summary['min_lead_gap_m']} m against a "
        f"{summary['trail_gap_floor_m']} m floor")
    add("the duck stayed on its planned route",
        summary["max_cross_track_m"] <= 0.45,
        f"max cross-track {summary['max_cross_track_m']} m")

    # -- the episodes --------------------------------------------------------
    add(f"at least {MIN_EPISODES} lag/loss episodes were DETECTED from "
        "measurement",
        summary["episode_count"] >= MIN_EPISODES,
        "; ".join(f"episode {e['index']}: {e['cause']} at {e['detected_at_s']}s "
                  f"(distance {e['distance_at_detect_m']} m, visible "
                  f"{e['visible_at_detect']}, lagging for "
                  f"{e['lagging_for_s']}s, unseen for {e['unseen_for_s']}s)"
                  for e in episodes))
    add("every declared stall produced a detected episode",
        all(s["detected"] for s in summary["declared_stalls"]),
        "; ".join(f"{s['label']!r} {s['start_s']}-{s['end_s']}s -> "
                  f"episodes {s['episode_indices']} "
                  f"(lag {s['detection_lag_s']}s)"
                  for s in summary["declared_stalls"]))
    add("the command was EXACTLY zero for every WAITING tick",
        all(e["max_command_while_waiting"] == 0.0 for e in episodes)
        and not summary["zero_command_violations"],
        "; ".join(f"episode {e['index']}: max command "
                  f"{e['max_command_while_waiting']} over "
                  f"{e['wait_only_seconds']}s of WAIT_FOR_PERSON, duck moved "
                  f"{e['duck_moved_while_waiting_m']} m"
                  for e in episodes)
        + f"; {len(summary['zero_command_violations'])} violation(s) in "
        f"{summary['zero_command_states']}")
    add("the duck was equally still while CHECKING, not only while waiting",
        all(e["squaring_up_path_m"] <= CHECK_STILL_PATH_M for e in episodes),
        "; ".join(f"episode {e['index']}: drifted {e['squaring_up_path_m']} m "
                  f"in CHECK_FOLLOWER against a {CHECK_STILL_PATH_M} m bound "
                  f"(10 s of exact zero MEASURES 0.0057 m of path)"
                  for e in episodes))
    add(f"every wait lasted at least {MIN_WAIT_SECONDS:.1f}s and she closed "
        f"at least {MIN_CLOSED_DISTANCE_M:.2f} m",
        len(episodes) > 0
        and all(e["wait_duration_s"] >= MIN_WAIT_SECONDS
                and e["follower_closed_m"] >= MIN_CLOSED_DISTANCE_M
                for e in episodes),
        "; ".join(f"episode {e['index']}: waited {e['wait_duration_s']}s, "
                  f"she closed {e['follower_closed_m']} m"
                  for e in episodes))
    add("every resume was justified by measured catch-up AND visibility",
        len(episodes) > 0
        and all(e["distance_at_resume_m"] <= summary["catchup_threshold_m"]
                and e["visible_at_resume"]
                and e["recovered_for_s"] >= summary["resume_confirm_s"]
                for e in episodes),
        "; ".join(f"episode {e['index']}: resumed at "
                  f"{e['distance_at_resume_m']} m <= "
                  f"{e['catchup_threshold_m']} m, visible "
                  f"{e['visible_at_resume']}, sustained {e['recovered_for_s']}s"
                  for e in episodes))
    add("each waiting spot kept positive clearance to the scenery",
        len(episodes) > 0
        and all(e["waiting_spot_scenery_clearance_m"] > 0.0 for e in episodes),
        "; ".join(f"episode {e['index']}: waited at {e['waiting_spot_xy']} "
                  f"with {e['waiting_spot_scenery_clearance_m']} m clearance"
                  for e in episodes))
    add("the duck never left her beyond the safety maximum for a prolonged "
        "interval",
        summary["max_safety_breach_s"] < summary["safety_max_interval_s"],
        f"longest continuous interval beyond "
        f"{summary['safety_max_distance_m']} m was "
        f"{summary['max_safety_breach_s']}s against a "
        f"{summary['safety_max_interval_s']}s limit "
        f"(max range {summary['max_follower_range_m']} m)")

    # -- the arrival ---------------------------------------------------------
    add("the duck reached the destination it was ASKED for",
        summary["arrived"]
        and summary["arrival"].get("destination")
        == summary["requested_destination"],
        f"arrived at {summary['arrival'].get('destination')!r} at "
        f"{summary['arrival'].get('arrived_at_s')}s, requested "
        f"{summary['requested_destination']!r}")
    add("the final distance to the destination is inside its band",
        summary["final_destination_distance_m"] is not None
        and FINAL_DISTANCE_BAND_M[0] <= summary["final_destination_distance_m"]
        <= FINAL_DISTANCE_BAND_M[1],
        f"{summary['final_destination_distance_m']} m against band "
        f"{summary['final_destination_band_m']}")
    add("the duck finished facing the destination",
        summary["final_facing_error_deg"] is not None
        and summary["final_facing_error_deg"] <= summary["face_tolerance_deg"],
        f"facing error {summary['final_facing_error_deg']} deg <= "
        f"{summary['face_tolerance_deg']} deg")
    add(f"the arrival was indicated for at least {INDICATE_SECONDS:.0f}s",
        summary["indicate_seconds"] >= INDICATE_SECONDS - 0.05,
        f"INDICATE for {summary['indicate_seconds']}s")
    add("the person finished safely nearby",
        summary["final_follower_distance_m"] <= summary["final_person_near_m"],
        f"she finished {summary['final_follower_distance_m']} m away, limit "
        f"{summary['final_person_near_m']} m")
    add("the duck arrived BEFORE her, as a guide must",
        summary["min_lead_gap_m"] > 0.0
        and summary["follower_trail_gap_final_m"] > 0.0,
        f"final trail gap {summary['follower_trail_gap_final_m']} m")

    # -- visibility ----------------------------------------------------------
    add(f"the follower was visible in >= {MIN_VISIBLE_WITH_LOS:.0%} of "
        "CHECK/WAIT steps where line of sight existed",
        summary["monitor_visible_fraction_with_los"] >= MIN_VISIBLE_WITH_LOS,
        f"{summary['monitor_visible_fraction_with_los'] * 100:.2f}% of "
        f"{summary['monitor_los_steps']} LOS steps in "
        f"{summary['monitor_steps']} monitor steps "
        f"({summary['visible_fraction_with_los'] * 100:.2f}% over the whole "
        f"run)")

    # -- the populated hall ---------------------------------------------------
    add(f"at least {MIN_MOVING_ADULTS} other adults were moving in the hall",
        summary["moving_adults"] >= MIN_MOVING_ADULTS,
        f"{summary['moving_adults']} of {len(summary['other_adults'])} other "
        f"adults changed position: {summary['other_adults']}")

    # -- safety ---------------------------------------------------------------
    add("positive clearance to every person at all times",
        summary["min_person_clearance_m"] > 0.0,
        f"min {summary['min_person_clearance_m']:.4f} m to "
        f"{summary['min_person_clearance_name']}")
    add("positive clearance to every obstacle, wall and fixture",
        summary["min_scenery_clearance_m"] > 0.0,
        f"min {summary['min_scenery_clearance_m']:.4f} m to "
        f"{summary['min_scenery_clearance_geom']}")
    add("zero contacts", summary["contact_steps"] == 0,
        f"{summary['contact_steps']} steps with non-positive clearance")

    # -- states ---------------------------------------------------------------
    add("every declared state was visited",
        set(summary["declared_states"]) <= set(summary["states_visited"]),
        f"visited {summary['states_visited']}")
    add("every visited state is one this behavior declares",
        set(summary["states_visited"]) <= set(summary["declared_states"]),
        f"visited {summary['states_visited']}")
    add("exactly zero ABANDON and SEARCH steps",
        all(v == 0 for v in summary["forbidden_state_steps"].values()),
        f"forbidden state steps: {summary['forbidden_state_steps']}")
    add("no phase hit its ceiling", not summary["timeouts"],
        f"timeouts: {summary['timeouts'] or 'none'}")

    # -- locomotion health ----------------------------------------------------
    add("zero falls", summary["fallen_steps"] == 0,
        f"{summary['fallen_steps']} steps below 0.09 m")
    add("trunk never below 0.09 m", summary["min_trunk_z_m"] >= 0.09,
        f"min {summary['min_trunk_z_m']:.4f} m")
    add("final trunk height near the nominal 0.116 m",
        abs(summary["final_trunk_z_m"] - 0.116) <= 0.012,
        f"{summary['final_trunk_z_m']:.4f} m")

    # -- the physical contract ------------------------------------------------
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
