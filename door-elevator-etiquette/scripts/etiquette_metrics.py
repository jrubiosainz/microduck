#!/usr/bin/env python3
"""The acceptance gate: every hard requirement, measured from the rollout.

Each gate is computed from the summary - which is computed from the RECORDS and
from the machine's own logs - never from the scenario's schedule.  "The duck let
them out first" is a count of measured occupancy ticks, not an assertion that
LET_OCCUPANTS_EXIT returns zero; "it did not walk through a closed door" is the
door's MEASURED open fraction at the instant the duck's own footprint entered the
aperture.

Split from the measurement so that judging and measuring live in different files.
``tests/test_gate_counterexamples.py`` mutates a real summary once per gate and
requires the named gate to reject it, so a gate that cannot fail is caught here
rather than shipped.
"""

from __future__ import annotations

from etiquette_states import STATES
from etiquette_summary import summarize  # noqa: F401  (re-exported for callers)
from etiquette_thresholds import (
    MAX_EARLY_ZONE_STEPS,
    MAX_GUARDIAN_GAP_M,
    MAX_OVERTAKE_STEPS,
    MAX_SHARED_APERTURE_STEPS,
    MIN_CABIN_MARGIN_M,
    MIN_CABIN_SECONDS,
    MIN_DOOR_EXITERS,
    MIN_EFFECTIVE_GAP_AT_CROSSING_M,
    MIN_MOVING_ADULTS,
    MIN_NET_M,
    MIN_OCCUPANTS_OUT,
    MIN_OPEN_FRACTION_AT_CROSSING,
    MIN_PATH_M,
    MIN_RIDE_SECONDS,
    MIN_VISIBLE_WITH_LOS,
    MIN_WAIT_SIDE_SECONDS,
    MIN_YIELD_SECONDS,
    UPSTREAM_POLICY_SHA,
    ZERO_STATE_PATH_M,
)


def gates(summary: dict) -> list[tuple[str, bool, str]]:
    """Every hard gate, as (label, passed, evidence)."""
    results: list[tuple[str, bool, str]] = []

    def add(label: str, ok: bool, evidence: str) -> None:
        results.append((label, bool(ok), evidence))

    yields = summary["yields"]
    boarding = summary["boarding"]
    crossings = {c["aperture"]: c for c in summary["crossings"]}

    # -- the route is physical -----------------------------------------------
    add(f"the duck physically walked at least {MIN_PATH_M:.1f} m",
        summary["path_m"] >= MIN_PATH_M,
        f"{summary['path_m']} m of path, {summary['walk_path_m']} m of it in "
        f"the walking states over {summary['walk_seconds']}s")
    add(f"the journey made at least {MIN_NET_M:.1f} m of net progress",
        summary["net_m"] >= MIN_NET_M,
        f"net {summary['net_m']} m from start to finish")
    add("every bend in the route is inside the MEASURED turning circle "
        "for its own sign",
        all(b["walkable"] for b in summary["route_bends"]),
        "; ".join(f"{b['hand']} {b['turn_deg']:+.1f} deg at r={b['radius_m']} m "
                  f"(needs {b['min_radius_for_hand_m']} m)"
                  for b in summary["route_bends"]))
    add("the route passes through the middle of every aperture",
        all(c.get("crossed") and c["margin_m"] > 0.20
            for c in summary["route_crossings"]),
        "; ".join(f"{c['aperture']}: {c['offset_from_centre_m']:+.4f} m off "
                  f"centre, {c['margin_m']} m to the jamb"
                  for c in summary["route_crossings"]))
    add("the duck stayed on its own route",
        summary["max_cross_track_m"] <= 0.45,
        f"max cross-track {summary['max_cross_track_m']} m")

    # -- A: the doorway -------------------------------------------------------
    add("the duck yielded at the doorway before entering it",
        len(yields) >= 1
        and all(y.get("duration_s", 0.0) >= MIN_YIELD_SECONDS for y in yields),
        "; ".join(f"yield {y['index']} ({y['kind']}): stopped {y['began_at_s']}s, "
                  f"released {y.get('ended_at_s')}s, "
                  f"{y.get('duration_s')}s with "
                  f"{y.get('exiters_pending_at_stop')} exiter(s) pending"
                  for y in yields) or "no yield recorded")
    add("NO THRESHOLD ENCROACHMENT before the exiters were clear",
        summary["zone_violation_steps"].get(
            "concourse_door_threshold", 0) <= MAX_EARLY_ZONE_STEPS,
        f"{summary['zone_violation_steps'].get('concourse_door_threshold', 0)} "
        f"early step(s) in the door threshold band; worst penetration over the "
        f"whole run "
        f"{summary['zone_worst'].get('concourse_door_threshold', {}).get('worst_m')} m "
        f"at "
        f"{summary['zone_worst'].get('concourse_door_threshold', {}).get('at_s')}s")
    add(f"at least {MIN_DOOR_EXITERS} people came OUT through the doorway",
        len(summary["exiters_used_door"]) >= MIN_DOOR_EXITERS,
        f"{summary['exiters_used_door']} occupied the door aperture, of "
        f"{summary['door_exiter_names']}")
    add("the duck was NEVER in the doorway at the same time as anybody else",
        summary["aperture_shared_steps"].get(
            "concourse_door", 0) <= MAX_SHARED_APERTURE_STEPS,
        f"{summary['aperture_shared_steps'].get('concourse_door', 0)} shared "
        f"step(s) of {summary['aperture_steps'].get('concourse_door', 0)} the "
        f"duck spent in it; the opening has "
        f"{summary['abreast_slack_m']['concourse_door']} m of slack for two "
        f"abreast, so this is a claim about the robot")
    add("the duck entered the doorway BEHIND the guardian",
        summary["overtake_steps"] <= MAX_OVERTAKE_STEPS
        and (summary["min_guardian_gap_m"] or 0.0) > 0.0,
        f"{summary['overtake_steps']} step(s) ahead of her over "
        f"{summary['guardian_gap_samples']} samples; minimum gap along the "
        f"shared route {summary['min_guardian_gap_m']} m")

    # -- B: the lift ----------------------------------------------------------
    add(f"the duck waited BESIDE the lift doors for at least "
        f"{MIN_WAIT_SIDE_SECONDS:.1f}s",
        summary["state_seconds"].get("WAIT_SIDE", 0.0) >= MIN_WAIT_SIDE_SECONDS,
        f"WAIT_SIDE for {summary['state_seconds'].get('WAIT_SIDE')}s at max "
        f"command {summary['state_command_max'].get('WAIT_SIDE')}")
    add("the duck NEVER stood in the lift's exit passage before boarding",
        summary["zone_violation_steps"].get(
            "lift_front_passage", 0) <= MAX_EARLY_ZONE_STEPS,
        f"{summary['zone_violation_steps'].get('lift_front_passage', 0)} early "
        f"step(s) in the door-centre passage; worst penetration over the whole "
        f"run "
        f"{summary['zone_worst'].get('lift_front_passage', {}).get('worst_m')} m")
    add(f"at least {MIN_OCCUPANTS_OUT} occupants EXITED before the duck entered",
        boarding.get("occupants_exited_before_entry", 0) >= MIN_OCCUPANTS_OUT,
        f"{boarding.get('occupants_exited_before_entry')} of "
        f"{len(summary['occupant_names'])} had cleared at "
        f"{boarding.get('cleared_at_s')}s; "
        f"{summary['max_occupants_exited']} exited in total, "
        f"{summary['occupants_used_lift']} used the aperture")
    add("the duck did not move until the LAST occupant was clear",
        summary["state_command_max"].get("LET_OCCUPANTS_EXIT", 1.0) == 0.0
        and summary["state_seconds"].get("LET_OCCUPANTS_EXIT", 0.0) > 0.0,
        f"LET_OCCUPANTS_EXIT for "
        f"{summary['state_seconds'].get('LET_OCCUPANTS_EXIT')}s at max command "
        f"{summary['state_command_max'].get('LET_OCCUPANTS_EXIT')}, drifting "
        f"{summary['state_path_m'].get('LET_OCCUPANTS_EXIT')} m")
    add("the duck boarded AFTER the guardian",
        bool(boarding.get("guardian_inside_at_entry")),
        f"she was inside the car at the duck's entry "
        f"({boarding.get('duck_entered_at_s')}s), gap "
        f"{boarding.get('guardian_gap_at_entry_m')} m")
    add("the duck was NEVER in the lift aperture at the same time as anybody "
        "else",
        summary["aperture_shared_steps"].get(
            "lift_front", 0) <= MAX_SHARED_APERTURE_STEPS
        and summary["aperture_shared_steps"].get(
            "lift_rear", 0) <= MAX_SHARED_APERTURE_STEPS,
        f"front {summary['aperture_shared_steps'].get('lift_front', 0)} of "
        f"{summary['aperture_steps'].get('lift_front', 0)}; rear "
        f"{summary['aperture_shared_steps'].get('lift_rear', 0)} of "
        f"{summary['aperture_steps'].get('lift_rear', 0)}")
    add("the duck's position inside the cabin is real and inside its bounds",
        summary["min_cabin_margin_m"] is not None
        and summary["min_cabin_margin_m"] >= MIN_CABIN_MARGIN_M
        and summary["cabin_seconds"] >= MIN_CABIN_SECONDS
        and summary["cabin_outside_while_riding_steps"] == 0,
        f"inside for {summary['cabin_seconds']}s with a minimum face margin of "
        f"{summary['min_cabin_margin_m']} m; "
        f"{summary['cabin_outside_while_riding_steps']} step(s) outside it "
        f"while riding")
    add(f"the ride was exactly still for at least {MIN_RIDE_SECONDS:.0f}s",
        summary["ride_seconds"] >= MIN_RIDE_SECONDS
        and summary["state_command_max"].get("RIDE", 1.0) == 0.0,
        f"RIDE for {summary['ride_seconds']}s at max command "
        f"{summary['state_command_max'].get('RIDE')}, drifting "
        f"{summary['state_path_m'].get('RIDE')} m against the MEASURED "
        f"{summary['zero_drift_reference_m']} m per 10 s")
    add("the guardian left the cabin FIRST at the target floor",
        boarding.get("guardian_exited_at_s") is not None
        and boarding.get("duck_exited_at_s") is not None
        and boarding["guardian_exited_at_s"] < boarding["duck_exited_at_s"],
        f"she was out at {boarding.get('guardian_exited_at_s')}s, the duck at "
        f"{boarding.get('duck_exited_at_s')}s")

    # -- doors ---------------------------------------------------------------
    add("NO MOVEMENT THROUGH A CLOSED DOOR: every aperture was open when the "
        "duck entered it",
        all(c.get("entered_at_s") is not None
            and c["open_fraction_at_entry"] >= MIN_OPEN_FRACTION_AT_CROSSING
            and c["effective_gap_at_entry_m"]
            >= MIN_EFFECTIVE_GAP_AT_CROSSING_M
            for c in crossings.values()),
        "; ".join(f"{name}: entered {c.get('entered_at_s')}s at "
                  f"{c.get('open_fraction_at_entry')} open "
                  f"({c.get('effective_gap_at_entry_m')} m clear)"
                  for name, c in sorted(crossings.items())))

    # -- order and following --------------------------------------------------
    add("the duck NEVER overtook or crossed the guardian",
        summary["overtake_steps"] <= MAX_OVERTAKE_STEPS,
        f"{summary['overtake_steps']} overtaking step(s); gap ranged "
        f"{summary['min_guardian_gap_m']} to {summary['max_guardian_gap_m']} m")
    add("the duck stayed close enough to have actually followed her",
        (summary["max_guardian_gap_m"] or 0.0) <= MAX_GUARDIAN_GAP_M,
        f"maximum gap {summary['max_guardian_gap_m']} m against a "
        f"{MAX_GUARDIAN_GAP_M} m limit")

    # -- stillness -------------------------------------------------------------
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
        all(path <= ZERO_STATE_PATH_M
            for path in summary["zero_state_path_m"].values()),
        "; ".join(f"{state} drifted {path} m"
                  for state, path in sorted(
                      summary["zero_state_path_m"].items()))
        + f" against a {ZERO_STATE_PATH_M} m bound "
        f"({summary['zero_drift_reference_m']} m per 10 s MEASURED)")

    # -- visibility ------------------------------------------------------------
    add(f"the active person was visible in >= {MIN_VISIBLE_WITH_LOS:.0%} of "
        "monitoring steps where line of sight existed",
        summary["monitor_visible_fraction_with_los"] >= MIN_VISIBLE_WITH_LOS,
        f"{summary['monitor_visible_fraction_with_los'] * 100:.2f}% of "
        f"{summary['monitor_los_steps']} LOS steps in "
        f"{summary['monitor_steps']} monitor steps "
        f"({summary['visible_fraction_with_los'] * 100:.2f}% over the whole "
        f"run)")
    add("the duck watched the right person, in the right order",
        summary["subject_role_order"] == summary["expected_subject_role_order"],
        f"watched {summary['subject_role_order']}, expected "
        f"{summary['expected_subject_role_order']}")

    # -- the populated building -------------------------------------------------
    add(f"at least {MIN_MOVING_ADULTS} adults besides the guardian were moving",
        summary["moving_adults"] >= MIN_MOVING_ADULTS,
        f"{summary['moving_adults']} of {len(summary['other_adults'])} changed "
        f"position: {summary['other_adults']}")

    # -- safety -----------------------------------------------------------------
    add("positive clearance to every person at all times",
        summary["min_person_clearance_m"] > 0.0,
        f"min {summary['min_person_clearance_m']:.4f} m to "
        f"{summary['min_person_clearance_name']}; per person "
        f"{summary['min_clearance_by_person_m']}")
    add("positive clearance to every wall, jamb, cabin panel and door leaf",
        summary["min_scenery_clearance_m"] > 0.0,
        f"min {summary['min_scenery_clearance_m']:.4f} m to "
        f"{summary['min_scenery_clearance_geom']}")
    add("zero contacts", summary["contact_steps"] == 0,
        f"{summary['contact_steps']} steps with non-positive clearance")

    # -- states -------------------------------------------------------------------
    add("every declared state was visited",
        set(summary["declared_states"]) <= set(summary["states_visited"]),
        f"visited {summary['states_visited']}")
    add("every visited state is one this behavior declares",
        set(summary["states_visited"]) <= set(summary["declared_states"]),
        f"visited {summary['states_visited']}")
    add("the states ran in the declared order",
        summary["state_order"] == list(STATES[1:]),
        f"{summary['state_order']} against the declared "
        f"{list(STATES[1:])}")
    add("exactly zero PUSH_THROUGH and BOARD_FIRST steps",
        all(v == 0 for v in summary["forbidden_state_steps"].values()),
        f"forbidden state steps: {summary['forbidden_state_steps']}")
    add("no phase hit its ceiling", not summary["timeouts"],
        f"timeouts: {summary['timeouts'] or 'none'}")

    # -- locomotion health -----------------------------------------------------
    add("zero falls", summary["fallen_steps"] == 0,
        f"{summary['fallen_steps']} steps below 0.09 m")
    add("trunk never below 0.09 m", summary["min_trunk_z_m"] >= 0.09,
        f"min {summary['min_trunk_z_m']:.4f} m")
    add("final trunk height near the nominal 0.116 m",
        abs(summary["final_trunk_z_m"] - 0.116) <= 0.012,
        f"{summary['final_trunk_z_m']:.4f} m")

    # -- the physical contract --------------------------------------------------
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
