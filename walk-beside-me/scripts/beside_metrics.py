#!/usr/bin/env python3
"""The acceptance gate: every hard requirement, measured from the rollout.

Each gate is computed from the RECORDS, never from the scenario's own schedule.
"The duck's side was blocked" is a count of measured verdicts, not a lookup of
when the script put a kiosk there; "the duck never cut in front of her" is the
maximum longitudinal offset over every tick, not an assertion that the planner
puts its waypoints astern.
"""

from __future__ import annotations

import numpy as np

from beside_cast import GUARDIAN
from beside_constants import (
    BESIDE_STATES,
    FORBIDDEN_STATES,
    STATES,
)
from beside_geometry import (
    BESIDE_MAX_M,
    BESIDE_MIN_M,
    BESIDE_LONG_TOLERANCE_M,
    CROSS_BEHIND_M,
    FORWARD_HALF_PLANE_M,
)

# The formation the duck must actually achieve and hold.
MIN_BESIDE_SECONDS = 18.0
MIN_BESIDE_PATH_M = 1.60
# Side decisions the run must contain, and switches it must complete.
MIN_SIDE_DECISIONS = 2
MIN_COMPLETED_SWITCHES = 1
# A completed switch must be a real manoeuvre, not a step sideways.
MIN_SWITCH_PATH_M = 0.90
MIN_SWITCH_NET_M = 0.45
MIN_SWITCH_LATERAL_M = 0.80
# Rear margin the crossing must have kept.
MIN_CROSS_REAR_MARGIN_M = CROSS_BEHIND_M * 0.72
# Visibility, conditioned on line of sight existing at all.
MIN_VISIBLE_WITH_LOS = 0.95
# The bends the duck must have followed.
MIN_BENDS_FOLLOWED = 3


def _fraction(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def summarize(rollout) -> dict:
    """Every measured quantity the gate and the README quote."""
    records = rollout.records
    machine = rollout.machine
    dt = rollout.dt

    states = [r["state"] for r in records]
    lateral = rollout.beside_lateral
    longitudinal = rollout.beside_longitudinal

    # -- the initial join ---------------------------------------------------
    first_beside = next((r for r in records if r["state"] in BESIDE_STATES),
                        None)
    pre_join = [r for r in records
                if first_beside is None or r["t"] < first_beside["t"]]
    join_path = pre_join[-1]["path_m"] if pre_join else 0.0
    join_start = np.array(pre_join[0]["duck_xy"]) if pre_join else np.zeros(2)
    join_end = np.array(pre_join[-1]["duck_xy"]) if pre_join else np.zeros(2)
    join_net = float(np.linalg.norm(join_end - join_start))
    initial_lateral = (abs(first_beside["lateral_m"]) if first_beside else 0.0)

    # -- side decisions and refusals ----------------------------------------
    decisions = machine.decisions
    refusals = [d for d in decisions
                if "blocked by" in d["reason"] or d["kind"] == "blocked"]
    # Every tick in which the duck's OWN side was measured unusable.
    blocked_ticks = [
        r for r in records
        if r["side"] is not None
        and not (r["verdict_left"] if r["side"] == 1
                 else r["verdict_right"])["usable"]
    ]
    # The control case: the far-side pedestrian who must NOT have caused a
    # switch.  Measured as "was any switch attributed to her".
    switch_causes = [(s["cause"], s["detail"]) for s in machine.switches]

    # -- the switches --------------------------------------------------------
    switches = []
    for switch in machine.switches:
        index = switch["index"]
        start_xy = rollout.switch_start_xy.get(index)
        end_xy = rollout.switch_end_xy.get(index)
        net = (float(np.linalg.norm(end_xy - start_xy))
               if start_xy is not None and end_xy is not None else 0.0)
        lateral_start = rollout.switch_lateral_start.get(index, 0.0)
        lateral_end = rollout.switch_lateral_end.get(index, 0.0)
        switches.append({
            **switch,
            "path_m": round(rollout.switch_path.get(index, 0.0), 4),
            "net_m": round(net, 4),
            "lateral_start_m": round(lateral_start, 4),
            "lateral_end_m": round(lateral_end, 4),
            "lateral_travel_m": round(abs(lateral_end - lateral_start), 4),
            "sides_are_opposite": bool(lateral_start * lateral_end < 0.0),
            "min_longitudinal_m": round(
                rollout.switch_min_longitudinal.get(index, float("nan")), 4),
            "max_longitudinal_m": round(
                rollout.switch_max_longitudinal.get(index, float("nan")), 4),
            "min_guardian_clearance_m": round(
                rollout.switch_min_clearance.get(index, float("nan")), 4),
        })

    # -- formation quality after the LAST switch -----------------------------
    last_switch_t = (machine.switches[-1]["joined_at_s"]
                     if machine.switches else None)
    post = [r for r in records
            if last_switch_t is not None and r["t"] >= last_switch_t
            and r["state"] in BESIDE_STATES]
    post_lateral = [abs(r["lateral_m"]) for r in post]
    post_long = [r["longitudinal_m"] for r in post]

    # -- bends ---------------------------------------------------------------
    bends = rollout.guardian_route.corner_report()
    bends_followed = []
    for bend in bends:
        window = [r for r in records
                  if bend["start_t_s"] <= r["t"] <= bend["end_t_s"]
                  and r["state"] in BESIDE_STATES]
        if not window:
            bends_followed.append({**bend, "beside_steps": 0, "followed": False})
            continue
        worst_lateral = max(abs(r["lateral_m"]) for r in window)
        worst_long = max(abs(r["longitudinal_m"]) for r in window)
        bends_followed.append({
            **bend,
            "beside_steps": len(window),
            "beside_seconds": round(len(window) * dt, 3),
            "max_lateral_m": round(worst_lateral, 4),
            "max_abs_longitudinal_m": round(worst_long, 4),
            "followed": bool(worst_lateral <= BESIDE_MAX_M + 0.12
                             and len(window) * dt >= 1.0),
        })

    return {
        "seconds": rollout.seconds,
        "control_steps": len(records),
        "control_hz": 1.0 / dt,
        "guardian": machine.guardian,

        # physics
        "observation_dim": 61,
        "action_scale": 0.9,
        "gyro_sensor": "imu_ang_vel",
        "policy_sha256": rollout.policy_sha256,
        "fallen_steps": rollout.fallen_steps,
        "contact_steps": rollout.contact_steps,
        "min_trunk_z_m": round(rollout.min_trunk_z, 5),
        "final_trunk_z_m": round(float(records[-1]["trunk_z"]), 5),
        "path_m": round(rollout.path_m, 4),
        "min_person_clearance_m": round(rollout.min_person_clearance, 4),
        "min_person_clearance_name": rollout.min_person_name,
        "min_guardian_clearance_m": round(rollout.min_guardian_clearance, 4),
        "min_scenery_clearance_m": round(rollout.min_scenery_clearance, 4),
        "min_scenery_clearance_geom": rollout.min_scenery_geom,
        "duck_planar_radius_m": round(rollout.duck_radius, 4),
        "duck_exact_radius_m": round(rollout.duck_exact_radius, 4),
        "adult_half_extent_m": round(rollout.adult_half_extent, 4),
        "adult_half_extent_basis": "pose-zero sample; not a gait maximum",

        # states
        "states_visited": sorted(set(states)),
        "declared_states": list(STATES),
        "forbidden_states": list(FORBIDDEN_STATES),
        "forbidden_state_steps": {
            state: rollout.state_steps.get(state, 0)
            for state in FORBIDDEN_STATES},
        "state_steps": dict(rollout.state_steps),
        "state_seconds": {k: round(v * dt, 3)
                          for k, v in rollout.state_steps.items()},
        "transitions": machine.transitions,

        # the initial join
        "join_path_m": round(join_path, 4),
        "join_net_m": round(join_net, 4),
        "join_completed": machine.joined,
        "first_beside_at_s": None if first_beside is None else first_beside["t"],
        "first_beside_side": None if first_beside is None
        else first_beside["side_name"],
        "initial_join_lateral_m": round(initial_lateral, 4),

        # the formation
        "beside_steps": rollout.beside_steps,
        "beside_seconds": round(rollout.beside_steps * dt, 3),
        "beside_path_m": round(rollout.beside_path_m, 4),
        "beside_side_seconds": {k: round(v * dt, 3)
                                for k, v in rollout.beside_side_steps.items()},
        "formation_steps": rollout.formation_steps,
        "formation_seconds": round(rollout.formation_steps * dt, 3),
        "formation_fraction_of_beside": round(
            _fraction(rollout.formation_steps, rollout.beside_steps), 4),
        "beside_lateral_min_m": round(min(lateral), 4) if lateral else None,
        "beside_lateral_max_m": round(max(lateral), 4) if lateral else None,
        "beside_lateral_mean_m": round(
            float(np.mean(lateral)), 4) if lateral else None,
        "beside_longitudinal_min_m": round(
            min(longitudinal), 4) if longitudinal else None,
        "beside_longitudinal_max_m": round(
            max(longitudinal), 4) if longitudinal else None,
        "beside_longitudinal_abs_max_m": round(
            max(abs(v) for v in longitudinal), 4) if longitudinal else None,
        "beside_band_m": [BESIDE_MIN_M, BESIDE_MAX_M],
        "longitudinal_tolerance_m": BESIDE_LONG_TOLERANCE_M,

        # the forward half-plane
        "max_forward_longitudinal_m": round(
            rollout.max_forward_longitudinal, 4),
        "max_forward_during_switch_m": (
            None if rollout.max_forward_during_switch == -float("inf")
            else round(rollout.max_forward_during_switch, 4)),
        "forward_half_plane_limit_m": FORWARD_HALF_PLANE_M,

        # side decisions
        "side_decisions": decisions,
        "side_decision_count": len(decisions),
        "refusal_count": len(refusals),
        "blocked_tick_count": len(blocked_ticks),
        "switch_causes": switch_causes,

        # switches
        "switches": switches,
        "completed_switches": len(switches),
        "cross_behind_threshold_m": CROSS_BEHIND_M,

        # the formation after the last switch
        "post_switch_steps": len(post),
        "post_switch_seconds": round(len(post) * dt, 3),
        "post_switch_lateral_min_m": round(
            min(post_lateral), 4) if post_lateral else None,
        "post_switch_lateral_max_m": round(
            max(post_lateral), 4) if post_lateral else None,
        "post_switch_longitudinal_abs_max_m": round(
            max(abs(v) for v in post_long), 4) if post_long else None,
        "post_switch_side": (records[-1]["side_name"] if records else None),

        # bends
        "route": rollout.guardian_route.as_record(),
        "bends": bends_followed,
        "bends_followed": sum(1 for b in bends_followed if b["followed"]),

        # visibility
        "visible_steps": rollout.visible_steps,
        "visible_fraction": round(
            _fraction(rollout.visible_steps, len(records)), 4),
        "los_steps": rollout.los_steps,
        "visible_with_los_steps": rollout.visible_with_los,
        "visible_fraction_with_los": round(
            _fraction(rollout.visible_with_los, rollout.los_steps), 4),
        "blocked_by": dict(rollout.blocked_by),

        # timeouts
        "timeouts": machine.timeouts,
    }


def gates(summary: dict) -> list[tuple[str, bool, str]]:
    """Every hard gate, as (label, passed, evidence)."""
    results: list[tuple[str, bool, str]] = []

    def add(label: str, ok: bool, evidence: str) -> None:
        results.append((label, bool(ok), evidence))

    switches = summary["switches"]

    # -- the initial join ----------------------------------------------------
    add("the duck walked into formation rather than spawning in it",
        summary["join_path_m"] >= 0.80 and summary["join_net_m"] >= 0.50
        and summary["join_completed"],
        f"path {summary['join_path_m']:.3f} m, net {summary['join_net_m']:.3f} m "
        f"before first BESIDE at {summary['first_beside_at_s']}s")
    add("the initial join is a real SIDE slot, not a trailing position",
        summary["initial_join_lateral_m"] >= summary["beside_band_m"][0]
        and summary["first_beside_side"] in ("left", "right"),
        f"lateral {summary['initial_join_lateral_m']:.3f} m on the "
        f"{summary['first_beside_side']} at first BESIDE")

    # -- meaningful beside time and distance ---------------------------------
    add(f"at least {MIN_BESIDE_SECONDS:.0f}s spent beside her",
        summary["beside_seconds"] >= MIN_BESIDE_SECONDS,
        f"{summary['beside_seconds']:.2f}s in BESIDE_LEFT/BESIDE_RIGHT "
        f"({summary['beside_side_seconds']})")
    add(f"at least {MIN_BESIDE_PATH_M:.1f} m walked while beside her",
        summary["beside_path_m"] >= MIN_BESIDE_PATH_M,
        f"{summary['beside_path_m']:.3f} m of path in formation")
    add("the lateral offset stayed inside the 0.45-0.75 m band while beside her",
        summary["beside_lateral_min_m"] is not None
        and summary["beside_lateral_min_m"] >= BESIDE_MIN_M - 0.02
        and summary["beside_lateral_max_m"] <= BESIDE_MAX_M + 0.12,
        f"lateral {summary['beside_lateral_min_m']}-"
        f"{summary['beside_lateral_max_m']} m "
        f"(mean {summary['beside_lateral_mean_m']})")
    add("the longitudinal error stayed bounded while beside her",
        summary["beside_longitudinal_abs_max_m"] is not None
        and summary["beside_longitudinal_abs_max_m"]
        <= summary["longitudinal_tolerance_m"] + 0.12,
        f"|longitudinal| <= {summary['beside_longitudinal_abs_max_m']} m "
        f"against a {summary['longitudinal_tolerance_m']} m tolerance")

    # -- the side decisions --------------------------------------------------
    add(f"at least {MIN_SIDE_DECISIONS} side decisions were made",
        summary["side_decision_count"] >= MIN_SIDE_DECISIONS,
        f"{summary['side_decision_count']} decision(s): "
        + "; ".join(f"{d['kind']}->{d['side_name']} ({d['reason']})"
                    for d in summary["side_decisions"]))
    add(f"at least {MIN_COMPLETED_SWITCHES} completed physical side switch",
        summary["completed_switches"] >= MIN_COMPLETED_SWITCHES,
        f"{summary['completed_switches']} completed switch(es): "
        + "; ".join(f"{s['from_side']}->{s['to_side']} at {s['blocked_at_s']}s"
                    for s in switches))
    add("every switch was caused by a MEASURED blockage of the duck's own side",
        len(switches) > 0
        and all(s["cause"] in ("static", "person") and s["detail"]
                and s["blocked_for_s"] > 0.0 for s in switches),
        "; ".join(f"switch {s['index']}: {s['cause']}:{s['detail']} measured "
                  f"for {s['blocked_for_s']}s "
                  f"(static gap {s['static_gap_m']} m, "
                  f"person gap {s['person_gap_m']} m)" for s in switches))
    add("an unsafe side was refused while the other stayed available",
        summary["refusal_count"] >= 1 and summary["blocked_tick_count"] > 0,
        f"{summary['refusal_count']} refusal decision(s), "
        f"{summary['blocked_tick_count']} tick(s) with the duck's own side "
        "measured unusable")

    # -- the crossover -------------------------------------------------------
    add("every switch crossed BEHIND her, never through her forward half-plane",
        len(switches) > 0
        and all(s["max_longitudinal_m"] <= FORWARD_HALF_PLANE_M
                for s in switches),
        "; ".join(f"switch {s['index']}: max longitudinal "
                  f"{s['max_longitudinal_m']} m <= "
                  f"{FORWARD_HALF_PLANE_M} m" for s in switches))
    add(f"every switch reached at least {MIN_CROSS_REAR_MARGIN_M:.2f} m astern "
        "before crossing",
        len(switches) > 0
        and all(s["min_longitudinal_m"] <= -MIN_CROSS_REAR_MARGIN_M
                for s in switches),
        "; ".join(f"switch {s['index']}: {s['min_longitudinal_m']} m astern"
                  for s in switches))
    add("the duck never got ahead of her at ANY tick of the whole rollout",
        summary["max_forward_longitudinal_m"] <= FORWARD_HALF_PLANE_M,
        f"max longitudinal {summary['max_forward_longitudinal_m']} m over "
        f"{summary['control_steps']} steps, limit "
        f"{FORWARD_HALF_PLANE_M} m")
    add("every switch is a real path with real net and lateral displacement",
        len(switches) > 0
        and all(s["path_m"] >= MIN_SWITCH_PATH_M
                and s["net_m"] >= MIN_SWITCH_NET_M
                and s["lateral_travel_m"] >= MIN_SWITCH_LATERAL_M
                and s["sides_are_opposite"] for s in switches),
        "; ".join(f"switch {s['index']}: path {s['path_m']} m, net "
                  f"{s['net_m']} m, lateral {s['lateral_start_m']} -> "
                  f"{s['lateral_end_m']} m" for s in switches))
    add("the duck kept clear of her legs throughout every switch",
        len(switches) > 0
        and all(s["min_guardian_clearance_m"] > 0.0 for s in switches),
        "; ".join(f"switch {s['index']}: min clearance "
                  f"{s['min_guardian_clearance_m']} m" for s in switches))

    # -- the formation after the switch --------------------------------------
    add("the opposite-side formation is stable after the switch",
        summary["post_switch_seconds"] >= 6.0
        and summary["post_switch_lateral_min_m"] is not None
        and summary["post_switch_lateral_min_m"] >= BESIDE_MIN_M - 0.02
        and summary["post_switch_lateral_max_m"] <= BESIDE_MAX_M + 0.12,
        f"{summary['post_switch_seconds']:.2f}s on the "
        f"{summary['post_switch_side']} at lateral "
        f"{summary['post_switch_lateral_min_m']}-"
        f"{summary['post_switch_lateral_max_m']} m")

    # -- the bends -----------------------------------------------------------
    add(f"the formation followed at least {MIN_BENDS_FOLLOWED} route bends",
        summary["bends_followed"] >= MIN_BENDS_FOLLOWED,
        "; ".join(f"{b['hand']} {b['turn_deg']:.1f} deg: "
                  f"{b.get('beside_seconds', 0)}s beside, max lateral "
                  f"{b.get('max_lateral_m', float('nan'))} m -> "
                  f"{'followed' if b['followed'] else 'NOT followed'}"
                  for b in summary["bends"]))
    add("the route contains both a left and a right bend",
        {b["hand"] for b in summary["route"]["bends"]} == {"left", "right"},
        f"bends: {[b['hand'] for b in summary['route']['bends']]}")

    # -- visibility ----------------------------------------------------------
    add(f"guardian visible in >= {MIN_VISIBLE_WITH_LOS:.0%} of steps with line "
        "of sight",
        summary["visible_fraction_with_los"] >= MIN_VISIBLE_WITH_LOS,
        f"{summary['visible_fraction_with_los'] * 100:.2f}% of "
        f"{summary['los_steps']} LOS steps "
        f"({summary['visible_fraction'] * 100:.2f}% overall)")

    # -- safety --------------------------------------------------------------
    add("positive clearance to every person at all times",
        summary["min_person_clearance_m"] > 0.0,
        f"min {summary['min_person_clearance_m']:.4f} m to "
        f"{summary['min_person_clearance_name']}")
    add("positive clearance to every obstacle and wall",
        summary["min_scenery_clearance_m"] > 0.0,
        f"min {summary['min_scenery_clearance_m']:.4f} m to "
        f"{summary['min_scenery_clearance_geom']}")
    add("zero contacts", summary["contact_steps"] == 0,
        f"{summary['contact_steps']} steps with non-positive clearance")

    # -- states --------------------------------------------------------------
    add("exactly zero HOLD and DONE steps",
        all(v == 0 for v in summary["forbidden_state_steps"].values()),
        f"forbidden state steps: {summary['forbidden_state_steps']}")
    add("every visited state is one this behavior declares",
        set(summary["states_visited"]) <= set(summary["declared_states"]),
        f"visited {summary['states_visited']}")
    add("no phase hit its ceiling", not summary["timeouts"],
        f"timeouts: {summary['timeouts'] or 'none'}")

    # -- locomotion health ---------------------------------------------------
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


# The stock walking policy this whole behavior is measured against.
UPSTREAM_POLICY_SHA = (
    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c")


def report(summary: dict) -> tuple[bool, list[tuple[str, bool, str]]]:
    results = gates(summary)
    return all(ok for _, ok, _ in results), results
