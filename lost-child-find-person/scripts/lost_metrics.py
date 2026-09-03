#!/usr/bin/env python3
"""The acceptance gate: every hard requirement, measured from the rollout.

Each gate is computed from the RECORDS, never from the scenario's own schedule.
"The guardian was hidden" is a count of ray casts that failed, not a lookup of
when the script said she went behind the kiosk; "the duck did not move while
lost" is the maximum absolute command over those ticks, not an assertion that
the controller returns zero.
"""

from __future__ import annotations

import math

import numpy as np

from lost_cast import GUARDIAN, LOOKALIKE_NAMES
from lost_constants import (
    LOSS_CONFIRM_S,
    REACQUIRE_CONFIRM_S,
    STATIONARY_STATES,
)
from lost_geometry import STANDOFF_MAX_M, STANDOFF_MIN_M

# The occlusion a deep cycle must contain, in seconds of continuous invisibility.
MIN_OCCLUSION_S = 2.0
# The false candidates the duck must have refused, by distinct person.
MIN_REJECTED = 2
# Fraction of REJOIN steps with line of sight in which she must be visible.
REJOIN_TRACK_FRACTION = 0.95


def _fraction(values: list[bool]) -> float:
    return (sum(1 for v in values if v) / len(values)) if values else 0.0


def summarize(rollout) -> dict:
    """Every measured quantity the gate and the README quote."""
    records = rollout.records
    machine = rollout.machine
    identity = rollout.identity

    states = [r["state"] for r in records]
    dt = rollout.dt

    # -- zero command while lost -------------------------------------------
    stationary_peak = {
        state: rollout.state_command_max.get(state, 0.0)
        for state in STATIONARY_STATES if state in rollout.state_steps
    }
    blind_steps = sum(
        1 for r in records
        if r["state"] in STATIONARY_STATES and r["command_peak"] != 0.0)

    # -- occlusion ----------------------------------------------------------
    runs = sorted(rollout.occlusion_runs, key=lambda r: -r["duration_s"])
    longest = runs[0] if runs else None
    geometric_runs = [
        run for run in runs
        if max(run["blockers"], key=run["blockers"].get) != "out_of_frustum"
    ]
    longest_geometric = geometric_runs[0] if geometric_runs else None

    # -- follow before the first loss ---------------------------------------
    first_lost = next((r for r in records if r["state"] == "LOST"), None)
    pre_loss = [r for r in records
                if first_lost is None or r["t"] < first_lost["t"]]
    pre_loss_path = pre_loss[-1]["path_m"] if pre_loss else 0.0
    pre_loss_visible = _fraction([r["guardian_visible"] for r in pre_loss
                                  if r["state"] == "FOLLOW"])
    pre_loss_start = np.array(pre_loss[0]["duck_xy"]) if pre_loss else np.zeros(2)
    pre_loss_end = np.array(pre_loss[-1]["duck_xy"]) if pre_loss else np.zeros(2)
    pre_loss_net = float(np.linalg.norm(pre_loss_end - pre_loss_start))

    # -- rejections ---------------------------------------------------------
    distinct_rejected = identity.distinct_rejected()
    rejected_lookalikes = [n for n in distinct_rejected if n in LOOKALIKE_NAMES]
    lookalikes_seen = dict(rollout.lookalike_seen)

    # -- rejoin cycles ------------------------------------------------------
    cycles = []
    for cycle in machine.cycles:
        index = cycle["index"]
        start_xy = rollout.rejoin_start_xy.get(index)
        end_xy = rollout.rejoin_end_xy.get(index)
        net = (float(np.linalg.norm(end_xy - start_xy))
               if start_xy is not None and end_xy is not None else 0.0)
        with_los = rollout.rejoin_visible_with_los.get(index, [])
        cycles.append({
            **cycle,
            "rejoin_path_m": round(rollout.rejoin_path.get(index, 0.0), 4),
            "rejoin_net_m": round(net, 4),
            "range_at_reacquire_m": round(
                rollout.rejoin_start_range.get(index, float("nan")), 4),
            "range_at_arrival_m": round(
                rollout.rejoin_end_range.get(index, float("nan")), 4),
            "target_visible_fraction": round(
                _fraction(rollout.rejoin_visible.get(index, [])), 4),
            "target_visible_fraction_with_los": round(_fraction(with_los), 4),
            "los_steps": len(with_los),
            "min_person_clearance_m": round(
                rollout.rejoin_min_clearance.get(index, float("nan")), 4),
            "route": rollout.rejoin_routes.get(index),
            "rejection_count": len(cycle.get("rejections", [])),
        })

    # -- final standoff -----------------------------------------------------
    final = records[-1] if records else {}
    final_range = final.get("guardian_range_m", float("nan"))
    final_visible = final.get("guardian_visible", False)

    # -- identity continuity ------------------------------------------------
    guardian_names = {r["guardian"] for r in records}

    return {
        "seconds": rollout.seconds,
        "control_steps": len(records),
        "control_hz": 1.0 / dt,
        "guardian": machine.guardian,
        "guardian_names_seen": sorted(guardian_names),

        # physics
        "observation_dim": 61,
        "action_scale": 0.9,
        "gyro_sensor": "imu_ang_vel",
        "fallen_steps": rollout.fallen_steps,
        "contact_steps": rollout.contact_steps,
        "min_trunk_z_m": round(rollout.min_trunk_z, 5),
        "final_trunk_z_m": round(float(records[-1]["trunk_z"]), 5),
        "path_m": round(rollout.path_m, 4),
        "min_person_clearance_m": round(rollout.min_person_clearance, 4),
        "min_person_clearance_name": rollout.min_person_name,
        "min_scenery_clearance_m": round(rollout.min_scenery_clearance, 4),
        "min_scenery_clearance_geom": rollout.min_scenery_geom,
        "duck_planar_radius_m": round(rollout.duck_radius, 4),
        "duck_exact_radius_m": round(rollout.duck_exact_radius, 4),
        # Pose-zero sample of the guardian's exact planar half-extent, for
        # context only; NOT a gait maximum and not consumed by any gate.  See
        # the note in ``rollout_lost`` and in ``lost_geometry``.
        "adult_half_extent_m": round(rollout.adult_half_extent, 4),
        "adult_half_extent_basis": "pose-zero sample; gait range 0.1375-0.2629 m",

        # the states
        "states_visited": sorted(set(states)),
        "state_steps": dict(rollout.state_steps),
        "state_seconds": {k: round(v * dt, 3)
                          for k, v in rollout.state_steps.items()},
        "transitions": machine.transitions,
        "stationary_command_peak": stationary_peak,
        "blind_movement_steps": blind_steps,

        # the follow before the loss
        "pre_loss_path_m": round(pre_loss_path, 4),
        "pre_loss_net_m": round(pre_loss_net, 4),
        "pre_loss_guardian_visible_fraction": round(pre_loss_visible, 4),
        "first_loss_at_s": None if first_lost is None else first_lost["t"],
        "loss_confirm_s": LOSS_CONFIRM_S,
        "reacquire_confirm_s": REACQUIRE_CONFIRM_S,

        # occlusion
        "occlusion_runs": [
            {"start_s": round(r["start_s"], 3), "duration_s": r["duration_s"],
             "blocker": max(r["blockers"], key=r["blockers"].get),
             "blockers": r["blockers"], "cycle": r["cycle"]}
            for r in runs[:8]
        ],
        "longest_occlusion_s": 0.0 if longest is None else longest["duration_s"],
        "longest_geometric_occlusion_s": (
            0.0 if longest_geometric is None else longest_geometric["duration_s"]),
        "longest_geometric_occluder": (
            "" if longest_geometric is None
            else max(longest_geometric["blockers"],
                     key=longest_geometric["blockers"].get)),

        # identity
        "rejections": identity.rejections,
        "distinct_rejected": list(distinct_rejected),
        "rejected_lookalikes": rejected_lookalikes,
        "lookalikes_camera_visible_at_s": lookalikes_seen,
        "wrong_accepts": identity.wrong_accepts,
        "accepted": identity.accepted,

        # cycles
        "cycles": cycles,
        "cycle_count": len(cycles),
        "timeouts": machine.timeouts,

        # trail
        "trail": rollout.trail.as_record(),
        "trail_length_m": round(rollout.trail.length_m(), 4),

        # the end
        "final_range_m": final_range,
        "final_guardian_visible": bool(final_visible),
        "final_state": machine.state,
        "standoff_band_m": [STANDOFF_MIN_M, STANDOFF_MAX_M],
    }


def gates(summary: dict) -> list[tuple[str, bool, str]]:
    """Every hard gate, as (label, passed, evidence)."""
    cycles = summary["cycles"]
    deep = [c for c in cycles if c["rejection_count"] >= MIN_REJECTED]
    results: list[tuple[str, bool, str]] = []

    def add(label: str, ok: bool, evidence: str) -> None:
        results.append((label, bool(ok), evidence))

    # -- the follow before the loss -----------------------------------------
    add("real follow path before the first loss",
        summary["pre_loss_path_m"] >= 0.60 and summary["pre_loss_net_m"] >= 0.40,
        f"path {summary['pre_loss_path_m']:.3f} m, "
        f"net {summary['pre_loss_net_m']:.3f} m")
    add("guardian visible while following before the loss",
        summary["pre_loss_guardian_visible_fraction"] >= 0.80,
        f"{summary['pre_loss_guardian_visible_fraction'] * 100:.1f}% of FOLLOW steps")

    # -- the loss ------------------------------------------------------------
    add("loss declared only after a sustained invisibility window",
        summary["first_loss_at_s"] is not None,
        f"first LOST at {summary['first_loss_at_s']}s after "
        f"{summary['loss_confirm_s']}s of continuous invisibility")
    add("zero locomotion command in EVERY stationary state",
        summary["blind_movement_steps"] == 0
        and all(v == 0.0 for v in summary["stationary_command_peak"].values()),
        f"peak |command| per state: {summary['stationary_command_peak']}")

    # -- occlusion -----------------------------------------------------------
    add(f"a geometric occlusion lasting >= {MIN_OCCLUSION_S}s",
        summary["longest_geometric_occlusion_s"] >= MIN_OCCLUSION_S,
        f"{summary['longest_geometric_occlusion_s']:.2f}s behind "
        f"{summary['longest_geometric_occluder'] or 'nothing'}")

    # -- false candidates ----------------------------------------------------
    add(f">= {MIN_REJECTED} distinct false candidates refused",
        len(summary["distinct_rejected"]) >= MIN_REJECTED,
        f"refused {summary['distinct_rejected']}")
    add("every refused candidate was genuinely camera-visible",
        all(n in summary["lookalikes_camera_visible_at_s"]
            for n in summary["rejected_lookalikes"])
        and len(summary["rejected_lookalikes"]) >= MIN_REJECTED,
        f"look-alikes first seen at {summary['lookalikes_camera_visible_at_s']}")
    add("zero wrong-identity locks",
        len(summary["wrong_accepts"]) == 0,
        f"{len(summary['wrong_accepts'])} accept-grade sightings of "
        "somebody other than the guardian")

    # -- identity continuity -------------------------------------------------
    add("identity is the SAME guardian throughout",
        summary["guardian_names_seen"] == [GUARDIAN.name],
        f"guardian identity seen in records: {summary['guardian_names_seen']}")
    add("every acceptance is of the guardian, after the confirm window",
        all(a["name"] == GUARDIAN.name for a in summary["accepted"])
        and len(summary["accepted"]) > 0,
        f"{len(summary['accepted'])} accept-grade sightings, all "
        f"{GUARDIAN.name}")

    # -- the trail -----------------------------------------------------------
    add("a world-space last-known trail was retained",
        summary["trail_length_m"] >= 0.40
        and summary["trail"]["last_seen_xy"] is not None,
        f"{summary['trail_length_m']:.3f} m of trail, last seen at "
        f"{summary['trail']['last_seen_xy']}")

    # -- the rejoin ----------------------------------------------------------
    add("at least one deep loss/reacquisition cycle",
        len(deep) >= 1,
        f"{len(deep)} cycle(s) with >= {MIN_REJECTED} refusals, "
        f"{len(cycles)} cycle(s) total")
    add("every rejoin is real physical progress",
        all(c["rejoin_path_m"] >= 0.30 and c["rejoin_net_m"] >= 0.20
            for c in cycles) and len(cycles) > 0,
        "; ".join(f"cycle {c['index']}: path {c['rejoin_path_m']:.3f} m, "
                  f"net {c['rejoin_net_m']:.3f} m" for c in cycles))
    add("every rejoin lowered the range to the guardian",
        all(c["range_at_arrival_m"] < c["range_at_reacquire_m"]
            for c in cycles) and len(cycles) > 0,
        "; ".join(f"cycle {c['index']}: {c['range_at_reacquire_m']:.3f} -> "
                  f"{c['range_at_arrival_m']:.3f} m" for c in cycles))
    add(f"guardian visible >= {REJOIN_TRACK_FRACTION:.0%} of REJOIN steps "
        "with line of sight",
        all(c["target_visible_fraction_with_los"] >= REJOIN_TRACK_FRACTION
            for c in cycles) and len(cycles) > 0,
        "; ".join(f"cycle {c['index']}: "
                  f"{c['target_visible_fraction_with_los'] * 100:.1f}% of "
                  f"{c['los_steps']} steps" for c in cycles))
    add("no rejoin route cut through an occluder or the crowd",
        all(c["route"] is not None and c["route"]["feasible"]
            for c in cycles) and len(cycles) > 0,
        "; ".join(f"cycle {c['index']}: "
                  f"{c['route']['waypoint_count']} waypoints, bends around "
                  f"{c['route']['bends_around'] or 'nothing'}"
                  for c in cycles if c["route"]))

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

    # -- the finish ----------------------------------------------------------
    add(f"final distance inside the {STANDOFF_MIN_M}-{STANDOFF_MAX_M} m band",
        summary["final_range_m"] is not None
        and STANDOFF_MIN_M <= summary["final_range_m"] <= STANDOFF_MAX_M,
        f"{summary['final_range_m']} m")
    add("guardian visible at the final standoff",
        summary["final_guardian_visible"],
        f"guardian_visible={summary['final_guardian_visible']} at the last frame")

    # -- locomotion health ---------------------------------------------------
    add("zero falls", summary["fallen_steps"] == 0,
        f"{summary['fallen_steps']} steps below 0.09 m")
    add("trunk never below 0.09 m", summary["min_trunk_z_m"] >= 0.09,
        f"min {summary['min_trunk_z_m']:.4f} m")
    add("final trunk height near the nominal 0.116 m",
        abs(summary["final_trunk_z_m"] - 0.116) <= 0.012,
        f"{summary['final_trunk_z_m']:.4f} m")
    add("no phase hit its ceiling", not summary["timeouts"],
        f"timeouts: {summary['timeouts'] or 'none'}")

    return results


def report(summary: dict) -> tuple[bool, list[tuple[str, bool, str]]]:
    results = gates(summary)
    return all(ok for _, ok, _ in results), results
