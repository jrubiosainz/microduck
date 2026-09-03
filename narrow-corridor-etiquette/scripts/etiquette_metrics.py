#!/usr/bin/env python3
"""Acceptance gates for narrow-corridor-etiquette.

Every gate is evaluated from the rollout records and the machine's own logs;
nothing here re-simulates or re-derives physics.  A gate never softens: if the
behavior cannot meet one, the behavior is changed, not the threshold.

Gate summary
------------
``state_order``            the nine states occur in the required order, and the
                          rollout ends in DONE
``real_forward_start``     the duck makes genuine forward progress toward the
                          destination before the first encounter
``detects_before_unsafe``  every encounter is detected while the adult is still
                          beyond the unsafe-proximity range
``counterfactual_recorded`` each cycle records the clearance the pass WOULD have
                          had, measured at detection, and it is unsafe
``evaluated_enough``       at least two alcoves scored per decision
``rejected_on_clearance``  at least one alcove refused for physical clearance
                          rather than for being out of reach or behind
``selection_is_viable``    every selected alcove satisfies BOTH reachability and
                          physical clearance
``pull_over_moved``        each pull-over produced real path length and real
                          lateral displacement
``footprint_cleared``      the duck's whole footprint left the centre passage
``yield_command_zero``     command EXACTLY zero for every YIELD step
``adult_passed``           the adult crossed the duck's station during the yield
``no_early_rejoin``        rejoin began only after the adult was receding and
                          beyond the justified clearance range
``rejoin_centred``         each rejoin ended near the corridor centreline
``resumed_forward``        real forward progress after the last rejoin
``reached_destination``    the duck's trunk reached the painted threshold
``person_clearance``       minimum geometric duck/adult clearance is positive
``wall_clearance``         minimum geometric duck/wall clearance is positive
``no_contacts``            zero steps with any non-positive clearance
``tracking``               the adult was in the PiP for >= 95% of every YIELD
                          step in which the duck could physically see out of
                          its recess; beyond the mouth's sightline the person
                          is behind an opaque wall
``no_decorative_commands``  every nonzero command clears its measured gait onset
``no_falls``               fallen_steps == 0
``min_trunk_z``            >= 0.09 m throughout
``final_trunk_z``          within 0.012 m of the nominal 0.116 m
``no_timeouts``            no phase hit its ceiling
"""

from __future__ import annotations

import numpy as np

from corridor import (
    ALCOVE_BY_NAME,
    CENTER_PASSAGE_HALF,
    CLEAR_ABS_Y,
    DESTINATION_X,
    REJOIN_TOLERANCE_M,
    corridor_passing_geometry,
)
from encounter import (
    CLEAR_RANGE_M,
    STATES,
    STATIONARY_STATES,
    UNSAFE_PROXIMITY_M,
    VX_MIN_EFFECTIVE,
    VY_MIN_EFFECTIVE_LEFT,
    VY_MIN_EFFECTIVE_RIGHT,
)

FALLEN_Z = 0.09
NOMINAL_Z = 0.116
FINAL_Z_TOLERANCE = 0.012
# Evidence thresholds for a pull-over being a real physical manoeuvre rather
# than a state label.  The shallowest usable park point is 0.3343 m off the
# centreline, so a genuine entry cannot be much under 0.25 m of lateral travel.
MIN_PULL_OVER_LATERAL_M = 0.25
MIN_PULL_OVER_PATH_M = 0.30
MIN_REJOIN_LATERAL_M = 0.25
# Forward progress required before the first encounter, so "starts with a real
# forward path" is measured rather than assumed.
MIN_OPENING_PROGRESS_M = 0.30
# Forward progress required after the final rejoin.
MIN_RESUME_PROGRESS_M = 0.30
# Fraction of YIELD steps in which the passing adult must be inside the PiP.
MIN_TRACKING_FRACTION = 0.95
# The whole-behavior minimum for both geometric clearances.
MIN_CLEARANCE_M = 0.0


def _state_runs(records: list[dict]) -> list[tuple[str, int, int]]:
    """Contiguous runs of the same state: ``(state, first_index, last_index)``."""
    runs: list[tuple[str, int, int]] = []
    for index, record in enumerate(records):
        if runs and runs[-1][0] == record["state"]:
            runs[-1] = (runs[-1][0], runs[-1][1], index)
        else:
            runs.append((record["state"], index, index))
    return runs


def _expected_sequence(cycles: int) -> list[str]:
    """The state order a rollout with ``cycles`` complete etiquette cycles must show."""
    sequence = ["CRUISE"]
    for index in range(cycles):
        sequence += ["DETECT", "SELECT_ALCOVE", "PULL_OVER", "YIELD",
                     "CLEAR", "REJOIN", "RESUME"]
    sequence.append("DONE")
    return sequence


def _command_below_onset(command) -> bool:
    """Is this a decorative command — nonzero but below its measured gait onset?

    A command between zero and onset shows motion in the HUD and produces none
    on the floor, which is exactly the kind of dishonesty the gates exist to
    catch.  Yaw is excluded: it is a trim on top of a walking gait rather than
    a gait of its own, and it is only ever emitted alongside one.
    """
    vx, vy, _wz = (float(v) for v in command)
    if vx != 0.0 and abs(vx) < VX_MIN_EFFECTIVE:
        return True
    if vy > 0.0 and vy < VY_MIN_EFFECTIVE_LEFT:
        return True
    if vy < 0.0 and abs(vy) < VY_MIN_EFFECTIVE_RIGHT:
        return True
    return False


def summarize(rollout) -> dict:
    records = rollout.records
    machine = rollout.machine
    runs = _state_runs(records)
    run_states = [state for state, _, _ in runs]
    cycles = list(machine.cycles)

    # -- opening: real forward path before anything happens ---------------
    first_detect = next(
        (i for i, r in enumerate(records) if r["state"] == "DETECT"), None)
    opening = records[:first_detect] if first_detect else records
    opening_progress = (
        opening[-1]["duck_xy"][0] - records[0]["duck_xy"][0]
        if opening else 0.0)

    # -- per-cycle evidence ------------------------------------------------
    cycle_evidence = []
    for entry in cycles:
        index = entry["index"]
        pull_records = [r for r in records
                        if r["state"] == "PULL_OVER" and r["cycle"] == index]
        yield_records = [r for r in records
                         if r["state"] == "YIELD" and r["cycle"] == index]
        rejoin_records = [r for r in records
                          if r["state"] == "REJOIN" and r["cycle"] == index]
        detect_records = [r for r in records
                          if r["state"] == "DETECT" and r["cycle"] == index]

        lateral = (
            abs(pull_records[-1]["duck_xy"][1] - pull_records[0]["duck_xy"][1])
            if pull_records else 0.0)
        rejoin_lateral = (
            abs(rejoin_records[-1]["duck_xy"][1]
                - rejoin_records[0]["duck_xy"][1])
            if rejoin_records else 0.0)
        tracking = rollout.yield_tracking.get(index, [])
        tracking_fraction = (
            sum(tracking) / len(tracking) if tracking else 0.0)
        in_view = rollout.yield_tracking_in_view.get(index, [])
        in_view_fraction = sum(in_view) / len(in_view) if in_view else 0.0
        alcove = ALCOVE_BY_NAME.get(entry.get("selected_alcove") or "")
        sightline = alcove.sightline_half_span_m if alcove else None
        sides = rollout.yield_person_side.get(index, [])
        # The adult genuinely crossed the duck's station during the yield: the
        # signed offset changed sign, which cannot happen unless it passed.
        passed = bool(sides) and (min(sides) < 0.0 < max(sides))

        decision = entry.get("decision", {})
        candidates = decision.get("candidates", [])
        # A rejection that counts: refused for PHYSICAL CLEARANCE while it was
        # still reachable and ahead of the duck.  A bay that is merely out of
        # reach proves nothing about the duck's judgement of the geometry.
        clearance_rejections = [
            c for c in candidates
            if not c["clears_passage"] and c["reachable"] and not c["behind"]
        ]
        selected = decision.get("selected")

        cycle_evidence.append({
            "index": index,
            "person": entry.get("person"),
            "head_on": entry.get("head_on"),
            "detected_at_s": entry.get("detected_at_s"),
            "detect_range_m": entry.get("detect_range_m"),
            "detect_time_to_meet_s": entry.get("detect_time_to_meet_s"),
            "detect_min_clearance_m": rollout.detect_proximity.get(index),
            "counterfactual_clearance_m":
                entry.get("counterfactual_clearance_m"),
            "alcoves_considered": entry.get("alcoves_considered"),
            "alcoves_rejected": entry.get("alcoves_rejected"),
            "clearance_rejections": [c["alcove"] for c in clearance_rejections],
            "selected_alcove": entry.get("selected_alcove"),
            "selected_park_y": entry.get("selected_park_y"),
            "selected_margin_s": entry.get("selected_margin_s"),
            "selected_clears_passage": (
                selected["clears_passage"] if selected else False),
            "selected_reachable": selected["reachable"] if selected else False,
            "pull_over_duration_s": entry.get("pull_over_duration_s"),
            "pull_over_path_m": rollout.pull_over_path.get(index, 0.0),
            "pull_over_lateral_m": lateral,
            "park_xy": entry.get("park_xy"),
            "park_clears_passage": (
                bool(yield_records[0]["clears_passage"])
                if yield_records else False),
            "min_passage_intrusion_m": (
                min(r["passage_intrusion_m"] for r in yield_records)
                if yield_records else None),
            "yield_duration_s": entry.get("yield_duration_s"),
            "yield_steps": len(yield_records),
            "yield_command_max": rollout.yield_command_max.get(index, 0.0),
            "adult_passed_during_yield": passed,
            "adult_offset_range_m": (
                [min(sides), max(sides)] if sides else None),
            "clear_range_m": entry.get("clear_range_m"),
            "tracking_fraction": tracking_fraction,
            "tracking_fraction_in_sightline": in_view_fraction,
            "sightline_half_span_m": sightline,
            "tracking_steps_in_sightline": len(in_view),
            "rejoin_duration_s": entry.get("rejoin_duration_s"),
            "rejoin_path_m": rollout.rejoin_path.get(index, 0.0),
            "rejoin_lateral_m": rejoin_lateral,
            "rejoin_xy": entry.get("rejoin_xy"),
            "detect_steps": len(detect_records),
        })

    # -- resume: real forward progress after the last rejoin ---------------
    last_rejoin = None
    for index in range(len(records) - 1, -1, -1):
        if records[index]["state"] == "REJOIN":
            last_rejoin = index
            break
    resume_progress = (
        records[-1]["duck_xy"][0] - records[last_rejoin]["duck_xy"][0]
        if last_rejoin is not None else 0.0)

    # -- whole-rollout measurements ----------------------------------------
    stationary = [r for r in records if r["state"] in STATIONARY_STATES]
    stationary_max = max(
        (float(np.max(np.abs(r["command"]))) for r in stationary), default=0.0)
    yield_records = [r for r in records if r["state"] == "YIELD"]
    yield_command_max = max(
        (float(np.max(np.abs(r["command"]))) for r in yield_records),
        default=0.0)
    decorative = [
        {"t": r["t"], "state": r["state"], "command": r["command"]}
        for r in records if _command_below_onset(r["command"])
    ]

    min_person = min(
        (r["nearest_clearance_m"] for r in records), default=float("inf"))
    min_wall = min(
        (r["wall_clearance_m"] for r in records), default=float("inf"))
    contact_steps = sum(
        r["nearest_clearance_m"] <= MIN_CLEARANCE_M
        or r["wall_clearance_m"] <= MIN_CLEARANCE_M for r in records)
    final_z = records[-1]["trunk_z_m"] if records else 0.0
    min_z = min((r["trunk_z_m"] for r in records), default=0.0)
    final_x = records[-1]["duck_xy"][0] if records else float("nan")

    tracking_fractions = [
        c["tracking_fraction_in_sightline"] for c in cycle_evidence]
    raw_tracking_fractions = [c["tracking_fraction"] for c in cycle_evidence]
    detect_clearances = [
        c["detect_min_clearance_m"] for c in cycle_evidence
        if c["detect_min_clearance_m"] is not None]

    gates = {
        "state_order": run_states == _expected_sequence(len(cycles)),
        "real_forward_start": opening_progress >= MIN_OPENING_PROGRESS_M,
        "detects_before_unsafe": bool(detect_clearances) and all(
            value > UNSAFE_PROXIMITY_M for value in detect_clearances),
        "counterfactual_recorded": bool(cycle_evidence) and all(
            c["counterfactual_clearance_m"] is not None
            and c["counterfactual_clearance_m"] < 0.0 for c in cycle_evidence),
        "evaluated_enough": bool(cycle_evidence) and all(
            (c["alcoves_considered"] or 0) >= 2 for c in cycle_evidence),
        "rejected_on_clearance": any(
            c["clearance_rejections"] for c in cycle_evidence),
        "selection_is_viable": bool(cycle_evidence) and all(
            c["selected_clears_passage"] and c["selected_reachable"]
            for c in cycle_evidence),
        "pull_over_moved": bool(cycle_evidence) and all(
            c["pull_over_lateral_m"] >= MIN_PULL_OVER_LATERAL_M
            and c["pull_over_path_m"] >= MIN_PULL_OVER_PATH_M
            for c in cycle_evidence),
        "footprint_cleared": bool(cycle_evidence) and all(
            c["park_clears_passage"]
            and (c["min_passage_intrusion_m"] or 1.0) <= 0.0
            for c in cycle_evidence),
        "yield_command_zero": yield_command_max == 0.0,
        "adult_passed": bool(cycle_evidence) and all(
            c["adult_passed_during_yield"] for c in cycle_evidence),
        "no_early_rejoin": bool(cycle_evidence) and all(
            (c["clear_range_m"] or 0.0) >= CLEAR_RANGE_M
            for c in cycle_evidence),
        "rejoin_centred": bool(cycle_evidence) and all(
            c["rejoin_xy"] is not None
            and abs(c["rejoin_xy"][1]) <= REJOIN_TOLERANCE_M
            and c["rejoin_lateral_m"] >= MIN_REJOIN_LATERAL_M
            for c in cycle_evidence),
        "resumed_forward": resume_progress >= MIN_RESUME_PROGRESS_M,
        "reached_destination": bool(records) and records[-1]["at_destination"],
        "person_clearance": min_person > MIN_CLEARANCE_M,
        "wall_clearance": min_wall > MIN_CLEARANCE_M,
        "no_contacts": contact_steps == 0,
        "tracking": bool(tracking_fractions) and all(
            value >= MIN_TRACKING_FRACTION for value in tracking_fractions),
        "no_decorative_commands": not decorative,
        "still_when_still": stationary_max == 0.0,
        "no_falls": sum(r["trunk_z_m"] < FALLEN_Z for r in records) == 0,
        "min_trunk_z": min_z >= FALLEN_Z,
        "final_trunk_z": abs(final_z - NOMINAL_Z) <= FINAL_Z_TOLERANCE,
        "no_timeouts": not machine.timeouts,
    }

    return {
        "duration_s": rollout.seconds,
        "control_steps": len(records),
        "ctrl_hz": 1.0 / rollout.dt,
        "decimation": rollout.decimation,
        "duck_planar_radius_m": rollout.duck_radius,
        "duck_exact_planar_radius_m": rollout.duck_exact_radius,
        "duck_lateral_half_m": rollout.duck_lateral_half,
        "adult_lateral_half_m": rollout.adult_lateral_half,
        "adult_exact_planar_radius_m": rollout.adult_exact_radius,
        "passing_geometry": corridor_passing_geometry(),
        "center_passage_half_m": CENTER_PASSAGE_HALF,
        "clear_abs_y_m": CLEAR_ABS_Y,
        "destination_x_m": DESTINATION_X,
        "state_sequence": run_states,
        "expected_state_sequence": _expected_sequence(len(cycles)),
        "state_step_counts": {
            state: sum(r["state"] == state for r in records)
            for state in STATES
        },
        "transitions": rollout.transitions,
        "cycles": cycle_evidence,
        "cycle_count": len(cycles),
        "decisions": machine.decisions,
        "opening_progress_m": opening_progress,
        "resume_progress_m": resume_progress,
        "total_path_m": rollout.path_m,
        "forward_progress_m": rollout.forward_progress_m,
        "final_x_m": final_x,
        "max_x_m": rollout.max_x,
        "min_person_clearance_m": min_person,
        "min_wall_clearance_m": min_wall,
        "min_wall_geom": rollout.min_wall_geom,
        "wall_geom_count": len(rollout.wall_geoms),
        "contact_steps": contact_steps,
        "yield_command_max": yield_command_max,
        "stationary_command_max": stationary_max,
        "stationary_steps": len(stationary),
        "decorative_commands": decorative[:20],
        "decorative_command_count": len(decorative),
        "unsafe_proximity_m": UNSAFE_PROXIMITY_M,
        "clear_range_required_m": CLEAR_RANGE_M,
        "raw_tracking_fractions": raw_tracking_fractions,
        "sightline_tracking_fractions": tracking_fractions,
        "min_trunk_z_m": min_z,
        "final_trunk_z_m": final_z,
        "nominal_trunk_z_m": NOMINAL_Z,
        "fallen_steps": sum(r["trunk_z_m"] < FALLEN_Z for r in records),
        "timeouts": list(machine.timeouts),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def format_report(summary: dict) -> str:
    geometry = summary["passing_geometry"]
    lines = [
        f"duration={summary['duration_s']:.1f}s  "
        f"steps={summary['control_steps']}  "
        f"cycles={summary['cycle_count']}",
        f"states={'→'.join(summary['state_sequence'])}",
        f"corridor {geometry['corridor_width_m']:.3f} m wide  "
        f"duck half {geometry['duck_lateral_half_m']:.4f}  "
        f"adult half {geometry['adult_lateral_half_m']:.4f}  "
        f"best possible surface gap "
        f"{geometry['best_possible_surface_gap_m']:+.4f} m  "
        f"needs {geometry['safe_gap_m']:.3f} → "
        f"fits_safely={geometry['fits_safely']} "
        f"(short by {geometry['shortfall_m']:+.4f} m)",
        f"min_z={summary['min_trunk_z_m']:.4f}  "
        f"final_z={summary['final_trunk_z_m']:.4f}  "
        f"falls={summary['fallen_steps']}  "
        f"min person clearance={summary['min_person_clearance_m']:.4f}  "
        f"min wall clearance={summary['min_wall_clearance_m']:.4f} "
        f"({summary['min_wall_geom']}, {summary['wall_geom_count']} geoms)",
        f"opening progress={summary['opening_progress_m']:.3f} m  "
        f"resume progress={summary['resume_progress_m']:.3f} m  "
        f"final x={summary['final_x_m']:+.3f} m "
        f"(destination {summary['destination_x_m']:+.3f})",
        f"yield command max={summary['yield_command_max']:.6f}  "
        f"stationary command max={summary['stationary_command_max']:.6f}  "
        f"decorative commands={summary['decorative_command_count']}",
        "",
        "etiquette cycles:",
    ]
    for entry in summary["cycles"]:
        lines.append(
            f"  #{entry['index']} {entry['person']:<6s} "
            f"{'head-on' if entry['head_on'] else 'overtaking':<10s} "
            f"detect t={entry['detected_at_s']:6.2f}s "
            f"range={entry['detect_range_m']:5.2f} m "
            f"ttm={entry['detect_time_to_meet_s']:5.2f}s "
            f"nearest={entry['detect_min_clearance_m']:.3f} m")
        lines.append(
            f"      counterfactual clearance "
            f"{entry['counterfactual_clearance_m']:+.4f} m  "
            f"considered {entry['alcoves_considered']}  "
            f"rejected {entry['alcoves_rejected']}  "
            f"on clearance {entry['clearance_rejections']}")
        lines.append(
            f"      → {entry['selected_alcove']} "
            f"park_y={entry['selected_park_y']:+.4f} "
            f"margin={entry['selected_margin_s']:+.2f}s  "
            f"pull-over {entry['pull_over_duration_s']:.2f}s "
            f"lateral {entry['pull_over_lateral_m']:.3f} m "
            f"path {entry['pull_over_path_m']:.3f} m")
        lines.append(
            f"      yield {entry['yield_duration_s']:.2f}s "
            f"({entry['yield_steps']} steps) cmd_max="
            f"{entry['yield_command_max']:.6f}  "
            f"adult passed={entry['adult_passed_during_yield']} "
            f"offsets={entry['adult_offset_range_m']}  "
            f"tracked {entry['tracking_fraction_in_sightline'] * 100:5.1f}% "
            f"in sightline (±{entry['sightline_half_span_m']:.3f} m, "
            f"{entry['tracking_steps_in_sightline']} steps; "
            f"{entry['tracking_fraction'] * 100:.1f}% over the whole yield)  "
            f"released at range {entry['clear_range_m']:.3f} m")
        lines.append(
            f"      rejoin {entry['rejoin_duration_s']:.2f}s "
            f"lateral {entry['rejoin_lateral_m']:.3f} m → "
            f"y={entry['rejoin_xy'][1]:+.4f} m  "
            f"min passage intrusion "
            f"{entry['min_passage_intrusion_m']:+.4f} m")
    if summary["timeouts"]:
        lines.append(f"\nTIMEOUTS: {', '.join(summary['timeouts'])}")
    lines.append("")
    for name, passed in summary["gates"].items():
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    lines.append(f"\nALL GATES: {'PASS' if summary['all_gates_pass'] else 'FAIL'}")
    return "\n".join(lines)
