#!/usr/bin/env python3
"""Every acceptance gate, declared as data.

Gates live here as a LIST rather than as scattered asserts, so the same
declaration can be printed, written into the metrics, counted, and required by
``tests/test_gate_counterexamples.py`` to have a counterexample each.  A gate
that exists only inside an ``if`` is a gate nobody can enumerate.

WHAT MAKES A GATE ADMISSIBLE HERE
-----------------------------------
Each must be graded on a quantity that was MEASURED from physics or from the
real camera, never on a state name, a command register, or a label.  This module
reads only the summary dictionary, so it cannot reach into a rollout and grade
something the summary did not record.
"""

from __future__ import annotations

from gest_cast import INSTRUCTOR
from gest_control import GAIT_ONSET_EPS
from gest_states import (
    BACK_UP_TARGET_M,
    CONFIRM_S,
    FORBIDDEN_STATES,
    STANDOFF_MIN_M,
    STOP_HOLD_S,
    TURN_TARGET_DEG,
    VX_REVERSE_ONSET,
)
from policy_runtime import CTRL_HZ, FALLEN_TRUNK_Z, NOMINAL_TRUNK_Z


def gates(summary: dict) -> list[tuple[str, bool, str]]:
    """Every acceptance gate as ``(name, passed, detail)``.

    Declared as data so the count is knowable, the metrics can carry it, and a
    meta-test can require each one to have a counterexample.
    """
    tally = summary["tally"]
    sequence = summary["sequence"]
    acquisition = summary["acquisition"]
    partial = summary["partial"]
    turns = summary["turns"]
    back = summary["back_up"]
    approach = summary["approach"]
    stop = summary["stop"]
    others = summary["wrong_person"]
    out: list[tuple[str, bool, str]] = []

    def gate(name: str, ok: bool, detail: str) -> None:
        out.append((name, bool(ok), detail))

    # -- the sequence ---------------------------------------------------------
    gate("sequence_exact_order", sequence["matches"],
         f"accepted {sequence['accepted']} == expected {sequence['expected']}")
    gate("sequence_six_commands", sequence["count"] == 6,
         f"{sequence['count']} commands accepted")
    gate("session_completed", summary["final_state"] == "DONE",
         f"final state {summary['final_state']}")
    gate("no_timeouts", not summary["timeouts"],
         f"{len(summary['timeouts'])} phase ceilings fired: {summary['timeouts']}")

    # -- who ------------------------------------------------------------------
    gate("locked_the_instructor", acquisition["locked"] == INSTRUCTOR,
         f"locked {acquisition['locked']!r}, wanted {INSTRUCTOR!r}")
    gate("acquisition_had_alternatives",
         len(acquisition["people_seen_during_search"]) >= 2,
         f"camera confirmed {acquisition['people_seen_during_search']} "
         "during the search")
    gate("every_episode_from_instructor",
         all(e["person"] == INSTRUCTOR for e in summary["episodes"]),
         f"{sum(1 for e in summary['episodes'] if e['person'] != INSTRUCTOR)}"
         " episodes from somebody else")

    # -- the wrong-person refusal ----------------------------------------------
    readable_distractors = [n for n, e in others.items()
                            if e["readable_command_ticks"] > 0]
    sustained = [n for n, e in others.items() if e["sustained_past_confirm"]]
    gate("distractor_gestures_were_readable", len(readable_distractors) >= 1,
         f"{readable_distractors} were classifiable, fully readable and in "
         "range - the state an identity-blind robot would have obeyed")
    gate("distractor_gesture_sustained_past_confirm", len(sustained) >= 1,
         f"{sustained} held a full command past the {CONFIRM_S:.2f}s confirm "
         "window and were still ignored")
    gate("zero_wrong_person_commands",
         all(e["person"] == INSTRUCTOR for e in summary["episodes"]),
         "no episode was opened on a distractor")

    # -- the ambiguous partial -------------------------------------------------
    gate("partial_rejected", partial["accepted"] == 0,
         f"{partial['accepted']} commands accepted during the partial gesture")
    gate("partial_was_visible", partial["visible_fraction"] >= 0.95,
         f"instructor visible on {partial['visible_fraction'] * 100:.1f}% of "
         "the partial's ticks - so it was refused on its measurement, not on "
         "not being seen")
    gate("partial_was_readable", partial["readable_fraction"] >= 0.95,
         f"arm fully readable on {partial['readable_fraction'] * 100:.1f}% of "
         "its ticks")
    gate("partial_rejection_logged", partial["logged_rejections"] >= 1,
         f"{partial['logged_rejections']} explicit rejections logged")

    # -- the camera gate --------------------------------------------------------
    worst_visible = min(
        (e["confirm_visible_fraction"] for e in summary["episodes"]),
        default=0.0)
    worst_readable = min(
        (e["confirm_arm_readable_fraction"] for e in summary["episodes"]),
        default=0.0)
    gate("every_command_camera_confirmed", worst_visible >= 0.95,
         f"worst episode had the instructor visible on "
         f"{worst_visible * 100:.1f}% of its confirm ticks")
    gate("every_command_arm_readable", worst_readable >= 0.95,
         f"worst episode had the arm fully readable on "
         f"{worst_readable * 100:.1f}% of its confirm ticks")
    gate("confirm_window_sustained",
         all(e["confirm_held_s"] >= CONFIRM_S - 1e-9
             for e in summary["episodes"]),
         f"every acceptance held at least {CONFIRM_S:.2f}s")
    gate("monitor_visibility", tally["monitor_visible_fraction"] >= 0.95,
         f"instructor visible on {tally['monitor_visible_fraction'] * 100:.2f}%"
         " of monitoring ticks")

    # -- the physical actions ----------------------------------------------------
    gate("approach_closed_real_distance",
         approach["range_reduction_m"] >= 0.30,
         f"range {approach['start_range_m']:.3f} -> "
         f"{approach['end_range_m']:.3f} m "
         f"({approach['range_reduction_m']:+.3f} m)")
    gate("approach_walked_real_path", approach["path_m"] >= 0.30,
         f"{approach['path_m']:.3f} m of real path")
    gate("stop_interrupted_real_motion",
         float(stop["command_before_stop"]) > 0.0,
         f"the command on the tick before the STOP was "
         f"{stop['command_before_stop']:.3f}, and it interrupted "
         f"{stop['interrupts_command']!r}")
    gate("stop_zeroed_within_one_tick",
         stop["ticks_to_zero"] is not None and stop["ticks_to_zero"] <= 1,
         f"exact zero reached {stop['ticks_to_zero']} ticks after the STOP "
         "state began")
    gate("stop_held_still",
         float(stop["hold_s"]) >= STOP_HOLD_S - 1.5 / CTRL_HZ,
         f"held {stop['hold_s']:.2f}s below the settled speed "
         f"(bar {STOP_HOLD_S:.2f}s, less the one control tick the measurement "
         "lags the machine by)")
    gate("stop_drift_negligible", float(stop["drift_m"]) <= 0.05,
         f"{stop['drift_m'] * 1000:.1f} mm of drift while stopped")
    gate("turn_left_real_heading_change",
         bool(turns.get("TURN_LEFT", {}).get("reached")),
         f"trunk yaw turned {turns.get('TURN_LEFT', {}).get('turned_deg')} deg "
         f"(target {TURN_TARGET_DEG:+.0f})")
    gate("turn_right_real_heading_change",
         bool(turns.get("TURN_RIGHT", {}).get("reached")),
         f"trunk yaw turned {turns.get('TURN_RIGHT', {}).get('turned_deg')} deg"
         f" (target {-TURN_TARGET_DEG:+.0f})")
    gate("turns_are_opposite", summary["turns_opposite"],
         "the two named turns produced opposite-signed real heading changes")
    gate("turns_were_walked_arcs",
         all(t["path_m"] >= 0.20 for t in turns.values()),
         "each turn walked a real arc rather than pivoting")
    gate("back_up_real_displacement", back["reached"],
         f"{back['back_m']:.3f} m backward along the pre-action heading "
         f"(target {BACK_UP_TARGET_M:.2f})")
    gate("back_up_used_reverse_gait",
         float(back["command_vx_min"]) <= VX_REVERSE_ONSET + GAIT_ONSET_EPS,
         f"reverse command reached {back['command_vx_min']:.3f}, at or past "
         f"the measured onset {VX_REVERSE_ONSET:.2f}")

    # -- the command contract -----------------------------------------------------
    gate("zero_states_exactly_zero", tally["zero_violation_count"] == 0,
         f"{tally['zero_violation_count']} ticks where a zero-command state "
         "emitted a nonzero command")
    gate("no_sub_gait_commands", tally["sub_gait_ticks"] == 0,
         f"{tally['sub_gait_ticks']} ticks commanded between zero and a "
         "measured gait onset")
    gate("no_lateral_command", tally["max_abs_vy"] == 0.0,
         f"max |vy| = {tally['max_abs_vy']}")
    gate("stillness_is_real", tally["worst_zero_episode_m"] <= 0.10,
         f"worst zero-command episode accumulated "
         f"{tally['worst_zero_episode_m'] * 1000:.1f} mm of path")

    # -- safety --------------------------------------------------------------------
    gate("zero_contacts", tally["contacts"] == 0,
         f"{tally['contacts']} ticks with a nonpositive surface clearance")
    gate("positive_clearance_to_people",
         tally["min_clearance_m"] is not None and tally["min_clearance_m"] > 0.0,
         f"closest approach to a person {tally['min_clearance_m']:.4f} m "
         f"({tally['min_clearance_body']} at "
         f"{tally['min_clearance_t_s']:.2f}s)")
    gate("clearance_outside_standoff_floor",
         tally["min_clearance_m"] is not None
         and tally["min_clearance_m"] >= STANDOFF_MIN_M - 0.02,
         f"never closer than {tally['min_clearance_m']:.4f} m to anybody")
    gate("positive_clearance_to_scenery",
         tally["min_scenery_gap_m"] is not None
         and tally["min_scenery_gap_m"] > 0.0,
         f"closest approach to scenery {tally['min_scenery_gap_m']:.4f} m "
         f"({tally['min_scenery_geom']})")
    gate("stayed_inside_the_area", tally["outside_area_ticks"] == 0,
         f"{tally['outside_area_ticks']} ticks outside the training floor")
    gate("zero_falls", tally["fallen_steps"] == 0,
         f"{tally['fallen_steps']} ticks below {FALLEN_TRUNK_Z:.2f} m")
    gate("trunk_height_held",
         tally["min_trunk_z"] >= FALLEN_TRUNK_Z,
         f"minimum trunk height {tally['min_trunk_z']:.4f} m")
    gate("final_height_nominal",
         abs(tally["final_trunk_z"] - NOMINAL_TRUNK_Z) <= 0.012,
         f"final trunk height {tally['final_trunk_z']:.4f} m against a nominal "
         f"{NOMINAL_TRUNK_Z:.3f}")
    gate("no_forbidden_states",
         all(t["to"] not in FORBIDDEN_STATES for t in summary["transitions"]),
         f"none of {list(FORBIDDEN_STATES)} was ever entered")

    # -- the policy ------------------------------------------------------------------
    policy = summary["policy"]
    gate("stock_policy_sha256", policy["sha256_matches_stock"],
         f"{policy['sha256'][:16]}... matches the frozen stock walking policy")
    gate("observation_is_61d", policy["obs_dim"] == 61,
         f"{policy['obs_dim']}-D observation")
    gate("action_scale_is_0_9", policy["action_scale"] == 0.9,
         f"action scale {policy['action_scale']}")
    gate("exact_gyro_sensor", policy["gyro_sensor"] == "imu_ang_vel",
         f"angular-velocity sensor {policy['gyro_sensor']!r}")
    gate("control_rate_50hz", summary["ctrl_hz"] == 50.0,
         f"{summary['ctrl_hz']} Hz with decimation "
         f"{summary['decimation']} from a {summary['timestep']} s timestep")

    return out
