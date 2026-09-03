#!/usr/bin/env python3
"""Turn one finished rollout into the summary every gate is graded on.

Split out of ``validate_gesture`` so that module stays about DECLARING GATES and
this stays about ASSEMBLING EVIDENCE.  The two are different jobs: this one
knows how to reach into a rollout, and the gate list must not.

Every quantity here is measured from physics or from the real camera.  Where
both a physical and a bookkeeping version of a number exist, the physical one
wins - a turn is the trunk-yaw delta the duck actually turned through, a reverse
is displacement projected on the heading it held when the command was accepted,
and an approach is the contact probe's own surface clearance.
"""

from __future__ import annotations

from gest_cast import INSTRUCTOR
from gest_control import GAIT_ONSET_EPS
from gest_gesture import MIN_CONFIDENCE, MOTION_WINDOW_S
from gest_script import EXPECTED_COMMANDS, INSTRUCTOR_CUES
from gest_states import (
    ACQUIRE_CONFIRM_S,
    BACK_UP_TARGET_M,
    BACK_UP_TOLERANCE_M,
    CONFIRM_MIN_FRACTION,
    CONFIRM_S,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
    STOP_HOLD_S,
    TURN_TARGET_DEG,
    TURN_TOLERANCE_DEG,
    VX_BACK_UP,
    VX_ONSET,
    VX_REVERSE_ONSET,
)
from policy_runtime import (
    ACTION_SCALE,
    CTRL_HZ,
    FALLEN_TRUNK_Z,
    GYRO_SENSOR,
    NOMINAL_TRUNK_Z,
    OBS_DIM,
)
from gest_arena import FIXTURES, INSTRUCTOR_MARK

# The SHA-256 of the frozen stock walking policy every sibling behavior uses.
# Compared rather than trusted: a behavior that quietly trained its own policy
# would still pass every other gate.
POLICY_SHA256 = "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"


def build_summary(rollout) -> dict:
    """Everything the gates and the README are graded on, from one real run."""
    records = rollout.records
    tally = rollout.tally.as_record()
    machine = rollout.machine.summary()
    episodes = machine["episodes"]

    by_command = {e["command"]: e for e in episodes}

    # -- the wrong-person evidence -------------------------------------------
    # For every distractor: how many ticks their gesture was CLASSIFIABLE, with
    # a fully readable arm, inside gesture range - which is precisely the state
    # in which an identity-blind robot would have obeyed them.
    others = {}
    for name, entry in rollout.detector.other_readings.items():
        windows = entry.get("windows", [])
        longest = max((w["ticks"] for w in windows), default=0)
        others[name] = {
            "readable_command_ticks": entry["readable_command_ticks"],
            "commands": entry["commands"],
            "max_confidence": round(float(entry["max_confidence"]), 4),
            "in_range_ticks": entry["in_range_ticks"],
            "first_s": entry["first_s"],
            "last_s": entry["last_s"],
            "windows": windows,
            "longest_window_ticks": longest,
            "longest_window_s": round(longest / CTRL_HZ, 3),
            # The whole point: sustained PAST the confirm window and still
            # ignored.  A distractor whose gesture was too brief to confirm
            # would have been refused by the clock rather than by identity.
            "sustained_past_confirm": bool(longest / CTRL_HZ >= CONFIRM_S),
        }

    # -- the partial gesture --------------------------------------------------
    partial_cue = next((c for c in INSTRUCTOR_CUES
                        if c.expect == "reject_partial"), None)
    partial_window = {"from_s": 0.0, "to_s": 0.0}
    if partial_cue is not None:
        partial_window = {"from_s": partial_cue.at_s, "to_s": partial_cue.ends_s}
    partial_ticks = [r for r in records
                     if partial_window["from_s"] <= r["t"] < partial_window["to_s"]]
    partial_visible = sum(1 for r in partial_ticks if r["instructor_visible"])
    partial_readable = sum(1 for r in partial_ticks
                           if r["instructor_arm_readable"])
    partial_accepted = [
        e for e in episodes
        if partial_window["from_s"] <= e["confirmed_at_s"] < partial_window["to_s"]]
    partial_rejections = [
        rej for rej in rollout.detector.rejections
        if partial_window["from_s"] <= rej["t"] < partial_window["to_s"]
        and rej["person"] == INSTRUCTOR]

    # -- the two turns, graded on real trunk yaw ------------------------------
    turns = {}
    for command, direction in (("TURN_LEFT", "left"), ("TURN_RIGHT", "right")):
        entry = by_command.get(command)
        if entry is None:
            continue
        turned = float(entry["execute_yaw_delta_deg"])
        wanted = TURN_TARGET_DEG if direction == "left" else -TURN_TARGET_DEG
        turns[command] = {
            "direction": direction,
            "turned_deg": round(turned, 2),
            "target_deg": wanted,
            "error_deg": round(abs(turned - wanted), 2),
            "sign_correct": bool((turned > 0) == (wanted > 0)),
            "reached": bool(abs(turned - wanted) <= TURN_TOLERANCE_DEG),
            "path_m": entry["execute_path_m"],
        }
    opposite = (
        bool(turns["TURN_LEFT"]["turned_deg"] > 0
             > turns["TURN_RIGHT"]["turned_deg"])
        if len(turns) == 2 else False)

    # -- the reverse -----------------------------------------------------------
    back = by_command.get("BACK_UP", {})
    # -- the approach ----------------------------------------------------------
    come = by_command.get("COME", {})
    # -- the stop --------------------------------------------------------------
    stop = by_command.get("STOP", {})

    # Ticks the STOP episode took to reach an exact zero, measured by index.
    stop_zero_ticks = None
    if stop:
        started = stop["executed_at_s"]
        after = [r for r in records if r["t"] >= started
                 and r["state"] == "EXECUTE_STOP"]
        for index, entry in enumerate(after):
            if entry["command_peak"] == 0.0:
                stop_zero_ticks = index
                break

    zero_state_records = [r for r in records if r["state"] in (
        "READY", "OBSERVE", "CONFIRM", "EXECUTE_STOP", "ACK", "GOODBYE",
        "DONE")]

    return {
        "seconds": rollout.seconds,
        "ticks": len(records),
        "ctrl_hz": CTRL_HZ,
        "decimation": rollout.decimation,
        "timestep": float(rollout.model.opt.timestep),

        "policy": {
            "path": str(rollout.policy.path),
            "sha256": rollout.policy_sha256,
            "sha256_matches_stock": rollout.policy_sha256 == POLICY_SHA256,
            "obs_dim": OBS_DIM,
            "action_scale": ACTION_SCALE,
            "gyro_sensor": GYRO_SENSOR,
        },

        "sequence": {
            "accepted": machine["accepted_commands"],
            "expected": list(EXPECTED_COMMANDS),
            "matches": machine["accepted_commands"] == list(EXPECTED_COMMANDS),
            "count": len(machine["accepted_commands"]),
        },

        "acquisition": rollout.detector.acquisition.as_record(),
        "wrong_person": others,
        "partial": {
            "window": partial_window,
            "ticks": len(partial_ticks),
            "visible_ticks": partial_visible,
            "readable_ticks": partial_readable,
            "visible_fraction": round(
                partial_visible / max(len(partial_ticks), 1), 4),
            "readable_fraction": round(
                partial_readable / max(len(partial_ticks), 1), 4),
            "accepted": len(partial_accepted),
            "logged_rejections": len(partial_rejections),
            "rejection_reasons": [r["reason"] for r in partial_rejections[:4]],
        },

        "turns": turns,
        "turns_opposite": opposite,
        "back_up": {
            "back_m": back.get("execute_back_m", 0.0),
            "target_m": BACK_UP_TARGET_M,
            "path_m": back.get("execute_path_m", 0.0),
            "reached": bool(
                back.get("execute_back_m", 0.0)
                >= BACK_UP_TARGET_M - BACK_UP_TOLERANCE_M),
            "command_vx_min": back.get("command_vx_min", 0.0),
        },
        "approach": {
            "start_range_m": come.get("start_range_m", 0.0),
            "end_range_m": come.get("execute_end_range_m", 0.0),
            "range_reduction_m": round(
                float(come.get("start_range_m", 0.0))
                - float(come.get("execute_end_range_m", 0.0)), 4),
            "path_m": come.get("execute_path_m", 0.0),
            "min_clearance_m": come.get("execute_min_clearance_m"),
            "interrupted_by": come.get("interrupted_by", ""),
        },
        "stop": {
            "command_before_stop": stop.get("command_before_stop", 0.0),
            "interrupts_command": stop.get("interrupts_command", ""),
            "hold_s": stop.get("stop_hold_s", 0.0),
            "drift_m": stop.get("stop_drift_m", 0.0),
            "ticks_to_zero": stop_zero_ticks,
        },

        "episodes": episodes,
        "transitions": machine["transitions"],
        "timeouts": machine["timeouts"],
        "interrupts": machine["interrupts"],
        "rejections": rollout.detector.rejections,
        "final_state": machine["state"],

        "tally": tally,
        "zero_state_ticks_checked": len(zero_state_records),
        "geometry": {
            "duck_planar_radius_m": round(float(rollout.duck_radius), 4),
            "duck_exact_radius_m": round(float(rollout.duck_exact_radius), 4),
            "duck_lateral_half_m": round(float(rollout.duck_lateral_half), 4),
            "instructor_mark": list(INSTRUCTOR_MARK),
            "fixtures": [f.name for f in FIXTURES],
            "scenery_geoms": list(rollout.scenery_geoms),
        },
        "thresholds": {
            "confirm_s": CONFIRM_S,
            "confirm_min_fraction": CONFIRM_MIN_FRACTION,
            "acquire_confirm_s": ACQUIRE_CONFIRM_S,
            "motion_window_s": MOTION_WINDOW_S,
            "min_confidence": MIN_CONFIDENCE,
            "standoff_min_m": STANDOFF_MIN_M,
            "standoff_max_m": STANDOFF_MAX_M,
            "turn_target_deg": TURN_TARGET_DEG,
            "turn_tolerance_deg": TURN_TOLERANCE_DEG,
            "back_up_target_m": BACK_UP_TARGET_M,
            "stop_hold_s": STOP_HOLD_S,
            "vx_onset": VX_ONSET,
            "vx_reverse_onset": VX_REVERSE_ONSET,
            "vx_back_up": VX_BACK_UP,
            "gait_onset_eps": GAIT_ONSET_EPS,
            "fallen_trunk_z": FALLEN_TRUNK_Z,
            "nominal_trunk_z": NOMINAL_TRUNK_Z,
        },
    }
