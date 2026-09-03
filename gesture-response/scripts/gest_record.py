#!/usr/bin/env python3
"""One control tick, as a flat JSON-safe record.

Split out of the rollout so the tick body stays about ORDER and this stays about
WHAT IS WRITTEN DOWN.  Every field here is consumed by at least one of: an
acceptance gate, the HUD, or the metrics summary - there are no fields kept
"just in case", because an unread field is a claim nobody checks.

The record is what both the headless gate and the renderer read, which is what
makes the video and the numbers the same run rather than two runs that agree.
"""

from __future__ import annotations

import math

import numpy as np

from gest_cast import ALL_NAMES, INSTRUCTOR
from gest_states import (
    BACK_UP_TARGET_M,
    CONFIRM_S,
    GESTURE_MAX_RANGE_M,
    STOP_HOLD_S,
    TURN_TARGET_DEG,
)


def build_record(*, display_t, state, machine, detector, view, command,
                 duck_xy, duck_yaw_after, duck_pos, min_trunk_z, camera_state,
                 clearances, nearest, scenery_gap, scenery_geom, bodies, sense,
                 instructor_visible, arm_readable, los_available, los_blocker,
                 path_m, state_elapsed, target_xy, interlock,
                 camera_active) -> dict:
    """Everything one tick needs to be graded and drawn."""
    acquisition = detector.acquisition
    episode = machine.episode
    return {
        "t": round(float(display_t), 3),
        "state": state,
        "state_elapsed_s": round(float(state_elapsed), 3),

        # -- the duck -------------------------------------------------------
        "duck_xy": [round(float(duck_xy[0]), 4), round(float(duck_xy[1]), 4)],
        "duck_yaw_deg": round(math.degrees(float(duck_yaw_after)), 2),
        "trunk_z": round(float(duck_pos[2]), 5),
        "min_trunk_z": round(float(min_trunk_z), 5),
        "path_m": round(float(path_m), 4),
        "speed_mps": round(float(sense.measured_speed_mps), 4),
        "settled": bool(sense.settled),

        # -- the command ----------------------------------------------------
        "command": [round(float(v), 6) for v in np.asarray(command)],
        "command_peak": round(float(np.max(np.abs(np.asarray(command)))), 6),

        # -- who ------------------------------------------------------------
        "locked": acquisition.locked,
        "acquisition_state": acquisition.state,
        "people_seen_during_search": list(acquisition.seen),
        "instructor_visible": bool(instructor_visible),
        "instructor_arm_readable": bool(arm_readable),
        "instructor_range_m": round(float(sense.instructor_range_m), 4),
        "los_available": bool(los_available),
        "los_blocker": los_blocker,

        # -- what -----------------------------------------------------------
        "candidate_command": view.get("candidate_command", ""),
        "candidate_held_s": round(float(view.get("candidate_held_s", 0.0)), 3),
        "candidate_fraction": round(
            float(view.get("candidate_fraction", 0.0)), 4),
        "candidate_confidence": round(
            float(view.get("candidate_confidence", 0.0)), 4),
        "confirm_progress": round(float(view.get("confirm_progress", 0.0)), 4),
        "detector_suspended": bool(view.get("suspended", False)),
        "accepted_commands": list(machine.accepted_commands),
        "pose": view.get("pose"),
        "reading": view.get("reading"),

        # -- progress of the action under way ---------------------------------
        "yaw_delta_deg": round(float(sense.yaw_delta_deg), 2),
        "back_along_heading_m": round(float(sense.back_along_heading_m), 4),
        "in_standoff_band": bool(sense.in_standoff_band),
        "stop_hold_s": round(float(sense.stop_hold_s), 3),
        "episode_index": None if episode is None else episode.index,
        "episode_command": "" if episode is None else episode.command,

        # -- safety -----------------------------------------------------------
        "clearances": {n: round(float(clearances[n]), 4) for n in ALL_NAMES},
        "nearest_body": nearest,
        "min_clearance_m": round(float(clearances[nearest]), 4),
        "scenery_gap_m": round(float(scenery_gap), 4),
        "scenery_geom": scenery_geom,
        "inside_area": bool(sense.inside_area),
        "interlock_blocked": bool(interlock.blocked),
        "interlock_reason": interlock.reason,
        "interlock_body": interlock.body,

        # -- the camera --------------------------------------------------------
        "camera_active": bool(camera_active),
        "visible_bodies": list(camera_state["visible_bodies"]),
        "gaze_yaw_deg": round(math.degrees(float(camera_state["gaze_yaw"])), 2),
        "gaze_pitch_deg": round(
            math.degrees(float(camera_state["gaze_pitch"])), 2),
        "arm_readable": {
            n: camera_state["bodies"][n].get("arm_readable",
                                             {"l": False, "r": False})
            for n in ALL_NAMES},

        # -- the world (for the HUD only; no decision layer reads this) -------
        "bodies": {
            n: {"xy": [round(float(bodies[n].pos[0]), 4),
                       round(float(bodies[n].pos[1]), 4)],
                "yaw_deg": round(math.degrees(float(bodies[n].yaw)), 2),
                "present": bool(bodies[n].present),
                "gesture": bodies[n].gesture,
                "speed": round(float(bodies[n].speed), 4)}
            for n in ALL_NAMES},
        "target_xy": (None if target_xy is None
                      else [round(float(target_xy[0]), 4),
                            round(float(target_xy[1]), 4)]),
        "instructor": INSTRUCTOR,
        # The bars the HUD draws are graded against the SAME constants the
        # gates use, carried on the record so the overlay cannot drift from the
        # thresholds by importing its own copy.
        "thresholds": {
            "confirm_s": CONFIRM_S,
            "gesture_max_range_m": GESTURE_MAX_RANGE_M,
            "turn_target_deg": TURN_TARGET_DEG,
            "back_up_target_m": BACK_UP_TARGET_M,
            "stop_hold_s": STOP_HOLD_S,
        },
    }
