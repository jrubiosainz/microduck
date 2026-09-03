#!/usr/bin/env python3
"""Per-tick record: everything the gate grades and the HUD draws, in one dict.

Kept in its own module so ``rollout_slalom`` stays about ORDER - what happens
before what, within a control tick - and this stays about CONTENT.
"""

from __future__ import annotations

import math

import numpy as np

from slalom_aim import role_of
from slalom_cast import BY_NAME


def build_record(*, display_t, state, machine, command, duck_xy, duck_yaw_after,
                 duck_pos, min_trunk_z, camera_state, clearances, nearest,
                 scenery_gap, scenery_geom, actors, sense, decision, tracks,
                 subject, subject_visible, subject_blocker, los_available,
                 los_blocker, path_m, state_elapsed, target_xy, target_kind,
                 interlock, lane_offset_m, bodies_in_lane, goal_distance_m,
                 predictions, careful, encounter_index) -> dict:
    """One control tick, flattened into the record the metrics read."""
    subject_entry = (camera_state["bodies"].get(subject, {}) if subject
                     else camera_state["goal"])
    return {
        "t": round(float(display_t), 4),
        "state": state,
        "state_elapsed_s": round(float(state_elapsed), 3),
        "command": [round(float(v), 4) for v in command],
        "command_peak": round(float(np.max(np.abs(command))), 6),
        "command_vx": round(float(command[0]), 6),
        "command_wz": round(float(command[2]), 6),
        "careful": bool(careful),
        "duck_xy": [round(float(duck_xy[0]), 4), round(float(duck_xy[1]), 4)],
        "duck_yaw_deg": round(math.degrees(float(duck_yaw_after)), 2),
        "trunk_z": round(float(duck_pos[2]), 5),
        "min_trunk_z": round(float(min_trunk_z), 5),
        "path_m": round(float(path_m), 4),

        # -- progress toward the goal -----------------------------------------
        "goal_remaining_m": round(float(sense.goal_remaining_m), 4),
        "goal_distance_m": round(float(goal_distance_m), 4),
        "at_goal": bool(sense.at_goal),
        "lane_offset_m": round(float(lane_offset_m), 4),
        "lateral_error_m": round(float(sense.lateral_error_m), 4),
        "target_xy": (None if target_xy is None else
                      [round(float(target_xy[0]), 4),
                       round(float(target_xy[1]), 4)]),
        "target_kind": target_kind,
        "encounter_index": int(encounter_index),

        # -- the prediction and the decision -----------------------------------
        "threat": sense.threat,
        "threat_ttc_s": (None if not np.isfinite(sense.threat_ttc_s)
                         else round(float(sense.threat_ttc_s), 3)),
        "threat_range_m": (None if not np.isfinite(sense.threat_range_m)
                           else round(float(sense.threat_range_m), 4)),
        "threat_receding": bool(sense.threat_receding),
        "decision": decision.as_record(),
        "decision_side": sense.decision_side,
        "chosen_clearance_m": round(float(sense.chosen_clearance_m), 4),
        "rejected_side": sense.rejected_side,
        "rejected_clearance_m": round(float(sense.rejected_clearance_m), 4),
        "predicted_occupancy": predictions,
        "tracks": {
            t.name: {
                "pos": [round(float(t.pos[0]), 4), round(float(t.pos[1]), 4)],
                "vel": [round(float(t.velocity[0]), 4),
                        round(float(t.velocity[1]), 4)],
                "speed_mps": round(float(np.linalg.norm(t.velocity)), 4),
                "radius_m": round(float(t.radius), 4),
            } for t in tracks},

        # -- the interlock -------------------------------------------------------
        "interlock_blocked": bool(interlock.blocked),
        "interlock_reason": interlock.reason,
        "interlock_body": interlock.body,

        # -- what the camera is watching ---------------------------------------
        "subject": subject or "goal",
        "subject_role": role_of(subject),
        "subject_visible": bool(subject_visible),
        "subject_sample_count": int(subject_entry.get("sample_count", 0)),
        "subject_range_m": round(float(subject_entry.get("range_m", 0.0)), 4),
        "subject_blocked_by": subject_blocker,
        "los_available": bool(los_available),
        "los_blocked_by": los_blocker,
        "goal_visible": bool(camera_state["goal"]["visible"]),
        "goal_sample_count": int(camera_state["goal"]["sample_count"]),

        # -- safety ----------------------------------------------------------
        "min_body_clearance_m": round(float(clearances[nearest]), 4),
        "nearest_body": nearest,
        "scenery_clearance_m": round(float(scenery_gap), 4),
        "nearest_scenery": scenery_geom,

        # -- the world -------------------------------------------------------
        "bodies_in_lane": list(bodies_in_lane),
        "passes_completed": machine.completed_passes,
        "pass_sides": list(machine.pass_sides),
        "actor_xy": {
            name: [round(float(s.pos[0]), 4), round(float(s.pos[1]), 4)]
            for name, s in actors.items()
        },
        "actor_kind": {name: BY_NAME[name].kind for name in actors},
        "actor_encounter": {name: BY_NAME[name].encounter for name in actors},
        "actor_speed": {name: round(float(s.speed), 4)
                        for name, s in actors.items()},
        "actor_visible": {
            name: bool(camera_state["bodies"][name]["visible"])
            for name in actors
        },
        "visible_bodies": list(camera_state["visible_bodies"]),
        "view_yaw_deg": round(math.degrees(camera_state["view_yaw"]), 2),
        "gaze_yaw_deg": round(math.degrees(camera_state["gaze_yaw"]), 2),
    }
