#!/usr/bin/env python3
"""Per-tick record: everything the gate grades and the HUD draws, in one dict.

Kept in its own module so ``rollout_guide`` stays about ORDER — what happens
before what, within a control tick — and this stays about CONTENT.
"""

from __future__ import annotations

import math

import numpy as np

from guide_cast import BY_NAME, FOLLOWER


def build_record(*, display_t, state, machine, command, duck_xy, duck_yaw_after,
                 duck_pos, min_trunk_z, camera_state, clearances, nearest,
                 scenery_gap, scenery_geom, people, follower, tracker,
                 follower_range, follower_visible, follower_blocker,
                 los_available, los_blocker, path_m, state_elapsed,
                 target_xy, target_kind, look_at_yaw, destination,
                 destination_distance, facing_error_deg, lagging, unseen,
                 waiting_spot, safety_breach_s) -> dict:
    """One control tick, flattened into the record the metrics read."""
    follower_entry = camera_state["people"].get(FOLLOWER.name, {})
    return {
        "t": round(float(display_t), 4),
        "state": state,
        "state_elapsed_s": round(float(state_elapsed), 3),
        "command": [round(float(v), 4) for v in command],
        "command_peak": round(float(np.max(np.abs(command))), 6),
        "command_vx": round(float(command[0]), 6),
        "duck_xy": [round(float(duck_xy[0]), 4), round(float(duck_xy[1]), 4)],
        "duck_yaw_deg": round(math.degrees(float(duck_yaw_after)), 2),
        "trunk_z": round(float(duck_pos[2]), 5),
        "min_trunk_z": round(float(min_trunk_z), 5),
        "path_m": round(float(path_m), 4),

        # -- the request and the plan --------------------------------------
        "requested_destination": machine.requested_key,
        "destination": None if destination is None else destination.key,
        "destination_label": None if destination is None else destination.label,
        "destination_xy": (None if destination is None else
                           [round(float(destination.xy[0]), 4),
                            round(float(destination.xy[1]), 4)]),
        "destination_distance_m": (None if destination_distance is None
                                   else round(float(destination_distance), 4)),
        "facing_error_deg": (None if facing_error_deg is None
                             else round(float(facing_error_deg), 2)),
        "candidates": list(machine.candidates),

        # -- progress along the route ---------------------------------------
        **({"route_" + k: v for k, v in tracker.as_record().items()}
           if tracker is not None else {}),
        "target_xy": (None if target_xy is None else
                      [round(float(target_xy[0]), 4),
                       round(float(target_xy[1]), 4)]),
        "target_kind": target_kind,
        "look_at_yaw_deg": (None if look_at_yaw is None
                            else round(math.degrees(float(look_at_yaw)), 2)),

        # -- the follower ----------------------------------------------------
        "follower": machine.follower,
        "follower_xy": [round(float(follower.pos[0]), 4),
                        round(float(follower.pos[1]), 4)],
        "follower_range_m": round(float(follower_range), 4),
        "follower_visible": bool(follower_visible),
        "follower_sample_count": int(follower_entry.get("sample_count", 0)),
        "follower_blocked_by": follower_blocker,
        "follower_speed_mps": round(float(follower.speed), 4),
        "follower_trail_gap_m": round(float(follower.trail_gap_m), 4),
        "follower_walked_m": round(float(follower.walked_m), 4),
        "follower_stall_label": follower.stall_label,
        "los_available": bool(los_available),
        "los_blocked_by": los_blocker,
        "lagging": bool(lagging),
        "unseen": bool(unseen),
        "safety_breach_s": round(float(safety_breach_s), 3),
        "waiting_spot": (None if waiting_spot is None else
                         [round(float(waiting_spot[0]), 4),
                          round(float(waiting_spot[1]), 4)]),
        "episodes_completed": machine.completed_episodes,

        # -- safety ----------------------------------------------------------
        "min_person_clearance_m": round(float(clearances[nearest]), 4),
        "nearest_person": nearest,
        "scenery_clearance_m": round(float(scenery_gap), 4),
        "nearest_scenery": scenery_geom,

        # -- the world -------------------------------------------------------
        "person_xy": {
            name: [round(float(s.pos[0]), 4), round(float(s.pos[1]), 4)]
            for name, s in people.items()
        },
        "person_role": {name: BY_NAME[name].role for name in people},
        "person_visible": {
            name: bool(camera_state["people"][name]["visible"])
            for name in people
        },
        "visible_people": list(camera_state["visible_people"]),
        "view_yaw_deg": round(math.degrees(camera_state["view_yaw"]), 2),
        "gaze_yaw_deg": round(math.degrees(camera_state["gaze_yaw"]), 2),
        "gesture_yaw_deg": round(camera_state["gesture_yaw_deg"], 2),
    }
