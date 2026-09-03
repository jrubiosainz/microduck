#!/usr/bin/env python3
"""Per-tick record: everything the gate grades and the HUD draws, in one dict.

Kept in its own module so ``rollout_etiquette`` stays about ORDER - what happens
before what, within a control tick - and this stays about CONTENT.
"""

from __future__ import annotations

import math

import numpy as np

from etiquette_aim import role_of
from etiquette_cast import BY_NAME, GUARDIAN


def build_record(*, display_t, state, machine, command, duck_xy, duck_yaw_after,
                 duck_pos, min_trunk_z, camera_state, clearances, nearest,
                 scenery_gap, scenery_geom, people, doors, sense, tracker,
                 subject, subject_visible, subject_blocker, los_available,
                 los_blocker, path_m, state_elapsed, target_xy, target_kind,
                 interlock, zone_depths, aperture_occupancy, cabin_margin_m,
                 guardian_gap_m, careful) -> dict:
    """One control tick, flattened into the record the metrics read."""
    subject_entry = camera_state["people"].get(subject, {})
    return {
        "t": round(float(display_t), 4),
        "state": state,
        "state_elapsed_s": round(float(state_elapsed), 3),
        "command": [round(float(v), 4) for v in command],
        "command_peak": round(float(np.max(np.abs(command))), 6),
        "command_vx": round(float(command[0]), 6),
        "careful": bool(careful),
        "duck_xy": [round(float(duck_xy[0]), 4), round(float(duck_xy[1]), 4)],
        "duck_yaw_deg": round(math.degrees(float(duck_yaw_after)), 2),
        "trunk_z": round(float(duck_pos[2]), 5),
        "min_trunk_z": round(float(min_trunk_z), 5),
        "path_m": round(float(path_m), 4),

        # -- progress along the route ---------------------------------------
        **({"route_" + k: v for k, v in tracker.as_record().items()}
           if tracker is not None else {}),
        "target_xy": (None if target_xy is None else
                      [round(float(target_xy[0]), 4),
                       round(float(target_xy[1]), 4)]),
        "target_kind": target_kind,

        # -- the doors --------------------------------------------------------
        "door_fraction": {name: round(float(d.fraction), 4)
                          for name, d in doors.items()},
        "door_gap_m": {name: round(float(d.effective_gap_m), 4)
                       for name, d in doors.items()},
        "door_passable": {name: bool(d.passable) for name, d in doors.items()},

        # -- the zones ---------------------------------------------------------
        "zone_depth_m": {name: round(float(v), 4)
                         for name, v in zone_depths.items()},
        "aperture_occupancy": {
            name: {"duck": bool(entry["duck"]),
                   "others": list(entry["others"])}
            for name, entry in aperture_occupancy.items()},
        "cabin_margin_m": round(float(cabin_margin_m), 4),
        "inside_cabin": bool(sense.inside_cabin),

        # -- the guardian and the traffic --------------------------------------
        "guardian": machine.guardian,
        "guardian_xy": [round(float(people[GUARDIAN.name].pos[0]), 4),
                        round(float(people[GUARDIAN.name].pos[1]), 4)],
        "guardian_gap_m": round(float(guardian_gap_m), 4),
        "guardian_through_door": bool(sense.guardian_through_door),
        "guardian_through_lift": bool(sense.guardian_through_lift),
        "guardian_inside_cabin": bool(sense.guardian_inside_cabin),
        "guardian_through_rear": bool(sense.guardian_through_rear),
        "exiters_pending": int(sense.exiters_pending),
        "exiters_in_aperture": int(sense.exiters_in_aperture),
        "all_exiters_clear": bool(sense.all_exiters_clear),
        "occupants_exited": int(sense.occupants_exited),
        "occupants_in_cabin": int(sense.occupants_in_cabin),
        "occupants_in_passage": int(sense.occupants_in_passage),
        "all_occupants_clear": bool(sense.all_occupants_clear),
        "yields_completed": machine.completed_yields,

        # -- what the camera is watching ---------------------------------------
        "subject": subject,
        "subject_role": role_of(subject),
        "subject_visible": bool(subject_visible),
        "subject_sample_count": int(subject_entry.get("sample_count", 0)),
        "subject_range_m": round(float(subject_entry.get("range_m", 0.0)), 4),
        "subject_blocked_by": subject_blocker,
        "los_available": bool(los_available),
        "los_blocked_by": los_blocker,

        # -- the interlock -------------------------------------------------------
        "interlock_blocked": bool(interlock.blocked),
        "interlock_reason": interlock.reason,
        "interlock_aperture": interlock.aperture,

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
        "person_speed": {name: round(float(s.speed), 4)
                         for name, s in people.items()},
        "person_visible": {
            name: bool(camera_state["people"][name]["visible"])
            for name in people
        },
        "visible_people": list(camera_state["visible_people"]),
        "view_yaw_deg": round(math.degrees(camera_state["view_yaw"]), 2),
        "gaze_yaw_deg": round(math.degrees(camera_state["gaze_yaw"]), 2),
    }
