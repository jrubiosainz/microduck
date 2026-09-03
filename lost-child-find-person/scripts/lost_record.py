#!/usr/bin/env python3
"""Per-tick record: everything the HUD draws and the gate grades, in one dict.

Kept in its own module so ``rollout_lost`` stays about ORDER — what happens
before what, within a control tick — and this stays about CONTENT.
"""

from __future__ import annotations

import math

import numpy as np

from lost_cast import BY_NAME, GUARDIAN


def build_record(*, display_t, state, machine, cycle_index, command, duck_xy,
                 duck_yaw_after, duck_pos, min_trunk_z, subject, camera_state,
                 clearances, nearest, scenery_gap, scenery_geom, people,
                 trail, sighting, guardian_range, guardian_visible,
                 guardian_blocker, confirmed_s, invisible_for, route,
                 route_target, los_available, los_blocker, path_m,
                 state_elapsed, rejections) -> dict:
    """One control tick, flattened into the record the HUD and metrics read."""
    guardian_entry = camera_state["people"].get(GUARDIAN.name, {})
    return {
        "t": round(float(display_t), 4),
        "state": state,
        "state_elapsed_s": round(float(state_elapsed), 3),
        "cycle_index": int(cycle_index),
        "command": [round(float(v), 4) for v in command],
        "command_peak": round(float(np.max(np.abs(command))), 6),
        "duck_xy": [round(float(duck_xy[0]), 4), round(float(duck_xy[1]), 4)],
        "duck_yaw_deg": round(math.degrees(float(duck_yaw_after)), 2),
        "trunk_z": round(float(duck_pos[2]), 5),
        "min_trunk_z": round(float(min_trunk_z), 5),
        "path_m": round(float(path_m), 4),

        # -- the guardian ------------------------------------------------
        "guardian": machine.guardian,
        "guardian_visible": bool(guardian_visible),
        "guardian_samples": list(guardian_entry.get("samples", [])),
        "guardian_sample_count": int(guardian_entry.get("sample_count", 0)),
        "guardian_range_m": (None if guardian_range is None
                             else round(float(guardian_range), 4)),
        "guardian_blocked_by": guardian_blocker,
        "guardian_readable": list(guardian_entry.get("readable", [])),
        "invisible_for_s": round(float(invisible_for), 3),
        "confirmed_s": round(float(confirmed_s), 3),

        # -- the head ----------------------------------------------------
        "subject": subject,
        "scanning": bool(camera_state["scanning"]),
        "view_yaw_deg": round(math.degrees(camera_state["view_yaw"]), 2),
        "gaze_yaw_deg": round(math.degrees(camera_state["gaze_yaw"]), 2),
        "scan_reversals": int(camera_state["scan_reversals"]),
        "visible_people": list(camera_state["visible_people"]),

        # -- the current sighting ---------------------------------------
        "sighting": None if sighting is None else sighting.as_record(),
        "rejections": [
            {"name": r["name"], "score": r["score"], "reason": r["reason"],
             "t": r["rejected_at_s"]}
            for r in rejections
        ],

        # -- memory and planning ----------------------------------------
        "trail": trail.as_record(),
        "route": None if route is None else route.as_record(),
        "route_target": (None if route_target is None else
                         [round(float(route_target[0]), 4),
                          round(float(route_target[1]), 4)]),
        "los_available": bool(los_available),
        "los_blocked_by": los_blocker,

        # -- safety ------------------------------------------------------
        "min_person_clearance_m": round(float(clearances[nearest]), 4),
        "nearest_person": nearest,
        "scenery_clearance_m": round(float(scenery_gap), 4),
        "nearest_scenery": scenery_geom,

        # -- the world ---------------------------------------------------
        "person_xy": {
            name: [round(float(s.pos[0]), 4), round(float(s.pos[1]), 4)]
            for name, s in people.items()
        },
        "person_role": {name: BY_NAME[name].role for name in people},
        "person_visible": {
            name: bool(camera_state["people"][name]["visible"])
            for name in people
        },
    }
