#!/usr/bin/env python3
"""Per-tick record: everything the gate grades and a HUD would draw, in one dict.

Kept in its own module so ``rollout_beside`` stays about ORDER — what happens
before what, within a control tick — and this stays about CONTENT.
"""

from __future__ import annotations

import math

import numpy as np

from beside_cast import BY_NAME, GUARDIAN


def build_record(*, display_t, state, machine, command, duck_xy, duck_yaw_after,
                 duck_pos, min_trunk_z, camera_state, clearances, nearest,
                 scenery_gap, scenery_geom, people, guardian_range,
                 guardian_visible, guardian_blocker, lateral, longitudinal,
                 verdicts, preferred, preference_reason, target_xy, target_kind,
                 los_available, los_blocker, path_m, state_elapsed,
                 formation_ok) -> dict:
    """One control tick, flattened into the record the metrics read."""
    guardian_entry = camera_state["people"].get(GUARDIAN.name, {})
    side = machine.side
    return {
        "t": round(float(display_t), 4),
        "state": state,
        "state_elapsed_s": round(float(state_elapsed), 3),
        "side": side,
        "side_name": (None if side is None
                      else ("left" if side == 1 else "right")),
        "target_side": machine.target_side,
        "command": [round(float(v), 4) for v in command],
        "command_peak": round(float(np.max(np.abs(command))), 6),
        "duck_xy": [round(float(duck_xy[0]), 4), round(float(duck_xy[1]), 4)],
        "duck_yaw_deg": round(math.degrees(float(duck_yaw_after)), 2),
        "trunk_z": round(float(duck_pos[2]), 5),
        "min_trunk_z": round(float(min_trunk_z), 5),
        "path_m": round(float(path_m), 4),

        # -- the formation ------------------------------------------------
        "lateral_m": round(float(lateral), 4),
        "lateral_abs_m": round(abs(float(lateral)), 4),
        "longitudinal_m": round(float(longitudinal), 4),
        "formation_ok": bool(formation_ok),
        "guardian": machine.guardian,
        "guardian_xy": [round(float(people[GUARDIAN.name].pos[0]), 4),
                        round(float(people[GUARDIAN.name].pos[1]), 4)],
        "guardian_yaw_deg": round(
            math.degrees(float(people[GUARDIAN.name].yaw)), 2),
        "guardian_speed_mps": round(float(people[GUARDIAN.name].speed), 4),
        "guardian_range_m": round(float(guardian_range), 4),
        "guardian_visible": bool(guardian_visible),
        "guardian_sample_count": int(guardian_entry.get("sample_count", 0)),
        "guardian_blocked_by": guardian_blocker,
        "los_available": bool(los_available),
        "los_blocked_by": los_blocker,

        # -- the side decision --------------------------------------------
        "verdict_left": verdicts[1].as_record(),
        "verdict_right": verdicts[-1].as_record(),
        "preferred_side": preferred,
        "preference_reason": preference_reason,
        "target_xy": (None if target_xy is None else
                      [round(float(target_xy[0]), 4),
                       round(float(target_xy[1]), 4)]),
        "target_kind": target_kind,

        # -- safety --------------------------------------------------------
        "min_person_clearance_m": round(float(clearances[nearest]), 4),
        "nearest_person": nearest,
        "scenery_clearance_m": round(float(scenery_gap), 4),
        "nearest_scenery": scenery_geom,

        # -- the world -----------------------------------------------------
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
    }
