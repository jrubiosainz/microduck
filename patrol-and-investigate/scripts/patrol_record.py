#!/usr/bin/env python3
"""Per-tick record: everything the gate grades and the HUD draws, in one dict.

Kept in its own module so ``rollout_patrol`` stays about ORDER - what happens
before what, within a control tick - and this stays about CONTENT.
"""

from __future__ import annotations

import math

import numpy as np

from patrol_aim import role_of
from patrol_cast import BY_NAME


def build_record(*, display_t, state, machine, plan, command, duck_xy,
                 duck_yaw_after, duck_pos, min_trunk_z, camera_state,
                 clearances, nearest, scenery_gap, scenery_geom, bodies, sense,
                 subject, subject_visible, subject_blocker, los_available,
                 los_blocker, path_m, state_elapsed, target_xy, target_kind,
                 interlock, standoff_plan, verdict, investigation,
                 zone_gap_m, scan_arc_deg, camera_active) -> dict:
    """One control tick, flattened into the record the metrics read."""
    open_interruption = plan.open_interruption
    investigated = machine.subject
    target_clearance = (clearances.get(investigated)
                        if investigated in clearances else None)
    return {
        "t": round(float(display_t), 4),
        "state": state,
        "state_elapsed_s": round(float(state_elapsed), 3),
        "command": [round(float(v), 4) for v in command],
        "command_peak": round(float(np.max(np.abs(command))), 6),
        "command_vx": round(float(command[0]), 6),
        "command_wz": round(float(command[2]), 6),
        "duck_xy": [round(float(duck_xy[0]), 4), round(float(duck_xy[1]), 4)],
        "duck_yaw_deg": round(math.degrees(float(duck_yaw_after)), 2),
        "trunk_z": round(float(duck_pos[2]), 5),
        "min_trunk_z": round(float(min_trunk_z), 5),
        "path_m": round(float(path_m), 4),

        # -- the patrol ---------------------------------------------------
        "target_name": sense.target_name,
        "target_remaining_m": round(float(sense.target_remaining_m), 4),
        "completed": int(sense.completed),
        "completed_names": list(plan.completed),
        "checkpoint_total": 5,
        "finished_circuit": bool(sense.finished_circuit),
        "target_xy": (None if target_xy is None else
                      [round(float(target_xy[0]), 4),
                       round(float(target_xy[1]), 4)]),
        "target_kind": target_kind,

        # -- the route memory ----------------------------------------------
        "interrupted": open_interruption is not None,
        "interrupted_target": ("" if open_interruption is None
                               else open_interruption.target_name),
        "resume_xy": (None if open_interruption is None else
                      [round(float(open_interruption.resume_xy[0]), 4),
                       round(float(open_interruption.resume_xy[1]), 4)]),
        "resume_remaining_m": (
            None if not np.isfinite(sense.resume_remaining_m)
            else round(float(sense.resume_remaining_m), 4)),
        "interruptions": len(plan.interruptions),

        # -- the detection and the verdict -----------------------------------
        "candidate": sense.candidate,
        "candidate_verdict": sense.candidate_verdict,
        "candidate_rule": sense.candidate_rule,
        "candidate_confidence": round(float(sense.candidate_confidence), 4),
        "candidate_range_m": (None if not np.isfinite(sense.candidate_range_m)
                              else round(float(sense.candidate_range_m), 4)),
        "candidate_investigate": bool(sense.candidate_investigate),
        "verdict": verdict or {},
        "verdicts_so_far": [v.get("verdict", "") for v in machine.verdicts],
        "verdict_targets": [v.get("target", "") for v in machine.verdicts],

        # -- the investigation -------------------------------------------------
        "investigation_index": (None if investigation is None
                                else investigation.index),
        "investigation_target": ("" if investigation is None
                                 else investigation.target),
        "target_range_m": (None if not np.isfinite(sense.target_range_m)
                           else round(float(sense.target_range_m), 4)),
        # THE MEASURED SURFACE CLEARANCE TO THE INVESTIGATED BODY, which is the
        # quantity the standoff BAND is defined in and the quantity the gate
        # grades.  ``target_range_m`` beside it is CENTRE-TO-CENTRE, and the two
        # differ by both bodies' radii - about 0.29 m for the crate.  The HUD
        # drew the range against the band at first, which put the tick outside a
        # window the duck was correctly inside: a bar that contradicts the
        # passing gate beside it is worse than no bar at all.
        "target_clearance_m": (None if target_clearance is None
                               else round(float(target_clearance), 4)),
        "standoff_m": (None if standoff_plan is None or not standoff_plan.ok
                       else round(float(standoff_plan.chosen.standoff_m), 4)),
        "standoff_xy": (None if standoff_plan is None or not standoff_plan.ok
                        else [round(float(standoff_plan.chosen.xy[0]), 4),
                              round(float(standoff_plan.chosen.xy[1]), 4)]),
        "standoff_rejected": (0 if standoff_plan is None else
                              sum(1 for c in standoff_plan.candidates
                                  if not c.ok)),
        "in_standoff_band": bool(sense.in_standoff_band),
        "observations_done": int(sense.observations_done),
        "observe_angle_deg": round(float(machine.observe_angle_deg), 2),

        # -- the interlock -------------------------------------------------------
        "interlock_blocked": bool(interlock.blocked),
        "interlock_reason": interlock.reason,
        "interlock_body": interlock.body,

        # -- what the camera is watching ---------------------------------------
        "subject": subject or "route",
        "subject_role": role_of(subject),
        "subject_visible": bool(subject_visible),
        "subject_blocked_by": subject_blocker,
        "camera_active": bool(camera_active),
        "los_available": bool(los_available),
        "los_blocked_by": los_blocker,
        "scan_arc_deg": round(float(scan_arc_deg), 2),

        # -- safety ----------------------------------------------------------
        "min_body_clearance_m": round(float(clearances[nearest]), 4),
        "nearest_body": nearest,
        "scenery_clearance_m": round(float(scenery_gap), 4),
        "nearest_scenery": scenery_geom,
        "zone_gap_m": round(float(zone_gap_m), 4),

        # -- the facility -------------------------------------------------------
        "actor_xy": {
            name: [round(float(s.pos[0]), 4), round(float(s.pos[1]), 4)]
            for name, s in bodies.items() if s.present
        },
        "actor_kind": {name: BY_NAME[name].kind for name in bodies},
        "actor_present": {name: bool(s.present) for name, s in bodies.items()},
        "actor_visible": {
            name: bool(camera_state["bodies"][name]["visible"])
            for name in bodies
        },
        "visible_bodies": list(camera_state["visible_bodies"]),
        "view_yaw_deg": round(math.degrees(camera_state["view_yaw"]), 2),
        "gaze_yaw_deg": round(math.degrees(camera_state["gaze_yaw"]), 2),
    }
