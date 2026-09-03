#!/usr/bin/env python3
"""Assembly of the per-tick record.

One dictionary per control step, carrying everything the gates, the HUD and the
metrics read.  Split out of the rollout so the integration loop stays legible:
the loop decides WHAT happens in what order, this decides what gets written
down.

Note ``person_arc_m``: the arc length of every queue member is recorded because
the no-overtaking gate is graded on arc length rather than on distance.  After
the fold the duck can be nearer the counter in a straight line while still being
genuinely behind in the queue, and only arc length distinguishes those.
"""

from __future__ import annotations

import math

from queue_geometry import AT_COUNTER_ARC_M


def build_record(*, display_t, state, machine, cycle_index, command, duck_xy,
                 duck_yaw_after, duck_pos, min_trunk_z, arc, cross, off_path,
                 subject, display_reading, truth, display_gaps,
                 rejected_available, camera_state, clearances, nearest,
                 scenery_gap, scenery_geom, display_people, path_m,
                 state_elapsed) -> dict:
    """Everything one control tick needs to record about itself."""
    return {
        "t": display_t,
        "state": state,
        "state_elapsed_s": state_elapsed,
        "cycle": cycle_index,
        "command": [float(v) for v in command],
        "duck_xy": list(duck_xy),
        "duck_yaw_deg": math.degrees(duck_yaw_after),
        "duck_arc_m": float(arc),
        "duck_cross_track_m": float(cross),
        "duck_off_path_m": float(off_path),
        "trunk_z_m": float(duck_pos[2]),
        "min_trunk_z_m": min_trunk_z,
        "target_arc_m": machine.target_arc,
        "predecessor": subject,
        "predecessor_arc_m": (
            display_reading.members[subject].arc_m
            if subject in display_reading.members else None),
        "standoff_m": (
            float(arc) - display_reading.members[subject].arc_m
            if subject in display_reading.members else None),
        "predecessors_remaining": sum(
            1 for name in display_reading.order
            if display_reading.members[name].arc_m < arc),
        "inferred_order": list(display_reading.order),
        "true_order": truth,
        "inferred_tail": display_reading.tail,
        "naive_range_tail": display_reading.naive_tail("by_range"),
        "naive_x_tail": display_reading.naive_tail("by_max_minus_x"),
        "excluded": {k: round(v, 4)
                     for k, v in display_reading.excluded.items()},
        "gaps": [g.as_record() for g in display_gaps],
        "rejected_gap_names": [g.name for g in rejected_available],
        "subject_visible": bool(camera_state["subject_visible"]),
        "subject_fraction": float(camera_state["subject_fraction"]),
        "visible_people": list(camera_state["visible_people"]),
        "view_yaw_deg": math.degrees(camera_state["view_yaw"]),
        "gaze_yaw_deg": math.degrees(camera_state["gaze_yaw"]),
        "nearest_person": nearest,
        "nearest_clearance_m": float(clearances[nearest]),
        "person_clearances": {k: float(v) for k, v in clearances.items()},
        "scenery_clearance_m": float(scenery_gap),
        "scenery_nearest_geom": scenery_geom,
        "person_xy": {
            name: [float(p.pos[0]), float(p.pos[1])]
            for name, p in display_people.items()},
        "person_in_queue": {
            name: bool(p.in_queue) for name, p in display_people.items()},
        # Arc length of every QUEUE MEMBER, which is what the no-overtaking
        # gate is graded on.  Distance from the counter cannot express it:
        # after the fold the duck can be nearer the counter in a straight
        # line while still being genuinely behind in the queue.
        "person_arc_m": {
            name: float(member.arc_m)
            for name, member in display_reading.members.items()},
        "at_counter": bool(arc <= AT_COUNTER_ARC_M),
        "path_m": path_m,
        "completed_cycles": len(machine.cycles),
    }
