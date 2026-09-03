#!/usr/bin/env python3
"""Posing the world-space marker discs: the route, the memory, the standoff.

Split out of ``rollout_patrol`` so the tick loop stays about ORDER.  These are
presentation, but they are driven from the SAME objects the behavior acts on -
the circuit the plan walks, the interruption point the plan remembered, the
standoff the planner chose - so a disc cannot show a decision the duck did not
make.

THE MEMORY DISCS ARE THE ONE WORTH EXPLAINING.  While an investigation is under
way they draw the line from where the duck broke off back to the checkpoint it
was walking to - the route it is holding in memory while it is somewhere else.
A viewer can watch the duck leave that line, do its work, and come back to the
exact point it left, which is the claim the whole behavior turns on.

Every body here is mocap with ``contype="0" conaffinity="0"``, so none can touch
the robot, and a marker with nothing to show is parked at ``z = -3`` below the
floor rather than deleted: MuJoCo has no way to remove a body from a compiled
model, and a disc under the floor is invisible from every camera.
"""

from __future__ import annotations

import numpy as np

# How many of each marker the scene carries.  These MUST match
# ``tools/build_scene.py``; ``test_the_scene_carries_every_marker`` pins them.
ROUTE_DISCS = 22
TRAIL_DISCS = 18
MEMORY_DISCS = 10
STANDOFF_DISCS = 8
# Every ``trail_i`` disc samples the duck's own recent path this far apart, in
# control ticks.  At 50 Hz and the MEASURED 0.128 m/s cruise that is 0.049 m
# between discs, which reads as a dotted line rather than a smear.
TRAIL_STRIDE = 19


def place(model, data, body_name: str, xy, z: float) -> None:
    """Move one marker, or park it below the floor when it has nothing to show."""
    body = model.body(body_name)
    mocap = int(model.body_mocapid[body.id])
    if xy is None:
        data.mocap_pos[mocap] = (0.0, 0.0, -3.0)
        return
    data.mocap_pos[mocap] = (float(xy[0]), float(xy[1]), z)


def memory_points(resume_xy, target_xy, count: int = MEMORY_DISCS
                  ) -> list[np.ndarray]:
    """The remembered route: from the interruption point to the checkpoint.

    Drawn only while an interruption is open, which is exactly when the duck is
    holding a route it is not currently walking.
    """
    start = np.asarray(resume_xy, dtype=np.float64)[:2]
    end = np.asarray(target_xy, dtype=np.float64)[:2]
    return [start + (end - start) * (i / max(count - 1, 1))
            for i in range(count)]


def standoff_points(duck_xy, standoff_xy, count: int = STANDOFF_DISCS
                    ) -> list[np.ndarray]:
    """The approach line the duck is walking to reach its standoff point."""
    start = np.asarray(duck_xy, dtype=np.float64)[:2]
    end = np.asarray(standoff_xy, dtype=np.float64)[:2]
    return [start + (end - start) * ((i + 1) / count) for i in range(count)]


def pose_markers(model, data, *, route_points, trail_points, memory,
                 standoff, target_xy, checkpoint_xy) -> None:
    """Show the circuit, the trail, the remembered route and the standoff."""
    for index in range(ROUTE_DISCS):
        point = route_points[index] if index < len(route_points) else None
        place(model, data, f"route_{index}", point, 0.005)

    for index in range(TRAIL_DISCS):
        point = trail_points[index] if index < len(trail_points) else None
        place(model, data, f"trail_{index}", point, 0.004)

    for index in range(MEMORY_DISCS):
        point = memory[index] if memory and index < len(memory) else None
        place(model, data, f"memory_{index}", point, 0.006)

    for index in range(STANDOFF_DISCS):
        point = standoff[index] if standoff and index < len(standoff) else None
        place(model, data, f"standoff_{index}", point, 0.006)

    place(model, data, "target_marker", target_xy, 0.008)
    place(model, data, "checkpoint_marker", checkpoint_xy, 0.009)
