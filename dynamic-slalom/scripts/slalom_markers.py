#!/usr/bin/env python3
"""Posing the world-space marker discs: the plan, the corridors, the predictions.

Split out of ``rollout_slalom`` so the tick loop stays about ORDER.  These are
presentation, but they are driven from the SAME objects the behavior acts on -
the corridor the planner scored, the predictions it scored it against, the trail
the duck actually walked - so the discs cannot show a decision the duck did not
make.

Every body here is mocap with ``contype="0" conaffinity="0"``, so none can touch
the robot, and a marker with nothing to show is parked at ``z = -3`` below the
floor rather than deleted: MuJoCo has no way to remove a body from a compiled
model, and a disc under the floor is invisible from every camera.
"""

from __future__ import annotations

import numpy as np

# How many of each marker the scene carries.  These MUST match
# ``tools/build_scene.py``; ``test_the_scene_carries_every_marker`` pins them.
ROUTE_DISCS = 26
TRAIL_DISCS = 18
CORRIDOR_DISCS = 8
PRED_DISCS = 10
# Every ``trail_i`` disc samples the duck's own recent path this far apart, in
# control ticks.  At 50 Hz and the MEASURED 0.129 m/s cruise that is 0.049 m
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


def corridor_points(duck_xy, offset_m: float, count: int = CORRIDOR_DISCS,
                    span_m: float = 1.7) -> list[np.ndarray]:
    """Sample one candidate corridor's line, for the discs that draw it.

    Uses the planner's OWN geometry, so the lane drawn on the floor is the lane
    that was scored rather than a parallel line drawn to look like it.
    """
    from slalom_plan import duck_at
    from slalom_states import SPEED_AT_WALK

    horizon = span_m / max(SPEED_AT_WALK, 1e-6)
    return [duck_at(duck_xy, offset_m, horizon * (i + 1) / count)
            for i in range(count)]


def pose_markers(model, data, *, plan_points, trail_points, left_points,
                 right_points, pred_points, goal_xy) -> None:
    """Show the duck's plan, both candidate corridors and the predictions.

    ``pred_points`` are the PREDICTED positions of the threatening bodies over
    the horizon - the actual occupancy the planner scored - so a viewer can see
    the thing the robot was reasoning about rather than only its conclusion.
    """
    for index in range(ROUTE_DISCS):
        point = plan_points[index] if index < len(plan_points) else None
        place(model, data, f"route_{index}", point, 0.005)

    for index in range(TRAIL_DISCS):
        point = trail_points[index] if index < len(trail_points) else None
        place(model, data, f"trail_{index}", point, 0.004)

    for index in range(CORRIDOR_DISCS):
        left = left_points[index] if left_points and index < len(left_points) \
            else None
        right = right_points[index] if right_points \
            and index < len(right_points) else None
        place(model, data, f"left_{index}", left, 0.006)
        place(model, data, f"right_{index}", right, 0.006)

    for index in range(PRED_DISCS):
        point = pred_points[index] if index < len(pred_points) else None
        place(model, data, f"pred_{index}", point, 0.005)

    place(model, data, "goal_marker", goal_xy, 0.009)


def plan_polyline(duck_xy, goal_xy, count: int = ROUTE_DISCS) -> list[np.ndarray]:
    """The duck's current straight plan to the goal, sampled.

    Recomputed from where the duck IS, every tick, which is why the plan visibly
    swings back onto the goal after each pass: that is the replan, drawn.
    """
    start = np.asarray(duck_xy, dtype=np.float64)[:2]
    goal = np.asarray(goal_xy, dtype=np.float64)[:2]
    return [start + (goal - start) * (i / (count - 1)) for i in range(count)]
