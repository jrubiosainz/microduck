#!/usr/bin/env python3
"""Posing the world-space marker discs.  Pure presentation, no physics.

Split out of ``rollout_guide`` so that the tick loop stays about ORDER.  Nothing
here is read by a gate: the markers exist so a viewer can SEE the searched route,
its waypoints, the destination and the spot the duck chose to wait at, drawn in
the world rather than only in the HUD.

They are mocap bodies with ``contype="0" conaffinity="0"``, so they cannot touch
the robot, and one parked marker sits at ``z = -3`` below the floor rather than
being deleted \u2014 MuJoCo has no way to remove a body from a compiled model, and a
disc under the floor is invisible from every camera in the scene.
"""

from __future__ import annotations

# How many of each marker the scene carries.  These MUST match
# ``tools/build_scene.py``; ``test_the_scene_carries_every_marker`` pins them.
ROUTE_DISCS = 26
WAYPOINT_DISCS = 8
TRAIL_DISCS = 14
# Every ``trail_i`` disc samples the duck's own recent path this far apart, in
# control ticks.  At 50 Hz and the measured 0.130 m/s lead pace that is 0.047 m
# between discs, which reads as a dotted line rather than a smear.
TRAIL_STRIDE = 18


def place(model, data, body_name: str, xy, z: float) -> None:
    """Move one marker, or park it below the floor when it has nothing to show."""
    body = model.body(body_name)
    mocap = int(model.body_mocapid[body.id])
    if xy is None:
        data.mocap_pos[mocap] = (0.0, 0.0, -3.0)
        return
    data.mocap_pos[mocap] = (float(xy[0]), float(xy[1]), z)


def pose_markers(model, data, *, state: str, route_points, plan, destination,
                 waiting_spot, records) -> None:
    """Show the planned route, its waypoints, the goal and the waiting spot.

    Every marker is driven from the SAME objects the behavior acts on \u2014 the
    plan's own waypoints, the machine's own destination, the spot the rollout
    recorded stopping at \u2014 so the discs cannot show a route the duck did not
    plan or a goal it did not resolve.
    """
    from guide_states import MONITOR_STATES

    for index in range(ROUTE_DISCS):
        point = route_points[index] if index < len(route_points) else None
        place(model, data, f"route_{index}", point, 0.005)

    waypoints = plan.waypoints if plan is not None else []
    for index in range(WAYPOINT_DISCS):
        point = waypoints[index] if index < len(waypoints) else None
        place(model, data, f"wp_{index}", point, 0.007)

    goal = destination.stand if destination is not None else None
    place(model, data, "goal_marker", goal, 0.008)
    place(model, data, "wait_marker",
          waiting_spot if state in MONITOR_STATES else None, 0.008)
    place(model, data, "arrive_marker",
          goal if state in ("INDICATE", "DONE") else None, 0.010)

    trail = [r["duck_xy"] for r in records[::-TRAIL_STRIDE]][:TRAIL_DISCS]
    for index in range(TRAIL_DISCS):
        place(model, data, f"trail_{index}",
              trail[index] if index < len(trail) else None, 0.004)
