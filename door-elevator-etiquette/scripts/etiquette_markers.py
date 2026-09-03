#!/usr/bin/env python3
"""Posing the world-space marker discs and the sliding door leaves.

Split out of ``rollout_etiquette`` so the tick loop stays about ORDER.  Only the
markers are presentation; the LEAVES are not.  Their poses come from the same
:class:`lobby_doors.DoorState` the gate measures, so the picture and the measured
clear gap cannot disagree - a leaf drawn open while the metrics called the door
shut would be a video that lied.

Every body here is mocap with ``contype="0" conaffinity="0"``, so none of them
can touch the robot, and a marker with nothing to show is parked at ``z = -3``
below the floor rather than deleted: MuJoCo has no way to remove a body from a
compiled model, and a disc under the floor is invisible from every camera.
"""

from __future__ import annotations

import numpy as np

# How many of each marker the scene carries.  These MUST match
# ``tools/build_scene.py``; ``test_the_scene_carries_every_marker`` pins them.
ROUTE_DISCS = 30
WAYPOINT_DISCS = 10
TRAIL_DISCS = 16
# Every ``trail_i`` disc samples the duck's own recent path this far apart, in
# control ticks.  At 50 Hz and the measured 0.129 m/s cruise that is 0.047 m
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


def pose_leaves(model, data, doors: dict) -> None:
    """Slide every door leaf to the position its MEASURED open fraction implies.

    Not presentation: this is the same ``DoorState`` the acceptance gate reads,
    so a door the metrics call 40% open is drawn 40% open.  The leaves sit at
    each aperture's own plane and centre, offset along the aperture axis by
    :meth:`lobby_doors.DoorState.leaf_offsets`.
    """
    for name, door in doors.items():
        south, north = door.leaf_offsets()
        for side, offset in (("s", south), ("n", north)):
            body = model.body(f"leaf_{name}_{side}")
            mocap = int(model.body_mocapid[body.id])
            half_h = 0.5 * float(model.geom_size[
                model.geom(f"leaf_{name}_{side}_panel").id][2] * 2.0)
            data.mocap_pos[mocap] = (door.plane_x,
                                     door.center_y + offset,
                                     half_h)


def pose_markers(model, data, *, route_points, waypoints, hold_xy, wait_xy,
                 records) -> None:
    """Show the duck's own route, its holding points and its recent trail.

    Every marker is driven from the SAME objects the behavior acts on - the
    route it walks, the holding points its states are gated on - so the discs
    cannot show a path the duck did not take.
    """
    for index in range(ROUTE_DISCS):
        point = route_points[index] if index < len(route_points) else None
        place(model, data, f"route_{index}", point, 0.005)

    for index in range(WAYPOINT_DISCS):
        point = waypoints[index] if index < len(waypoints) else None
        place(model, data, f"wp_{index}", point, 0.007)

    place(model, data, "hold_marker", hold_xy, 0.008)
    place(model, data, "wait_marker", wait_xy, 0.008)

    trail = [r["duck_xy"] for r in records[::-TRAIL_STRIDE]][:TRAIL_DISCS]
    for index in range(TRAIL_DISCS):
        place(model, data, f"trail_{index}",
              trail[index] if index < len(trail) else None, 0.004)


def route_polyline(route, count: int = ROUTE_DISCS) -> list[np.ndarray]:
    """The whole filleted route, sampled, for the markers and the plan view."""
    return [route.pose_at_arc(route.length * i / (count - 1))[0]
            for i in range(count)]
