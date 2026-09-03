#!/usr/bin/env python3
"""World markers: the duck's trail, its commanded heading, and what it watches.

Real MuJoCo geometry rather than HUD annotation, because a viewer must be able
to see WHERE the duck was told to go and where it has been without trusting a
number in a corner.  The heading ray is the one that matters: it is drawn from
the duck along the heading the current command is closing on, so a LEFT turn and
a RIGHT turn are visibly opposite in the wide shot rather than only in the HUD.
"""

from __future__ import annotations

import math

import numpy as np

# Every Nth control tick contributes a trail disc, so the trail spans a useful
# stretch of history with a bounded number of markers.
TRAIL_STRIDE = 14
TRAIL_DISCS = 18
HEADING_DISCS = 8
# How far the commanded-heading ray extends from the duck.
HEADING_RAY_M = 0.62


def heading_points(duck_xy, heading_rad: float) -> list[np.ndarray]:
    """Discs along the heading the current command is closing on."""
    duck = np.asarray(duck_xy, dtype=np.float64)[:2]
    direction = np.array([math.cos(heading_rad), math.sin(heading_rad)])
    return [duck + direction * (HEADING_RAY_M * (i + 1) / HEADING_DISCS)
            for i in range(HEADING_DISCS)]


def _park(model, data, name: str) -> None:
    body = model.body(name)
    mocap = int(model.body_mocapid[body.id])
    data.mocap_pos[mocap] = (0.0, 0.0, -3.0)


def _place(model, data, name: str, xy, z: float = 0.004) -> None:
    body = model.body(name)
    mocap = int(model.body_mocapid[body.id])
    data.mocap_pos[mocap] = (float(xy[0]), float(xy[1]), z)


def pose_markers(model, data, *, trail_points=None, heading=None,
                 target_xy=None, focus_xy=None) -> None:
    """Write every marker's mocap pose for this tick.

    Anything not supplied is parked below the floor, which is how a marker
    stops existing rather than lingering where it last was - a stale standoff
    disc left on the floor would be a claim about a plan that is no longer
    being followed.
    """
    points = list(trail_points or [])[:TRAIL_DISCS]
    for index in range(TRAIL_DISCS):
        if index < len(points):
            _place(model, data, f"trail_{index}", points[index])
        else:
            _park(model, data, f"trail_{index}")

    rays = list(heading or [])[:HEADING_DISCS]
    for index in range(HEADING_DISCS):
        if index < len(rays):
            _place(model, data, f"heading_{index}", rays[index], 0.006)
        else:
            _park(model, data, f"heading_{index}")

    if target_xy is not None:
        _place(model, data, "target_marker", target_xy, 0.007)
    else:
        _park(model, data, "target_marker")

    if focus_xy is not None:
        _place(model, data, "focus_marker", focus_xy, 0.008)
    else:
        _park(model, data, "focus_marker")
