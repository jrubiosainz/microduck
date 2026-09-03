#!/usr/bin/env python3
"""What the duck aims at, given its state.  Pure geometry, no physics.

Split out of ``rollout_guide`` so that the tick loop stays about ORDER and this
stays about WHERE THE TARGET IS.  Nothing here steps physics, advances the
machine or touches MuJoCo, which is what lets every branch be unit-tested on
hand-built inputs.

There are only four kinds of target in this behavior, and the list is short on
purpose:

* ``route_pursuit`` \u2014 a point a fixed arc length ahead of the duck's own
  projection onto the planned route.  Driving at the next WAYPOINT instead would
  make the duck cut every corner: it would aim at the vertex, reach the fillet,
  and turn late.
* ``standing_point`` \u2014 the destination's own standing point, on the last
  approach.
* ``look_back`` / ``waiting`` \u2014 no position at all, only a bearing for the head.
  The duck is stopped in both, so a walking target would be a contradiction.
* nothing \u2014 in every state whose command is a literal zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Aim:
    """Where to walk, where to look, and how far is left."""

    target_xy: np.ndarray | None = None
    kind: str = ""
    look_at_yaw: float | None = None
    cross_track_m: float = 0.0
    remaining_m: float = 1e9


def select(state: str, *, duck_xy, tracker, destination, follower_yaw: float,
           arrive_radius_m: float) -> Aim:
    """The aim for this tick.

    ``tracker`` may be ``None`` before a route exists, and ``destination`` may be
    ``None`` before a request has been resolved; both are normal early in the
    run rather than error cases, so they produce an empty aim and the controller
    emits its zero.
    """
    if state in ("LEAD", "RESUME") and tracker is not None:
        tracker.project(duck_xy)
        return Aim(target_xy=tracker.pursuit_point(),
                   kind="route_pursuit",
                   cross_track_m=tracker.cross_track_m,
                   remaining_m=tracker.remaining_m)

    if state == "CHECK_FOLLOWER":
        return Aim(kind="look_back", look_at_yaw=follower_yaw)

    if state == "WAIT_FOR_PERSON":
        return Aim(kind="waiting", look_at_yaw=follower_yaw)

    if state == "ARRIVE" and destination is not None:
        gap = float(np.linalg.norm(
            destination.stand - np.asarray(duck_xy, dtype=np.float64)))
        if gap > 0.10:
            return Aim(target_xy=destination.stand, kind="standing_point",
                       remaining_m=gap)
        return Aim(kind="arrived", remaining_m=0.0)

    return Aim()


def facing_error_deg(duck_xy, duck_yaw: float, destination) -> float | None:
    """How far off the DESTINATION FIXTURE the duck is pointing, in degrees.

    Measured against the fixture rather than against the standing point: a guide
    that arrived and then faced the patch of floor it is standing on has not
    indicated anything.
    """
    if destination is None:
        return None
    delta = destination.position - np.asarray(duck_xy, dtype=np.float64)
    bearing = math.atan2(float(delta[1]), float(delta[0]))
    error = math.atan2(math.sin(bearing - duck_yaw),
                       math.cos(bearing - duck_yaw))
    return abs(math.degrees(error))


def bearing_to(duck_xy, target_xy) -> float:
    """World bearing from the duck to a point, in radians."""
    delta = (np.asarray(target_xy, dtype=np.float64)
             - np.asarray(duck_xy, dtype=np.float64))
    return math.atan2(float(delta[1]), float(delta[0]))


def reached_standing_point(duck_xy, destination, arrive_radius_m: float) -> bool:
    """Is the duck inside the arrival radius of the destination it was asked for?"""
    if destination is None:
        return False
    return float(np.linalg.norm(
        destination.stand - np.asarray(duck_xy, dtype=np.float64))) \
        <= arrive_radius_m
