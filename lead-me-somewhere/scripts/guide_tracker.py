#!/usr/bin/env python3
"""Where the duck is on its own route: projection, pursuit point, cross-track.

Pure geometry against a :class:`guide_route.Route`.  No MuJoCo, no state machine,
no physics — so every property below is unit-tested directly.

THE CURSOR IS MONOTONIC, AND THAT IS A SCAR
---------------------------------------------
``lost-child-find-person`` learned this the hard way: a stateless "nearest point
on the path" selector re-acquires an already-passed part of the route as soon as
the duck's distance to it grows again, which on a route that doubles back
produces an endless loop.  The arc cursor here only ever moves FORWARD, and the
projection searches a bounded window ahead of it, so a route that comes back
near itself cannot pull the duck backwards.

WHY REMAINING DISTANCE IS ARC LENGTH AND NOT EUCLIDEAN
--------------------------------------------------------
"How far is left to walk" is what decides when the duck eases in and when it has
arrived.  On a route that bends back on itself the straight-line distance to the
end can be small while several metres of path remain; using it would make the
duck slow down in the middle of the hall.  Remaining distance is therefore
``route.length - arc_s``, and the arrival test is a separate check against the
standing point itself.
"""

from __future__ import annotations

import math

import numpy as np

from guide_route import Route
from guide_states import PROJECTION_WINDOW_M, PURSUIT_LOOKAHEAD_M


class RouteTracker:
    """A monotonic cursor along one planned route, with a pursuit point."""

    def __init__(self, route: Route, *,
                 lookahead_m: float = PURSUIT_LOOKAHEAD_M,
                 window_m: float = PROJECTION_WINDOW_M,
                 samples: int = 48):
        self.route = route
        self.lookahead_m = float(lookahead_m)
        self.window_m = float(window_m)
        self.samples = int(samples)
        self.arc_s = 0.0
        self.cross_track_m = 0.0

    # -- projection --------------------------------------------------------
    def project(self, duck_xy) -> float:
        """Advance the MONOTONIC cursor to the duck's projection on the route.

        Searches only ``[arc_s, arc_s + window_m]``, so a route that doubles
        back cannot re-acquire the duck at an earlier arc length.  The cursor
        never decreases.
        """
        point = np.asarray(duck_xy, dtype=np.float64)
        lo = self.arc_s
        hi = min(self.arc_s + self.window_m, self.route.length)
        best_s, best_d = lo, float("inf")
        for index in range(self.samples + 1):
            s = lo + (hi - lo) * (index / self.samples)
            position, _ = self.route.pose_at_arc(s)
            distance = float(np.linalg.norm(position - point))
            if distance < best_d:
                best_d, best_s = distance, s
        self.arc_s = max(self.arc_s, best_s)
        self.cross_track_m = best_d
        return self.arc_s

    # -- the pursuit point -------------------------------------------------
    @property
    def remaining_m(self) -> float:
        """Arc length left to walk.  Never negative."""
        return max(0.0, self.route.length - self.arc_s)

    def pursuit_point(self) -> np.ndarray:
        """The point ``lookahead_m`` further along the route than the cursor.

        Clamped to the route's end, so the last stretch drives at the standing
        point itself rather than at a phantom beyond it.
        """
        return self.route.pose_at_arc(
            min(self.arc_s + self.lookahead_m, self.route.length))[0]

    def route_yaw(self) -> float:
        """The route's own heading at the cursor, for reporting and the HUD."""
        _, tangent = self.route.pose_at_arc(self.arc_s)
        return math.atan2(float(tangent[1]), float(tangent[0]))

    def polyline(self, count: int = 48) -> list[np.ndarray]:
        """The whole filleted route, sampled, for markers and the plan view."""
        return [self.route.pose_at_arc(self.route.length * i / (count - 1))[0]
                for i in range(count)]

    def as_record(self) -> dict:
        return {
            "arc_s_m": round(self.arc_s, 4),
            "route_length_m": round(self.route.length, 4),
            "remaining_m": round(self.remaining_m, 4),
            "cross_track_m": round(self.cross_track_m, 4),
            "progress": round(self.arc_s / max(self.route.length, 1e-9), 4),
        }
