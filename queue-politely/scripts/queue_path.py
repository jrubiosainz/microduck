#!/usr/bin/env python3
"""The queue as an EXPLICIT CURVED WORLD-SPACE PATH, and projection onto it.

This module is the whole reason the behavior is not trivial.  A queue is not a
set of people sorted by distance from the counter, and it is not a set of
people sorted by a coordinate.  It is an ordered occupancy of a *path*, and the
only honest way to read the order is to PROJECT each person onto that path and
sort by arc length.

WHY A MAX-COORDINATE HEURISTIC IS WRONG HERE, ARITHMETICALLY
------------------------------------------------------------
The path in this scene is a hairpin: a leg out from the counter, a 180 deg
fold, and a return leg.  That is an ordinary rope-barrier queue, and it breaks
both naive heuristics at once.  With the six adults at their initial stations:

    slot  arc s   world (x, y)          |p| from counter   -x
      0   0.00    ( 0.000,  0.000)            0.000       0.000
      1   0.55    (-0.549, -0.018)            0.549       0.549
      2   1.10    (-0.960, -0.355)            1.024       0.960
      3   1.65    (-0.960, -0.887)            1.307       0.960
      4   2.55    (-0.198, -1.240)            1.256       0.198   <- TRUE TAIL

MEASURED, by ``naive_orders`` on exactly those five stations:

    by_range        alvarez bianchi chandra ERIKSSON DUBOIS   -> tail dubois
    by_max_minus_x  alvarez ERIKSSON bianchi dubois CHANDRA   -> tail chandra
    by_arc_length   alvarez bianchi chandra dubois eriksson   -> tail eriksson

The two naive readings do not merely fail; they fail DIFFERENTLY, naming two
different wrong people as the back of the queue:

* **Farthest from the counter** picks ``dubois``, who is 4th, because the
  return leg folds back TOWARD the counter - the tail is physically nearer the
  counter than the person two places ahead of it.
* **Largest -x** picks ``chandra``, who is 3rd, and additionally ranks the true
  tail SECOND, because the fold puts the tail back at almost the same x as the
  head.
* The duck's own join station (arc 3.13) sits at |p| = 1.298, which is NEARER
  the counter than both ``dubois`` and ``eriksson`` - so a distance-sorted
  reading would rank the newly joined duck FOURTH of six while it is genuinely
  last.

Only arc length along the path gets it right, and the arc length is what this
module computes.  ``naive_orders`` reproduces both wrong answers on purpose so
the metrics can state the difference rather than assert it.

Only arc length along the path gets it right, and the arc length is what this
module computes.  ``naive_orders`` reproduces both wrong answers on purpose so
the metrics can state the difference rather than assert it.

Two people standing beside the queue rather than in it are handled by the same
projection: their perpendicular distance to the path exceeds ``QUEUE_BAND_M``,
so they are not queue members at all.  A max-coordinate reading has no way to
express that at all - the bystander is the farthest thing from the counter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Spacing between consecutive queue stations, centre to centre.  Adults have a
# measured planar half-extent of 0.104 m and the duck about 0.13 m, so 0.55 m
# leaves a real surface gap of roughly 0.32 m: a queue, not a scrum.
SLOT_SPACING_M = 0.55
# Perpendicular distance within which a body counts as standing IN the queue.
# Wider than any queueing person's own sloppiness, narrower than the 0.45 m
# barrier lane, so somebody beside the rope is outside it.
QUEUE_BAND_M = 0.30
# Where the served person stands, and which way they face.
SERVICE_XY = (0.0, 0.0)


def _unit(heading_deg: float) -> np.ndarray:
    rad = math.radians(heading_deg)
    return np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)


def _rot90(v: np.ndarray) -> np.ndarray:
    """Rotate a planar vector +90 deg (counter-clockwise)."""
    return np.array([-v[1], v[0]], dtype=np.float64)


@dataclass(frozen=True)
class _Line:
    start: np.ndarray
    direction: np.ndarray     # unit, points AWAY from the counter
    length: float

    def point(self, u: float) -> np.ndarray:
        return self.start + self.direction * u

    def away_heading(self, u: float) -> float:
        return math.atan2(float(self.direction[1]), float(self.direction[0]))

    def project(self, xy: np.ndarray) -> tuple[float, float, float]:
        raw = float((xy - self.start) @ self.direction)
        u = min(max(raw, 0.0), self.length)
        closest = self.point(u)
        normal = _rot90(self.direction)
        cross = float((xy - closest) @ normal)
        return u, cross, float(np.linalg.norm(xy - closest))


@dataclass(frozen=True)
class _Arc:
    center: np.ndarray
    radius: float
    theta0: float             # position angle at u = 0, radians
    sign: float               # +1: away-heading increases with u
    length: float

    def _theta(self, u: float) -> float:
        return self.theta0 + self.sign * (u / self.radius)

    def point(self, u: float) -> np.ndarray:
        theta = self._theta(u)
        return self.center + self.radius * np.array(
            [math.cos(theta), math.sin(theta)], dtype=np.float64)

    def away_heading(self, u: float) -> float:
        theta = self._theta(u)
        tangent = self.sign * np.array(
            [-math.sin(theta), math.cos(theta)], dtype=np.float64)
        return math.atan2(float(tangent[1]), float(tangent[0]))

    def project(self, xy: np.ndarray) -> tuple[float, float, float]:
        offset = xy - self.center
        radial = float(np.linalg.norm(offset))
        angle = math.atan2(float(offset[1]), float(offset[0]))
        # Signed sweep from theta0 to the query angle, in the arc's own sense.
        delta = self.sign * (angle - self.theta0)
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        if delta < 0.0:
            delta += 2.0 * math.pi
        u = min(max(delta * self.radius, 0.0), self.length)
        closest = self.point(u)
        # Inward normal for sign=+1 (the centre is +90 deg from the tangent).
        cross = (self.radius - radial) * self.sign
        if 0.0 < u < self.length:
            return u, float(cross), abs(float(radial - self.radius))
        return u, float(cross), float(np.linalg.norm(xy - closest))


class QueuePath:
    """A polyline-and-arc path in world coordinates, parameterised by arc length.

    ``s`` grows AWAY from the counter, so the head of the queue is at ``s = 0``
    and a larger ``s`` means further back in line.  The duck travels toward
    DECREASING ``s``.
    """

    def __init__(self, start_xy, start_heading_deg: float, spec):
        point = np.array(start_xy, dtype=np.float64)
        direction = _unit(start_heading_deg)
        self.segments: list[_Line | _Arc] = []
        self.starts: list[float] = []
        total = 0.0
        for entry in spec:
            self.starts.append(total)
            if entry[0] == "line":
                length = float(entry[1])
                segment = _Line(point.copy(), direction.copy(), length)
                point = segment.point(length)
            elif entry[0] == "arc":
                radius, sweep_deg = float(entry[1]), float(entry[2])
                sign = 1.0 if sweep_deg >= 0.0 else -1.0
                center = point + radius * sign * _rot90(direction)
                offset = point - center
                theta0 = math.atan2(float(offset[1]), float(offset[0]))
                length = abs(math.radians(sweep_deg)) * radius
                segment = _Arc(center, radius, theta0, sign, length)
                point = segment.point(length)
                heading = segment.away_heading(length)
                direction = _unit(math.degrees(heading))
            else:  # pragma: no cover - spec is a module constant
                raise ValueError(f"unknown queue path segment {entry[0]!r}")
            self.segments.append(segment)
            total += segment.length
        self.length = total

    # -- forward evaluation ---------------------------------------------
    def _locate(self, s: float) -> tuple[int, float]:
        s = min(max(float(s), 0.0), self.length)
        for index in range(len(self.segments) - 1, -1, -1):
            if s >= self.starts[index] - 1e-12:
                return index, s - self.starts[index]
        return 0, 0.0

    def point_at(self, s: float) -> np.ndarray:
        index, u = self._locate(s)
        return self.segments[index].point(min(u, self.segments[index].length))

    def away_heading_at(self, s: float) -> float:
        """Heading, in radians, pointing from the counter toward the tail."""
        index, u = self._locate(s)
        return self.segments[index].away_heading(
            min(u, self.segments[index].length))

    def travel_heading_at(self, s: float) -> float:
        """Heading a queueing body faces: toward the counter, i.e. decreasing s."""
        return (self.away_heading_at(s) + math.pi + math.pi) % (
            2.0 * math.pi) - math.pi

    def polyline(self, step: float = 0.04) -> np.ndarray:
        count = max(2, int(round(self.length / step)) + 1)
        return np.asarray(
            [self.point_at(i * self.length / (count - 1)) for i in range(count)])

    # -- inverse ---------------------------------------------------------
    def project(self, xy) -> tuple[float, float, float]:
        """Arc length, SIGNED cross-track and true distance of the nearest point.

        Cross-track is positive to the LEFT of the away-direction.  Distance is
        the honest planar distance to the path, which exceeds ``|cross|`` when
        the nearest point is an endpoint - that matters for a body standing
        beyond the tail rather than beside the queue.
        """
        query = np.asarray(xy, dtype=np.float64)[:2]
        best = (0.0, 0.0, float("inf"))
        for index, segment in enumerate(self.segments):
            u, cross, distance = segment.project(query)
            if distance < best[2]:
                best = (self.starts[index] + u, cross, distance)
        return best

    def arc_of(self, xy) -> float:
        return self.project(xy)[0]

    def cross_track_of(self, xy) -> float:
        return self.project(xy)[1]

    def on_path(self, xy, band: float = QUEUE_BAND_M) -> bool:
        return self.project(xy)[2] <= band


# THE PATH.  A short leg out from the counter, a 180 deg fold at radius 0.62 m,
# and a long return leg: an ordinary rope-barrier queue, and the shape that
# makes both naive orderings wrong in two DIFFERENT ways (module docstring).
#
# The fold radius is NOT arbitrary.  The duck advances one station per service,
# and while it is on the fold it must hold a turn of radius R.  MEASURED on
# this scene (tools/measure_advance.py, 3 s windows):
#
#     vx=0.38 wz=-0.55  ->  R = 0.455 m
#     vx=0.38 wz=-0.42  ->  R = 0.715 m
#     vx=0.34 wz=-0.42  ->  R = 0.630 m   <- the fold sits inside this
#     vx=0.28 wz=-0.34  ->  R = 0.613 m
#
# So 0.62 m is a radius the stock policy demonstrably holds at the advance
# speed, with authority in hand on both sides.  The scene geometry is set by
# the measured turn authority rather than the other way round, and the turn is
# taken in the NEGATIVE wz sense, which the sweep measured as by far the
# stronger direction (+0.18 yields R=4.74 m; -0.18 yields R=1.65 m).
FOLD_RADIUS_M = 0.62
PATH = QueuePath(
    SERVICE_XY,
    180.0,
    (
        ("line", 0.40),                    # out from the counter, along -x
        ("arc", FOLD_RADIUS_M, 180.0),     # the fold
        ("line", 2.30),                    # return leg, along +x
    ),
)

# Station of the n-th place in line, head first.
def station_arc(index: int) -> float:
    return index * SLOT_SPACING_M


def station_xy(index: int) -> np.ndarray:
    return PATH.point_at(station_arc(index))


def naive_orders(positions: dict[str, tuple[float, float]]) -> dict[str, list[str]]:
    """The two wrong answers, computed on purpose.

    ``by_range``      sorted by Euclidean distance from the service point.
    ``by_max_minus_x`` sorted by -x, the "furthest back along the aisle" reading.

    Reported by the metrics so the projection's advantage is a measured
    difference rather than a claim.
    """
    service = np.asarray(SERVICE_XY, dtype=np.float64)
    names = list(positions)
    by_range = sorted(
        names,
        key=lambda n: float(np.linalg.norm(
            np.asarray(positions[n], dtype=np.float64) - service)))
    by_minus_x = sorted(names, key=lambda n: -float(positions[n][0]))
    return {"by_range": by_range, "by_max_minus_x": by_minus_x}
