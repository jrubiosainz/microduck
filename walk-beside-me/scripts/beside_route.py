#!/usr/bin/env python3
"""Constant-speed arc-length routes with real, rounded bends.

WHY NOT THE SMOOTHERSTEP WALKER THE SIBLING BEHAVIORS USE
----------------------------------------------------------
Every earlier behavior in this lab moved its actors by interpolating between
timed waypoints with a smootherstep.  That is fine when the scenario only needs
somebody to *be* somewhere at a given moment, but it has two properties this
behavior cannot tolerate:

* **the actor stops at every waypoint** — smootherstep has zero derivative at
  both ends of each leg, so a "continuous walk" is really a sequence of
  accelerations from rest.  A robot asked to hold station beside a companion
  whose speed keeps returning to zero is being graded on the companion's
  stuttering, not on its own formation keeping;
* **the heading changes discontinuously at a waypoint** — the turn happens in a
  single control tick, which no walking person does and which the duck, whose
  MEASURED yaw authority is a few degrees per second, could never follow.

So the guardian here walks a genuinely continuous route: a polyline whose
interior corners are replaced by circular arcs of radius :data:`CORNER_RADIUS`,
parameterized by ARC LENGTH and traversed at constant speed.  Heading is the
path tangent, which is continuous everywhere, and curvature is bounded by
``1 / CORNER_RADIUS`` — which is what makes "the duck followed the bends" a
measurable claim rather than a hope.

The construction is pure geometry with no MuJoCo, no time-stepping and no
state, so every property below is unit-tested directly:

* total length equals the sum of the segment and arc lengths;
* position is continuous across every segment/arc boundary;
* the tangent is continuous across every boundary (that is what the fillet buys);
* speed is constant except during an explicit start delay or terminal hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Radius of the circular fillet inserted at every interior corner.  Chosen
# against the duck's MEASURED yaw authority rather than for looks: at the
# guardian's 0.175 m/s cruise a 0.90 m radius turns the formation at
# 0.175 / 0.90 = 0.194 rad/s = 11.1 deg/s, which the duck can follow on its
# weaker (left) yaw sign.  A tighter corner would make the bend gate a test of
# the policy's turning ceiling rather than of the beside controller.
CORNER_RADIUS = 0.90


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("cannot take the direction of a zero-length segment")
    return vector / norm


@dataclass(frozen=True)
class _Piece:
    """One traversable piece of the path: a straight run or a circular arc."""

    kind: str                 # "line" | "arc"
    length: float
    start_s: float
    # line
    a: np.ndarray | None = None
    b: np.ndarray | None = None
    # arc
    center: np.ndarray | None = None
    radius: float = 0.0
    theta0: float = 0.0
    sweep: float = 0.0        # signed; positive is counter-clockwise

    def at(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and unit tangent at arc length ``s`` into this piece."""
        u = 0.0 if self.length <= 0.0 else min(max(s / self.length, 0.0), 1.0)
        if self.kind == "line":
            direction = _unit(self.b - self.a)
            return self.a + (self.b - self.a) * u, direction
        angle = self.theta0 + self.sweep * u
        radial = np.array([math.cos(angle), math.sin(angle)])
        tangent = np.array([-math.sin(angle), math.cos(angle)])
        if self.sweep < 0.0:
            tangent = -tangent
        return self.center + self.radius * radial, tangent


def _build(corners: list[np.ndarray], radius: float) -> list[_Piece]:
    """Straight runs joined by tangent circular arcs at every interior corner.

    Each corner is cut back by ``t = radius * tan(theta / 2)`` along both of its
    legs and the gap bridged by an arc, which is exactly the fillet that makes
    the tangent continuous.  A corner whose legs are too short for its own
    cutback, or which is nearly straight, is left as a plain vertex; both cases
    are reported by :meth:`Route.corner_report` rather than silently accepted.
    """
    pieces: list[_Piece] = []
    cursor = corners[0].copy()
    s = 0.0
    for index in range(1, len(corners) - 1):
        previous, corner, following = corners[index - 1], corners[index], corners[index + 1]
        incoming = _unit(corner - previous)
        outgoing = _unit(following - corner)
        cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        dot = float(np.clip(incoming @ outgoing, -1.0, 1.0))
        turn = math.atan2(cross, dot)
        if abs(turn) < math.radians(1.0):
            continue
        cutback = radius * math.tan(abs(turn) * 0.5)
        room = min(float(np.linalg.norm(corner - cursor)),
                   float(np.linalg.norm(following - corner)) * 0.5)
        if cutback >= room:
            continue
        entry = corner - incoming * cutback
        exit_point = corner + outgoing * cutback
        straight = float(np.linalg.norm(entry - cursor))
        if straight > 1e-9:
            pieces.append(_Piece("line", straight, s, a=cursor.copy(),
                                 b=entry.copy()))
            s += straight
        sign = 1.0 if turn > 0.0 else -1.0
        normal = np.array([-incoming[1], incoming[0]]) * sign
        center = entry + normal * radius
        theta0 = math.atan2(*(entry - center)[::-1])
        sweep = sign * abs(turn)
        arc_length = radius * abs(turn)
        pieces.append(_Piece("arc", arc_length, s, center=center.copy(),
                             radius=radius, theta0=theta0, sweep=sweep))
        s += arc_length
        cursor = exit_point
    final = float(np.linalg.norm(corners[-1] - cursor))
    if final > 1e-9:
        pieces.append(_Piece("line", final, s, a=cursor.copy(),
                             b=corners[-1].copy()))
    if not pieces:
        raise ValueError("route collapsed to zero pieces")
    return pieces


@dataclass
class Route:
    """A constant-speed walk along a filleted polyline.

    ``start_s`` delays departure so an actor can be standing when the rollout
    begins; once the far end is reached the actor holds there, which is a real
    standstill rather than a wrap-around.
    """

    name: str
    corners: tuple[tuple[float, float], ...]
    speed: float
    start_t: float = 0.0
    radius: float = CORNER_RADIUS

    def __post_init__(self) -> None:
        points = [np.asarray(c, dtype=np.float64) for c in self.corners]
        if len(points) < 2:
            raise ValueError(f"route {self.name!r} needs at least two corners")
        self.pieces = _build(points, self.radius)
        self.length = float(sum(p.length for p in self.pieces))

    # -- parameterization ------------------------------------------------
    def arc_at(self, t: float) -> float:
        return float(np.clip(self.speed * (t - self.start_t), 0.0, self.length))

    def finish_t(self) -> float:
        return self.start_t + self.length / self.speed

    def _piece_at(self, s: float) -> tuple[_Piece, float]:
        for piece in self.pieces:
            if s <= piece.start_s + piece.length:
                return piece, s - piece.start_s
        last = self.pieces[-1]
        return last, last.length

    def pose_at_arc(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        piece, local = self._piece_at(float(np.clip(s, 0.0, self.length)))
        return piece.at(local)

    # -- public ----------------------------------------------------------
    def pos_at(self, t: float) -> np.ndarray:
        return self.pose_at_arc(self.arc_at(t))[0]

    def tangent_at(self, t: float) -> np.ndarray:
        return self.pose_at_arc(self.arc_at(t))[1]

    def yaw_at(self, t: float) -> float:
        tangent = self.tangent_at(t)
        return math.atan2(float(tangent[1]), float(tangent[0]))

    def speed_at(self, t: float) -> float:
        s = self.speed * (t - self.start_t)
        return self.speed if 0.0 < s < self.length else 0.0

    def moving(self, t: float) -> bool:
        return self.speed_at(t) > 0.0

    def corner_report(self) -> list[dict]:
        """Every bend actually built into the path, with its signed turn.

        Reported rather than assumed so a test can require that the route
        contains at least one LEFT and one RIGHT bend, and so a corner that was
        silently dropped for want of room shows up as missing.
        """
        report: list[dict] = []
        for piece in self.pieces:
            if piece.kind != "arc":
                continue
            report.append({
                "turn_deg": round(math.degrees(piece.sweep), 3),
                "hand": "left" if piece.sweep > 0.0 else "right",
                "radius_m": round(piece.radius, 4),
                "arc_len_m": round(piece.length, 4),
                "start_s_m": round(piece.start_s, 4),
                "start_t_s": round(self.start_t + piece.start_s / self.speed, 3),
                "end_t_s": round(
                    self.start_t + (piece.start_s + piece.length) / self.speed, 3),
            })
        return report

    def as_record(self) -> dict:
        return {
            "name": self.name,
            "corners": [[round(float(x), 4), round(float(y), 4)]
                        for x, y in self.corners],
            "length_m": round(self.length, 4),
            "speed_mps": self.speed,
            "start_t_s": self.start_t,
            "finish_t_s": round(self.finish_t(), 3),
            "corner_radius_m": self.radius,
            "bends": self.corner_report(),
        }
