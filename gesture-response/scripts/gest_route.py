#!/usr/bin/env python3
"""Constant-speed arc-length routes with real, rounded bends.

Used here for the four distracting adults, who walk the training area for the
whole session.  Their heading has to be continuous for two measurable reasons:
a cornered polyline turns its walker through a whole corner in ONE control tick,
which is a teleport of the body axis, and the arm keypoints this behavior reads
are expressed in the person's own frame - so a heading discontinuity would make
a distractor's gesture features jump for a tick and could hand the classifier a
pose that was never physically held.

WHY NOT A SMOOTHERSTEP WALKER
------------------------------
Interpolating between timed waypoints with a smootherstep gives an actor zero
derivative at both ends of every leg, so a "continuous walk" is really a
sequence of accelerations from rest, and the heading changes discontinuously at
each waypoint.  Neither is something a walking person does, and the second is
something the duck — whose MEASURED yaw authority is a few degrees per second —
could never follow.

So a route here is a polyline whose interior corners are replaced by circular
arcs of radius :data:`CORNER_RADIUS`, parameterized by ARC LENGTH and traversed
at constant speed.  Heading is the path tangent, which is continuous everywhere,
and curvature is bounded by ``1 / CORNER_RADIUS``.

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

# Radius of the circular fillet inserted at every interior corner.  Smaller than
# in the sibling behaviors because this is an open training area with short
# route legs around its edges, and ``_build`` raises rather than silently
# leaving a hard vertex when a cutback does not fit.  Nothing the DUCK walks
# uses this radius - its turns are closed-loop on measured yaw error - so this
# is purely the scripted adults' path smoothness.
CORNER_RADIUS = 0.40


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


def _build(corners: list[np.ndarray], radius: float,
           name: str = "") -> list[_Piece]:
    """Straight runs joined by tangent circular arcs at every interior corner.

    Each corner is cut back by ``t = radius * tan(theta / 2)`` along both of its
    legs and the gap bridged by an arc, which is exactly the fillet that makes
    the tangent continuous.

    A CORNER THAT CANNOT BE FILLETED IS A LOUD FAILURE, NOT A SILENT VERTEX.
    Earlier this function skipped any corner whose cutback did not fit the legs
    either side of it and left a hard vertex in its place - which turns the
    walker through the whole corner in ONE control tick.  Two scripted routes in
    this behavior hit it, and the symptom surfaced far away: a 51 deg
    single-tick heading change that ``tools/check_layout.py`` refused, with
    nothing in the route to point at.  Skipping is now an exception naming the
    corner and the radius that would fit, so the fix happens where the geometry
    is rather than where the consequence shows up.

    A corner that is very nearly straight is still skipped, because there is no
    turn to fillet.
    """
    pieces: list[_Piece] = []
    cursor = corners[0].copy()
    s = 0.0
    for index in range(1, len(corners) - 1):
        previous, corner, following = (
            corners[index - 1], corners[index], corners[index + 1])
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
            fits = room / max(math.tan(abs(turn) * 0.5), 1e-9)
            raise ValueError(
                f"route {name or '?'!r}: the {math.degrees(turn):+.1f} deg "
                f"corner {index} at ({corner[0]:.3f}, {corner[1]:.3f}) needs "
                f"{cutback:.4f} m of cutback but has only {room:.4f} m of "
                f"room.  Leaving it as a hard vertex would turn the walker "
                f"through the whole corner in one control tick.  Use a radius "
                f"below {fits:.4f} m, or move the corner.")
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
    """A constant-speed walk along a filleted polyline, with optional holds.

    ``start_t`` delays departure so an actor can be standing when the rollout
    begins; once the far end is reached the actor holds there, which is a real
    standstill rather than a wrap-around.

    ``hold_windows`` are ``(start_s, end_s)`` intervals during which the walker
    STOPS WHERE IT IS and its arc length does not advance.  This exists for one
    reason and it is a physical one: a person riding a lift stands still inside
    the car for the whole ride.  Without a hold, the guardian's arc length would
    keep growing while the car was sealed and she would walk out through a
    closed door — which is precisely the failure this behavior grades the duck
    on.  The hold makes the SCENARIO obey the same rule the robot does.

    Arc length is therefore ``speed * (elapsed - held)`` rather than
    ``speed * elapsed``, computed in closed form so it stays a pure function of
    ``t`` with no accumulated state.  A test asserts position continuity across
    both edges of every window.
    """

    name: str
    corners: tuple[tuple[float, float], ...]
    speed: float
    start_t: float = 0.0
    radius: float = CORNER_RADIUS
    hold_windows: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        points = [np.asarray(c, dtype=np.float64) for c in self.corners]
        if len(points) < 2:
            raise ValueError(f"route {self.name!r} needs at least two corners")
        for start, end in self.hold_windows:
            if end < start:
                raise ValueError(
                    f"route {self.name!r} has a hold window ending before it "
                    f"starts: ({start}, {end})")
        self.pieces = _build(points, self.radius, self.name)
        self.length = float(sum(p.length for p in self.pieces))

    # -- parameterization ------------------------------------------------
    def _held_before(self, t: float) -> float:
        """Seconds of hold already elapsed by ``t``.  Closed form, no state."""
        total = 0.0
        for start, end in self.hold_windows:
            total += max(0.0, min(t, end) - start)
        return total

    def arc_at(self, t: float) -> float:
        elapsed = (t - self.start_t) - self._held_before(t)
        return float(np.clip(self.speed * elapsed, 0.0, self.length))

    def finish_t(self) -> float:
        held = sum(max(0.0, end - start) for start, end in self.hold_windows)
        return self.start_t + held + self.length / self.speed

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
        """Ground speed at ``t``: zero before departure, inside a hold, and after
        the far end is reached."""
        for start, end in self.hold_windows:
            if start <= t < end:
                return 0.0
        s = self.speed * ((t - self.start_t) - self._held_before(t))
        return self.speed if 0.0 < s < self.length else 0.0

    def moving(self, t: float) -> bool:
        return self.speed_at(t) > 0.0

    def corner_report(self) -> list[dict]:
        """Every bend actually built into the path, with its signed turn.

        Reported rather than assumed so a test can require that the route
        contains at least three bends and both hands, and so a corner that was
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
                "end_s_m": round(piece.start_s + piece.length, 4),
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
            "hold_windows_s": [[float(a), float(b)]
                               for a, b in self.hold_windows],
            "finish_t_s": round(self.finish_t(), 3),
            "corner_radius_m": self.radius,
            "bends": self.corner_report(),
        }
