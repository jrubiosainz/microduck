#!/usr/bin/env python3
"""The person being led: she walks the duck's OWN accumulated path, behind it.

THIS IS THE MODULE THAT MAKES "THE DUCK LED HER" FALSIFIABLE
--------------------------------------------------------------
A follower who walked a route of her own would prove nothing.  She would arrive
at the destination whether or not the duck ever moved, and every "she followed
me" claim in the metrics would be a coincidence of two independent scripts.

So her path is not hers.  The duck lays down a breadcrumb trail as it walks, in
world coordinates, and she advances along THAT trail by arc length.  Three
consequences follow structurally rather than by assertion:

* she can only go where the duck has already been, so the route she walks is the
  route the duck chose;
* she is always strictly behind the duck along the path, because her arc length
  is clamped below the duck's own by :data:`MIN_TRAIL_GAP_M` — the guide cannot
  accidentally end up following the follower;
* if the duck stops, her arc ceiling stops growing, so she closes the gap and
  catches up.  "Waiting worked" is then a measurement, not a hope.

The trail is seeded with the segment from where she stands to where the duck
starts, so arc zero is her own feet and she never has to be teleported onto a
path that begins somewhere else.

WHAT IS SCRIPTED, STATED PLAINLY
---------------------------------
Her SPEED is scripted: :data:`STALLS` names the windows in which she deliberately
slows to a crawl or stops, which is what creates the episodes the duck has to
detect.  That is the scenario, and it is declared here in one place.

What is NOT scripted is anything the duck knows.  The duck never reads this
module.  It measures her range with the same probe it measures everybody else
with, and her visibility through the real head camera, and it decides to wait or
resume from those two numbers alone.  A stall window and a WAIT episode are
therefore different objects, and the acceptance gate compares them rather than
conflating them.

Her stall speeds are ramped with a smootherstep at both edges rather than
switched, because a person who goes from 0.16 m/s to 0 in one control tick is a
teleport with extra steps, and the duck's lag detector would be grading an
artifact of the script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# -- her walking speeds -----------------------------------------------------
# Comfortable following pace.  Slightly above the duck's 0.116 m/s cruise so
# that, absent a stall, she holds station rather than drifting back.
FOLLOW_SPEED_MPS = 0.132
# What she does once she has fallen a long way behind and starts hurrying.
CATCHUP_SPEED_MPS = 0.196
# Arc gap beyond which she hurries.
CATCHUP_GAP_M = 1.35
# She never comes nearer than this along the trail, so the duck is always the
# one in front.  It is an arc-length clamp, not a repulsion force: she simply
# runs out of path she is allowed to occupy.
MIN_TRAIL_GAP_M = 0.62

# -- why she does not walk in the duck's footprints -------------------------
# She follows the trail OFFSET TO ONE SIDE by this much, measured normal to the
# path.  This is not a convenience: it is the difference between a scenario the
# robot can perform and one it cannot.
#
# MEASURED: with a zero offset she sits at a relative bearing of 173-180 deg —
# exactly astern, because the trail she walks IS the duck's own heading history.
# The head's MEASURED yaw range is +/-170 deg, so at those bearings the camera
# CANNOT see her however well the tracker aims, and turn-in-place is MEASURED at
# 1.6 deg/s so the body cannot help.  Two drafts tried to solve it on the robot
# side, by walking a bounded arc to square up; both worked and both cost 11 s per
# episode, spent the wait budget on turning, and still grazed the visibility
# gate.
#
# The offset fixes it at the source, and it is also what people actually do:
# somebody following a guide walks a little to one side, because that is how you
# see past them.  At the 0.62 m minimum trail gap a 0.30 m offset puts her at
# about 154 deg — comfortably inside the head's reach at every range in the run,
# with no manoeuvre required of the robot at all.
#
# The side is fixed rather than alternating: a follower who swapped sides would
# cross the duck's own path, which is neither natural nor safe.
FOLLOW_OFFSET_M = 0.30
# Arc length the trail tangent is averaged over before the offset is applied.
# About two of the duck's strides; see :meth:`Trail.pose_at` for why the raw
# per-segment tangent cannot be used.
TANGENT_SMOOTH_M = 0.25

# -- the scripted episodes --------------------------------------------------
# Each stall is (start_s, end_s, speed_factor, ramp_s, label).  A factor of 0.0
# is a full stop.  The ramps are what keep her velocity continuous.
#
# THE TWO EPISODES ARE DIFFERENT ON PURPOSE.  The first is a pure DISTANCE
# event in open floor with the sightline intact, so the duck must detect it from
# measured range alone.  The second happens while the route is rounding the
# partition, so she is both far behind AND behind a full-height body: the duck
# loses sight of her, and the detector has to fire on VISIBILITY rather than on
# range.  A guide that only watched one of the two would pass one episode and
# fail the other.
STALLS: tuple[tuple[float, float, float, float, str], ...] = (
    (17.0, 27.0, 0.00, 1.20, "stops to look at the departures board"),
    (44.0, 55.0, 0.10, 1.20, "slows to a crawl while rounding the partition"),
)


def _smootherstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def stall_factor(t: float) -> tuple[float, str]:
    """Her speed multiplier at ``t``, and the label of the active stall.

    Ramped at both edges, so her velocity is continuous and the duck's lag
    detector is measuring a person slowing down rather than a discontinuity.
    """
    factor, label = 1.0, ""
    for start, end, level, ramp, name in STALLS:
        if t <= start - ramp or t >= end + ramp:
            continue
        if t < start:
            blend = _smootherstep((t - (start - ramp)) / ramp)
        elif t > end:
            blend = 1.0 - _smootherstep((t - end) / ramp)
        else:
            blend = 1.0
        candidate = 1.0 + (level - 1.0) * blend
        if candidate < factor:
            factor, label = candidate, name
    return factor, label


class Trail:
    """The duck's accumulated world path, indexed by arc length.

    Append-only.  Interpolation is linear between consecutive samples, which at
    50 Hz and a 0.13 m/s walk means samples 2.6 mm apart: far finer than any
    quantity measured against it.
    """

    def __init__(self, seed: list[np.ndarray]):
        if len(seed) < 2:
            raise ValueError("the trail needs at least a seed segment")
        self.points: list[np.ndarray] = [
            np.asarray(p, dtype=np.float64).copy() for p in seed]
        self.arcs: list[float] = [0.0]
        for index in range(1, len(self.points)):
            self.arcs.append(self.arcs[-1] + float(np.linalg.norm(
                self.points[index] - self.points[index - 1])))

    def append(self, xy) -> None:
        point = np.asarray(xy, dtype=np.float64)
        step = float(np.linalg.norm(point - self.points[-1]))
        # Samples closer than a tenth of a millimetre carry no information and
        # would make the arc index dense with duplicates while the duck stands
        # still.  Dropping them keeps the search below O(n) per tick in practice.
        if step < 1e-4:
            return
        self.points.append(point.copy())
        self.arcs.append(self.arcs[-1] + step)

    @property
    def total_m(self) -> float:
        return self.arcs[-1]

    def pose_at(self, s: float, smooth_m: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray]:
        """Position and unit tangent at arc length ``s`` along the trail.

        ``smooth_m`` averages the tangent over a window of that arc length
        instead of taking it from the single segment ``s`` falls in.  THIS IS
        NOT COSMETIC.  The trail is sampled at 50 Hz from a walking robot, so
        consecutive samples sit about 2.6 mm apart and their individual
        directions swing wildly with the gait — the per-segment tangent flips
        through tens of degrees between ticks.  An offset applied along that raw
        normal makes the follower jitter sideways by twice the offset every few
        ticks: MEASURED, it turned a 10 m walk into 163 m of accumulated path.
        Averaging over about two of the duck's strides removes the gait entirely
        while still following every real bend.
        """
        s = float(np.clip(s, 0.0, self.arcs[-1]))
        position = self._point_at(s)
        if smooth_m > 0.0:
            lo = self._point_at(max(0.0, s - 0.5 * smooth_m))
            hi = self._point_at(min(self.arcs[-1], s + 0.5 * smooth_m))
            delta = hi - lo
        else:
            index = int(np.clip(
                int(np.searchsorted(self.arcs, s, side="right")) - 1,
                0, len(self.points) - 2))
            delta = self.points[index + 1] - self.points[index]
        norm = float(np.linalg.norm(delta))
        tangent = (delta / norm) if norm > 1e-12 else np.array([1.0, 0.0])
        return position, tangent

    def _point_at(self, s: float) -> np.ndarray:
        """Position only, without the tangent.  Used by the smoothing window."""
        s = float(np.clip(s, 0.0, self.arcs[-1]))
        index = int(np.searchsorted(self.arcs, s, side="right")) - 1
        index = int(np.clip(index, 0, len(self.points) - 2))
        span = self.arcs[index + 1] - self.arcs[index]
        u = 0.0 if span <= 1e-12 else (s - self.arcs[index]) / span
        a, b = self.points[index], self.points[index + 1]
        return a + (b - a) * u


@dataclass
class Follower:
    """Her state, and the arc-length rule that keeps her behind the duck."""

    start_xy: tuple[float, float]
    duck_start_xy: tuple[float, float]
    arc_s: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    stall_label: str = ""
    stall_factor_now: float = 1.0
    # Total path length she has actually walked, so "she really moved" is
    # measured rather than inferred from her arc index.
    walked_m: float = 0.0
    _pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    trail: Trail | None = None

    def __post_init__(self) -> None:
        self.trail = Trail([np.asarray(self.start_xy, dtype=np.float64),
                            np.asarray(self.duck_start_xy, dtype=np.float64)])
        # Her starting pose is the offset trail at arc zero, so she does not
        # jump sideways on the first tick she moves.
        position, tangent = self.trail.pose_at(0.0, TANGENT_SMOOTH_M)
        normal = np.array([-tangent[1], tangent[0]])
        self._pos = position + normal * FOLLOW_OFFSET_M
        delta = (np.asarray(self.duck_start_xy, dtype=np.float64) - self._pos)
        self.yaw = math.atan2(float(delta[1]), float(delta[0]))

    @property
    def pos(self) -> np.ndarray:
        return self._pos

    @property
    def ceiling_m(self) -> float:
        """The furthest arc length she is allowed to occupy right now."""
        return max(0.0, self.trail.total_m - MIN_TRAIL_GAP_M)

    @property
    def trail_gap_m(self) -> float:
        """How far behind the duck she is ALONG THE PATH, in metres."""
        return self.trail.total_m - self.arc_s

    def push_duck(self, duck_xy) -> None:
        """Extend the trail with the duck's measured position for this tick."""
        self.trail.append(duck_xy)

    def update(self, t: float, dt: float, *, moving: bool) -> None:
        """Advance her one control tick.

        ``moving`` is the scenario's own gate on whether she has set off at all:
        she stands where she is until the duck has acknowledged and started, so
        the opening seconds are a person waiting to be led rather than a person
        already walking.
        """
        factor, label = stall_factor(t)
        self.stall_factor_now = factor
        self.stall_label = label
        if not moving:
            self.speed = 0.0
            return

        base = (CATCHUP_SPEED_MPS if self.trail_gap_m > CATCHUP_GAP_M
                else FOLLOW_SPEED_MPS)
        desired = base * factor
        proposed = min(self.arc_s + desired * dt, self.ceiling_m)
        proposed = max(proposed, self.arc_s)
        previous = self._pos.copy()
        self.arc_s = proposed
        position, tangent = self.trail.pose_at(self.arc_s, TANGENT_SMOOTH_M)
        # Offset normal to the path, so she walks a little to one side of the
        # duck's own line rather than in its footprints.  See FOLLOW_OFFSET_M:
        # this is what keeps her inside the head camera's MEASURED +/-170 deg
        # reach instead of exactly astern of it.
        normal = np.array([-tangent[1], tangent[0]])
        self._pos = position + normal * FOLLOW_OFFSET_M
        step = float(np.linalg.norm(self._pos - previous))
        self.walked_m += step
        self.speed = step / dt if dt > 0.0 else 0.0
        if step > 1e-6:
            self.yaw = math.atan2(float(tangent[1]), float(tangent[0]))

    def as_record(self) -> dict:
        return {
            "arc_s_m": round(self.arc_s, 4),
            "trail_total_m": round(self.trail.total_m, 4),
            "trail_gap_m": round(self.trail_gap_m, 4),
            "walked_m": round(self.walked_m, 4),
            "speed_mps": round(self.speed, 4),
            "stall_factor": round(self.stall_factor_now, 4),
            "stall_label": self.stall_label,
        }


def stall_windows() -> list[dict]:
    """The declared stall windows, for the metrics to compare episodes against."""
    return [{"start_s": start, "end_s": end, "speed_factor": level,
             "ramp_s": ramp, "label": name}
            for start, end, level, ramp, name in STALLS]
