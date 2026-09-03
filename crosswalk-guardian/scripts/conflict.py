#!/usr/bin/env python3
"""Geometric crossing-occupancy prediction and the gap decision.

Pure geometry and pure Python/numpy: no MuJoCo, no ONNX, no rendering.  This is
the decision layer and it is fully unit-tested in ``tests/``.

What "the duck predicts a conflict" actually means
--------------------------------------------------
The predictor is an honest **simulator semantic/geometric proxy for vehicle
perception**.  Each road user's lane, position, velocity and body size come
from the simulator; there is no detector, no tracker, no radar and no
time-to-collision estimated from pixels anywhere in this behavior.  What IS
real is the camera geometry used by the LOOK phases — a scan phase only counts
when the corresponding road sector is genuinely inside the frustum of the exact
camera the PiP renders from.

The prediction itself
---------------------
Both the duck and every vehicle are reduced to **1-D occupancy intervals** on
the quantity that actually matters, and the decision is whether those intervals
are disjoint by a margin.

* The **duck** crosses along +x at a measured speed, so for each lane it
  occupies that lane over a closed interval of time
  ``[t_enter(lane), t_exit(lane)]`` relative to the moment it steps off.  Both
  ends are inflated by the duck's planar radius, so the interval covers its
  whole footprint rather than its centre point.
* A **vehicle** travels along y at constant speed, so it occupies the crossing
  corridor over ``[t_in, t_out]``, inflated by its own half-length and by the
  duck's radius plus a lateral buffer.

A vehicle is a conflict for this crossing if its corridor interval comes within
``SAFETY_MARGIN_S`` of the duck's interval **for the lane that vehicle is in**.
A vehicle in the far lane is irrelevant while the duck is still in the near
lane, and the predictor says so directly instead of treating the road as one
undivided block — which is both more realistic and strictly harder to satisfy
at the moment of commitment, because the duck must be clear of the far lane
*later*, when it is most exposed.

The decision is taken over the **entire estimated crossing interval**, using a
crossing-duration estimate inflated by a measured pessimism factor.  A gap that
is safe only if the duck walks at exactly its nominal speed is rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from street import (
    CROSSWALK_HALF_SPAN,
    DUCK_PLANAR_RADIUS,
    LANE_SPANS,
    ROAD_EXIT_X,
)

# --- measured locomotion constants -------------------------------------------
# MEASURED on scene_crosswalk_guardian.xml with the stock alpha_walking policy
# at action scale 0.9 and the real imu_ang_vel sensor
# (tools/sweep_commands.py, 5 s and 6 s rollouts).
#
# FORWARD GAIT ONSET IS A CLIFF, NOT A RAMP:
#   vx=0.16 -> 0.008 m in 6 s   (no gait)
#   vx=0.20 -> 0.010 m in 6 s   (no gait)
#   vx=0.24 -> 0.516 m in 6 s   (walking)
# A command below onset produces NO motion, so a crossing can never be slowed
# by shrinking vx: it is walked at speed or not at all.
VX_MIN_EFFECTIVE: float = 0.24
# Approach to the kerb.  MEASURED vx=0.52 -> 1.516 m in 6 s (0.253 m/s).
VX_APPROACH: float = 0.52
# The crossing command.  MEASURED vx=0.58 -> 1.794 m in 6 s (0.299 m/s), min
# trunk z 0.111, final 0.119.  Faster commands are stable too (vx=0.75 reaches
# 0.455 m/s) but they cost heading stability: the measured yaw drift grows from
# -2.9 deg/s at 0.58 to -6.4 deg/s at 0.75, and a crossing that curves is a
# crossing that ends up in a lane.
VX_CROSS: float = 0.58

# THE POLICY DRIFTS RIGHT AT SPEED, and the drift is not small.  MEASURED yaw
# over 5-6 s of straight-line walking:
#   vx=0.46 -> -18.0 deg      vx=0.52 -> -8.8 deg     vx=0.58 -> -17.7 deg
#   vx=0.60 -> -23.4 deg      vx=0.65 -> -13.7 deg    vx=0.70 -> -24.5 deg
# An open-loop crossing therefore curves several tens of degrees off the zebra
# and can leave the crossing entirely.  The crossing MUST be flown closed-loop
# on heading, which is why the controller below exists at all.
#
# Yaw authority while walking, MEASURED at vx=0.58 over 5 s:
#   wz=+0.10 -> +12.0 deg   (+2.4 deg/s)    wz=-0.10 -> -43.3 deg  (-8.7 deg/s)
#   wz=+0.18 -> +36.0 deg   (+7.2 deg/s)    wz=-0.18 -> -55.0 deg (-11.0 deg/s)
# The two signs are NOT mirror images — the right side is 3-5x stronger for the
# same magnitude — so they get independent gains and independent dead zones.
KP_YAW_LEFT: float = 1.35
KP_YAW_RIGHT: float = 0.42
WZ_MAX_LEFT: float = 0.30
WZ_MAX_RIGHT: float = 0.16
# Below these the measured yaw response is lost in the gait's own variation, so
# emitting them burns control ticks without steering.
WZ_MIN_LEFT: float = 0.06
WZ_MIN_RIGHT: float = 0.03

# --- crossing-duration estimate ----------------------------------------------
# The nominal crossing speed, MEASURED by tools/measure_crossing.py, which runs
# the EXACT crossing primitive with the EXACT heading controller from the real
# kerb stop and times the true lane occupancies.  Three starting offsets gave
# 0.2977, 0.2977 and 0.2980 m/s; the slowest is used.
# Replace this constant only with a number that tool printed.
CROSS_SPEED_MPS: float = 0.2977

# THE PREDICTION MUST BRACKET THE MEASUREMENT ON BOTH SIDES, and a single
# pessimism factor cannot do that.
#
# MEASURED (tools/measure_crossing.py, closed loop from x=-0.95):
#     near lane occupied [1.28, 3.58] s     far lane occupied [3.00, 5.38] s
#
# The first draft simply divided the nominal speed by 1.30 and used the result
# for both ends.  That produced near [1.38, 4.63]: the predicted EXIT was
# safely late, but the predicted ENTRY was 0.10 s LATER than the duck actually
# entered.  A vehicle clearing the lane in that window would have been judged
# clear while the duck was already in it.  Stretching a schedule uniformly is
# not conservatism — it moves both ends the same way, and only one of them is
# the safe way.
#
# The interval is therefore widened OUTWARD from both ends:
#
#   * the ENTRY is computed at a speed FASTER than measured, so the duck is
#     predicted to be in the lane before it can possibly get there;
#   * the EXIT is computed at a speed SLOWER than measured AND delayed by the
#     measured gait-onset dead time, so the duck is predicted to still be in
#     the lane well after it has actually left.
CROSS_SPEED_FAST_FACTOR: float = 1.15
CROSS_DURATION_PESSIMISM: float = 1.30
# MEASURED gait-onset dead time: from the kerb stop the duck's footprint
# reached the near lane at t=1.28 s, where nominal speed alone predicts
# 0.2697 m / 0.2977 m/s = 0.91 s.  The 0.37 s difference is the gait crossing
# its onset threshold.  0.55 s carries that measurement with margin, and it is
# applied ONLY to the exit end, where being late is the safe direction.
ONSET_DEAD_TIME_S: float = 0.55

# --- safety margins ----------------------------------------------------------
# Required clear time between the duck's occupancy of a lane and any vehicle's
# occupancy of the crossing corridor in that same lane.
#
# JUSTIFICATION, from measured quantities rather than taste:
#   * gait onset latency, MEASURED: the duck needs about 0.9 s from the command
#     to leaving standstill, which is dead time inside every crossing;
#   * crossing-duration uncertainty at the 1.30 pessimism factor above is
#     already folded into the interval, so this margin covers what is left:
#     the vehicle's own speed being read one tick stale (0.02 s x 1.55 m/s =
#     0.031 m) and the fact that a vehicle is not obliged to hold its speed;
#   * a pedestrian who clears a lane 1.5 s before a car arrives has visibly
#     cleared it; 0.5 s would be a near miss even if the arithmetic passed.
SAFETY_MARGIN_S: float = 1.50
# Lateral buffer added to the duck's radius when deciding how much of the road
# corridor counts as "the duck is in the way".
CORRIDOR_BUFFER_M: float = 0.12
# How far ahead the predictor looks.  Beyond this a vehicle is not yet a
# participant in this crossing decision; the horizon is comfortably longer than
# the pessimistic crossing duration so nothing relevant is truncated.
PREDICT_HORIZON_S: float = 22.0

# --- timings -----------------------------------------------------------------
STOP_HOLD_S: float = 1.30
LOOK_LEFT_S: float = 2.40
LOOK_RIGHT_S: float = 2.40
LOOK_LEFT_AGAIN_S: float = 1.90
# The gap decision may not commit the instant the machine enters WAIT_FOR_GAP:
# a gap must be predicted safe continuously for this long, so a single tick of
# favourable arithmetic cannot launch the duck into the road.
GAP_CONFIRM_S: float = 0.40
# Hard ceiling on waiting, so a rollout cannot hang forever.  Exceeding it is
# recorded as a timeout and fails the gate rather than being papered over.
WAIT_MAX_S: float = 40.0
SAFE_HOLD_S: float = 3.0
# A crossing that has not finished by now has gone wrong; recorded as a
# timeout, which the gate treats as a failure.
CROSSING_MAX_S: float = 16.0

STATES = (
    "APPROACH_CURB", "STOP", "LOOK_LEFT", "LOOK_RIGHT", "LOOK_LEFT_AGAIN",
    "WAIT_FOR_GAP", "CROSSING", "SAFE",
)
LOOK_STATES = ("LOOK_LEFT", "LOOK_RIGHT", "LOOK_LEFT_AGAIN")
# Every state in which the locomotion command must be EXACTLY zero.
STATIONARY_STATES = (
    "STOP", "LOOK_LEFT", "LOOK_RIGHT", "LOOK_LEFT_AGAIN", "WAIT_FOR_GAP",
    "SAFE",
)
MOVING_STATES = ("APPROACH_CURB", "CROSSING")


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Interval:
    """A closed time interval, or the empty interval when ``start > end``."""

    start: float
    end: float

    @property
    def empty(self) -> bool:
        return self.start > self.end

    def gap_to(self, other: "Interval") -> float:
        """Clear time between two intervals; negative when they overlap.

        The magnitude of a negative value is the overlap duration, which is
        what makes a rejected gap reportable as "the van would have been on the
        crossing for 1.2 s while the duck was in its lane".
        """
        if self.empty or other.empty:
            return float("inf")
        if self.end < other.start:
            return other.start - self.end
        if other.end < self.start:
            return self.start - other.end
        return -(min(self.end, other.end) - max(self.start, other.start))


def duck_lane_intervals(
    start_x: float,
    speed: float = CROSS_SPEED_MPS,
    *,
    pessimism: float = CROSS_DURATION_PESSIMISM,
    fast_factor: float = CROSS_SPEED_FAST_FACTOR,
    dead_time: float = ONSET_DEAD_TIME_S,
    radius: float = DUCK_PLANAR_RADIUS,
) -> dict[str, Interval]:
    """When the duck's footprint occupies each lane, relative to stepping off.

    Both ends are inflated by the duck's planar radius, and the interval is
    widened OUTWARD in time: the entry is computed at a speed faster than
    measured, the exit at a speed slower than measured plus the measured
    gait-onset dead time.  The result strictly contains the measured
    occupancy, which is the property the gap decision depends on and which a
    single uniform pessimism factor does not provide.
    """
    if speed <= 0.0:
        raise ValueError("crossing speed must be positive")
    fast = speed * fast_factor          # earliest plausible entry
    slow = speed / pessimism            # latest plausible exit
    intervals: dict[str, Interval] = {}
    for lane, (low, high) in LANE_SPANS.items():
        enter_x = low - radius     # leading edge reaches the lane
        exit_x = high + radius     # trailing edge leaves the lane
        intervals[lane] = Interval(
            start=max(0.0, (enter_x - start_x) / fast),
            end=max(0.0, dead_time + (exit_x - start_x) / slow),
        )
    return intervals


def crossing_duration(
    start_x: float,
    goal_x: float = ROAD_EXIT_X,
    speed: float = CROSS_SPEED_MPS,
    *,
    pessimism: float = CROSS_DURATION_PESSIMISM,
    dead_time: float = ONSET_DEAD_TIME_S,
) -> float:
    """Pessimistic estimate of how long the whole crossing takes."""
    return max(0.0, dead_time + (goal_x - start_x) / (speed / pessimism))


def vehicle_corridor_interval(
    y: float,
    velocity_y: float,
    half_length: float,
    *,
    corridor_half: float = DUCK_PLANAR_RADIUS + CORRIDOR_BUFFER_M,
    horizon: float = PREDICT_HORIZON_S,
) -> Interval:
    """When a vehicle's body occupies the pedestrian corridor.

    The corridor is the strip of road the duck's own body sweeps through, so a
    vehicle "arrives" when its front bumper reaches the far edge of that strip
    and "leaves" when its rear bumper clears the near edge.  A vehicle that is
    stationary or driving away returns the empty interval.
    """
    reach = corridor_half + half_length
    if abs(velocity_y) < 1e-9:
        # Not moving: it is either parked on the crossing forever or nowhere
        # near it.  Both are answered exactly.
        return Interval(0.0, horizon) if abs(y) <= reach else Interval(1.0, 0.0)
    t_a = (-reach - y) / velocity_y
    t_b = (+reach - y) / velocity_y
    start, end = (t_a, t_b) if t_a <= t_b else (t_b, t_a)
    if end < 0.0:
        return Interval(1.0, 0.0)          # already gone
    if start > horizon:
        return Interval(1.0, 0.0)          # beyond the horizon
    return Interval(max(start, 0.0), min(end, horizon))


@dataclass(frozen=True)
class Conflict:
    """One vehicle's predicted interaction with a candidate crossing."""

    name: str
    lane: str
    margin_s: float
    vehicle_window: Interval
    duck_window: Interval
    range_m: float
    speed_mps: float
    approaching: bool

    @property
    def blocks(self) -> bool:
        return self.margin_s < SAFETY_MARGIN_S

    @property
    def overlaps(self) -> bool:
        return self.margin_s < 0.0


@dataclass(frozen=True)
class GapDecision:
    """Everything behind one accept/reject of a candidate crossing."""

    safe: bool
    worst_margin_s: float
    limiting_vehicle: str | None
    conflicts: tuple[Conflict, ...]
    crossing_duration_s: float
    start_x: float

    @property
    def blocking(self) -> tuple[Conflict, ...]:
        return tuple(c for c in self.conflicts if c.blocks)

    def as_record(self) -> dict:
        return {
            "safe": self.safe,
            "worst_margin_s": self.worst_margin_s,
            "limiting_vehicle": self.limiting_vehicle,
            "crossing_duration_s": self.crossing_duration_s,
            "start_x": self.start_x,
            "blocking": [
                {
                    "vehicle": c.name,
                    "lane": c.lane,
                    "margin_s": c.margin_s,
                    "vehicle_window_s": [c.vehicle_window.start,
                                         c.vehicle_window.end],
                    "duck_window_s": [c.duck_window.start, c.duck_window.end],
                    "range_m": c.range_m,
                    "speed_mps": c.speed_mps,
                }
                for c in self.blocking
            ],
        }


def evaluate_gap(
    traffic: dict,
    duck_xy,
    *,
    start_x: float | None = None,
    speed: float = CROSS_SPEED_MPS,
    margin: float = SAFETY_MARGIN_S,
    horizon: float = PREDICT_HORIZON_S,
) -> GapDecision:
    """Decide whether a crossing started NOW would stay clear of every vehicle.

    ``traffic`` maps a name to anything with ``lane``, ``pos``, ``vel`` and
    ``half_length`` — the ``VehicleState`` records from ``traffic.py``, or plain
    stubs in the tests.

    The decision is per-lane and covers the ENTIRE estimated crossing interval,
    not the instant of commitment.  It is deliberately conservative in three
    independent ways: the duck's schedule is stretched by
    ``CROSS_DURATION_PESSIMISM``, both bodies are inflated to their real
    extents, and every vehicle must clear ``margin`` seconds rather than merely
    not collide.
    """
    duck_xy = np.asarray(duck_xy, dtype=np.float64)
    x0 = float(duck_xy[0]) if start_x is None else float(start_x)
    lane_windows = duck_lane_intervals(x0, speed)
    conflicts: list[Conflict] = []
    worst = float("inf")
    limiting: str | None = None

    for name, vehicle in traffic.items():
        lane = vehicle.lane
        duck_window = lane_windows[lane]
        y = float(vehicle.pos[1])
        vy = float(vehicle.vel[1])
        window = vehicle_corridor_interval(
            y, vy, float(vehicle.half_length), horizon=horizon)
        margin_s = duck_window.gap_to(window)
        # A vehicle driving AWAY from the crossing has an empty window and an
        # infinite margin, which is correct: it cannot come back inside the
        # horizon.  Reported so the HUD can show it as cleared rather than
        # silently dropping it.
        approaching = not window.empty
        conflict = Conflict(
            name=name, lane=lane, margin_s=margin_s,
            vehicle_window=window, duck_window=duck_window,
            range_m=float(np.linalg.norm(vehicle.pos - duck_xy)),
            speed_mps=abs(vy), approaching=approaching,
        )
        conflicts.append(conflict)
        if margin_s < worst:
            worst = margin_s
            limiting = name

    conflicts.sort(key=lambda c: c.margin_s)
    return GapDecision(
        safe=worst >= margin,
        worst_margin_s=worst,
        limiting_vehicle=limiting,
        conflicts=tuple(conflicts),
        crossing_duration_s=crossing_duration(x0, speed=speed),
        start_x=x0,
    )
