#!/usr/bin/env python3
"""Six road users on continuous, non-stop scripted trajectories.

Pure analytic functions of ``t``: position, velocity and heading are all
closed-form, so the conflict predictor can be unit-tested without MuJoCo.

Design constraints this file exists to satisfy
----------------------------------------------
* **Nobody teleports, nobody freezes, nobody yields.**  Every vehicle drives its
  lane at a constant speed for the whole rollout and **wraps** around a long
  loop when it runs off the end.  There is no "stop so the duck can cross"
  branch anywhere, and no vehicle ever reacts to the duck.  The safe gap the
  duck eventually takes is a gap the *schedule* produced, not one the scenario
  handed it.
* **Two travel directions.**  The near lane runs −Y, the far lane runs +Y.
* **The wrap is invisible.**  ``LOOP_HALF_Y`` is far outside both the camera's
  useful range and the predictor's horizon, so a vehicle that wraps has left
  the scene entirely before it reappears at the other end.  The wrap is
  therefore not a teleport *in the scenario*; it is how a finite road models an
  endless stream.  ``max_visible_jump`` measures that claim instead of asserting
  it, and a test pins it.
* **The gap is engineered, and then MEASURED.**  Vehicle phases are chosen so
  that the first several seconds are genuinely unsafe and a single wide gap
  opens later.  Whether that gap is actually safe is decided by
  ``conflict.py`` from geometry — this module only supplies kinematics.

Every vehicle is a MOCAP body with ``contype="0" conaffinity="0"``: kinematic
scenery that cannot touch or push the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from street import LANE_DIRECTION, NEAR_LANE_X, FAR_LANE_X

VEHICLE_NAMES: tuple[str, ...] = (
    "hatch", "sedan", "scooter", "van", "bike", "taxi", "courier",
)

# Half-length of the loop each vehicle drives before wrapping.
#
# The binding constraint is NOT visibility, it is the PREDICTOR'S HORIZON.  A
# wrap moves a vehicle from one end of the loop to the other in a single tick;
# if the far end is close enough that the vehicle re-reaches the crossing
# within `conflict.PREDICT_HORIZON_S`, that jump appears to the predictor as a
# road user materialising inside its reach, and a gap decision can flip between
# two consecutive ticks for no physical reason.
#
# MEASURED: the first draft wrapped at 26 m, which looked comfortable — 25 m
# from the duck, far outside the PiP.  But the fastest vehicle (sedan,
# 1.55 m/s) covers that in 16.8 s, INSIDE the 22 s horizon.  The wrap was
# therefore inside the decision, not merely off-screen.
#
# The requirement is `LOOP_HALF_Y > max_speed * PREDICT_HORIZON_S` = 34.1 m.
# 42 m gives the fastest vehicle 27.1 s to return, clearing the horizon with
# margin, and a test pins the relation rather than the number.
LOOP_HALF_Y: float = 42.0

# Vehicle bounding half-extents (metres), mirroring the geoms the scene
# generator emits.  The conflict predictor inflates by these, so a "clear gap"
# accounts for the whole body rather than a centre point.
HALF_LENGTH: dict[str, float] = {
    "hatch": 0.229, "sedan": 0.229, "van": 0.229, "taxi": 0.229,
    "courier": 0.229, "scooter": 0.208, "bike": 0.208,
}
HALF_WIDTH: dict[str, float] = {
    "hatch": 0.100, "sedan": 0.100, "van": 0.100, "taxi": 0.100,
    "courier": 0.100, "scooter": 0.088, "bike": 0.088,
}
KIND: dict[str, str] = {
    "hatch": "car", "sedan": "car", "van": "car", "taxi": "car",
    "courier": "car", "scooter": "scooter", "bike": "bicycle",
}


@dataclass(frozen=True)
class Vehicle:
    """One road user driving a lane at constant speed, wrapping at the ends.

    ``y_at_zero`` is where the vehicle is at ``t = 0``; the trajectory is then
    ``y(t) = wrap(y_at_zero + direction * speed * t)``.  Specifying the schedule
    this way — rather than as "arrive at the crossing at time T" — keeps the
    kinematics honest: nothing in this file knows where the duck is or when it
    would like to cross.
    """

    name: str
    lane: str
    speed: float
    y_at_zero: float
    lateral_offset: float = 0.0

    @property
    def direction(self) -> float:
        return LANE_DIRECTION[self.lane]

    @property
    def x(self) -> float:
        base = NEAR_LANE_X if self.lane == "near" else FAR_LANE_X
        return base + self.lateral_offset

    @property
    def half_length(self) -> float:
        return HALF_LENGTH[self.name]

    @property
    def half_width(self) -> float:
        return HALF_WIDTH[self.name]

    @property
    def yaw(self) -> float:
        """Heading: +Y traffic faces +90 deg, −Y traffic faces −90 deg."""
        return math.pi / 2.0 if self.direction > 0 else -math.pi / 2.0

    def y_at(self, t: float) -> float:
        raw = self.y_at_zero + self.direction * self.speed * t
        return (raw + LOOP_HALF_Y) % (2.0 * LOOP_HALF_Y) - LOOP_HALF_Y

    def pos_at(self, t: float) -> np.ndarray:
        return np.array([self.x, self.y_at(t)], dtype=np.float64)

    def vel_at(self, t: float) -> np.ndarray:
        return np.array([0.0, self.direction * self.speed], dtype=np.float64)


@dataclass(frozen=True)
class VehicleState:
    name: str
    lane: str
    pos: np.ndarray
    vel: np.ndarray
    yaw: float
    speed: float
    half_length: float
    half_width: float
    kind: str


# THE TRAFFIC SCHEDULE.
#
# Speeds are in metres per second at the scene's scale.  The duck walks at a
# MEASURED 0.25-0.36 m/s (tools/sweep_commands.py), so vehicles at 0.9-1.7 m/s
# are three to six times its speed — the same ratio a real street presents to a
# child-sized robot, which is what makes the gap decision non-trivial.
#
# The schedule is built so the crossing is genuinely blocked at first:
#
#   * three vehicles reach the crossing inside the first ~11 s, alternating
#     lanes, so no gap wide enough for a 6-7 s crossing exists early;
#   * between those bursts the road is momentarily EMPTY, which is the trap the
#     predictor exists to catch: at t~9 s and again at t~18 s nothing is on the
#     crossing at all, yet a crossing started then would meet the next arrival
#     mid-road.  A naive "is the road clear right now" rule crosses into both;
#   * ``taxi`` is slower than ``bike`` in the same lane, so it can never catch
#     and drive through it (``min_vehicle_separation`` measures this).
#
# After ``van`` clears at ~t=24.5 s the next arrival is ``sedan`` on its second
# loop at ~t=40 s, which is the first genuinely wide gap.
#
# None of that is asserted here.  ``conflict.py`` decides safety from geometry
# and the metrics gate reports which gaps were rejected and why.
SCHEDULE: tuple[Vehicle, ...] = (
    # near lane, travelling −Y: arrives from the duck's LEFT
    Vehicle("hatch", lane="near", speed=1.25, y_at_zero=7.6),
    Vehicle("bike", lane="near", speed=0.92, y_at_zero=13.4),
    Vehicle("taxi", lane="near", speed=0.86, y_at_zero=19.9),
    # far lane, travelling +Y: arrives from the duck's RIGHT
    Vehicle("sedan", lane="far", speed=1.55, y_at_zero=-11.2),
    Vehicle("scooter", lane="far", speed=1.08, y_at_zero=-17.6),
    # The van must never CATCH the scooter ahead of it in the same lane, or the
    # scene shows one vehicle driving through another.  On the original 26 m
    # loop a faster van simply never got the chance; on the 42 m loop it laps
    # far enough to close the gap, and ``min_vehicle_separation`` measured a
    # -0.433 m overlap.  Keeping the van slower than the vehicle in front of it
    # is the structural fix, and the measured minimum same-lane gap is +6.98 m.
    Vehicle("van", lane="far", speed=1.06, y_at_zero=-25.02),
    # THE GAP IS BOUNDED AT BOTH ENDS.  Without a late arrival the road is
    # simply empty after t~24.5 s, and "the duck waited for a safe gap" would
    # be indistinguishable from "the duck waited for the traffic to end".
    # ``courier`` reaches the crossing at ~t=40 s, so the gap the duck takes is
    # a genuine window between two vehicles, and the safety margin at
    # commitment is measured against a road user that is still coming.
    Vehicle("courier", lane="near", speed=1.18, y_at_zero=48.0),
)

VEHICLE_BY_NAME: dict[str, Vehicle] = {v.name: v for v in SCHEDULE}


def traffic_at(t: float) -> dict[str, VehicleState]:
    """Every road user's state at time ``t``."""
    states: dict[str, VehicleState] = {}
    for vehicle in SCHEDULE:
        pos = vehicle.pos_at(t)
        vel = vehicle.vel_at(t)
        states[vehicle.name] = VehicleState(
            name=vehicle.name,
            lane=vehicle.lane,
            pos=pos,
            vel=vel,
            yaw=vehicle.yaw,
            speed=vehicle.speed,
            half_length=vehicle.half_length,
            half_width=vehicle.half_width,
            kind=KIND[vehicle.name],
        )
    return states


def crossing_arrivals(seconds: float, dt: float = 0.02,
                      window: float = 0.60) -> list[dict]:
    """Every time a vehicle's body overlaps the crossing corridor.

    Reported rather than assumed, so the README can state when the crossing was
    actually blocked instead of quoting the phases the schedule was written
    with.  ``window`` is the half-span in y that counts as "on the crossing".
    """
    events: list[dict] = []
    open_pass: dict[str, dict] = {}
    steps = int(seconds / dt) + 1
    for index in range(steps):
        t = index * dt
        for name, state in traffic_at(t).items():
            y = float(state.pos[1])
            occupying = abs(y) <= window + state.half_length
            if occupying and name not in open_pass:
                open_pass[name] = {"vehicle": name, "lane": state.lane,
                                   "enter_s": t, "speed_mps": state.speed}
            elif not occupying and name in open_pass:
                entry = open_pass.pop(name)
                entry["exit_s"] = t
                entry["duration_s"] = t - entry["enter_s"]
                events.append(entry)
    for entry in open_pass.values():
        entry["exit_s"] = seconds
        entry["duration_s"] = seconds - entry["enter_s"]
        events.append(entry)
    events.sort(key=lambda e: e["enter_s"])
    return events


def max_visible_jump(seconds: float, dt: float = 0.02,
                     visible_half_y: float = 12.0) -> tuple[float, str, float]:
    """Largest single-tick position jump while a vehicle is anywhere near.

    The loop wrap is a discontinuity by construction, so the honest claim is
    not "there is no jump" but "no jump happens where it could be seen or where
    it could affect a prediction".  This measures exactly that: the biggest
    step taken while the vehicle is inside ``visible_half_y`` of the crossing.
    A test pins it below one tick of ordinary travel.
    """
    worst = (0.0, "", 0.0)
    steps = int(seconds / dt)
    previous = {name: state.pos[1]
                for name, state in traffic_at(0.0).items()}
    for index in range(1, steps + 1):
        t = index * dt
        for name, state in traffic_at(t).items():
            y = float(state.pos[1])
            jump = abs(y - previous[name])
            if abs(y) <= visible_half_y and jump > worst[0]:
                worst = (jump, name, t)
            previous[name] = y
    return worst


def min_vehicle_separation(seconds: float, dt: float = 0.05) -> tuple[float, str, str]:
    """Closest any two vehicles come to each other over the rollout.

    Two vehicles in the SAME lane must never overlap, or the scene shows one
    car driving through another.  Cross-lane pairs are allowed to be close —
    that is just traffic — so the reported worst case is filtered to same-lane
    pairs, which is the one that would look broken.
    """
    worst = (float("inf"), "", "")
    steps = int(seconds / dt) + 1
    for index in range(steps):
        traffic = traffic_at(index * dt)
        names = list(traffic)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                a, b = traffic[first], traffic[second]
                if a.lane != b.lane:
                    continue
                gap = abs(float(a.pos[1] - b.pos[1])) - (
                    a.half_length + b.half_length)
                if gap < worst[0]:
                    worst = (gap, first, second)
    return worst


def pose_traffic(model, data, traffic: dict[str, VehicleState], t: float) -> None:
    """Write every vehicle's mocap pose.

    Kinematic scenery written straight into mocap slots: none of it is
    simulated and none of it can touch the robot.  Two-wheelers get a small
    lean that scales with speed, purely so they read as moving vehicles rather
    than sliding props.
    """
    for name, state in traffic.items():
        body = model.body(f"vehicle_{name}")
        mocap = int(model.body_mocapid[body.id])
        data.mocap_pos[mocap, 0] = float(state.pos[0])
        data.mocap_pos[mocap, 1] = float(state.pos[1])
        data.mocap_pos[mocap, 2] = 0.0
        yaw = state.yaw
        if state.kind in ("scooter", "bicycle"):
            lean = math.radians(4.0) * math.sin(2.0 * math.pi * 0.9 * t)
            half_yaw, half_roll = yaw / 2.0, lean / 2.0
            data.mocap_quat[mocap] = np.array([
                math.cos(half_yaw) * math.cos(half_roll),
                math.cos(half_yaw) * math.sin(half_roll),
                math.sin(half_yaw) * math.sin(half_roll),
                math.sin(half_yaw) * math.cos(half_roll),
            ])
        else:
            data.mocap_quat[mocap] = np.array(
                [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
