#!/usr/bin/env python3
"""Five adults milling around the duck, and the caller's raised-arm wave.

Pure analytic functions of ``t``: position, velocity and heading are all
differentiable and reproducible, so caller selection can be unit-tested without
MuJoCo.

Design constraints this file exists to satisfy:

* **Nobody freezes and nobody teleports.**  Every adult paces a slow closed
  ellipse around their own anchor for the whole rollout.  There is no "stand
  perfectly still so the duck can reach me" branch: the caller keeps drifting
  while the duck walks to them, so the approach controller has to track a
  moving goal.
* **They stay apart.**  The anchors are >= 1.5 m apart and the pacing amplitude
  is small, so the five never interpenetrate and each stays readable in the
  wide shot.  ``min_person_separation`` measures this instead of asserting it.
* **The caller is unmistakable without the HUD.**  Whoever is calling turns to
  face the duck and raises one arm overhead, waving it.  The shoulder hinge
  range in the generated scene reaches ``-195 deg`` precisely so the arm can go
  past vertical.

Identity, position and velocity all come from the simulator.  This is a
semantic proxy for "an adult calls the robot", not audio recognition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ADULT_NAMES: tuple[str, ...] = ("blue", "green", "red", "yellow", "purple")

# The three adults that actually call, in the order they call.  Blue and purple
# are never callers; blue only ever issues the interrupting call that the
# no-steal rule refuses.
CALLER_NAMES: tuple[str, ...] = ("red", "yellow", "green")

# Adult body geometry, mirroring assets/scene_come_here_recall.xml.
TORSO_RADIUS = 0.078
STRIDE_HZ = 0.95
# Wave: the raised arm oscillates about this shoulder angle.  0 rad hangs the
# arm straight down and -pi points it straight up, so -165 deg is overhead and
# tilted slightly forward, which reads clearly from the duck's low camera.
WAVE_CENTER_DEG = -165.0
WAVE_AMPLITUDE_DEG = 22.0
WAVE_HZ = 1.6


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Station:
    """One adult's slow pacing loop around a fixed anchor.

    The anchors are placed from the duck's PREDICTED trajectory rather than on
    a symmetric ring.  A ring looks tidy and plays badly: after walking to one
    caller the duck sits ``ring_radius - standoff`` from the plaza centre, so
    the next caller on the same ring is 2.0-2.4 m away no matter which one is
    chosen.  Choosing anchors so that each successive recall pulls the duck
    back across the middle keeps every approach inside 2.4 m, keeps the wide
    shot framed, and still gives three widely separated call bearings.
    """

    name: str
    anchor: tuple[float, float]
    radii: tuple[float, float]
    period: float
    rotation: float
    phase: float
    reverse: bool = False

    @property
    def omega(self) -> float:
        return (-1.0 if self.reverse else 1.0) * 2.0 * math.pi / self.period

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """World position and velocity at time ``t``."""
        rx, ry = self.radii
        angle = self.phase + self.omega * t
        local = np.array([rx * math.cos(angle), ry * math.sin(angle)])
        local_v = np.array(
            [-rx * self.omega * math.sin(angle), ry * self.omega * math.cos(angle)]
        )
        cos_r, sin_r = math.cos(self.rotation), math.sin(self.rotation)
        rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        return rot @ local + np.asarray(self.anchor, dtype=np.float64), rot @ local_v


@dataclass(frozen=True)
class AdultState:
    name: str
    pos: np.ndarray
    vel: np.ndarray
    yaw: float
    speed: float

    def clearance_to(self, point: np.ndarray) -> float:
        """Planar clearance from ``point`` to this adult's torso surface."""
        delta = np.asarray(point, dtype=np.float64) - self.pos
        return float(np.linalg.norm(delta)) - TORSO_RADIUS


# Anchors SOLVED by tools/solve_anchors.py against the duck's own predicted
# recall path, for the call order RED -> YELLOW -> GREEN.
#
# Two measured facts drive this layout:
#
# 1. **The stock policy cannot turn in place.**  Measured on this scene, a pure
#    yaw command produces almost nothing: ``wz=+/-0.85`` with ``vx=0`` yields
#    +7.8 deg and -9.5 deg over SIX seconds.  Every heading change must be flown
#    as an arc while walking, so a call bearing is paid for in both time AND
#    ground covered.
# 2. **Right turns are faster than left.**  At ``vx=0.28``: ``wz=-0.85`` gives
#    -31.0 deg/s while ``wz=+0.85`` gives only +26.8 deg/s, and the gap widens
#    at lower speed (-8.0 vs +0.7 deg/s at ``vx=0.24``, ``wz=+/-0.25``).
#
# Placing five anchors on a ring is the obvious approach and it is wrong: the
# three approach ranges are CHAINED, because each recall starts from where the
# previous one stopped.  The solver walks the sequence forward instead, so
# every range and bearing is correct by construction.  The MEASURED result:
#
#   leg 1  red     range 1.94 m   bearing   -1.8 deg   turn   -1.8 deg
#   leg 2  yellow  range 1.89 m   bearing -120.7 deg   turn -118.9 deg (RIGHT)
#   leg 3  green   range 2.23 m   bearing +114.1 deg   turn -125.2 deg (RIGHT)
#
# Every pair of call bearings is more than 110 deg apart, the two large turns
# are both to the faster side, min separation between any two adults over the
# rollout is 1.501 m, and the duck's resting points stay 0.898 m clear of every
# bystander.  The first call comes from nearly straight ahead on purpose: the
# opening search should be a short sweep, and the LATER calls are the ones that
# force a genuine rear search.
#
# Blue and purple never call.  They exist so acquisition has to reject four
# wrong identities, and blue additionally issues the interrupting call that the
# no-steal rule must refuse.
#
# The pacing loops are deliberately slow (about 0.04-0.09 m/s, roughly a fifth
# of the duck's walking speed).  They must not be static - a caller who freezes
# turns the approach into a fixed-point problem - but a caller who strides
# around at walking pace would make the standoff band a chase rather than a
# recall, and that is a different behavior.
STATIONS: tuple[Station, ...] = (
    Station("red", anchor=(1.65, -0.04), radii=(0.30, 0.20),
            period=29.0, rotation=math.radians(-20.0), phase=math.radians(60.0),
            reverse=True),
    Station("yellow", anchor=(0.40, -1.38), radii=(0.32, 0.19),
            period=31.0, rotation=math.radians(105.0), phase=math.radians(310.0)),
    Station("green", anchor=(-0.35, 1.06), radii=(0.31, 0.21),
            period=26.0, rotation=math.radians(35.0), phase=math.radians(200.0)),
    Station("blue", anchor=(-2.25, -0.48), radii=(0.33, 0.21),
            period=24.0, rotation=math.radians(150.0), phase=math.radians(120.0),
            reverse=True),
    Station("purple", anchor=(1.65, 1.60), radii=(0.29, 0.24),
            period=27.5, rotation=math.radians(70.0), phase=math.radians(25.0)),
)

STATION_BY_NAME = {station.name: station for station in STATIONS}


def crowd_at(t: float) -> dict[str, AdultState]:
    """Every adult's state at time ``t``."""
    states: dict[str, AdultState] = {}
    for station in STATIONS:
        pos, vel = station.at(t)
        speed = float(np.linalg.norm(vel))
        yaw = math.atan2(float(vel[1]), float(vel[0]))
        states[station.name] = AdultState(
            name=station.name, pos=pos, vel=vel, yaw=yaw, speed=speed
        )
    return states


def min_person_separation(seconds: float, dt: float = 0.1) -> tuple[float, str, str]:
    """Smallest centre-to-centre distance between any two adults over a rollout.

    Reported rather than assumed: the pacing loops are independent, so nothing
    structurally prevents two of them from overlapping.  Tests pin the measured
    value so a later anchor change cannot quietly create a merged pair.
    """
    worst = (float("inf"), "", "")
    steps = int(seconds / dt) + 1
    for index in range(steps):
        crowd = crowd_at(index * dt)
        for i, first in enumerate(ADULT_NAMES):
            for second in ADULT_NAMES[i + 1:]:
                distance = float(
                    np.linalg.norm(crowd[first].pos - crowd[second].pos)
                )
                if distance < worst[0]:
                    worst = (distance, first, second)
    return worst


def pose_crowd(model, data, crowd: dict[str, AdultState], t: float, *,
               caller: str | None = None, duck_xy: np.ndarray | None = None,
               wave: bool = False) -> None:
    """Write every adult's mocap pose, walking gait and the caller's wave.

    The active caller turns to FACE the duck and raises one arm overhead; the
    others keep pacing and swinging their arms normally.  All of it is
    kinematic scenery written straight into mocap slots and hinge qpos - none
    of it is simulated, and none of it can touch the robot.
    """
    for order, name in enumerate(ADULT_NAMES):
        adult = crowd[name]
        body = model.body(f"person_{name}")
        mocap = int(model.body_mocapid[body.id])
        phase_t = t + 0.53 * order

        yaw = adult.yaw
        is_caller = wave and name == caller
        if is_caller and duck_xy is not None:
            # Look at whoever you are calling.
            delta = np.asarray(duck_xy, dtype=np.float64) - adult.pos
            yaw = math.atan2(float(delta[1]), float(delta[0]))

        data.mocap_pos[mocap, :2] = adult.pos
        data.mocap_pos[mocap, 2] = 0.36 + 0.006 * abs(
            math.sin(2.0 * math.pi * STRIDE_HZ * phase_t)
        )
        data.mocap_quat[mocap] = np.array(
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        )

        amplitude = math.radians(10.0 + 120.0 * min(adult.speed, 0.12))
        stride = amplitude * math.sin(2.0 * math.pi * STRIDE_HZ * phase_t)
        swing = {
            f"{name}_hip_l": stride,
            f"{name}_hip_r": -stride,
            f"{name}_shoulder_l": -0.6 * stride,
            f"{name}_shoulder_r": 0.6 * stride,
        }
        if is_caller:
            # Right arm overhead, waving; left arm keeps a small natural swing.
            swing[f"{name}_shoulder_r"] = math.radians(
                WAVE_CENTER_DEG
                + WAVE_AMPLITUDE_DEG
                * math.sin(2.0 * math.pi * WAVE_HZ * t)
            )
            swing[f"{name}_shoulder_l"] = -0.3 * stride
        for joint_name, value in swing.items():
            joint = model.joint(joint_name).id
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.qvel[int(model.jnt_dofadr[joint])] = 0.0
