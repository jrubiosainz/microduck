#!/usr/bin/env python3
"""Posing the world each control tick: bodies on their routes.

WHAT THE DUCK NEVER READS
--------------------------
This module and :mod:`pps_script`.  ``tests/test_rollout_and_hygiene.py`` parses
the import graph with ``ast`` and fails if ``pps_machine``, ``pps_threat``,
``pps_geometry``, ``pps_control``, ``pps_sense`` or ``pps_camera`` ever imports
either.  The duck measures every person's position through the same per-tick
world state its contact probe uses, and sees them through the real head camera.

THE PEOPLE DO NOT REACT TO THE ROBOT, AND THAT IS THE POINT
-------------------------------------------------------------
Every route here is a pure function of ``t``.  Nobody swerves around the duck,
nobody waits for it, and nobody is repelled by it.  Two things follow, and both
are load-bearing:

* **The duck cannot "block" anybody.**  The actors are non-colliding, so an
  intrusion is never stopped by the robot being in the way.  What the gates
  measure instead is whether the robot got ONTO the bearing between the two
  people, with positive clearance to both, before the intruder arrived - which
  is the honest simulation-only form of the claim.
* **Every clearance is the robot's own doing.**  A person who politely stepped
  around the duck would make "zero contacts" a fact about the choreography.

A PERSON'S GAIT IS ANIMATED EVEN WHEN THEY ARE STANDING STILL, at a low
amplitude and rate: a body frozen mid-stride reads as a mannequin, and several
of these people hold position for seven seconds at a time.
"""

from __future__ import annotations

import math

import numpy as np

from pps_cast import ALL_NAMES, BY_NAME
from pps_script import ROUTES

STRIDE_HZ: float = 1.05


class BodyState:
    """One person's world state at one instant.  Positions, never percepts."""

    __slots__ = ("name", "pos", "yaw", "speed", "velocity", "present")

    def __init__(self, name, pos, yaw, speed, velocity, present):
        self.name = name
        self.pos = pos
        self.yaw = yaw
        self.speed = speed
        self.velocity = velocity
        self.present = present


def bodies_at(t: float) -> dict[str, BodyState]:
    """Every person's world state at ``t``, in cast order."""
    states: dict[str, BodyState] = {}
    for name in ALL_NAMES:
        route = ROUTES[name]
        speed = route.speed_at(t)
        yaw = route.yaw_at(t)
        planar = route.pos_at(t)
        position = np.array([planar[0], planar[1]])
        present = t >= route.start_t
        velocity = (np.array([math.cos(yaw), math.sin(yaw)]) * speed
                    if speed > 0.0 else np.zeros(2))
        states[name] = BodyState(name, position, yaw, speed, velocity, present)
    return states


def pose_bodies(model, data, states: dict[str, BodyState], t: float) -> None:
    """Write mocap poses and animate the gait."""
    for order, name in enumerate(ALL_NAMES):
        state = states[name]
        spec = BY_NAME[name]
        body = model.body(f"actor_{name}")
        mocap = int(model.body_mocapid[body.id])

        if not state.present:
            data.mocap_pos[mocap] = (float(state.pos[0]), float(state.pos[1]),
                                     -3.0)
            data.mocap_quat[mocap] = np.array([1.0, 0.0, 0.0, 0.0])
            continue

        phase = t + 0.47 * order
        data.mocap_pos[mocap, :2] = state.pos
        data.mocap_pos[mocap, 2] = spec.origin_z + 0.008 * abs(
            math.sin(2.0 * math.pi * STRIDE_HZ * phase))
        data.mocap_quat[mocap] = np.array(
            [math.cos(state.yaw / 2.0), 0.0, 0.0, math.sin(state.yaw / 2.0)])

        if state.speed > 1e-3:
            amplitude = math.radians(18.0 + 78.0 * min(state.speed, 0.30))
            rate = STRIDE_HZ
        else:
            amplitude = math.radians(3.4)
            rate = 0.29
        stride = amplitude * math.sin(2.0 * math.pi * rate * phase)
        for joint_name, value in ((f"{name}_hip_l", stride),
                                  (f"{name}_hip_r", -stride)):
            joint = model.joint(joint_name).id
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.qvel[int(model.jnt_dofadr[joint])] = 0.0


def max_heading_step(seconds: float, dt: float = 0.02) -> tuple[float, str, float]:
    """Largest single-tick heading change any scripted person makes, in degrees.

    The property a filleted route buys and a cornered polyline does not.  A
    person is only counted while they are actually walking: a stationary walker's
    "heading" is the tangent at a frozen arc length, which does not move, and a
    hold window's edges are therefore not turns.
    """
    worst = (0.0, "", 0.0)
    previous = {n: s.yaw for n, s in bodies_at(0.0).items()}
    for index in range(1, int(seconds / dt) + 1):
        t = index * dt
        for name, state in bodies_at(t).items():
            delta = abs(math.degrees(math.atan2(
                math.sin(state.yaw - previous[name]),
                math.cos(state.yaw - previous[name]))))
            if delta > worst[0] and state.speed > 0.0:
                worst = (delta, name, t)
            previous[name] = state.yaw
    return worst


def moving_fraction(seconds: float, dt: float = 0.10) -> dict[str, float]:
    """Fraction of the session each scripted person spends actually walking."""
    counts = {name: 0 for name in ALL_NAMES}
    steps = int(seconds / dt)
    for index in range(steps):
        for name, state in bodies_at(index * dt).items():
            if state.speed > 0.0:
                counts[name] += 1
    return {name: counts[name] / max(steps, 1) for name in ALL_NAMES}


def route_records() -> list[dict]:
    """Every scripted route, for the metrics to publish."""
    records = []
    for name in ALL_NAMES:
        record = ROUTES[name].as_record()
        record["role"] = BY_NAME[name].role
        records.append(record)
    return records
