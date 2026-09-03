#!/usr/bin/env python3
"""Posing the world each control tick: bodies on their routes, arms on the cues.

WHAT THE DUCK NEVER READS
--------------------------
This module and :mod:`gest_script`.  ``tests/test_rollout_and_hygiene.py``
parses the import graph with ``ast`` and fails if ``gest_machine``,
``gest_gesture``, ``gest_pose``, ``gest_control`` or ``gest_acquire`` ever
imports either.  The duck measures every person's position through the same
per-tick world state its contact probe uses, sees them through the real head
camera, and reads their arms from the world positions of real keypoint bodies.

THE INSTRUCTOR STANDS STILL, AND THAT IS A MEASURED CHOICE
------------------------------------------------------------
She is on her mark for the entire session and never moves.  If she walked, "the
duck reduced the range on COME" would be partly a fact about her closing on it,
and "BACK_UP moved the duck backward" would be partly a fact about her retreating.
A stationary instructor makes every range change the robot's own doing.  Her
GAIT is still animated at a low amplitude, because a body frozen mid-stride
reads as a mannequin.

WHY THE ARMS ARE POSED THROUGH REAL JOINTS AND NOT WRITTEN AS GEOM OFFSETS
---------------------------------------------------------------------------
Each arm is three nested bodies with three hinge joints.  Writing ``qpos`` for
those joints and calling ``mj_forward`` means MuJoCo computes the shoulder,
elbow and hand world positions - which are then the SAME positions the camera
ray-casts against and the SAME positions :mod:`gest_pose` measures.  Drawing the
arm as a decoration would break that identity and make the "camera-visible
gesture" claim unfalsifiable.
"""

from __future__ import annotations

import math

import numpy as np

from gest_arm import JOINT_KEYS, REST, arm_targets
from gest_arena import INSTRUCTOR_FACING_DEG, INSTRUCTOR_MARK
from gest_cast import ALL_NAMES, BY_NAME, INSTRUCTOR
from gest_script import ROUTES, active_cue

STRIDE_HZ: float = 1.05


class BodyState:
    """One person's world state at one instant.  Positions, never percepts."""

    __slots__ = ("name", "pos", "yaw", "speed", "velocity", "present",
                 "gesture", "gesture_elapsed_s", "gesture_span_s",
                 "gesture_envelope")

    def __init__(self, name, pos, yaw, speed, velocity, present,
                 gesture, gesture_elapsed_s, gesture_span_s, gesture_envelope):
        self.name = name
        self.pos = pos
        self.yaw = yaw
        self.speed = speed
        self.velocity = velocity
        self.present = present
        # The SCENARIO's own record of what this person is doing.  Published
        # only into the metrics for cross-checking against what the duck
        # concluded; no decision layer can reach it.
        self.gesture = gesture
        self.gesture_elapsed_s = gesture_elapsed_s
        self.gesture_span_s = gesture_span_s
        self.gesture_envelope = gesture_envelope


def bodies_at(t: float) -> dict[str, BodyState]:
    """Every person's world state at ``t``, in cast order."""
    states: dict[str, BodyState] = {}
    for name in ALL_NAMES:
        cue = active_cue(name, t)
        gesture = cue.gesture if cue is not None else REST
        elapsed = (t - cue.at_s) if cue is not None else 0.0
        span = cue.span_s if cue is not None else 0.0

        if name == INSTRUCTOR:
            position = np.asarray(INSTRUCTOR_MARK, dtype=np.float64)
            yaw = math.radians(INSTRUCTOR_FACING_DEG)
            speed = 0.0
            velocity = np.zeros(2)
            present = True
        else:
            route = ROUTES[name]
            speed = route.speed_at(t)
            yaw = route.yaw_at(t)
            planar = route.pos_at(t)
            position = np.array([planar[0], planar[1]])
            present = t >= route.start_t
            velocity = (np.array([math.cos(yaw), math.sin(yaw)]) * speed
                        if speed > 0.0 else np.zeros(2))

        from gest_arm import envelope
        states[name] = BodyState(
            name, position, yaw, speed, velocity, present,
            gesture, elapsed, span,
            envelope(elapsed, span) if cue is not None else 0.0)
    return states


def pose_bodies(model, data, states: dict[str, BodyState], t: float) -> None:
    """Write mocap poses, animate the gait, and drive the six arm joints.

    A PERSON'S GAIT IS ANIMATED EVEN WHEN THEY ARE STANDING STILL, at a low
    amplitude and rate: a body frozen mid-stride reads as a mannequin.  The
    ARMS, by contrast, are driven entirely by the gesture templates - including
    the idle sway - so an arm is never both walking and gesturing at once, which
    would make the measured features depend on stride phase.
    """
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

        # -- the legs ------------------------------------------------------
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

        # -- the arms, which are what this behavior reads ------------------
        targets, _ = arm_targets(state.gesture, state.gesture_elapsed_s,
                                 state.gesture_span_s, phase)
        for key in JOINT_KEYS:
            joint = model.joint(f"{name}_{key}").id
            data.qpos[int(model.jnt_qposadr[joint])] = targets[key]
            data.qvel[int(model.jnt_dofadr[joint])] = 0.0


def max_heading_step(seconds: float, dt: float = 0.02) -> tuple[float, str, float]:
    """Largest single-tick heading change any scripted person makes, in degrees.

    The property a filleted route buys and a cornered polyline does not.  The
    instructor is excluded because she never turns; including her would average
    the number down and make the gate weaker.
    """
    worst = (0.0, "", 0.0)
    previous = {n: s.yaw for n, s in bodies_at(0.0).items()}
    for index in range(1, int(seconds / dt) + 1):
        t = index * dt
        for name, state in bodies_at(t).items():
            if name == INSTRUCTOR:
                continue
            delta = abs(math.degrees(math.atan2(
                math.sin(state.yaw - previous[name]),
                math.cos(state.yaw - previous[name]))))
            if delta > worst[0]:
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
        if name in ROUTES:
            record = ROUTES[name].as_record()
        else:
            record = {"name": name, "corners": [list(INSTRUCTOR_MARK)],
                      "length_m": 0.0, "speed_mps": 0.0, "start_t_s": 0.0,
                      "hold_windows_s": [], "finish_t_s": 0.0, "bends": []}
        record["role"] = BY_NAME[name].role
        records.append(record)
    return records
