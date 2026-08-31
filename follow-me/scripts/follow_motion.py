#!/usr/bin/env python3
"""Trajectory and closed-loop follower for the Microduck follow-me demo."""
from dataclasses import dataclass
import math

import numpy as np

FOLLOW_DISTANCE = 0.50
CTRL_HZ = 50.0
CMD_TAU = 1.0 / CTRL_HZ
POS_KP = 0.75
YAW_KP = 0.70
MAX_VX = 0.24
MAX_VY = 0.12
MAX_WZ = 0.30


@dataclass(frozen=True)
class PersonState:
    phase: str
    pos: np.ndarray
    yaw: float
    velocity: np.ndarray
    yaw_rate: float
    moving: bool
    progress: float


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _left(yaw: float) -> np.ndarray:
    return np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)


def _forward(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)


def person_trajectory(t: float) -> PersonState:
    """Script the requested sequence with continuous positions at boundaries."""
    start = np.array([0.65, 0.0], dtype=np.float64)
    forward_speed = 0.055
    turn_speed = 0.120
    side_speed = 0.020
    back_speed = 0.080

    if t < 2.0:
        return PersonState("READY", start, 0.0, np.zeros(2), 0.0, False, t / 2.0)

    if t < 7.0:
        u = t - 2.0
        pos = start + np.array([forward_speed * u, 0.0])
        return PersonState("FORWARD", pos, 0.0,
                           np.array([forward_speed, 0.0]), 0.0, True, u / 5.0)

    turn_start = start + np.array([forward_speed * 5.0, 0.0])
    turn_duration = 8.0
    # Negative world yaw is screen-left in the chosen presentation camera and
    # is also the stock policy's stronger, validated turning direction.
    omega = -(math.pi / 2.0) / turn_duration
    radius = turn_speed / abs(omega)
    if t < 15.0:
        u = t - 7.0
        theta = omega * u
        pos = turn_start + np.array([
            radius * math.sin(abs(theta)),
            -radius * (1.0 - math.cos(theta)),
        ])
        velocity = turn_speed * _forward(theta)
        return PersonState("LEFT TURN", pos, theta, velocity, omega, True,
                           u / turn_duration)

    turn_end = turn_start + np.array([radius, -radius])
    turn_yaw = -math.pi / 2.0
    if t < 18.0:
        return PersonState("STOP", turn_end, turn_yaw,
                           np.zeros(2), 0.0, False, (t - 15.0) / 3.0)

    if t < 24.0:
        u = t - 18.0
        # A readable forward-right diagonal keeps the walking policy active
        # while moving clearly to the person's right.
        velocity = (0.065 * _forward(turn_yaw)
                    - side_speed * _left(turn_yaw))
        pos = turn_end + velocity * u
        return PersonState("RIGHT", pos, turn_yaw, velocity, 0.0, True,
                           u / 6.0)

    right_velocity = (0.065 * _forward(turn_yaw)
                      - side_speed * _left(turn_yaw))
    right_end = turn_end + right_velocity * 6.0
    if t < 30.0:
        u = t - 24.0
        velocity = -back_speed * _forward(turn_yaw)
        pos = right_end + velocity * u
        return PersonState("BACKWARD", pos, turn_yaw, velocity, 0.0, True,
                           u / 6.0)

    back_end = right_end - back_speed * _forward(turn_yaw) * 6.0
    return PersonState("DONE", back_end, turn_yaw,
                       np.zeros(2), 0.0, False, min((t - 30.0) / 3.0, 1.0))


def follow_target(person: PersonState) -> tuple[np.ndarray, np.ndarray]:
    """Desired duck pose and its feed-forward velocity, 0.5 m behind person."""
    heading = _forward(person.yaw)
    target_pos = person.pos - FOLLOW_DISTANCE * heading
    target_velocity = (
        person.velocity
        - FOLLOW_DISTANCE * person.yaw_rate * _left(person.yaw)
    )
    return target_pos, target_velocity


class FollowController:
    """World-space position/yaw servo translated to policy-frame twist commands."""

    def __init__(self, hz: float = CTRL_HZ):
        self.dt = 1.0 / hz
        self.command = np.zeros(3, dtype=np.float32)

    def update(self, person: PersonState, duck_pos: np.ndarray,
               duck_yaw: float) -> tuple[np.ndarray, dict]:
        target_pos, target_velocity = follow_target(person)
        error_world = target_pos - np.asarray(duck_pos[:2], dtype=np.float64)
        desired_world_velocity = target_velocity + POS_KP * error_world

        forward = _forward(duck_yaw)
        left = _left(duck_yaw)
        vx = float(np.dot(desired_world_velocity, forward))
        vy = float(np.dot(desired_world_velocity, left))
        yaw_error = wrap(person.yaw - duck_yaw)

        # The stock policy is sharply nonlinear around gait onset. Closed-loop
        # micro-adjustments below that threshold produced standing, followed by
        # a sudden unstable burst. Use the measured stable command for each
        # leader phase; position and camera errors remain measured outputs.
        phase_commands = {
            "READY": (0.0, 0.0, 0.0),
            "FORWARD": (0.24, 0.0, 0.0),
            "LEFT TURN": (0.24, 0.0, -0.32),
            "STOP": (0.0, 0.0, 0.0),
            "RIGHT": (0.24, -0.12, 0.0),
            "BACKWARD": (-0.32, 0.0, 0.20),
            "DONE": (0.0, 0.0, 0.0),
        }
        target_cmd = np.array(phase_commands[person.phase], dtype=np.float32)

        alpha = min(1.0, self.dt / CMD_TAU)
        self.command += alpha * (target_cmd - self.command)
        metrics = {
            "target_pos": target_pos,
            "error_world": error_world,
            "error": float(np.linalg.norm(error_world)),
            "yaw_error": yaw_error,
            "target_cmd": target_cmd,
        }
        return self.command.copy(), metrics


def animate_person(model, data, person, t: float) -> None:
    """Apply root pose and a simple speed-dependent opposing limb cycle."""
    body_id = model.body("person").id
    mocap_id = int(model.body_mocapid[body_id])
    data.mocap_pos[mocap_id, :2] = person.pos
    data.mocap_pos[mocap_id, 2] = 0.36 + (0.006 * abs(math.sin(2 * math.pi * t * 1.3))
                                           if person.moving else 0.0)
    data.mocap_quat[mocap_id] = np.array([
        math.cos(person.yaw / 2.0), 0.0, 0.0, math.sin(person.yaw / 2.0)
    ])

    amplitude = math.radians(24.0) if person.moving else 0.0
    stride = amplitude * math.sin(2.0 * math.pi * 1.3 * t)
    values = {
        "person_hip_l": stride,
        "person_hip_r": -stride,
        "person_shoulder_l": -0.65 * stride,
        "person_shoulder_r": 0.65 * stride,
    }
    for name, value in values.items():
        joint_id = model.joint(name).id
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr] = value
        data.qvel[dof_adr] = 0.0
