#!/usr/bin/env python3
"""Leader choreography and spatial footstep queue for follow-me-footsteps."""
from bisect import bisect_right
from dataclasses import dataclass
import math

import numpy as np

TRAIL_DISTANCE = 0.65
CTRL_HZ = 50.0
CMD_TAU = 1.0 / CTRL_HZ


@dataclass(frozen=True)
class PersonState:
    phase: str
    pos: np.ndarray
    yaw: float
    velocity: np.ndarray
    yaw_rate: float
    moving: bool
    progress: float


@dataclass(frozen=True)
class TrailState:
    phase: str
    pos: np.ndarray
    yaw: float
    path_s: float
    leader_path_s: float
    moving: bool


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _left(yaw: float) -> np.ndarray:
    return np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)


def _forward(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)


def person_trajectory(t: float) -> PersonState:
    """Script the leader's deterministic world-space route."""
    start = np.array([0.65, 0.0], dtype=np.float64)
    forward_speed = 0.055
    turn_speed = 0.120
    side_speed = 0.050
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

    right_velocity = (0.065 * _forward(turn_yaw)
                      - side_speed * _left(turn_yaw))
    if t < 28.0:
        u = t - 18.0
        pos = turn_end + right_velocity * u
        return PersonState("RIGHT", pos, turn_yaw, right_velocity, 0.0, True,
                           u / 10.0)

    right_end = turn_end + right_velocity * 10.0
    if t < 34.0:
        u = t - 28.0
        velocity = -back_speed * _forward(turn_yaw)
        pos = right_end + velocity * u
        return PersonState("BACKWARD", pos, turn_yaw, velocity, 0.0, True,
                           u / 6.0)

    back_end = right_end - back_speed * _forward(turn_yaw) * 6.0
    return PersonState("DONE", back_end, turn_yaw,
                       np.zeros(2), 0.0, False, min((t - 34.0) / 3.0, 1.0))


class FootstepTrail:
    """Return the world-space point walked TRAIL_DISTANCE metres earlier.

    This is the semantic correction over v1. The target is sampled from the
    leader's accumulated path, not reconstructed from the leader's current pose.
    Therefore a corner remains in the queue: the duck continues straight until
    its delayed target reaches the exact place where the leader turned.
    """

    def __init__(self, initial: PersonState, gap: float = TRAIL_DISTANCE):
        self.gap = gap
        self.path_s = 0.0
        self.previous_pos = initial.pos.copy()
        virtual_start = initial.pos - gap * _forward(initial.yaw)
        self.distances = [-gap, 0.0]
        self.samples = [
            TrailState("FORWARD", virtual_start, initial.yaw,
                       -gap, 0.0, False),
            TrailState("FORWARD", initial.pos.copy(), initial.yaw,
                       0.0, 0.0, False),
        ]

    def update(self, leader: PersonState) -> TrailState:
        delta = float(np.linalg.norm(leader.pos - self.previous_pos))
        if delta > 1e-8:
            self.path_s += delta
            self.distances.append(self.path_s)
            self.samples.append(TrailState(
                leader.phase, leader.pos.copy(), leader.yaw,
                self.path_s, self.path_s, leader.moving))
        self.previous_pos = leader.pos.copy()

        target_s = self.path_s - self.gap
        upper = min(max(bisect_right(self.distances, target_s), 1),
                    len(self.distances) - 1)
        lower = upper - 1
        a, b = self.samples[lower], self.samples[upper]
        span = self.distances[upper] - self.distances[lower]
        fraction = 0.0 if span <= 1e-10 else (
            target_s - self.distances[lower]) / span
        pos = a.pos + fraction * (b.pos - a.pos)
        yaw = wrap(a.yaw + fraction * wrap(b.yaw - a.yaw))
        phase = b.phase
        return TrailState(
            phase=phase,
            pos=pos,
            yaw=yaw,
            path_s=target_s,
            leader_path_s=self.path_s,
            moving=leader.moving,
        )


class FollowController:
    """Replay the motion stored at the delayed world-space trail point."""

    def __init__(self, hz: float = CTRL_HZ):
        self.dt = 1.0 / hz
        self.command = np.zeros(3, dtype=np.float32)

    def update(self, leader: PersonState, trail: TrailState,
               duck_pos: np.ndarray, duck_yaw: float) -> tuple[np.ndarray, dict]:
        error_world = trail.pos - np.asarray(duck_pos[:2], dtype=np.float64)
        yaw_error = wrap(trail.yaw - duck_yaw)

        phase_commands = {
            "FORWARD": (0.24, 0.0, 0.0),
            "LEFT TURN": (0.24, 0.0, -0.32),
            "RIGHT": (0.24, -0.12, 0.0),
            "BACKWARD": (-0.32, 0.0, 0.20),
        }
        # A stopped leader freezes the spatial queue. Reversal is the one
        # safety exception: when the leader backs toward the follower, the duck
        # backs immediately so the gap cannot collapse, while turns and lateral
        # legs still come from the queued world-space footsteps.
        replay_phase = ("BACKWARD" if leader.phase == "BACKWARD"
                        else trail.phase)
        target_cmd = np.array(
            phase_commands.get(replay_phase, (0.0, 0.0, 0.0))
            if leader.moving else (0.0, 0.0, 0.0),
            dtype=np.float32,
        )
        alpha = min(1.0, self.dt / CMD_TAU)
        self.command += alpha * (target_cmd - self.command)
        metrics = {
            "target_pos": trail.pos,
            "error_world": error_world,
            "error": float(np.linalg.norm(error_world)),
            "yaw_error": yaw_error,
            "target_cmd": target_cmd,
            "trail_phase": trail.phase,
            "replay_phase": replay_phase,
            "trail_path_s": trail.path_s,
            "leader_path_s": trail.leader_path_s,
            "spatial_lag": trail.leader_path_s - trail.path_s,
        }
        return self.command.copy(), metrics


def animate_person(model, data, person, t: float) -> None:
    """Apply root pose and a speed-dependent opposing limb cycle."""
    body_id = model.body("person").id
    mocap_id = int(model.body_mocapid[body_id])
    data.mocap_pos[mocap_id, :2] = person.pos
    data.mocap_pos[mocap_id, 2] = 0.36 + (
        0.006 * abs(math.sin(2 * math.pi * t * 1.3)) if person.moving else 0.0)
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
        data.qpos[int(model.jnt_qposadr[joint_id])] = value
        data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0
