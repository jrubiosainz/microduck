#!/usr/bin/env python3
"""Independent pedestrians, target trails and search/follow state machine."""
from bisect import bisect_right
from dataclasses import dataclass
import math

import numpy as np

CTRL_HZ = 50.0
TRAIL_DISTANCE = 0.55
COLORS = ("BLUE", "GREEN", "RED", "YELLOW", "PURPLE")
TARGET_SEQUENCE = ("BLUE", "GREEN", "RED", "BLUE")


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PersonState:
    color: str
    pos: np.ndarray
    yaw: float
    velocity: np.ndarray
    moving: bool = True


@dataclass(frozen=True)
class TrailState:
    pos: np.ndarray
    yaw: float
    path_s: float
    leader_path_s: float


def _ellipse(color: str, t: float, *, center, radii, omega, phase,
             wobble=0.0) -> PersonState:
    """A smooth closed pedestrian route with a small independent wobble."""
    angle = phase + omega * t
    rx, ry = radii
    pos = np.array([
        center[0] + rx * math.cos(angle) + wobble * math.sin(0.17 * t + phase),
        center[1] + ry * math.sin(angle) + 0.5 * wobble * math.sin(0.11 * t),
    ], dtype=np.float64)
    velocity = np.array([
        -rx * omega * math.sin(angle) + 0.17 * wobble * math.cos(0.17 * t + phase),
        ry * omega * math.cos(angle) + 0.055 * wobble * math.cos(0.11 * t),
    ], dtype=np.float64)
    yaw = math.atan2(float(velocity[1]), float(velocity[0]))
    return PersonState(color, pos, yaw, velocity, True)


def crowd_trajectory(t: float) -> dict[str, PersonState]:
    """Five logical but deliberately unsynchronised walking routes."""
    return {
        "BLUE": _ellipse("BLUE", t, center=(1.80, 0.75), radii=(0.52, 0.25),
                         omega=0.080, phase=0.10, wobble=0.025),
        "GREEN": _ellipse("GREEN", t, center=(0.92, 0.52), radii=(0.46, 0.24),
                          omega=-0.073, phase=1.75, wobble=0.030),
        "RED": _ellipse("RED", t, center=(1.20, 1.15), radii=(0.56, 0.24),
                        omega=0.068, phase=3.75, wobble=0.020),
        "YELLOW": _ellipse("YELLOW", t, center=(0.82, 1.00), radii=(0.62, 0.20),
                           omega=-0.092, phase=5.05, wobble=0.035),
        "PURPLE": _ellipse("PURPLE", t, center=(0.90, -1.00), radii=(0.52, 0.22),
                           omega=0.087, phase=2.75, wobble=0.028),
    }


class FootstepTrail:
    """Interpolate a point a fixed arc length behind one pedestrian."""

    def __init__(self, initial: PersonState, gap: float = TRAIL_DISTANCE):
        self.gap = gap
        self.path_s = 0.0
        self.previous_pos = initial.pos.copy()
        virtual_start = initial.pos - gap * np.array(
            [math.cos(initial.yaw), math.sin(initial.yaw)])
        self.distances = [-gap, 0.0]
        self.positions = [virtual_start, initial.pos.copy()]
        self.yaws = [initial.yaw, initial.yaw]

    def update(self, person: PersonState) -> TrailState:
        delta = float(np.linalg.norm(person.pos - self.previous_pos))
        if delta > 1e-8:
            self.path_s += delta
            self.distances.append(self.path_s)
            self.positions.append(person.pos.copy())
            self.yaws.append(person.yaw)
        self.previous_pos = person.pos.copy()
        target_s = self.path_s - self.gap
        upper = min(max(bisect_right(self.distances, target_s), 1),
                    len(self.distances) - 1)
        lower = upper - 1
        span = self.distances[upper] - self.distances[lower]
        fraction = 0.0 if span <= 1e-10 else (
            target_s - self.distances[lower]) / span
        pos = self.positions[lower] + fraction * (
            self.positions[upper] - self.positions[lower])
        yaw = wrap(self.yaws[lower] + fraction * wrap(
            self.yaws[upper] - self.yaws[lower]))
        return TrailState(pos, yaw, target_s, self.path_s)


class SearchFollowStateMachine:
    """Enforce SEARCH→FOUND→FOLLOW→STOP for four target selections."""

    FOUND_SECONDS = 1.0
    FOLLOW_SECONDS = 9.0
    STOP_SECONDS = 1.5
    MIN_SEARCH_SECONDS = 0.4
    MAX_SEARCH_SECONDS = 8.0

    def __init__(self):
        self.index = 0
        self.state = "SEARCH"
        self.state_since = 0.0
        self.cycles = []
        self.current = {
            "selection": 1,
            "target": TARGET_SEQUENCE[0],
            "search_start_s": 0.0,
        }

    @property
    def target(self) -> str:
        return TARGET_SEQUENCE[min(self.index, len(TARGET_SEQUENCE) - 1)]

    @property
    def done(self) -> bool:
        return self.index >= len(TARGET_SEQUENCE)

    def update(self, t: float, camera: dict) -> tuple[str, str, bool]:
        if self.done:
            return "DONE", TARGET_SEQUENCE[-1], False
        elapsed = t - self.state_since
        changed = False
        if self.state == "SEARCH":
            seen = camera.get("target_visible", False)
            centered = camera.get("target_off_axis", math.pi) < math.radians(8.0)
            if elapsed >= self.MIN_SEARCH_SECONDS and seen and centered:
                self.state = "FOUND"
                self.current["found_s"] = t
                self.current["search_duration_s"] = elapsed
                changed = True
            elif elapsed >= self.MAX_SEARCH_SECONDS:
                raise RuntimeError(f"camera failed to find {self.target} in {elapsed:.2f}s")
        elif self.state == "FOUND" and elapsed >= self.FOUND_SECONDS:
            self.state = "FOLLOW"
            self.current["follow_start_s"] = t
            changed = True
        elif self.state == "FOLLOW" and elapsed >= self.FOLLOW_SECONDS:
            self.state = "STOP"
            self.current["stop_s"] = t
            changed = True
        elif self.state == "STOP" and elapsed >= self.STOP_SECONDS:
            self.current["cycle_end_s"] = t
            self.cycles.append(dict(self.current))
            self.index += 1
            if self.done:
                return "DONE", TARGET_SEQUENCE[-1], True
            self.state = "SEARCH"
            self.current = {
                "selection": self.index + 1,
                "target": self.target,
                "search_start_s": t,
            }
            changed = True
        if changed:
            self.state_since = t
        return self.state, self.target, changed


class CrowdFollowController:
    """Walk toward the selected pedestrian's delayed world-space footprint."""

    def __init__(self):
        self.command = np.zeros(3, dtype=np.float32)

    def update(self, active: bool, trail: TrailState, duck_pos: np.ndarray,
               duck_yaw: float) -> tuple[np.ndarray, dict]:
        error = trail.pos - np.asarray(duck_pos[:2], dtype=np.float64)
        distance = float(np.linalg.norm(error))
        desired_yaw = (math.atan2(float(error[1]), float(error[0]))
                       if distance > 0.10 else trail.yaw)
        yaw_error = wrap(desired_yaw - duck_yaw)
        if not active:
            raw = (0.0, 0.0, 0.0)
        else:
            # The stock walking policy has a hard gait-onset threshold.  The
            # old 0.16 command used for large heading errors can leave it
            # standing still (the RED hand-off exposed this).  Keep the
            # measured 0.24 walking command while turning, as in the validated
            # left/right base, so every target switch actually initiates gait.
            vx = 0.24
            if distance < 0.14:
                vx = 0.0
            if abs(yaw_error) < math.radians(4):
                wz = 0.0
            elif yaw_error > 0.0:
                wz = min(1.0, max(0.60, 1.25 * yaw_error))
            else:
                wz = -min(0.32, max(0.18, 1.25 * -yaw_error))
            raw = (vx, 0.0, wz)
        target = np.asarray(raw, dtype=np.float32)
        self.command += target - self.command
        return self.command.copy(), {
            "target_pos": trail.pos,
            "error": distance,
            "desired_yaw": desired_yaw,
            "yaw_error": yaw_error,
            "spatial_lag": trail.leader_path_s - trail.path_s,
        }


def animate_crowd(model, data, crowd: dict[str, PersonState], t: float) -> None:
    """Pose all pedestrians and animate independent gait phases."""
    for order, (color, person) in enumerate(crowd.items()):
        prefix = color.lower()
        body = model.body(f"person_{prefix}")
        mocap = int(model.body_mocapid[body.id])
        phase_t = t + 0.73 * order
        data.mocap_pos[mocap, :2] = person.pos
        data.mocap_pos[mocap, 2] = 0.36 + 0.006 * abs(
            math.sin(2.0 * math.pi * 1.15 * phase_t))
        data.mocap_quat[mocap] = np.array([
            math.cos(person.yaw / 2.0), 0.0, 0.0,
            math.sin(person.yaw / 2.0),
        ])
        stride = math.radians(24.0) * math.sin(2.0 * math.pi * 1.15 * phase_t)
        values = {
            f"{prefix}_hip_l": stride,
            f"{prefix}_hip_r": -stride,
            f"{prefix}_shoulder_l": -0.65 * stride,
            f"{prefix}_shoulder_r": 0.65 * stride,
        }
        for name, value in values.items():
            joint = model.joint(name).id
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.qvel[int(model.jnt_dofadr[joint])] = 0.0
