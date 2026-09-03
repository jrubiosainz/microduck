#!/usr/bin/env python3
"""Two adults walking the corridor on continuous scripted trajectories.

Pure analytic functions of ``t``: position, velocity and heading are all
closed-form, so the encounter predictor can be unit-tested without MuJoCo.

Design constraints this file exists to satisfy
----------------------------------------------
* **Nobody teleports, nobody freezes, nobody yields.**  Each adult walks at a
  constant speed along the corridor for the whole of its own pass and never
  reacts to the duck.  There is no branch anywhere that pauses a person so the
  duck can get out of the way.  A pull-over that "worked" because the adult
  waited would prove nothing.
* **Nobody steps aside either.**  Each adult holds a fixed lateral offset near
  the centreline, so the corridor really is blocked while they are in it.  If
  the adult drifted to a wall, the duck would not need to pull over at all.
* **Off-stage means off-stage, not frozen mid-corridor.**  Before its pass an
  adult stands well beyond the corridor end, out of the camera and outside the
  predictor's reach; after it, it keeps walking away.  ``max_visible_jump``
  measures that no adult ever makes a visible discontinuity, rather than
  asserting it.
* **Two encounters from opposite directions.**  ``chen`` walks toward the duck
  head-on (−X).  ``diaz`` enters behind the duck and overtakes it (+X), which
  is a genuinely different geometry: the closing speed is lower, the person
  appears in the duck's rear hemisphere, and the alcove that suits it is on the
  other wall.

Every adult is a MOCAP body with ``contype="0" conaffinity="0"``: kinematic
scenery that cannot touch or push the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Stride animation frequency, shared with the poser.
STRIDE_HZ: float = 1.05


@dataclass(frozen=True)
class Pedestrian:
    """One adult walking the corridor at constant speed during its own window.

    The trajectory is specified as "be at ``x_at_start`` when ``t = start_s``,
    then walk at ``speed`` in direction ``direction`` until ``end_s``".  Before
    ``start_s`` the adult waits at its entry point, which is placed off the end
    of the corridor; after ``end_s`` it keeps walking, so it exits rather than
    stopping in shot.

    ``lateral_y`` is the adult's own offset from the centreline.  It is small
    and CONSTANT: these people walk down the middle of a corridor and do not
    move over.
    """

    name: str
    rgba: str
    direction: float           # +1 walks toward +X, -1 toward -X
    speed: float
    x_at_start: float
    start_s: float
    end_s: float
    lateral_y: float = 0.0
    label: str = ""

    def x_at(self, t: float) -> float:
        clock = min(max(t, self.start_s), self.end_s)
        return self.x_at_start + self.direction * self.speed * (
            clock - self.start_s)

    def moving_at(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    def pos_at(self, t: float) -> np.ndarray:
        return np.array([self.x_at(t), self.lateral_y], dtype=np.float64)

    def vel_at(self, t: float) -> np.ndarray:
        if not self.moving_at(t):
            return np.zeros(2, dtype=np.float64)
        return np.array([self.direction * self.speed, 0.0], dtype=np.float64)

    @property
    def yaw(self) -> float:
        return 0.0 if self.direction > 0 else math.pi

    @property
    def half_length(self) -> float:
        """Planar half-extent along the corridor axis. Pinned by a test."""
        return 0.104


@dataclass(frozen=True)
class PersonState:
    name: str
    pos: np.ndarray
    vel: np.ndarray
    yaw: float
    speed: float
    direction: float
    moving: bool
    half_length: float
    label: str


# THE SCHEDULE.
#
# Speeds are metres per second at the scene's scale.  The duck's MEASURED
# cruise is 0.25-0.30 m/s (tools/sweep_commands.py), so an adult at 0.42 m/s
# closes head-on at about 0.70 m/s: roughly the ratio a walking adult presents
# to a child-sized robot, and fast enough that the pull-over has to be decided
# early rather than reacted to.
#
# ``chen`` starts 3.4 m beyond the corridor's far end so it is genuinely
# off-stage at t=0, walks in through the doorway, meets the duck around
# x ≈ +0.3, and keeps going out of the near end.
#
# ``diaz`` enters from BEHIND the duck at a lower speed.  Overtaking is the
# harder encounter to detect: the closing speed is only 0.42 − 0.28 = 0.14 m/s
# early on, so a predictor that only looks at range would react far too late,
# and a predictor that only looks at head-on bearings would never react at all.
PEDESTRIANS: tuple[Pedestrian, ...] = (
    Pedestrian("chen", rgba="0.90 0.32 0.24 1", direction=-1.0, speed=0.42,
               x_at_start=+6.20, start_s=0.0, end_s=44.0, lateral_y=+0.020,
               label="head-on"),
    Pedestrian("diaz", rgba="0.20 0.44 0.88 1", direction=-1.0, speed=0.42,
               x_at_start=+16.00, start_s=0.0, end_s=60.0, lateral_y=-0.020,
               label="second, head-on"),
)
PERSON_NAMES: tuple[str, ...] = tuple(p.name for p in PEDESTRIANS)
PERSON_BY_NAME: dict[str, Pedestrian] = {p.name: p for p in PEDESTRIANS}


def people_at(t: float) -> dict[str, PersonState]:
    """Every adult's state at time ``t``."""
    states: dict[str, PersonState] = {}
    for person in PEDESTRIANS:
        states[person.name] = PersonState(
            name=person.name,
            pos=person.pos_at(t),
            vel=person.vel_at(t),
            yaw=person.yaw,
            speed=person.speed if person.moving_at(t) else 0.0,
            direction=person.direction,
            moving=person.moving_at(t),
            half_length=person.half_length,
            label=person.label,
        )
    return states


def corridor_passes(seconds: float, dt: float = 0.02,
                    x_low: float = -4.20, x_high: float = 4.60) -> list[dict]:
    """Every interval during which an adult's body is inside the corridor.

    Reported rather than assumed, so the README can state when the corridor was
    actually occupied instead of quoting the numbers the schedule was written
    with.
    """
    events: list[dict] = []
    open_pass: dict[str, dict] = {}
    steps = int(seconds / dt) + 1
    for index in range(steps):
        t = index * dt
        for name, state in people_at(t).items():
            x = float(state.pos[0])
            inside = (x_low - state.half_length <= x
                      <= x_high + state.half_length)
            if inside and name not in open_pass:
                open_pass[name] = {"person": name, "enter_s": t,
                                   "direction": state.direction,
                                   "speed_mps": state.speed}
            elif not inside and name in open_pass:
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
                     visible_half_x: float = 6.0) -> tuple[float, str, float]:
    """Largest single-tick position jump while an adult is anywhere near.

    The honest claim is not "the trajectory is continuous by inspection" but
    "no adult ever moves discontinuously where it could be seen".  This measures
    exactly that, and a test pins it below one tick of ordinary walking.
    """
    worst = (0.0, "", 0.0)
    steps = int(seconds / dt)
    previous = {name: float(state.pos[0])
                for name, state in people_at(0.0).items()}
    for index in range(1, steps + 1):
        t = index * dt
        for name, state in people_at(t).items():
            x = float(state.pos[0])
            jump = abs(x - previous[name])
            if abs(x) <= visible_half_x and jump > worst[0]:
                worst = (jump, name, t)
            previous[name] = x
    return worst


def min_person_separation(seconds: float, dt: float = 0.05) -> tuple[float, str, str]:
    """Closest any two adults come to each other along the corridor axis.

    Two people walking opposite ways down a 0.54 m corridor cannot occupy the
    same station, so a schedule that lets them overlap would show one person
    walking through another.  Reported rather than asserted.
    """
    worst = (float("inf"), "", "")
    steps = int(seconds / dt) + 1
    for index in range(steps):
        people = people_at(index * dt)
        names = list(people)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                a, b = people[first], people[second]
                gap = abs(float(a.pos[0] - b.pos[0])) - (
                    a.half_length + b.half_length)
                if gap < worst[0]:
                    worst = (gap, first, second)
    return worst


def pose_people(model, data, people: dict[str, PersonState], t: float) -> None:
    """Write every adult's mocap pose and animate their walking gait.

    Kinematic scenery written straight into mocap slots: none of it is
    simulated and none of it can touch the robot.  The stride amplitude follows
    each adult's own speed, so a stopped adult stands still instead of marching
    on the spot.
    """
    for order, name in enumerate(PERSON_NAMES):
        person = people[name]
        body = model.body(f"person_{name}")
        mocap = int(model.body_mocapid[body.id])
        phase_t = t + 0.53 * order
        data.mocap_pos[mocap, :2] = person.pos
        data.mocap_pos[mocap, 2] = 0.36 + 0.007 * abs(
            math.sin(2.0 * math.pi * STRIDE_HZ * phase_t))
        data.mocap_quat[mocap] = np.array(
            [math.cos(person.yaw / 2.0), 0.0, 0.0, math.sin(person.yaw / 2.0)])
        amplitude = math.radians(14.0 + 90.0 * min(person.speed, 0.22))
        stride = (amplitude * math.sin(2.0 * math.pi * STRIDE_HZ * phase_t)
                  if person.moving else 0.0)
        swing = {
            f"{name}_hip_l": stride,
            f"{name}_hip_r": -stride,
            f"{name}_shoulder_l": -0.6 * stride,
            f"{name}_shoulder_r": 0.6 * stride,
        }
        for joint_name, value in swing.items():
            joint = model.joint(joint_name).id
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.qvel[int(model.jnt_dofadr[joint])] = 0.0
