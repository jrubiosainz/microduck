#!/usr/bin/env python3
"""Where everybody walks: one continuous guardian route and four other adults.

THE GUARDIAN'S ROUTE IS THE SCENARIO
------------------------------------
She walks ONE continuous arc-length path — never a sequence of waypoints she
accelerates from rest between — with three real bends: LEFT off the south
straight, RIGHT back onto an east straight, LEFT onto the north straight.  Two
lefts and a right, because a formation keeper that only ever turns one way has
not been tested on the sign its yaw controller is weakest on.

Constant speed and continuous heading are not cosmetic.  A companion robot
graded on how well it holds a slot beside somebody must be graded against
somebody who walks; against an actor who stops dead at every corner, the
robot's lateral error is a measurement of the actor's stutter.

WHY EACH SIDE DECISION HAS A CAUSE, AND WHY THE CAUSES DIFFER
--------------------------------------------------------------
* **The initial join.**  The guardian's RIGHT-hand slot on the south straight
  lies inside ``hedge_s``.  The duck therefore cannot simply default to a side:
  it measures both, refuses the one that is occupied by a static body, and joins
  the LEFT.  The hedge ends part-way along the straight, so the right side
  becomes usable later — which is what makes the refusal a live measurement
  rather than a permanent scenery fact.
* **The first switch is caused by a MOVING body.**  ``tomas`` walks the length
  of the promenade toward them, offset 0.43 m from the duck's left lane.  That
  is closer than the 0.55 m the duck requires between itself and a pedestrian,
  so the lane it is standing in stops being usable while it is standing in it.
* **The second switch is caused by a STATIC body.**  The duck is by then on the
  guardian's right; ``kiosk`` stands in that lane on the east straight, and the
  left has become clear again.
* **``iris`` is the control case.**  She passes the duck's own lane at 0.68 m —
  closer than anybody else comes, and comfortably inside "somebody is near" —
  but outside the measured 0.55 m lane requirement.  A behavior that switched
  sides whenever a pedestrian came near would switch for her, and would fail the
  gate that requires every side decision to name a measured cause.

``rafa`` and ``lena`` cross the promenade on their own loops.  They are what
makes "the duck kept positive clearance to every person" a claim about a
populated space rather than an empty one.
"""

from __future__ import annotations

import math

import numpy as np

from beside_cast import ALL_NAMES, BY_NAME
from beside_route import Route

STRIDE_HZ: float = 1.05

# MEASURED-AGAINST-THE-DUCK CRUISE.  See ``beside_constants`` for the sweep this
# was chosen from: it sits between the duck's 0.098 m/s (vx=0.26) and 0.150 m/s
# (vx=0.38) walking speeds, so the duck can both fall behind her and catch her
# up using commands that are above the measured gait-onset cliff.
GUARDIAN_SPEED = 0.118

GUARDIAN_CORNERS = (
    (-4.10, -2.35),    # south straight, heading east
    (0.40, -2.35),     # LEFT bend, delayed until side switch is complete
    (1.80, -1.05),     # RIGHT bend onto the east straight
    (4.05, -1.05),     # LEFT bend onto the north straight
    (4.45, 1.55),
)

ROUTES: dict[str, Route] = {
    "nadia": Route("nadia", GUARDIAN_CORNERS, GUARDIAN_SPEED),
    # Oncoming, brisk, straight down the promenade and away to the north-west.
    # Offset 0.43 m from the duck's left lane: inside its 0.55 m pedestrian
    # requirement, so the lane stops being usable, and outside its own body, so
    # the measured clearance stays positive.
    "tomas": Route("tomas", ((5.10, -1.40), (-3.40, -1.40), (-4.75, 1.35)),
                   0.30),
    # The control case: passes the duck's own lane at 0.68 m and must NOT cause
    # a side change.
    "iris": Route("iris", ((2.00, -1.15), (-2.40, -1.15), (-4.80, 0.35)), 0.34),
    "rafa": Route("rafa", ((-1.20, 2.85), (3.60, 2.30), (4.90, -0.20),
                           (2.60, 2.60), (-2.00, 2.90)), 0.22),
    "lena": Route("lena", ((4.90, 2.70), (0.20, 2.60), (-4.30, 1.70)), 0.20),
}


class PersonState:
    """One adult's world state at one instant.  Positions, never percepts."""

    __slots__ = ("name", "pos", "yaw", "speed", "velocity", "role")

    def __init__(self, name: str, pos: np.ndarray, yaw: float, speed: float,
                 velocity: np.ndarray, role: str):
        self.name = name
        self.pos = pos
        self.yaw = yaw
        self.speed = speed
        self.velocity = velocity
        self.role = role


def people_at(t: float) -> dict[str, PersonState]:
    """Every adult's world state at ``t``.

    The duck's controller never receives this dictionary.  It receives only what
    ``beside_awareness`` reports, which is a stated proxy for an omnidirectional
    range sensor, and what ``beside_camera`` can actually see.
    """
    states: dict[str, PersonState] = {}
    for name in ALL_NAMES:
        route = ROUTES[name]
        speed = route.speed_at(t)
        yaw = route.yaw_at(t)
        velocity = (np.array([math.cos(yaw), math.sin(yaw)]) * speed
                    if speed > 0.0 else np.zeros(2))
        states[name] = PersonState(name, route.pos_at(t), yaw, speed, velocity,
                                   BY_NAME[name].role)
    return states


def max_visible_jump(seconds: float, dt: float = 0.02) -> tuple[float, str, float]:
    """Largest single-tick position jump any adult makes, over the rollout."""
    worst = (0.0, "", 0.0)
    previous = {n: s.pos.copy() for n, s in people_at(0.0).items()}
    for index in range(1, int(seconds / dt) + 1):
        t = index * dt
        for name, state in people_at(t).items():
            jump = float(np.linalg.norm(state.pos - previous[name]))
            if jump > worst[0]:
                worst = (jump, name, t)
            previous[name] = state.pos.copy()
    return worst


def max_heading_step(seconds: float, dt: float = 0.02) -> tuple[float, str, float]:
    """Largest single-tick heading change any adult makes, in degrees.

    This is the property the filleted route buys and the smootherstep walker
    used by the sibling behaviors does not have: a cornered polyline turns its
    walker through the whole corner in ONE tick.
    """
    worst = (0.0, "", 0.0)
    previous = {n: s.yaw for n, s in people_at(0.0).items()}
    for index in range(1, int(seconds / dt) + 1):
        t = index * dt
        for name, state in people_at(t).items():
            delta = abs(math.degrees(math.atan2(
                math.sin(state.yaw - previous[name]),
                math.cos(state.yaw - previous[name]))))
            if delta > worst[0]:
                worst = (delta, name, t)
            previous[name] = state.yaw
    return worst


def moving_fraction(seconds: float, dt: float = 0.10) -> dict[str, float]:
    """Fraction of the rollout each adult spends actually walking."""
    counts = {name: 0 for name in ALL_NAMES}
    steps = int(seconds / dt)
    for index in range(steps):
        for name, state in people_at(index * dt).items():
            if state.speed > 0.0:
                counts[name] += 1
    return {name: counts[name] / max(steps, 1) for name in ALL_NAMES}


def pose_people(model, data, people: dict[str, PersonState], t: float) -> None:
    """Write mocap poses and animate the gait.  Kinematic scenery, no contacts."""
    for order, name in enumerate(ALL_NAMES):
        person = people[name]
        spec = BY_NAME[name]
        body = model.body(f"person_{name}")
        mocap = int(model.body_mocapid[body.id])
        phase = t + 0.47 * order
        data.mocap_pos[mocap, :2] = person.pos
        data.mocap_pos[mocap, 2] = spec.origin_z + 0.008 * abs(
            math.sin(2.0 * math.pi * STRIDE_HZ * phase))
        data.mocap_quat[mocap] = np.array(
            [math.cos(person.yaw / 2.0), 0.0, 0.0, math.sin(person.yaw / 2.0)])
        if person.speed > 1e-3:
            amplitude = math.radians(18.0 + 78.0 * min(person.speed, 0.26))
            rate = STRIDE_HZ
        else:
            amplitude = math.radians(3.6)
            rate = 0.31
        stride = amplitude * math.sin(2.0 * math.pi * rate * phase)
        swing = {
            f"{name}_hip_l": stride, f"{name}_hip_r": -stride,
            f"{name}_shoulder_l": -0.62 * stride,
            f"{name}_shoulder_r": 0.62 * stride,
        }
        for joint_name, value in swing.items():
            joint = model.joint(joint_name).id
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.qvel[int(model.jnt_dofadr[joint])] = 0.0
