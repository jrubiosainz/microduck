#!/usr/bin/env python3
"""The six adults who are not the follower, and how everybody is posed.

WHY EACH ROUTE IS WHAT IT IS
-----------------------------
* ``noor``, ``pablo`` cross the middle of the concourse during the planning
  window.  Their measured velocities at that instant produce swept tubes that
  cover cells the route would otherwise have used, which is what makes the
  planner's crowd term bite.  ``test_the_plan_refuses_cells_because_of_people``
  requires that.
* ``ivan`` walks the far aisle at a brisk pace and passes near the middle leg,
  so the per-tick clearance gate is graded against somebody who actually comes
  close rather than against a hall of distant extras.
* ``sena``, ``omar``, ``tessa`` loop through the space on their own paths.
  Two of them cross the duck's sightline to the follower, so the visibility
  measurement is taken in a hall where bodies genuinely pass through the line of
  sight rather than in an empty one.

All six walk continuous filleted routes at constant speed, so a linear
extrapolation of their heading over the planner's horizon is close to their true
path except inside a bend.  The follower is NOT here: her motion comes from
``guide_follower``, because she walks the duck's own trail.
"""

from __future__ import annotations

import math

import numpy as np

from guide_cast import ALL_NAMES, BY_NAME, CROWD_NAMES, FOLLOWER
from guide_route import Route

STRIDE_HZ: float = 1.05

ROUTES: dict[str, Route] = {
    # Crosses the corridor between the barriers, northbound, during the planning
    # window.  Her tube covers cells the route would otherwise have used.
    "noor": Route("noor", ((0.35, -1.55), (-0.15, 0.15), (-0.55, 1.45),
                           (-2.15, 1.05)), 0.156),
    # Crosses the other way, so the two tubes are not a single moving wall the
    # planner could route around once.
    "pablo": Route("pablo", ((-0.35, 1.55), (0.35, 0.35), (1.75, -0.85),
                             (3.15, -0.55)), 0.148),
    # The far aisle, brisk, WEST TO EAST.  Direction matters and was measured:
    # walking east-to-west put him inside the corridor between the barriers at
    # plan time, and his 0.55 m swept planning tube narrowed the one way through
    # below what the search needs.  Reversed, he is at the west end at
    # t = 1.6 s and crosses the duck's own route later, which is what makes him
    # the adult the per-tick clearance gate grades.
    "ivan": Route("ivan", ((-3.15, -1.05), (-1.55, -1.45), (0.55, -1.15),
                           (2.35, -0.35)), 0.205),
    # Loops the north half, well clear of the duck's start.  An earlier route
    # began 0.53 m from the duck's own start: her swept planning tube then
    # covered the duck's only exit and the planner reported the concourse
    # SEALED.  A background walker must populate the hall without standing in
    # the doorway.
    "sena": Route("sena", ((-0.15, 1.65), (1.55, 1.35), (2.95, 0.35),
                           (3.35, -0.85)), 0.128),
    "omar": Route("omar", ((3.15, 0.25), (1.85, 1.55), (-0.85, 1.75),
                           (-3.15, 1.15)), 0.142),
    # Reversed relative to the first draft, which started her at (-3.05, 1.55):
    # that put her 0.64 m from the HELPDESK standing point at plan time, and her
    # swept tube isolated that goal cell from every neighbour — the destination
    # existed but was unreachable, which would have quietly reduced the three
    # candidates to two.  A candidate the duck could not have walked to is not a
    # candidate it declined.
    "tessa": Route("tessa", ((1.45, -1.55), (-0.55, -1.75), (-1.95, -1.15),
                             (-2.35, 0.55), (-3.05, 1.55)), 0.118),
}

# Every non-follower name, in the order they are posed.  The follower is posed
# separately because her state comes from the duck's own trail.
ACTOR_NAMES: tuple[str, ...] = tuple(
    name for name in ALL_NAMES if name != FOLLOWER.name)


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


def actors_at(t: float) -> dict[str, PersonState]:
    """Every NON-follower adult's world state at ``t``.

    The duck's controller never receives this dictionary.  It receives only what
    its own probes measure and what its camera can actually see.
    """
    states: dict[str, PersonState] = {}
    for name in ACTOR_NAMES:
        route = ROUTES[name]
        speed = route.speed_at(t)
        yaw = route.yaw_at(t)
        velocity = (np.array([math.cos(yaw), math.sin(yaw)]) * speed
                    if speed > 0.0 else np.zeros(2))
        states[name] = PersonState(name, route.pos_at(t), yaw, speed, velocity,
                                   BY_NAME[name].role)
    return states


def follower_state(follower) -> PersonState:
    """Wrap the follower in the same state object the actors use."""
    velocity = (np.array([math.cos(follower.yaw), math.sin(follower.yaw)])
                * follower.speed if follower.speed > 1e-6 else np.zeros(2))
    return PersonState(FOLLOWER.name, follower.pos.copy(), follower.yaw,
                       follower.speed, velocity, FOLLOWER.role)


def people_at(t: float, follower) -> dict[str, PersonState]:
    """Everybody: the scripted actors plus the follower, in cast order."""
    actors = actors_at(t)
    combined: dict[str, PersonState] = {}
    for name in ALL_NAMES:
        combined[name] = (follower_state(follower) if name == FOLLOWER.name
                          else actors[name])
    return combined


def max_heading_step(seconds: float, dt: float = 0.02) -> tuple[float, str, float]:
    """Largest single-tick heading change any scripted actor makes, in degrees.

    This is the property the filleted route buys and a smootherstep walker does
    not have: a cornered polyline turns its walker through the whole corner in
    ONE tick.
    """
    worst = (0.0, "", 0.0)
    previous = {n: s.yaw for n, s in actors_at(0.0).items()}
    for index in range(1, int(seconds / dt) + 1):
        t = index * dt
        for name, state in actors_at(t).items():
            delta = abs(math.degrees(math.atan2(
                math.sin(state.yaw - previous[name]),
                math.cos(state.yaw - previous[name]))))
            if delta > worst[0]:
                worst = (delta, name, t)
            previous[name] = state.yaw
    return worst


def moving_fraction(seconds: float, dt: float = 0.10) -> dict[str, float]:
    """Fraction of the rollout each scripted actor spends actually walking."""
    counts = {name: 0 for name in ACTOR_NAMES}
    steps = int(seconds / dt)
    for index in range(steps):
        for name, state in actors_at(index * dt).items():
            if state.speed > 0.0:
                counts[name] += 1
    return {name: counts[name] / max(steps, 1) for name in ACTOR_NAMES}


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
