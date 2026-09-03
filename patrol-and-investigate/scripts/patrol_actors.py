#!/usr/bin/env python3
"""Where every body is at every instant, and how it is posed.

WHAT IS SCRIPTED, STATED PLAINLY
---------------------------------
The three staff walk declared routes at declared speeds for the whole run, so
"the facility was populated" is a working floor rather than an empty one.  The
intruder walks a declared route into the restricted annex and stops there.  The
two objects are placed by a declared APPEARANCE TIME - before it they are parked
below the floor, out of every camera - which is how a facility gains an
unattended crate partway through a patrol.  **None of them reacts to the duck.**

WHAT IS *NOT* SCRIPTED IS ANYTHING THE DUCK KNOWS
---------------------------------------------------
The duck never reads this module.  It measures every body's position through the
same per-tick world state its contact probe uses, differentiates those
measurements into velocities, sees them through the real head camera, and
decides what they are from geometry it measured itself.

THE APPEARANCE TIMES ARE SOLVED AGAINST THE DUCK'S OWN MEASURED PROGRESS
--------------------------------------------------------------------------
Each anomaly must become present while the duck is somewhere it can plausibly
find it, and - crucially - while it is NOT already investigating something else.
``tools/tune_timing.py`` runs the REAL rollout and reports when the duck reaches
each checkpoint; the times below are read off that measurement.  Guessing them
from the cruise speed alone fails the same way it failed in the sibling
behavior: the duck's schedule is a consequence of its MEASURED 0.129 m/s cruise,
its checkpoint stops, its scans and the time each investigation takes, none of
which can be predicted without modelling the controller that is under test.

WHY EVERY ROUTE IS FILLETED
-----------------------------
``patrol_route.Route`` replaces every interior corner with a circular arc, so
each walker's heading is continuous.  A cornered polyline turns its walker
through the whole corner in ONE control tick, which is a teleport of the body
axis and would make the duck's measured velocity estimate jump.
"""

from __future__ import annotations

import math

import numpy as np

from patrol_cast import ALL_NAMES, BY_NAME, OBJECT_HEIGHT_M
from patrol_route import Route

STRIDE_HZ: float = 1.05

# Corner radius for the scripted routes.  Smaller than the module default
# because these routes have short legs on a compact floor, and
# ``patrol_route._build`` raises rather than silently leaving a hard vertex when
# a cutback does not fit.
ACTOR_CORNER_RADIUS = 0.26

# Where a body waits before it exists.  Far below the floor, so it is outside
# every camera frustum and every ray cast, and the duck genuinely cannot see it.
PARKED_Z = -3.0


# -- where the anomalies are, and when they appear ---------------------------
#
# THE PLACES ARE SOLVED, NOT CHOSEN.  ``tools/check_layout.py`` requires each
# anomaly to sit far enough outside the circuit that walking from the checkpoint
# it is found at to the planned standoff point REDUCES the range by at least
# 0.45 m.  That is the constraint that makes an approach a physical approach.
# A first draft put all three about 0.9 m from their checkpoints, which is
# inside the 0.92-0.98 m standoff range the planner asks for: the duck would
# have been told to walk to a point it was already standing on, and two of the
# three approaches would have been 0.04 m and -0.06 m long.
#
# ``crate`` sits in the open north-east bay, clear of the stow pallet and of the
# shelf stack, where the east-aisle post's scan sweeps.
#
# ``trolley`` sits ON the stow pallet - the rule that makes it benign - out in
# the south-east, where the dock-gate post's scan sweeps.  ``emil`` is routed to
# stand beside it, which is the second rule.
#
# ``visitor`` walks into the restricted annex and stays there, in view of the
# server-door post.
CRATE_XY = (2.15, 0.72)
CRATE_APPEARS_S = 10.0
TROLLEY_XY = (1.16, -1.86)
TROLLEY_APPEARS_S = 3.0

# The intruder's route: in from the north-west, into the annex, and a stop.
VISITOR_SPEED = 0.26
VISITOR_START_S = 30.0

ROUTES: dict[str, Route] = {
    # -- staff, moving for the whole run --------------------------------
    # Rosa works the west and north aisles.  Her route is deliberately kept
    # clear of the north-east bay for the whole run: the crate's "nobody is
    # near it" rule is a MEASUREMENT the duck takes, and a member of staff who
    # wandered past it would correctly make it benign and destroy the scenario.
    # ``tools/check_layout.py`` measures the closest anybody comes to the crate
    # and fails below the attendance radius - it caught an earlier route that
    # brought her within 0.449 m of it at 16.8 s.
    "rosa": Route(
        "rosa",
        ((0.35, 1.92), (-0.20, 1.98), (-1.30, 1.85), (-1.70, 0.35),
         (-1.12, -0.85), (-1.05, -1.70), (-0.10, -1.98), (0.75, -1.78)),
        0.136, start_t=0.5, radius=ACTOR_CORNER_RADIUS),

    # Emil walks in from the west and then STANDS BESIDE THE TROLLEY, which is
    # the measurement that makes it benign.  The hold window is what puts him
    # there, and the route ENDS at the standing position rather than passing
    # through it: an earlier draft held him mid-route and left him 2.17 m away,
    # which the layout probe caught.  He stops 0.55 m from the trolley - inside
    # the 0.90 m attendance radius, outside the pallet itself.
    "emil": Route(
        "emil",
        ((-1.30, -1.95), (-0.30, -2.00), (0.55, -1.95), (1.14, -1.32)),
        0.112, start_t=1.0, radius=ACTOR_CORNER_RADIUS,
        hold_windows=((26.0, 260.0),)),

    # Nadia works the west side.  She walks NEAR the restricted zone without
    # ever entering it, which is what stops "somebody was on the west side" from
    # being the thing that triggers the intrusion call.
    #
    # HER ROUTE WAS SOLVED AGAINST THE FIXTURES, NOT DRAWN.  Three successive
    # hand-drawn lines clipped ``obs_zone_post_nw`` by 0.034 m, ``obs_shelf_sw``
    # by 0.013 m and ``obs_column_w`` by 0.047 m - each caught by
    # ``tools/check_layout.py`` and invisible by eye.  The line below threads
    # the aisle between the zone stanchions, the west column and the south-west
    # shelf with a MEASURED 0.217 m of clearance at its worst point.
    "nadia": Route(
        "nadia",
        ((-2.58, 0.20), (-1.30, 0.25), (-1.18, -0.90), (-1.30, -1.86),
         (-0.40, -1.95), (0.30, -1.85)),
        0.104, start_t=1.2, radius=ACTOR_CORNER_RADIUS),

    # -- the intruder ------------------------------------------------------
    # In from the north-west corner, into the marked annex, and a stop.  The
    # hold is what makes it an intrusion rather than somebody walking past: a
    # person who crossed the rectangle and kept going would be a transient, and
    # the duck's own dwell rule would correctly decline to call it.
    "visitor": Route(
        "visitor",
        ((-2.75, 2.05), (-2.45, 1.70), (-2.16, 1.40), (-2.10, 1.15)),
        VISITOR_SPEED, start_t=VISITOR_START_S, radius=0.18),
}

# The two objects do not walk; they APPEAR.  Kept in the same shape as a route
# so ``bodies_at`` has one code path.
APPEARANCES: dict[str, dict] = {
    "crate": {"xy": CRATE_XY, "at_s": CRATE_APPEARS_S, "yaw_deg": 18.0},
    "trolley": {"xy": TROLLEY_XY, "at_s": TROLLEY_APPEARS_S, "yaw_deg": -6.0},
}

ACTOR_NAMES: tuple[str, ...] = tuple(ROUTES) + tuple(APPEARANCES)


class BodyState:
    """One body's world state at one instant.  Positions, never percepts."""

    __slots__ = ("name", "pos", "yaw", "speed", "velocity", "kind", "present")

    def __init__(self, name: str, pos: np.ndarray, yaw: float, speed: float,
                 velocity: np.ndarray, kind: str, present: bool):
        self.name = name
        self.pos = pos
        self.yaw = yaw
        self.speed = speed
        self.velocity = velocity
        self.kind = kind
        self.present = present


def bodies_at(t: float) -> dict[str, BodyState]:
    """Every body's world state at ``t``, in cast order.

    A body that has not appeared yet is reported at :data:`PARKED_Z` and marked
    ``present=False``.  The duck's own perception never special-cases that: a
    parked body is simply outside every frustum and every ray cast, so it cannot
    be seen, which is exactly what "it is not there yet" should mean.
    """
    states: dict[str, BodyState] = {}
    for name in ALL_NAMES:
        spec = BY_NAME[name]
        if name in ROUTES:
            route = ROUTES[name]
            speed = route.speed_at(t)
            yaw = route.yaw_at(t)
            planar = route.pos_at(t)
            present = t >= route.start_t
            velocity = (np.array([math.cos(yaw), math.sin(yaw)]) * speed
                        if speed > 0.0 else np.zeros(2))
            position = (np.array([planar[0], planar[1]]) if present
                        else np.array([planar[0], planar[1]]))
        else:
            entry = APPEARANCES[name]
            present = t >= entry["at_s"]
            position = np.asarray(entry["xy"], dtype=np.float64)
            yaw = math.radians(entry["yaw_deg"])
            speed = 0.0
            velocity = np.zeros(2)
        states[name] = BodyState(name, position, yaw, speed, velocity,
                                 spec.kind, present)
    return states


def present_at(name: str, t: float) -> bool:
    """Is this body physically in the facility at ``t``?"""
    if name in APPEARANCES:
        return t >= APPEARANCES[name]["at_s"]
    return t >= ROUTES[name].start_t


def appearance_times() -> dict[str, float]:
    """When each body first becomes present, for the metrics to publish."""
    out = {name: float(entry["at_s"]) for name, entry in APPEARANCES.items()}
    out.update({name: float(route.start_t) for name, route in ROUTES.items()})
    return out


def pose_bodies(model, data, states: dict[str, BodyState], t: float) -> None:
    """Write mocap poses and animate the gait.  Kinematic scenery, no contacts.

    A PERSON'S GAIT IS ANIMATED EVEN WHEN THEY ARE STANDING STILL, at a low
    amplitude and rate: a body frozen mid-stride reads as a mannequin, and the
    requirement is that the population is continuously animated.  An OBJECT does
    not move at all, which is what makes "it has been stationary for six
    seconds" a measurement the duck can take rather than an artefact of noise.
    """
    for order, name in enumerate(ALL_NAMES):
        state = states[name]
        spec = BY_NAME[name]
        body = model.body(f"actor_{name}")
        mocap = int(model.body_mocapid[body.id])

        if not state.present:
            data.mocap_pos[mocap] = (float(state.pos[0]), float(state.pos[1]),
                                     PARKED_Z)
            data.mocap_quat[mocap] = np.array([1.0, 0.0, 0.0, 0.0])
            continue

        phase = t + 0.47 * order
        data.mocap_pos[mocap, :2] = state.pos
        if spec.is_person:
            data.mocap_pos[mocap, 2] = spec.origin_z + 0.008 * abs(
                math.sin(2.0 * math.pi * STRIDE_HZ * phase))
        else:
            data.mocap_pos[mocap, 2] = 0.0
        data.mocap_quat[mocap] = np.array(
            [math.cos(state.yaw / 2.0), 0.0, 0.0, math.sin(state.yaw / 2.0)])

        if not spec.is_person:
            continue
        if state.speed > 1e-3:
            amplitude = math.radians(18.0 + 78.0 * min(state.speed, 0.30))
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


def max_heading_step(seconds: float, dt: float = 0.02) -> tuple[float, str, float]:
    """Largest single-tick heading change any scripted PERSON makes, in degrees.

    This is the property a filleted route buys and a cornered polyline does not.
    Objects are excluded because they never turn; including them would average
    the number down and make the gate weaker.
    """
    worst = (0.0, "", 0.0)
    previous = {n: s.yaw for n, s in bodies_at(0.0).items()}
    for index in range(1, int(seconds / dt) + 1):
        t = index * dt
        for name, state in bodies_at(t).items():
            if not BY_NAME[name].is_person:
                continue
            delta = abs(math.degrees(math.atan2(
                math.sin(state.yaw - previous[name]),
                math.cos(state.yaw - previous[name]))))
            if delta > worst[0]:
                worst = (delta, name, t)
            previous[name] = state.yaw
    return worst


def moving_fraction(seconds: float, dt: float = 0.10) -> dict[str, float]:
    """Fraction of the rollout each scripted body spends actually moving."""
    counts = {name: 0 for name in ALL_NAMES}
    steps = int(seconds / dt)
    for index in range(steps):
        for name, state in bodies_at(index * dt).items():
            if state.speed > 0.0:
                counts[name] += 1
    return {name: counts[name] / max(steps, 1) for name in ALL_NAMES}


def zone_occupancy(seconds: float, dt: float = 0.10) -> dict[str, float]:
    """Seconds each PERSON spends inside the restricted rectangle.

    Published beside the duck's own intrusion call so the claim "only one person
    entered the zone" is arithmetic over the choreography rather than a caption.
    """
    from patrol_facility import RESTRICTED_ZONE

    seconds_in = {name: 0.0 for name in ALL_NAMES}
    for index in range(int(seconds / dt)):
        for name, state in bodies_at(index * dt).items():
            if state.present and RESTRICTED_ZONE.contains(state.pos):
                seconds_in[name] += dt
    return {k: round(v, 2) for k, v in seconds_in.items()}


def route_records() -> list[dict]:
    """Every scripted route and appearance, for the metrics to publish."""
    records = []
    for name in ALL_NAMES:
        if name in ROUTES:
            record = ROUTES[name].as_record()
        else:
            entry = APPEARANCES[name]
            record = {"name": name, "corners": [list(entry["xy"])],
                      "length_m": 0.0, "speed_mps": 0.0,
                      "start_t_s": entry["at_s"], "hold_windows_s": [],
                      "finish_t_s": entry["at_s"], "bends": []}
        record["kind"] = BY_NAME[name].kind
        record["role"] = BY_NAME[name].role
        record["appears_at_s"] = appearance_times()[name]
        records.append(record)
    return records
