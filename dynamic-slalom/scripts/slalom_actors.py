#!/usr/bin/env python3
"""Where every moving body is, at every instant, and how it is posed.

WHAT IS SCRIPTED, STATED PLAINLY
---------------------------------
All seven bodies here walk declared routes at declared speeds.  Six of them
stage the five encounters the duck has to solve; the seventh crosses the depot
throughout so "positive clearance to every moving body" is a claim about a
working floor rather than an empty one.  **None of them reacts to the duck**,
none can be pushed by it, and none stops because it arrived — which is
deliberate.  Traffic that yielded would make every slalom claim vacuous: "the
duck predicted the gap and took it" would be true of a duck walking in a
straight line with its eyes shut.

WHAT IS *NOT* SCRIPTED IS ANYTHING THE DUCK KNOWS
---------------------------------------------------
The duck never reads this module.  It measures every body's range with the same
contact probe it measures everything with, their positions through the same
per-tick world state, their velocities by finite-differencing those positions,
and their visibility through the real head camera.  Its planner consumes those
measurements and nothing else, which is why the acceptance gate can COMPARE the
declared choreography against what the duck actually did rather than conflating
the two.

THE CROSSING TIMES ARE SOLVED, NOT CHOSEN
-------------------------------------------
Each crossing actor is declared by WHERE it crosses the duck's lane and WHEN it
gets there — ``cross_x`` and ``cross_t`` — and :func:`_solve_start` then finds
the departure time that puts it on the lane at that instant, by measuring the
arc length from its own start to its own ``y = 0`` crossing.  Nudging a start
time by hand until a run looked right would make the encounter geometry an
accident; solving it means moving an encounter is one number.

WHY EVERY ROUTE IS FILLETED, AND WHERE THE BENDS ARE
------------------------------------------------------
``slalom_route.Route`` replaces every interior corner with a circular arc, so
each walker's heading is continuous.  A cornered polyline turns its walker
through the whole corner in ONE control tick, which is a teleport of the body
axis and would make the duck's measured velocity jump.

The bends are placed at ``|y| >= 1.3``, well outside the duck's lane, for a
specific reason: the duck's predictor is a CONSTANT-VELOCITY model, so it is
genuinely wrong wherever somebody is turning.  Putting the bends off the lane
means the mispredictions happen where they are informative — they show up as
prediction error in the metrics — without making the safety-critical part of
each crossing unpredictable in a way the scenario, rather than the robot,
controls.

THE FOURTH ENCOUNTER IS THE ONE THAT FORCES A WAIT
----------------------------------------------------
``karl`` walks north at 0.30 m/s and ``dev`` pushes a cart south at 0.16 m/s,
crossing the lane 0.40 m apart in x and about a second apart in time.  At the
moment the duck must decide, karl is just north of the lane and dev just south
of nothing — both are within a corridor's reach of it, so **neither the left nor
the right corridor is predicted safe and the duck stops**.

It then resolves LEFT, and that too is geometry rather than a rule: karl is fast
and keeps going north until he is 1.6 m clear, while dev is slow and is still
descending through the south when the duck moves off.  The north therefore
vacates first.  A duck that had simply picked a favourite side would have taken
the right and walked into the cart.
"""

from __future__ import annotations

import math

import numpy as np

from slalom_cast import (
    ALL_NAMES,
    BOX_AHEAD_M,
    BY_NAME,
    CART_AHEAD_M,
)
from slalom_route import Route

STRIDE_HZ: float = 1.05

# Corner radius for the scripted routes.  Smaller than the module default
# because these routes have short legs, and ``slalom_route._build`` raises
# rather than silently leaving a hard vertex when a cutback does not fit.
# MEASURED: at 0.30 m the tightest corner in this cast fillets cleanly and the
# largest single-tick heading change any body makes is under 2 deg.
ACTOR_CORNER_RADIUS = 0.30

# How far north and south a crossing route runs.  Beyond the perimeter's own
# 2.85 m half-width would put a body inside a wall, and much shorter would let
# somebody appear from nowhere just as the duck arrived.
CROSS_EXTENT = 2.52
# Length of the perimeter legs before and after each crossing.  These exist so
# the traffic is CONTINUOUSLY MOVING rather than standing at the edge of the
# floor waiting for its cue: without them each crossing body walks for about a
# fifth of the run and is a mannequin either side of it.  They also put every
# route bend out at |y| >= 1.3, which is where the constant-velocity predictor is
# allowed to be wrong (see the module docstring).
PERIMETER_LEG = 1.60
# The lane the perimeter legs run along, 0.33 m clear of the wall.
PERIMETER_Y = 2.52


def _crossing_corners(cross_x: float, northbound: bool
                      ) -> tuple[tuple[float, float], ...]:
    """A crossing route: approach along the perimeter, cross, leave along it.

    ``northbound`` starts in the south and ends in the north.  The two turns on
    and off the perimeter are what give the route real bends to fillet, and they
    sit at ``|y| >= 1.3`` - off the duck's lane - for the reason in the module
    docstring.
    """
    s = -1.0 if northbound else 1.0
    return (
        (cross_x - PERIMETER_LEG - 0.44, s * PERIMETER_Y),
        (cross_x - 0.44, s * PERIMETER_Y),
        (cross_x - 0.26, s * 1.42),
        (cross_x, s * 0.18),
        (cross_x + 0.16, -s * 1.34),
        (cross_x + 0.34, -s * PERIMETER_Y),
        (cross_x + PERIMETER_LEG + 0.44, -s * PERIMETER_Y),
    )


def _arc_to_lane(route: Route, samples: int = 2000) -> float:
    """Arc length at which a route first crosses ``y = 0``.

    Measured on the FILLETED path rather than computed from the corner list,
    because the fillets shorten the route near every bend and a corner-list sum
    would not agree with the curve the body actually walks.
    """
    previous = None
    for index in range(samples + 1):
        s = route.length * index / samples
        position, _ = route.pose_at_arc(s)
        y = float(position[1])
        if previous is not None and (previous[1] < 0.0 <= y
                                     or previous[1] > 0.0 >= y):
            return float(previous[0])
        previous = (s, y)
    return route.length * 0.5


def _solve_start(corners, speed: float, cross_t: float) -> Route:
    """The route whose walker is on the duck's lane at exactly ``cross_t``.

    Built twice: once to measure the arc length to the lane, then again with the
    departure time that arrival implies.  Solving it means an encounter is moved
    by editing ``cross_t``, not by nudging a start time until a run looks right.
    """
    probe = Route("probe", tuple(corners), speed, radius=ACTOR_CORNER_RADIUS)
    lane_arc = _arc_to_lane(probe)
    return Route("", tuple(corners), speed, start_t=cross_t - lane_arc / speed,
                 radius=ACTOR_CORNER_RADIUS)


def _crossing(name: str, cross_x: float, cross_t: float, speed: float,
              northbound: bool) -> Route:
    route = _solve_start(_crossing_corners(cross_x, northbound), speed, cross_t)
    route.name = name
    return route


# WHERE AND WHEN EACH ENCOUNTER HAPPENS.  ``cross_x`` is where that body crosses
# the duck's lane and ``cross_t`` when it gets there.
#
# THE TIMES ARE SOLVED AGAINST THE DUCK'S MEASURED PROGRESS, NOT CHOSEN.
# ``tools/tune_phasing.py`` runs the REAL rollout and records when the duck's
# trunk first comes within its own engage lead of each crossing point; each
# ``cross_t`` below is that measured instant.  The duck's schedule is a
# consequence of its MEASURED 0.129 m/s cruise, its 0.097 m/s careful command,
# the 0.64 m of course each sidestep costs and the time it spends waiting - none
# of which can be predicted analytically without modelling the controller, which
# is the thing under test.
#
# A FIRST DRAFT GUESSED THESE FROM THE CRUISE SPEED ALONE AND IT FAILED
# MEASURABLY.  The duck reached x = -2.60 at 11.6 s while ``mara`` did not cross
# until 15.4 s, so it walked through the crossing point 3.8 s early, the 7.0 s
# prediction horizon never saw a conflict, and the encounter that eventually
# happened was a late surprise with a MEASURED -0.038 m overlap.  The symptom
# looked like a planner choosing the wrong side; the cause was a scenario whose
# timing was wishful.
ENCOUNTERS: dict[str, dict] = {
    "E1": {"cross_x": -2.62, "cross_t": 10.60},
    "E2": {"cross_x": -1.06, "cross_t": 24.60},
    "E3": {"cross_x": 0.36, "cross_t": 40.20},
    "E4": {"cross_x": 1.55, "cross_t": 53.40},
    "E5": {"cross_x": 2.74, "cross_t": 59.20},
}

ROUTES: dict[str, Route] = {
    # E1 -- from the SOUTH, so the south vacates and the duck should pass RIGHT.
    "mara": _crossing("mara", ENCOUNTERS["E1"]["cross_x"],
                      ENCOUNTERS["E1"]["cross_t"], 0.255, northbound=True),

    # E2 -- from the NORTH with a cart, so the duck should pass LEFT.  The
    # traffic cone at (0.55, 0.86) is NOT what prunes this one; the cone bites at
    # E3.  Here the cart's own 0.48 m planning radius is what closes the right.
    "tobin": _crossing("tobin", ENCOUNTERS["E2"]["cross_x"],
                       ENCOUNTERS["E2"]["cross_t"], 0.208, northbound=False),

    # E3 -- from the SOUTH carrying a box: pass RIGHT.  This is the encounter
    # where ``obs_cone_mid`` prunes the two WIDER left corridors, so the static
    # check bites on the real run rather than only under a test mutation.
    "ines": _crossing("ines", ENCOUNTERS["E3"]["cross_x"],
                      ENCOUNTERS["E3"]["cross_t"], 0.232, northbound=True),

    # E4 -- BOTH SIDES AT ONCE, AND THE ONE THAT FORCES A WAIT.
    #
    # The geometry is the whole point, so it is spelled out.  A NORTHBOUND body
    # is in the south before it crosses and in the north after, so the duck
    # passes behind it through the SOUTH (right).  A SOUTHBOUND body is the
    # mirror: the duck passes behind it through the NORTH (left).
    #
    # Here ``dev`` (southbound cart, slow) is still NORTH of the lane while
    # ``karl`` (northbound, fast) is still SOUTH of it, both close enough to
    # matter at the same instant.  The north is blocked by the cart the duck has
    # not yet let past and the south by the walker coming up into it, so
    # NEITHER corridor is predicted safe and the duck stops.
    #
    # It then resolves LEFT, and that is geometry rather than a rule: ``dev``
    # crosses first and continues south, which vacates the north, while ``karl``
    # is still working his way up through the south.  A duck with a favourite
    # side would have taken the right and walked into him.
    "dev": _crossing("dev", ENCOUNTERS["E4"]["cross_x"] - 0.24,
                     ENCOUNTERS["E4"]["cross_t"], 0.160,
                     northbound=False),
    "karl": _crossing("karl", ENCOUNTERS["E4"]["cross_x"] + 0.26,
                      ENCOUNTERS["E4"]["cross_t"] + 5.60, 0.300,
                      northbound=True),

    # E5 -- from the SOUTH carrying a box: pass RIGHT, in front of the arrival
    # band, so the last thing the video shows is a decision rather than a stroll.
    "noor": _crossing("noor", ENCOUNTERS["E5"]["cross_x"],
                      ENCOUNTERS["E5"]["cross_t"], 0.238, northbound=True),

    # BACKGROUND.  Walks the depot's north side for the whole run, well clear of
    # the lane, so it never confuses an encounter but is always a body the
    # clearance gate has to account for.
    "pilar": Route(
        "pilar",
        ((-4.30, 2.05), (-2.60, 1.90), (-0.80, 2.15), (1.20, 1.88),
         (3.10, 2.10), (4.20, 1.80)),
        0.112, start_t=1.0, radius=ACTOR_CORNER_RADIUS),
}

ACTOR_NAMES: tuple[str, ...] = tuple(ROUTES)


class ActorState:
    """One body's world state at one instant.  Positions, never percepts."""

    __slots__ = ("name", "pos", "yaw", "speed", "velocity", "kind", "encounter")

    def __init__(self, name: str, pos: np.ndarray, yaw: float, speed: float,
                 velocity: np.ndarray, kind: str, encounter: str):
        self.name = name
        self.pos = pos
        self.yaw = yaw
        self.speed = speed
        self.velocity = velocity
        self.kind = kind
        self.encounter = encounter


def actors_at(t: float) -> dict[str, ActorState]:
    """Every body's world state at ``t``, in cast order.

    The duck's planner never receives this dictionary.  It receives only what its
    own probes measured and what its camera could actually see.
    """
    states: dict[str, ActorState] = {}
    for name in ALL_NAMES:
        route = ROUTES[name]
        speed = route.speed_at(t)
        yaw = route.yaw_at(t)
        velocity = (np.array([math.cos(yaw), math.sin(yaw)]) * speed
                    if speed > 0.0 else np.zeros(2))
        spec = BY_NAME[name]
        states[name] = ActorState(name, route.pos_at(t), yaw, speed, velocity,
                                  spec.kind, spec.encounter)
    return states


def max_heading_step(seconds: float, dt: float = 0.02) -> tuple[float, str, float]:
    """Largest single-tick heading change any scripted body makes, in degrees.

    This is the property the filleted route buys and a cornered polyline does
    not have.  A test requires it to stay below what a walking person could do.
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
    """Fraction of the rollout each scripted body spends actually moving."""
    counts = {name: 0 for name in ACTOR_NAMES}
    steps = int(seconds / dt)
    for index in range(steps):
        for name, state in actors_at(index * dt).items():
            if state.speed > 0.0:
                counts[name] += 1
    return {name: counts[name] / max(steps, 1) for name in ACTOR_NAMES}


def lane_crossings(seconds: float, dt: float = 0.02) -> list[dict]:
    """When each body actually crosses the duck's lane, MEASURED from the routes.

    Reported rather than assumed so a test can require the solved ``cross_t`` to
    have produced the crossing it was solved for, and so the metrics can publish
    the choreography beside what the duck did about it.
    """
    out: list[dict] = []
    previous = {n: float(s.pos[1]) for n, s in actors_at(0.0).items()}
    for index in range(1, int(seconds / dt) + 1):
        t = index * dt
        for name, state in actors_at(t).items():
            y = float(state.pos[1])
            if previous[name] < 0.0 <= y or previous[name] > 0.0 >= y:
                out.append({
                    "actor": name,
                    "encounter": BY_NAME[name].encounter,
                    "t_s": round(t, 3),
                    "x_m": round(float(state.pos[0]), 4),
                    "northbound": y > previous[name],
                })
            previous[name] = y
    return out


def pose_actors(model, data, actors: dict[str, ActorState], t: float) -> None:
    """Write mocap poses and animate the gait.  Kinematic scenery, no contacts.

    THE GAIT IS ANIMATED EVEN WHEN SOMEBODY IS STANDING STILL, at a low
    amplitude and rate.  A body frozen mid-stride reads as a mannequin, and the
    requirement is that the traffic is continuously animated.

    A body pushing a cart or carrying a box swings its arms LESS, because both
    hands are occupied.  That is not decoration: the arms are real geoms the
    clearance probe measures against, so a carrier whose arms swung freely would
    be a wider body than the one the viewer sees holding a box.
    """
    for order, name in enumerate(ALL_NAMES):
        actor = actors[name]
        spec = BY_NAME[name]
        body = model.body(f"actor_{name}")
        mocap = int(model.body_mocapid[body.id])
        phase = t + 0.47 * order
        data.mocap_pos[mocap, :2] = actor.pos
        data.mocap_pos[mocap, 2] = spec.origin_z + 0.008 * abs(
            math.sin(2.0 * math.pi * STRIDE_HZ * phase))
        data.mocap_quat[mocap] = np.array(
            [math.cos(actor.yaw / 2.0), 0.0, 0.0, math.sin(actor.yaw / 2.0)])
        if actor.speed > 1e-3:
            amplitude = math.radians(18.0 + 78.0 * min(actor.speed, 0.30))
            rate = STRIDE_HZ
        else:
            amplitude = math.radians(3.6)
            rate = 0.31
        stride = amplitude * math.sin(2.0 * math.pi * rate * phase)
        # Hands full: a carrier's arms barely move.
        arm_scale = 0.18 if spec.kind in ("cart", "box") else 0.62
        swing = {
            f"{name}_hip_l": stride, f"{name}_hip_r": -stride,
            f"{name}_shoulder_l": -arm_scale * stride,
            f"{name}_shoulder_r": arm_scale * stride,
        }
        for joint_name, value in swing.items():
            joint = model.joint(joint_name).id
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.qvel[int(model.jnt_dofadr[joint])] = 0.0


def route_records() -> list[dict]:
    """Every scripted route, for the metrics to publish beside the duck's."""
    records = []
    for name in ACTOR_NAMES:
        record = ROUTES[name].as_record()
        record["kind"] = BY_NAME[name].kind
        record["encounter"] = BY_NAME[name].encounter
        records.append(record)
    return records
