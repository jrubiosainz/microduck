#!/usr/bin/env python3
"""Where every scripted person is, at every instant, and how they are posed.

WHAT IS SCRIPTED, STATED PLAINLY
---------------------------------
All eight people here walk declared routes at declared speeds.  The guardian's
route through both apertures and the cabin is the scenario; the two door exiters
and the three lift occupants are the traffic the duck has to yield to.  None of
them reacts to the duck, none of them can be pushed by it, and none of them
stops because the duck arrived — which is deliberate.  Traffic that waited for
the robot would make every etiquette claim vacuous: "the duck let them out
first" would be true of a duck that walked straight in.

WHAT IS *NOT* SCRIPTED IS ANYTHING THE DUCK KNOWS
---------------------------------------------------
The duck never reads this module.  It measures every person's range with the
same contact probe it measures everybody with, their positions through the same
per-tick world state, and their visibility through the real head camera.  Its
state machine consumes those measurements and nothing else, which is why the
acceptance gate can COMPARE the declared choreography against what the duck
actually did rather than conflating the two.

WHY EVERY ROUTE IS FILLETED AND CONTINUOUS
--------------------------------------------
``etiquette_route.Route`` replaces every interior corner with a circular arc, so
each walker's heading is continuous.  A cornered polyline turns its walker
through the whole corner in ONE control tick, which is a teleport of the body
axis and would make the duck's measured bearing to that person jump.  Nobody in
this scene turns faster than a person can.

THE TWO DOOR EXITERS USE OPPOSITE LANES OF THE SAME OPENING
-------------------------------------------------------------
``tomas`` leaves in the south lane and ``leila`` in the north lane of the 0.66 m
aperture, about 0.9 s apart.  That matters for one specific gate: the duck may
not be inside the aperture box while anybody else is, and with two people using
it in sequence there is no instant in the yield window when the opening is free.
A duck that "waited" only until the first of them was through would be caught.
"""

from __future__ import annotations

import math

import numpy as np

from etiquette_cast import ALL_NAMES, BY_NAME, GUARDIAN
from etiquette_route import Route
from lobby_layout import (
    CABIN_Y,
    DOOR_CENTER_Y,
    DOOR_WALL_X,
    LIFT_FRONT_Y,
    LIFT_WALL_X,
    REAR_WALL_X,
    REAR_Y,
)

STRIDE_HZ: float = 1.05

# Corner radius for the SCRIPTED PEOPLE's routes.  Smaller than the module
# default because these routes have short legs - the guardian steps round a jamb
# in under a metre - and ``etiquette_route._build`` silently leaves a corner as a
# hard vertex when its cutback does not fit the legs either side of it.  A hard
# vertex turns its walker through the whole corner in ONE control tick, which
# ``tools/check_layout.py`` catches as a 51 deg single-tick heading change and
# refuses.  MEASURED: at 0.28 m the tightest corner in this cast fillets cleanly
# and the largest single-tick turn any person makes is under 2 deg.
ACTOR_CORNER_RADIUS = 0.28

# Lane offsets inside the 0.66 m concourse aperture.  +/-0.15 m puts each
# exiter's centreline 0.15 m off the door axis, so the two of them use genuinely
# different halves of the opening rather than the same line at different times.
DOOR_LANE = 0.15

ROUTES: dict[str, Route] = {
    # THE GUARDIAN.  She walks the whole journey the duck must follow: up to the
    # automatic door, through it once the exiters are out, across the lobby,
    # into the cabin, and out through the rear doors at the target floor.
    #
    # HER TIMING IS SOLVED, NOT CHOSEN.  ``tools/tune_phasing.py`` checks, at
    # 20 Hz over the whole run, that her arc length on the DUCK'S OWN ROUTE
    # stays strictly ahead of the duck's and never more than
    # ``MAX_GUARDIAN_GAP_M`` ahead, that she is through each aperture before the
    # duck's leg into it begins, and - this one caught a real bug - that SHE
    # never walks through a closed door either.  An earlier draft had her
    # strolling through the sealed lift at 40.3 s and the sealed rear doors at
    # 69.7 s, which would have made the duck's own no-closed-doors gate look
    # arbitrary.  Her holds were then solved against the door schedule rather
    # than nudged.
    #
    # SHE WAITS BESIDE THE DOOR, NOT IN FRONT OF IT, AND THAT IS TWO MEASURED
    # FIXES AT ONCE.  A first draft had her waiting on the door's own axis, 0.24
    # m from the duck's holding point.  Two things broke.  The MEASURED surface
    # clearance between the two bodies went NEGATIVE - -0.0609 m over 163 control
    # steps - because a 0.116 m robot and a 0.114 m adult cannot both stand on
    # the same line.  And she sat squarely between the duck's head camera and the
    # doorway, so ``nadia_torso`` blocked the view of the exiters the duck was
    # waiting for in 323 ticks.  Starting her north of the axis fixes both, and
    # is what a person actually does at a door somebody else is coming through.
    #
    # THREE PHASES, EACH TIMED AGAINST THE DOOR SCHEDULE:
    #   1. she flows through the concourse door at 16.5 s, as the exiters finish,
    #      so she is never in it at the same time as them or as the duck;
    #   2. 32.1-47.5 s waiting in the lobby for the lift, which is what the duck
    #      is doing beside her.  She enters the lift aperture at 50.2 s, by which
    #      time the MEASURED open fraction is past 0.6;
    #   3. 59.5-82.1 s riding the sealed car, standing at (2.35, -0.30).  Without this her arc length would
    #      keep growing while the car was closed and she would walk out through
    #      a shut rear door - the very failure the duck is gated on.  She reaches
    #      the rear aperture at 88.0 s, after it opens at 86.8 s, and is 0.30 m
    #      through it shortly after - which is what the duck waits for in
    #      DOORS_OPEN_TARGET before following her out.
    #
    # WHERE SHE STANDS DURING THE RIDE IS A MEASUREMENT, NOT A POSE.  A draft
    # parked her at (1.68, 0.08), just inside the cabin door - which projects to
    # arc 5.746 on the DUCK's route, BEHIND the duck's own cabin holding point at
    # 6.009.  The measured gap went to -0.23 m and the never-overtook check
    # failed on a run where the duck passed nobody: she had simply stopped short
    # of it.  Riding at (2.35, -0.30) puts her at arc 6.813, a clear 0.80 m
    # ahead, on the south side of the car with the duck on the north - which is
    # also how two bodies actually share a lift.
    #
    # Her pace, 0.128 m/s, is just below the duck's MEASURED 0.133 m/s cruise -
    # so an unimpeded duck closes on her slowly rather than falling behind, and
    # the follow gap is bounded at both ends by the walking rather than by a
    # controller trying to hold station.
    "nadia": Route(
        "nadia",
        ((-1.95, 0.52), (-1.62, 0.56), (-1.34, 0.28),
         (DOOR_WALL_X, DOOR_CENTER_Y), (-0.42, 0.06), (0.55, 0.10),
         (LIFT_WALL_X, LIFT_FRONT_Y), (2.35, -0.30),
         (REAR_WALL_X, REAR_Y), (4.05, -0.50)),
        0.128, start_t=8.27, radius=ACTOR_CORNER_RADIUS, hold_windows=(
            (32.11, 47.48),
            (59.47, 82.13),
        )),

    # THE TWO DOOR EXITERS.  Both start EAST of the divider and walk WEST
    # through the opening, which is the whole point: they are coming towards the
    # duck, in its only way through.  They must be clear before the duck's yield
    # can end at 18.7 s, which ``tools/tune_phasing.py`` checks.
    "tomas": Route(
        "tomas",
        ((0.30, DOOR_CENTER_Y - DOOR_LANE - 0.05),
         (-0.55, DOOR_CENTER_Y - DOOR_LANE),
         (DOOR_WALL_X, DOOR_CENTER_Y - DOOR_LANE), (-1.85, -0.42),
         (-2.75, -0.85), (-3.85, -1.05)),
        0.200, start_t=4.40, radius=ACTOR_CORNER_RADIUS),
    "leila": Route(
        "leila",
        ((0.48, DOOR_CENTER_Y + DOOR_LANE + 0.05),
         (-0.45, DOOR_CENTER_Y + DOOR_LANE),
         (DOOR_WALL_X, DOOR_CENTER_Y + DOOR_LANE), (-1.90, 0.44),
         (-2.80, 0.86), (-3.85, 1.02)),
        0.194, start_t=5.20, radius=ACTOR_CORNER_RADIUS),

    # THE THREE LIFT OCCUPANTS.  Each starts INSIDE the cabin and walks out
    # through the front aperture into the lobby once the doors open at 48.8 s,
    # in order, then away.  They leave one at a time through a 0.72 m opening,
    # which is what gives "the last occupant cleared before the duck moved" a
    # tail long enough to measure.  All three must be clear before the duck
    # boards at 59.4 s.
    "priya": Route(
        "priya",
        ((2.30, 0.24), (LIFT_WALL_X, LIFT_FRONT_Y), (0.60, 0.30),
         (-0.30, 0.78), (-1.05, 1.35)),
        0.238, start_t=49.20, radius=ACTOR_CORNER_RADIUS),
    "marek": Route(
        "marek",
        ((2.55, -0.06), (LIFT_WALL_X, LIFT_FRONT_Y - 0.02), (0.55, -0.30),
         (-0.25, -0.85), (-1.05, -1.35)),
        0.230, start_t=50.00, radius=ACTOR_CORNER_RADIUS),
    "odile": Route(
        "odile",
        ((2.15, -0.42), (LIFT_WALL_X + 0.10, LIFT_FRONT_Y - 0.04),
         (0.62, 0.02), (-0.15, 0.52), (-0.95, 1.72)),
        0.224, start_t=50.80, radius=ACTOR_CORNER_RADIUS),

    # BACKGROUND.  Neither enters the cabin, neither uses the doorway during the
    # yield window, and both stay clear of the lift passage: a background walker
    # who wandered into the exit corridor would be indistinguishable from an
    # occupant and would make the occupant-clearance measurement ambiguous.
    # Their routes are long and slow enough to keep them walking for most of the
    # run, which is what makes "positive clearance to every person" a claim about
    # a populated building.
    "sami": Route(
        "sami",
        ((-3.85, 1.70), (-2.70, 1.15), (-2.05, 0.55), (-2.35, -1.15),
         (-3.55, -1.75), (-3.90, -0.60), (-3.20, 1.30)),
        0.104, radius=ACTOR_CORNER_RADIUS),
    "vera": Route(
        "vera",
        ((0.25, -1.95), (-0.60, -1.65), (-1.55, -1.25), (-2.20, -1.70),
         (-3.00, -1.95), (-3.60, -0.95), (-2.90, 0.40)),
        0.098, start_t=3.0, radius=ACTOR_CORNER_RADIUS),
}

ACTOR_NAMES: tuple[str, ...] = tuple(ROUTES)


class PersonState:
    """One person's world state at one instant.  Positions, never percepts."""

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
    """Every person's world state at ``t``, in cast order.

    The duck's controller never receives this dictionary.  It receives only what
    its own probes measure and what its camera can actually see.
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


def max_heading_step(seconds: float, dt: float = 0.02) -> tuple[float, str, float]:
    """Largest single-tick heading change any scripted person makes, in degrees.

    This is the property the filleted route buys and a cornered polyline does
    not have.  A test requires it to stay below what a walking person could do.
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
    """Fraction of the rollout each scripted person spends actually walking."""
    counts = {name: 0 for name in ACTOR_NAMES}
    steps = int(seconds / dt)
    for index in range(steps):
        for name, state in people_at(index * dt).items():
            if state.speed > 0.0:
                counts[name] += 1
    return {name: counts[name] / max(steps, 1) for name in ACTOR_NAMES}


def pose_people(model, data, people: dict[str, PersonState], t: float) -> None:
    """Write mocap poses and animate the gait.  Kinematic scenery, no contacts.

    THE GAIT IS ANIMATED EVEN WHEN SOMEBODY IS STANDING STILL, at a low
    amplitude and rate.  A person frozen mid-stride inside a lift for forty
    seconds reads as a mannequin, and the requirement is that the people are
    continuously animated.
    """
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


def route_records() -> list[dict]:
    """Every scripted route, for the metrics to publish beside the duck's."""
    return [ROUTES[name].as_record() for name in ACTOR_NAMES]
