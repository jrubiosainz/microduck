#!/usr/bin/env python3
"""Turning the world into what the duck MEASURED: occupancy, clearance, order,
and the aperture interlock.

This module is the boundary.  Everything above it (the machine, the controller)
sees only a :class:`etiquette_machine.Sense` and an
:class:`etiquette_control.Interlock`; everything below it is the simulator.  If
a quantity is not built here, the decision layers cannot reach it - which is how
"the duck never read the choreography" is enforced rather than promised.

EVERY FIELD IS SOMETHING A ROBOT COULD HAVE OBTAINED
------------------------------------------------------
* positions and ranges come from the same per-tick world state its contact probe
  measures against, labelled by MuJoCo body id - a **semantic proxy** for person
  recognition, not an RGB classifier;
* door open fractions come from :mod:`lobby_doors`, which is the interface a real
  door-state sensor would present;
* order relative to the guardian is computed by projecting HER measured position
  onto the duck's OWN route, which is the only place "behind her on the same
  path" has a meaning.

WHY THE GUARDIAN GAP IS AN ARC LENGTH AND NOT A BEARING
---------------------------------------------------------
A body-frame test ("is she in front of me") calls the follower ahead every time
the leader turns a corner, and this route turns five times.  Projecting her onto
the duck's own filleted route and comparing arc lengths is stable through every
bend, and it is what makes "the duck never overtook her" a statement about the
shared path rather than about a camera angle.

WHY CLEARANCE IS COUNTED IN TWO WAYS
--------------------------------------
An exiter is CLEAR when they are both far enough past the aperture plane and far
enough off the duck's own route corridor.  Either alone is insufficient: somebody
who has stepped out of a doorway and then stopped dead in front of it is still
somebody to wait for, and somebody who is off to one side but still on the sill
is still in the opening.
"""

from __future__ import annotations

import numpy as np

from etiquette_cast import (
    DOOR_EXITER_NAMES,
    GUARDIAN,
    OCCUPANT_NAMES,
)
from etiquette_control import Interlock
from etiquette_machine import Sense
from etiquette_states import (
    CABIN_HOLD_RADIUS_M,
    DUCK_PLANAR_RADIUS,
    EXITER_CLEAR_M,
    EXITER_LATERAL_CLEAR_M,
    GUARDIAN_THROUGH_M,
    OCCUPANT_EXITED_M,
)
from etiquette_zones import (
    CABIN_HOLD_XY,
    DOOR_APERTURE,
    LIFT_APERTURE,
    LIFT_PASSAGE,
    REAR_APERTURE,
    WAIT_SIDE_XY,
    cabin_contains,
)
from lobby_doors import APERTURES
from lobby_layout import CABIN_X, CABIN_Y, occluder_between

# Planar radius each person is treated as when deciding whether they BLOCK a
# sightline.  The widest adult's MEASURED lateral half-width is 0.1071 m; this
# rounds it up, because a body seen edge-on still screens what is behind it.
PERSON_OCCLUDER_RADIUS_M = 0.12

# The three aperture boxes, keyed by the door that fills them.
APERTURE_BOX = {
    "concourse_door": DOOR_APERTURE,
    "lift_front": LIFT_APERTURE,
    "lift_rear": REAR_APERTURE,
}

# Half-width of the duck's own route corridor, used ONLY for the "is this person
# still in my way" test.  A person outside it is not blocking the duck even if
# they are near the door.  Derived from the duck's conservative radius plus a
# body, so it is generous in the direction of waiting longer.
ROUTE_CORRIDOR_HALF_M = DUCK_PLANAR_RADIUS + 0.22


def bodies_in_aperture(people, name: str, radius: float = 0.0) -> list[str]:
    """Everybody whose footprint is inside one aperture box right now."""
    box = APERTURE_BOX[name]
    return [person for person, state in people.items()
            if box.contains(state.pos, radius)]


def in_cabin(xy) -> bool:
    """Is a point inside the cabin's raw footprint (not the inset interior)?"""
    point = np.asarray(xy, dtype=np.float64)
    return bool(CABIN_X[0] <= float(point[0]) <= CABIN_X[1]
                and CABIN_Y[0] <= float(point[1]) <= CABIN_Y[1])


def past_plane(xy, name: str, sign: float = -1.0) -> float:
    """How far a point is beyond an aperture plane, in the exit direction.

    ``sign`` is -1 when leaving towards lower x (out of the doorway, out of the
    lift into the lobby) and +1 when leaving towards higher x (out of the rear
    doors).  Positive means through.
    """
    plane = float(APERTURES[name]["plane_x"])
    return sign * (float(np.asarray(xy, dtype=np.float64)[0]) - plane)


def distance_to_route(route, xy, arc_from: float, arc_to: float,
                      samples: int = 64) -> float:
    """Shortest distance from a point to a WINDOW of the duck's own route.

    Windowed rather than whole-route because a person standing near a later leg
    is not in the way of the leg the duck is walking now.
    """
    point = np.asarray(xy, dtype=np.float64)
    best = float("inf")
    lo = max(0.0, arc_from)
    hi = min(route.length, arc_to)
    if hi <= lo:
        return best
    for index in range(samples + 1):
        s = lo + (hi - lo) * (index / samples)
        position, _ = route.pose_at_arc(s)
        best = min(best, float(np.linalg.norm(position - point)))
    return best


def guardian_arc_on_duck_route(route, guardian_xy, samples: int = 900) -> float:
    """The guardian's arc length along the DUCK's route.

    Her nearest point on the path the duck is walking.  Unwindowed, because she
    can legitimately be far ahead, and monotonic behavior is not required of her
    - only of the duck's own cursor.

    PAST THE END OF THE ROUTE THE PROJECTION IS EXTENDED, NOT SATURATED.
    A plain nearest-point projection clamps at ``route.length``, so once the
    guardian walks beyond the duck's own arrival point her arc stops growing
    while the duck's catches up - and the gap collapses to exactly zero at the
    end of the run.  That is an artifact of the measurement, not an overtake:
    she is metres ahead, on the far side of the finish.  Measured on the first
    scheduled run it reported a minimum gap of +0.0000 m at 103.1 s and would
    have failed the never-overtook gate on a run where she was never passed.

    So beyond the final point the arc is continued along the route's own
    terminal tangent, which is the honest continuation of "how far along is
    she".
    """
    point = np.asarray(guardian_xy, dtype=np.float64)
    best_s, best_d = 0.0, float("inf")
    for index in range(samples + 1):
        s = route.length * index / samples
        position, _ = route.pose_at_arc(s)
        distance = float(np.linalg.norm(position - point))
        if distance < best_d:
            best_d, best_s = distance, s
    if best_s < route.length - 1e-9:
        return best_s
    end_point, end_tangent = route.pose_at_arc(route.length)
    overshoot = float((point - end_point) @ end_tangent)
    return route.length + max(0.0, overshoot)


def los_blocked_by(eye_xy, subject: str, people, margin: float = 0.0) -> str:
    """Name of whatever stands in the planar sightline to ``subject``.

    Static geometry first, then OTHER PEOPLE.  Both are returned as a single
    "line of sight did not exist" answer, because the visibility gate excludes
    exactly the ticks where seeing the subject was geometrically impossible - and
    a person in the way makes it just as impossible as a wall does.

    THIS FUNCTION EXISTS BECAUSE OF A MEASURED FAILURE.  With people excluded
    from the predicate, the guardian standing between the duck's head camera and
    the exiter it was waiting for counted as 323 ticks of the duck failing to
    look properly, and the monitoring visibility gate read 28.7%.  The duck was
    aiming correctly the whole time; there was a person in the way.  A body is an
    occluder, and pretending otherwise grades the robot for the scenario's
    geometry.

    Each person is treated as a disc of :data:`PERSON_OCCLUDER_RADIUS_M`, which
    is the widest adult's MEASURED lateral half-width rounded up.  The subject
    themselves is never their own occluder.
    """
    static = occluder_between(eye_xy, people[subject].pos, margin)
    if static is not None:
        return static
    eye = np.asarray(eye_xy, dtype=np.float64)[:2]
    target = np.asarray(people[subject].pos, dtype=np.float64)[:2]
    span = target - eye
    length = float(np.linalg.norm(span))
    if length < 1e-9:
        return ""
    direction = span / length
    for name, state in people.items():
        if name == subject:
            continue
        offset = np.asarray(state.pos, dtype=np.float64)[:2] - eye
        along = float(offset @ direction)
        # Only bodies BETWEEN the eye and the target can occlude.  The 0.08 m
        # standoff keeps a person standing essentially at the target from
        # counting as their own screen.
        if along <= 0.08 or along >= length - 0.08:
            continue
        lateral = abs(float(offset[0] * direction[1]
                            - offset[1] * direction[0]))
        if lateral <= PERSON_OCCLUDER_RADIUS_M + margin:
            return name
    return ""


def build_sense(*, t: float, duck_xy, route, arc_s: float, leg_end_m: float,
                people, doors) -> Sense:
    """Everything the duck measured this tick, as one object.

    ``arc_s`` is the duck's own monotonic cursor and ``leg_end_m`` the end of the
    leg its current state authorises, so ``route_remaining_m`` is distance to
    the holding point rather than to the far end of the building.
    """
    duck = np.asarray(duck_xy, dtype=np.float64)
    guardian = people[GUARDIAN.name]

    door = doors["concourse_door"]
    lift = doors["lift_front"]
    rear = doors["lift_rear"]

    # -- the doorway ---------------------------------------------------
    exiters_in_aperture = len(
        [n for n in bodies_in_aperture(people, "concourse_door", 0.0)
         if n in DOOR_EXITER_NAMES])
    # An exiter is PENDING while they have not yet cleared the opening in the
    # duck's direction.  Measured from their position alone.
    pending = 0
    for name in DOOR_EXITER_NAMES:
        through = past_plane(people[name].pos, "concourse_door", -1.0)
        off_route = distance_to_route(route, people[name].pos,
                                      arc_s - 0.2, arc_s + 2.4)
        if through < EXITER_CLEAR_M or off_route < EXITER_LATERAL_CLEAR_M:
            pending += 1
    all_exiters_clear = pending == 0 and exiters_in_aperture == 0

    # -- the guardian ---------------------------------------------------
    guardian_arc = guardian_arc_on_duck_route(route, guardian.pos)

    # -- the lift -------------------------------------------------------
    occupants_in_cabin = sum(1 for n in OCCUPANT_NAMES
                             if in_cabin(people[n].pos))
    occupants_in_passage = sum(
        1 for n in OCCUPANT_NAMES
        if LIFT_PASSAGE.contains(people[n].pos, 0.0)
        or LIFT_APERTURE.contains(people[n].pos, 0.0))
    occupants_exited = sum(
        1 for n in OCCUPANT_NAMES
        if past_plane(people[n].pos, "lift_front", -1.0) >= OCCUPANT_EXITED_M
        and not LIFT_APERTURE.contains(people[n].pos, 0.0))
    all_occupants_clear = (
        occupants_in_cabin == 0
        and occupants_in_passage == 0
        and all(past_plane(people[n].pos, "lift_front", -1.0)
                >= OCCUPANT_EXITED_M for n in OCCUPANT_NAMES))

    return Sense(
        route_remaining_m=max(0.0, leg_end_m - arc_s),
        # ARRIVING IS A TOLERANCE, NOT AN EQUALITY.  With the pursuit point
        # clamped to the leg's end the duck settles a few centimetres short and
        # then stops, because the residual pursuit vector falls under the
        # controller's own dead zone.  A test written as ``remaining <= 0`` never
        # fires; the first full rollout spent 70 s standing at holding points
        # waiting for phase ceilings because of exactly that.
        leg_arrived=bool((leg_end_m - arc_s) <= _leg_arrived_m()),
        at_door_threshold=bool(
            float(np.linalg.norm(duck - _door_hold())) <= _hold_radius()),
        at_lift_hold=bool(
            float(np.linalg.norm(duck - WAIT_SIDE_XY)) <= _hold_radius()),
        at_cabin_hold=bool(
            float(np.linalg.norm(duck - CABIN_HOLD_XY)) <= CABIN_HOLD_RADIUS_M),
        inside_cabin=cabin_contains(duck, DUCK_PLANAR_RADIUS),
        beyond_rear_m=past_plane(duck, "lift_rear", +1.0),

        door_open_fraction=door.fraction,
        door_passable=door.passable,
        exiters_in_aperture=exiters_in_aperture,
        exiters_pending=pending,
        all_exiters_clear=all_exiters_clear,

        guardian_through_door=past_plane(
            guardian.pos, "concourse_door", -1.0) <= -GUARDIAN_THROUGH_M,
        guardian_through_lift=past_plane(
            guardian.pos, "lift_front", -1.0) <= -GUARDIAN_THROUGH_M,
        guardian_inside_cabin=in_cabin(guardian.pos),
        guardian_through_rear=past_plane(
            guardian.pos, "lift_rear", +1.0) >= GUARDIAN_THROUGH_M,
        guardian_gap_m=float(guardian_arc - arc_s),

        lift_open_fraction=lift.fraction,
        lift_passable=lift.passable,
        occupants_exited=occupants_exited,
        occupants_in_cabin=occupants_in_cabin,
        occupants_in_passage=occupants_in_passage,
        all_occupants_clear=all_occupants_clear,
        rear_open_fraction=rear.fraction,
        rear_passable=rear.passable,
    )


def _door_hold():
    from etiquette_path import door_hold_xy
    return door_hold_xy()


def _hold_radius() -> float:
    from etiquette_control import HOLD_RADIUS_M
    return HOLD_RADIUS_M


def _leg_arrived_m() -> float:
    from etiquette_control import LEG_ARRIVED_M
    return LEG_ARRIVED_M


def build_interlock(*, duck_xy, people, doors, route, arc_s: float,
                    lookahead_m: float = 0.34) -> Interlock:
    """The independent refusal to advance into an aperture.

    Computed from THIS TICK's raw occupancy and door fractions, never from the
    state machine.  Three separate reasons, each of which alone holds the duck:

    * the aperture it is about to enter is not open far enough to pass;
    * somebody else's footprint is inside that aperture;
    * the guardian is inside it, which is the side-by-side case specifically.

    ``lookahead_m`` is how far ahead along the route the duck is considered to be
    "about to enter": far enough that it stops before the sill rather than in it.
    """
    duck = np.asarray(duck_xy, dtype=np.float64)
    ahead, _ = route.pose_at_arc(min(route.length, arc_s + lookahead_m))

    for name, box in APERTURE_BOX.items():
        entering = box.contains(ahead, DUCK_PLANAR_RADIUS) \
            or box.contains(duck, DUCK_PLANAR_RADIUS)
        if not entering:
            continue
        door = doors[name]
        if not door.passable:
            return Interlock(True, "aperture not open enough to pass", name)
        others = bodies_in_aperture(people, name, 0.0)
        if GUARDIAN.name in others:
            return Interlock(
                True, "the guardian is in this aperture; never abreast", name)
        if others:
            return Interlock(
                True, f"occupied by {', '.join(sorted(others))}", name)
    return Interlock(False, "", "")
