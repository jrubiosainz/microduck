#!/usr/bin/env python3
"""Who the duck watches, and what it walks at, given its state.  Pure geometry.

Split out of ``rollout_etiquette`` so the tick loop stays about ORDER and this
stays about WHO AND WHERE.  Nothing here steps physics, advances the machine or
touches MuJoCo, which is what lets every branch be unit-tested on hand-built
inputs.

THE SUBJECT IS THE PERSON THE CURRENT STATE IS WAITING ON
-----------------------------------------------------------
This is the behavior's acquisition claim, and it is deliberately narrow.  There
is no search, no scanning and no "interesting object" heuristic: in each state
there is exactly one person whose position the decision depends on, and that is
the person the head tracks.

* ``APPROACH_DOOR`` / ``YIELD_EXITERS`` - the exiter who is **least clear** of
  the doorway, because that is the one whose progress decides when the duck may
  move.  As each finishes clearing, the subject switches to the next, so the
  sequence of subjects is itself a measurement of the yield resolving.
* ``FOLLOW_THROUGH`` / ``APPROACH_LIFT`` / ``FOLLOW_GUARDIAN_IN`` /
  ``POSITION_INSIDE`` / ``FOLLOW_OUT`` - the **guardian**, because the duck's
  order relative to her is the constraint in all five.
* ``WAIT_SIDE`` / ``DOORS_OPEN`` / ``LET_OCCUPANTS_EXIT`` - the occupant who is
  **least clear** of the cabin, which is the last one out and therefore the one
  the duck must wait for.  Before anybody moves that is whoever is deepest in
  the car, which is also who a person would be watching.
* ``RIDE`` / ``DOORS_OPEN_TARGET`` - the **guardian**, who is in the car with
  it, and whose exit ends the wait.
* ``DONE`` - the guardian, arrived.

The ORDER these subjects are acquired in is recorded and gated:
``expected_subject_order`` states it, and the metrics require the run to have
produced exactly that sequence of distinct subjects.  A duck that watched the
right person by accident in one state and the wrong one in another would fail
that even if every other gate passed.

WHY THE HEAD SOMETIMES LOOKS AT A PLACE INSTEAD OF A PERSON
-------------------------------------------------------------
While crossing an aperture the duck looks THROUGH it - at the far side of the
opening it is walking into - because that is what the decision needs and what a
person does.  The subject whose visibility is GRADED is still the guardian in
those states; the aim point is simply not her body.  Keeping the two separable is
why :class:`Aim` carries both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from etiquette_cast import DOOR_EXITER_NAMES, GUARDIAN, OCCUPANT_NAMES
from etiquette_sense import past_plane
from lobby_doors import APERTURES

# States in which the head tracks the guardian.
GUARDIAN_STATES = ("FOLLOW_THROUGH", "APPROACH_LIFT", "FOLLOW_GUARDIAN_IN",
                   "POSITION_INSIDE", "RIDE", "DOORS_OPEN_TARGET",
                   "FOLLOW_OUT", "DONE")
# States in which it tracks whichever door exiter is least clear.
EXITER_STATES = ("APPROACH_DOOR", "YIELD_EXITERS")
# States in which it tracks whichever lift occupant is least clear.
OCCUPANT_STATES = ("WAIT_SIDE", "DOORS_OPEN", "LET_OCCUPANTS_EXIT")

# The aperture whose far side the head looks through, per state.  Absent means
# the head aims at the subject's body instead.
LOOK_THROUGH: dict[str, str] = {
    "FOLLOW_THROUGH": "concourse_door",
    "FOLLOW_GUARDIAN_IN": "lift_front",
    "FOLLOW_OUT": "lift_rear",
}
# How far beyond an aperture plane the look-through point sits.
LOOK_THROUGH_M = 0.90
# Height of a look-through aim point, at about an adult's chest.
LOOK_THROUGH_Z = 0.52


@dataclass
class Aim:
    """Where to walk, whom to watch, and where to point the head."""

    target_xy: np.ndarray | None = None
    kind: str = ""
    subject: str = ""
    look_at: np.ndarray | None = None
    remaining_m: float = 1e9
    cross_track_m: float = 0.0


def least_clear(people, names, aperture: str, sign: float) -> str:
    """Whichever of ``names`` is least far through ``aperture``.

    "Least clear" rather than "nearest": the person the duck is waiting on is
    the one who still has furthest to go, which on a shared exit path is not the
    same as the closest body.  Deterministic on ties by falling back to cast
    order, so the acquisition sequence is reproducible.
    """
    best, best_through = names[0], float("inf")
    for name in names:
        through = past_plane(people[name].pos, aperture, sign)
        if through < best_through - 1e-9:
            best, best_through = name, through
    return best


def subject_for(state: str, people) -> str:
    """The one person this state's decision depends on."""
    if state in EXITER_STATES:
        return least_clear(people, DOOR_EXITER_NAMES, "concourse_door", -1.0)
    if state in OCCUPANT_STATES:
        return least_clear(people, OCCUPANT_NAMES, "lift_front", -1.0)
    return GUARDIAN.name


def expected_subject_order() -> tuple[str, ...]:
    """The sequence of DISTINCT subjects a correct run produces.

    Stated here rather than derived from the run, so the metrics compare a
    measurement against a declaration instead of against itself.  The exiters
    and occupants appear as roles because which individual is least clear at a
    given instant is a measurement; what is claimed is the ROLE order.
    """
    return ("door_exiter", "guardian", "occupant", "guardian")


def role_of(name: str) -> str:
    if name == GUARDIAN.name:
        return "guardian"
    if name in DOOR_EXITER_NAMES:
        return "door_exiter"
    if name in OCCUPANT_NAMES:
        return "occupant"
    return "background"


def look_through_point(state: str) -> np.ndarray | None:
    """The far side of the aperture this state is crossing, if any."""
    aperture = LOOK_THROUGH.get(state)
    if aperture is None:
        return None
    spec = APERTURES[aperture]
    # The rear doors are crossed towards higher x; the other two towards it too,
    # since the whole route runs west to east.  Derived from the plane rather
    # than declared per state.
    return np.array([float(spec["plane_x"]) + LOOK_THROUGH_M,
                     float(spec["center_y"]), LOOK_THROUGH_Z])


def select(state: str, *, duck_xy, tracker, people) -> Aim:
    """The aim for this tick.

    ``tracker`` may be ``None`` before the route exists, which produces an empty
    aim and the controller's zero.
    """
    from etiquette_path import STATE_LEG

    subject = subject_for(state, people)
    look_at = look_through_point(state)

    if state not in STATE_LEG or tracker is None:
        return Aim(kind="holding", subject=subject, look_at=look_at,
                   remaining_m=0.0)

    tracker.project(duck_xy)
    return Aim(target_xy=tracker.pursuit_point(),
               kind="route_pursuit",
               subject=subject,
               look_at=look_at,
               remaining_m=tracker.remaining_m,
               cross_track_m=tracker.cross_track_m)


def bearing_to(duck_xy, target_xy) -> float:
    """World bearing from the duck to a point, in radians."""
    delta = (np.asarray(target_xy, dtype=np.float64)[:2]
             - np.asarray(duck_xy, dtype=np.float64)[:2])
    return math.atan2(float(delta[1]), float(delta[0]))
