#!/usr/bin/env python3
"""Turning the world into what the duck MEASURED: the boundary module.

Everything above this line (the machine, the detector, the controller) sees only
a :class:`gest_episode.Sense` and a :class:`gest_control.Interlock`; everything
below it is the simulator.  If a quantity is not built here, the decision layers
cannot reach it - which is how "the duck never read the choreography" is
enforced rather than promised.  ``tests/test_rollout_and_hygiene.py`` parses the
import graph with ``ast`` and fails if ``gest_machine``, ``gest_detect``,
``gest_gesture``, ``gest_pose`` or ``gest_control`` ever imports
``gest_actors`` or ``gest_script``.

EVERY FIELD IS SOMETHING A ROBOT COULD HAVE OBTAINED
------------------------------------------------------
* positions come from the same per-tick world state its contact probe measures
  against, labelled by MuJoCo body id - a **semantic proxy** for person
  recognition, not an RGB classifier;
* visibility and arm readability come from the real head camera: frustum
  containment plus real MuJoCo occlusion ray casts;
* the arm features come from the world positions of real keypoint bodies;
* ranges, yaw deltas and heading-projected displacements are quantities the duck
  could have measured from its own odometry.

THE TURN AND REVERSE PROGRESS ARE MEASURED HERE, WHICH IS WHY THEY ARE HONEST
------------------------------------------------------------------------------
:func:`turn_progress_deg` and :func:`back_progress_m` take the pose the duck
holds NOW and the pose it held when the command was accepted, and return the
trunk-yaw delta and the displacement projected on the pre-action heading.
Neither can be satisfied by a command register or a state name.
"""

from __future__ import annotations

import math

import numpy as np

from gest_arena import (
    FLOOR_HALF,
    inside_area,
    occluder_between,
    static_gap,
)
from gest_cast import ALL_NAMES
from gest_control import Interlock
from gest_states import (
    DUCK_PLANAR_RADIUS,
    GESTURE_MAX_RANGE_M,
    SETTLED_MPS,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
)

# Planar radius each person is treated as when deciding whether they BLOCK a
# sightline.  The widest body's measured lateral half-width rounded up, because
# a body seen edge-on still screens what is behind it.
BODY_OCCLUDER_RADIUS_M = 0.14
# How far ahead the interlock looks when refusing to advance.  Far enough that
# the duck stops BEFORE a person rather than beside one.
INTERLOCK_LOOKAHEAD_M = 0.34
# The surface separation at which the interlock refuses.  Above the duck's own
# planar radius, so the refusal happens with room to spare rather than at the
# point of contact.  It is deliberately BELOW the standoff band's lower edge of
# 0.45 m: the interlock is a backstop for a failure of the approach controller,
# not the thing that normally stops the duck.
INTERLOCK_CLEARANCE_M = 0.30
# How close to the perimeter wall the duck may be driven.  The reverse leg is
# what this guards: backing up is the one action whose direction is away from
# what the camera is watching.
AREA_MARGIN_M = DUCK_PLANAR_RADIUS + 0.12


def measured_positions(bodies) -> dict[str, np.ndarray]:
    """Every present person's position, as the duck's own sensing would report.

    A person who has not entered yet is omitted entirely rather than reported at
    a parking position: the duck cannot see them, so they do not exist as far as
    every layer above this one is concerned.
    """
    return {name: np.asarray(bodies[name].pos, dtype=np.float64)[:2]
            for name in ALL_NAMES if bodies[name].present}


def measured_yaws(bodies) -> dict[str, float]:
    """Every present person's facing.

    Part of the pose SEMANTIC PROXY and stated as such: a real system would get
    this from the same estimator that gives it the keypoints, and the gesture
    features are expressed in this frame so that the instructor's "left" is a
    property of the instructor rather than of where the robot is standing.
    """
    return {name: float(bodies[name].yaw)
            for name in ALL_NAMES if bodies[name].present}


def range_to(duck_xy, position) -> float:
    """Planar centre-to-centre range from the duck to one body."""
    return float(np.linalg.norm(
        np.asarray(position, dtype=np.float64)[:2]
        - np.asarray(duck_xy, dtype=np.float64)[:2]))


def ranges_from(duck_xy, positions: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: range_to(duck_xy, pos) for name, pos in positions.items()}


def turn_progress_deg(duck_yaw: float, reference_yaw: float) -> float:
    """Trunk-yaw delta since the command was accepted, in degrees.

    Signed and wrapped, so a left turn reports positive and a right turn
    negative.  THE ONLY quantity a turn gate should ever be graded on.
    """
    return math.degrees(
        math.atan2(math.sin(duck_yaw - reference_yaw),
                   math.cos(duck_yaw - reference_yaw)))


def back_progress_m(duck_xy, reference_xy, reference_yaw: float) -> float:
    """Displacement BACKWARD along the pre-action heading, in metres.

    Positive means the duck genuinely retreated relative to the way it was
    facing when the reverse was ordered.  Distance travelled is deliberately not
    used: the MEASURED reverse gait drifts -50 deg in 6 s, so a robot that
    curled sideways would rack up path without going backwards.
    """
    delta = (np.asarray(duck_xy, dtype=np.float64)[:2]
             - np.asarray(reference_xy, dtype=np.float64)[:2])
    heading = np.array([math.cos(reference_yaw), math.sin(reference_yaw)])
    return float(-(delta @ heading))


def forward_progress_m(duck_xy, reference_xy, reference_yaw: float) -> float:
    """Displacement FORWARD along the pre-action heading.  The mirror of above."""
    return -back_progress_m(duck_xy, reference_xy, reference_yaw)


def in_standoff_band(clearance_m: float) -> bool:
    """Is the MEASURED SURFACE clearance inside the required band?

    Graded on the contact probe's own surface number - not on a centre-to-centre
    range, and not on a planned point.  The two differ by both bodies' radii,
    which is about 0.3 m here: enough to put a duck that is correctly inside the
    band outside a window drawn in the wrong units.
    """
    return bool(STANDOFF_MIN_M <= clearance_m <= STANDOFF_MAX_M)


def los_blocked_by(eye_xy, target_xy, bodies, exclude: str = "",
                   margin: float = 0.0) -> str:
    """Name of whatever stands in the planar sightline to a point.

    Static geometry first, then OTHER PEOPLE.  Both are returned as a single
    "line of sight did not exist" answer, because the visibility gate excludes
    exactly the ticks where seeing the target was geometrically impossible - and
    a person in the way makes it just as impossible as a rack does.  Holding the
    duck responsible for not seeing through somebody would grade the layout
    rather than the robot.
    """
    static = occluder_between(eye_xy, target_xy, margin)
    if static is not None:
        return static
    eye = np.asarray(eye_xy, dtype=np.float64)[:2]
    target = np.asarray(target_xy, dtype=np.float64)[:2]
    span = target - eye
    length = float(np.linalg.norm(span))
    if length < 1e-9:
        return ""
    direction = span / length
    for name in ALL_NAMES:
        if name == exclude or not bodies[name].present:
            continue
        offset = np.asarray(bodies[name].pos, dtype=np.float64)[:2] - eye
        along = float(offset @ direction)
        if along <= 0.08 or along >= length - 0.08:
            continue
        lateral = abs(float(offset[0] * direction[1]
                            - offset[1] * direction[0]))
        if lateral <= BODY_OCCLUDER_RADIUS_M + margin:
            return name
    return ""


def build_interlock(*, duck_xy, duck_yaw: float, bodies,
                    clearances: dict[str, float], state: str,
                    target_xy=None) -> Interlock:
    """The independent refusal to walk into somebody, or out of the area.

    Computed from THIS TICK's MEASURED surface clearance, the person's bearing
    and the duck's own distance to the boundary - never from the detector or the
    machine.  The executor reasons about carrying out a command; this reasons
    about where the duck is right now.

    A REFUSAL MUST NOT BE A TRAP, AND ON THIS ROBOT THAT IS A REAL CONSTRAINT.
    Turning in place is MEASURED to be unavailable - at most 1.6 deg/s at
    ``vx = 0`` - so a duck whose command is refused cannot turn away either: the
    yaw it needs comes from walking.  Both refusals are therefore about the
    DIRECTION OF TRAVEL rather than the pose: they fire only when continuing
    would make the situation worse, and moving away is always allowed.

    THE REVERSE DIRECTION IS HANDLED EXPLICITLY, because it is the one action
    whose direction of travel is opposite to the duck's heading.  A proximity
    check written only for forward motion would refuse a reverse that was
    moving AWAY from the person - and, being unable to turn, the duck would then
    stand there until the ceiling fired.
    """
    duck = np.asarray(duck_xy, dtype=np.float64)[:2]
    heading = np.array([math.cos(duck_yaw), math.sin(duck_yaw)])
    reversing = state == "EXECUTE_BACK_UP"
    travel = -heading if reversing else heading

    # THE AREA BOUNDARY, checked one lookahead along the ACTUAL direction of
    # travel.  Reversing toward a wall is refused; reversing away from one is
    # not, which is the same escape rule the proximity check uses below and for
    # the same reason: a robot that cannot turn in place must always be allowed
    # to improve its situation.
    ahead = duck + travel * INTERLOCK_LOOKAHEAD_M
    if not inside_area(ahead, AREA_MARGIN_M) \
            and _boundary_gap(ahead) < _boundary_gap(duck):
        return Interlock(
            True,
            f"the next {INTERLOCK_LOOKAHEAD_M:.2f} m would leave the "
            "training area",
            "")

    for name, gap in sorted(clearances.items(), key=lambda kv: kv[1]):
        if gap >= INTERLOCK_CLEARANCE_M or not bodies[name].present:
            continue
        offset = np.asarray(bodies[name].pos, dtype=np.float64)[:2] - duck
        # Only somebody in the direction of TRAVEL can be a reason to stop.
        if float(offset @ travel) <= 0.0:
            continue
        if float(np.linalg.norm(offset)) > INTERLOCK_LOOKAHEAD_M + 0.9:
            continue
        if target_xy is not None and not reversing:
            span = np.asarray(target_xy, dtype=np.float64)[:2] - duck
            if float(np.linalg.norm(span)) > 1e-9 \
                    and float(offset @ (span / np.linalg.norm(span))) <= 0.0:
                continue
        return Interlock(
            True,
            f"{name} is {gap:.3f} m away in the direction of travel; below "
            f"the {INTERLOCK_CLEARANCE_M:.2f} m bar",
            name)
    return Interlock(False, "", "")


def _boundary_gap(xy) -> float:
    """Planar gap from a point to the nearest perimeter wall's inner face."""
    point = np.asarray(xy, dtype=np.float64)[:2]
    return float(min(FLOOR_HALF[0] - abs(float(point[0])),
                     FLOOR_HALF[1] - abs(float(point[1]))))


def build_sense(*, detector_view: dict, confirmed: dict | None,
                instructor_visible: bool, arm_readable: bool,
                instructor_range_m: float, measured_speed_mps: float,
                duck_yaw: float, reference_yaw: float, reference_xy,
                duck_xy, instructor_clearance_m: float, stop_hold_s: float,
                measured_min_clearance_m: float):
    """Everything the duck measured and concluded this tick, as one object."""
    from gest_episode import Sense

    return Sense(
        locked=detector_view.get("locked", ""),
        acquisition_state=detector_view.get("acquisition_state", "search"),
        instructor_visible=bool(instructor_visible),
        arm_readable=bool(arm_readable),
        instructor_range_m=float(instructor_range_m),
        in_gesture_range=bool(instructor_range_m <= GESTURE_MAX_RANGE_M),

        candidate_command=detector_view.get("candidate_command", ""),
        candidate_held_s=float(detector_view.get("candidate_held_s", 0.0)),
        candidate_fraction=float(detector_view.get("candidate_fraction", 0.0)),
        candidate_confidence=float(
            detector_view.get("candidate_confidence", 0.0)),
        confirm_progress=float(detector_view.get("confirm_progress", 0.0)),
        confirmed=confirmed,

        measured_speed_mps=float(measured_speed_mps),
        settled=bool(measured_speed_mps <= SETTLED_MPS),
        duck_yaw_deg=math.degrees(float(duck_yaw)),
        yaw_delta_deg=turn_progress_deg(duck_yaw, reference_yaw),
        back_along_heading_m=back_progress_m(duck_xy, reference_xy,
                                             reference_yaw),
        range_to_instructor_m=float(instructor_range_m),
        in_standoff_band=in_standoff_band(instructor_clearance_m),
        stop_hold_s=float(stop_hold_s),

        measured_min_clearance_m=float(measured_min_clearance_m),
        inside_area=bool(inside_area(duck_xy, DUCK_PLANAR_RADIUS)),
    )


def duck_static_gap(duck_xy) -> tuple[str, float]:
    """Nearest static surface and the planar gap to it, for reporting."""
    return static_gap(duck_xy)
