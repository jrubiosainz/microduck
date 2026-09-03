#!/usr/bin/env python3
"""Turning the world into what the duck MEASURED: the boundary module.

Everything above this line (the machine, the branch, the controller, the
detector) sees only a :class:`patrol_episode.Sense` and a
:class:`patrol_control.Interlock`; everything below it is the simulator.  If a
quantity is not built here, the decision layers cannot reach it - which is how
"the duck never read the choreography" is enforced rather than promised.
``tests/test_rollout_and_hygiene.py`` parses the import graph with ``ast`` and
fails if ``patrol_machine``, ``patrol_branch``, ``patrol_detect`` or
``patrol_control`` ever imports ``patrol_actors``.

EVERY FIELD IS SOMETHING A ROBOT COULD HAVE OBTAINED
------------------------------------------------------
* positions come from the same per-tick world state its contact probe measures
  against, labelled by MuJoCo body id - a **semantic proxy** for object
  recognition, not an RGB classifier;
* visibility comes from the real head camera: frustum containment plus a real
  MuJoCo occlusion ray cast;
* the distance to the restricted zone is computed from the SAME rectangle the
  scene paints and the detector tests;
* ranges are planar distances the duck could have measured itself.

THE ZONE INTERLOCK IS THE ONE THAT MATTERS HERE
-------------------------------------------------
The duck has to investigate an intruder who is standing INSIDE an area the duck
itself may not enter.  That is the whole difficulty of the intrusion case, and
it is guarded twice, independently: the standoff planner prunes any observation
point inside the rectangle, and :func:`build_interlock` refuses to advance when
the duck's own next step would carry it within :data:`ZONE_STANDOFF_M` of the
marked edge.  Neither is derived from the other, so one of them being wrong
cannot produce a robot in the restricted zone.
"""

from __future__ import annotations

import math

import numpy as np

from patrol_cast import ALL_NAMES
from patrol_control import Interlock
from patrol_facility import (
    RESTRICTED_ZONE,
    occluder_between,
    static_gap,
)
from patrol_states import (
    DUCK_PLANAR_RADIUS,
    SETTLED_MPS,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
)
# Planar radius each body is treated as when deciding whether it BLOCKS a
# sightline.  The widest body's measured lateral half-width rounded up, because
# a body seen edge-on still screens what is behind it.
BODY_OCCLUDER_RADIUS_M = 0.14
# How far ahead the interlock looks when refusing to advance.  Far enough that
# the duck stops BEFORE a body rather than beside one.
INTERLOCK_LOOKAHEAD_M = 0.34
# The surface separation at which the interlock refuses.  Above the duck's own
# planar radius, so the refusal happens with room to spare rather than at the
# point of contact.  It is deliberately BELOW the standoff band's lower edge of
# 0.45 m: the interlock is a backstop for a failure of the approach controller,
# not the thing that normally stops the duck.
INTERLOCK_CLEARANCE_M = 0.30
# The margin the ZONE interlock grows the marked rectangle by.  It is the duck's
# own planar half-extent and NOTHING MORE, so what the interlock forbids is
# exactly what the rule forbids: any part of the robot's footprint inside the
# marked rectangle.
#
# THIS IS DELIBERATELY LOOSER THAN THE STANDOFF PLANNER'S OWN ZONE MARGIN, AND
# THE SPLIT IS A MEASURED NECESSITY RATHER THAN A RELAXATION.  The planner uses
# ``ZONE_STANDOFF_M`` on top of the radius when it CHOOSES where to stand, which
# is a design rule about picking a comfortable observation point and can afford
# to be generous.  The interlock governs whether the duck may MOVE AT ALL, and
# on this robot that is a kinematic question: turning in place is MEASURED to be
# unavailable at 1.6 deg/s, so a duck whose forward command is refused cannot
# turn away either - the yaw it needs comes from walking.
#
# MEASURED, with the planner's generous margin used here as well: after
# observing the intruder the duck stood at (-1.054, 0.970) facing the annex at
# -173 deg, 0.61 m from the rectangle.  Its next 0.34 m crossed the GROWN
# rectangle by 7 mm, so it was refused; being refused it could not turn; and it
# stood there for the whole 40 s ``RETURN_TO_PATROL`` ceiling.  The left turn it
# needed would have carried it no closer than 0.298 m from the real rectangle -
# comfortably outside the robot's own footprint, and forbidden only by a margin
# that had been borrowed from a different question.
ZONE_INTERLOCK_MARGIN_M = DUCK_PLANAR_RADIUS


def measured_positions(bodies) -> dict[str, np.ndarray]:
    """Every present body's position, as the duck's own sensing would report it.

    A body that has not appeared yet is omitted entirely rather than reported at
    its parking position: the duck cannot see it, so it does not exist as far as
    every layer above this one is concerned.
    """
    return {name: np.asarray(bodies[name].pos, dtype=np.float64)[:2]
            for name in ALL_NAMES if bodies[name].present}


def range_to(duck_xy, position) -> float:
    """Planar centre-to-centre range from the duck to one body."""
    return float(np.linalg.norm(
        np.asarray(position, dtype=np.float64)[:2]
        - np.asarray(duck_xy, dtype=np.float64)[:2]))


def zone_gap_m(duck_xy) -> float:
    """Signed distance from the duck to the restricted zone's edge.

    Positive outside, negative inside.  This is the quantity the "it never
    entered the restricted zone" gate is measured on, every control tick.
    """
    return float(-RESTRICTED_ZONE.depth_inside(duck_xy))


def los_blocked_by(eye_xy, target_xy, bodies, exclude: str = "",
                   margin: float = 0.0) -> str:
    """Name of whatever stands in the planar sightline to a point.

    Static geometry first, then OTHER BODIES.  Both are returned as a single
    "line of sight did not exist" answer, because the visibility gate excludes
    exactly the ticks where seeing the target was geometrically impossible - and
    a body in the way makes it just as impossible as the central rack does.
    Holding the duck responsible for not seeing through somebody would grade the
    facility's layout rather than the robot.
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
        # Only bodies BETWEEN the eye and the target can occlude.  The 0.08 m
        # standoff keeps a body standing essentially at the target from counting
        # as its own screen.
        if along <= 0.08 or along >= length - 0.08:
            continue
        lateral = abs(float(offset[0] * direction[1]
                            - offset[1] * direction[0]))
        if lateral <= BODY_OCCLUDER_RADIUS_M + margin:
            return name
    return ""


def in_standoff_band(range_m: float, target: str) -> bool:
    """Is the MEASURED surface standoff inside the required band?

    Converted from the centre-to-centre range through the same planning radii
    the standoff planner used, so the band the machine stops on and the band the
    planner aimed at are the same band.  The GATE measures the real geoms with
    ``ContactProbe`` and is what any safety claim rests on.
    """
    from patrol_investigate import standoff_from_range

    surface = standoff_from_range(target, range_m)
    return bool(STANDOFF_MIN_M <= surface <= STANDOFF_MAX_M)


def build_interlock(*, duck_xy, duck_yaw: float, bodies,
                    clearances: dict[str, float], target_xy=None) -> Interlock:
    """The independent refusal to walk into somebody, or into the marked zone.

    Computed from THIS TICK's MEASURED surface clearance, the body's bearing and
    the duck's own distance to the zone - never from the detector or the
    standoff planner.  The planner reasons about where it would be safe to
    stand; this reasons about where the duck is right now.  The two are computed
    from different quantities, so a mistake in either alone cannot produce a
    duck touching an anomaly or standing in a restricted area.

    A REFUSAL MUST NOT BE A TRAP, AND ON THIS ROBOT THAT IS A REAL CONSTRAINT.
    Turning in place is MEASURED to be unavailable - at most 1.6 deg/s at
    ``vx = 0`` - so a duck whose forward command is refused cannot turn away
    either: the yaw it needs comes from walking.  A refusal that fired on
    heading alone would therefore be permanent.

    MEASURED: it was.  With a heading-only zone check the duck finished its
    intrusion observation facing the restricted annex, was refused, could not
    turn, and stood still for the whole 30 s ``RETURN_TO_PATROL`` ceiling,
    ending 1.249 m from the point it was supposed to return to.

    The refusal is therefore about the DIRECTION OF TRAVEL, not the pose: it
    fires only when advancing would take the duck CLOSER to the zone's edge.
    Walking away from a restricted area is always allowed, which is both the
    physically correct rule and the only one this robot can obey.
    """
    duck = np.asarray(duck_xy, dtype=np.float64)[:2]
    heading = np.array([math.cos(duck_yaw), math.sin(duck_yaw)])

    # THE ZONE COMES FIRST, because it is a rule rather than a hazard: the duck
    # may not enter the marked rectangle even if doing so would be perfectly
    # safe.  Checked one lookahead ahead, so it stops at the line rather than
    # discovering it has crossed.
    ahead = duck + heading * INTERLOCK_LOOKAHEAD_M
    if RESTRICTED_ZONE.contains(ahead, ZONE_INTERLOCK_MARGIN_M) \
            and zone_gap_m(ahead) < zone_gap_m(duck):
        return Interlock(
            True,
            f"the next {INTERLOCK_LOOKAHEAD_M:.2f} m would cross into the "
            f"{RESTRICTED_ZONE.name} restricted zone",
            "")

    for name, gap in sorted(clearances.items(), key=lambda kv: kv[1]):
        if gap >= INTERLOCK_CLEARANCE_M or not bodies[name].present:
            continue
        offset = np.asarray(bodies[name].pos, dtype=np.float64)[:2] - duck
        if float(offset @ heading) <= 0.0:
            continue
        if float(np.linalg.norm(offset)) > INTERLOCK_LOOKAHEAD_M + 0.9:
            continue
        # The same escape rule as the zone: a body the duck is walking AWAY
        # from cannot be a reason to stand still, or the refusal becomes a trap
        # on a robot that steers by walking.
        if target_xy is not None:
            span = np.asarray(target_xy, dtype=np.float64)[:2] - duck
            if float(np.linalg.norm(span)) > 1e-9 \
                    and float(offset @ (span / np.linalg.norm(span))) <= 0.0:
                continue
        return Interlock(
            True,
            f"{name} is {gap:.3f} m away and ahead; below the "
            f"{INTERLOCK_CLEARANCE_M:.2f} m bar",
            name)
    return Interlock(False, "", "")


def build_sense(*, plan, duck_xy, measured_speed_mps: float,
                scan_arc_deg: float, scan_complete: bool,
                bodies_seen: tuple[str, ...], candidate: str,
                candidate_verdict: str, candidate_rule: str,
                candidate_confidence: float, candidate_investigate: bool,
                candidate_visible: bool, candidate_range_m: float,
                target_range_m: float, standoff_ready: bool,
                standoff_remaining_m: float, in_band: bool,
                observe_elapsed_s: float, observations_done: int,
                measured_min_clearance_m: float):
    """Everything the duck measured and concluded this tick, as one object."""
    from patrol_branch import RESUME_TOLERANCE_M
    from patrol_episode import Sense
    from patrol_states import CHECKPOINT_ARRIVE_M

    duck = np.asarray(duck_xy, dtype=np.float64)[:2]
    remaining = plan.remaining_m(duck)
    resume = plan.resume_xy
    resume_remaining = (float(np.linalg.norm(duck - resume))
                        if resume is not None else 1e9)

    return Sense(
        target_name=plan.target_name,
        target_remaining_m=remaining,
        at_target=bool(remaining <= CHECKPOINT_ARRIVE_M),
        at_home=bool(plan.finished_circuit
                     and remaining <= CHECKPOINT_ARRIVE_M),
        finished_circuit=plan.finished_circuit,
        completed=len(plan.completed),

        measured_speed_mps=float(measured_speed_mps),
        settled=bool(measured_speed_mps <= SETTLED_MPS),

        scan_arc_deg=float(scan_arc_deg),
        scan_complete=bool(scan_complete),
        bodies_seen=tuple(bodies_seen),

        candidate=candidate,
        candidate_verdict=candidate_verdict,
        candidate_rule=candidate_rule,
        candidate_confidence=float(candidate_confidence),
        candidate_range_m=float(candidate_range_m),
        candidate_investigate=bool(candidate_investigate),
        candidate_visible=bool(candidate_visible),

        target_range_m=float(target_range_m),
        standoff_ready=bool(standoff_ready),
        standoff_remaining_m=float(standoff_remaining_m),
        in_standoff_band=bool(in_band),
        observe_elapsed_s=float(observe_elapsed_s),
        observations_done=int(observations_done),

        resume_remaining_m=resume_remaining,
        at_resume_point=bool(resume_remaining <= RESUME_TOLERANCE_M),

        measured_min_clearance_m=float(measured_min_clearance_m),
        zone_gap_m=zone_gap_m(duck),
    )


def duck_static_gap(duck_xy) -> tuple[str, float]:
    """Nearest static surface and the planar gap to it, for reporting."""
    return static_gap(duck_xy)
