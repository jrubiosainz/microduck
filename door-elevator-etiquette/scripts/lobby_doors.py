#!/usr/bin/env python3
"""The three door pairs: their schedule, their open fraction, and the effective
clear gap that follows from it.

Each door is a SCRIPTED KINEMATIC SEMANTIC PROXY.  There is no door controller,
no motion sensor and no lift call button anywhere in this behavior: a door's
open fraction is a function of time alone, declared in :data:`DOOR_SCHEDULE`,
and the leaves are mocap bodies posed from it.  That is stated plainly because a
robot demo about doors invites the reader to assume the robot operated them.  It
did not.

WHAT IS *NOT* A PROXY IS THE CONSEQUENCE
-----------------------------------------
The duck never reads this schedule.  It reads :meth:`DoorState.open_fraction`
through the same interface a real sensor would provide, and every gate about
doors is graded on the MEASURED geometry that the leaves actually had:

* ``no movement through a closed door`` is checked by measuring the duck's own
  x against each aperture plane every control tick and requiring the door to
  have been open whenever it crossed;
* the duck's clearance to the leaves is measured by the same analytic probe that
  measures every other surface, so a leaf sweeping shut onto the robot would
  show up as a clearance failure rather than as a passing run.

WHY THE FRACTION IS RAMPED AND NOT SWITCHED
---------------------------------------------
A door that goes from shut to open in one control tick is a teleport, and a
clearance gate graded against it would be grading a discontinuity.  Every edge
here is a smootherstep over :data:`DOOR_RAMP_S`, so the leaves have continuous
velocity and the duck's measured clearance to them is a real curve.

THE OPEN FRACTION IS THE ONLY THING THE REST OF THE BEHAVIOR SEES
------------------------------------------------------------------
``effective_gap_m`` is DERIVED from the fraction and the aperture, never
declared, so a schedule edit cannot leave a stale width behind.  A fraction of
0 is a gap of 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from lobby_layout import (
    DOOR_CENTER_Y,
    DOOR_CLEAR_W,
    DOOR_LEAF_TRAVEL,
    DOOR_WALL_X,
    LIFT_CLEAR_W,
    LIFT_FRONT_Y,
    LIFT_LEAF_TRAVEL,
    LIFT_WALL_X,
    REAR_CLEAR_W,
    REAR_LEAF_TRAVEL,
    REAR_WALL_X,
    REAR_Y,
)

# Seconds each leaf takes to travel its full stroke.  Slow enough to read in the
# video and to be a real constraint on when the duck may enter.
DOOR_RAMP_S = 1.30
# A door is treated as OPEN ENOUGH to walk through at this fraction.  Derived:
# the duck's conservative planar diameter is 0.2606 m, so at 0.55 of the 0.66 m
# concourse aperture the clear gap is 0.363 m and the robot has 0.05 m either
# side.  Below it the aperture is narrower than a safe pass and the duck must
# not be in it.
DOOR_PASSABLE_FRACTION = 0.55

# The three apertures, and which axis they lie on.  All three are x-planes here,
# so crossing one is a sign change of ``duck_x - plane_x``; the plane is stored
# rather than assumed so a test can pin it against the built scene.
APERTURES: dict[str, dict] = {
    "concourse_door": {
        "plane_x": DOOR_WALL_X,
        "center_y": DOOR_CENTER_Y,
        "clear_w": DOOR_CLEAR_W,
        "travel": DOOR_LEAF_TRAVEL,
        "label": "automatic door in the concourse divider",
    },
    "lift_front": {
        "plane_x": LIFT_WALL_X,
        "center_y": LIFT_FRONT_Y,
        "clear_w": LIFT_CLEAR_W,
        "travel": LIFT_LEAF_TRAVEL,
        "label": "lift doors, lobby side",
    },
    "lift_rear": {
        "plane_x": REAR_WALL_X,
        "center_y": REAR_Y,
        "clear_w": REAR_CLEAR_W,
        "travel": REAR_LEAF_TRAVEL,
        "label": "lift doors, target floor",
    },
}
APERTURE_NAMES: tuple[str, ...] = tuple(APERTURES)

# (door, open_at_s, close_at_s) - the instant each edge STARTS.  A close_at of
# None means it never shuts again within the run.
#
# EVERY NUMBER HERE IS DERIVED FROM A MEASUREMENT, NOT CHOSEN.
# ``tools/measure_legs.py`` walks the whole route with the real policy and no
# state machine and reports when the duck actually arrives at each holding
# point; ``tools/tune_phasing.py`` turns those arrivals into this timeline and
# FAILS if any edge falls on the wrong side of one.  Re-run both after any
# geometry change - the schedule is a consequence of the building, and a
# building 0.4 m shorter moves every edge below.
#
# The measured arrivals are 10.5 s (door hold), 23.8 s (through the doorway),
# 36.2 s (beside the lift), 50.2 s (positioned in the cabin) and 67.3 s (target
# floor), each cumulative over the scripted pauses.
#
# WHAT EACH ENTRY IS FOR:
#
# * the concourse door opens at 5.0 s - fully open by 6.3 s, comfortably before
#   the duck reaches its holding point at 10.5 s - because two people are about
#   to come OUT through it, which is what the duck must yield to.  It stays open
#   for the rest of the run, so "the duck waited outside the threshold" cannot
#   be confused with "the duck waited for the door".
# * the lift front doors open at 48.8 s, which is 3.2 s AFTER the duck reaches
#   the holding spot beside them.  That gap is the WAIT_SIDE state, and it is
#   why the duck is standing beside a closed lift rather than walking up to an
#   open one.  They shut at 79.0 s, once the duck is aboard and positioned, so
#   the ride happens in a sealed car.
# * the lift rear doors open at 86.8 s at the target floor, 8.1 s into the ride.
#   Until that instant the duck is in a sealed box, so "it did not walk through
#   a closed door" is a claim with something to fail against.
DOOR_SCHEDULE: tuple[tuple[str, float, float | None], ...] = (
    ("concourse_door", 5.00, None),
    ("lift_front", 48.76, 79.00),
    ("lift_rear", 86.80, None),
)


def _smootherstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


@dataclass(frozen=True)
class DoorState:
    """One door at one instant.  A measurement, not a command."""

    name: str
    fraction: float
    plane_x: float
    center_y: float
    clear_w: float
    travel: float

    @property
    def effective_gap_m(self) -> float:
        """The clear width actually available right now.

        DERIVED from the fraction and the leaf travel, never declared: at
        fraction 0 both leaves cover their half of the aperture and the gap is
        exactly 0; at fraction 1 each has retracted its full stroke and the gap
        is the whole clear width.
        """
        return 2.0 * self.travel * self.fraction

    @property
    def passable(self) -> bool:
        return self.fraction >= DOOR_PASSABLE_FRACTION

    @property
    def closed(self) -> bool:
        return self.fraction <= 1e-9

    def leaf_offsets(self) -> tuple[float, float]:
        """Centre offsets of the south and north leaves along the aperture axis.

        A leaf's centre sits half its own width from the aperture centre when
        shut, and retracts outward by the travel as the fraction rises.
        """
        shut = 0.5 * self.clear_w
        slide = self.travel * self.fraction
        return (-(shut + slide), +(shut + slide))


def open_fraction(name: str, t: float) -> float:
    """This door's open fraction at ``t``, ramped at both edges."""
    fraction = 0.0
    for door, open_at, close_at in DOOR_SCHEDULE:
        if door != name:
            continue
        opening = _smootherstep((t - open_at) / DOOR_RAMP_S)
        closing = (1.0 if close_at is None
                   else 1.0 - _smootherstep((t - close_at) / DOOR_RAMP_S))
        fraction = max(fraction, min(opening, closing))
    return float(min(max(fraction, 0.0), 1.0))


def doors_at(t: float) -> dict[str, DoorState]:
    """Every door's measured state at ``t``."""
    return {
        name: DoorState(name=name, fraction=open_fraction(name, t),
                        plane_x=float(spec["plane_x"]),
                        center_y=float(spec["center_y"]),
                        clear_w=float(spec["clear_w"]),
                        travel=float(spec["travel"]))
        for name, spec in APERTURES.items()
    }


def schedule_windows() -> list[dict]:
    """The declared schedule, for the metrics to report beside what happened.

    The gates never read this to decide anything; it is published so a reader
    can see exactly what was scripted and compare it against the measured
    crossing times.
    """
    return [{"door": door, "opens_at_s": open_at, "closes_at_s": close_at,
             "ramp_s": DOOR_RAMP_S, "label": APERTURES[door]["label"]}
            for door, open_at, close_at in DOOR_SCHEDULE]
