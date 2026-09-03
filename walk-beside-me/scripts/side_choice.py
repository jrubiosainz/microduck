#!/usr/bin/env python3
"""Is this side usable?  A measurement, with a named cause, every control tick.

This is the module the whole behavior turns on, so it is pure: no MuJoCo, no
state, no time-stepping.  It takes the guardian's pose, a set of predicted
pedestrian tracks and a side, and returns a verdict with the reason attached.
Every side decision the duck makes can therefore be replayed and explained, and
the acceptance gate can require that each switch names a MEASURED cause rather
than merely happening.

WHAT MAKES A SIDE UNUSABLE
--------------------------
Two things, and they are checked against different margins because they are
different hazards:

* **A static surface.**  The slot, or the lane leading to it, comes within
  :data:`SIDE_STATIC_MARGIN_M` of an obstacle or a wall.  Static bodies do not
  move, so the present measurement is the whole story.
* **A predicted pedestrian.**  Somebody is projected to come within
  :data:`SIDE_PERSON_MARGIN_M` of the slot at any point in the next
  :data:`SIDE_LOOKAHEAD_S` seconds.  The margin is larger than the static one
  because a person moves and swings their arms, and because the prediction is
  linear and therefore wrong in detail.

THE LANE, NOT JUST THE SLOT
---------------------------
Checking only the slot's current position would let the duck walk into a body
that its slot is about to reach.  The slot is therefore swept along the
guardian's own predicted motion over the lookahead window, and every sample of
that swept lane is graded.  This is what makes the kiosk refusal happen BEFORE
the duck is level with the kiosk rather than as it arrives.

WHY A LINEAR PREDICTION IS HONEST HERE, AND WHERE IT IS NOT
------------------------------------------------------------
The pedestrians walk continuous filleted routes at constant speed, so over a
3 s window a constant-velocity extrapolation of their current heading is close
to their true path except in a bend.  It is not a perception system and it is
not a tracker; it is a stated proxy for one.  What matters for the gate is that
the duck's decision is made from PREDICTED positions it computed itself, never
from the scenario's schedule, and that the prediction error is bounded and
reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from beside_geometry import (
    BESIDE_TARGET_M,
    SIDE_LOOKAHEAD_S,
    SIDE_LOOKAHEAD_SAMPLES,
    SIDE_PERSON_MARGIN_M,
    SIDE_STATIC_MARGIN_M,
    slot_point,
)
from promenade_layout import static_gap


@dataclass(frozen=True)
class Track:
    """One predicted pedestrian: where they are and where they are going.

    Constructed from what the duck could measure — a position and a velocity —
    and never from a route object, so the side chooser cannot accidentally
    consult the scenario's own schedule.
    """

    name: str
    pos: np.ndarray
    velocity: np.ndarray

    def at(self, dt: float) -> np.ndarray:
        return self.pos + self.velocity * dt


@dataclass
class SideVerdict:
    """Whether a side is usable, and the measured reason it is not."""

    side: int
    usable: bool
    static_gap_m: float
    static_name: str
    person_gap_m: float
    person_name: str
    person_dt_s: float
    cause: str = ""
    detail: str = ""
    samples: list[dict] = field(default_factory=list)

    def as_record(self) -> dict:
        return {
            "side": self.side,
            "usable": bool(self.usable),
            "static_gap_m": round(float(self.static_gap_m), 4),
            "static_name": self.static_name,
            "person_gap_m": round(float(self.person_gap_m), 4),
            "person_name": self.person_name,
            "person_dt_s": round(float(self.person_dt_s), 3),
            "cause": self.cause,
            "detail": self.detail,
        }


def evaluate_side(guardian_xy, guardian_yaw: float, guardian_velocity,
                  side: int, tracks: list[Track], *,
                  lookahead_s: float = SIDE_LOOKAHEAD_S,
                  samples: int = SIDE_LOOKAHEAD_SAMPLES,
                  lateral: float = BESIDE_TARGET_M,
                  static_margin: float = SIDE_STATIC_MARGIN_M,
                  person_margin: float = SIDE_PERSON_MARGIN_M) -> SideVerdict:
    """Grade one side of the guardian over the lookahead window.

    The slot is swept forward along the guardian's own predicted motion, so the
    verdict covers the lane the duck would occupy rather than a single point it
    happens to stand on now.
    """
    guardian_xy = np.asarray(guardian_xy, dtype=np.float64)
    guardian_velocity = np.asarray(guardian_velocity, dtype=np.float64)

    worst_static = float("inf")
    worst_static_name = ""
    worst_person = float("inf")
    worst_person_name = ""
    worst_person_dt = 0.0
    trace: list[dict] = []

    for index in range(samples):
        dt = lookahead_s * index / max(samples - 1, 1)
        future_xy = guardian_xy + guardian_velocity * dt
        slot = slot_point(future_xy, guardian_yaw, side, lateral=lateral)

        name, gap = static_gap(slot)
        if gap < worst_static:
            worst_static, worst_static_name = gap, name

        for track in tracks:
            distance = float(np.linalg.norm(track.at(dt) - slot))
            if distance < worst_person:
                worst_person = distance
                worst_person_name = track.name
                worst_person_dt = dt
        trace.append({"dt_s": round(dt, 3), "static_gap_m": round(gap, 4)})

    static_blocked = worst_static < static_margin
    person_blocked = worst_person < person_margin
    usable = not (static_blocked or person_blocked)

    cause, detail = "", ""
    if static_blocked and person_blocked:
        # Report the tighter of the two, scaled by its own margin, so the cause
        # names the hazard that actually decided rather than whichever was
        # tested first.
        if worst_static / static_margin <= worst_person / person_margin:
            cause, detail = "static", worst_static_name
        else:
            cause, detail = "person", worst_person_name
    elif static_blocked:
        cause, detail = "static", worst_static_name
    elif person_blocked:
        cause, detail = "person", worst_person_name

    return SideVerdict(
        side=side, usable=usable,
        static_gap_m=worst_static, static_name=worst_static_name,
        person_gap_m=worst_person, person_name=worst_person_name,
        person_dt_s=worst_person_dt, cause=cause, detail=detail, samples=trace)


def evaluate_both(guardian_xy, guardian_yaw: float, guardian_velocity,
                  tracks: list[Track], **kwargs) -> dict[int, SideVerdict]:
    """Grade both sides at once, keyed by side (+1 left, -1 right)."""
    return {
        side: evaluate_side(guardian_xy, guardian_yaw, guardian_velocity,
                            side, tracks, **kwargs)
        for side in (1, -1)
    }


def prefer_side(verdicts: dict[int, SideVerdict],
                current_side: int | None = None) -> tuple[int | None, str]:
    """Which side to take, and why.

    * Both usable: keep the current one if it is usable (switching sides for no
      reason is worse than either choice), otherwise take the side with the
      larger static clearance, tie-broken to the LEFT deterministically.
    * One usable: that one.
    * Neither: ``None``, which the machine treats as "hold what you have and
      keep measuring" rather than as a licence to stop.
    """
    left, right = verdicts[1], verdicts[-1]
    if left.usable and right.usable:
        if current_side in (1, -1) and verdicts[current_side].usable:
            return current_side, "current side remains usable"
        if left.static_gap_m >= right.static_gap_m:
            return 1, "both usable; left has more static clearance"
        return -1, "both usable; right has more static clearance"
    if left.usable:
        return 1, f"right blocked by {right.cause}:{right.detail}"
    if right.usable:
        return -1, f"left blocked by {left.cause}:{left.detail}"
    return None, (f"both blocked (left {left.cause}:{left.detail}, "
                  f"right {right.cause}:{right.detail})")


def tracks_from_states(people, exclude: str) -> list[Track]:
    """Build predicted tracks from a people snapshot, excluding the guardian.

    Uses each person's CURRENT position and velocity only.  Nothing here reads a
    route, a waypoint list or a schedule, which is what keeps the duck's side
    decisions measurements rather than lookups.
    """
    return [
        Track(name, np.asarray(state.pos, dtype=np.float64).copy(),
              np.asarray(state.velocity, dtype=np.float64).copy())
        for name, state in people.items() if name != exclude
    ]


def prediction_error(track: Track, actual_xy, dt: float) -> float:
    """How wrong the linear prediction was, in metres, at horizon ``dt``.

    Reported by the metrics so the lookahead's honesty is a measured quantity
    rather than an assumption.
    """
    return float(np.linalg.norm(track.at(dt) - np.asarray(
        actual_xy, dtype=np.float64)))


def closing_speed(track: Track, guardian_velocity) -> float:
    """Speed at which a track approaches the guardian's frame, m/s."""
    relative_velocity = track.velocity - np.asarray(
        guardian_velocity, dtype=np.float64)
    return float(np.linalg.norm(relative_velocity))


def bearing_of(track: Track, guardian_xy, guardian_yaw: float) -> float:
    """Where a track sits in the guardian's frame, in degrees off her nose."""
    delta = track.pos - np.asarray(guardian_xy, dtype=np.float64)
    return math.degrees(math.atan2(
        float(delta[1]), float(delta[0])) - guardian_yaw)
