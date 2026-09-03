#!/usr/bin/env python3
"""Scene layout, the duck's footprint, and what makes a gap a legitimate place.

Single source of truth for every distance the scene generator paints and the
decision layer reasons about, so the picture and the arithmetic cannot drift
apart.

THE CENTRAL DISTINCTION THIS MODULE ENCODES
-------------------------------------------
A gap in a queue being WIDE ENOUGH is a geometric fact.  A gap being YOURS is a
social one, and it is the second that this behavior is about.  The two are kept
deliberately separate:

* :func:`gap_fits_duck` answers only "could the duck physically stand here",
  from the measured footprint and the measured separation.
* :func:`classify_gap` answers "may the duck stand here", and the ONLY gap it
  admits is the one behind the last person in line.

The scene is built so those two answers disagree.  The straggler leaves a
0.90 m hole in the middle of the queue - 0.35 m more than the nominal spacing,
and comfortably wider than the 0.26 m the duck's own footprint needs - so
refusing it cannot be explained by "it did not fit".  The same is true of the
space beside the person at the counter.  Every rejection in this behavior is a
rejection of a gap the duck could have taken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from queue_path import PATH, QUEUE_BAND_M, SLOT_SPACING_M

# -- hall and barrier layout ------------------------------------------------
# Half-width of the roped lane, centre to post centre, and of the floor paint.
BARRIER_HALF_M = 0.42
LANE_PAINT_HALF_M = 0.40
# The barrier run stops this far short of the path's open end, leaving a mouth
# the duck walks in through.  Roping the lane all the way round would enclose
# it and force the duck to pass through a rope to join at all.
BARRIER_MOUTH_M = 0.60
# Front face of the service counter.  The queue runs out along -x from the
# origin, so the counter sits at NEGATIVE x and the person being served stands
# at the origin facing it.
COUNTER_FRONT_X = 0.30
# Where the duck enters the hall: outside the lane, beyond its open mouth,
# angled in toward the entrance.  The approach is therefore a real traverse
# from outside the queue rather than a spawn inside it.
DUCK_START_XY = (2.55, -0.72)
DUCK_START_YAW_DEG = 208.0

# -- the duck -------------------------------------------------------------
# Conservative planar half-extent of the robot, from each geom's BOUNDING
# SPHERE.  Over-states the robot, which is the safe direction for every gate:
# a fatter duck makes clearance, lane-keeping and gap-fitting all HARDER.
# ``test_duck_planar_radius_matches_the_model`` pins this against the built
# scene, so it can never drift from the robot it describes.
DUCK_PLANAR_RADIUS = 0.1303

# -- what the duck must achieve -------------------------------------------
# Standoff behind the person in front, trunk centre to their centre, measured
# along the queue path.  Wider than the sum of the two half-extents by a clear
# margin, and narrow enough that the duck is unmistakably IN the queue rather
# than loitering near it.
STANDOFF_MIN_M = 0.45
STANDOFF_MAX_M = 0.75
STANDOFF_TARGET_M = 0.58
# How far off the lane centreline the duck may sit and still count as being in
# the queue rather than beside it.
JOIN_LATERAL_BAND_M = 0.22
# How far behind the tail's arc length the duck must be to have joined BEHIND
# rather than beside or in front.
JOIN_LONGITUDINAL_MIN_M = 0.40
JOIN_LONGITUDINAL_MAX_M = 0.85
# Cross-track budget while advancing round the bend.  A duck that cut the
# corner would show up here as a large NEGATIVE cross-track (inside the bend).
CROSS_TRACK_LIMIT_M = 0.20
# Corner cutting is graded separately and one-sided, because only the inside of
# a bend can be cut.  A duck that swings wide is clumsy; a duck that cuts the
# corner has left the queue's own path.
CORNER_CUT_LIMIT_M = 0.13
# The duck has arrived at the counter when its trunk is within this of the
# service station's arc length.
AT_COUNTER_ARC_M = 0.24


def duck_footprint_gap(separation_m: float, other_half_extent_m: float) -> float:
    """Surface gap left over if the duck stands midway in a gap of this size."""
    return 0.5 * separation_m - DUCK_PLANAR_RADIUS - other_half_extent_m


def gap_fits_duck(separation_m: float, other_half_extent_m: float,
                  min_surface_gap_m: float = 0.06) -> bool:
    """Purely geometric: could the duck physically stand in a gap this wide?

    Deliberately generous.  The point of this function is to establish that the
    gaps the duck refuses are gaps it COULD have taken, so making it strict
    would quietly weaken the behavior's central claim.
    """
    return duck_footprint_gap(separation_m, other_half_extent_m) >= min_surface_gap_m


@dataclass(frozen=True)
class Gap:
    """A candidate standing place, in WORLD coordinates.

    A candidate is not always on the queue path: the most tempting way to jump
    a queue is to walk up BESIDE the person being served, which is off the path
    entirely.  So a gap carries its own world station, and its arc length and
    off-path distance are measured by projection like everything else.

    ``ahead`` is the person whose back the duck would be at; ``behind`` is the
    person the duck would be standing in front of, or ``None`` for the gap
    behind the tail.
    """

    name: str
    xy: tuple[float, float]
    kind: str                 # "side" | "cut_in" | "tail"
    ahead: str | None
    behind: str | None
    separation_m: float
    fits: bool
    verdict: str
    reason: str

    @property
    def accepted(self) -> bool:
        return self.verdict == "join"

    @property
    def arc(self) -> float:
        return PATH.project(self.xy)[0]

    @property
    def off_path_m(self) -> float:
        return PATH.project(self.xy)[2]

    def as_record(self) -> dict:
        return {
            "gap": self.name, "kind": self.kind,
            "xy": [round(float(self.xy[0]), 4), round(float(self.xy[1]), 4)],
            "arc_s_m": round(self.arc, 4),
            "off_path_m": round(self.off_path_m, 4),
            "ahead": self.ahead, "behind": self.behind,
            "separation_m": round(self.separation_m, 4),
            "physically_fits": bool(self.fits),
            "verdict": self.verdict, "reason": self.reason,
        }


# Where the "walk up beside the person being served" candidate stands: in front
# of the counter, one body's width to the side of the service station.  It is
# genuinely empty floor - nothing is there and nothing is near it - so refusing
# it is a decision about order rather than an observation about space.
COUNTER_SIDE_XY = (0.30, -0.60)


def enumerate_gaps(order: list[str], arcs: dict[str, float],
                   half_extent_m: float) -> list[Gap]:
    """Every standing place the queue offers, from the counter to behind the tail.

    ``order`` is the inferred order, head first; ``arcs`` each member's arc
    length.  The gaps are enumerated front to back so the HUD shows the duck
    working from the tempting end of the queue backwards - it looks at the
    places that would get it served soonest FIRST, and says no to them.

    THE CLASSIFICATION IS THE BEHAVIOR:

    * ``side``    beside the person being served, off the queue path.  Empty
      floor, physically available, and the purest form of queue-jumping.
    * ``cut_in``  between two people already in line.  This is where the
      straggler's 0.90 m hole lives.
    * ``tail``    behind the last person.  The only legitimate place.

    ``fits`` is computed for EVERY candidate, not just the accepted one.  A
    refusal only means something if the thing refused was actually available.
    """
    gaps: list[Gap] = []
    if not order:
        return gaps

    head = order[0]
    gaps.append(Gap(
        name="beside_counter", xy=COUNTER_SIDE_XY, kind="side",
        ahead=None, behind=head,
        separation_m=float(np.linalg.norm(
            np.asarray(COUNTER_SIDE_XY) - PATH.point_at(arcs[head]))),
        # Empty floor beside the counter: nothing to fit between, so the only
        # question is whether the duck's own footprint has room, and it does.
        fits=True,
        verdict="reject",
        reason=(f"empty floor beside {head}, who is being served: taking it "
                "would put the duck at the counter ahead of the whole queue")))

    for ahead, behind in zip(order, order[1:]):
        separation = abs(arcs[behind] - arcs[ahead])
        fits = gap_fits_duck(separation, half_extent_m)
        midpoint = PATH.point_at(0.5 * (arcs[ahead] + arcs[behind]))
        if fits:
            reason = (f"{separation:.2f} m between {ahead} and {behind} is wide "
                      f"enough for the duck, but standing there cuts in front "
                      f"of {behind}")
        else:
            reason = (f"{separation:.2f} m between {ahead} and {behind} is too "
                      f"narrow, and would cut in front of {behind} anyway")
        gaps.append(Gap(
            name=f"between_{ahead}_{behind}",
            xy=(float(midpoint[0]), float(midpoint[1])), kind="cut_in",
            ahead=ahead, behind=behind, separation_m=separation, fits=fits,
            verdict="reject", reason=reason))

    tail = order[-1]
    tail_station = PATH.point_at(arcs[tail] + STANDOFF_TARGET_M)
    gaps.append(Gap(
        name="behind_tail",
        xy=(float(tail_station[0]), float(tail_station[1])), kind="tail",
        ahead=tail, behind=None, separation_m=STANDOFF_TARGET_M,
        fits=gap_fits_duck(2.0 * STANDOFF_TARGET_M, half_extent_m),
        verdict="join",
        reason=f"behind {tail}, the last person in line"))
    return gaps


def classify_gap(gap: Gap) -> str:
    """The verdict, re-derived from the gap's own structure.

    Exists so the acceptance gate can recompute every verdict independently of
    the enumeration that produced it, and so a synthetic counterexample can
    build a gap by hand and test the RULE rather than the bookkeeping.  The rule
    is one line: the only admissible place is the one with nobody behind it.
    """
    return "join" if gap.behind is None else "reject"


def join_station_arc(tail_arc: float) -> float:
    return tail_arc + STANDOFF_TARGET_M


def advance_station_arc(predecessor_arc: float) -> float:
    return predecessor_arc + STANDOFF_TARGET_M


def in_join_band(duck_arc: float, duck_cross_m: float,
                 tail_arc: float) -> tuple[bool, float, float]:
    """Is the duck physically behind the tail, in the lane, at a queue standoff?

    Returns the verdict and the two measurements it is made of, so a failure
    reports which band it missed rather than a bare False.
    """
    longitudinal = duck_arc - tail_arc
    lateral = abs(duck_cross_m)
    ok = (JOIN_LONGITUDINAL_MIN_M <= longitudinal <= JOIN_LONGITUDINAL_MAX_M
          and lateral <= JOIN_LATERAL_BAND_M)
    return bool(ok), float(longitudinal), float(lateral)


def counter_arc() -> float:
    return 0.0


def lane_clearance(xy) -> float:
    """Distance from the duck's footprint to the nearer barrier line.

    Geometric, computed from the path rather than from the scene's geoms, so it
    can be evaluated in a pure unit test.  The scene's real posts and ropes are
    graded separately by the exact surface probe.
    """
    _, cross, _ = PATH.project(xy)
    return BARRIER_HALF_M - abs(cross) - DUCK_PLANAR_RADIUS


def naive_tail(order_by_range: list[str]) -> str:
    return order_by_range[-1]


def queue_geometry_summary() -> dict:
    """Everything a reader needs to check the scenario's premise arithmetically."""
    from queue_people import ADULT_HALF_EXTENT_M
    nominal = SLOT_SPACING_M
    straggler = 0.90
    return {
        "slot_spacing_m": nominal,
        "straggler_gap_m": straggler,
        "duck_planar_radius_m": DUCK_PLANAR_RADIUS,
        "adult_half_extent_m": ADULT_HALF_EXTENT_M,
        "duck_needs_m": round(
            2.0 * (DUCK_PLANAR_RADIUS + ADULT_HALF_EXTENT_M + 0.06), 4),
        "straggler_gap_surface_slack_m": round(
            duck_footprint_gap(straggler, ADULT_HALF_EXTENT_M), 4),
        "straggler_gap_fits_duck": gap_fits_duck(straggler, ADULT_HALF_EXTENT_M),
        "nominal_gap_fits_duck": gap_fits_duck(nominal, ADULT_HALF_EXTENT_M),
        "lane_half_width_m": BARRIER_HALF_M,
        "lane_slack_each_side_m": round(
            BARRIER_HALF_M - DUCK_PLANAR_RADIUS, 4),
        "queue_band_m": QUEUE_BAND_M,
        "path_length_m": round(PATH.length, 4),
        "fold_sweep_deg": 180.0,
    }
