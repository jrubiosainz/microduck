#!/usr/bin/env python3
"""The planner: predict where everybody will be, score both corridors, choose.

This is the module the whole behavior turns on, and it is deliberately pure:
positions, velocities and geometry in, a decision out.  No MuJoCo, no policy, no
state machine, no clock — so every property below is unit-tested directly on
hand-built inputs.

WHAT IT PREDICTS, AND WHY THE MODEL IS DELIBERATELY NAIVE
----------------------------------------------------------
Each tracked body is propagated at CONSTANT VELOCITY over a
:data:`~slalom_states.PREDICT_HORIZON_S` horizon, sampled every
:data:`~slalom_states.PREDICT_DT_S`.  The velocity is finite-differenced from
that body's own two most recent MEASURED positions — never read from its route,
which the duck cannot see.

A constant-velocity model is the simplest thing that can be WRONG, and that is
the point.  The actors walk filleted routes with real bends, so a straight-line
extrapolation genuinely mispredicts wherever somebody is turning.  The
acceptance gate then requires the predicted clearance to conservatively BRACKET
the closest approach that actually occurred — a real test, precisely because the
predictor is not the choreography.

HOW A CORRIDOR IS SCORED
-------------------------
A corridor is a candidate lateral offset from the duck's current line to the
goal: :data:`~slalom_states.LATERAL_OFFSETS` to the left, the same to the right.
For each candidate the planner walks the duck forward along that offset line at
its MEASURED cruise, and at every horizon sample measures the distance from the
duck's predicted position to every predicted body position, inflated by that
body's planning radius.  The corridor's score is the WORST clearance over the
whole horizon.

Three things can disqualify a corridor, and all three are recorded so a refusal
can be explained rather than merely counted:

* ``unsafe``     — its worst predicted clearance is below
  :data:`~slalom_states.SAFE_CLEARANCE_M`;
* ``static``     — its line comes inside :data:`~slalom_states.STATIC_MARGIN_M`
  of a crate, pallet or cone.  The duck knows exactly where those are;
* ``reachable``  — whether the duck could physically get that far sideways in
  the time available, given the MEASURED lateral rate.  This one is REPORTED
  rather than vetoed, because :func:`duck_at` already ramps the offset in at
  that rate: an unreachable corridor is one whose predicted duck position never
  gets there, and its score already reflects that.  Vetoing on it as well
  double-counted and made the planner refuse every corridor on an empty floor.

WHY BOTH SIDES ARE ALWAYS EVALUATED
-------------------------------------
:func:`choose_corridor` scores every candidate on both hands even when the first
one it looks at is fine.  The decision record therefore always carries the
REJECTED side's predicted clearance and the reason it lost, which is what makes
"the duck chose left because right was predicted unsafe" a measured comparison
rather than a caption.  A planner that returned early on the first safe corridor
would produce the same motion and no evidence.

AND WHY "NEITHER" IS A REAL ANSWER
------------------------------------
When no candidate on either hand survives, :func:`choose_corridor` returns a
decision with ``side = "wait"``.  Given the MEASURED lateral budget — a 0.34 m
sidestep costs 0.64 m of course and 5.8 s — committing to a corridor that closes
is strictly worse than stopping, so waiting is the correct answer rather than a
failure to find one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from slalom_cast import planning_radius
from slalom_course import STATIC_OBSTACLES, GOAL_XY, LANE_HALF_W
from slalom_states import (
    DUCK_PLANAR_RADIUS,
    LATERAL_OFFSETS,
    LATERAL_RATE_MPS,
    PREDICT_DT_S,
    PREDICT_SAMPLES,
    PREDICTION_SLOP_M,
    SAFE_CLEARANCE_M,
    SPEED_AT_WALK,
    STATIC_MARGIN_M,
    THREAT_CLEARANCE_M,
    THREAT_RANGE_M,
    TRUNCATED_SAFE_M,
)


@dataclass(frozen=True)
class Track:
    """One body as the DUCK sees it: where it is, and how fast it is going.

    Built by ``slalom_sense`` from finite differences of measured positions.
    The planner never receives a route, a schedule or a destination — only this.
    """

    name: str
    pos: np.ndarray
    velocity: np.ndarray
    radius: float

    def predict(self, dt: float) -> np.ndarray:
        """Constant-velocity extrapolation.  Deliberately naive; see the module
        docstring."""
        return self.pos + self.velocity * dt


@dataclass
class Corridor:
    """One candidate lateral offset, scored over the whole horizon."""

    side: str                 # "left" | "right" | "straight"
    offset_m: float
    worst_clearance_m: float = float("inf")
    worst_at_s: float = 0.0
    worst_body: str = ""
    static_gap_m: float = float("inf")
    static_body: str = ""
    reachable: bool = True
    reject_reason: str = ""
    # The WORLD-SPACE line this corridor represents, captured when it was
    # scored: the point the offset was measured from and the unit direction to
    # the goal.  Stored because a corridor has to survive being committed to.
    #
    # THIS IS A SCAR.  A first draft rebuilt the pursuit target every tick from
    # the duck's CURRENT position, so the "corridor" was always 0.26 m to the
    # side of wherever the duck happened to be - a line that receded exactly as
    # fast as the duck approached it.  The lateral error never fell below the
    # 0.12 m arrival test, CHOOSE_RIGHT ran into its 12 s ceiling, and the duck
    # crabbed sideways across the floor without ever being "on" the corridor it
    # had chosen.  A corridor is a place, not an offset from yourself.
    origin: np.ndarray | None = None
    direction: np.ndarray | None = None

    def line_point(self, along_m: float) -> np.ndarray:
        """A point ``along_m`` further down this corridor's fixed world line."""
        if self.origin is None or self.direction is None:
            raise ValueError("corridor has no world line; it was never scored")
        normal = np.array([-self.direction[1], self.direction[0]])
        return (self.origin + self.direction * along_m
                + normal * self.offset_m)

    @property
    def safe(self) -> bool:
        return not self.reject_reason

    def as_record(self) -> dict:
        return {
            "side": self.side,
            "offset_m": round(float(self.offset_m), 4),
            "worst_clearance_m": (
                None if not np.isfinite(self.worst_clearance_m)
                else round(float(self.worst_clearance_m), 4)),
            "worst_at_s": round(float(self.worst_at_s), 3),
            "worst_body": self.worst_body,
            "static_gap_m": (
                None if not np.isfinite(self.static_gap_m)
                else round(float(self.static_gap_m), 4)),
            "static_body": self.static_body,
            "reachable": bool(self.reachable),
            "safe": bool(self.safe),
            "reject_reason": self.reject_reason,
            "origin": (None if self.origin is None
                       else [round(float(self.origin[0]), 4),
                             round(float(self.origin[1]), 4)]),
        }


@dataclass
class Decision:
    """The chosen side, the rejected one, and why — all measured.

    ``rejected`` always carries the best corridor on the losing hand, so the
    justification is a comparison of two numbers rather than an assertion.
    """

    side: str                 # "left" | "right" | "straight" | "wait"
    corridor: Corridor | None = None
    rejected: Corridor | None = None
    threat: str = ""
    threat_ttc_s: float = float("inf")
    threat_range_m: float = float("inf")
    all_corridors: list[Corridor] = field(default_factory=list)

    def as_record(self) -> dict:
        return {
            "side": self.side,
            "chosen": None if self.corridor is None else self.corridor.as_record(),
            "rejected": (None if self.rejected is None
                         else self.rejected.as_record()),
            "threat": self.threat,
            "threat_ttc_s": (None if not np.isfinite(self.threat_ttc_s)
                             else round(float(self.threat_ttc_s), 3)),
            "threat_range_m": (None if not np.isfinite(self.threat_range_m)
                               else round(float(self.threat_range_m), 4)),
            "corridors": [c.as_record() for c in self.all_corridors],
        }


def horizon_times() -> list[float]:
    """The prediction sample times, in seconds ahead of now."""
    return [(i + 1) * PREDICT_DT_S for i in range(PREDICT_SAMPLES)]


def predict_occupancy(tracks: list[Track]) -> list[dict]:
    """Every tracked body's predicted position at every horizon sample.

    Returned as plain records so the HUD can draw exactly what the planner
    scored, and so a test can compare a prediction against what later happened
    without re-deriving it.
    """
    out: list[dict] = []
    for dt in horizon_times():
        out.append({
            "dt_s": round(float(dt), 3),
            "bodies": {t.name: t.predict(dt).tolist() for t in tracks},
        })
    return out


def duck_line(duck_xy, offset_m: float, goal_xy=GOAL_XY) -> tuple[np.ndarray, np.ndarray]:
    """The duck's candidate line: from where it is, to the goal, offset laterally.

    The offset is applied PERPENDICULAR to the heading toward the goal, so a
    corridor is a parallel lane rather than a rotation, and its lateral meaning
    does not change as the duck advances.
    """
    start = np.asarray(duck_xy, dtype=np.float64)[:2]
    goal = np.asarray(goal_xy, dtype=np.float64)[:2]
    span = goal - start
    length = float(np.linalg.norm(span))
    if length < 1e-9:
        return start.copy(), np.array([1.0, 0.0])
    direction = span / length
    normal = np.array([-direction[1], direction[0]])
    return start + normal * offset_m, direction


def duck_at(duck_xy, offset_m: float, dt: float,
            speed: float = SPEED_AT_WALK, goal_xy=GOAL_XY) -> np.ndarray:
    """Where the duck would be at ``dt`` if it walked this corridor now.

    The lateral offset is ramped in at the MEASURED lateral rate rather than
    applied instantly: the duck cannot teleport sideways, and a predictor that
    assumed it could would score a corridor the robot can never occupy.
    """
    base, direction = duck_line(duck_xy, 0.0, goal_xy)
    normal = np.array([-direction[1], direction[0]])
    reached = min(abs(offset_m), LATERAL_RATE_MPS * dt)
    lateral = math.copysign(reached, offset_m)
    return base + direction * (speed * dt) + normal * lateral


def score_corridor(duck_xy, offset_m: float, side: str, tracks: list[Track],
                   goal_xy=GOAL_XY, ttc_s: float = float("inf")) -> Corridor:
    """Worst predicted clearance along one candidate lane, plus its statics.

    Every disqualification is recorded rather than short-circuited, so the
    decision record can explain a refusal.

    THE PREDICTED CLEARANCE IS DELIBERATELY PESSIMISED, AND THAT IS A
    MEASURED CORRECTION.
    The duck is treated as a disc of :data:`~slalom_states.DUCK_PLANAR_RADIUS`
    and each body's predicted position is inflated by
    :data:`~slalom_states.PREDICTION_SLOP_M`.  Without both, the planner's
    promise was routinely OPTIMISTIC against what was later measured: over the
    first full run, predicted 0.630 m against a measured 0.249 m for ``mara``
    and 0.817 m against 0.252 m for ``dev``.

    The cause is not a bad model - it is that a predicted CENTRE-to-centre gap
    is not a surface-to-surface clearance, and that a body walking a filleted
    route covers more ground than a straight-line extrapolation of its current
    velocity says.  A safety prediction has to err toward under-promising, so
    the score subtracts the duck's own radius and a slop term sized to the
    MEASURED worst-case error of the constant-velocity model over the horizon.
    """
    corridor = Corridor(side=side, offset_m=offset_m)
    base, direction = duck_line(duck_xy, 0.0, goal_xy)
    corridor.origin = base.copy()
    corridor.direction = direction.copy()

    # -- moving bodies, over the whole horizon -------------------------------
    for dt in horizon_times():
        here = duck_at(duck_xy, offset_m, dt, goal_xy=goal_xy)
        for track in tracks:
            gap = (float(np.linalg.norm(track.predict(dt) - here))
                   - track.radius - DUCK_PLANAR_RADIUS - PREDICTION_SLOP_M)
            if gap < corridor.worst_clearance_m:
                corridor.worst_clearance_m = gap
                corridor.worst_at_s = dt
                corridor.worst_body = track.name

    # -- static bodies, along the whole candidate lane ------------------------
    # Sampled over the reachable part of the lane rather than to the goal: a
    # crate 6 m away does not disqualify a sidestep taken now.
    lane_end = duck_at(duck_xy, offset_m, horizon_times()[-1], goal_xy=goal_xy)
    for obstacle in STATIC_OBSTACLES:
        gap = min(obstacle.distance_to(duck_at(duck_xy, offset_m, dt,
                                               goal_xy=goal_xy))
                  for dt in horizon_times())
        gap = min(gap, obstacle.distance_to(lane_end))
        if gap < corridor.static_gap_m:
            corridor.static_gap_m = gap
            corridor.static_body = obstacle.name

    # -- can the duck actually get there in time? -----------------------------
    # RECORDED, NOT VETOED, AND THAT DISTINCTION WAS A BUG.
    #
    # :func:`duck_at` already ramps the lateral offset in at the MEASURED
    # :data:`~slalom_states.LATERAL_RATE_MPS`, so a corridor the duck cannot
    # reach in time is one whose PREDICTED DUCK POSITION never gets far enough
    # sideways - and its worst-case clearance is scored against that true
    # position.  Reachability is therefore ALREADY INSIDE THE SCORE.
    #
    # A first draft ALSO vetoed on it, which double-counted: every corridor was
    # rejected as "unreachable" against a time-to-conflict shorter than the
    # manoeuvre itself, and the planner answered "wait" to a single body 1.3 m
    # away on an otherwise empty floor.  The flag is kept because it explains
    # WHY a corridor scored badly, but it no longer disqualifies one on its own.
    if np.isfinite(ttc_s) and abs(offset_m) > 1e-9:
        corridor.reachable = (
            abs(offset_m) / LATERAL_RATE_MPS) <= max(ttc_s, 0.0)
    else:
        corridor.reachable = True

    # -- the two ways to lose, in priority order ------------------------------
    #
    # A THIRD, SUBTLER ONE COMES FIRST: A CORRIDOR WHOSE WORST MOMENT IS THE
    # LAST HORIZON SAMPLE HAS NOT BEEN SCORED, IT HAS BEEN TRUNCATED.
    #
    # When the worst predicted clearance occurs exactly at the end of the
    # horizon, the conflict is still getting worse when the prediction stops
    # looking - so the number is an artifact of where the horizon was cut, not a
    # property of the corridor.  MEASURED: the duck engaged ``mara`` while she
    # was 2.2 m south of the lane, both corridors bottomed out at the 7.0 s
    # sample, the north scored +0.332 m on that truncated view, the duck
    # committed LEFT - and she then walked north into exactly that corridor,
    # producing a MEASURED -0.085 m overlap.  Every one of the four bad passes
    # in that run had its worst moment at the horizon edge.
    #
    # Refusing to call such a corridor safe means the duck keeps walking and
    # re-scores when the whole crossing fits inside the horizon, which is the
    # only point at which the score means anything.
    at_edge = abs(corridor.worst_at_s - horizon_times()[-1]) < 1e-9
    if at_edge and corridor.worst_clearance_m < TRUNCATED_SAFE_M:
        corridor.reject_reason = (
            f"truncated: the worst moment is the last horizon sample "
            f"(+{corridor.worst_at_s:.1f} s), so {corridor.worst_body} is still "
            f"closing when the prediction stops; not scored, cut off")
    elif corridor.static_gap_m < STATIC_MARGIN_M:
        corridor.reject_reason = (
            f"static: {corridor.static_gap_m:.3f} m to {corridor.static_body} "
            f"is inside the {STATIC_MARGIN_M:.2f} m margin")
    elif corridor.worst_clearance_m < SAFE_CLEARANCE_M:
        corridor.reject_reason = (
            f"unsafe: predicted {corridor.worst_clearance_m:.3f} m to "
            f"{corridor.worst_body} at +{corridor.worst_at_s:.1f} s, below the "
            f"{SAFE_CLEARANCE_M:.2f} m bar")
    return corridor


def nearest_threat(duck_xy, duck_heading, tracks: list[Track],
                   goal_xy=GOAL_XY) -> tuple[str, float, float]:
    """The most urgent body predicted to conflict, its TTC and its range.

    A body counts as a threat only if it comes within
    :data:`~slalom_states.THREAT_CLEARANCE_M` of the duck's UNOFFSET line within
    the horizon, and is within :data:`~slalom_states.THREAT_RANGE_M` along the
    course.  Returns ``("", inf, inf)`` when the way ahead is clear.

    TTC is the horizon sample at which the predicted clearance is worst, which
    is the quantity the corridor reachability report needs: how long the duck
    has before the conflict actually arrives.
    """
    best_name, best_gap = "", float("inf")
    best_ttc, best_range = float("inf"), float("inf")
    for dt in horizon_times():
        here = duck_at(duck_xy, 0.0, dt, goal_xy=goal_xy)
        for track in tracks:
            gap = (float(np.linalg.norm(track.predict(dt) - here))
                   - track.radius - DUCK_PLANAR_RADIUS)
            range_m = float(np.linalg.norm(
                track.pos - np.asarray(duck_xy, dtype=np.float64)[:2]))
            if gap < best_gap and range_m <= THREAT_RANGE_M:
                best_gap, best_name, best_ttc, best_range = (
                    gap, track.name, dt, range_m)
    if best_name and best_gap <= THREAT_CLEARANCE_M:
        return best_name, best_ttc, best_range
    return "", float("inf"), float("inf")


def choose_corridor(duck_xy, tracks: list[Track], *, ttc_s: float = float("inf"),
                    goal_xy=GOAL_XY, threat: str = "",
                    threat_range_m: float = float("inf")) -> Decision:
    """Score every corridor on BOTH hands and pick the best survivor.

    Left candidates are positive offsets and right candidates negative, matching
    the world-frame convention: with the course running toward +x, +y is the
    duck's left.  ``tools/check_course.py`` pins that convention against a
    measured turn rather than trusting the label.

    The winner is the surviving corridor with the greatest worst-case predicted
    clearance; ties break toward the SMALLER sidestep, because a smaller one is
    cheaper in both course and time.  When nothing survives on either hand the
    answer is ``"wait"``.
    """
    left = [score_corridor(duck_xy, +offset, "left", tracks, goal_xy, ttc_s)
            for offset in LATERAL_OFFSETS]
    right = [score_corridor(duck_xy, -offset, "right", tracks, goal_xy, ttc_s)
             for offset in LATERAL_OFFSETS]
    everything = left + right

    def best_of(candidates: list[Corridor]) -> Corridor | None:
        safe = [c for c in candidates if c.safe]
        if not safe:
            return None
        return max(safe, key=lambda c: (round(c.worst_clearance_m, 3),
                                        -abs(c.offset_m)))

    best_left, best_right = best_of(left), best_of(right)

    def fallback(candidates: list[Corridor]) -> Corridor:
        """The least-bad corridor on a hand where nothing survived.

        Reported so a WAIT still names what it rejected and by how much.
        """
        return max(candidates, key=lambda c: c.worst_clearance_m)

    if best_left is None and best_right is None:
        return Decision(side="wait", corridor=None,
                        rejected=fallback(everything), threat=threat,
                        threat_ttc_s=ttc_s, threat_range_m=threat_range_m,
                        all_corridors=everything)

    if best_left is not None and best_right is not None:
        if (round(best_left.worst_clearance_m, 3),
                -abs(best_left.offset_m)) >= (
                round(best_right.worst_clearance_m, 3),
                -abs(best_right.offset_m)):
            chosen, other = best_left, best_right
        else:
            chosen, other = best_right, best_left
    elif best_left is not None:
        chosen, other = best_left, fallback(right)
    else:
        chosen, other = best_right, fallback(left)

    return Decision(side=chosen.side, corridor=chosen, rejected=other,
                    threat=threat, threat_ttc_s=ttc_s,
                    threat_range_m=threat_range_m, all_corridors=everything)


def lane_intrusion_m(xy) -> float:
    """How far into the nominal lane band a point sits.  Negative is outside.

    Used only for reporting which bodies are in the duck's way at all; the
    decisions are made on predicted clearance, not on this.
    """
    return LANE_HALF_W - abs(float(np.asarray(xy, dtype=np.float64)[1]))
