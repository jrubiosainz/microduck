#!/usr/bin/env python3
"""Encounter prediction, alcove scoring, and the measured locomotion constants.

Pure geometry and pure Python/numpy: no MuJoCo, no ONNX, no rendering.  This is
the decision layer and it is fully unit-tested in ``tests/``.

What "the duck predicts an encounter" actually means
----------------------------------------------------
The predictor is an honest **simulator semantic/geometric proxy for pedestrian
perception**.  Each adult's position, velocity and body size come from the
simulator; there is no detector, no tracker and no time-to-collision estimated
from pixels anywhere in this behavior.  What IS real is the camera geometry —
tracking the passing adult in the PiP is measured through the exact frustum the
PiP renders from, with occlusion ray casts against actual scene geometry.

The prediction itself
---------------------
The corridor is one-dimensional, so the encounter is too.  Both bodies are
reduced to intervals along **x**, and the questions are *where* and *when* they
would meet:

* the **meeting station** is where the two paths cross, integrating the duck's
  own measured cruise speed forward rather than assuming it stops;
* the **time to meet** is the closing range divided by the closing speed;
* the **counterfactual clearance** is the surface gap the pass would have had
  if the duck simply kept walking down the centreline.  That number is the
  reason to act, and it is recorded at the moment of detection so it cannot be
  reconstructed favourably afterwards.

A pull-over is triggered when the predicted encounter is close enough in time
AND the counterfactual pass is not safe.  Both conditions are necessary: an
adult who is far away is not yet this decision's business, and an adult the
duck could pass safely does not need one.  In this corridor the second
condition is *structurally* true — ``corridor.corridor_passing_geometry``
reports that no side-by-side pass clears the required gap anywhere in the plain
corridor — but the predictor still evaluates it per encounter rather than
assuming it, so the same code would decline to pull over in a wider hallway.

Choosing where to go
--------------------
Every alcove is scored against the SAME predicted meeting, and a candidate must
satisfy two independent requirements:

* **physical clearance** — the duck's whole footprint must fit inside the
  recess *and* out of the centre passage.  This is computed from the recess's
  usable depth, which accounts for obstructions, so a bay full of crates fails
  here rather than being special-cased by name;
* **reachability** — the duck must be able to get there, settle, and be
  stationary before the adult arrives, using the MEASURED lateral speed and a
  measured settle time.  An alcove the duck would still be entering as the
  adult walks past is refused however roomy it is.

Among the survivors the choice is the one that leaves the largest time margin,
with the meeting station used to break ties.  Every rejection is recorded with
its reason and its numbers, so "the duck rejected the blocked bay" is evidence
rather than narration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from corridor import (
    ADULT_PLANAR_RADIUS,
    ALCOVES,
    Alcove,
    CENTER_PASSAGE_HALF,
    CLEAR_ABS_Y,
    DESTINATION_X,
    DUCK_LATERAL_HALF,
    DUCK_PLANAR_RADIUS,
    counterfactual_pass_clearance,
)

# --- measured locomotion constants -------------------------------------------
# MEASURED on scene_narrow_corridor.xml with the stock alpha_walking policy at
# action scale 0.9 and the real imu_ang_vel sensor (tools/sweep_commands.py,
# 4 s and 6 s rollouts from the corridor start).
#
# FORWARD GAIT ONSET IS A CLIFF, NOT A RAMP:
#   vx=0.16 -> 0.008 m in 6 s   (no gait)
#   vx=0.20 -> 0.010 m in 6 s   (no gait)
#   vx=0.24 -> 0.516 m in 6 s   (walking, 0.086 m/s)
# A command below onset produces NO motion, so the duck cannot creep down the
# corridor: it walks at speed or it stands still.
VX_MIN_EFFECTIVE: float = 0.24
# The corridor cruise.  MEASURED vx=0.36 -> 0.819 m in 6 s (0.136 m/s forward),
# min trunk z 0.112, final 0.115.  Faster is available (vx=0.52 reaches
# 0.253 m/s) but costs heading stability in a 0.42 m corridor: measured yaw
# drift grows from -17.4 deg at 0.36 to -9.7 deg at 0.52 with a 0.27 m lateral
# excursion, and a cruise that wanders is a cruise that scrapes a wall.
VX_CRUISE: float = 0.36
# The approach to a chosen alcove mouth, slower so the arrival is not
# overshot.  MEASURED vx=0.28 -> 0.639 m in 6 s (0.106 m/s).
VX_APPROACH: float = 0.28

# LATERAL GAIT ONSET IS ALSO A CLIFF, AND IT IS ASYMMETRIC.
# MEASURED pure-lateral commands over 4-6 s:
#   vy=+0.20 -> 0.003 m (no gait)     vy=-0.20 -> 0.002 m (no gait)
#   vy=+0.28 -> 0.005 m (no gait)     vy=-0.24 -> 0.000 m (no gait)
#   vy=+0.30 -> 0.186 m (walking)     vy=-0.26 -> 0.149 m (walking)
# So the two signs cross onset at different magnitudes: about -0.26 to the
# right and +0.30 to the left.  A single symmetric threshold would either stall
# one direction or over-drive the other.
VY_MIN_EFFECTIVE_LEFT: float = 0.30
VY_MIN_EFFECTIVE_RIGHT: float = 0.26

# THE LATERAL COMMAND IS STRONGLY YAW-COUPLED, AND ONLY ON ONE SIDE.
# MEASURED over 4 s of pure lateral command:
#   vy=+0.46, wz=0    ->  +0.476 m sideways,  yaw   -1.3 deg
#   vy=-0.46, wz=0    ->  -0.402 m sideways,  yaw  +93.6 deg  (!!)
#   vy=-0.46, wz=-0.45 -> -0.388 m sideways,  yaw   -5.8 deg
#   vy=-0.60, wz=-0.45 -> -0.522 m sideways,  yaw   +5.3 deg
#   vy=-0.70, wz=-0.45 -> -0.628 m sideways,  yaw   -5.4 deg
# Stepping RIGHT with no yaw command spins the duck through ninety degrees in
# four seconds; stepping LEFT barely rotates it at all.  In a 0.42 m corridor a
# ninety-degree spin puts the duck's nose into a wall, so the right-hand step
# carries a large FEED-FORWARD yaw term and the left-hand step carries almost
# none.  The two signs are therefore not mirror images and never share a gain.
VY_PULLOVER_LEFT: float = 0.60
VY_PULLOVER_RIGHT: float = -0.60
WZ_FEEDFORWARD_LEFT: float = -0.05
WZ_FEEDFORWARD_RIGHT: float = -0.45
# MEASURED lateral ground speeds from the REAL closed-loop pull-over primitive
# (tools/measure_pullover.py), which is what the reachability estimate must be
# built on rather than an open-loop sweep:
#   into bay_open  (step RIGHT): 0.367 m in 2.22 s after onset = 0.1653 m/s
#   into bay_far   (step LEFT):  0.367 m in 2.82 s after onset = 0.1301 m/s
# The SLOWER of the two is used for every estimate, because using the faster
# one would make a marginal alcove look reachable when approached from the
# wrong side.  Rounded down to 0.128 so the constant can never exceed the
# measurement, and a test pins that relation.
VY_SPEED_MPS: float = 0.128

# YAW AUTHORITY WHILE CRUISING is asymmetric too.  MEASURED at vx=0.40 over 6 s:
#   wz=+0.08 ->  +7.3 deg     wz=-0.08 -> -29.5 deg
#   wz=+0.16 -> +50.1 deg     wz=-0.16 -> -48.7 deg
#   wz=+0.24 -> +61.1 deg     wz=-0.24 -> -66.9 deg
# and the zero-command drift at cruise is itself -10.0 deg over 6 s.  The right
# side is roughly four times stronger for a small command, so the two signs get
# independent gains and independent dead zones; mirroring one onto the other
# would make every right correction a violent over-correction.
KP_YAW_LEFT: float = 1.15
KP_YAW_RIGHT: float = 0.34
WZ_MAX_LEFT: float = 0.16
WZ_MAX_RIGHT: float = 0.09
# Below these the measured yaw response is lost in the gait's own variation, so
# emitting them burns control ticks without steering.
WZ_MIN_LEFT: float = 0.05
WZ_MIN_RIGHT: float = 0.03

# MEASURED settle: how long after the lateral command is released the trunk
# keeps drifting before it is stationary.  From the real pull-over primitive:
# 0.18 s and 0.16 s, drifting a further 0.031 m and 0.023 m.  0.40 s carries
# both with margin and is applied to every reachability estimate.
SETTLE_S: float = 0.40
# MEASURED gait-onset dead time for the lateral step: the command must cross
# onset and the gait must start before the trunk moves at all.  Measured 0.08 s
# and 0.10 s on the two sides; 0.25 s carries both with margin.
LATERAL_DEAD_TIME_S: float = 0.25

# THE TWO LEGS OF A PULL-OVER HAPPEN AT THE SAME TIME, NOT ONE AFTER THE OTHER.
# The controller drives vx and vy together, so charging the forward leg and the
# lateral leg sequentially is not conservatism — it describes a different
# manoeuvre, and it rejects alcoves the duck reaches comfortably.  MEASURED by
# tools/measure_pullover.py, which times the REAL controller from a lead
# distance short of the mouth to a parked, stationary duck, against the longer
# of the two modelled legs:
#
#   lead      bay_open            bay_far
#   (m)    parked  ratio       parked  ratio
#   0.10    2.52   0.88         3.68   1.28   <- worst
#   0.30    2.50   0.87         3.18   1.11
#   0.60    4.60   0.96         5.20   1.08
#   0.90    6.56   0.91         7.42   1.03
#   1.30    9.44   0.91        10.26   0.99
#   1.80   13.26   0.92        13.94   0.97
#   2.40   17.56   0.91        18.22   0.95
#
# The right-hand entry into ``bay_far`` is the expensive one, exactly as the
# command sweep predicted: that side carries the large yaw coupling and spends
# part of its budget fighting it.  The whole manoeuvre costs the LONGER of the
# two legs scaled by a factor that must exceed the worst measured ratio of
# 1.28; 1.30 clears every one of them, and a test pins the constant above all
# fourteen measurements rather than pinning the number itself.
CONCURRENT_LEG_PESSIMISM: float = 1.30

# --- prediction --------------------------------------------------------------
# The duck's MEASURED ground speed at cruise, which is what the predictor
# integrates.  Integrating the COMMAND instead would over-state the duck's
# progress by nearly a factor of three (0.36 commanded against 0.140 measured)
# and put every predicted meeting station well past where it really is.
# MEASURED closed-loop over 12 s: 1.677 m, i.e. 0.1397 m/s.
CRUISE_SPEED_MPS: float = 0.140
# Ground speed used for the along-corridor leg of every reachability estimate.
# The pull-over walks at CRUISE speed until it is close to the alcove mouth and
# only then slows, so the honest figure is near the cruise speed rather than
# the slow final approach.  MEASURED effective forward speed over the whole
# manoeuvre (tools/measure_pullover.py, leads of 0.6-2.4 m): 0.130-0.137 m/s.
# 0.125 is below every one of them, so the estimate cannot flatter a marginal
# alcove, and a test pins the constant under the measured minimum.
APPROACH_SPEED_MPS: float = 0.125

# How far ahead the predictor looks.  Beyond this an adult is not yet a
# participant in this decision.
PREDICT_HORIZON_S: float = 20.0
# An encounter must be predicted within this long to trigger a pull-over.  It
# is not a taste parameter: it must be long enough to complete the manoeuvre.
# The worst case is the deepest usable alcove entered from the centreline:
# 0.3670 m of lateral travel at the MEASURED 0.128 m/s is 2.87 s, which at the
# concurrency factor costs 3.73 s, plus 0.25 s of lateral gait-onset dead time
# and 0.40 s of settling = 4.38 s, plus the 0.60 s DETECT dwell and the 0.50 s
# SELECT dwell = 5.48 s.  Detection at 9.0 s therefore leaves 3.5 s of slack
# for the walk to the alcove's mouth.  A test pins the relation to the measured
# constants rather than the number.
DETECT_HORIZON_S: float = 9.0
# Required time margin between the duck being settled in an alcove and the
# adult's body reaching that alcove's mouth.  Justified from measurement: the
# duck's own settle is 0.90 s and the adult covers 0.42 m/s x 0.8 s = 0.34 m in
# this margin, which is one full adult body length of visible daylight.
REACH_MARGIN_S: float = 0.80
# How close the adult may come before the situation counts as unsafe proximity.
# The pull-over must be DECIDED before this, which is what "detects the
# encounter before unsafe proximity" means as a number.  Set to the range at
# which a side-by-side pass would already be committed: two adult body lengths.
UNSAFE_PROXIMITY_M: float = 0.70

# How long the machine waits before re-opening a decision that found no viable
# alcove.  Without it the machine oscillates between CRUISE and DETECT at the
# dwell period, which is both unreadable in the state trace and pointless: the
# geometry cannot change in 0.6 s.  Walking on for a couple of seconds brings
# new bays into reach, which is the only thing that can change the answer.
REDETECT_COOLDOWN_S: float = 2.0

# --- yielding ----------------------------------------------------------------
# The adult must be past the duck AND opening the range by at least this much
# before the duck may rejoin.  JUSTIFIED: the duck's own rejoin manoeuvre takes
# it back through the centre passage, which at the MEASURED lateral speed takes
# 0.3670 / 0.128 = 2.87 s (measured directly: 2.10 s and 1.62 s).  An adult
# receding at 0.42 m/s covers 1.21 m in that time, so requiring 0.55 m of range
# before starting means the adult is never closer than its own body length
# while the duck is moving back into the corridor.  A test pins the relation.
CLEAR_RANGE_M: float = 0.55
# The adult's range must be increasing for this long before it counts as
# receding, so one noisy tick cannot release the yield.
RECEDING_CONFIRM_S: float = 0.30

# --- timings -----------------------------------------------------------------
DETECT_HOLD_S: float = 0.60      # dwell in DETECT before selecting
SELECT_HOLD_S: float = 0.50      # dwell in SELECT_ALCOVE before moving
YIELD_MIN_S: float = 0.60        # a yield is never instantaneous
CLEAR_HOLD_S: float = 0.50       # dwell in CLEAR before rejoining
DONE_HOLD_S: float = 0.0
# Hard ceilings, so a rollout cannot hang.  Exceeding one is recorded as a
# timeout, which the gate treats as a failure rather than papering over.
PULL_OVER_MAX_S: float = 9.0
YIELD_MAX_S: float = 14.0
REJOIN_MAX_S: float = 9.0
RESUME_MAX_S: float = 30.0
CRUISE_MAX_S: float = 30.0

STATES = (
    "CRUISE", "DETECT", "SELECT_ALCOVE", "PULL_OVER", "YIELD",
    "CLEAR", "REJOIN", "RESUME", "DONE",
)
# Every state in which the locomotion command must be EXACTLY zero.
STATIONARY_STATES = ("YIELD", "CLEAR", "DONE")
MOVING_STATES = ("CRUISE", "PULL_OVER", "REJOIN", "RESUME")
# DETECT and SELECT_ALCOVE keep walking: stopping dead in the middle of a
# corridor the moment a person appears is not etiquette, it is an obstruction.
# The duck decides while still making progress, which is also what makes the
# reachability arithmetic non-trivial.


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Encounter:
    """One predicted meeting between the duck and one adult."""

    name: str
    range_m: float
    closing_speed_mps: float
    time_to_meet_s: float
    meet_x: float
    adult_x: float
    adult_y: float
    adult_speed_mps: float
    adult_direction: float
    counterfactual_clearance_m: float
    head_on: bool
    approaching: bool

    @property
    def imminent(self) -> bool:
        return self.approaching and self.time_to_meet_s <= DETECT_HORIZON_S

    @property
    def unsafe_to_share(self) -> bool:
        """Would passing without pulling over be too tight?"""
        return self.counterfactual_clearance_m < 0.0

    def as_record(self) -> dict:
        return {
            "person": self.name,
            "range_m": self.range_m,
            "closing_speed_mps": self.closing_speed_mps,
            "time_to_meet_s": self.time_to_meet_s,
            "meet_x": self.meet_x,
            "adult_x": self.adult_x,
            "adult_y": self.adult_y,
            "adult_speed_mps": self.adult_speed_mps,
            "adult_direction": self.adult_direction,
            "counterfactual_clearance_m": self.counterfactual_clearance_m,
            "head_on": self.head_on,
        }


def predict_encounter(
    person,
    duck_xy,
    duck_speed: float = CRUISE_SPEED_MPS,
    *,
    horizon: float = PREDICT_HORIZON_S,
) -> Encounter:
    """Where and when the duck would meet one adult if neither changed course.

    ``duck_speed`` is the duck's MEASURED ground speed, not its command.

    An adult walking the same way as the duck but slower is still an encounter
    if the duck is catching it, and an adult walking away faster is not one at
    all.  Both fall out of the closing-speed arithmetic rather than needing a
    head-on special case, which is what lets the same predictor handle the
    overtaking encounter without a second code path.
    """
    duck_xy = np.asarray(duck_xy, dtype=np.float64)
    duck_x, duck_y = float(duck_xy[0]), float(duck_xy[1])
    adult_x = float(person.pos[0])
    adult_y = float(person.pos[1])
    adult_vx = float(person.vel[0])

    separation = adult_x - duck_x
    ahead = separation >= 0.0
    # Positive closing speed means the gap is shrinking, whichever side the
    # adult is on and whichever way it is walking.
    closing = (duck_speed - adult_vx) if ahead else (adult_vx - duck_speed)
    range_m = abs(separation) - person.half_length - DUCK_PLANAR_RADIUS
    counterfactual = counterfactual_pass_clearance(duck_y, adult_y)
    # Head-on means the adult is walking toward the duck rather than away.
    head_on = (adult_vx < 0.0) if ahead else (adult_vx > 0.0)

    if closing <= 1e-6:
        return Encounter(
            name=person.name, range_m=range_m, closing_speed_mps=closing,
            time_to_meet_s=float("inf"), meet_x=float("nan"),
            adult_x=adult_x, adult_y=adult_y, adult_speed_mps=abs(adult_vx),
            adult_direction=float(person.direction),
            counterfactual_clearance_m=counterfactual,
            head_on=head_on, approaching=False,
        )

    time_to_meet = max(0.0, abs(separation) / closing)
    meet_x = duck_x + duck_speed * min(time_to_meet, horizon)
    return Encounter(
        name=person.name, range_m=range_m, closing_speed_mps=closing,
        time_to_meet_s=time_to_meet, meet_x=meet_x,
        adult_x=adult_x, adult_y=adult_y, adult_speed_mps=abs(adult_vx),
        adult_direction=float(person.direction),
        counterfactual_clearance_m=counterfactual,
        head_on=head_on, approaching=True,
    )


def predict_encounters(people: dict, duck_xy,
                       duck_speed: float = CRUISE_SPEED_MPS) -> list[Encounter]:
    """Every adult's predicted encounter, soonest first."""
    encounters = [
        predict_encounter(person, duck_xy, duck_speed)
        for person in people.values()
    ]
    encounters.sort(key=lambda e: e.time_to_meet_s)
    return encounters


def most_urgent(people: dict, duck_xy,
                duck_speed: float = CRUISE_SPEED_MPS) -> Encounter | None:
    """The soonest encounter that is both imminent and unsafe to share."""
    for encounter in predict_encounters(people, duck_xy, duck_speed):
        if encounter.imminent and encounter.unsafe_to_share:
            return encounter
    return None


# --- alcove scoring ----------------------------------------------------------
@dataclass(frozen=True)
class AlcoveScore:
    """One alcove evaluated against one predicted encounter."""

    name: str
    side: int
    center_x: float
    usable_outer_y: float
    max_trunk_abs_y: float
    park_y: float
    clears_passage: bool
    clearance_headroom_m: float
    reachable: bool
    travel_time_s: float
    time_available_s: float
    time_margin_s: float
    behind: bool
    reasons: tuple[str, ...]

    @property
    def viable(self) -> bool:
        return not self.reasons

    def as_record(self) -> dict:
        return {
            "alcove": self.name,
            "side": self.side,
            "center_x": self.center_x,
            "usable_outer_y": self.usable_outer_y,
            "max_trunk_abs_y": self.max_trunk_abs_y,
            "park_y": self.park_y,
            "clears_passage": self.clears_passage,
            "clearance_headroom_m": self.clearance_headroom_m,
            "reachable": self.reachable,
            "behind": self.behind,
            "travel_time_s": self.travel_time_s,
            "time_available_s": self.time_available_s,
            "time_margin_s": self.time_margin_s,
            "viable": self.viable,
            "rejected_because": list(self.reasons),
        }


def score_alcove(
    alcove: Alcove,
    encounter: Encounter,
    duck_xy,
    *,
    forward_speed: float = APPROACH_SPEED_MPS,
    lateral_speed: float = VY_SPEED_MPS,
    settle_s: float = SETTLE_S,
    dead_time_s: float = LATERAL_DEAD_TIME_S,
    margin_s: float = REACH_MARGIN_S,
    concurrency: float = CONCURRENT_LEG_PESSIMISM,
) -> AlcoveScore:
    """Grade one alcove against one predicted encounter.

    Two independent requirements, and BOTH must hold:

    * **physical clearance** — the duck's whole footprint must fit inside the
      recess and out of the centre passage.  Computed from the recess's usable
      depth, so an obstruction fails here on geometry rather than by name.
    * **reachability** — the duck must arrive, settle and be stationary at
      least ``margin_s`` before the adult's body reaches the mouth.

    The travel estimate models the manoeuvre as MEASURED: the forward and
    lateral legs run concurrently, so the cost is the longer of the two scaled
    by ``CONCURRENT_LEG_PESSIMISM``, plus the gait-onset dead time and the
    settle.  It is deliberately pessimistic in four independent ways — the
    concurrency factor exceeds every measured ratio, the lateral leg uses the
    SLOWER of the two measured lateral speeds, the forward speed is below the
    measured clean approach, and the along-corridor leg is measured to the near
    edge of the mouth rather than to its centre.
    """
    duck_xy = np.asarray(duck_xy, dtype=np.float64)
    duck_x, duck_y = float(duck_xy[0]), float(duck_xy[1])
    reasons: list[str] = []

    # -- physical clearance ------------------------------------------------
    if not alcove.clears_passage:
        if alcove.blocked:
            reasons.append("obstructed: usable depth too small")
        else:
            reasons.append("too shallow: footprint cannot clear the passage")
    if alcove.x_headroom_m < 0.0:
        reasons.append("mouth shorter than the duck's footprint")

    # -- reachability -------------------------------------------------------
    # The duck only ever moves along +x here, and the controller begins its
    # lateral step at the alcove's ``entry_x``.  The estimate uses THE SAME
    # station, so the duck is never scored against a point it does not drive
    # to — scoring against the alcove's centre while the controller stops at
    # its near end rejects bays the duck reaches comfortably.
    target_x = alcove.entry_x
    forward_distance = target_x - duck_x
    behind = forward_distance < -1e-9
    if behind:
        # The duck has no reverse primitive in this behavior, so an alcove it
        # has already walked past is genuinely unavailable.
        reasons.append("behind the duck")
        travel_time = float("inf")
    else:
        forward_time = (forward_distance / forward_speed
                        if forward_distance > 1e-9 else 0.0)
        lateral_time = abs(alcove.park_y - duck_y) / lateral_speed
        travel_time = (
            dead_time_s + settle_s
            + concurrency * max(forward_time, lateral_time)
        )

    # How long the duck has: when the adult's body reaches this alcove's mouth.
    time_available = _time_until_adult_reaches(encounter, alcove)
    time_margin = time_available - travel_time
    reachable = bool(time_margin >= margin_s)
    if not reachable and not behind:
        reasons.append(
            f"not reachable in time: needs {travel_time:.2f}s, "
            f"has {time_available:.2f}s")

    return AlcoveScore(
        name=alcove.name, side=alcove.side, center_x=alcove.center_x,
        usable_outer_y=alcove.usable_outer_y,
        max_trunk_abs_y=alcove.max_trunk_abs_y, park_y=alcove.park_y,
        clears_passage=alcove.clears_passage,
        clearance_headroom_m=alcove.clearance_headroom_m,
        reachable=reachable, travel_time_s=travel_time,
        time_available_s=time_available, time_margin_s=time_margin,
        behind=behind, reasons=tuple(reasons),
    )


def _time_until_adult_reaches(encounter: Encounter, alcove: Alcove) -> float:
    """When the adult's body first reaches this alcove's mouth.

    Measured to the edge of the mouth the adult arrives at, inflated by the
    adult's own half-length, so "the duck was settled before the adult got
    there" covers the adult's whole body rather than its centre point.  An
    adult that has already passed the mouth returns 0.0, which correctly makes
    that alcove unreachable rather than infinitely available.

    NOTE that this is deliberately the MOUTH, not the duck's park station.  A
    duck tucked into the back of a recess is safe long before the adult draws
    level with it, but requiring it to be settled before the adult reaches the
    opening is the stricter and more honest test: it is the point from which
    the person can see into the bay.
    """
    low, high = alcove.x_span
    speed = encounter.adult_speed_mps
    if speed <= 1e-6:
        return float("inf")
    if encounter.adult_direction < 0.0:
        # Walking toward -x: it reaches the mouth's high edge first.
        distance = encounter.adult_x - (high + ADULT_PLANAR_RADIUS)
    else:
        distance = (low - ADULT_PLANAR_RADIUS) - encounter.adult_x
    return max(0.0, distance / speed)


@dataclass
class AlcoveDecision:
    """Everything behind one choice of pull-over zone."""

    selected: AlcoveScore | None
    candidates: tuple[AlcoveScore, ...]
    encounter: Encounter
    considered: int = 0
    rejected: tuple[AlcoveScore, ...] = field(default_factory=tuple)

    @property
    def viable(self) -> tuple[AlcoveScore, ...]:
        return tuple(c for c in self.candidates if c.viable)

    def as_record(self) -> dict:
        return {
            "encounter": self.encounter.as_record(),
            "considered": self.considered,
            "selected": self.selected.as_record() if self.selected else None,
            "candidates": [c.as_record() for c in self.candidates],
            "rejected": [c.as_record() for c in self.rejected],
            "viable_count": len(self.viable),
        }


def choose_alcove(
    encounter: Encounter,
    duck_xy,
    alcoves: tuple[Alcove, ...] = ALCOVES,
    **kwargs,
) -> AlcoveDecision:
    """Score every alcove against this encounter and pick the best viable one.

    The winner is the viable candidate with the largest time margin; ties are
    broken by the nearer mouth, so the duck does not walk past a perfectly good
    recess to reach an equally good one further away.  Every rejection keeps
    its reasons.
    """
    scores = tuple(
        score_alcove(alcove, encounter, duck_xy, **kwargs)
        for alcove in alcoves
    )
    viable = [s for s in scores if s.viable]
    rejected = tuple(s for s in scores if not s.viable)
    selected = None
    if viable:
        selected = max(
            viable, key=lambda s: (round(s.time_margin_s, 3), -s.center_x))
    return AlcoveDecision(
        selected=selected, candidates=scores, encounter=encounter,
        considered=len(scores), rejected=rejected,
    )
