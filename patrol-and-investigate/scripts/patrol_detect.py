#!/usr/bin/env python3
"""Detection and classification: what the duck decided each new thing WAS, from
what it measured itself.

THE CAMERA GATE IS THE WHOLE POINT
------------------------------------
A body may become a candidate ONLY while the duck can actually see it through
the real head camera - frustum containment plus a real MuJoCo occlusion ray cast
- and only after it has stayed visible for :data:`DETECT_CONFIRM_S`.  Nothing in
this module can be reached by a body the camera could not resolve, which is what
makes "it detected the intruder" a perception claim rather than a schedule
lookup.  ``feed`` takes the camera's own per-body visibility dictionary; it
cannot ask the world where anybody is.

THE THREE VERDICTS, AND THE RULES BEHIND THEM
-----------------------------------------------
Each rule is a decision over MEASURED geometric features, and each carries a
RULE MARGIN - how far past its own threshold the evidence sits - which is what
the reported confidence proxy is built from.

* **intrusion** - a PERSON whose measured position is inside the marked
  restricted rectangle, and who has stayed there for :data:`INTRUSION_DWELL_S`.
  The dwell is what separates an intruder from somebody walking past the edge of
  the zone, and it is measured from the duck's own successive observations.
  Margin: how deep inside the rectangle they are.

* **benign** - an OBJECT standing in a designated stow area, or an object with a
  person within :data:`ATTENDED_RADIUS_M` of it.  Either rule alone is enough,
  and the distractor in this scenario satisfies BOTH, which is what makes the
  dismissal robust rather than a coin flip.  Margin: how far inside the stow
  area, or how much nearer than the attendance radius the person is.

* **suspicious** - an OBJECT that is none of those things and has been
  stationary for :data:`UNATTENDED_S`.  It is the RESIDUAL category, deliberately:
  a guard robot should have to rule out the innocent explanations before it
  escalates, and ordering the rules this way means the escalation cannot fire on
  something a cheaper rule already explained.  Margin: how long past the
  unattended threshold it has stood there.

WHAT THIS IS NOT
------------------
Identity comes from the simulator, so this is a **semantic proxy** for object
and behaviour recognition rather than an RGB classifier, and the confidence is a
**rule-margin proxy** rather than a learned probability.  Both are labelled as
such wherever they surface - in the HUD, in the metrics and in the README.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from patrol_cast import BY_NAME, PERSON_NAMES
from patrol_facility import RESTRICTED_ZONE, stowed_on
from patrol_states import (
    ATTENDED_RADIUS_M,
    DETECT_CONFIRM_S,
    DETECT_MAX_RANGE_M,
    UNATTENDED_S,
)

# How long a person must remain inside the marked rectangle before it is called
# an intrusion.  DERIVED from the traffic: the fastest staff member covers the
# zone's 0.88 m width in 6.0 s, so a dwell longer than that cannot be produced by
# somebody merely crossing it.  It is the rule that makes the intrusion call a
# statement about BEHAVIOUR rather than about a single position sample.
INTRUSION_DWELL_S = 2.5
# How far a body may move between two observations and still count as standing
# in the same place.  DERIVED from the MEASURED traffic: an object posed
# analytically shows exactly 0.000 m of movement, while the slowest walking
# staff member covers 0.104 m/s - so at the 50 Hz observation rate a walking
# person moves 0.0021 m per tick and a parked object moves none.  0.02 m is an
# order of magnitude above the object and an order below anything a person does
# between two glances.
STATIONARY_STEP_M = 0.02
# The equivalent per-second rate, kept beside it because the two describe the
# same measurement and a reader deserves both.
STATIONARY_MPS = 0.02


@dataclass
class Observation:
    """What the duck has accumulated about one body it has seen.

    Every field is something the robot could have obtained: how long it has been
    visible, where it was measured, how long it has stood still, whether anybody
    was near it, and whether it was inside the marked rectangle.
    """

    name: str
    kind: str
    first_seen_s: float
    last_seen_s: float
    seen_s: float = 0.0
    position: tuple[float, float] = (0.0, 0.0)
    range_m: float = 0.0
    stationary_s: float = 0.0
    in_zone_s: float = 0.0
    nearest_person: str = ""
    nearest_person_m: float = float("inf")
    stow_area: str = ""
    zone_depth_m: float = -1.0
    # When the body was first observed at the place it is now.  See
    # :meth:`Detector.feed`: the stationary time is the ELAPSED time since that
    # observation, not the time the duck spent watching.
    settled_since_s: float | None = None
    _previous: tuple[float, float] | None = None

    @property
    def confirmed(self) -> bool:
        """Has it been visible long enough to be acted on?"""
        return self.seen_s >= DETECT_CONFIRM_S


@dataclass
class Verdict:
    """One classification: what it is, why, and how far past the rule it sits."""

    name: str
    verdict: str                 # "suspicious" | "intrusion" | "benign"
    rule: str
    confidence: float
    margin_m: float = 0.0
    margin_s: float = 0.0
    evidence: dict = field(default_factory=dict)

    @property
    def investigate(self) -> bool:
        """Does this verdict warrant breaking off the patrol?

        Only the two escalating verdicts do.  A benign call is a decision NOT to
        investigate, and recording it as a verdict rather than as silence is
        what makes the dismissal checkable.
        """
        return self.verdict in ("suspicious", "intrusion")

    def as_record(self) -> dict:
        return {
            "target": self.name,
            "verdict": self.verdict,
            "rule": self.rule,
            "confidence": round(float(self.confidence), 4),
            "margin_m": round(float(self.margin_m), 4),
            "margin_s": round(float(self.margin_s), 3),
            "evidence": self.evidence,
        }


def _confidence(margin: float, scale: float) -> float:
    """A bounded RULE-MARGIN proxy in ``[0.5, 0.99]``.

    Not a probability and never presented as one: it maps how far the evidence
    sits past a rule's own threshold onto a readable number, saturating so a
    wildly-clear case does not report certainty.  0.5 at the threshold itself,
    because a decision taken exactly at its bar is a coin flip and should look
    like one.
    """
    return float(min(0.99, 0.5 + 0.49 * math.tanh(max(margin, 0.0) / scale)))


class Detector:
    """The duck's own record of what it has seen, and what it concluded.

    One instance per rollout.  ``feed`` is called every tick with the camera's
    visibility dictionary and the measured world positions; it updates the
    observations and returns the candidates that are ready to be classified.
    """

    def __init__(self, dt: float):
        self.dt = float(dt)
        self.observations: dict[str, Observation] = {}
        self.verdicts: dict[str, Verdict] = {}
        # Every classification in the order it was made, which is what the gate
        # compares against the scenario's expected verdicts.
        self.sequence: list[Verdict] = []
        # Bodies already handled, so a body is not re-detected forever.
        self.settled: set[str] = set()
        # Per body, the number of ticks it was inside the camera gate.  Reported
        # so "it was detected through the camera" is a count, not a claim.
        self.gate_ticks: dict[str, int] = {}
        self.first_gate_s: dict[str, float] = {}

    # -- accumulate -------------------------------------------------------
    def feed(self, t: float, *, visibility: dict, positions: dict,
             duck_xy) -> list[str]:
        """Update every observation from THIS tick's camera and measurements.

        Returns the names of bodies that are confirmed, unsettled and therefore
        ready to be classified.  A body that is not visible this tick is simply
        not updated: its accumulated evidence stays, but it cannot gain any.
        """
        duck = np.asarray(duck_xy, dtype=np.float64)[:2]
        people = {name: np.asarray(positions[name], dtype=np.float64)[:2]
                  for name in PERSON_NAMES if name in positions}

        ready: list[str] = []
        for name, entry in visibility.items():
            if name not in positions:
                continue
            if not entry.get("visible"):
                continue
            position = np.asarray(positions[name], dtype=np.float64)[:2]
            range_m = float(np.linalg.norm(position - duck))
            # THE CAMERA GATE.  Visible, and near enough that the camera could
            # resolve what it is.
            if range_m > DETECT_MAX_RANGE_M:
                continue

            self.gate_ticks[name] = self.gate_ticks.get(name, 0) + 1
            self.first_gate_s.setdefault(name, float(t))

            observation = self.observations.get(name)
            if observation is None:
                observation = Observation(
                    name=name, kind=BY_NAME[name].kind, first_seen_s=float(t),
                    last_seen_s=float(t))
                self.observations[name] = observation

            observation.last_seen_s = float(t)
            observation.seen_s += self.dt
            observation.position = (float(position[0]), float(position[1]))
            observation.range_m = range_m

            # HOW LONG IT HAS STOOD THERE, from the duck's OWN successive
            # observations.
            #
            # THIS IS ELAPSED TIME SINCE IT WAS FIRST SEEN IN THIS PLACE, NOT
            # TIME SPENT WATCHING IT, AND THE DISTINCTION IS A MEASURED FIX.
            # Accumulating only the ticks the object was in frustum makes the
            # rule depend on how long the duck happened to look: MEASURED, the
            # crate was in the camera gate for 5.58 s of a single checkpoint
            # scan and missed a 6.0 s bar it had physically satisfied for far
            # longer.  A robot that glanced twice, ten seconds apart, and saw
            # the same box in the same place both times has evidence it stood
            # there for ten seconds - that is what a person concludes, and it is
            # what makes the rule a statement about the OBJECT rather than about
            # the patrol's timing.
            #
            # The assumption it carries is stated as a limitation in the README:
            # an object could in principle leave and return between two
            # observations.  The conservative direction is available - counting
            # only observed time - and was rejected because it makes the verdict
            # depend on the observer rather than the observed.
            moved = (float(np.linalg.norm(
                position - np.asarray(observation._previous)))
                if observation._previous is not None else 0.0)
            if observation.settled_since_s is None or moved > STATIONARY_STEP_M:
                observation.settled_since_s = float(t)
            observation.stationary_s = float(t) - observation.settled_since_s
            observation._previous = (float(position[0]), float(position[1]))

            # Who is with it, and where it is standing.
            nearest, nearest_m = "", float("inf")
            for person, where in people.items():
                if person == name:
                    continue
                gap = float(np.linalg.norm(where - position))
                if gap < nearest_m:
                    nearest, nearest_m = person, gap
            observation.nearest_person = nearest
            observation.nearest_person_m = nearest_m
            observation.stow_area = stowed_on(position)

            # Zone membership and dwell.
            depth = RESTRICTED_ZONE.depth_inside(position)
            observation.zone_depth_m = depth
            if BY_NAME[name].is_person and depth >= 0.0:
                observation.in_zone_s += self.dt
            elif BY_NAME[name].is_person:
                observation.in_zone_s = 0.0

            if observation.confirmed and name not in self.settled:
                ready.append(name)
        return ready

    # -- classify ----------------------------------------------------------
    def classify(self, name: str) -> Verdict | None:
        """Decide what one confirmed body is, from its accumulated observation.

        Returns ``None`` when the evidence does not yet support ANY verdict -
        an object that has been visible for half a second but has not yet been
        still for six is not innocent and not suspicious; it is unresolved, and
        saying so is more honest than defaulting either way.

        THE RULES ARE ORDERED CHEAPEST-INNOCENT-FIRST.  A guard robot must rule
        out the ordinary explanations before it escalates, so the benign rules
        are checked before the suspicious one and the residual category is the
        one that costs an investigation.
        """
        observation = self.observations.get(name)
        if observation is None or not observation.confirmed:
            return None
        spec = BY_NAME[name]

        if spec.is_person:
            return self._classify_person(observation)
        return self._classify_object(observation)

    def _classify_person(self, o: Observation) -> Verdict | None:
        """A person is only ever an intrusion, or nothing at all.

        Staff walking the floor are not classified as anything: a patrol robot
        that produced a verdict about every person it saw would be reporting its
        own colleagues.  The ONLY thing that makes a person a case is being
        inside the marked rectangle for longer than crossing it takes.
        """
        if o.in_zone_s < INTRUSION_DWELL_S:
            return None
        return Verdict(
            name=o.name,
            verdict="intrusion",
            rule=(f"person inside the marked restricted zone for "
                  f"{o.in_zone_s:.1f}s, past the {INTRUSION_DWELL_S:.1f}s "
                  f"dwell that separates entering from crossing"),
            confidence=_confidence(o.in_zone_s - INTRUSION_DWELL_S, 3.0),
            margin_m=float(o.zone_depth_m),
            margin_s=float(o.in_zone_s - INTRUSION_DWELL_S),
            evidence={
                "zone": RESTRICTED_ZONE.name,
                "position": [round(o.position[0], 4), round(o.position[1], 4)],
                "depth_inside_m": round(float(o.zone_depth_m), 4),
                "dwell_s": round(float(o.in_zone_s), 3),
                "dwell_bar_s": INTRUSION_DWELL_S,
                "range_m": round(float(o.range_m), 4),
                "seen_s": round(float(o.seen_s), 3),
            })

    def _classify_object(self, o: Observation) -> Verdict | None:
        """An object is benign if something explains it, suspicious otherwise."""
        attended = o.nearest_person_m <= ATTENDED_RADIUS_M
        if o.stow_area or attended:
            reasons = []
            if o.stow_area:
                reasons.append(f"standing in the designated stow area "
                               f"{o.stow_area}")
            if attended:
                reasons.append(f"{o.nearest_person} is "
                               f"{o.nearest_person_m:.2f} m away, inside the "
                               f"{ATTENDED_RADIUS_M:.2f} m attendance radius")
            margin = ATTENDED_RADIUS_M - o.nearest_person_m if attended else 0.0
            return Verdict(
                name=o.name,
                verdict="benign",
                rule="; ".join(reasons),
                confidence=_confidence(max(margin, 0.20), 0.45),
                margin_m=float(max(margin, 0.0)),
                evidence={
                    "stow_area": o.stow_area,
                    "attended_by": o.nearest_person if attended else "",
                    "nearest_person_m": (
                        None if not np.isfinite(o.nearest_person_m)
                        else round(float(o.nearest_person_m), 4)),
                    "attendance_radius_m": ATTENDED_RADIUS_M,
                    "position": [round(o.position[0], 4),
                                 round(o.position[1], 4)],
                    "range_m": round(float(o.range_m), 4),
                    "seen_s": round(float(o.seen_s), 3),
                })

        if o.stationary_s < UNATTENDED_S:
            return None
        return Verdict(
            name=o.name,
            verdict="suspicious",
            rule=(f"stationary {o.stationary_s:.1f}s outside any designated "
                  f"stow area with nobody within "
                  f"{ATTENDED_RADIUS_M:.2f} m"),
            confidence=_confidence(o.stationary_s - UNATTENDED_S, 6.0),
            margin_s=float(o.stationary_s - UNATTENDED_S),
            margin_m=float(min(o.nearest_person_m - ATTENDED_RADIUS_M, 9.0)
                           if np.isfinite(o.nearest_person_m) else 9.0),
            evidence={
                "stationary_s": round(float(o.stationary_s), 3),
                "unattended_bar_s": UNATTENDED_S,
                "stow_area": "",
                "nearest_person": o.nearest_person,
                "nearest_person_m": (None if not np.isfinite(o.nearest_person_m)
                                     else round(float(o.nearest_person_m), 4)),
                "attendance_radius_m": ATTENDED_RADIUS_M,
                "position": [round(o.position[0], 4), round(o.position[1], 4)],
                "range_m": round(float(o.range_m), 4),
                "seen_s": round(float(o.seen_s), 3),
            })

    def record(self, verdict: Verdict) -> None:
        """Commit a verdict and stop reconsidering that body."""
        self.verdicts[verdict.name] = verdict
        self.sequence.append(verdict)
        self.settled.add(verdict.name)

    # -- reporting -----------------------------------------------------------
    def summary(self) -> dict:
        return {
            "verdicts": [v.as_record() for v in self.sequence],
            "verdict_order": [v.name for v in self.sequence],
            "verdict_by_name": {v.name: v.verdict for v in self.sequence},
            "investigated": [v.name for v in self.sequence if v.investigate],
            "dismissed": [v.name for v in self.sequence if not v.investigate],
            "camera_gate_ticks": dict(self.gate_ticks),
            "first_in_camera_gate_s": {
                k: round(v, 3) for k, v in self.first_gate_s.items()},
            "observations": {
                name: {
                    "first_seen_s": round(o.first_seen_s, 3),
                    "seen_s": round(o.seen_s, 3),
                    "stationary_s": round(o.stationary_s, 3),
                    "settled_since_s": (None if o.settled_since_s is None
                                        else round(o.settled_since_s, 3)),
                    "in_zone_s": round(o.in_zone_s, 3),
                    "nearest_person": o.nearest_person,
                    "nearest_person_m": (
                        None if not np.isfinite(o.nearest_person_m)
                        else round(float(o.nearest_person_m), 4)),
                    "stow_area": o.stow_area,
                } for name, o in sorted(self.observations.items())},
        }
