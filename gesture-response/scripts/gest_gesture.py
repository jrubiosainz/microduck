#!/usr/bin/env python3
"""The gesture classifier: measured arm features -> a command, or nothing.

WHAT THIS IS, WITHOUT DRESSING IT UP
--------------------------------------
An **honest semantic simulator proxy**.  It is a rule set over features
:mod:`gest_pose` measured from real world geometry - extension, elevation,
forward and lateral components in the person's own frame, and how far the hand
moved over the last window.  It is NOT an RGB gesture recogniser, and the
number it reports is a **rule-margin proxy**: how far the measured evidence sits
past each window's own edge, normalised to 0..1.  It is not a probability, and
it is labelled as a proxy in the HUD, in the metrics and in the README.

WHAT MAKES IT HONEST RATHER THAN A LOOKUP
-------------------------------------------
Three properties, each of which a scripted lookup would fail:

* **It reads only measured geometry.**  It never receives the gesture's name,
  the schedule, or the person's role.  ``tests/test_rollout_and_hygiene.py``
  parses the import graph and fails if this module ever imports
  ``gest_actors`` or ``gest_script``.
* **It can refuse.**  A pose that matches nothing well enough returns ``None``,
  and the scenario contains a deliberate PARTIAL pose that must land there.
  A lookup cannot fail to recognise the thing it looked up.
* **Its windows overlap in every dimension but one.**  COME and STOP share a
  raised forward arm; STOP and BACK_UP share a straight forward push; WAVE and
  STOP share a high hand.  Each pair is separated by ONE measured feature -
  hand motion, arm count, elevation - so the classifier has to measure rather
  than pattern-match on a whole silhouette.

THE COMMANDS, WHICH ARE NOT THE TEMPLATE NAMES
------------------------------------------------
A gesture is read in the PERSON's frame; the command is what the DUCK must do.
The instructor faces the duck, so her raised LEFT arm points to the duck's
right.  That mapping happens once, in :func:`command_for`, and it is the only
place the two frames meet - which is why the two turn commands are opposite
physical heading changes rather than opposite labels.
"""

from __future__ import annotations

from dataclasses import dataclass

# How much of its own arm span the hand must TRAVEL over the motion window for
# a gesture to count as OSCILLATING.  Travel is accumulated PATH, not the
# distance between the window's endpoints: a beckon at 1.15 Hz sampled over a
# short window can return to almost exactly where it started, and an endpoint
# measure reported 0.07 for a hand that had swept 0.40 of its span.
#
# THE WINDOW MUST BE LONGER THAN THE SLOWEST OSCILLATING GESTURE'S OWN CYCLE,
# AND THAT IS A MEASURED REQUIREMENT RATHER THAN A PREFERENCE.  The animation
# beckons at 1.15 Hz (period 0.870 s), waves at 1.05 Hz (0.952 s) and pushes at
# 0.95 Hz (1.053 s).  A window shorter than a period can only ever contain a
# half-swing - a one-way hand movement whose ``wander`` collapses to about 1.0,
# which is indistinguishable from an arm on its way up.
#
# MEASURED consequence of getting this wrong, with the original 0.60 s window:
# COME's own confidence fell under the bar twice per second, the confirm window
# reset every ~0.7 s, and the duck read COME thirteen times in twelve seconds
# without ever confirming it.
#
# MEASURED steady-state worst-case wander against window length, which is what
# sets the value rather than the periods alone:
#
#     window   COME    WAVE
#     0.60 s    1.46    1.21   <- barely above a raise; no margin at all
#     0.70 s    2.51    1.73
#     0.80 s    7.06    3.18
#     1.10 s   16.5    28.4    <- both an order of magnitude clear
#
# 1.10 s exceeds every oscillation period above, so each window contains a whole
# cycle of the slowest of them, and it is the value the wander bar below is
# measured against.  BACK_UP's push period is the longest at 1.053 s, and its
# template does not test wander at all - it is separated by arm COUNT - but the
# window still covers it, so no oscillating gesture is graded on a partial
# cycle.
MOTION_WINDOW_S = 1.10
MOVING_BAR = 0.30
# And the bar a STILL template must stay BELOW.  Far above the measured 0.000 of
# a static pose and far below the least-moving oscillation, so neither side is a
# boundary case.
STILL_BAR = 0.12

# PATH ALONE CANNOT TELL AN OSCILLATION FROM AN ARM ON ITS WAY UP, AND THAT IS A
# MEASURED FAILURE RATHER THAN A HYPOTHETICAL ONE.  The ambiguous PARTIAL pose
# is a half-lifted arm held perfectly still, which must be refused - but at the
# instant it finished RISING, the 0.60 s window behind it was full of the raise,
# reporting a path of 1.94 spans.  With path as the only motion feature the
# classifier accepted that single tick as a COME.
#
# The honest separator is that a RAISE IS ONE-WAY and an OSCILLATION COMES BACK.
# ``wander`` is the path divided by the net displacement over the same window:
# a monotonic raise measures about 1.0, while a hand that swung out and back
# measures far more.
#
# THE BAR IS SET FROM THE STEADY-STATE MEASUREMENT, NOT FROM THE TRANSIENT.
# MEASURED over a 0.90 s window fully inside each hold: COME 19.2 at worst,
# WAVE 10.6, BACK_UP 3.6, and every static pose exactly 0.00.  MEASURED over a
# window that still straddles the raise, a rising arm reaches at most 1.40 -
# including the ambiguous PARTIAL's own raise, which is the pose this bar exists
# to refuse.  1.60 sits above every raise measured and an order of magnitude
# below the least-oscillating gesture, so neither side is a boundary case.
OSCILLATION_WANDER_BAR = 1.60

# The confidence a rule must reach to be offered as a candidate at all.  A pose
# that scores below this is reported as an unrecognised pose, which is what the
# ambiguous PARTIAL gesture must produce.
MIN_CONFIDENCE = 0.55



from gest_commands import COMMANDS  # noqa: F401
from gest_templates import (  # noqa: F401
    BY_TEMPLATE,
    TEMPLATES,
    Template,
    Window,
)


@dataclass(frozen=True)
class Reading:
    """One classification attempt on one measured pose."""

    template: str
    command: str
    confidence: float
    rule: str
    features: dict

    @property
    def accepted(self) -> bool:
        return bool(self.template) and self.confidence >= MIN_CONFIDENCE

    def as_record(self) -> dict:
        return {
            "template": self.template,
            "command": self.command,
            "confidence": round(float(self.confidence), 4),
            "rule": self.rule,
            "features": {k: round(float(v), 4) if isinstance(v, float) else v
                         for k, v in self.features.items()},
        }


def _motion_ok(template: Template, hand_travel: float,
               hand_wander: float) -> bool:
    """Does this pose's hand motion satisfy the template's motion rule?

    A ``"moving"`` template needs BOTH a real path and evidence that the hand
    came back - see :data:`OSCILLATION_WANDER_BAR`.  A ``"still"`` template
    needs the path to be small; no wander test applies, because a still hand has
    no meaningful direction to have returned from.
    """
    if template.motion == "moving":
        return (hand_travel >= MOVING_BAR
                and hand_wander >= OSCILLATION_WANDER_BAR)
    if template.motion == "still":
        return hand_travel <= STILL_BAR
    return True


def score(template: Template, pose, hand_travel: float,
          hand_wander: float) -> float:
    """How well one measured pose matches one template.  0 means no match.

    The score is the MINIMUM of the per-feature margins, not their mean.  A mean
    lets a pose that is badly wrong in one dimension be rescued by being
    perfect in three others, which is exactly how a half-raised arm would talk
    its way into being a STOP.  The minimum means every feature has to be inside
    its own window on its own merits.
    """
    if pose.raised_count != template.arms:
        return 0.0
    if not _motion_ok(template, hand_travel, hand_wander):
        return 0.0
    arms = pose.raised_arms
    if not arms:
        return 0.0
    # For a two-armed template EVERY raised arm must satisfy the windows, and
    # the score is the worst of them.  Measuring only the "primary" arm would
    # let one arm be anywhere at all.
    worst = 1.0
    for arm in arms:
        margins = (
            template.extension.score(arm.extension),
            template.elevation.score(arm.elevation_deg),
            template.forward.score(arm.forward),
            template.lateral.score(_lateral_for(arm, template)),
        )
        worst = min(worst, min(margins))
        if worst <= 0.0:
            return 0.0
    return float(worst)


def _lateral_for(arm, template: Template) -> float:
    """The lateral value this template's window is applied to.

    A two-armed push has one arm slightly to the person's left and one slightly
    to their right, and both are legitimate, so the two-armed templates carry a
    window symmetric about zero and are handed the SIGNED value directly - which
    a symmetric window accepts either way round.

    A one-armed POINT is specifically to one side, and its window is one-sided,
    so the signed value is what must be tested.  Both cases therefore pass the
    signed value; the distinction lives in the window, which is where a rule
    about sides belongs.  An earlier version took ``abs()`` and re-applied a
    sign here, which made the LEFT and RIGHT windows interchangeable and would
    have let the two pointing commands collapse into each other.
    """
    return float(arm.lateral)


def classify(pose, hand_travel: float, hand_wander: float = 0.0) -> Reading:
    """The best-matching template for a measured pose, or an empty Reading.

    Returning the BEST rather than the first match matters: several windows
    genuinely overlap, and a first-match rule would make the answer depend on
    the order of a tuple rather than on the evidence.
    """
    features = {
        "raised_arms": pose.raised_count,
        "hand_travel": float(hand_travel),
        "hand_wander": float(hand_wander),
        "fully_readable": bool(pose.fully_readable),
    }
    primary = pose.primary
    if primary is not None:
        features.update({
            "extension": float(primary.extension),
            "elevation_deg": float(primary.elevation_deg),
            "forward": float(primary.forward),
            "lateral": float(primary.lateral),
        })

    best: Template | None = None
    best_score = 0.0
    for template in TEMPLATES:
        value = score(template, pose, hand_travel, hand_wander)
        if value > best_score:
            best, best_score = template, value

    if best is None or best_score < MIN_CONFIDENCE:
        return Reading(
            "", "", float(best_score),
            (f"no template's measured margin reached {MIN_CONFIDENCE:.2f}"
             f" (best {best.name if best else 'none'} at {best_score:.2f})"),
            features)
    return Reading(best.name, best.command, float(best_score),
                   f"{best.label}; margin {best_score:.2f} past every window "
                   "edge (rule-margin proxy, not a probability)", features)


def command_for(template_name: str) -> str:
    """The command a template maps to, or ``""``."""
    template = BY_TEMPLATE.get(template_name)
    return template.command if template is not None else ""
