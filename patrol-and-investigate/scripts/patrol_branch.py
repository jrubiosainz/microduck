#!/usr/bin/env python3
"""The investigation branch: DETECT -> INVESTIGATE_PLAN -> APPROACH -> OBSERVE
-> CLASSIFY -> RETURN_TO_PATROL -> RESUME.

Split out of ``patrol_machine`` so each module stays inside the size budget and,
more usefully, so the two halves of the behavior can be read separately: the
machine file is about walking a circuit and stopping on it, and this file is
about what happens when the circuit is interrupted.  Both are transition logic
only - no physics, no MuJoCo, no commands - and :class:`InvestigationBranch` is
mixed into :class:`patrol_machine.PatrolMachine`.

THE FOUR RULES THIS BRANCH EXISTS TO ENFORCE
----------------------------------------------
* **A benign candidate costs no walking.**  :meth:`_detect_state` sends a
  dismissed candidate straight to CLASSIFY, so the duck never takes a step
  toward something it has already explained.  The dismissal is recorded exactly
  as an escalation is, which is what makes "it explicitly dismissed the
  distractor" checkable rather than an absence of evidence.

* **The approach stops on MEASURED range.**  :meth:`_approach_state` ends on the
  range to the target being inside the standoff band - not on arrival at the
  planned point.  The two normally agree; conflating them would let a
  badly-placed standoff carry the duck closer than the band allows.

* **The observation is a count of held bearings.**  Each angle is opened, held
  for its own dwell, and closed with its own visibility fraction, so
  "multi-angle" is arithmetic over recorded holds.

* **Resuming means going back, not skipping ahead.**  RETURN_TO_PATROL ends on
  the duck being within :data:`RESUME_TOLERANCE_M` of the remembered
  interruption point, and only then does RESUME hand control back to the patrol
  - still aimed at the checkpoint it was aimed at before.
"""

from __future__ import annotations

from patrol_episode import Observation
from patrol_states import (
    APPROACH_MAX_S,
    CLASSIFY_MAX_S,
    CLASSIFY_S,
    DETECT_MAX_S,
    INVESTIGATE_PLAN_MAX_S,
    OBSERVE_ANGLES_DEG,
    OBSERVE_HOLD_S,
    OBSERVE_MAX_S,
    RESUME_MAX_S,
    RETURN_MAX_S,
)

# How long the duck holds INVESTIGATE_PLAN.  A real beat rather than an
# instantaneous transition, because the plan is a decision the video has to
# show: the duck stops, the candidate standoff points appear on the floor, the
# rejected ones are marked, and one is chosen.  Long enough to read.
INVESTIGATE_PLAN_S = 1.6
# How long RESUME is held once the duck is back on its route, so the resumption
# is a visible beat in the timeline rather than a single-tick transition.
RESUME_S = 1.0
# How near the remembered resume point counts as being back on the route.
# DERIVED from the duck's own conservative planar radius (0.1162 m): being
# within its own footprint of the point it left is as close as "back where it
# was" can meaningfully mean for a body of this size.
RESUME_TOLERANCE_M = 0.16
# How long DETECT is held before the duck commits either way, so the detection
# is visible in the video and the verdict is not taken on a single tick.
DETECT_HOLD_S = 0.6


class InvestigationBranch:
    """The DETECT..RESUME handlers.  Mixed into :class:`PatrolMachine`."""

    # -- deciding whether to go and look ---------------------------------
    def _detect_state(self, t: float, s) -> None:
        """Something is confirmed in the camera.  Is it worth leaving the route?

        BOTH ANSWERS ARE RECORDED.  A candidate the detector explains away goes
        straight to CLASSIFY without a step being taken toward it, and the
        dismissal appears in the log exactly as an escalation does.  A behavior
        that simply stayed silent about what it ignored could not be shown to
        have considered it.
        """
        if not self._subject and s.candidate:
            self._subject = s.candidate
        if self._elapsed(t) < DETECT_HOLD_S:
            return
        if not s.candidate_investigate:
            self._dismissed = True
            self._go(t, "CLASSIFY", target=self._subject,
                     verdict=s.candidate_verdict,
                     reason=f"{self._subject} explained by a benign rule; "
                            "dismissed without leaving the route")
            return
        self._dismissed = False
        self._go(t, "INVESTIGATE_PLAN", target=self._subject,
                 verdict=s.candidate_verdict,
                 reason=f"{self._subject} warrants a closer look; planning a "
                        "safe standoff")

    def _investigate_plan_state(self, t: float, s) -> None:
        """Choose where to stand, and break off the patrol - remembering where.

        The interruption itself is recorded by the ROLLOUT, which owns the plan
        object; this state holds long enough for the decision to be visible and
        then commits to the approach.
        """
        if self._elapsed(t) >= INVESTIGATE_PLAN_S and s.standoff_ready:
            self._go(t, "APPROACH", target=self._subject,
                     reason="standoff chosen; approaching to a safe "
                            "observation distance")
            return
        if self._timeout(t, INVESTIGATE_PLAN_MAX_S, "INVESTIGATE_PLAN"):
            self._go(t, "APPROACH" if s.standoff_ready else "CLASSIFY",
                     reason="investigate-plan ceiling reached")

    # -- closing on it ------------------------------------------------------
    def _approach_state(self, t: float, s) -> None:
        """Close on the target and STOP inside the safe standoff band.

        THE EXIT IS THE MEASURED RANGE, NOT ARRIVAL AT THE PLANNED POINT.  The
        two normally agree, but conflating them would mean a standoff point
        placed badly - by a planner bug, or by the target having moved since -
        could carry the duck closer than the band allows.  Ending on the
        measured range means the safety property is enforced by the quantity it
        is about.
        """
        if s.in_standoff_band:
            self._go(t, "OBSERVE", target=self._subject,
                     range_m=round(s.target_range_m, 4),
                     reason=f"stopped {s.target_range_m:.3f} m from "
                            f"{self._subject}; observing from a standoff")
            return
        if self._timeout(t, APPROACH_MAX_S, "APPROACH"):
            self._go(t, "OBSERVE", reason="approach ceiling reached")

    # -- looking at it from several angles -----------------------------------
    def _observe_state(self, t: float, s) -> None:
        """Hold several viewing angles on the target, at an exact zero command.

        The duck cannot orbit - turning in place is MEASURED to be unavailable
        at 1.6 deg/s - so the angles are swept by the HEAD from a fixed
        standoff.  Each angle is held for its own dwell and recorded separately,
        which is what makes "multi-angle" a count of held bearings rather than a
        description of the motion.
        """
        if self._observation is None:
            self._begin_angle(t)

        observation = self._observation
        if observation is not None:
            observation.held_s = t - observation.began_at_s
            observation.steps += 1
            if s.candidate_visible:
                observation.visible_steps += 1
            observation.min_range_m = min(observation.min_range_m,
                                          s.target_range_m)
            if observation.held_s >= OBSERVE_HOLD_S:
                self._end_angle()
                if self._angle_index >= len(OBSERVE_ANGLES_DEG):
                    self._go(t, "CLASSIFY", target=self._subject,
                             angles=len(OBSERVE_ANGLES_DEG),
                             reason=f"{len(OBSERVE_ANGLES_DEG)} viewing angles "
                                    "held; classifying")
                    return
                self._begin_angle(t)
        if self._timeout(t, OBSERVE_MAX_S, "OBSERVE"):
            self._end_angle()
            self._go(t, "CLASSIFY", reason="observe ceiling reached")

    def _begin_angle(self, t: float) -> None:
        if self._angle_index >= len(OBSERVE_ANGLES_DEG):
            return
        self._observation = Observation(
            angle_deg=OBSERVE_ANGLES_DEG[self._angle_index], began_at_s=t)

    def _end_angle(self) -> None:
        if self._observation is None:
            return
        if self._investigation is not None:
            self._investigation.observations.append(self._observation)
        self._observation = None
        self._angle_index += 1

    @property
    def observe_angle_deg(self) -> float:
        """The bearing the head is currently holding, for the camera to aim."""
        if self._observation is not None:
            return self._observation.angle_deg
        if self._angle_index < len(OBSERVE_ANGLES_DEG):
            return OBSERVE_ANGLES_DEG[self._angle_index]
        return 0.0

    # -- recording it and getting back ---------------------------------------
    def _classify_state(self, t: float, s) -> None:
        """Record the verdict, then go back to the patrol.

        A DISMISSED candidate returns straight to PATROL, because the duck never
        left its route for it.  An INVESTIGATED one has to walk back to the
        point it broke off at, which is RETURN_TO_PATROL.
        """
        if self._elapsed(t) < CLASSIFY_S:
            if not self._timeout(t, CLASSIFY_MAX_S, "CLASSIFY"):
                return
        if self._dismissed:
            self._close_visit(t, "detect", self._subject)
            subject, self._subject = self._subject, ""
            self._dismissed = False
            self._go(t, "PATROL", target=subject,
                     reason=f"{subject} dismissed; the patrol was never "
                            "interrupted")
            return
        self._go(t, "RETURN_TO_PATROL", target=self._subject,
                 reason="investigation recorded; returning to the point the "
                        "patrol was interrupted")

    def _return_to_patrol_state(self, t: float, s) -> None:
        """Walk back to the exact point the patrol was broken off at."""
        if s.at_resume_point:
            self._go(t, "RESUME", error_m=round(s.resume_remaining_m, 4),
                     reason=f"back within {s.resume_remaining_m:.3f} m of the "
                            "interruption point")
            return
        if self._timeout(t, RETURN_MAX_S, "RETURN_TO_PATROL"):
            self._go(t, "RESUME", reason="return ceiling reached")

    def _resume_state(self, t: float, s) -> None:
        """Pick the patrol up, still aimed at the SAME checkpoint as before."""
        if self._elapsed(t) >= RESUME_S:
            self._close_visit(t, "detect", self._subject)
            subject, self._subject = self._subject, ""
            self._go(t, "PATROL", target=subject,
                     next_checkpoint=s.target_name,
                     reason=f"patrol resumed toward {s.target_name}")
            return
        if self._timeout(t, RESUME_MAX_S, "RESUME"):
            self._close_visit(t, "detect", self._subject)
            self._subject = ""
            self._go(t, "PATROL", reason="resume ceiling reached")
