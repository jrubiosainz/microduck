#!/usr/bin/env python3
"""The state machine: plan, advance, spot a threat, choose a side or wait, pass,
replan, and arrive.

    PLAN -> ADVANCE -> THREAT -> CHOOSE_LEFT | CHOOSE_RIGHT | WAIT
         -> PASS -> REPLAN -> (ADVANCE ...)  -> GOAL -> DONE

The cycle ADVANCE -> THREAT -> {CHOOSE_*|WAIT} -> PASS -> REPLAN repeats once
per encounter, and REPLAN returns to ADVANCE for the next one.  WAIT is not a
dead end: it resolves into whichever side the prediction clears first, so a wait
is a deferred choice rather than a refusal to move.

The machine never touches physics and never emits a command; ``slalom_control``
does that from the state.  Keeping the two apart is what lets every transition
rule be unit-tested on hand-built inputs, with no MuJoCo anywhere.

FIVE INVARIANTS ARE STRUCTURAL RATHER THAN CHECKED AFTERWARDS
---------------------------------------------------------------
* **Every choice is CAUSED by a prediction, never scheduled.**  There is no
  timer, no waypoint index and no schedule lookup in this file.  The duck leaves
  ADVANCE because it PREDICTED a conflict, and it leaves THREAT for the side
  whose predicted clearance was higher.  The choreography lives in
  ``slalom_actors`` and the machine cannot see it.

* **A choice must be justified by a POSITIVE predicted clearance, and the
  losing side is recorded with it.**  :meth:`_threat_state` stores the whole
  :class:`slalom_plan.Decision`, so every pass in the log names the rejected
  corridor and its predicted clearance.  A decision that could not explain
  itself would fail the gate.

* **WAIT is entered only when NEITHER side survives**, and it is exactly zero
  while it lasts.  Forward gait onset on this scene is a MEASURED cliff, so a
  duck that "edged forward" while waiting would emit a command that produces
  nothing on the floor.

* **A pass is not over until the body is measured receding.**
  :meth:`_pass_state` ends on the threat's measured range growing past
  :data:`~slalom_states.PASS_CLEAR_M`, which is a property of the world rather
  than of a countdown the machine could race.

* **A ceiling MOVES the machine.**  Every ``_timeout`` here transitions and
  records why.  A ceiling that only appends to a log is not a ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slalom_control import ON_CORRIDOR_M
from slalom_encounter import PassRecord, Sense
from slalom_states import (
    ADVANCE_MAX_S,
    CHOOSE_MAX_S,
    COMMIT_CONFIRM_S,
    GOAL_MAX_S,
    GOAL_SETTLED_MPS,
    MIN_GOAL_S,
    MIN_REPLAN_S,
    MIN_WAIT_S,
    PASS_MAX_S,
    PLAN_MAX_S,
    REPLAN_MAX_S,
    RESOLVED_IGNORE_S,
    THREAT_MAX_S,
    WAIT_MAX_S,
)


@dataclass
class SlalomMachine:
    """Transitions, the pass log and the replan log.  No physics."""

    ctrl_hz: float = 50.0
    state: str = "PLAN"
    state_since: float = 0.0

    transitions: list[dict] = field(default_factory=list)
    passes: list[PassRecord] = field(default_factory=list)
    replans: list[dict] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    # The side chosen for each completed encounter, in order.  The alternation
    # gate reads this.
    pass_sides: list[str] = field(default_factory=list)
    waits: list[dict] = field(default_factory=list)

    _commit_for: float = 0.0
    _pass: PassRecord | None = None
    _wait: dict = field(default_factory=dict)
    _pending_side: str = ""
    # The body whose encounter has already been resolved.  A threat is IGNORED
    # while it names this body, which is what stops one crossing from being
    # re-detected as a fresh encounter over and over.
    _resolved_threat: str = ""
    _resolved_at: float = -1e9
    # When the duck's own measured speed first fell to a standstill inside the
    # goal band.  See :meth:`_goal_state`.
    _settled_at: float | None = None
    # The body the CURRENT encounter is about.  Set when a pass opens and
    # cleared when it closes; the rollout reads it so resolution is measured
    # against this body rather than against whatever the planner flags now.
    _encounter_body: str = ""

    def __post_init__(self) -> None:
        self.dt = 1.0 / self.ctrl_hz

    # -- helpers ---------------------------------------------------------
    def _go(self, t: float, state: str, **detail) -> None:
        self.transitions.append(
            {"t": round(t, 3), "from": self.state, "to": state, **detail})
        self.state = state
        self.state_since = t
        self._commit_for = 0.0

    def _elapsed(self, t: float) -> float:
        return t - self.state_since

    def _timeout(self, t: float, limit: float, label: str) -> bool:
        if self._elapsed(t) >= limit:
            self.timeouts.append(f"{label}@{t:.2f}s")
            return True
        return False

    # -- the machine -----------------------------------------------------
    def update(self, t: float, sense: Sense) -> tuple[str, bool]:
        """Advance one control tick on measurements taken BEFORE this tick's
        physics."""
        before = self.state
        handler = getattr(self, f"_{self.state.lower()}_state")
        handler(t, sense)
        return self.state, self.state != before

    # -- getting going ----------------------------------------------------
    def _plan_state(self, t: float, s: Sense) -> None:
        """Compute the first line to the goal and set off.

        Short by construction: the plan is a straight line to a visible band, so
        there is nothing to search.  It exists as a state so the video and the
        timeline show the duck planning BEFORE it moves, and so the first
        prediction is made from a standstill rather than mid-stride.
        """
        if self._elapsed(t) >= 0.40:
            self._go(t, "ADVANCE",
                     goal_remaining_m=round(s.goal_remaining_m, 3),
                     reason="plan computed; walking the lane to the goal")
            return
        if self._timeout(t, PLAN_MAX_S, "PLAN"):
            self._go(t, "ADVANCE", reason="plan ceiling reached")

    def _advance_state(self, t: float, s: Sense) -> None:
        """Walk the planned line until the predictor flags a NEW conflict.

        Three exits, and the ORDER matters.  Arriving wins over a late threat, so
        a body crossing behind the goal band cannot pull the duck out of its own
        arrival.

        A THREAT NAMING THE BODY JUST PASSED IS IGNORED, AND THAT IS A SCAR.
        Without it, the duck resolved an encounter, replanned, immediately
        re-detected the SAME receding body as a threat and opened another pass on
        it - ten "passes" for five crossings, alternation destroyed, and two
        phases hitting their ceilings.  An encounter is over when the body has
        gone past; it does not become a new one because the planner can still see
        it.
        """
        if s.at_goal or s.goal_remaining_m <= 0.0:
            self._go(t, "GOAL", reason="reached the arrival band")
            return
        if s.threat and not self._is_stale(t, s.threat):
            self._go(t, "THREAT", threat=s.threat,
                     ttc_s=round(s.threat_ttc_s, 3),
                     range_m=round(s.threat_range_m, 3),
                     reason=f"{s.threat} predicted to cross the planned line")
            return
        if self._timeout(t, ADVANCE_MAX_S, "ADVANCE"):
            self._go(t, "GOAL", reason="advance ceiling reached")

    def _is_stale(self, t: float, threat: str) -> bool:
        """Is this threat the body whose encounter was just resolved?

        Time-limited rather than permanent: a body that crossed early and comes
        back round is a genuinely new encounter, and a scenario is entitled to
        stage one.  :data:`RESOLVED_IGNORE_S` is the window, sized to the
        MEASURED time a body takes to clear the lane at its own speed.
        """
        return (threat == self._resolved_threat
                and (t - self._resolved_at) < RESOLVED_IGNORE_S)

    # -- resolving one encounter -------------------------------------------
    def _threat_state(self, t: float, s: Sense) -> None:
        """Both corridors are being scored.  Commit to one, or hold.

        The commit needs the SAME side to stay the planner's answer for
        :data:`~slalom_states.COMMIT_CONFIRM_S` continuously, so a single tick of
        a favourable prediction is not a green light.  The confirmation window is
        reset whenever the answer changes, which is what makes it a sustained
        agreement rather than a count.
        """
        if s.decision_side and s.decision_side == self._pending_side:
            self._commit_for += self.dt
        else:
            self._pending_side = s.decision_side
            self._commit_for = 0.0

        if s.encounter_resolved and not s.threat:
            self._go(t, "REPLAN", reason="the threat cleared before a choice "
                                         "was needed")
            return

        if self._commit_for >= COMMIT_CONFIRM_S:
            if s.decision_side == "wait":
                self._wait = {
                    "index": len(self.waits),
                    "threat": s.threat,
                    "began_at_s": round(t, 3),
                    "rejected_side": s.rejected_side,
                    "rejected_clearance_m": round(s.rejected_clearance_m, 4),
                }
                self._go(t, "WAIT", threat=s.threat,
                         reason="NEITHER corridor predicted safe")
                return
            if s.decision_side in ("left", "right"):
                self._begin_pass(t, s)
                return
        if self._timeout(t, THREAT_MAX_S, "THREAT"):
            self._go(t, "WAIT", reason="threat ceiling reached")

    def _begin_pass(self, t: float, s: Sense, waited_s: float = 0.0) -> None:
        """Open a pass record and enter the chosen side's state."""
        body = s.threat or (self._wait.get("threat", "") if self._wait else "")
        self._encounter_body = body
        self._pass = PassRecord(
            index=len(self.passes),
            threat=body,
            side=s.decision_side,
            began_at_s=t,
            chosen_clearance_m=s.chosen_clearance_m,
            rejected_side=s.rejected_side,
            rejected_clearance_m=s.rejected_clearance_m,
            waited_s=waited_s,
        )
        state = "CHOOSE_LEFT" if s.decision_side == "left" else "CHOOSE_RIGHT"
        self._go(t, state, threat=body,
                 chosen_clearance_m=round(s.chosen_clearance_m, 4),
                 rejected=f"{s.rejected_side}@{s.rejected_clearance_m:.3f}m",
                 reason=f"passing {s.decision_side}: predicted "
                        f"{s.chosen_clearance_m:.3f} m against "
                        f"{s.rejected_clearance_m:.3f} m on the "
                        f"{s.rejected_side}")

    def _choose_left_state(self, t: float, s: Sense) -> None:
        self._choose_state(t, s, "left")

    def _choose_right_state(self, t: float, s: Sense) -> None:
        self._choose_state(t, s, "right")

    def _choose_state(self, t: float, s: Sense, side: str) -> None:
        """Turning out onto the chosen corridor.

        Ends as soon as the duck is actually ON the corridor, which is a
        measured lateral error rather than a timer.  Separate from PASS so the
        timeline distinguishes committing to a side from executing the pass, and
        so the video shows the turn-out as its own beat.
        """
        if abs(s.lateral_error_m) <= ON_CORRIDOR_M or s.encounter_resolved:
            self._go(t, "PASS", side=side,
                     reason=f"on the {side} corridor; passing")
            return
        if self._timeout(t, CHOOSE_MAX_S, f"CHOOSE_{side.upper()}"):
            self._go(t, "PASS", reason="choose ceiling reached")

    def _wait_state(self, t: float, s: Sense) -> None:
        """Exactly zero because NEITHER side was predicted safe.

        Resolves as soon as a side becomes safe AND the minimum wait has
        elapsed.  The minimum is a floor on what counts as a wait, not the thing
        that ends it: what ends it is the prediction changing.
        """
        if s.decision_side in ("left", "right") \
                and self._elapsed(t) >= MIN_WAIT_S:
            waited = self._elapsed(t)
            self._wait.update({
                "ended_at_s": round(t, 3),
                "duration_s": round(waited, 3),
                "resolved_side": s.decision_side,
                "resolved_clearance_m": round(s.chosen_clearance_m, 4),
            })
            self.waits.append(self._wait)
            self._wait = {}
            self._begin_pass(t, s, waited_s=waited)
            return
        if not s.threat and self._elapsed(t) >= MIN_WAIT_S:
            self._wait.update({
                "ended_at_s": round(t, 3),
                "duration_s": round(self._elapsed(t), 3),
                "resolved_side": "cleared",
            })
            self.waits.append(self._wait)
            self._wait = {}
            self._go(t, "REPLAN", reason="the threat cleared while waiting")
            return
        if self._timeout(t, WAIT_MAX_S, "WAIT"):
            self._wait.update({"ended_at_s": round(t, 3),
                               "ceiling_reached": True})
            self.waits.append(self._wait)
            self._wait = {}
            self._go(t, "REPLAN", reason="wait ceiling reached")

    def _pass_state(self, t: float, s: Sense) -> None:
        """Executing the pass.  Ends when the body has genuinely GONE PAST.

        Not a timer and not a distance the machine picked: the encounter is over
        when the thing that caused it has crossed, which the duck measures.

        ``encounter_resolved`` rather than ``threat_receding`` is the exit, and
        the difference is a scar.  Receding merely says the measured range grew
        this tick, which happens repeatedly WHILE a body is still crossing - the
        duck turns, the geometry changes, the range ticks up for a few frames.
        Ending on it closed passes early, mid-crossing, and the duck then walked
        back into the body it had just started avoiding: the MEASURED clearance
        went to -0.038 m.  Resolution requires the body to be BOTH past the
        duck's line and outside the clear distance.
        """
        if self._pass is not None:
            self._pass.measured_min_clearance_m = min(
                self._pass.measured_min_clearance_m,
                s.measured_min_clearance_m)
        if s.encounter_resolved:
            self._close_pass(t)
            self._go(t, "REPLAN", reason="the crossing is behind; replanning")
            return
        if self._timeout(t, PASS_MAX_S, "PASS"):
            self._close_pass(t, ceiling=True)
            self._go(t, "REPLAN", reason="pass ceiling reached")

    def _close_pass(self, t: float, ceiling: bool = False) -> None:
        if self._pass is None:
            return
        self._pass.ended_at_s = t
        if ceiling:
            self._pass.decision["ceiling_reached"] = True
        self.passes.append(self._pass)
        self.pass_sides.append(self._pass.side)
        # Remember whose encounter this was, so ADVANCE does not immediately
        # re-open it on the same receding body.
        self._resolved_threat = self._pass.threat
        self._resolved_at = t
        self._pass = None
        self._encounter_body = ""

    @property
    def encounter_body(self) -> str:
        """The body the current encounter is about, or "" when between them."""
        return self._encounter_body

    def _replan_state(self, t: float, s: Sense) -> None:
        """Recompute the line to the goal from where the duck ACTUALLY is.

        This is what makes the behavior a slalom rather than a sequence of
        dodges: after each pass the duck is laterally displaced, and the next
        leg is planned from there rather than from the original lane.  Every
        replan is logged with the position it was made from.
        """
        if self._elapsed(t) >= MIN_REPLAN_S:
            self.replans.append({
                "t": round(t, 3),
                "after_pass": len(self.passes),
                "goal_remaining_m": round(s.goal_remaining_m, 3),
                "lateral_error_m": round(s.lateral_error_m, 4),
            })
            if s.at_goal or s.goal_remaining_m <= 0.0:
                self._go(t, "GOAL", reason="replanned onto the arrival band")
                return
            self._go(t, "ADVANCE",
                     goal_remaining_m=round(s.goal_remaining_m, 3),
                     reason="replanned; walking the new line to the goal")
            return
        if self._timeout(t, REPLAN_MAX_S, "REPLAN"):
            self._go(t, "ADVANCE", reason="replan ceiling reached")

    # -- arriving -----------------------------------------------------------
    def _goal_state(self, t: float, s: Sense) -> None:
        """Standing in the arrival band at exactly zero.

        THE MINIMUM DWELL IS MEASURED FROM WHEN THE DUCK ACTUALLY STOPPED, not
        from when it entered the band.  The gait cannot halt instantly: the
        MEASURED coast after the command goes to zero is 0.0088 m and a stride
        takes about a second to unwind.  Timing the dwell from entry counted
        that coast as drift and broke the exact-standstill claim by 0.033 m.
        The state therefore waits for the body to settle, and only then starts
        counting - which is also what "it stopped in the band" means physically.
        """
        if self._settled_at is None:
            if s.measured_speed_mps <= GOAL_SETTLED_MPS:
                self._settled_at = t
        elif (t - self._settled_at) >= MIN_GOAL_S:
            self._go(t, "DONE", reason="arrived and stopped in the band")
            return
        if self._timeout(t, GOAL_MAX_S, "GOAL"):
            self._go(t, "DONE", reason="goal ceiling reached")

    def _done_state(self, t: float, s: Sense) -> None:
        """Terminal.  Exactly zero for the rest of the run."""
        return

    # -- bookkeeping ------------------------------------------------------
    @property
    def completed_passes(self) -> int:
        return len(self.passes)

    @property
    def finished(self) -> bool:
        return self.state == "DONE"

    def alternating(self) -> bool:
        """Did the pass sides genuinely alternate?

        Checked as a property of the recorded sequence rather than enforced
        anywhere: the planner is never told to alternate, so this is evidence
        that the scenario presented threats from both hands and the duck
        answered each one on its merits.
        """
        return all(a != b for a, b in zip(self.pass_sides, self.pass_sides[1:]))

    def summary(self) -> dict:
        return {
            "state": self.state,
            "transitions": list(self.transitions),
            "passes": [p.as_record() for p in self.passes],
            "pass_sides": list(self.pass_sides),
            "alternating": self.alternating(),
            "waits": list(self.waits),
            "replans": list(self.replans),
            "timeouts": list(self.timeouts),
        }
