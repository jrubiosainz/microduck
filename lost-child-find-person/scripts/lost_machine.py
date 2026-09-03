#!/usr/bin/env python3
"""The state machine: losing somebody, refusing look-alikes, and rejoining.

    FOLLOW -> LOST -> STOP -> SEARCH_SWEEP -> CANDIDATE -> REJECT -> (sweep)
           -> REACQUIRED -> REJOIN -> FOLLOW ... -> SAFE -> DONE

``CANDIDATE`` and ``REJECT`` alternate with ``SEARCH_SWEEP`` once per body the
duck evaluates, so the refusals are a loop rather than a scripted pair.  The
whole cycle can run more than once, and does.

The machine never touches physics and never emits a command; ``lost_control``
does that from the state.  Keeping the two apart is what lets every transition
rule be unit-tested on hand-built inputs, with no MuJoCo anywhere.

THREE INVARIANTS ARE STRUCTURAL RATHER THAN CHECKED AFTERWARDS
---------------------------------------------------------------
* **The duck cannot move while it does not know where its guardian is.**  Every
  state from ``LOST`` to ``REACQUIRED`` is in ``STATIONARY_STATES``, and the
  controller returns exact zero for those without consulting anything else.  It
  is not a rule the machine remembers to apply; it is the only thing the
  controller can do in those states.

* **A reacquisition requires the SAME identity, confirmed over time.**  The
  machine leaves ``REACQUIRED`` only when the identity tracker reports a
  continuous accept-grade confirmation of the ORIGINAL guardian name.  A
  look-alike cannot reach that transition even momentarily, because the accept
  verdict itself is name-checked in ``lost_identity.evaluate`` and again here.

* **The loss must be SUSTAINED.**  ``LOST`` is entered only after the guardian
  has been continuously invisible in the exact PiP camera for
  ``LOSS_CONFIRM_S``.  A single stride of somebody crossing the sightline is not
  a loss, and the machine will not let it become one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lost_constants import (
    FOLLOW_MAX_S,
    LOSS_CONFIRM_S,
    REACQUIRE_CONFIRM_S,
    REJECT_HOLD_S,
    REJOIN_MAX_S,
    SEARCH_MAX_S,
)

# How long the duck stands at its final standoff before the run is DONE.
SAFE_HOLD_S = 1.6
# How long STOP lasts: a deliberate halt that a viewer can see, spent entirely
# at exactly zero command before the search begins.
STOP_S = 0.8


@dataclass
class LostMachine:
    """Transitions and the cycle log.  No physics, no commands, no MuJoCo."""

    ctrl_hz: float = 50.0
    state: str = "FOLLOW"
    state_since: float = 0.0
    # The identity the duck is looking for.  Set once, never reassigned; the
    # machine refuses any attempt to change it.
    guardian: str = ""
    # The body the head should track right now: the guardian while following,
    # or the candidate being evaluated.
    subject: str | None = None
    # Filled when a candidate is being evaluated.
    candidate: str | None = None
    candidate_since: float = 0.0
    # Set at REACQUIRED, cleared when the rejoin completes.
    rejoin_goal: tuple[float, float] | None = None

    cycles: list[dict] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    _cycle: dict = field(default_factory=dict)
    _invisible_for: float = 0.0
    _safe_since: float | None = None

    def __post_init__(self) -> None:
        self.dt = 1.0 / self.ctrl_hz

    # -- helpers ---------------------------------------------------------
    def _go(self, t: float, state: str, **detail) -> None:
        self.transitions.append(
            {"t": round(t, 3), "from": self.state, "to": state, **detail})
        self.state = state
        self.state_since = t

    def _elapsed(self, t: float) -> float:
        return t - self.state_since

    def _timeout(self, t: float, limit: float, label: str) -> bool:
        if self._elapsed(t) >= limit:
            self.timeouts.append(f"{label}@{t:.2f}s")
            return True
        return False

    @property
    def cycle_index(self) -> int:
        return len(self.cycles)

    def set_guardian(self, name: str) -> None:
        """Record the identity being followed.  Callable once, by construction."""
        if self.guardian and self.guardian != name:
            raise ValueError(
                f"guardian identity is {self.guardian!r} and cannot be "
                f"reassigned to {name!r}: identity continuity is the behavior")
        self.guardian = name
        self.subject = name

    # -- the machine -----------------------------------------------------
    def update(self, t: float, *, guardian_visible: bool,
               guardian_confirmed_s: float, best_candidate,
               reached_goal: bool, tracker) -> tuple[str, bool]:
        """Advance one control tick.

        ``guardian_visible`` is the EXACT-PiP visibility of the guardian this
        tick.  ``guardian_confirmed_s`` is the continuous accept-grade
        confirmation time from the identity tracker.  ``best_candidate`` is the
        strongest evaluable sighting, or ``None``.  ``reached_goal`` is whether
        the duck has arrived at its rejoin standoff.

        Returns the state after the update and whether it changed this tick.
        """
        before = self.state

        if self.state == "FOLLOW":
            self._follow(t, guardian_visible)
        elif self.state == "LOST":
            self._lost(t)
        elif self.state == "STOP":
            self._stop(t)
        elif self.state == "SEARCH_SWEEP":
            self._search(t, best_candidate, tracker)
        elif self.state == "CANDIDATE":
            self._candidate(t, best_candidate, guardian_confirmed_s, tracker)
        elif self.state == "REJECT":
            self._reject(t)
        elif self.state == "REACQUIRED":
            self._reacquired(t, guardian_confirmed_s)
        elif self.state == "REJOIN":
            self._rejoin(t, reached_goal)
        elif self.state == "SAFE":
            self._safe(t)

        return self.state, self.state != before

    # -- per-state rules -------------------------------------------------
    def _follow(self, t: float, guardian_visible: bool) -> None:
        """Follow while she is in sight; declare a loss only when it is sustained."""
        self.subject = self.guardian
        if guardian_visible:
            self._invisible_for = 0.0
            return
        self._invisible_for += self.dt
        if self._invisible_for >= LOSS_CONFIRM_S:
            self._cycle = {
                "index": self.cycle_index,
                "lost_at_s": round(t, 3),
                "invisible_for_s": round(self._invisible_for, 3),
                "rejections": [],
            }
            self._go(t, "LOST",
                     reason=f"guardian not visible for {self._invisible_for:.2f}s")

    def _lost(self, t: float) -> None:
        """A single tick of acknowledged loss, then the deliberate halt.

        LOST exists as its own state rather than folding into STOP so the HUD,
        the timeline and the metrics can all name the instant the duck realised,
        separately from the halt that follows.
        """
        self.subject = None
        self._go(t, "STOP", reason="halting; guardian's position unknown")

    def _stop(self, t: float) -> None:
        self.subject = None
        if self._elapsed(t) >= STOP_S:
            self._go(t, "SEARCH_SWEEP", reason="beginning head sweep")

    def _search(self, t: float, best_candidate, tracker) -> None:
        """Sweep until something worth evaluating enters the frame."""
        self.subject = None
        if best_candidate is not None:
            self.candidate = best_candidate.name
            self.candidate_since = t
            self.subject = best_candidate.name
            self._go(t, "CANDIDATE", candidate=best_candidate.name,
                     score=round(best_candidate.score, 4))
            return
        self._timeout(t, SEARCH_MAX_S, "SEARCH_SWEEP")

    def _candidate(self, t: float, best_candidate, guardian_confirmed_s: float,
                   tracker) -> None:
        """Evaluate the body in frame: confirm it, or refuse it by name.

        THE GUARDIAN IS NOT EXEMPT FROM THE CONFIRMATION DURATION.  She is
        promoted only once the tracker reports a continuous accept-grade
        confirmation of HER name; until then she sits in CANDIDATE exactly as a
        look-alike would.
        """
        self.subject = self.candidate
        if (self.candidate == self.guardian
                and guardian_confirmed_s >= REACQUIRE_CONFIRM_S):
            self._go(t, "REACQUIRED", identity=self.guardian,
                     confirmed_s=round(guardian_confirmed_s, 3))
            return

        # A candidate that is no longer the strongest sighting, or has stopped
        # being evaluable, or has been in frame long enough to have been
        # confirmed and was not, is refused.
        gone = (best_candidate is None or best_candidate.name != self.candidate)
        stale = self._elapsed(t) >= REACQUIRE_CONFIRM_S
        if self.candidate != self.guardian and (gone or stale):
            sighting = (best_candidate if (best_candidate is not None
                                           and best_candidate.name == self.candidate)
                        else None)
            self._pending_reject = sighting
            self._go(t, "REJECT", candidate=self.candidate)
            return
        if self.candidate == self.guardian and gone:
            # She was in frame and slipped out again before confirming.  That is
            # not a rejection — it is a failed confirmation, and the duck goes
            # back to sweeping without recording a refusal of its own guardian.
            self.candidate = None
            self._go(t, "SEARCH_SWEEP", reason="guardian sighting lost before "
                                               "confirmation completed")

    def _reject(self, t: float) -> None:
        """Hold the refusal briefly so it is a visible decision, then resume."""
        self.subject = self.candidate
        if self._elapsed(t) >= REJECT_HOLD_S:
            self.candidate = None
            self._go(t, "SEARCH_SWEEP", reason="candidate refused; resuming sweep")

    def _reacquired(self, t: float, guardian_confirmed_s: float) -> None:
        """One tick to publish the reacquisition, then plan and walk the rejoin."""
        self.subject = self.guardian
        self._go(t, "REJOIN", identity=self.guardian,
                 confirmed_s=round(guardian_confirmed_s, 3))

    def _rejoin(self, t: float, reached_goal: bool) -> None:
        self.subject = self.guardian
        if reached_goal:
            self._close_cycle(t, "rejoined")
            self._go(t, "FOLLOW", reason="standoff reached; resuming follow")
            self._invisible_for = 0.0
            return
        if self._timeout(t, REJOIN_MAX_S, "REJOIN"):
            self._close_cycle(t, "timeout")
            self._go(t, "FOLLOW", reason="rejoin ceiling reached")
            self._invisible_for = 0.0

    def _safe(self, t: float) -> None:
        self.subject = self.guardian
        if self._safe_since is None:
            self._safe_since = t
        if t - self._safe_since >= SAFE_HOLD_S:
            self._go(t, "DONE", reason="standing at safe standoff")

    # -- cycle bookkeeping ------------------------------------------------
    def _close_cycle(self, t: float, outcome: str) -> None:
        if not self._cycle:
            return
        self._cycle.update({
            "rejoined_at_s": round(t, 3),
            "outcome": outcome,
            "duration_s": round(t - self._cycle["lost_at_s"], 3),
        })
        self.cycles.append(self._cycle)
        self._cycle = {}

    def note_rejection(self, record: dict) -> None:
        """Attach a refusal to the cycle it happened in, and to the run log."""
        self.rejections.append(record)
        if self._cycle:
            self._cycle["rejections"].append(record)

    def finish(self, t: float) -> None:
        """Enter SAFE from a rollout that has run its course while following."""
        if self.state in ("FOLLOW", "REJOIN"):
            if self.state == "REJOIN":
                self._close_cycle(t, "ended in rejoin")
            self._go(t, "SAFE", reason="rollout complete at safe standoff")

    def summary(self) -> dict:
        return {
            "guardian": self.guardian,
            "state": self.state,
            "cycles": list(self.cycles),
            "cycle_count": len(self.cycles),
            "transitions": list(self.transitions),
            "rejections": list(self.rejections),
            "timeouts": list(self.timeouts),
        }
