#!/usr/bin/env python3
"""The queueing state machine: transitions, cycles and the join decision.

    APPROACH -> OBSERVE_QUEUE -> IDENTIFY_TAIL -> EVALUATE_GAPS
             -> JOIN -> WAIT -> ADVANCE -> WAIT -> ... -> AT_COUNTER -> DONE

``WAIT`` and ``ADVANCE`` alternate, once per person served ahead of the duck, so
the cycle is a loop rather than a one-shot script.  ``AT_COUNTER`` is reachable
only after every predecessor has been served and left, which the machine checks
against the LIVE QUEUE rather than against the clock.

The machine never touches physics and never emits a command; ``queue_control``
does that from the state.  Keeping the two apart is what lets every transition
rule be unit-tested on hand-built inputs, with no MuJoCo anywhere.

TWO INVARIANTS ARE STRUCTURAL RATHER THAN CHECKED AFTERWARDS
------------------------------------------------------------
* **The duck may never pass its own target arc.**  Every moving state closes on
  an arc-length setpoint and stops when it reaches it, so overtaking is
  impossible by construction as well as being graded afterwards.
* **Reaching the counter is a consequence of the queue emptying**, never of
  elapsed time: ``predecessors_remaining`` is counted from the live reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from queue_constants import (
    ADVANCE_MAX_S,
    ADVANCE_TRIGGER_M,
    APPROACH_MAX_S,
    ARRIVE_TOLERANCE_M,
    AT_COUNTER_S,
    EVALUATE_S,
    IDENTIFY_S,
    JOIN_MAX_S,
    OBSERVE_S,
    SETTLE_S,
    WAIT_MAX_S,
)
from queue_geometry import AT_COUNTER_ARC_M, STANDOFF_TARGET_M
from queue_path import PATH


@dataclass
class QueueMachine:
    """The queueing state machine.  Owns transitions and the cycle log only.

    It never touches physics and never emits a command; the controller does
    that from the state.  Keeping the two apart is what lets every transition
    rule be unit-tested on hand-built inputs.
    """

    ctrl_hz: float = 50.0
    state: str = "APPROACH"
    state_since: float = 0.0
    # Where the duck is trying to stand, as an arc length on the queue path.
    target_arc: float | None = None
    # Who the duck is queueing behind right now.
    predecessor: str | None = None
    # The tail identified at join time, and the order read then.
    joined_behind: str | None = None
    identified_order: list[str] = field(default_factory=list)
    rejected_gaps: list[dict] = field(default_factory=list)
    accepted_gap: dict | None = None
    cycles: list[dict] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    _cycle: dict = field(default_factory=dict)
    _settled_since: float | None = None

    # -- helpers ---------------------------------------------------------
    def _advance_state(self, t: float, state: str, **detail) -> None:
        self.transitions.append(
            {"t": round(t, 3), "from": self.state, "to": state, **detail})
        self.state = state
        self.state_since = t
        self._settled_since = None

    def _elapsed(self, t: float) -> float:
        return t - self.state_since

    def _arrived(self, duck_arc: float, t: float) -> bool:
        """Has the duck reached its target arc AND held it briefly?

        THE SENSE OF THIS COMPARISON IS THE WHOLE MANOEUVRE.  The duck travels
        toward the counter, which is arc length ZERO, so it approaches every
        target from ABOVE and arrives when its arc has FALLEN to the setpoint.
        The first version tested ``duck_arc >= target_arc``, which is true from
        the instant the rollout starts - the duck begins near arc 4.5 and every
        target is smaller - so it 'arrived' before taking a step, joined 1.95 m
        behind the tail instead of 0.58 m, and then dribbled forward in 0.17 m
        advances for the rest of the rollout.  Thirty-one cycles were logged
        where there should have been five.  ``join_band`` and
        ``advances_are_real`` both failed, and both were this one inverted
        comparison.

        The settle window matters too: the coast after an exact-zero command was
        MEASURED at 0.011 m, so a duck that merely touched the setpoint is
        genuinely stopped a fraction of a second later, and the standoff the
        gate grades should be the settled one.
        """
        if self.target_arc is None:
            return False
        if duck_arc <= self.target_arc + ARRIVE_TOLERANCE_M:
            if self._settled_since is None:
                self._settled_since = t
            return (t - self._settled_since) >= SETTLE_S
        self._settled_since = None
        return False

    # -- the machine -----------------------------------------------------
    def update(self, t: float, *, duck_arc: float, duck_off_path_m: float,
               reading, gaps: list, predecessor_arc: float | None,
               predecessors_remaining: int) -> tuple[str, bool]:
        """Advance one control tick.

        ``reading`` is the current :class:`queue_model.QueueReading`; ``gaps``
        the judged candidate places.  ``predecessor_arc`` is where the person
        the duck is queueing behind currently stands, or ``None`` once they have
        been served.  ``predecessors_remaining`` counts queue members still
        ahead of the duck.
        """
        previous = self.state
        elapsed = self._elapsed(t)

        if self.state == "APPROACH":
            # The approach ends where the duck can SEE the whole queue and is
            # near the lane's open end - not at a fixed clock time.
            if duck_arc <= PATH.length - 0.05 and duck_off_path_m <= 0.55:
                self._advance_state(t, "OBSERVE_QUEUE", duck_arc=duck_arc)
            elif elapsed >= APPROACH_MAX_S:
                self.timeouts.append("approach_timeout")
                self._advance_state(t, "OBSERVE_QUEUE", reason="timeout")

        elif self.state == "OBSERVE_QUEUE":
            if elapsed >= OBSERVE_S:
                self._advance_state(t, "IDENTIFY_TAIL",
                                    seen=len(reading.order))

        elif self.state == "IDENTIFY_TAIL":
            if elapsed >= IDENTIFY_S and reading.tail is not None:
                self.identified_order = list(reading.order)
                self._advance_state(t, "EVALUATE_GAPS", tail=reading.tail)

        elif self.state == "EVALUATE_GAPS":
            if elapsed >= EVALUATE_S and gaps:
                accepted = next((g for g in gaps if g.accepted), None)
                if accepted is not None:
                    self.rejected_gaps = [
                        g.as_record() for g in gaps if not g.accepted]
                    self.accepted_gap = accepted.as_record()
                    self.joined_behind = accepted.ahead
                    self.predecessor = accepted.ahead
                    self.target_arc = float(accepted.arc)
                    self._cycle = {"kind": "join", "started_s": t,
                                   "behind": accepted.ahead,
                                   "target_arc_m": self.target_arc}
                    self._advance_state(t, "JOIN", target_arc=self.target_arc,
                                        behind=accepted.ahead)

        elif self.state == "JOIN":
            if self._arrived(duck_arc, t):
                self._close_cycle(t, duck_arc)
                self._advance_state(t, "WAIT", duck_arc=duck_arc)
            elif elapsed >= JOIN_MAX_S:
                self.timeouts.append("join_timeout")
                self._close_cycle(t, duck_arc, reason="timeout")
                self._advance_state(t, "WAIT", reason="timeout")

        elif self.state == "WAIT":
            if predecessors_remaining == 0:
                # Everybody ahead has been served: the counter is the duck's.
                self.predecessor = None
                self.target_arc = 0.0
                self._cycle = {"kind": "to_counter", "started_s": t,
                               "target_arc_m": 0.0,
                               "from_arc_m": duck_arc}
                self._advance_state(t, "ADVANCE", to_counter=True)
            elif predecessor_arc is not None:
                # The person in front has moved up: follow them.  The trigger is
                # the SLACK that has opened - how much further back the duck now
                # sits than the standoff it wants - which is a positive quantity
                # only once the predecessor has genuinely advanced.
                desired = predecessor_arc + STANDOFF_TARGET_M
                if duck_arc - desired >= ADVANCE_TRIGGER_M:
                    self.target_arc = float(desired)
                    self._cycle = {"kind": "advance", "started_s": t,
                                   "behind": self.predecessor,
                                   "target_arc_m": self.target_arc,
                                   "from_arc_m": duck_arc}
                    self._advance_state(t, "ADVANCE",
                                        target_arc=self.target_arc)
            if self.state == "WAIT" and elapsed >= WAIT_MAX_S:
                self.timeouts.append("wait_timeout")

        elif self.state == "ADVANCE":
            to_counter = bool(self._cycle.get("to_counter")
                              or self._cycle.get("kind") == "to_counter")
            # THE TARGET TRACKS THE PERSON IN FRONT WHILE THEY ARE MOVING.
            # A queue advance is not a move to a station fixed at the moment the
            # trigger fired: the predecessor is still walking when the duck sets
            # off, and freezing the setpoint then leaves the duck stopping
            # wherever the predecessor HAPPENED to be a second earlier.  That is
            # exactly what the first version did, and it stopped 1.149 m behind
            # the tail against a 0.45-0.75 m band, on four consecutive cycles,
            # with only 0.20-0.55 m of arc progress where 0.55 m was due.
            # Re-deriving the setpoint every tick from the predecessor's CURRENT
            # arc is both the fix and the honest description of following
            # somebody up a queue.
            if not to_counter and predecessor_arc is not None:
                self.target_arc = max(
                    float(predecessor_arc + STANDOFF_TARGET_M), 0.0)
            if self._arrived(duck_arc, t):
                self._close_cycle(t, duck_arc)
                if to_counter or duck_arc <= AT_COUNTER_ARC_M:
                    self._advance_state(t, "AT_COUNTER", duck_arc=duck_arc)
                else:
                    self._advance_state(t, "WAIT", duck_arc=duck_arc)
            elif elapsed >= ADVANCE_MAX_S:
                self.timeouts.append("advance_timeout")
                self._close_cycle(t, duck_arc, reason="timeout")
                self._advance_state(
                    t, "AT_COUNTER" if to_counter else "WAIT", reason="timeout")

        elif self.state == "AT_COUNTER":
            if elapsed >= AT_COUNTER_S:
                self._advance_state(t, "DONE", duck_arc=duck_arc)

        return self.state, self.state != previous

    def _close_cycle(self, t: float, duck_arc: float, **detail) -> None:
        if not self._cycle:
            return
        entry = dict(self._cycle)
        entry.update(detail)
        entry["completed_s"] = t
        entry["final_arc_m"] = float(duck_arc)
        entry["duration_s"] = t - float(entry.get("started_s", t))
        self.cycles.append(entry)
        self._cycle = {}

    @property
    def wait_advance_cycles(self) -> list[dict]:
        return [c for c in self.cycles if c.get("kind") == "advance"]
