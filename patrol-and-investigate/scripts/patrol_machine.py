#!/usr/bin/env python3
"""The state machine: patrol, stop, scan, and - when something is wrong - break
off, approach, observe, classify, go back and carry on.

    PATROL -> CHECKPOINT_STOP -> SCAN -> CLEAR ------------------> PATROL
                                     -> DETECT -> INVESTIGATE_PLAN
                                        -> APPROACH -> OBSERVE -> CLASSIFY
                                        -> RETURN_TO_PATROL -> RESUME -> PATROL
    ... five checkpoints ...          -> HOME -> DONE

The cycle CHECKPOINT_STOP -> SCAN -> CLEAR runs once per checkpoint, and the
DETECT..RESUME branch runs once per anomaly.  Both rejoin PATROL, which is what
makes the patrol RESUMABLE rather than restartable.

This file owns the circuit half; ``patrol_branch`` owns the investigation half
and is mixed in.  Neither touches physics and neither emits a command -
``patrol_control`` does that from the state.  Keeping them apart is what lets
every transition rule be unit-tested on hand-built inputs, with no MuJoCo
anywhere.

FIVE INVARIANTS ARE STRUCTURAL RATHER THAN CHECKED AFTERWARDS
---------------------------------------------------------------
* **Every detection is CAUSED by the camera, never scheduled.**  There is no
  timer and no schedule lookup in either file.  The duck leaves SCAN for DETECT
  because its own detector confirmed a candidate it could SEE, and the detector
  cannot be reached by a body outside the frustum.

* **An investigation cannot lose the patrol's place.**  Breaking off calls
  ``PatrolPlan.interrupt``, which snapshots the target checkpoint and the resume
  point; nothing in the diversion may touch ``target_index``.  RESUME then
  compares the plan's own live target against that snapshot, so "the route was
  preserved" is a comparison of two values written at different times rather
  than an assertion.

* **The duck stops before it looks.**  SCAN is entered only from
  CHECKPOINT_STOP, and CHECKPOINT_STOP is left only once the duck's own MEASURED
  speed says it has settled.  A robot that scanned while still rolling would
  never reach SCAN at all.

* **A checkpoint is completed by the ROLLOUT, once, on leaving it.**  The
  machine records the visit; the plan advances the index.  Splitting them is
  what makes it impossible for a diversion to advance the patrol.

* **A ceiling MOVES the machine.**  Every ``_timeout`` here transitions and
  records why.  A ceiling that only appends to a log is not a ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from patrol_branch import InvestigationBranch
from patrol_episode import CheckpointVisit, Investigation, Observation, Sense
from patrol_states import (
    CHECKPOINT_RESULT_S,
    CHECKPOINT_STOP_MAX_S,
    CHECKPOINT_STOP_S,
    CLEAR_MAX_S,
    HOME_MAX_S,
    PATROL_MAX_S,
    SCAN_MAX_S,
    SCAN_PERIOD_S,
)

# How long the duck must be standing on the guard post before the patrol is
# DONE.  Long enough that the completed circuit is a visible beat rather than a
# transition, and measured from when the body actually settled.
HOME_DWELL_S = 1.5


@dataclass
class PatrolMachine(InvestigationBranch):
    """Transitions, the checkpoint log and the investigation log.  No physics."""

    ctrl_hz: float = 50.0
    state: str = "PATROL"
    state_since: float = 0.0

    transitions: list[dict] = field(default_factory=list)
    visits: list[CheckpointVisit] = field(default_factory=list)
    investigations: list[Investigation] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    # Every verdict in the order it was reached, INCLUDING the dismissals.
    verdicts: list[dict] = field(default_factory=list)

    _visit: CheckpointVisit | None = None
    _investigation: Investigation | None = None
    _observation: Observation | None = None
    _angle_index: int = 0
    # The candidate the current DETECT..CLASSIFY branch is about.  Set when the
    # branch opens and cleared when it closes, so every downstream state acts on
    # the body the detection was about rather than on whatever is nearest now.
    _subject: str = ""
    _dismissed: bool = False
    _settled_at: float | None = None

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

    # -- the machine -----------------------------------------------------
    def update(self, t: float, sense: Sense) -> tuple[str, bool]:
        """Advance one control tick on measurements taken BEFORE this tick's
        physics."""
        before = self.state
        getattr(self, f"_{self.state.lower()}_state")(t, sense)
        return self.state, self.state != before

    # -- walking the circuit ------------------------------------------------
    def _patrol_state(self, t: float, s: Sense) -> None:
        """Walk the leg to the next checkpoint, or home when the circuit is done.

        THE ORDER OF THE EXITS MATTERS.  Arriving wins over a detection, so a
        body that becomes visible in the last stride before a checkpoint does
        not stop the duck short of it: the checkpoint is reached, the scan
        happens, and the detection is made from the checkpoint - which is where
        a guard robot is supposed to be looking from.
        """
        if s.at_target:
            if s.finished_circuit:
                self._go(t, "HOME",
                         reason="the circuit is complete; standing down at the "
                                "guard post")
                return
            self._open_visit(t, s)
            return
        if self._timeout(t, PATROL_MAX_S, "PATROL"):
            if s.finished_circuit:
                self._go(t, "HOME", reason="patrol leg ceiling reached")
            else:
                self._open_visit(t, s)

    def _open_visit(self, t: float, s: Sense) -> None:
        self._visit = CheckpointVisit(
            index=len(self.visits), name=s.target_name, arrived_at_s=t,
            arrival_error_m=s.target_remaining_m)
        self._go(t, "CHECKPOINT_STOP", checkpoint=s.target_name,
                 arrival_error_m=round(s.target_remaining_m, 4),
                 reason=f"arrived at {s.target_name}: stopping")

    def _checkpoint_stop_state(self, t: float, s: Sense) -> None:
        """Come to a complete stop before looking at anything.

        The exit requires BOTH a minimum dwell and the duck's own MEASURED speed
        having fallen to a standstill.  Timing the dwell alone would count the
        gait's coast as part of the stop; the MEASURED 0.0091 m coast is small,
        but a checkpoint stop that began while the robot was still rolling is
        not a stop.
        """
        if self._elapsed(t) >= CHECKPOINT_STOP_S and s.settled:
            if self._visit is not None:
                self._visit.stopped_s = self._elapsed(t)
            self._go(t, "SCAN", checkpoint=s.target_name,
                     reason="stopped; sweeping the head across the facility")
            return
        if self._timeout(t, CHECKPOINT_STOP_MAX_S, "CHECKPOINT_STOP"):
            self._go(t, "SCAN", reason="checkpoint stop ceiling reached")

    def _scan_state(self, t: float, s: Sense) -> None:
        """Sweep the head through its arc, at an exact zero command.

        The scan ends when the SWEEP is complete - a property of the head's own
        travel rather than a countdown the machine could race - or when the
        detector confirms something worth acting on, whichever comes first.
        Leaving early on a detection is correct: a guard that finished its sweep
        before reacting to an intruder it had already identified would be a
        guard following a script rather than watching.
        """
        if self._visit is not None:
            self._visit.scan_s = self._elapsed(t)
            self._visit.scan_arc_deg = max(self._visit.scan_arc_deg,
                                           s.scan_arc_deg)
            self._visit.bodies_seen = s.bodies_seen

        if s.candidate and s.candidate_visible:
            self._go(t, "DETECT", target=s.candidate,
                     verdict=s.candidate_verdict,
                     range_m=round(s.candidate_range_m, 4),
                     reason=f"{s.candidate} confirmed in the head camera "
                            "during the sweep")
            return
        if s.scan_complete or self._elapsed(t) >= SCAN_PERIOD_S:
            self._go(t, "CLEAR", checkpoint=s.target_name,
                     arc_deg=round(s.scan_arc_deg, 1), seen=len(s.bodies_seen),
                     reason="sweep complete, nothing to investigate")
            return
        if self._timeout(t, SCAN_MAX_S, "SCAN"):
            self._go(t, "CLEAR", reason="scan ceiling reached")

    def _clear_state(self, t: float, s: Sense) -> None:
        """Record the checkpoint as clear and move on to the next one."""
        if self._elapsed(t) >= CHECKPOINT_RESULT_S \
                or self._timeout(t, CLEAR_MAX_S, "CLEAR"):
            self._close_visit(t, "clear")
            self._go(t, "PATROL",
                     reason="checkpoint clear; walking the next leg")

    def _close_visit(self, t: float, result: str, detected: str = "") -> None:
        if self._visit is None:
            return
        self._visit.result = result
        self._visit.detected = detected
        self._visit.ended_at_s = t
        self.visits.append(self._visit)
        self._visit = None

    # -- finishing ------------------------------------------------------------
    def _home_state(self, t: float, s: Sense) -> None:
        """Standing on the guard post at exactly zero, patrol complete.

        THE DWELL IS MEASURED FROM WHEN THE DUCK ACTUALLY STOPPED, not from when
        it reached the pad.  The gait cannot halt instantly - the MEASURED coast
        is 0.0091 m - so timing from arrival would count that coast as drift
        during a state whose whole claim is stillness.
        """
        if self._settled_at is None:
            if s.settled:
                self._settled_at = t
        elif (t - self._settled_at) >= HOME_DWELL_S:
            self._go(t, "DONE",
                     reason="patrol complete: five checkpoints visited in "
                            "order, two investigations, home")
            return
        if self._timeout(t, HOME_MAX_S, "HOME"):
            self._go(t, "DONE", reason="home ceiling reached")

    def _done_state(self, t: float, s: Sense) -> None:
        """Terminal.  Exactly zero for the rest of the run."""
        return

    # -- bookkeeping ---------------------------------------------------------
    def open_investigation(self, investigation: Investigation) -> None:
        """Attach the investigation record the rollout built."""
        self._investigation = investigation
        self._angle_index = 0
        self._observation = None

    def close_investigation(self, t: float) -> None:
        if self._investigation is None:
            return
        self._investigation.ended_at_s = t
        self.investigations.append(self._investigation)
        self._investigation = None

    def record_verdict(self, t: float, verdict: dict) -> None:
        self.verdicts.append({"t": round(float(t), 3), **verdict})

    @property
    def investigation(self) -> Investigation | None:
        return self._investigation

    @property
    def subject(self) -> str:
        """The body the current investigation branch is about, or ``""``."""
        return self._subject

    @property
    def dismissing(self) -> bool:
        return self._dismissed

    @property
    def visited_names(self) -> list[str]:
        return [v.name for v in self.visits]

    @property
    def finished(self) -> bool:
        return self.state == "DONE"

    def summary(self) -> dict:
        return {
            "state": self.state,
            "transitions": list(self.transitions),
            "visits": [v.as_record() for v in self.visits],
            "visited_order": self.visited_names,
            "investigations": [i.as_record() for i in self.investigations],
            "verdicts": list(self.verdicts),
            "timeouts": list(self.timeouts),
        }
