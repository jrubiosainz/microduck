#!/usr/bin/env python3
"""What one encounter records: the pass, the wait, and the decision behind them.

Split out of ``slalom_machine`` so the machine stays about TRANSITIONS - what
causes the duck to change state - and this stays about EVIDENCE.  Neither module
touches physics.

WHY THE REJECTED SIDE LIVES IN THE RECORD
-------------------------------------------
A pass that only recorded the side it took could not distinguish a decision from
a habit.  Every :class:`PassRecord` therefore carries the corridor the planner
REJECTED and its predicted clearance beside the one it chose, so the acceptance
gate can require the choice to have been justified by a measured comparison
rather than by a caption.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PassRecord:
    """One completed encounter, with the decision that produced it."""

    index: int
    threat: str
    side: str
    began_at_s: float
    chosen_clearance_m: float
    rejected_side: str
    rejected_clearance_m: float
    waited_s: float = 0.0
    ended_at_s: float | None = None
    measured_min_clearance_m: float = 1e9
    decision: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        return {
            "index": self.index,
            "threat": self.threat,
            "side": self.side,
            "began_at_s": round(float(self.began_at_s), 3),
            "ended_at_s": (None if self.ended_at_s is None
                           else round(float(self.ended_at_s), 3)),
            "waited_s": round(float(self.waited_s), 3),
            "chosen_clearance_m": round(float(self.chosen_clearance_m), 4),
            "rejected_side": self.rejected_side,
            "rejected_clearance_m": round(float(self.rejected_clearance_m), 4),
            "measured_min_clearance_m": (
                round(float(self.measured_min_clearance_m), 4)
                if self.measured_min_clearance_m < 1e8 else None),
            "decision": self.decision,
        }


@dataclass
class Sense:
    """Everything the duck MEASURED or PREDICTED this tick, and nothing else.

    Every field is a quantity the robot could have obtained from its own pose,
    its own contact probe, its own camera or its own predictor.  Bundling them
    into one object is what makes it impossible for the machine to reach
    sideways into the scenario: if a value is not here, the machine cannot use
    it.
    """

    # -- the duck's own progress ---------------------------------------
    goal_remaining_m: float = 1e9
    at_goal: bool = False
    leg_arrived: bool = False
    lateral_error_m: float = 0.0

    # -- what the predictor says ----------------------------------------
    threat: str = ""
    threat_ttc_s: float = 1e9
    threat_range_m: float = 1e9
    threat_receding: bool = False
    # True once the body THIS ENCOUNTER IS ABOUT has genuinely gone past.
    #
    # BOUND TO THE COMMITTED BODY, NOT TO WHATEVER THE PLANNER CURRENTLY FLAGS,
    # AND THAT DISTINCTION IS A SCAR.  An earlier version ended a pass as soon
    # as ``threat`` went empty - but the threat goes empty precisely BECAUSE the
    # sidestep worked, while the crossing body has not crossed yet.  One real
    # crossing was therefore logged as two encounters (``mara`` at 1.4 s and
    # again at 13.0 s, ``tobin`` at 14.9 s and again at 28.7 s), which inflated
    # the pass count, destroyed the alternation claim and produced "passes"
    # 1.0 s long.  A manoeuvre that removes a predicted conflict has succeeded,
    # not finished.
    encounter_resolved: bool = False
    decision_side: str = ""            # "left" | "right" | "wait" | ""
    chosen_clearance_m: float = 0.0
    rejected_side: str = ""
    rejected_clearance_m: float = 0.0
    any_side_safe: bool = False

    # -- what actually happened ------------------------------------------
    measured_min_clearance_m: float = 1e9
    goal_visible: bool = False
    # The duck's own measured ground speed this tick, from its two most recent
    # measured positions.  Used only to decide when it has genuinely stopped.
    measured_speed_mps: float = 0.0
