#!/usr/bin/env python3
"""What one patrol records: checkpoint visits, investigations, and the sense
object the machine decides on.

Split out of ``patrol_machine`` so the machine stays about TRANSITIONS - what
causes the duck to change state - and this stays about EVIDENCE.  Neither module
touches physics.

WHY EACH RECORD CARRIES WHAT IT DOES
--------------------------------------
A checkpoint visit that recorded only "visited" could not distinguish a robot
that stopped and looked from one that walked past.  Each :class:`CheckpointVisit`
therefore carries the time the body was actually stationary, the arc the head
swept, how many distinct bodies the camera resolved during the sweep, and the
result - so "it stopped and scanned" is four measurements rather than a caption.

An investigation that recorded only its verdict could not distinguish a decision
from a default.  Each :class:`Investigation` carries the range at which the
anomaly was first detected, the range the approach ended at, the standoff band it
had to land in, every observation angle actually held, and the point the patrol
was interrupted at - so the whole episode can be checked rather than believed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckpointVisit:
    """One checkpoint: the stop, the scan, and what came of it."""

    index: int
    name: str
    arrived_at_s: float
    stopped_s: float = 0.0
    scan_s: float = 0.0
    scan_arc_deg: float = 0.0
    bodies_seen: tuple[str, ...] = ()
    result: str = ""              # "clear" | "detect"
    detected: str = ""
    ended_at_s: float | None = None
    # The MEASURED path the duck walked during the stop and the scan.  This is
    # the number that makes "it physically stopped" a claim about the floor
    # rather than about the command.
    still_path_m: float = 0.0
    arrival_error_m: float = 0.0

    def as_record(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "arrived_at_s": round(float(self.arrived_at_s), 3),
            "ended_at_s": (None if self.ended_at_s is None
                           else round(float(self.ended_at_s), 3)),
            "stopped_s": round(float(self.stopped_s), 3),
            "scan_s": round(float(self.scan_s), 3),
            "scan_arc_deg": round(float(self.scan_arc_deg), 2),
            "bodies_seen": list(self.bodies_seen),
            "result": self.result,
            "detected": self.detected,
            "still_path_m": round(float(self.still_path_m), 5),
            "arrival_error_m": round(float(self.arrival_error_m), 4),
        }


@dataclass
class Observation:
    """One held viewing angle during a multi-angle observation."""

    angle_deg: float
    began_at_s: float
    held_s: float = 0.0
    visible_steps: int = 0
    steps: int = 0
    min_range_m: float = 1e9

    @property
    def visible_fraction(self) -> float:
        return self.visible_steps / self.steps if self.steps else 0.0

    def as_record(self) -> dict:
        return {
            "angle_deg": round(float(self.angle_deg), 2),
            "began_at_s": round(float(self.began_at_s), 3),
            "held_s": round(float(self.held_s), 3),
            "visible_fraction": round(self.visible_fraction, 4),
            "steps": int(self.steps),
            "min_range_m": (None if self.min_range_m > 1e8
                            else round(float(self.min_range_m), 4)),
        }


@dataclass
class Investigation:
    """One complete DETECT -> ... -> RESUME episode, with all its evidence."""

    index: int
    target: str
    detected_at_s: float
    detect_range_m: float
    interrupted_checkpoint: str
    interrupted_index: int
    # -- the approach ----------------------------------------------------
    approach_began_s: float = 0.0
    approach_start_range_m: float = 0.0
    approach_end_range_m: float = 0.0
    approach_path_m: float = 0.0
    standoff_m: float = 0.0
    standoff_xy: tuple[float, float] = (0.0, 0.0)
    rejected_standoffs: int = 0
    # -- the observation --------------------------------------------------
    observations: list[Observation] = field(default_factory=list)
    observe_command_max: float = 0.0
    # -- the verdict -------------------------------------------------------
    verdict: str = ""
    rule: str = ""
    confidence: float = 0.0
    # -- getting back ------------------------------------------------------
    resumed_at_s: float = 0.0
    return_error_m: float = 0.0
    resumed_checkpoint: str = ""
    # -- safety ------------------------------------------------------------
    min_clearance_m: float = 1e9
    min_zone_gap_m: float = 1e9
    ended_at_s: float | None = None

    @property
    def angles_held(self) -> int:
        return len(self.observations)

    @property
    def range_reduction_m(self) -> float:
        """How much the approach actually closed.  MEASURED, both ends."""
        return float(self.approach_start_range_m - self.approach_end_range_m)

    def as_record(self) -> dict:
        return {
            "index": self.index,
            "target": self.target,
            "detected_at_s": round(float(self.detected_at_s), 3),
            "detect_range_m": round(float(self.detect_range_m), 4),
            "interrupted_checkpoint": self.interrupted_checkpoint,
            "interrupted_index": int(self.interrupted_index),
            "approach_began_s": round(float(self.approach_began_s), 3),
            "approach_start_range_m": round(
                float(self.approach_start_range_m), 4),
            "approach_end_range_m": round(float(self.approach_end_range_m), 4),
            "range_reduction_m": round(self.range_reduction_m, 4),
            "approach_path_m": round(float(self.approach_path_m), 4),
            "standoff_m": round(float(self.standoff_m), 4),
            "standoff_xy": [round(float(self.standoff_xy[0]), 4),
                            round(float(self.standoff_xy[1]), 4)],
            "rejected_standoffs": int(self.rejected_standoffs),
            "observations": [o.as_record() for o in self.observations],
            "angles_held": self.angles_held,
            "observe_command_max": round(float(self.observe_command_max), 6),
            "verdict": self.verdict,
            "rule": self.rule,
            "confidence": round(float(self.confidence), 4),
            "resumed_at_s": round(float(self.resumed_at_s), 3),
            "return_error_m": round(float(self.return_error_m), 4),
            "resumed_checkpoint": self.resumed_checkpoint,
            "route_preserved": bool(
                self.resumed_checkpoint == self.interrupted_checkpoint),
            "min_clearance_m": (None if self.min_clearance_m > 1e8
                                else round(float(self.min_clearance_m), 4)),
            "min_zone_gap_m": (None if self.min_zone_gap_m > 1e8
                               else round(float(self.min_zone_gap_m), 4)),
            "ended_at_s": (None if self.ended_at_s is None
                           else round(float(self.ended_at_s), 3)),
        }


@dataclass
class Sense:
    """Everything the duck MEASURED or CONCLUDED this tick, and nothing else.

    Every field is a quantity the robot could have obtained from its own pose,
    its own contact probe, its own camera or its own detector.  Bundling them
    into one object is what makes it impossible for the machine to reach
    sideways into the scenario: if a value is not here, the machine cannot use
    it.
    """

    # -- where the duck is on its route ---------------------------------
    target_name: str = ""
    target_remaining_m: float = 1e9
    at_target: bool = False
    at_home: bool = False
    finished_circuit: bool = False
    completed: int = 0

    # -- has it stopped -------------------------------------------------
    measured_speed_mps: float = 0.0
    settled: bool = False

    # -- the scan --------------------------------------------------------
    scan_arc_deg: float = 0.0
    scan_complete: bool = False
    bodies_seen: tuple[str, ...] = ()

    # -- what the detector concluded -------------------------------------
    # The candidate this tick, its verdict, and whether it is worth breaking
    # the patrol for.  ``candidate`` is empty when nothing is confirmed.
    candidate: str = ""
    candidate_verdict: str = ""
    candidate_rule: str = ""
    candidate_confidence: float = 0.0
    candidate_range_m: float = 1e9
    candidate_investigate: bool = False
    candidate_visible: bool = False

    # -- the investigation ------------------------------------------------
    target_range_m: float = 1e9
    standoff_ready: bool = False
    standoff_remaining_m: float = 1e9
    in_standoff_band: bool = False
    observe_elapsed_s: float = 0.0
    observations_done: int = 0

    # -- getting back ------------------------------------------------------
    resume_remaining_m: float = 1e9
    at_resume_point: bool = False

    # -- what actually happened --------------------------------------------
    measured_min_clearance_m: float = 1e9
    zone_gap_m: float = 1e9
