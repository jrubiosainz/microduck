#!/usr/bin/env python3
"""What one session records: the commands, their execution, and the sense object
the machine decides on.

Split out of ``gest_machine`` so the machine stays about TRANSITIONS - what
causes the duck to change state - and this stays about EVIDENCE.  Neither module
touches physics.

WHY EACH RECORD CARRIES WHAT IT DOES
--------------------------------------
An episode that recorded only "COME was executed" could not distinguish a robot
that walked to a safe standoff from one that logged a state.  Each
:class:`Episode` therefore carries the range at both ends of the action, the
path actually walked, the trunk-yaw delta actually turned, the displacement
along the pre-action heading, and the peak command emitted - so every hard gate
is graded on a quantity that was measured rather than intended.

The turn and reverse fields are the ones that matter most.  ``yaw_delta_deg`` is
the trunk yaw the duck ACTUALLY turned through, and ``back_along_heading_m`` is
its displacement projected on the heading it held when the command was accepted.
Both are the honest form of a claim that is trivially faked by a label: a turn
that is "left" only because a state was named ``EXECUTE_TURN_LEFT`` proves
nothing, and neither does a reverse whose evidence is a negative command.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Episode:
    """One accepted command, from confirmation through to acknowledgment."""

    index: int
    command: str
    template: str
    person: str
    # -- how it was confirmed -------------------------------------------
    observed_at_s: float = 0.0
    confirmed_at_s: float = 0.0
    confirm_held_s: float = 0.0
    confirm_fraction: float = 0.0
    confirm_readable_fraction: float = 0.0
    confidence: float = 0.0
    rule: str = ""
    features: dict = field(default_factory=dict)
    # Ticks during the OBSERVE and CONFIRM states on which the instructor was
    # visible and her arm fully readable.  The acceptance gate requires both to
    # be complete, so these are what make "every acceptance required visibility"
    # a per-episode measurement rather than a run-wide average.
    confirm_ticks: int = 0
    confirm_visible_ticks: int = 0
    confirm_readable_ticks: int = 0
    # -- how it was executed --------------------------------------------
    executed_at_s: float = 0.0
    ended_at_s: float | None = None
    execute_state: str = ""
    start_xy: tuple[float, float] = (0.0, 0.0)
    end_xy: tuple[float, float] = (0.0, 0.0)
    start_yaw_deg: float = 0.0
    end_yaw_deg: float = 0.0
    start_range_m: float = 0.0
    end_range_m: float = 0.0
    path_m: float = 0.0
    yaw_delta_deg: float = 0.0
    back_along_heading_m: float = 0.0
    forward_along_heading_m: float = 0.0
    command_peak: float = 0.0
    command_vx_peak: float = 0.0
    command_vx_min: float = 0.0
    min_clearance_m: float = 1e9
    # -- the STOP case ----------------------------------------------------
    # The command magnitude on the tick BEFORE the stop was confirmed, and how
    # many ticks it took to reach exactly zero afterwards.  A STOP that
    # interrupted nothing would show a zero here, which is why it is recorded
    # rather than assumed.
    command_before_stop: float = 0.0
    ticks_to_zero: int | None = None
    stop_hold_s: float = 0.0
    stop_drift_m: float = 0.0
    # Which command, if any, cut this episode short.  A completed command and
    # one the duck was told to stop doing are different outcomes, and a log that
    # could not tell them apart would make "STOP interrupted the approach" an
    # inference rather than a record.
    interrupted_by: str = ""
    # The command this STOP episode itself interrupted, filled in by the
    # rollout, so each side of the interruption names the other.
    interrupts_command: str = ""
    # -- what the ACTION itself achieved, frozen when the execute state ends --
    #
    # THESE, NOT THE RUNNING FIELDS ABOVE, ARE WHAT EVERY PHYSICAL GATE IS
    # GRADED ON, AND THE DIFFERENCE IS A SCAR.  The running fields keep updating
    # for as long as the episode is open, which includes the ACK that follows
    # the action - and during ACK the duck holds an exact zero while its gait
    # unwinds, which rotates the trunk back a few degrees.  MEASURED: a turn
    # that reached +64.5 deg and satisfied its own exit test was recorded as
    # +57.8 deg once ACK had finished, so the episode log contradicted the
    # transition that closed it.  Freezing at the moment the action ends means
    # the number graded is the one the manoeuvre actually produced.
    execute_ended_s: float = 0.0
    execute_yaw_delta_deg: float = 0.0
    execute_back_m: float = 0.0
    execute_path_m: float = 0.0
    execute_end_range_m: float = 0.0
    execute_min_clearance_m: float = 1e9
    execute_in_standoff_band: bool = False

    @property
    def range_reduction_m(self) -> float:
        return float(self.start_range_m - self.end_range_m)

    def as_record(self) -> dict:
        return {
            "index": self.index,
            "command": self.command,
            "template": self.template,
            "person": self.person,
            "observed_at_s": round(float(self.observed_at_s), 3),
            "confirmed_at_s": round(float(self.confirmed_at_s), 3),
            "confirm_held_s": round(float(self.confirm_held_s), 3),
            "confirm_fraction": round(float(self.confirm_fraction), 4),
            "confirm_readable_fraction": round(
                float(self.confirm_readable_fraction), 4),
            "confirm_ticks": int(self.confirm_ticks),
            "confirm_visible_ticks": int(self.confirm_visible_ticks),
            "confirm_readable_ticks": int(self.confirm_readable_ticks),
            "confirm_visible_fraction": round(
                self.confirm_visible_ticks / max(self.confirm_ticks, 1), 4),
            "confirm_arm_readable_fraction": round(
                self.confirm_readable_ticks / max(self.confirm_ticks, 1), 4),
            "confidence": round(float(self.confidence), 4),
            "rule": self.rule,
            "features": {
                k: (round(float(v), 4) if isinstance(v, float) else v)
                for k, v in self.features.items()},
            "executed_at_s": round(float(self.executed_at_s), 3),
            "ended_at_s": (None if self.ended_at_s is None
                           else round(float(self.ended_at_s), 3)),
            "execute_state": self.execute_state,
            "start_xy": [round(float(self.start_xy[0]), 4),
                         round(float(self.start_xy[1]), 4)],
            "end_xy": [round(float(self.end_xy[0]), 4),
                       round(float(self.end_xy[1]), 4)],
            "start_yaw_deg": round(float(self.start_yaw_deg), 2),
            "end_yaw_deg": round(float(self.end_yaw_deg), 2),
            "start_range_m": round(float(self.start_range_m), 4),
            "end_range_m": round(float(self.end_range_m), 4),
            "range_reduction_m": round(self.range_reduction_m, 4),
            "path_m": round(float(self.path_m), 4),
            "yaw_delta_deg": round(float(self.yaw_delta_deg), 2),
            "back_along_heading_m": round(float(self.back_along_heading_m), 4),
            "forward_along_heading_m": round(
                float(self.forward_along_heading_m), 4),
            "command_peak": round(float(self.command_peak), 6),
            "command_vx_peak": round(float(self.command_vx_peak), 6),
            "command_vx_min": round(float(self.command_vx_min), 6),
            "min_clearance_m": (None if self.min_clearance_m > 1e8
                                else round(float(self.min_clearance_m), 4)),
            "command_before_stop": round(float(self.command_before_stop), 6),
            "ticks_to_zero": self.ticks_to_zero,
            "stop_hold_s": round(float(self.stop_hold_s), 3),
            "stop_drift_m": round(float(self.stop_drift_m), 5),
            "interrupted_by": self.interrupted_by,
            "interrupts_command": self.interrupts_command,
            "execute_ended_s": round(float(self.execute_ended_s), 3),
            "execute_yaw_delta_deg": round(float(self.execute_yaw_delta_deg), 2),
            "execute_back_m": round(float(self.execute_back_m), 4),
            "execute_path_m": round(float(self.execute_path_m), 4),
            "execute_end_range_m": round(float(self.execute_end_range_m), 4),
            "execute_min_clearance_m": (
                None if self.execute_min_clearance_m > 1e8
                else round(float(self.execute_min_clearance_m), 4)),
            "execute_in_standoff_band": bool(self.execute_in_standoff_band),
        }


@dataclass
class Sense:
    """Everything the duck MEASURED or CONCLUDED this tick, and nothing else.

    Every field is a quantity the robot could have obtained from its own pose,
    its own contact probe, its own camera or its own gesture reader.  Bundling
    them into one object is what makes it impossible for the machine to reach
    sideways into the scenario: if a value is not here, the machine cannot use
    it.
    """

    # -- who ---------------------------------------------------------------
    locked: str = ""
    acquisition_state: str = "search"
    instructor_visible: bool = False
    arm_readable: bool = False
    instructor_range_m: float = 1e9
    in_gesture_range: bool = False

    # -- what --------------------------------------------------------------
    candidate_command: str = ""
    candidate_held_s: float = 0.0
    candidate_fraction: float = 0.0
    candidate_confidence: float = 0.0
    confirm_progress: float = 0.0
    confirmed: dict | None = None

    # -- the duck's own state ------------------------------------------------
    measured_speed_mps: float = 0.0
    settled: bool = False
    duck_yaw_deg: float = 0.0
    # Progress of whatever physical action is under way, all MEASURED.
    yaw_delta_deg: float = 0.0
    back_along_heading_m: float = 0.0
    range_to_instructor_m: float = 1e9
    in_standoff_band: bool = False
    stop_hold_s: float = 0.0

    # -- safety --------------------------------------------------------------
    measured_min_clearance_m: float = 1e9
    inside_area: bool = True
