#!/usr/bin/env python3
"""The guardian state machine and the two locomotion controllers.

Pure Python/numpy: no MuJoCo, no ONNX, no rendering.  Fully unit-tested.

    APPROACH_CURB → STOP → LOOK_LEFT → LOOK_RIGHT → LOOK_LEFT_AGAIN
                  → WAIT_FOR_GAP → CROSSING → SAFE

Three properties are structural rather than hoped for:

* **The scan cannot be skipped, and it cannot be faked.**  Each LOOK phase
  advances only after its dwell time AND after the corresponding road sector
  has been genuinely visible through the camera for a continuous
  ``SECTOR_CONFIRM_S``.  The camera verdict is supplied by the caller; the
  machine never infers visibility from geometry it computed itself.
* **The gap decision happens in WAIT_FOR_GAP and nowhere else.**  Every
  candidate gap is recorded with its worst margin and its limiting vehicle,
  whether accepted or rejected, so "the duck rejected an unsafe gap" is
  evidence rather than narration.
* **A crossing, once started, is not re-decided.**  Stopping in the middle of a
  live traffic lane to reconsider is the single worst thing a pedestrian can
  do.  The commitment is taken with a margin that already covers the entire
  crossing, so the correct response to a surprise is to keep walking, and the
  controller has no command that would stop mid-road.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from conflict import (
    CROSSING_MAX_S,
    GAP_CONFIRM_S,
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    LOOK_LEFT_AGAIN_S,
    LOOK_LEFT_S,
    LOOK_RIGHT_S,
    SAFE_HOLD_S,
    STOP_HOLD_S,
    VX_APPROACH,
    VX_CROSS,
    WAIT_MAX_S,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
    clamp,
    wrap_angle,
)
from street import CROSS_GOAL_X, CURB_STOP_X, in_safe_zone

# Each LOOK phase, its dwell time, and the road sector that must be visible.
LOOK_PHASES: tuple[tuple[str, float, str], ...] = (
    ("LOOK_LEFT", LOOK_LEFT_S, "left"),
    ("LOOK_RIGHT", LOOK_RIGHT_S, "right"),
    ("LOOK_LEFT_AGAIN", LOOK_LEFT_AGAIN_S, "left"),
)
LOOK_SECTOR: dict[str, str] = {name: sector for name, _, sector in LOOK_PHASES}
LOOK_DWELL: dict[str, float] = {name: dwell for name, dwell, _ in LOOK_PHASES}
# A sector must be continuously visible for this long before the phase is
# allowed to advance, so one frame of the head sweep clipping the sector cannot
# count as having looked.
SECTOR_CONFIRM_S: float = 0.50
# Ceiling on a single LOOK phase, so a camera that can never satisfy the sector
# gate fails loudly instead of hanging the rollout.
LOOK_MAX_S: float = 9.0

# Where the approach releases its command.  MEASURED overshoot at the approach
# speed (vx=0.52) is ~60 mm from release to rest — far more than the 5-9 mm
# coast measured after a slower cruise in the sibling behaviors, so a
# one-centimetre allowance is not enough.  The release is placed 60 mm short of
# the stop target and the residual overshoot lands the duck on it.
APPROACH_RELEASE_X: float = CURB_STOP_X - 0.060


@dataclass
class GuardianMachine:
    """The eight-state crossing machine.

    The machine owns WHICH phase is active and WHY it advanced.  It does not
    own geometry: the duck's position, the camera's sector verdict and the gap
    decision are all supplied by the caller.  That split is what lets the whole
    machine be tested without MuJoCo.
    """

    ctrl_hz: float = 50.0
    state: str = "APPROACH_CURB"
    state_since: float = 0.0
    events: list[dict] = field(default_factory=list)
    gap_decisions: list[dict] = field(default_factory=list)
    scan_log: list[dict] = field(default_factory=list)
    commit: dict = field(default_factory=dict)
    timeouts: list[str] = field(default_factory=list)
    _sector_seen_for: float = 0.0
    _gap_safe_for: float = 0.0
    _rejected: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_hz

    @property
    def moving(self) -> bool:
        return self.state in ("APPROACH_CURB", "CROSSING")

    @property
    def committed(self) -> bool:
        return self.state in ("CROSSING", "SAFE")

    @property
    def rejected_gaps(self) -> list[dict]:
        """Distinct unsafe gaps that were explicitly refused, one per vehicle.

        Collapsed by limiting vehicle rather than per tick: rejecting the same
        van on 300 consecutive ticks is one refusal, and counting it as 300
        would make the acceptance gate trivially satisfiable.
        """
        return list(self._rejected.values())

    def _advance(self, t: float, state: str, **event) -> None:
        self.events.append({"t": t, "from": self.state, "to": state, **event})
        self.state = state
        self.state_since = t
        self._sector_seen_for = 0.0
        self._gap_safe_for = 0.0

    def update(
        self,
        t: float,
        *,
        trunk_x: float,
        sector_visible: bool,
        decision=None,
    ) -> tuple[str, bool]:
        """Advance one tick. Returns ``(state, changed)``.

        ``sector_visible`` is the CAMERA's verdict for the sector this LOOK
        phase requires, measured through the exact camera the PiP renders from.
        ``decision`` is a :class:`conflict.GapDecision` and is consulted only in
        ``WAIT_FOR_GAP``.
        """
        elapsed = t - self.state_since
        previous = self.state

        if self.state == "APPROACH_CURB":
            if trunk_x >= APPROACH_RELEASE_X:
                self._advance(t, "STOP", trunk_x=trunk_x,
                              reason="reached wait line")
        elif self.state == "STOP":
            if elapsed >= STOP_HOLD_S:
                self._advance(t, "LOOK_LEFT", trunk_x=trunk_x)
        elif self.state in LOOK_SECTOR:
            sector = LOOK_SECTOR[self.state]
            if sector_visible:
                self._sector_seen_for += self.dt
            else:
                self._sector_seen_for = 0.0
            dwell_done = elapsed >= LOOK_DWELL[self.state]
            seen_enough = self._sector_seen_for >= SECTOR_CONFIRM_S
            if dwell_done and seen_enough:
                self.scan_log.append({
                    "phase": self.state, "sector": sector,
                    "start_s": self.state_since, "end_s": t,
                    "duration_s": elapsed, "sector_confirmed": True,
                })
                index = [p[0] for p in LOOK_PHASES].index(self.state)
                nxt = (LOOK_PHASES[index + 1][0]
                       if index + 1 < len(LOOK_PHASES) else "WAIT_FOR_GAP")
                self._advance(t, nxt, trunk_x=trunk_x, sector=sector)
            elif elapsed >= LOOK_MAX_S:
                # The sector gate could not be satisfied.  Record it and move
                # on so the gate FAILS with evidence instead of hanging.
                self.scan_log.append({
                    "phase": self.state, "sector": sector,
                    "start_s": self.state_since, "end_s": t,
                    "duration_s": elapsed, "sector_confirmed": False,
                })
                self.timeouts.append(f"{self.state}_sector_timeout")
                index = [p[0] for p in LOOK_PHASES].index(self.state)
                nxt = (LOOK_PHASES[index + 1][0]
                       if index + 1 < len(LOOK_PHASES) else "WAIT_FOR_GAP")
                self._advance(t, nxt, trunk_x=trunk_x, sector=sector,
                              reason="sector timeout")
        elif self.state == "WAIT_FOR_GAP":
            if decision is not None:
                if decision.safe:
                    self._gap_safe_for += self.dt
                else:
                    self._gap_safe_for = 0.0
                    limiting = decision.limiting_vehicle
                    if limiting is not None:
                        record = self._rejected.get(limiting)
                        if record is None:
                            entry = decision.as_record()
                            entry["first_rejected_at_s"] = t
                            entry["last_rejected_at_s"] = t
                            entry["ticks"] = 1
                            self._rejected[limiting] = entry
                            self.gap_decisions.append(entry)
                        else:
                            record["last_rejected_at_s"] = t
                            record["ticks"] += 1
                            if decision.worst_margin_s < record["worst_margin_s"]:
                                record["worst_margin_s"] = decision.worst_margin_s
                                record["blocking"] = decision.as_record()["blocking"]
                if self._gap_safe_for >= GAP_CONFIRM_S:
                    accepted = decision.as_record()
                    accepted["accepted_at_s"] = t
                    accepted["wait_duration_s"] = elapsed
                    self.gap_decisions.append(accepted)
                    self.commit = {
                        "committed_at_s": t,
                        "wait_duration_s": elapsed,
                        "start_x": trunk_x,
                        "worst_margin_s": decision.worst_margin_s,
                        "limiting_vehicle": decision.limiting_vehicle,
                        "crossing_duration_estimate_s":
                            decision.crossing_duration_s,
                        "rejected_gaps": len(self._rejected),
                    }
                    self._advance(t, "CROSSING", trunk_x=trunk_x,
                                  margin_s=decision.worst_margin_s)
            if self.state == "WAIT_FOR_GAP" and elapsed >= WAIT_MAX_S:
                self.timeouts.append("wait_timeout")
                self._advance(t, "CROSSING", trunk_x=trunk_x,
                              reason="wait timeout")
        elif self.state == "CROSSING":
            # NEVER re-decided.  The commitment already covered the whole
            # crossing, and stopping in a live lane is the worst possible
            # response to a surprise.
            #
            # The finish line is the GOAL, not the safe zone's near edge.
            # Ending the crossing the instant the trunk centre clips the edge
            # (x=0.80) would leave the duck's trailing surface at 0.67 m, only
            # 0.12 m clear of the road, and would cut the measured crossing
            # path short of the evidence threshold the gate requires.
            if trunk_x >= CROSS_GOAL_X and in_safe_zone(trunk_x):
                self.commit["arrived_at_s"] = t
                self.commit["crossing_duration_s"] = elapsed
                self.commit["end_x"] = trunk_x
                self._advance(t, "SAFE", trunk_x=trunk_x)
            elif elapsed >= CROSSING_MAX_S:
                self.timeouts.append("crossing_timeout")
                self.commit["crossing_duration_s"] = elapsed
                self.commit["end_x"] = trunk_x
                self._advance(t, "SAFE", trunk_x=trunk_x,
                              reason="crossing timeout")
        elif self.state == "SAFE":
            pass

        changed = self.state != previous
        return self.state, changed


@dataclass
class GuardianController:
    """Produce ``(vx, vy, wz)`` from the state, the trunk pose and the goal.

    Two things are enforced here rather than hoped for:

    * **Zero means exactly zero.**  Every stationary state returns
      ``(0.0, 0.0, 0.0)`` with no filter tail, because the acceptance gate
      tests for exact zero and a decaying tail would fail it — correctly, since
      a decaying command is still a command.
    * **No sub-onset commands, ever.**  The measured gait onset means a ``vx``
      between zero and ``VX_MIN_EFFECTIVE`` looks like motion in the HUD and
      produces none on the floor.  The controller emits either a walking
      command or exactly zero.

    The crossing is flown CLOSED-LOOP ON HEADING because the policy drifts
    measurably right at speed (−17.7 deg over 5 s at ``vx=0.58``).  Left and
    right corrections use independent gains and dead zones: the measured yaw
    authority at this speed is +2.4 deg/s at ``wz=+0.10`` against −8.7 deg/s at
    ``wz=−0.10``, so mirroring one gain onto the other side would make every
    left correction a violent over-correction.
    """

    ctrl_hz: float = 50.0
    command: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))

    def reset(self) -> None:
        self.command[:] = 0.0

    def raw_command(self, state: str, trunk_x: float, trunk_yaw: float,
                    trunk_y: float = 0.0) -> tuple[float, float, float]:
        """Unfiltered target command for this state and trunk pose."""
        if state == "APPROACH_CURB":
            if trunk_x >= APPROACH_RELEASE_X:
                return (0.0, 0.0, 0.0)
            return (VX_APPROACH, 0.0, self._heading_hold(trunk_yaw, trunk_y))
        if state == "CROSSING":
            if trunk_x >= CROSS_GOAL_X:
                return (0.0, 0.0, 0.0)
            return (VX_CROSS, 0.0, self._heading_hold(trunk_yaw, trunk_y))
        return (0.0, 0.0, 0.0)

    def _heading_hold(self, trunk_yaw: float, trunk_y: float) -> float:
        """Yaw command that holds the duck pointing along +x, down the zebra.

        The setpoint is not simply "yaw = 0": a duck that has drifted off the
        zebra centreline must aim slightly BACK toward it, so the target
        heading carries a cross-track term.  The term is clamped hard, because
        a large cross-track correction inside a traffic lane would be worse
        than the lateral error it fixes.
        """
        cross_track = clamp(-1.10 * trunk_y, -math.radians(20.0),
                            math.radians(20.0))
        error = wrap_angle(cross_track - trunk_yaw)
        if error >= 0.0:
            wz = clamp(KP_YAW_LEFT * error, 0.0, WZ_MAX_LEFT)
            return 0.0 if wz < WZ_MIN_LEFT else wz
        wz = -clamp(KP_YAW_RIGHT * abs(error), 0.0, WZ_MAX_RIGHT)
        return 0.0 if abs(wz) < WZ_MIN_RIGHT else wz

    def update(self, state: str, trunk_x: float, trunk_yaw: float,
               trunk_y: float = 0.0) -> np.ndarray:
        target = self.raw_command(state, trunk_x, trunk_yaw, trunk_y)
        # Applied directly.  A low-pass filter here would spend the first ticks
        # BELOW the measured gait onset, which is not a gentle start — it is no
        # motion at all followed by a jump.
        self.command[:] = np.asarray(target, dtype=np.float32)
        return self.command.copy()
