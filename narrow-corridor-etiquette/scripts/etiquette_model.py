#!/usr/bin/env python3
"""The etiquette state machine and the corridor locomotion controllers.

Pure Python/numpy: no MuJoCo, no ONNX, no rendering.  Fully unit-tested.

    CRUISE → DETECT → SELECT_ALCOVE → PULL_OVER → YIELD
           → CLEAR → REJOIN → RESUME → DONE

and ``RESUME`` re-enters ``DETECT`` when a second adult appears, so the cycle
is a loop rather than a one-shot script.

Four properties are structural rather than hoped for:

* **The duck keeps walking while it decides.**  ``DETECT`` and
  ``SELECT_ALCOVE`` are *moving* states.  Stopping dead in the middle of a
  corridor the moment a person appears is not etiquette, it is an obstruction —
  and it would also make the reachability arithmetic trivial, because a
  stationary duck's alcove options never change.
* **YIELD is exactly still.**  Not "slow", not "a decaying command": the
  controller returns hard zeros, because a decaying tail is still a command and
  the acceptance gate tests for exact zero.
* **The yield cannot be released early.**  ``CLEAR`` is reachable only once the
  adult is genuinely past the duck, opening the range, and beyond a clearance
  distance justified by the duck's own measured rejoin time.  A duck that
  stepped back out while the adult was level with it would have pulled over for
  nothing.
* **A pull-over is never re-decided mid-manoeuvre.**  Changing target halfway
  would leave the duck in the middle of the passage at the worst moment.  The
  selection is taken once, with a margin that already covers the whole
  manoeuvre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from corridor import (
    ALCOVE_BY_NAME,
    DUCK_PLANAR_RADIUS,
    REJOIN_TOLERANCE_M,
    at_destination,
    clears_center_passage,
)
from encounter import (
    CLEAR_HOLD_S,
    CLEAR_RANGE_M,
    CRUISE_MAX_S,
    DETECT_HOLD_S,
    KP_YAW_LEFT,
    KP_YAW_RIGHT,
    PULL_OVER_MAX_S,
    RECEDING_CONFIRM_S,
    REDETECT_COOLDOWN_S,
    REJOIN_MAX_S,
    RESUME_MAX_S,
    SELECT_HOLD_S,
    VX_APPROACH,
    VX_CRUISE,
    VY_PULLOVER_LEFT,
    VY_PULLOVER_RIGHT,
    WZ_FEEDFORWARD_LEFT,
    WZ_FEEDFORWARD_RIGHT,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
    YIELD_MAX_S,
    YIELD_MIN_S,
    choose_alcove,
    clamp,
    wrap_angle,
)

# Beyond this distance from the target station the pull-over walks at CRUISE
# speed rather than the slow approach speed: the duck covers the corridor at
# its normal pace and only slows for the last stretch into the mouth.
FAR_APPROACH_M: float = 0.45
# How close to its park position the duck must be for the pull-over to count as
# complete.  TWO separate conditions, both required: its whole footprint out of
# the centre passage, AND within this of the park point — so "arrived" is not
# satisfied by merely clipping the passage edge.
PARK_TOLERANCE_M: float = 0.055
# The same, along the corridor axis.  Looser than the lateral tolerance because
# the mouth is far longer than the footprint, but bounded so a pull-over cannot
# be declared complete while the duck is still near a cheek.
PARK_X_TOLERANCE_M: float = 0.10
# The duck must be measurably stationary before YIELD begins, so the yield's
# exact-zero gate is not satisfied while the trunk is still coasting.
PARK_SPEED_TOLERANCE_MPS: float = 0.05


@dataclass
class EtiquetteMachine:
    """The nine-state corridor-etiquette machine.

    The machine owns WHICH phase is active and WHY it advanced.  It does not
    own geometry: the duck's pose, the predicted encounter and the alcove
    decision are all supplied by the caller.  That split is what lets the whole
    machine be tested without MuJoCo.
    """

    ctrl_hz: float = 50.0
    state: str = "CRUISE"
    state_since: float = 0.0
    events: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    cycles: list[dict] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    target: str | None = None
    target_side: int = 0
    target_x: float | None = None
    park_y: float = 0.0
    yielding_to: str | None = None
    no_alcove_events: list[dict] = field(default_factory=list)
    _pending: object | None = None
    _receding_for: float = 0.0
    _redetect_after: float = 0.0
    _cycle: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_hz

    @property
    def completed_cycles(self) -> int:
        return len(self.cycles)

    @property
    def in_manoeuvre(self) -> bool:
        return self.state in ("PULL_OVER", "YIELD", "CLEAR", "REJOIN")

    def _advance(self, t: float, state: str, **event) -> None:
        self.events.append({"t": t, "from": self.state, "to": state, **event})
        self.state = state
        self.state_since = t
        self._receding_for = 0.0

    def update(
        self,
        t: float,
        *,
        duck_xy,
        duck_speed_mps: float = 0.0,
        encounter=None,
        person_range_m: float | None = None,
        person_receding: bool | None = None,
        person_behind: bool | None = None,
    ) -> tuple[str, bool]:
        """Advance one tick. Returns ``(state, changed)``.

        ``encounter`` is the caller's most-urgent :class:`encounter.Encounter`,
        or ``None``.  The range/receding/behind arguments describe the adult
        currently being yielded to and are consulted only from ``YIELD`` on.
        """
        duck_xy = np.asarray(duck_xy, dtype=np.float64)
        duck_x, duck_y = float(duck_xy[0]), float(duck_xy[1])
        elapsed = t - self.state_since
        previous = self.state

        if self.state in ("CRUISE", "RESUME"):
            if at_destination(duck_x):
                self._advance(t, "DONE", trunk_x=duck_x,
                              reason="reached destination")
            elif encounter is not None and t >= self._redetect_after:
                # Record the counterfactual AT THE MOMENT OF DETECTION, before
                # the duck has done anything about it.  Reconstructing it later
                # from a duck already tucked into an alcove would report the
                # clearance of the manoeuvre, not the clearance of doing
                # nothing — which is the only thing that justifies acting.
                self._cycle = {
                    "index": len(self.cycles) + 1,
                    "person": encounter.name,
                    "detected_at_s": t,
                    "detected_x": duck_x,
                    "detected_y": duck_y,
                    "detect_range_m": encounter.range_m,
                    "detect_time_to_meet_s": encounter.time_to_meet_s,
                    "counterfactual_clearance_m":
                        encounter.counterfactual_clearance_m,
                    "head_on": encounter.head_on,
                    "adult_direction": encounter.adult_direction,
                    "predicted_meet_x": encounter.meet_x,
                }
                self._pending = encounter
                self._advance(t, "DETECT", trunk_x=duck_x,
                              person=encounter.name,
                              time_to_meet_s=encounter.time_to_meet_s)
            elif (self.state == "CRUISE" and elapsed >= CRUISE_MAX_S) or (
                    self.state == "RESUME" and elapsed >= RESUME_MAX_S):
                self.timeouts.append(f"{self.state.lower()}_timeout")
                self._advance(t, "DONE", trunk_x=duck_x, reason="timeout")

        elif self.state == "DETECT":
            if elapsed >= DETECT_HOLD_S:
                # The selection is taken on the caller's FRESHEST encounter if
                # one is still live, so a duck that kept walking through the
                # dwell chooses against where it is now rather than against a
                # stale prediction it never rechecked.
                live = encounter if encounter is not None else self._pending
                decision = choose_alcove(live, (duck_x, duck_y))
                record = decision.as_record()
                record["decided_at_s"] = t
                self.decisions.append(record)
                self._cycle["decision"] = record
                if decision.selected is None:
                    # Nowhere to go yet.  Recorded, and the machine waits a
                    # measured cooldown before re-opening the question: the
                    # geometry cannot change inside one dwell, so re-deciding
                    # immediately only produces a CRUISE/DETECT oscillation.
                    # Walking on for a couple of seconds brings new bays into
                    # reach, which is the only thing that CAN change the answer.
                    self.no_alcove_events.append(
                        {"t": t, "trunk_x": duck_x,
                         "person": getattr(live, "name", None)})
                    self._redetect_after = t + REDETECT_COOLDOWN_S
                    self._cycle = {}
                    self._advance(t, "CRUISE", trunk_x=duck_x,
                                  reason="no viable alcove")
                else:
                    self.target = decision.selected.name
                    self.target_side = decision.selected.side
                    self.target_x = ALCOVE_BY_NAME[self.target].park_x
                    self.park_y = decision.selected.park_y
                    self._pending = live
                    self._cycle.update({
                        "selected_alcove": self.target,
                        "selected_park_y": self.park_y,
                        "selected_margin_s": decision.selected.time_margin_s,
                        "alcoves_considered": decision.considered,
                        "alcoves_rejected": [r.name for r in decision.rejected],
                        "alcoves_viable": [c.name for c in decision.viable],
                    })
                    self._advance(t, "SELECT_ALCOVE", trunk_x=duck_x,
                                  alcove=self.target,
                                  margin_s=decision.selected.time_margin_s)

        elif self.state == "SELECT_ALCOVE":
            if elapsed >= SELECT_HOLD_S:
                self._cycle["pull_over_started_s"] = t
                self._cycle["pull_over_start_xy"] = [duck_x, duck_y]
                self._advance(t, "PULL_OVER", trunk_x=duck_x,
                              alcove=self.target)

        elif self.state == "PULL_OVER":
            # NEVER re-decided.  Changing target halfway leaves the duck in the
            # middle of the passage at the worst possible moment.
            #
            # "Parked" requires THREE independent things, because any two of
            # them can be true while the duck is in a bad place: its whole
            # footprint between the recess's cheeks, its whole footprint out of
            # the centre passage, and the trunk near the park point AND
            # stationary.  The first draft omitted the cheek test, and the duck
            # finished a pull-over pressed into the mouth's low cheek with
            # -0.109 m of measured wall overlap — it had reached its park y
            # before it had walked far enough along the corridor to be inside
            # the recess at all.
            alcove = ALCOVE_BY_NAME.get(self.target) if self.target else None
            inside = (alcove is not None
                      and alcove.footprint_inside(duck_x, duck_y))
            parked = (
                inside
                and clears_center_passage(duck_y)
                and abs(duck_y - self.park_y) <= PARK_TOLERANCE_M
                and (alcove is None
                     or abs(duck_x - alcove.park_x) <= PARK_X_TOLERANCE_M)
            )
            if parked and duck_speed_mps <= PARK_SPEED_TOLERANCE_MPS:
                self.yielding_to = self._cycle.get("person")
                self._cycle["yield_started_s"] = t
                self._cycle["park_xy"] = [duck_x, duck_y]
                self._cycle["pull_over_duration_s"] = (
                    t - self._cycle.get("pull_over_started_s", t))
                self._advance(t, "YIELD", trunk_x=duck_x, trunk_y=duck_y)
            elif elapsed >= PULL_OVER_MAX_S:
                self.timeouts.append("pull_over_timeout")
                self.yielding_to = self._cycle.get("person")
                self._cycle["yield_started_s"] = t
                self._cycle["park_xy"] = [duck_x, duck_y]
                self._cycle["pull_over_duration_s"] = elapsed
                self._advance(t, "YIELD", trunk_x=duck_x,
                              reason="pull-over timeout")

        elif self.state == "YIELD":
            # The adult must be PAST the duck and OPENING the range before the
            # yield may end.  Either condition alone releases too early:
            # "past" is true the instant the bodies are level, and "opening"
            # can flicker while they are alongside each other.
            if person_receding:
                self._receding_for += self.dt
            else:
                self._receding_for = 0.0
            cleared = (
                elapsed >= YIELD_MIN_S
                and bool(person_behind)
                and self._receding_for >= RECEDING_CONFIRM_S
                and person_range_m is not None
                and person_range_m >= CLEAR_RANGE_M
            )
            if cleared:
                self._cycle["yield_duration_s"] = elapsed
                self._cycle["cleared_at_s"] = t
                self._cycle["clear_range_m"] = person_range_m
                self._advance(t, "CLEAR", trunk_x=duck_x,
                              range_m=person_range_m)
            elif elapsed >= YIELD_MAX_S:
                self.timeouts.append("yield_timeout")
                self._cycle["yield_duration_s"] = elapsed
                self._advance(t, "CLEAR", trunk_x=duck_x,
                              reason="yield timeout")

        elif self.state == "CLEAR":
            if elapsed >= CLEAR_HOLD_S:
                self._cycle["rejoin_started_s"] = t
                self._advance(t, "REJOIN", trunk_x=duck_x)

        elif self.state == "REJOIN":
            if abs(duck_y) <= REJOIN_TOLERANCE_M:
                self._cycle["rejoin_duration_s"] = elapsed
                self._cycle["rejoined_at_s"] = t
                self._cycle["rejoin_xy"] = [duck_x, duck_y]
                self._finish_cycle(t)
                self._advance(t, "RESUME", trunk_x=duck_x, trunk_y=duck_y)
            elif elapsed >= REJOIN_MAX_S:
                self.timeouts.append("rejoin_timeout")
                self._cycle["rejoin_duration_s"] = elapsed
                self._cycle["rejoin_xy"] = [duck_x, duck_y]
                self._finish_cycle(t)
                self._advance(t, "RESUME", trunk_x=duck_x,
                              reason="rejoin timeout")

        elif self.state == "DONE":
            pass

        changed = self.state != previous
        return self.state, changed

    def _finish_cycle(self, t: float) -> None:
        entry = dict(self._cycle)
        entry["completed_at_s"] = t
        self.cycles.append(entry)
        self._cycle = {}
        self.target = None
        self.target_side = 0
        self.target_x = None
        self.yielding_to = None
        self._pending = None


@dataclass
class EtiquetteController:
    """Produce ``(vx, vy, wz)`` from the state, the trunk pose and the target.

    Three things are enforced here rather than hoped for:

    * **Zero means exactly zero.**  Every stationary state returns
      ``(0.0, 0.0, 0.0)`` with no filter tail, because the acceptance gate
      tests for exact zero and a decaying command is still a command.
    * **No sub-onset commands, ever.**  The measured gait onsets mean a command
      between zero and onset looks like motion in the HUD and produces none on
      the floor.  The controller emits either a walking command or exact zero.
    * **The lateral step carries its own measured yaw feed-forward.**  A
      right-hand step with no yaw command was MEASURED spinning the duck
      +93.6° in four seconds; a left-hand step barely rotates it at all.  The
      two signs therefore get completely independent feed-forward terms, and
      the heading loop closes on top of that.

    Heading is held closed-loop in every moving state, because the policy's
    open-loop drift at cruise was MEASURED at −17.4° over 6 s and a corridor
    only 0.42 m wide cannot absorb it.
    """

    ctrl_hz: float = 50.0
    command: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))

    def reset(self) -> None:
        self.command[:] = 0.0

    def raw_command(self, state: str, trunk_x: float, trunk_y: float,
                    trunk_yaw: float, *, park_y: float = 0.0,
                    target_x: float | None = None,
                    alcove_name: str | None = None
                    ) -> tuple[float, float, float]:
        """Unfiltered target command for this state and trunk pose."""
        if state in ("CRUISE", "DETECT", "SELECT_ALCOVE", "RESUME"):
            if at_destination(trunk_x):
                return (0.0, 0.0, 0.0)
            return (VX_CRUISE, 0.0, self._heading_hold(trunk_yaw, trunk_y))

        if state == "PULL_OVER":
            return self._pull_over(trunk_x, trunk_y, trunk_yaw, park_y,
                                   target_x, alcove_name)

        if state == "REJOIN":
            return self._rejoin(trunk_y, trunk_yaw)

        # YIELD, CLEAR, DONE
        return (0.0, 0.0, 0.0)

    # -- primitives ------------------------------------------------------
    def _pull_over(self, trunk_x: float, trunk_y: float, trunk_yaw: float,
                   park_y: float, target_x: float | None,
                   alcove_name: str | None = None
                   ) -> tuple[float, float, float]:
        """Walk to the recess, then step into it, holding the corridor axis.

        THE LATERAL STEP MAY NOT BEGIN BEFORE THE DUCK IS BETWEEN THE CHEEKS.
        Stepping sideways while still short of the mouth drives the robot into
        the wall beside it — measured at -0.109 m of overlap when the first
        draft omitted this test.  So the duck holds the centreline until its
        whole footprint is inside the mouth's x-span, and only then steps.

        The two legs still overlap: the mouth is far longer than the footprint,
        so the sideways move starts well before the duck reaches the park
        station and the forward loop keeps running underneath it.  That is the
        manoeuvre ``tools/measure_pullover.py`` timed and the reachability
        estimate is built on.
        """
        alcove = ALCOVE_BY_NAME.get(alcove_name) if alcove_name else None
        may_step = True
        if alcove is not None:
            low, high = alcove.x_span
            may_step = (trunk_x - DUCK_PLANAR_RADIUS >= low
                        and trunk_x + DUCK_PLANAR_RADIUS <= high)

        error_y = park_y - trunk_y
        if not may_step or abs(error_y) <= 0.5 * PARK_TOLERANCE_M:
            vy = 0.0
        elif error_y > 0.0:
            vy = VY_PULLOVER_LEFT
        else:
            vy = VY_PULLOVER_RIGHT

        # The MEASURED yaw coupling of this exact lateral command, fed forward.
        if vy > 0.0:
            wz_ff = WZ_FEEDFORWARD_LEFT
        elif vy < 0.0:
            wz_ff = WZ_FEEDFORWARD_RIGHT
        else:
            wz_ff = 0.0

        vx = 0.0
        if target_x is not None:
            error_x = target_x - trunk_x
            # Walk to the alcove at CRUISE speed and only slow for the last
            # stretch.  Approaching a mouth two metres away at the slow
            # approach speed would be a different manoeuvre from the one the
            # duck actually performs, and the reachability estimate is built on
            # this exact schedule.
            if error_x > FAR_APPROACH_M:
                vx = VX_CRUISE
            elif error_x > 0.06:
                vx = VX_APPROACH
            elif not may_step:
                # Still short of the mouth but inside the slow band: keep
                # closing, or the duck stalls forever just before its target.
                vx = VX_APPROACH

        # While the duck is still walking up to the mouth it is ON the
        # centreline, so the heading loop keeps its cross-track term; once the
        # step begins the duck is deliberately off-centre and the term is
        # dropped so it cannot fight the manoeuvre.
        wz = wz_ff + self._heading_hold(
            trunk_yaw, trunk_y, cross_track=(vy == 0.0 and not may_step))
        return (vx, vy, float(clamp(wz, -0.62, 0.35)))

    def _rejoin(self, trunk_y: float, trunk_yaw: float
                ) -> tuple[float, float, float]:
        """Step back out to the corridor centreline, without walking forward.

        Rejoining is a pure lateral move for the same reason the pull-over is:
        the duck cannot turn in place, so any heading change would have to be
        made while walking, and a curve out of an alcove mouth ends against the
        opposite wall of a 0.42 m corridor.
        """
        error_y = -trunk_y
        if abs(error_y) <= 0.5 * REJOIN_TOLERANCE_M:
            return (0.0, 0.0, 0.0)
        if error_y > 0.0:
            vy, wz_ff = VY_PULLOVER_LEFT, WZ_FEEDFORWARD_LEFT
        else:
            vy, wz_ff = VY_PULLOVER_RIGHT, WZ_FEEDFORWARD_RIGHT
        wz = wz_ff + self._heading_hold(trunk_yaw, 0.0, cross_track=False)
        return (0.0, vy, float(clamp(wz, -0.62, 0.35)))

    def _heading_hold(self, trunk_yaw: float, trunk_y: float,
                      cross_track: bool = True) -> float:
        """Yaw command that holds the duck pointing along +x, down the corridor.

        The setpoint is not simply "yaw = 0": a duck that has drifted off the
        centreline must aim slightly back toward it, so the target heading
        carries a cross-track term.  That term is clamped hard, because a large
        cross-track correction inside a 0.42 m corridor swings the nose into a
        wall faster than it fixes the offset.

        During a lateral manoeuvre the cross-track term is disabled entirely:
        the duck is *deliberately* off the centreline there, and a loop that
        tried to correct it would fight the step it is making.
        """
        setpoint = 0.0
        if cross_track:
            setpoint = clamp(-0.90 * trunk_y, -math.radians(14.0),
                             math.radians(14.0))
        error = wrap_angle(setpoint - trunk_yaw)
        if error >= 0.0:
            wz = clamp(KP_YAW_LEFT * error, 0.0, WZ_MAX_LEFT)
            return 0.0 if wz < WZ_MIN_LEFT else wz
        wz = -clamp(KP_YAW_RIGHT * abs(error), 0.0, WZ_MAX_RIGHT)
        return 0.0 if abs(wz) < WZ_MIN_RIGHT else wz

    def update(self, state: str, trunk_x: float, trunk_y: float,
               trunk_yaw: float, *, park_y: float = 0.0,
               target_x: float | None = None,
               alcove_name: str | None = None) -> np.ndarray:
        target = self.raw_command(state, trunk_x, trunk_y, trunk_yaw,
                                  park_y=park_y, target_x=target_x,
                                  alcove_name=alcove_name)
        # Applied directly.  A low-pass filter here would spend its first ticks
        # BELOW the measured gait onsets, which is not a gentle start — it is
        # no motion at all followed by a jump.
        self.command[:] = np.asarray(target, dtype=np.float32)
        return self.command.copy()
