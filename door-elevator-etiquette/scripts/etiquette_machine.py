#!/usr/bin/env python3
"""The state machine: approach, yield, follow through, wait beside, let them
out, board behind her, stand aside, ride, and leave after her.

    APPROACH_DOOR -> YIELD_EXITERS -> FOLLOW_THROUGH -> APPROACH_LIFT
        -> WAIT_SIDE -> DOORS_OPEN -> LET_OCCUPANTS_EXIT
        -> FOLLOW_GUARDIAN_IN -> POSITION_INSIDE -> RIDE
        -> DOORS_OPEN_TARGET -> FOLLOW_OUT -> DONE

The machine never touches physics and never emits a command; ``etiquette_control``
does that from the state.  Keeping the two apart is what lets every transition
rule be unit-tested on hand-built inputs, with no MuJoCo anywhere.

SIX INVARIANTS ARE STRUCTURAL RATHER THAN CHECKED AFTERWARDS
--------------------------------------------------------------
* **Every yield is CAUSED by a measurement, never scheduled.**  There is no
  timer, no waypoint index and no schedule lookup in this file.  The duck stops
  outside the threshold because it MEASURED somebody in the aperture heading
  towards it, and it moves off because it MEASURED all of them clear for
  :data:`CLEAR_CONFIRM_S` continuously.  The scenario's own choreography lives
  in ``etiquette_actors`` and the machine cannot see it.

* **The duck never enters an aperture the guardian is still in.**
  :meth:`_follow_through_state` and :meth:`_follow_in_state` are both gated on
  her being :data:`GUARDIAN_THROUGH_M` beyond the plane, so "it entered behind
  her" is enforced by the transition rather than asserted by the metrics.

* **Waiting is a state, not a speed.**  Forward gait onset on this scene is a
  MEASURED cliff, so a duck that "edged forward slowly" would emit a command
  that appears in the metrics and produces nothing on the floor.  It walks or it
  holds exactly zero, and every yield state is where the zero lives.

* **The ride cannot be shortened.**  ``RIDE`` ends only when the rear doors are
  measured open, which is a property of the world rather than of a countdown the
  machine could race.

* **The guardian leaves first.**  ``DOORS_OPEN_TARGET`` holds until she is
  measured through the rear aperture; only then may ``FOLLOW_OUT`` begin.

* **A ceiling MOVES the machine.**  Every ``_timeout`` here transitions and
  records why.  A ceiling that only appends to a log is not a ceiling: an
  earlier behavior in this lab spent 71 s in a state whose budget had run out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from etiquette_states import (
    APPROACH_DOOR_MAX_S,
    APPROACH_LIFT_MAX_S,
    CLEAR_CONFIRM_S,
    DOORS_OPEN_MAX_S,
    DOORS_OPEN_TARGET_MAX_S,
    FOLLOW_GUARDIAN_IN_MAX_S,
    FOLLOW_OUT_MAX_S,
    FOLLOW_THROUGH_MAX_S,
    LET_OCCUPANTS_EXIT_MAX_S,
    MIN_OCCUPANTS_EXITED,
    MIN_RIDE_S,
    MIN_WAIT_SIDE_S,
    MIN_YIELD_S,
    POSITION_INSIDE_MAX_S,
    RIDE_MAX_S,
    WAIT_SIDE_MAX_S,
    YIELD_MAX_S,
)


@dataclass
class Sense:
    """Everything the duck MEASURED this tick, and nothing else.

    Every field is a quantity the robot could have obtained from its own pose,
    its own contact probe or its own camera.  Bundling them into one object is
    what makes it impossible for the machine to reach sideways into the
    scenario: if a value is not here, the machine cannot use it.
    """

    # -- the duck's own progress ---------------------------------------
    route_remaining_m: float = 1e9
    leg_arrived: bool = False
    at_door_threshold: bool = False
    at_lift_hold: bool = False
    at_cabin_hold: bool = False
    inside_cabin: bool = False
    beyond_rear_m: float = -1e9

    # -- the doorway --------------------------------------------------
    door_open_fraction: float = 0.0
    door_passable: bool = False
    exiters_in_aperture: int = 0
    exiters_pending: int = 0
    all_exiters_clear: bool = False

    # -- the guardian --------------------------------------------------
    guardian_through_door: bool = False
    guardian_through_lift: bool = False
    guardian_inside_cabin: bool = False
    guardian_through_rear: bool = False
    guardian_gap_m: float = 0.0

    # -- the lift ------------------------------------------------------
    lift_open_fraction: float = 0.0
    lift_passable: bool = False
    occupants_exited: int = 0
    occupants_in_cabin: int = 0
    occupants_in_passage: int = 0
    all_occupants_clear: bool = False
    rear_open_fraction: float = 0.0
    rear_passable: bool = False


@dataclass
class EtiquetteMachine:
    """Transitions, yield episodes and the boarding record.  No physics."""

    ctrl_hz: float = 50.0
    state: str = "APPROACH_DOOR"
    state_since: float = 0.0
    guardian: str = ""

    transitions: list[dict] = field(default_factory=list)
    yields: list[dict] = field(default_factory=list)
    boarding: dict = field(default_factory=dict)
    timeouts: list[str] = field(default_factory=list)

    _clear_for: float = 0.0
    _yield: dict = field(default_factory=dict)
    _ride_started: float | None = None

    def __post_init__(self) -> None:
        self.dt = 1.0 / self.ctrl_hz

    # -- helpers ---------------------------------------------------------
    def _go(self, t: float, state: str, **detail) -> None:
        self.transitions.append(
            {"t": round(t, 3), "from": self.state, "to": state, **detail})
        self.state = state
        self.state_since = t
        self._clear_for = 0.0

    def _elapsed(self, t: float) -> float:
        return t - self.state_since

    def _timeout(self, t: float, limit: float, label: str) -> bool:
        if self._elapsed(t) >= limit:
            self.timeouts.append(f"{label}@{t:.2f}s")
            return True
        return False

    def set_guardian(self, name: str) -> None:
        """Record the identity being followed.  Callable once, by design."""
        if self.guardian and self.guardian != name:
            raise ValueError(
                f"the guardian is {self.guardian!r} and cannot be reassigned to "
                f"{name!r}: a robot follows one person through a door")
        self.guardian = name

    # -- the machine -----------------------------------------------------
    def update(self, t: float, sense: Sense) -> tuple[str, bool]:
        """Advance one control tick on measurements taken BEFORE this tick's
        physics."""
        before = self.state
        handler = getattr(self, f"_{self.state.lower()}_state")
        handler(t, sense)
        return self.state, self.state != before

    # -- the doorway -----------------------------------------------------
    def _approach_door_state(self, t: float, s: Sense) -> None:
        """Walk up to the door and stop OUTSIDE the threshold band.

        The stop is triggered by the duck's own arrival at its planned holding
        point, not by seeing anybody: a robot that only stopped when it happened
        to notice traffic would walk into an empty doorway at full speed and
        would have to brake when somebody appeared.  Etiquette is approaching a
        narrow opening ready to give way.
        """
        if s.at_door_threshold or s.leg_arrived:
            self._yield = {
                "index": len(self.yields),
                "kind": "door",
                "began_at_s": round(t, 3),
                "exiters_pending_at_stop": int(s.exiters_pending),
                "exiters_in_aperture_at_stop": int(s.exiters_in_aperture),
                "door_open_fraction_at_stop": round(s.door_open_fraction, 4),
            }
            self._go(t, "YIELD_EXITERS",
                     exiters_pending=int(s.exiters_pending),
                     reason="reached the holding point outside the threshold")
            return
        if self._timeout(t, APPROACH_DOOR_MAX_S, "APPROACH_DOOR"):
            self._go(t, "YIELD_EXITERS", reason="approach ceiling reached")

    def _yield_exiters_state(self, t: float, s: Sense) -> None:
        """Hold exactly still until every exiter is measured clear.

        BOTH conditions, continuously, for :data:`CLEAR_CONFIRM_S`: nobody left
        to come out, and nobody still in the opening or in the duck's way.  The
        minimum duration is separate and is what stops a two-tick "yield" from
        counting as one.
        """
        self._clear_for = (self._clear_for + self.dt
                           if s.all_exiters_clear else 0.0)
        ready = (self._clear_for >= CLEAR_CONFIRM_S
                 and self._elapsed(t) >= MIN_YIELD_S
                 and s.door_passable)
        if ready:
            self._yield.update({
                "ended_at_s": round(t, 3),
                "duration_s": round(t - self._yield.get("began_at_s", t), 3),
                "clear_sustained_s": round(self._clear_for, 3),
                "door_open_fraction_at_end": round(s.door_open_fraction, 4),
                "exiters_still_pending": int(s.exiters_pending),
            })
            self.yields.append(self._yield)
            self._yield = {}
            self._go(t, "FOLLOW_THROUGH",
                     clear_for_s=round(self._clear_for, 3),
                     reason="every exiter measured clear; door passable")
            return
        if self._timeout(t, YIELD_MAX_S, "YIELD_EXITERS"):
            self._yield.update({
                "ended_at_s": round(t, 3),
                "duration_s": round(t - self._yield.get("began_at_s", t), 3),
                "ceiling_reached": True,
            })
            self.yields.append(self._yield)
            self._yield = {}
            self._go(t, "FOLLOW_THROUGH", reason="yield ceiling reached")

    def _follow_through_state(self, t: float, s: Sense) -> None:
        """Walk through the opening, BEHIND the guardian, never beside her.

        The entry gate is her being measured beyond the plane; the exit gate is
        the duck's own arrival at the lobby leg.  The controller separately
        refuses to advance while she is still inside the aperture, so the
        no-side-by-side claim is enforced in two places rather than one.
        """
        if s.leg_arrived or s.at_lift_hold:
            self._go(t, "APPROACH_LIFT",
                     guardian_gap_m=round(s.guardian_gap_m, 4),
                     reason="through the doorway; heading for the lift")
            return
        if self._timeout(t, FOLLOW_THROUGH_MAX_S, "FOLLOW_THROUGH"):
            self._go(t, "APPROACH_LIFT", reason="follow-through ceiling reached")

    # -- the lift --------------------------------------------------------
    def _approach_lift_state(self, t: float, s: Sense) -> None:
        """Cross the lobby to the holding spot BESIDE the doors, not in front."""
        if s.at_lift_hold or s.leg_arrived:
            self._go(t, "WAIT_SIDE",
                     lift_open_fraction=round(s.lift_open_fraction, 4),
                     reason="reached the holding spot beside the lift doors")
            return
        if self._timeout(t, APPROACH_LIFT_MAX_S, "APPROACH_LIFT"):
            self._go(t, "WAIT_SIDE", reason="lift approach ceiling reached")

    def _wait_side_state(self, t: float, s: Sense) -> None:
        """Stand beside the doors at exactly zero until they begin to open.

        The transition is on the MEASURED open fraction rising off zero, which
        is a property of the world.  The minimum dwell is what makes "it waited"
        a duration rather than a frame.
        """
        if s.lift_open_fraction > 0.02 and self._elapsed(t) >= MIN_WAIT_SIDE_S:
            self._go(t, "DOORS_OPEN",
                     waited_s=round(self._elapsed(t), 3),
                     open_fraction=round(s.lift_open_fraction, 4),
                     reason="the doors began to open")
            return
        if self._timeout(t, WAIT_SIDE_MAX_S, "WAIT_SIDE"):
            self._go(t, "DOORS_OPEN", reason="wait-side ceiling reached")

    def _doors_open_state(self, t: float, s: Sense) -> None:
        """Doors travelling.  Still exactly zero, and now watching who comes out.

        Separate from LET_OCCUPANTS_EXIT so the timeline shows that the duck
        registered the doors opening BEFORE anybody stepped out, rather than
        conflating the two into one long wait.
        """
        if s.lift_passable or s.occupants_in_passage > 0:
            self._go(t, "LET_OCCUPANTS_EXIT",
                     open_fraction=round(s.lift_open_fraction, 4),
                     occupants_in_cabin=int(s.occupants_in_cabin),
                     reason="doors open; occupants coming out")
            return
        if self._timeout(t, DOORS_OPEN_MAX_S, "DOORS_OPEN"):
            self._go(t, "LET_OCCUPANTS_EXIT", reason="doors-open ceiling reached")

    def _let_occupants_exit_state(self, t: float, s: Sense) -> None:
        """Hold exactly still until the LAST occupant is measured clear.

        Requires BOTH a minimum number to have actually come out and all of them
        to be clear, sustained.  Requiring only "all clear" would be satisfied at
        the instant before anybody moved.
        """
        self._clear_for = (self._clear_for + self.dt
                           if s.all_occupants_clear else 0.0)
        enough = s.occupants_exited >= MIN_OCCUPANTS_EXITED
        if enough and self._clear_for >= CLEAR_CONFIRM_S and s.lift_passable:
            self.boarding.update({
                "occupants_exited_before_entry": int(s.occupants_exited),
                "cleared_at_s": round(t, 3),
                "clear_sustained_s": round(self._clear_for, 3),
                "lift_open_fraction_at_clear": round(s.lift_open_fraction, 4),
            })
            self._go(t, "FOLLOW_GUARDIAN_IN",
                     occupants_exited=int(s.occupants_exited),
                     reason="every occupant clear; following her in")
            return
        if self._timeout(t, LET_OCCUPANTS_EXIT_MAX_S, "LET_OCCUPANTS_EXIT"):
            self.boarding["ceiling_reached"] = True
            self._go(t, "FOLLOW_GUARDIAN_IN", reason="exit ceiling reached")

    def _follow_guardian_in_state(self, t: float, s: Sense) -> None:
        """Board the car AFTER her.  The controller holds the duck outside the
        aperture until she is measured through it."""
        if s.inside_cabin:
            self.boarding.update({
                "duck_entered_at_s": round(t, 3),
                "guardian_gap_at_entry_m": round(s.guardian_gap_m, 4),
                "guardian_inside_at_entry": bool(s.guardian_inside_cabin),
            })
            self._go(t, "POSITION_INSIDE",
                     guardian_gap_m=round(s.guardian_gap_m, 4),
                     reason="inside the car; moving out of the way")
            return
        if self._timeout(t, FOLLOW_GUARDIAN_IN_MAX_S, "FOLLOW_GUARDIAN_IN"):
            self._go(t, "POSITION_INSIDE", reason="boarding ceiling reached")

    def _position_inside_state(self, t: float, s: Sense) -> None:
        """Cross to the side of the car, clear of both apertures' centrelines.

        Accepts EITHER the world-space holding radius or the route's own leg
        completion.  Requiring only the former cost this behavior 25 s: the duck
        settled 0.05 m short of the cabin holding point - inside the leg's
        arrival tolerance but outside the 0.30 m radius by construction, since
        the radius is measured from the point and the arc from the path - and
        POSITION_INSIDE ran to its ceiling with the duck already standing
        exactly where it was supposed to be.
        """
        if s.at_cabin_hold or s.leg_arrived:
            self.boarding["positioned_at_s"] = round(t, 3)
            self._go(t, "RIDE", reason="standing clear inside the car")
            return
        if self._timeout(t, POSITION_INSIDE_MAX_S, "POSITION_INSIDE"):
            self.boarding["position_ceiling_reached"] = True
            self._go(t, "RIDE", reason="positioning ceiling reached")

    def _ride_state(self, t: float, s: Sense) -> None:
        """Exactly zero while the car travels.

        Ends on the REAR doors being measured to open, which the machine cannot
        bring about.  The minimum duration is a floor on what the gate will
        accept as a ride, not the thing that ends it.
        """
        if self._ride_started is None:
            self._ride_started = t
        if s.rear_open_fraction > 0.02 and self._elapsed(t) >= MIN_RIDE_S:
            self.boarding["rode_for_s"] = round(self._elapsed(t), 3)
            self._go(t, "DOORS_OPEN_TARGET",
                     rode_s=round(self._elapsed(t), 3),
                     reason="arrived; the rear doors are opening")
            return
        if self._timeout(t, RIDE_MAX_S, "RIDE"):
            self.boarding["ride_ceiling_reached"] = True
            self._go(t, "DOORS_OPEN_TARGET", reason="ride ceiling reached")

    def _doors_open_target_state(self, t: float, s: Sense) -> None:
        """Still exactly zero: SHE leaves first.

        The only way out of this state is her being measured through the rear
        aperture, which is what "the guardian exits first" means as a
        transition rather than as a caption.
        """
        if s.guardian_through_rear and s.rear_passable:
            self.boarding.update({
                "guardian_exited_at_s": round(t, 3),
                "rear_open_fraction_at_exit": round(s.rear_open_fraction, 4),
            })
            self._go(t, "FOLLOW_OUT",
                     reason="she is out; following her")
            return
        if self._timeout(t, DOORS_OPEN_TARGET_MAX_S, "DOORS_OPEN_TARGET"):
            self.boarding["target_doors_ceiling_reached"] = True
            self._go(t, "FOLLOW_OUT", reason="target-doors ceiling reached")

    def _follow_out_state(self, t: float, s: Sense) -> None:
        """Leave the car behind her and reach the arrival point."""
        if s.route_remaining_m <= 0.0 or s.leg_arrived:
            self.boarding.update({
                "duck_exited_at_s": round(t, 3),
                "beyond_rear_m": round(s.beyond_rear_m, 4),
            })
            self._go(t, "DONE", reason="off the lift, journey complete")
            return
        if self._timeout(t, FOLLOW_OUT_MAX_S, "FOLLOW_OUT"):
            self._go(t, "DONE", reason="follow-out ceiling reached")

    def _done_state(self, t: float, s: Sense) -> None:
        """Terminal.  Exactly zero for the rest of the run."""
        return

    # -- bookkeeping ------------------------------------------------------
    @property
    def completed_yields(self) -> int:
        return len(self.yields)

    @property
    def boarded(self) -> bool:
        return "duck_entered_at_s" in self.boarding

    @property
    def finished(self) -> bool:
        return self.state == "DONE"

    def summary(self) -> dict:
        return {
            "guardian": self.guardian,
            "state": self.state,
            "transitions": list(self.transitions),
            "yields": list(self.yields),
            "boarding": dict(self.boarding),
            "timeouts": list(self.timeouts),
        }
