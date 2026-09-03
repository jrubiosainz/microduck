#!/usr/bin/env python3
"""The state machine: find the instructor, watch her, confirm what she said,
carry it out, acknowledge, and go back to ready.

    READY -> OBSERVE -> CONFIRM -> EXECUTE_APPROACH -> ACK -> READY
                                -> EXECUTE_STOP      -> ACK -> READY
                                -> EXECUTE_TURN_LEFT -> ACK -> READY
                                -> EXECUTE_TURN_RIGHT-> ACK -> READY
                                -> EXECUTE_BACK_UP   -> ACK -> READY
                                -> GOODBYE -> DONE
            <- CONFIRM returns to OBSERVE when the reading does not hold

This file owns the machine; ``gest_control`` emits the commands from its state
and ``gest_detect`` decides what was said.  Neither touches physics.  Keeping
them apart is what lets every transition rule be unit-tested on hand-built
inputs, with no MuJoCo anywhere.

FIVE INVARIANTS ARE STRUCTURAL RATHER THAN CHECKED AFTERWARDS
---------------------------------------------------------------
* **No execute state can be entered except from CONFIRM**, and CONFIRM exits to
  one only when the detector's own confirm window is complete.  There is no
  timer, no schedule and no other path into an action.

* **The command decides the state, not the other way round.**  The transition
  reads ``STATE_FOR_COMMAND``, so a state cannot exist that no confirmed command
  maps to, and a confirmed command cannot silently execute the wrong action.

* **A STOP is entered from wherever the duck is**, including mid-walk, and
  ``EXECUTE_STOP`` is a zero-command state.  The interruption is therefore
  structural: the tick CONFIRM ends is the tick the command register goes to
  zero, with nothing in between that could ramp it down.

* **Gesture reading is SUSPENDED for the whole of every execute state.**  The
  duck does not take a new order while carrying one out, which also stops a
  half-finished action from being graded against a command it was not doing.

* **A ceiling MOVES the machine.**  Every ``_timeout`` here transitions and
  records why.  A ceiling that only appends to a log is not a ceiling.

WHY THERE IS AN OBSERVE STATE AT ALL, DISTINCT FROM READY
-----------------------------------------------------------
READY is "I have nobody to watch"; OBSERVE is "I am watching the person I have
locked and nothing has been read yet"; CONFIRM is "something is being read and I
am timing it".  Collapsing the three would make the HUD unable to show why the
duck is standing still, and would lose the distinction between a robot that has
not found its instructor and one that is waiting for her to say something.  All
three hold an exact zero, so the distinction costs nothing physically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gest_episode import Episode, Sense
from gest_states import (
    ACK_MAX_S,
    ACK_S,
    APPROACH_MAX_S,
    BACK_UP_MAX_S,
    BACK_UP_TARGET_M,
    BACK_UP_TOLERANCE_M,
    CONFIRM_MAX_S,
    GOODBYE_MAX_S,
    GOODBYE_S,
    INTERRUPT_COMMAND,
    OBSERVE_MAX_S,
    READY_MAX_S,
    STATE_FOR_COMMAND,
    STOP_HOLD_S,
    STOP_MAX_S,
    TURN_MAX_S,
    TURN_TARGET_DEG,
    TURN_TOLERANCE_DEG,
    WALKING_STATES,
)


@dataclass
class GestureMachine:
    """Transitions and the episode log.  No physics, no commands, no camera."""

    ctrl_hz: float = 50.0
    state: str = "READY"
    state_since: float = 0.0

    transitions: list[dict] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    # Every time a confirmed STOP cut into a manoeuvre already under way.
    interrupts: list[dict] = field(default_factory=list)

    _episode: Episode | None = None
    _observed_at: float = 0.0

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
        # THE INTERRUPT IS CHECKED BEFORE THE STATE'S OWN HANDLER, and it is the
        # only transition in the machine that does not originate in CONFIRM.
        # That is deliberate, and it is what makes a STOP a stop: the command
        # exists to interrupt motion already under way, so a duck that could
        # only accept it while standing still would be following a script.
        #
        # It is still a fully confirmed command - the detector applies the same
        # locked-person, arm-readability and sustained-window gate to it as to
        # any other - so this is a narrower ENTRY, never a weaker one.
        if self.state in WALKING_STATES and sense.confirmed is not None \
                and sense.confirmed["command"] == INTERRUPT_COMMAND:
            self._interrupt(t, sense)
            return self.state, True
        getattr(self, f"_{self.state.lower()}_state")(t, sense)
        return self.state, self.state != before

    def _interrupt(self, t: float, s: Sense) -> None:
        """A confirmed STOP arriving mid-manoeuvre.

        The episode under way is CLOSED AS INTERRUPTED rather than silently
        abandoned, so the log distinguishes a command the duck completed from
        one it was told to stop doing - and the STOP opens its own episode,
        which is what the acceptance gate grades the zero-command claim on.
        """
        interrupted = self._episode.command if self._episode else ""
        if self._episode is not None:
            self._episode.interrupted_by = INTERRUPT_COMMAND
            self.close_episode(t)
        self.interrupts.append({
            "t": round(t, 3), "interrupted": interrupted,
            "from_state": self.state})
        self._open_episode(t, s)
        self._go(t, "EXECUTE_STOP", command=INTERRUPT_COMMAND,
                 confidence=round(float(s.confirmed["confidence"]), 3),
                 held_s=round(float(s.confirmed["held_s"]), 3),
                 interrupted=interrupted,
                 reason=f"STOP confirmed mid-{interrupted}: interrupting a "
                        "manoeuvre already under way")

    # -- waiting ----------------------------------------------------------
    def _ready_state(self, t: float, s: Sense) -> None:
        """Standing at exactly zero, sweeping the head for the instructor.

        Leaves for OBSERVE only once acquisition has LOCKED, which requires the
        requested identity to have been confirmed in the real camera for the
        measured dwell.  A person merely being in frame is not enough.
        """
        if s.locked and s.instructor_visible:
            self._observed_at = t
            self._go(t, "OBSERVE", person=s.locked,
                     reason=f"{s.locked} acquired; watching for a command")
            return
        if self._timeout(t, READY_MAX_S, "READY"):
            self._go(t, "DONE", reason="ready ceiling reached with no lock")

    def _observe_state(self, t: float, s: Sense) -> None:
        """Watching the locked instructor, at exactly zero, nothing read yet.

        Enters CONFIRM as soon as a template matches, and the confirm window is
        timed THERE - so the HUD can show a timer that means something and the
        episode can record when watching turned into reading.
        """
        if s.candidate_command:
            self._go(t, "CONFIRM", command=s.candidate_command,
                     confidence=round(s.candidate_confidence, 3),
                     reason=f"reading a possible {s.candidate_command}")
            return
        if not s.instructor_visible and self._elapsed(t) >= OBSERVE_MAX_S:
            self.timeouts.append(f"OBSERVE@{t:.2f}s")
            self._go(t, "READY", reason="lost sight of the instructor")

    def _confirm_state(self, t: float, s: Sense) -> None:
        """Timing a sustained reading.  Acts only when the window completes.

        THE RETURN PATH MATTERS AS MUCH AS THE ACCEPTANCE.  A reading that stops
        holding sends the duck back to OBSERVE with nothing executed, which is
        exactly what the ambiguous partial gesture must produce: it is refused
        here, not by a special case somewhere else.
        """
        if s.confirmed is not None:
            command = s.confirmed["command"]
            target = STATE_FOR_COMMAND.get(command, "")
            if not target:
                self._go(t, "OBSERVE",
                         reason=f"confirmed {command} maps to no action")
                return
            self._open_episode(t, s)
            self._go(t, target, command=command,
                     confidence=round(float(s.confirmed["confidence"]), 3),
                     held_s=round(float(s.confirmed["held_s"]), 3),
                     reason=f"{command} confirmed after "
                            f"{s.confirmed['held_s']:.2f}s sustained")
            return
        if not s.candidate_command:
            self._go(t, "OBSERVE",
                     reason="the reading did not hold; nothing executed")
            return
        if self._timeout(t, CONFIRM_MAX_S, "CONFIRM"):
            self._go(t, "OBSERVE", reason="confirm ceiling reached")

    def _open_episode(self, t: float, s: Sense) -> None:
        confirmed = s.confirmed or {}
        self._episode = Episode(
            index=len(self.episodes),
            command=confirmed.get("command", ""),
            template=confirmed.get("template", ""),
            person=s.locked,
            observed_at_s=self._observed_at,
            confirmed_at_s=t,
            confirm_held_s=float(confirmed.get("held_s", 0.0)),
            confirm_fraction=float(confirmed.get("match_fraction", 0.0)),
            confirm_readable_fraction=float(
                confirmed.get("readable_fraction", 0.0)),
            confidence=float(confirmed.get("confidence", 0.0)),
            rule=confirmed.get("rule", ""),
            features=dict(confirmed.get("features", {})),
            executed_at_s=t,
            execute_state=STATE_FOR_COMMAND.get(confirmed.get("command", ""), ""),
        )

    # -- carrying it out ---------------------------------------------------
    def _execute_approach_state(self, t: float, s: Sense) -> None:
        """Walk toward the instructor and stop inside the safe standoff band.

        The exit is on MEASURED surface clearance, not on having reached a
        planned point: a target placed badly still cannot produce a close
        approach, because the band is checked against the contact probe's own
        number every tick.
        """
        if s.in_standoff_band:
            self._go(t, "ACK", reason="inside the safe standoff band")
            return
        if self._timeout(t, APPROACH_MAX_S, "EXECUTE_APPROACH"):
            self._go(t, "ACK", reason="approach ceiling reached")

    def _execute_stop_state(self, t: float, s: Sense) -> None:
        """Exactly zero, held, having interrupted whatever was under way."""
        if s.stop_hold_s >= STOP_HOLD_S:
            self._go(t, "ACK", reason="stopped and held")
            return
        if self._timeout(t, STOP_MAX_S, "EXECUTE_STOP"):
            self._go(t, "ACK", reason="stop ceiling reached")

    def _turn_done(self, t: float, s: Sense, direction: str) -> None:
        """Shared exit for both turns: the MEASURED yaw delta reached target.

        Graded on the trunk yaw the duck actually turned through, per sign, with
        the sign REQUIRED to match the direction.  A left turn that drifted
        right would fail rather than pass on magnitude alone.
        """
        wanted = TURN_TARGET_DEG if direction == "left" else -TURN_TARGET_DEG
        turned = s.yaw_delta_deg
        reached = (turned >= wanted - TURN_TOLERANCE_DEG if direction == "left"
                   else turned <= wanted + TURN_TOLERANCE_DEG)
        if reached:
            self._go(t, "ACK",
                     turned_deg=round(turned, 2),
                     reason=f"turned {turned:+.1f} deg "
                            f"(target {wanted:+.0f})")
            return
        if self._timeout(t, TURN_MAX_S, f"EXECUTE_TURN_{direction.upper()}"):
            self._go(t, "ACK", turned_deg=round(turned, 2),
                     reason="turn ceiling reached")

    def _execute_turn_left_state(self, t: float, s: Sense) -> None:
        self._turn_done(t, s, "left")

    def _execute_turn_right_state(self, t: float, s: Sense) -> None:
        self._turn_done(t, s, "right")

    def _execute_back_up_state(self, t: float, s: Sense) -> None:
        """Reverse until the MEASURED displacement along the PRE-ACTION heading
        reaches the target.

        Projected on the heading held when the command was accepted, so a robot
        that curled sideways under the reverse gait's large measured yaw drift
        does not get credit for the distance it travelled off-axis.
        """
        if s.back_along_heading_m >= BACK_UP_TARGET_M - BACK_UP_TOLERANCE_M:
            self._go(t, "ACK",
                     back_m=round(s.back_along_heading_m, 3),
                     reason=f"reversed {s.back_along_heading_m:.3f} m along "
                            "the pre-action heading")
            return
        if self._timeout(t, BACK_UP_MAX_S, "EXECUTE_BACK_UP"):
            self._go(t, "ACK", back_m=round(s.back_along_heading_m, 3),
                     reason="back-up ceiling reached")

    # -- finishing each command ---------------------------------------------
    def _ack_state(self, t: float, s: Sense) -> None:
        """Acknowledge at exactly zero, then go back to ready for the next one."""
        if self._elapsed(t) >= ACK_S or self._timeout(t, ACK_MAX_S, "ACK"):
            self.close_episode(t)
            self._go(t, "READY",
                     reason="acknowledged; ready for the next command")

    def _goodbye_state(self, t: float, s: Sense) -> None:
        """The final acknowledgment: a longer hold, then the session is over."""
        if self._elapsed(t) >= GOODBYE_S \
                or self._timeout(t, GOODBYE_MAX_S, "GOODBYE"):
            self.close_episode(t)
            self._go(t, "DONE",
                     reason="waved off: the session is complete")

    def _done_state(self, t: float, s: Sense) -> None:
        """Terminal.  Exactly zero for the rest of the run."""
        return

    # -- bookkeeping ---------------------------------------------------------
    def close_episode(self, t: float) -> None:
        if self._episode is None:
            return
        self._episode.ended_at_s = t
        self.episodes.append(self._episode)
        self._episode = None

    @property
    def episode(self) -> Episode | None:
        return self._episode

    @property
    def executing(self) -> bool:
        return self.state.startswith("EXECUTE_") or self.state == "GOODBYE"

    @property
    def accepted_commands(self) -> list[str]:
        return [e.command for e in self.episodes]

    @property
    def finished(self) -> bool:
        return self.state == "DONE"

    def summary(self) -> dict:
        return {
            "state": self.state,
            "transitions": list(self.transitions),
            "episodes": [e.as_record() for e in self.episodes],
            "accepted_commands": self.accepted_commands,
            "timeouts": list(self.timeouts),
            "interrupts": list(self.interrupts),
        }
