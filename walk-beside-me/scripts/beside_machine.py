#!/usr/bin/env python3
"""The state machine: joining a side, losing it, and crossing to the other one.

    ACQUIRE -> JOIN_SIDE -> BESIDE_LEFT/BESIDE_RIGHT
                                 |
                                 v
                          SIDE_BLOCKED -> FALL_BACK -> CROSS_BEHIND
                                 |                          |
                                 +--------------------------+
                                 v
                        JOIN_OTHER_SIDE -> BESIDE_LEFT/BESIDE_RIGHT ...

``BESIDE`` is the family name for the two concrete formation states; it is
reported in the metrics as the union so "time spent beside her" is one number,
while the machine itself always knows WHICH side, because that is the whole
point of the behavior.

The machine never touches physics and never emits a command; ``beside_control``
does that from the state.  Keeping the two apart is what lets every transition
rule be unit-tested on hand-built inputs, with no MuJoCo anywhere.

FOUR INVARIANTS ARE STRUCTURAL RATHER THAN CHECKED AFTERWARDS
---------------------------------------------------------------
* **A switch is caused, never scheduled.**  ``SIDE_BLOCKED`` is entered only
  after the CURRENT side has been continuously measured unusable for
  ``BLOCK_CONFIRM_S``, and the machine records the measured cause it was given
  at that moment.  There is no timer, no waypoint index and no schedule lookup
  anywhere in this file.

* **A switch is committed to before it is begun.**  The far side must be
  continuously usable for ``CLEAR_CONFIRM_S`` before ``FALL_BACK`` is entered.
  Crossing into a lane that is about to close would be worse than staying in a
  bad one, because the duck spends the crossing behind the guardian where it
  cannot react.

* **The crossing goes astern.**  ``FALL_BACK`` ends only when the duck is at
  least ``CROSS_BEHIND_M`` behind her, and ``CROSS_BEHIND`` ends only when the
  duck has committed laterally to the far side.  The duck is therefore behind
  her for the whole of the lateral transit, by construction rather than by luck.

* **A switch cannot chatter.**  After a completed switch the machine refuses to
  consider another for ``SWITCH_COOLDOWN_S``.  Without it, a duck on the
  boundary of two marginal lanes ping-pongs and the switch count becomes a count
  of ticks rather than of decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from beside_constants import (
    ACQUIRE_MAX_S,
    BLOCK_CONFIRM_S,
    CLEAR_CONFIRM_S,
    CROSS_MAX_S,
    FALL_BACK_MAX_S,
    JOIN_MAX_S,
    JOIN_OTHER_MAX_S,
    JOIN_SETTLE_S,
    SWITCH_COOLDOWN_S,
)
from beside_geometry import (
    CROSS_BEHIND_M,
    CROSS_COMMIT_M,
    side_name,
)


@dataclass
class BesideMachine:
    """Transitions, side decisions and the switch log.  No physics, no MuJoCo."""

    ctrl_hz: float = 50.0
    state: str = "ACQUIRE"
    state_since: float = 0.0
    # The side the duck is currently holding or heading for, +1 left / -1 right.
    side: int | None = None
    # The side a switch in progress is heading TO.
    target_side: int | None = None
    guardian: str = ""

    transitions: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    switches: list[dict] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)

    _blocked_for: float = 0.0
    _far_clear_for: float = 0.0
    _formed_for: float = 0.0
    _last_switch_t: float | None = None
    _switch: dict = field(default_factory=dict)
    _joined: bool = False

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

    def set_guardian(self, name: str) -> None:
        """Record the identity being accompanied.  Callable once, by design."""
        if self.guardian and self.guardian != name:
            raise ValueError(
                f"guardian is {self.guardian!r} and cannot be reassigned to "
                f"{name!r}: the formation is defined relative to one person")
        self.guardian = name

    @property
    def beside_state(self) -> str:
        return "BESIDE_LEFT" if self.side == 1 else "BESIDE_RIGHT"

    def cooldown_active(self, t: float) -> bool:
        return (self._last_switch_t is not None
                and t - self._last_switch_t < SWITCH_COOLDOWN_S)

    def note_decision(self, t: float, kind: str, side: int | None,
                      reason: str, verdicts: dict) -> None:
        """Log a side decision with the measurement that produced it.

        Every entry carries both sides' measured gaps, so a decision can be
        audited against what the duck could actually see rather than against the
        label it gave itself.
        """
        self.decisions.append({
            "t": round(t, 3),
            "kind": kind,
            "side": side,
            "side_name": None if side is None else side_name(side),
            "reason": reason,
            "left": verdicts[1].as_record(),
            "right": verdicts[-1].as_record(),
        })

    # -- the machine -----------------------------------------------------
    def update(self, t: float, *, formation_ok: bool, lateral: float,
               longitudinal: float, verdicts: dict, preferred: int | None,
               preference_reason: str) -> tuple[str, bool]:
        """Advance one control tick.

        ``formation_ok`` is the measured predicate from ``beside_geometry``:
        correct side, lateral inside the band, bounded longitudinal error.
        ``verdicts`` maps side to a ``SideVerdict``.  ``preferred`` is the side
        the chooser recommends right now, or ``None`` if neither is usable.
        """
        before = self.state

        if self.state == "ACQUIRE":
            self._acquire(t, preferred, preference_reason, verdicts)
        elif self.state == "JOIN_SIDE":
            self._join(t, formation_ok, verdicts)
        elif self.state in ("BESIDE_LEFT", "BESIDE_RIGHT"):
            self._beside(t, formation_ok, verdicts, preferred)
        elif self.state == "SIDE_BLOCKED":
            self._blocked(t, verdicts)
        elif self.state == "FALL_BACK":
            self._fall_back(t, longitudinal)
        elif self.state == "CROSS_BEHIND":
            self._cross(t, lateral)
        elif self.state == "JOIN_OTHER_SIDE":
            self._join_other(t, formation_ok, verdicts)

        return self.state, self.state != before

    # -- per-state rules -------------------------------------------------
    def _acquire(self, t: float, preferred: int | None, reason: str,
                 verdicts: dict) -> None:
        """Walk up to her and pick a side, refusing the one that is occupied."""
        if preferred is not None:
            self.side = preferred
            self.note_decision(t, "initial", preferred, reason, verdicts)
            self._go(t, "JOIN_SIDE", side=side_name(preferred), reason=reason)
            return
        self._timeout(t, ACQUIRE_MAX_S, "ACQUIRE")

    def _join(self, t: float, formation_ok: bool, verdicts: dict) -> None:
        """Close into the chosen slot and hold it long enough to count."""
        if formation_ok:
            self._formed_for += self.dt
            if self._formed_for >= JOIN_SETTLE_S:
                self._joined = True
                self._formed_for = 0.0
                self._go(t, self.beside_state, side=side_name(self.side),
                         reason="formation established")
                return
        else:
            self._formed_for = 0.0
        # A side that becomes unusable DURING the join is abandoned before it is
        # ever occupied: the duck re-picks rather than completing a join into a
        # slot it has just measured as blocked.
        if self.side is not None and not verdicts[self.side].usable:
            other = -self.side
            if verdicts[other].usable:
                self.side = other
                self.note_decision(
                    t, "join_reroute", other,
                    f"chosen side became unusable "
                    f"({verdicts[-other].cause}:{verdicts[-other].detail})",
                    verdicts)
                self._go(t, "JOIN_SIDE", side=side_name(other),
                         reason="chosen side blocked before arrival")
                return
        self._timeout(t, JOIN_MAX_S, "JOIN_SIDE")

    def _beside(self, t: float, formation_ok: bool, verdicts: dict,
                preferred: int | None) -> None:
        """Hold the formation, and watch the lane the duck is standing in."""
        if self.side is None:
            return
        if verdicts[self.side].usable:
            self._blocked_for = 0.0
            self._far_clear_for = 0.0
            return
        self._blocked_for += self.dt
        # The far side must be provably clear for its own window before the duck
        # will commit; the two counters run independently.
        if verdicts[-self.side].usable:
            self._far_clear_for += self.dt
        else:
            self._far_clear_for = 0.0
        if self.cooldown_active(t):
            return
        if (self._blocked_for >= BLOCK_CONFIRM_S
                and self._far_clear_for >= CLEAR_CONFIRM_S):
            verdict = verdicts[self.side]
            self.target_side = -self.side
            self._switch = {
                "index": len(self.switches),
                "from_side": side_name(self.side),
                "to_side": side_name(self.target_side),
                "blocked_at_s": round(t, 3),
                "cause": verdict.cause,
                "detail": verdict.detail,
                "blocked_for_s": round(self._blocked_for, 3),
                "far_clear_for_s": round(self._far_clear_for, 3),
                "static_gap_m": round(verdict.static_gap_m, 4),
                "person_gap_m": round(verdict.person_gap_m, 4),
            }
            self.note_decision(
                t, "blocked", self.target_side,
                f"{side_name(self.side)} blocked by "
                f"{verdict.cause}:{verdict.detail}", verdicts)
            self._go(t, "SIDE_BLOCKED", cause=verdict.cause,
                     detail=verdict.detail,
                     blocked_for_s=round(self._blocked_for, 3))

    def _blocked(self, t: float, verdicts: dict) -> None:
        """One deliberate tick acknowledging the blockage, then fall back.

        SIDE_BLOCKED exists as its own state rather than folding into FALL_BACK
        so the metrics and the timeline can name the instant the duck decided,
        separately from the manoeuvre that follows.
        """
        self._go(t, "FALL_BACK", to_side=side_name(self.target_side),
                 reason="dropping behind to cross")

    def _fall_back(self, t: float, longitudinal: float) -> None:
        """Drop astern until there is room to cross behind her."""
        if longitudinal <= -CROSS_BEHIND_M:
            self._switch["fell_back_at_s"] = round(t, 3)
            self._switch["longitudinal_at_cross_m"] = round(longitudinal, 4)
            self._go(t, "CROSS_BEHIND", longitudinal_m=round(longitudinal, 4),
                     reason="clear astern; crossing")
            return
        if self._timeout(t, FALL_BACK_MAX_S, "FALL_BACK"):
            self._go(t, "CROSS_BEHIND", reason="fall-back ceiling reached")

    def _cross(self, t: float, lateral: float) -> None:
        """Transit laterally, behind her, until committed to the far side."""
        if self.target_side is None:
            return
        committed = (lateral * self.target_side) >= CROSS_COMMIT_M
        if committed:
            self.side = self.target_side
            self._switch["crossed_at_s"] = round(t, 3)
            self._go(t, "JOIN_OTHER_SIDE", side=side_name(self.side),
                     reason="crossed astern; closing into the far slot")
            return
        if self._timeout(t, CROSS_MAX_S, "CROSS_BEHIND"):
            self.side = self.target_side
            self._go(t, "JOIN_OTHER_SIDE", side=side_name(self.side),
                     reason="crossing ceiling reached")

    def _join_other(self, t: float, formation_ok: bool, verdicts: dict) -> None:
        """Come up the far side into formation, and close the switch record."""
        if formation_ok:
            self._formed_for += self.dt
            if self._formed_for >= JOIN_SETTLE_S:
                self._formed_for = 0.0
                self._blocked_for = 0.0
                self._far_clear_for = 0.0
                self._last_switch_t = t
                self._switch["joined_at_s"] = round(t, 3)
                self._switch["duration_s"] = round(
                    t - self._switch["blocked_at_s"], 3)
                self.switches.append(self._switch)
                self._switch = {}
                self.target_side = None
                self._go(t, self.beside_state, side=side_name(self.side),
                         reason="formation re-established on the other side")
                return
        else:
            self._formed_for = 0.0
        self._timeout(t, JOIN_OTHER_MAX_S, "JOIN_OTHER_SIDE")

    # -- bookkeeping ------------------------------------------------------
    @property
    def completed_switches(self) -> int:
        return len(self.switches)

    @property
    def joined(self) -> bool:
        return self._joined

    def summary(self) -> dict:
        return {
            "guardian": self.guardian,
            "state": self.state,
            "side": self.side,
            "transitions": list(self.transitions),
            "decisions": list(self.decisions),
            "switches": list(self.switches),
            "timeouts": list(self.timeouts),
        }
