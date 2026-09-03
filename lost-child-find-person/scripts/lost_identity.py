#!/usr/bin/env python3
"""Identity: scoring a sighting, and refusing to lock onto a look-alike.

Pure logic.  No MuJoCo, no rendering, no physics — every rule here is unit-
testable on hand-built inputs, which is the point of keeping it separate from
the camera that produces those inputs.

THE THREE RULES THAT MAKE A LOCK HONEST
----------------------------------------
1. **A partial view can never confirm.**  Confirmation requires every one of the
   four appearance features to have been READABLE, not merely to have matched.
   "Everything I could see matched" is the classic re-identification failure,
   and here it is refused explicitly by :func:`evaluate`.

2. **Confirmation is a duration, not an instant.**  A sighting must hold at or
   above the accept score, with a complete descriptor, for
   ``REACQUIRE_CONFIRM_S`` of continuous camera time.  A single frame in which a
   look-alike happens to be side-on and unreadable cannot promote her.

3. **A rejected candidate stays rejected.**  Once a body has been evaluated and
   refused, it is not re-scored for ``REJECT_COOLDOWN_S``.  Without that, the
   head sweep re-acquires the same look-alike on every pass and the count of
   rejections becomes a count of ticks rather than a count of people.

WHY THE LOOK-ALIKES SCORE WHERE THEY DO
----------------------------------------
With the weights in ``lost_cast`` (shirt 0.45, stature 0.35, cap 0.10,
satchel 0.10) and a complete descriptor:

* ``mira``  — shirt ~0.03 off, same height, same bag, but a cap.
  Score ~= 1 - (0.45*0.03 + 0.35*0.07 + 0.10*1.0 + 0) ~= 0.86
* ``sofia`` — shirt ~0.03 off, same bag, no cap, 12 cm shorter.
  Score ~= 1 - (0.45*0.03 + 0.35*0.48 + 0 + 0) ~= 0.82

Both sit above the 0.55 candidate threshold and below the 0.90 accept
threshold: high enough that they must be evaluated, low enough that they must be
refused.  A distractor in a red shirt would score ~0.4 and would never test the
identity layer at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lost_cast import WEIGHTS, match_score, rejection_reason
from lost_constants import (
    ACCEPT_SCORE,
    CANDIDATE_MIN_S,
    CANDIDATE_SCORE,
    READ_CONE_DEG,
    REJECT_COOLDOWN_S,
)


@dataclass(frozen=True)
class Sighting:
    """One evaluated look at one body."""

    name: str
    t: float
    score: float
    penalties: dict[str, float]
    readable: tuple[str, ...]
    complete: bool
    range_m: float
    off_axis_deg: float
    verdict: str            # "accept" | "candidate" | "ignore"
    reason: str

    def as_record(self) -> dict:
        return {
            "name": self.name, "t": round(self.t, 3),
            "score": round(self.score, 4),
            "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
            "readable": list(self.readable),
            "complete_descriptor": self.complete,
            "range_m": round(self.range_m, 4),
            "off_axis_deg": round(self.off_axis_deg, 2),
            "verdict": self.verdict, "reason": self.reason,
        }


def evaluate(name: str, t: float, reference: dict, entry: dict) -> Sighting:
    """Score one camera observation of one body against the guardian's descriptor.

    ``entry`` is a person's record from ``LostCamera.update``.  A body that is
    not visible, or is too far off the optical axis for its appearance to be
    read reliably, is ignored rather than scored: a smear at the edge of the
    frame is not evidence either way.
    """
    observed = entry.get("observed", {})
    readable = tuple(sorted(entry.get("readable", ())))
    complete = set(readable) == set(WEIGHTS)
    score, penalties = match_score(reference, observed)

    if not entry.get("visible"):
        return Sighting(name, t, 0.0, penalties, readable, complete,
                        entry.get("range_m", float("nan")),
                        entry.get("off_axis_deg", 180.0),
                        "ignore", "not visible")
    if entry.get("off_axis_deg", 180.0) > READ_CONE_DEG:
        return Sighting(name, t, score, penalties, readable, complete,
                        entry["range_m"], entry["off_axis_deg"],
                        "ignore", "too far off the optical axis to read")
    if score >= ACCEPT_SCORE and complete:
        return Sighting(name, t, score, penalties, readable, complete,
                        entry["range_m"], entry["off_axis_deg"],
                        "accept", "appearance matches the guardian on every feature")
    if score >= CANDIDATE_SCORE:
        reason = (rejection_reason(name, penalties, set(readable), observed)
                  if complete else
                  f"incomplete descriptor: only {', '.join(readable) or 'nothing'} readable")
        return Sighting(name, t, score, penalties, readable, complete,
                        entry["range_m"], entry["off_axis_deg"],
                        "candidate", reason)
    return Sighting(name, t, score, penalties, readable, complete,
                    entry["range_m"], entry["off_axis_deg"],
                    "ignore", "appearance too dissimilar to evaluate")


@dataclass
class IdentityTracker:
    """Holds the guardian's descriptor and the running candidate bookkeeping.

    Owns three pieces of state and nothing else: how long the current candidate
    has been seen, how long an accept-grade sighting has held, and who has
    already been refused.
    """

    reference: dict
    guardian: str
    dt: float = 0.02
    seen_time: dict[str, float] = field(default_factory=dict)
    confirm_time: float = 0.0
    confirm_name: str | None = None
    rejected_until: dict[str, float] = field(default_factory=dict)
    rejections: list[dict] = field(default_factory=list)
    accepted: list[dict] = field(default_factory=list)
    # Every accept-grade sighting of somebody who is NOT the guardian.  This
    # must stay empty for the whole rollout; it is the wrong-lock counter.
    wrong_accepts: list[dict] = field(default_factory=list)

    def on_cooldown(self, name: str, t: float) -> bool:
        return t < self.rejected_until.get(name, -1.0)

    def note_visible(self, name: str, visible: bool) -> float:
        """Accumulate continuous visible time for one body, resetting on a gap."""
        if visible:
            self.seen_time[name] = self.seen_time.get(name, 0.0) + self.dt
        else:
            self.seen_time[name] = 0.0
        return self.seen_time[name]

    def ready_to_evaluate(self, name: str) -> bool:
        """Has this body been visible long enough to be worth scoring?"""
        return self.seen_time.get(name, 0.0) >= CANDIDATE_MIN_S

    def confirm(self, sighting: Sighting) -> float:
        """Accumulate continuous accept-grade time for one body.

        Resets whenever the accepting body changes or the sighting stops being
        accept-grade, so the returned duration is always CONTINUOUS.  A
        wrong-identity accept is recorded the instant it happens, before any
        duration logic, so the gate can never be satisfied by one that was
        merely too brief to promote.
        """
        if sighting.verdict != "accept":
            self.confirm_time = 0.0
            self.confirm_name = None
            return 0.0
        if sighting.name != self.guardian:
            self.wrong_accepts.append(sighting.as_record())
        if self.confirm_name != sighting.name:
            self.confirm_name = sighting.name
            self.confirm_time = 0.0
        self.confirm_time += self.dt
        return self.confirm_time

    def reject(self, sighting: Sighting, t: float) -> dict:
        """Record a refusal and put that body on cooldown."""
        self.rejected_until[sighting.name] = t + REJECT_COOLDOWN_S
        record = {**sighting.as_record(), "rejected_at_s": round(t, 3)}
        self.rejections.append(record)
        return record

    def accept(self, sighting: Sighting, t: float) -> dict:
        record = {**sighting.as_record(), "accepted_at_s": round(t, 3)}
        self.accepted.append(record)
        return record

    def distinct_rejected(self) -> tuple[str, ...]:
        """The distinct people the duck refused, in the order it refused them."""
        order: list[str] = []
        for record in self.rejections:
            if record["name"] not in order:
                order.append(record["name"])
        return tuple(order)

    def best_candidate(self, t: float, camera_people: dict) -> Sighting | None:
        """The strongest evaluable sighting this tick, cooldowns respected.

        The guardian is NOT exempt from the visible-time requirement.  Handing
        the real guardian a shortcut would make the confirmation gate untestable:
        it has to be the same gate for everybody, and the only way she wins is by
        actually scoring higher.
        """
        best: Sighting | None = None
        for name, entry in camera_people.items():
            self.note_visible(name, bool(entry.get("visible")))
            if self.on_cooldown(name, t) or not self.ready_to_evaluate(name):
                continue
            sighting = evaluate(name, t, self.reference, entry)
            if sighting.verdict == "ignore":
                continue
            if best is None or sighting.score > best.score:
                best = sighting
        return best
