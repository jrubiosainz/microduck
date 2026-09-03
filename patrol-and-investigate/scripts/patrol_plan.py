#!/usr/bin/env python3
"""The patrol plan and its MEMORY: where the duck is on the circuit, and where
it must come back to when something interrupts it.

THIS MODULE IS WHAT MAKES "IT RESUMED THE PATROL" A CLAIM RATHER THAN A CAPTION
--------------------------------------------------------------------------------
An investigation is an interruption, and the whole behavior turns on the
interruption being RECOVERABLE.  When the duck breaks off, three things are
recorded before it takes a single step toward the anomaly:

* the checkpoint it was walking to, **by name and by index**;
* the world position it was standing at when it broke off - the *resume point*;
* how many checkpoints it had already completed.

When the investigation ends, the duck walks back to that resume point, and only
then continues to the checkpoint it was originally heading for.  The gate checks
that the checkpoint index after resuming EQUALS the one recorded at
interruption, that the duck physically returned within a measured distance of
the resume point, and that no checkpoint was skipped, reordered or visited
twice.

Returning to the RESUME POINT rather than jumping straight to the checkpoint is
the honest reading of "resume where you left off", and it is also the only one
that is physically checkable: a duck that broke off two metres into a leg and
then approached the checkpoint from the far side would have completed the
checkpoint without ever walking the leg.

NOTHING HERE TOUCHES PHYSICS OR MUJOCO
----------------------------------------
The plan is pure geometry and pure bookkeeping, so every property below is
unit-tested on hand-built inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from patrol_facility import (
    CHECKPOINTS,
    CHECKPOINT_NAMES,
    HOME,
    LOOP_RADIUS_M,
)
from patrol_states import PURSUIT_LOOKAHEAD_M


@dataclass
class Interruption:
    """One interruption of the patrol, and everything needed to undo it.

    THE FIELDS ARE THE MEMORY.  ``target_index`` and ``target_name`` are what
    the duck must go back to doing; ``resume_xy`` is where it must be standing
    before it counts as having gone back to it.  ``completed`` is carried so a
    resumed patrol cannot silently lose or repeat a checkpoint.
    """

    index: int
    at_s: float
    target_index: int
    target_name: str
    resume_xy: tuple[float, float]
    completed: tuple[str, ...]
    reason: str
    target: str = ""
    resumed_at_s: float | None = None
    return_error_m: float | None = None
    resumed_target_name: str = ""

    def as_record(self) -> dict:
        return {
            "index": self.index,
            "at_s": round(float(self.at_s), 3),
            "target_index": int(self.target_index),
            "target_name": self.target_name,
            "resume_xy": [round(float(self.resume_xy[0]), 4),
                          round(float(self.resume_xy[1]), 4)],
            "completed_before": list(self.completed),
            "reason": self.reason,
            "anomaly": self.target,
            "resumed_at_s": (None if self.resumed_at_s is None
                             else round(float(self.resumed_at_s), 3)),
            "return_error_m": (None if self.return_error_m is None
                               else round(float(self.return_error_m), 4)),
            "resumed_target_name": self.resumed_target_name,
            # THE CLAIM, as one boolean computed HERE from the two names rather
            # than asserted downstream: the duck went back to the same
            # checkpoint it had been walking to.
            "route_preserved": bool(
                self.resumed_target_name == self.target_name),
        }


@dataclass
class PatrolPlan:
    """Progress round the circuit, and the memory of every interruption.

    ``target_index`` walks 0..len(CHECKPOINTS), where the final value means the
    duck is on its way HOME.  It is advanced ONLY by :meth:`complete_checkpoint`
    and is never touched by an interruption, which is the structural reason a
    diversion cannot lose the duck's place.
    """

    target_index: int = 0
    completed: list[str] = field(default_factory=list)
    interruptions: list[Interruption] = field(default_factory=list)
    _open: Interruption | None = None

    # -- where the duck is going ------------------------------------------
    @property
    def finished_circuit(self) -> bool:
        """Have all five checkpoints been completed?"""
        return self.target_index >= len(CHECKPOINTS)

    @property
    def target_name(self) -> str:
        """The name of the place the duck is currently walking to."""
        if self.finished_circuit:
            return HOME.name
        return CHECKPOINT_NAMES[self.target_index]

    @property
    def target_xy(self) -> np.ndarray:
        if self.finished_circuit:
            return HOME.position
        return CHECKPOINTS[self.target_index].position

    @property
    def target_watch_deg(self) -> float:
        if self.finished_circuit:
            return HOME.watch_deg
        return CHECKPOINTS[self.target_index].watch_deg

    @property
    def progress(self) -> tuple[int, int]:
        """``(completed, total)`` over the five numbered checkpoints."""
        return len(self.completed), len(CHECKPOINTS)

    # -- completing one -----------------------------------------------------
    def complete_checkpoint(self, name: str) -> None:
        """Record a checkpoint as visited and advance the target.

        Raises on an out-of-order completion rather than silently accepting it.
        A patrol that recorded checkpoints in whatever order it happened to
        reach them would make the ordering gate vacuous, so the ordering is
        enforced at the point the record is made, and the gate then checks the
        recorded sequence independently.
        """
        expected = self.target_name
        if name != expected:
            raise ValueError(
                f"checkpoint {name!r} completed while the plan's target was "
                f"{expected!r}; the patrol order is a requirement, not an "
                "outcome")
        self.completed.append(name)
        self.target_index += 1

    # -- interrupting and resuming -------------------------------------------
    def interrupt(self, t: float, duck_xy, reason: str,
                  target: str = "") -> Interruption:
        """Break off the patrol, remembering exactly where to come back to."""
        if self._open is not None:
            raise RuntimeError(
                "the patrol is already interrupted; a second interruption "
                "would overwrite the memory of the first")
        entry = Interruption(
            index=len(self.interruptions),
            at_s=float(t),
            target_index=int(self.target_index),
            target_name=self.target_name,
            resume_xy=(float(duck_xy[0]), float(duck_xy[1])),
            completed=tuple(self.completed),
            reason=reason,
            target=target,
        )
        self.interruptions.append(entry)
        self._open = entry
        return entry

    @property
    def open_interruption(self) -> Interruption | None:
        return self._open

    @property
    def resume_xy(self) -> np.ndarray | None:
        """The world point the duck must return to, or ``None``."""
        if self._open is None:
            return None
        return np.asarray(self._open.resume_xy, dtype=np.float64)

    def resume(self, t: float, duck_xy) -> Interruption:
        """Close the open interruption, measuring how well the duck got back.

        ``return_error_m`` is the MEASURED distance from the duck's trunk to the
        remembered resume point at the moment the patrol resumes, and
        ``resumed_target_name`` is read from the plan's own state - which was
        never modified by the diversion, so the two names agreeing is evidence
        that the memory survived rather than that it was reassigned.
        """
        if self._open is None:
            raise RuntimeError("no interruption is open; nothing to resume")
        entry = self._open
        entry.resumed_at_s = float(t)
        entry.return_error_m = float(np.linalg.norm(
            np.asarray(duck_xy, dtype=np.float64)[:2]
            - np.asarray(entry.resume_xy, dtype=np.float64)))
        entry.resumed_target_name = self.target_name
        self._open = None
        return entry

    # -- the route, for the controller and the HUD ---------------------------
    def leg(self, duck_xy) -> tuple[np.ndarray, np.ndarray]:
        """The straight leg the duck is currently walking: ``(from, to)``.

        The ``from`` end is the duck's own position rather than the previous
        checkpoint, because after an interruption the duck rejoins the leg part
        way along it and the line it should walk is the one from where it
        actually is.
        """
        return (np.asarray(duck_xy, dtype=np.float64)[:2], self.target_xy)

    def remaining_m(self, duck_xy) -> float:
        """Distance still to walk to the current target."""
        return float(np.linalg.norm(
            self.target_xy - np.asarray(duck_xy, dtype=np.float64)[:2]))

    def as_record(self) -> dict:
        return {
            "checkpoint_order": list(CHECKPOINT_NAMES),
            "completed": list(self.completed),
            "target_index": int(self.target_index),
            "target_name": self.target_name,
            "finished_circuit": self.finished_circuit,
            "interruptions": [i.as_record() for i in self.interruptions],
        }


def circuit_polyline(count: int = 22) -> list[np.ndarray]:
    """The whole patrol circuit, sampled, for the route markers.

    Drawn from the SAME checkpoint objects the plan walks, so the route a viewer
    sees on the floor is the route the duck is following.
    """
    corners = [HOME.position] + [c.position for c in CHECKPOINTS] \
        + [HOME.position]
    spans = [float(np.linalg.norm(b - a))
             for a, b in zip(corners, corners[1:])]
    total = sum(spans)
    points: list[np.ndarray] = []
    for index in range(count):
        distance = total * index / max(count - 1, 1)
        walked = 0.0
        for (a, b), span in zip(zip(corners, corners[1:]), spans):
            if walked + span >= distance or span <= 0.0:
                fraction = (distance - walked) / max(span, 1e-9)
                points.append(a + (b - a) * min(max(fraction, 0.0), 1.0))
                break
            walked += span
        else:
            points.append(corners[-1])
    return points


def pursuit_point(duck_xy, target_xy,
                  lookahead_m: float = PURSUIT_LOOKAHEAD_M) -> np.ndarray:
    """A point on the leg ahead of the duck, or the target when it is nearer.

    Pursuing a lookahead point rather than the target itself is what keeps the
    heading loop from oscillating as the duck closes: the bearing to a point
    0.34 m ahead changes slowly, while the bearing to a target 0.05 m away
    swings through 180 deg for a few millimetres of lateral error.
    """
    duck = np.asarray(duck_xy, dtype=np.float64)[:2]
    target = np.asarray(target_xy, dtype=np.float64)[:2]
    span = target - duck
    distance = float(np.linalg.norm(span))
    if distance <= lookahead_m:
        return target
    return duck + span / distance * lookahead_m


def circuit_length_m() -> float:
    """Total length of the patrol circuit, MEASURED from its own geometry."""
    corners = [HOME.position] + [c.position for c in CHECKPOINTS] \
        + [HOME.position]
    return float(sum(np.linalg.norm(b - a)
                     for a, b in zip(corners, corners[1:])))


def corner_turns_deg() -> list[float]:
    """The signed heading change at each checkpoint, walking the circuit.

    Reported rather than assumed so a test can require every corner to be a LEFT
    turn - the circuit runs counter-clockwise into the policy's weak yaw sign on
    purpose - and to be within what the MEASURED yaw ceiling can carry.
    """
    corners = [HOME.position] + [c.position for c in CHECKPOINTS] \
        + [HOME.position]
    turns: list[float] = []
    for previous, corner, following in zip(corners, corners[1:], corners[2:]):
        incoming = corner - previous
        outgoing = following - corner
        incoming = incoming / max(float(np.linalg.norm(incoming)), 1e-9)
        outgoing = outgoing / max(float(np.linalg.norm(outgoing)), 1e-9)
        cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        dot = float(np.clip(incoming @ outgoing, -1.0, 1.0))
        turns.append(round(float(np.degrees(np.arctan2(cross, dot))), 3))
    return turns
