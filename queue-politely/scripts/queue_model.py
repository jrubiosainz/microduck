#!/usr/bin/env python3
"""Reading a queue: who is in it, in what order, and which gaps are legitimate.

This module contains no physics, no rendering and no MuJoCo.  It turns a set of
world positions into an ORDER, a TAIL and a list of judged gaps, and it is
unit-testable on hand-built inputs precisely because it is pure.

WHAT IS PERCEIVED AND WHAT IS GIVEN
-----------------------------------
Given (simulator semantic proxy): each person's world position, and the queue
path itself.  There is no detector, no tracker and no RGB classification
anywhere in this behavior; ``queue_metrics`` states that as a limitation rather
than burying it.

Derived, and this is the behavior: membership, order, tail, and the verdict on
every candidate standing place.  In particular the duck is NEVER told anybody's
place in line.  It receives positions and works the order out by projecting
each one onto the explicit curved world-space queue path and sorting by ARC
LENGTH.

WHY ARC LENGTH AND NOT A COORDINATE
------------------------------------
Because a queue that bends breaks every coordinate heuristic, and this one
bends through 180 deg.  ``queue_path`` documents the measured failure: ordering
by distance from the counter names the WRONG tail, and ordering by -x names a
DIFFERENT wrong tail.  Both are computed alongside the real answer on every
tick, so the HUD and the metrics can show the difference rather than assert it.

Membership is a cross-track test, which is the other thing a coordinate reading
cannot express: somebody standing beside the rope is not at the back of the
queue, they are not in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from queue_geometry import (
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
    STANDOFF_TARGET_M,
    classify_gap,
    enumerate_gaps,
)
from queue_path import PATH, QUEUE_BAND_M, naive_orders

# A person past this arc length is behind the duck's own entry point and is not
# treated as part of the line to join; nobody in this scenario ever is, but the
# reading should not depend on that.
MAX_QUEUE_ARC_M = 4.60


@dataclass(frozen=True)
class QueueMember:
    name: str
    arc_m: float
    cross_m: float
    off_path_m: float
    xy: tuple[float, float]


@dataclass
class QueueReading:
    """One tick's complete reading of the queue.

    ``order`` is head-first.  ``tail`` is the last person in line.  ``excluded``
    records everybody who was seen but is not a queue member, with the measured
    off-path distance that excluded them, so a bystander's rejection is a number
    rather than a name.
    """

    order: list[str]
    tail: str | None
    members: dict[str, QueueMember]
    excluded: dict[str, float]
    naive: dict[str, list[str]] = field(default_factory=dict)

    @property
    def tail_arc(self) -> float:
        return self.members[self.tail].arc_m if self.tail else 0.0

    def arcs(self) -> dict[str, float]:
        return {name: member.arc_m for name, member in self.members.items()}

    def naive_tail(self, key: str) -> str | None:
        names = self.naive.get(key) or []
        return names[-1] if names else None

    def as_record(self) -> dict:
        return {
            "order": list(self.order),
            "tail": self.tail,
            "arcs_m": {n: round(m.arc_m, 4) for n, m in self.members.items()},
            "excluded_off_path_m": {
                n: round(v, 4) for n, v in self.excluded.items()},
            "naive_orders": {k: list(v) for k, v in self.naive.items()},
            "naive_tails": {k: (v[-1] if v else None)
                            for k, v in self.naive.items()},
        }


def read_queue(positions: dict[str, tuple[float, float]],
               *, band_m: float = QUEUE_BAND_M,
               exclude: tuple[str, ...] = ()) -> QueueReading:
    """Project every person onto the queue path and sort by arc length.

    ``exclude`` removes bodies that are not candidates at all - the duck
    itself, and anybody who has already been served and walked away.  Everybody
    else is judged purely on geometry: within ``band_m`` of the path is in the
    queue, beyond it is not.
    """
    members: dict[str, QueueMember] = {}
    excluded: dict[str, float] = {}
    for name, xy in positions.items():
        if name in exclude:
            continue
        arc, cross, distance = PATH.project(xy)
        if distance > band_m or arc > MAX_QUEUE_ARC_M:
            excluded[name] = float(distance)
            continue
        members[name] = QueueMember(
            name=name, arc_m=float(arc), cross_m=float(cross),
            off_path_m=float(distance),
            xy=(float(xy[0]), float(xy[1])))

    order = sorted(members, key=lambda n: members[n].arc_m)
    naive = naive_orders({n: members[n].xy for n in members})
    return QueueReading(
        order=order, tail=(order[-1] if order else None),
        members=members, excluded=excluded, naive=naive)


def judge_gaps(reading: QueueReading, adult_half_extent_m: float) -> list:
    """Every candidate standing place, judged.  The refusals are the behavior."""
    if not reading.order:
        return []
    gaps = enumerate_gaps(reading.order, reading.arcs(), adult_half_extent_m)
    # Re-derive each verdict from the gap's own structure, so a bookkeeping slip
    # in the enumeration cannot quietly admit a cut-in.
    for gap in gaps:
        if classify_gap(gap) != gap.verdict:
            raise RuntimeError(
                f"gap {gap.name!r} enumerated as {gap.verdict!r} but classifies "
                f"as {classify_gap(gap)!r}")
    return gaps


def rejected_available_gaps(gaps: list) -> list:
    """Refused candidates the duck could PHYSICALLY have taken.

    The gate counts these, not bare refusals.  Refusing a gap too narrow to
    stand in demonstrates nothing about queueing; refusing one that fits is the
    whole claim.
    """
    return [gap for gap in gaps if gap.verdict == "reject" and gap.fits]


def accepted_gap(gaps: list):
    for gap in gaps:
        if gap.accepted:
            return gap
    return None


def standoff_of(duck_arc: float, predecessor_arc: float) -> float:
    """Arc-length separation from the duck to the person in front of it."""
    return float(duck_arc - predecessor_arc)


def standoff_ok(gap_m: float) -> bool:
    return STANDOFF_MIN_M <= gap_m <= STANDOFF_MAX_M


def target_arc_behind(predecessor_arc: float) -> float:
    return float(predecessor_arc + STANDOFF_TARGET_M)


def order_is_correct(inferred: list[str], truth: list[str]) -> bool:
    return list(inferred) == list(truth)


def overtaking_violations(duck_arc: float, arcs: dict[str, float],
                          order: list[str]) -> list[str]:
    """Anybody still in line whom the duck has got in FRONT of.

    The honest no-overtaking test, and it is a test about arc length rather
    than about distance: the duck must remain behind every queue member for the
    whole rollout, right up until the last of them has been served and left.
    A duck that cut the corner of the fold might be nearer the counter in a
    straight line while still being behind in the queue, and a duck that took
    the straggler's gap would be ahead in the queue while looking similar on a
    range plot.  Only arc length distinguishes them.
    """
    return [name for name in order
            if name in arcs and duck_arc < arcs[name]]
