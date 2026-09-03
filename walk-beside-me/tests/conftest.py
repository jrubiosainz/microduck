#!/usr/bin/env python3
"""Import path, and the synthetic rollout every gate counterexample mutates.

A gate that cannot fail is decoration.  Each counterexample in
``test_gate_counterexamples`` takes the :func:`baseline` fixture — which passes
every gate — and breaks exactly one invariant.  Without a baseline that provably
passes, a failing mutation would prove nothing: the fixture itself might simply
be malformed, so ``test_the_baseline_passes_every_gate`` is the first test in
that module and the rest depend on it.

The fixture is hand-built rather than produced by perturbing a real rollout.  A
real rollout that fails one gate usually fails several, which isolates nothing.
It does, however, carry the REAL guardian route object, because the bend gates
grade records against the route's own ``corner_report`` and a fake route would
let the bend windows drift away from the ones the behavior is actually measured
on.

The record stream is sampled every 0.1 s rather than every control tick: the
per-state step counters ``summarize`` reads are attributes on the rollout, not
derived from the record count, so a coarse stream keeps the fixture cheap while
still putting enough records inside each bend window for the 1.0 s occupancy
requirement to be met honestly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tools"))

from beside_actors import ROUTES  # noqa: E402
from beside_metrics import UPSTREAM_POLICY_SHA, gates, summarize  # noqa: E402

RECORD_DT = 0.1
CTRL_DT = 0.02


class FakeVerdict:
    """The surface ``SideVerdict.as_record`` presents, without the geometry."""

    def __init__(self, side: int, usable: bool, cause: str = "",
                 detail: str = "", static_gap: float = 1.0,
                 person_gap: float = 2.0):
        self.side = side
        self.usable = usable
        self.cause = cause
        self.detail = detail
        self.static_gap_m = static_gap
        self.person_gap_m = person_gap

    def as_record(self) -> dict:
        return {
            "side": self.side, "usable": bool(self.usable),
            "static_gap_m": round(self.static_gap_m, 4),
            "static_name": "hedge_s" if self.detail == "" else self.detail,
            "person_gap_m": round(self.person_gap_m, 4),
            "person_name": "iris", "person_dt_s": 3.0,
            "cause": self.cause, "detail": self.detail,
        }


class FakeMachine:
    """Exactly the machine surface ``summarize`` reads."""

    def __init__(self):
        self.guardian = "nadia"
        self.joined = True
        self.timeouts: list[str] = []
        self.transitions = [
            {"t": 0.0, "from": "ACQUIRE", "to": "JOIN_SIDE", "side": "left",
             "reason": "right blocked by static:hedge_s"},
            {"t": 6.26, "from": "JOIN_SIDE", "to": "BESIDE_LEFT",
             "side": "left", "reason": "formation established"},
            {"t": 8.86, "from": "BESIDE_LEFT", "to": "SIDE_BLOCKED",
             "cause": "static", "detail": "kiosk", "blocked_for_s": 1.0},
            {"t": 8.88, "from": "SIDE_BLOCKED", "to": "FALL_BACK",
             "to_side": "right", "reason": "dropping behind to cross"},
            {"t": 13.32, "from": "FALL_BACK", "to": "CROSS_BEHIND",
             "longitudinal_m": -0.6215, "reason": "clear astern; crossing"},
            {"t": 17.84, "from": "CROSS_BEHIND", "to": "JOIN_OTHER_SIDE",
             "side": "right", "reason": "crossed astern; closing"},
            {"t": 29.36, "from": "JOIN_OTHER_SIDE", "to": "BESIDE_RIGHT",
             "side": "right", "reason": "formation re-established"},
        ]
        self.decisions = [
            {"t": 0.0, "kind": "initial", "side": 1, "side_name": "left",
             "reason": "right blocked by static:hedge_s",
             "left": FakeVerdict(1, True).as_record(),
             "right": FakeVerdict(-1, False, "static", "hedge_s",
                                  -0.09).as_record()},
            {"t": 8.86, "kind": "blocked", "side": -1, "side_name": "right",
             "reason": "left blocked by static:kiosk",
             "left": FakeVerdict(1, False, "static", "kiosk", 0.1082).as_record(),
             "right": FakeVerdict(-1, True, static_gap=0.27).as_record()},
        ]
        self.switches = [{
            "index": 0, "from_side": "left", "to_side": "right",
            "blocked_at_s": 8.86, "cause": "static", "detail": "kiosk",
            "blocked_for_s": 1.0, "far_clear_for_s": 1.0,
            "static_gap_m": 0.1082, "person_gap_m": 1.0028,
            "fell_back_at_s": 13.32, "longitudinal_at_cross_m": -0.6215,
            "crossed_at_s": 17.84, "joined_at_s": 29.36, "duration_s": 20.5,
        }]


class FakeRollout:
    """The smallest object ``summarize`` accepts, built to pass every gate."""

    def __init__(self, records):
        self.seconds = 86.0
        self.dt = CTRL_DT
        self.records = records
        self.machine = FakeMachine()
        self.policy_sha256 = UPSTREAM_POLICY_SHA

        # physics
        self.fallen_steps = 0
        self.contact_steps = 0
        self.min_trunk_z = 0.11113
        self.path_m = 16.1122
        self.min_person_clearance = 0.2846
        self.min_person_name = "nadia"
        self.min_guardian_clearance = 0.2846
        self.min_scenery_clearance = 0.004
        self.min_scenery_geom = "obs_hedge_s"
        self.duck_radius = 0.1162
        self.duck_exact_radius = 0.0827
        self.adult_half_extent = 0.1155

        # states
        self.state_steps = {
            "JOIN_SIDE": 313, "BESIDE_LEFT": 130, "SIDE_BLOCKED": 1,
            "FALL_BACK": 222, "CROSS_BEHIND": 226, "JOIN_OTHER_SIDE": 576,
            "BESIDE_RIGHT": 2832}

        # formation
        self.beside_steps = 2962
        self.beside_path_m = 10.8483
        self.beside_side_steps = {"BESIDE_LEFT": 130, "BESIDE_RIGHT": 2832}
        self.formation_steps = 2962
        self.beside_lateral = [0.4757, 0.5784, 0.7237]
        self.beside_longitudinal = [-0.6444, -0.12, 0.0938]
        self.max_forward_longitudinal = 0.0938
        self.max_forward_during_switch = -0.42

        # switches
        self.switch_path = {0: 3.9514}
        self.switch_start_xy = {0: np.array([-3.05, -1.62])}
        self.switch_end_xy = {0: np.array([-1.10, -3.32])}
        self.switch_lateral_start = {0: 0.7269}
        self.switch_lateral_end = {0: -0.6276}
        self.switch_min_longitudinal = {0: -1.1511}
        self.switch_max_longitudinal = {0: -0.42}
        self.switch_min_clearance = {0: 0.5628}

        # visibility
        self.visible_steps = 4300
        self.los_steps = 4300
        self.visible_with_los = 4300
        self.blocked_by: dict[str, int] = {}

    @property
    def guardian_route(self):
        return ROUTES["nadia"]


def make_record(t: float, state: str, *, side, lateral: float,
                longitudinal: float, path_m: float, duck_xy,
                left_usable: bool = True, right_usable: bool = True,
                trunk_z: float = 0.116) -> dict:
    """One record with exactly the keys ``summarize`` and the gates read."""
    return {
        "t": round(t, 4),
        "state": state,
        "side": side,
        "side_name": None if side is None else ("left" if side == 1 else "right"),
        "lateral_m": round(lateral, 4),
        "lateral_abs_m": round(abs(lateral), 4),
        "longitudinal_m": round(longitudinal, 4),
        "path_m": round(path_m, 4),
        "duck_xy": [round(float(duck_xy[0]), 4), round(float(duck_xy[1]), 4)],
        "trunk_z": trunk_z,
        "verdict_left": FakeVerdict(
            1, left_usable, "" if left_usable else "static",
            "" if left_usable else "kiosk").as_record(),
        "verdict_right": FakeVerdict(
            -1, right_usable, "" if right_usable else "static",
            "" if right_usable else "hedge_s").as_record(),
    }


# The phases of the reference run, as (state, side, lateral, longitudinal,
# left_usable, right_usable) over [start, end) in seconds.  These mirror the
# measured 86 s rollout closely enough that every gate is exercised on the same
# shape of evidence the real one produces.
PHASES = (
    (0.0, 6.3, "JOIN_SIDE", 1, 0.30, -0.45, True, False),
    (6.3, 8.9, "BESIDE_LEFT", 1, 0.58, -0.12, True, True),
    (8.9, 9.0, "SIDE_BLOCKED", 1, 0.72, -0.20, False, True),
    (9.0, 13.4, "FALL_BACK", 1, 0.70, -0.50, False, True),
    (13.4, 17.9, "CROSS_BEHIND", 1, 0.20, -0.95, False, True),
    (17.9, 29.4, "JOIN_OTHER_SIDE", -1, -0.45, -0.60, False, True),
    (29.4, 86.0, "BESIDE_RIGHT", -1, -0.58, -0.12, True, True),
)


def build_records() -> list[dict]:
    """The reference record stream: one sample every ``RECORD_DT`` seconds."""
    records: list[dict] = []
    path = 0.0
    x, y = -4.65, -2.72
    for start, end, state, side, lateral, longitudinal, left, right in PHASES:
        steps = int(round((end - start) / RECORD_DT))
        for index in range(steps):
            t = start + index * RECORD_DT
            path += 0.019
            x += 0.017
            y += 0.008
            records.append(make_record(
                t, state, side=side, lateral=lateral,
                longitudinal=longitudinal, path_m=path, duck_xy=(x, y),
                left_usable=left, right_usable=right))
    return records


@pytest.fixture
def baseline() -> FakeRollout:
    """A synthetic rollout that passes every acceptance gate."""
    return FakeRollout(build_records())


def gate_map(rollout) -> dict[str, bool]:
    """Every gate label mapped to whether it passed, for one rollout."""
    return {label: ok for label, ok, _ in gates(summarize(rollout))}


def failing(rollout) -> set[str]:
    """The labels of the gates that FAILED, for one rollout."""
    return {label for label, ok in gate_map(rollout).items() if not ok}


def only_failure(rollout, fragment: str) -> None:
    """Assert exactly one gate failed and its label contains ``fragment``."""
    failed = failing(rollout)
    assert len(failed) == 1, f"expected one failure, got {sorted(failed)}"
    label = failed.pop()
    assert fragment in label, f"expected {fragment!r} in {label!r}"
