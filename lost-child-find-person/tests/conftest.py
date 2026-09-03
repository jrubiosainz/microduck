#!/usr/bin/env python3
"""The synthetic rollout fixture every gate counterexample mutates.

A gate that cannot fail is decoration.  Each counterexample in
``test_gate_counterexamples`` takes THIS fixture — which passes all 25 gates —
breaks exactly one invariant, runs the REAL ``summarize``/``gates``, and
requires that one gate to report False.  Without a baseline that provably
passes, a failing mutation would prove nothing: the fixture itself might simply
be malformed, so ``test_the_baseline_fixture_passes_every_gate`` is what makes
the rest of that module meaningful.

The fixture is deliberately NOT a recording of the real rollout.  It is the
smallest object ``summarize`` accepts, with every field set to a value the gate
it feeds would accept, so a mutation isolates one gate instead of tripping
several at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lost_cast import GUARDIAN  # noqa: E402
from lost_memory import GuardianTrail  # noqa: E402

POLICY = ROOT / "onnx" / "alpha_walking.onnx"
SCENE = ROOT / "assets" / "scene_lost_child.xml"
# The stock walking policy this whole behavior is measured against.
UPSTREAM_SHA = "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"

DT = 0.02


def gate_map(results) -> dict[str, bool]:
    """``gates()`` returns (label, ok, evidence) triples; index them by label."""
    return {label: ok for label, ok, _ in results}


def find_gate(results, fragment: str) -> tuple[str, bool, str]:
    """The single gate whose label contains ``fragment``."""
    hits = [r for r in results if fragment in r[0]]
    assert len(hits) == 1, f"{fragment!r} matched {[h[0] for h in hits]}"
    return hits[0]


class FakeMachine:
    def __init__(self):
        self.guardian = GUARDIAN.name
        self.state = "SAFE"
        self.transitions = [{"t": 16.6, "from": "FOLLOW", "to": "LOST"}]
        self.timeouts: list[str] = []
        self.cycles = [
            {"index": 0, "lost_at_s": 16.6, "rejoined_at_s": 34.0,
             "outcome": "rejoined", "duration_s": 17.4,
             "rejections": [{"name": "sofia"}, {"name": "mira"}]},
            {"index": 1, "lost_at_s": 44.0, "rejoined_at_s": 52.0,
             "outcome": "rejoined", "duration_s": 8.0,
             "rejections": [{"name": "faruq"}]},
        ]


class FakeIdentity:
    def __init__(self):
        self.rejections = [
            {"name": "sofia", "score": 0.82, "rejected_at_s": 21.0,
             "reason": "stands 1.60 m, guardian 1.72 m"},
            {"name": "mira", "score": 0.86, "rejected_at_s": 25.0,
             "reason": "wears a cap; guardian does not"},
            {"name": "faruq", "score": 0.58, "rejected_at_s": 47.0,
             "reason": "shirt colour differs from the guardian's teal"},
        ]
        self.accepted = [
            {"name": GUARDIAN.name, "score": 1.0, "accepted_at_s": 33.0},
            {"name": GUARDIAN.name, "score": 1.0, "accepted_at_s": 52.4},
        ]
        self.wrong_accepts: list[dict] = []

    def distinct_rejected(self) -> tuple[str, ...]:
        order: list[str] = []
        for record in self.rejections:
            if record["name"] not in order:
                order.append(record["name"])
        return tuple(order)


def _record(t, state, *, xy, command_peak=0.0, visible=True, path_m=0.0,
            trunk_z=0.116, guardian_range=0.9):
    return {
        "t": round(t, 4), "state": state, "command_peak": command_peak,
        "duck_xy": [float(xy[0]), float(xy[1])],
        "trunk_z": trunk_z, "path_m": path_m,
        "guardian": GUARDIAN.name, "guardian_visible": bool(visible),
        "guardian_range_m": guardian_range,
    }


class FakeRollout:
    """The smallest object ``summarize`` accepts, built to pass every gate."""

    def __init__(self):
        self.seconds = 60.0
        self.dt = DT
        self.machine = FakeMachine()
        self.identity = FakeIdentity()

        # physics
        self.fallen_steps = 0
        self.contact_steps = 0
        self.min_trunk_z = 0.1114
        self.path_m = 6.2767
        self.min_person_clearance = 0.105
        self.min_person_name = "dahl"
        self.min_scenery_clearance = 0.1875
        self.min_scenery_geom = "obs_kiosk"
        self.duck_radius = 0.1162
        self.duck_exact_radius = 0.0827
        self.adult_half_extent = 0.1375

        # per-state accumulators: every stationary state at EXACT zero.
        self.state_command_max = {
            "FOLLOW": 0.46, "REJOIN": 0.46, "LOST": 0.0, "STOP": 0.0,
            "SEARCH_SWEEP": 0.0, "CANDIDATE": 0.0, "REJECT": 0.0,
            "REACQUIRED": 0.0, "SAFE": 0.0}
        self.state_steps = {
            "FOLLOW": 900, "LOST": 1, "STOP": 40, "SEARCH_SWEEP": 500,
            "CANDIDATE": 90, "REJECT": 105, "REACQUIRED": 2, "REJOIN": 729,
            "SAFE": 80}

        # occlusion: one long geometric run behind the kiosk.
        self.occlusion_runs = [
            {"start_s": 16.60, "end_s": 24.06, "duration_s": 7.48,
             "blockers": {"obs_kiosk": 330, "out_of_frustum": 44}, "cycle": 0},
            {"start_s": 44.20, "end_s": 45.40, "duration_s": 1.22,
             "blockers": {"out_of_frustum": 61}, "cycle": 1},
        ]
        self.lookalike_seen = {"faruq": 0.06, "mira": 0.08, "bekele": 0.16,
                               "arun": 14.64, "sofia": 17.44, "costa": 26.48}

        # rejoin bookkeeping, per cycle
        self.rejoin_path = {0: 2.744, 1: 0.432}
        self.rejoin_start_range = {0: 2.729, 1: 1.027}
        self.rejoin_end_range = {0: 0.895, 1: 0.709}
        self.rejoin_start_xy = {0: np.array([1.60, -0.30]),
                                1: np.array([-0.60, 1.20])}
        self.rejoin_end_xy = {0: np.array([-0.20, 0.65]),
                              1: np.array([-0.75, 1.50])}
        self.rejoin_visible = {0: [True] * 605, 1: [True] * 124}
        self.rejoin_visible_with_los = {0: [True] * 605, 1: [True] * 124}
        self.rejoin_min_clearance = {0: 0.31, 1: 0.44}
        self.rejoin_routes = {
            0: {"waypoints": [[1.6, -0.3], [1.26, 0.86], [-0.2, 0.65]],
                "length_m": 2.71, "direct_blocked_by": "kiosk",
                "bends_around": ["kiosk"], "feasible": True,
                "waypoint_count": 3},
            1: {"waypoints": [[-0.6, 1.2], [-0.75, 1.5]], "length_m": 0.43,
                "direct_blocked_by": "", "bends_around": [], "feasible": True,
                "waypoint_count": 2},
        }

        # a real trail, built by the real class
        self.trail = GuardianTrail()
        for step in range(18):
            self.trail.observe(step * 0.5,
                               np.array([2.0 - 0.22 * step, -0.4 + 0.05 * step]))

        self.records = _baseline_records()


def _baseline_records() -> list[dict]:
    """A record stream whose measured quantities satisfy every record-fed gate."""
    records: list[dict] = []
    t = 0.0
    path = 0.0

    # FOLLOW: a real traverse, guardian visible on all but a couple of ticks.
    for step in range(830):
        path += 0.0036
        records.append(_record(
            t, "FOLLOW", xy=(2.15 - 0.0028 * step, -1.90 + 0.0022 * step),
            command_peak=0.46, visible=(step % 200 != 7), path_m=path,
            guardian_range=0.85))
        t += DT
    lost_xy = (2.15 - 0.0028 * 829, -1.90 + 0.0022 * 829)

    # LOST -> STOP -> SEARCH_SWEEP -> CANDIDATE -> REJECT: EXACT zero, standing.
    stationary = ([("LOST", 1), ("STOP", 40), ("SEARCH_SWEEP", 200),
                   ("CANDIDATE", 45), ("REJECT", 35), ("SEARCH_SWEEP", 180),
                   ("CANDIDATE", 45), ("REJECT", 35), ("REACQUIRED", 1)])
    for state, count in stationary:
        for _ in range(count):
            records.append(_record(t, state, xy=lost_xy, command_peak=0.0,
                                   visible=False, path_m=path,
                                   guardian_range=2.729))
            t += DT

    # REJOIN: real progress, guardian visible throughout.
    for step in range(605):
        path += 0.0045
        records.append(_record(
            t, "REJOIN", xy=(lost_xy[0] - 0.0030 * step,
                             lost_xy[1] + 0.0016 * step),
            command_peak=0.46, visible=True, path_m=path, guardian_range=0.895))
        t += DT

    # A second, shallower cycle, then the final standoff.
    for state, count in (("SEARCH_SWEEP", 120), ("CANDIDATE", 45),
                         ("REJECT", 35), ("REACQUIRED", 1)):
        for _ in range(count):
            records.append(_record(t, state, xy=(-0.60, 1.20),
                                   command_peak=0.0, visible=False,
                                   path_m=path, guardian_range=1.027))
            t += DT
    for step in range(124):
        path += 0.0035
        records.append(_record(
            t, "REJOIN", xy=(-0.60 - 0.0012 * step, 1.20 + 0.0024 * step),
            command_peak=0.46, visible=True, path_m=path, guardian_range=0.709))
        t += DT
    for _ in range(80):
        records.append(_record(t, "SAFE", xy=(-0.75, 1.50), command_peak=0.0,
                               visible=True, path_m=path, trunk_z=0.11627,
                               guardian_range=0.7057))
        t += DT
    return records


@pytest.fixture
def baseline() -> FakeRollout:
    """A synthetic rollout that passes every acceptance gate."""
    return FakeRollout()
