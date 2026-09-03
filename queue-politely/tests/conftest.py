#!/usr/bin/env python3
"""The synthetic rollout fixture every gate counterexample mutates.

A gate that cannot fail is decoration, so each counterexample takes THIS
fixture - which passes every gate - and breaks exactly one invariant.  Without a
baseline that provably passes, a failing mutation would prove nothing: the
fixture itself might simply be malformed.

The fixture models departures, because the queue SHRINKS as people are served.
An earlier version kept all five stations populated for the whole rollout, so
once the duck reached the counter it appeared to be in front of five people who
had actually left long before, and ``no_overtaking`` failed on the baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from queue_geometry import STANDOFF_TARGET_M  # noqa: E402
from queue_path import PATH  # noqa: E402
from queue_people import QUEUE, QUEUE_NAMES  # noqa: E402

TRUTH = list(QUEUE_NAMES)
STATIONS = {adult.name: adult.initial_arc for adult in QUEUE}

class FakePolicy:
    def __init__(self, path):
        self.path = path


class FakeMachine:
    def __init__(self):
        self.transitions = []
        self.timeouts = []
        self.accepted_gap = {
            "gap": "behind_tail", "kind": "tail", "ahead": "eriksson",
            "behind": None, "separation_m": STANDOFF_TARGET_M,
            "physically_fits": True, "verdict": "join", "reason": "tail"}
        self.rejected_gaps = []
        self.cycles = []


class FakeRollout:
    """The smallest object ``summarize`` accepts, built to pass every gate."""

    def __init__(self, policy_path):
        self.seconds = 58.0
        self.policy = FakePolicy(policy_path)
        self.machine = FakeMachine()
        self.duck_radius = 0.1303
        self.duck_exact_radius = 0.0978
        self.adult_half_extent = 0.1155
        self.path_m = 6.4
        self.min_person_clearance = 0.21
        self.min_person_name = "okafor"
        self.min_scenery_clearance = 0.10
        self.min_scenery_geom = "post_r_8"
        self.records = []
        self.order_samples = []
        self.stationary_command_max = {
            "OBSERVE_QUEUE": 0.0, "IDENTIFY_TAIL": 0.0, "EVALUATE_GAPS": 0.0,
            "WAIT": 0.0, "AT_COUNTER": 0.0, "DONE": 0.0}
        self.cycle_path = {}
        self.cycle_start_arc = {}
        self.cycle_tracking = {}
        self.cycle_tracking_after_service = {}
        self.cycle_command_max = {}
        self.cycle_cross_track = {}
        self.join_evidence = None
        self.first_reading = None


def _record(t, state, *, arc, cross=0.0, cycle=0, order=None, tail="eriksson",
            predecessor=None, standoff=None, remaining=0, command=(0.0, 0.0, 0.0),
            arcs=None):
    order = TRUTH if order is None else order
    arcs = STATIONS if arcs is None else arcs
    return {
        "t": t, "state": state, "state_elapsed_s": 0.0, "cycle": cycle,
        "command": list(command),
        "duck_xy": list(PATH.point_at(arc)),
        "duck_yaw_deg": 0.0,
        "duck_arc_m": arc, "duck_cross_track_m": cross, "duck_off_path_m": abs(cross),
        "trunk_z_m": 0.116, "min_trunk_z_m": 0.1112,
        "target_arc_m": arc, "predecessor": predecessor,
        "predecessor_arc_m": arcs.get(predecessor) if predecessor else None,
        "standoff_m": standoff, "predecessors_remaining": remaining,
        "inferred_order": list(order), "true_order": list(order),
        "inferred_tail": tail,
        "naive_range_tail": "dubois", "naive_x_tail": "chandra",
        "excluded": {"nakamura": 0.80, "okafor": 0.39},
        "gaps": [], "rejected_gap_names": [],
        "subject_visible": True, "subject_fraction": 1.0,
        "visible_people": list(order), "view_yaw_deg": 0.0, "gaze_yaw_deg": 0.0,
        "nearest_person": "okafor", "nearest_clearance_m": 0.21,
        "person_clearances": {"okafor": 0.21},
        "scenery_clearance_m": 0.10, "scenery_nearest_geom": "post_r_8",
        "person_xy": {n: list(PATH.point_at(a)) for n, a in arcs.items()},
        "person_in_queue": {n: True for n in arcs},
        "person_arc_m": dict(arcs),
        "at_counter": arc <= 0.24, "path_m": 6.4,
        "completed_cycles": cycle,
    }


@pytest.fixture
def baseline(tmp_path):
    """A synthetic rollout that passes every gate."""
    policy = tmp_path / "alpha_walking.onnx"
    policy.write_bytes(b"stub")
    rollout = FakeRollout(policy)

    sequence = [
        ("APPROACH", 4.60, 0.0, 5.0), ("OBSERVE_QUEUE", 4.30, 0.0, 8.0),
        ("IDENTIFY_TAIL", 4.30, 0.0, 10.0), ("EVALUATE_GAPS", 4.30, 0.0, 12.0),
    ]
    records = []
    for state, arc, cross, when in sequence:
        records.append(_record(when, state, arc=arc, cross=cross,
                               remaining=5))
    # JOIN
    for step in range(20):
        records.append(_record(14.0 + step * 0.02, "JOIN", arc=3.60 - step * 0.02,
                               cycle=0, predecessor="eriksson", remaining=5,
                               command=(0.46, 0.0, 0.0)))
    join_arc = STATIONS["eriksson"] + STANDOFF_TARGET_M
    rollout.cycle_path[0] = 1.2
    rollout.cycle_start_arc[0] = 3.60
    rollout.cycle_tracking[0] = [True] * 20
    rollout.machine.cycles.append({
        "kind": "join", "started_s": 14.0, "completed_s": 18.0,
        "duration_s": 4.0, "behind": "eriksson", "target_arc_m": join_arc})
    rollout.join_evidence = {
        "t": 18.0, "duck_arc_m": join_arc, "behind": "eriksson",
        "longitudinal_m": STANDOFF_TARGET_M, "lateral_m": 0.01,
        "in_band": True, "duck_xy": list(PATH.point_at(join_arc))}
    rollout.first_reading = {
        "order": TRUTH, "tail": "eriksson", "truth": TRUTH,
        "naive_tails": {"by_range": "dubois", "by_max_minus_x": "chandra"},
        "gaps": [],
        "rejected_available": [
            {"gap": "beside_counter", "kind": "side", "separation_m": 0.67,
             "physically_fits": True, "verdict": "reject", "reason": "side",
             "ahead": None, "behind": "alvarez"},
            {"gap": "between_dubois_eriksson", "kind": "cut_in",
             "separation_m": 0.90, "physically_fits": True,
             "verdict": "reject", "reason": "cut", "ahead": "dubois",
             "behind": "eriksson"}],
    }
    rollout.machine.rejected_gaps = rollout.first_reading["rejected_available"]

    # Three WAIT -> ADVANCE cycles, then the run to the counter.
    # THE QUEUE SHRINKS AS PEOPLE ARE SERVED.  An earlier fixture kept all five
    # stations populated for the whole rollout, so once the duck reached the
    # counter it was in front of five people who had actually left long before,
    # and ``no_overtaking`` failed on the BASELINE.  The fixture has to model
    # departures for the same reason the rollout does.
    arc = join_arc
    t = 20.0
    for index in range(1, 4):
        remaining_names = TRUTH[index:]
        arcs = {name: STATIONS[name] - 0.55 * index
                for name in remaining_names}
        arcs["eriksson"] = arc - 0.59
        for step in range(10):
            records.append(_record(t + step * 0.02, "WAIT", arc=arc,
                                   cycle=index - 1, predecessor="eriksson",
                                   standoff=0.59, remaining=1,
                                   order=remaining_names, arcs=arcs))
        t += 1.0
        start = arc
        arc -= 0.55
        moved = dict(arcs)
        moved["eriksson"] = arc - 0.59
        for step in range(10):
            records.append(_record(
                t + step * 0.02, "ADVANCE", arc=start - 0.055 * step,
                cross=0.05, cycle=index, predecessor="eriksson",
                standoff=0.59, remaining=1, command=(0.46, 0.0, -0.2),
                order=remaining_names, arcs=moved))
        rollout.cycle_path[index] = 0.74
        rollout.cycle_start_arc[index] = start
        rollout.cycle_tracking[index] = [True] * 10
        rollout.machine.cycles.append({
            "kind": "advance", "started_s": t, "completed_s": t + 3.0,
            "duration_s": 3.0, "behind": "eriksson", "target_arc_m": arc,
            "from_arc_m": start})
        t += 4.0

    # To the counter, after the last service: the queue is now EMPTY.
    start = arc
    for step in range(10):
        records.append(_record(48.0 + step * 0.02, "ADVANCE",
                               arc=start - 0.06 * step, cross=0.04, cycle=4,
                               remaining=0, command=(0.46, 0.0, 0.0),
                               order=[], arcs={}, tail=None))
    rollout.cycle_path[4] = 0.74
    rollout.cycle_start_arc[4] = start
    rollout.machine.cycles.append({
        "kind": "to_counter", "started_s": 48.0, "completed_s": 50.0,
        "duration_s": 2.0, "target_arc_m": 0.0, "from_arc_m": start})
    for step in range(6):
        records.append(_record(50.0 + step * 0.02, "AT_COUNTER", arc=0.05,
                               remaining=0, order=[], arcs={}, tail=None))
    for step in range(6):
        records.append(_record(53.0 + step * 0.02, "DONE", arc=0.05,
                               remaining=0, order=[], arcs={}, tail=None))

    rollout.records = records
    rollout.order_samples = [
        {"t": r["t"], "state": r["state"], "inferred": r["inferred_order"],
         "truth": r["true_order"], "correct": True, "tail": r["inferred_tail"],
         "true_tail": r["inferred_tail"], "tail_correct": True}
        for r in records if r["true_order"]]
    return rollout


