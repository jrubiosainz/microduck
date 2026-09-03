#!/usr/bin/env python3
"""Synthetic counterexamples: proof that every acceptance gate can FAIL.

A gate that cannot fail is decoration.  Each test here builds a fake rollout
that satisfies the behavior in every respect except one, mutates exactly that
one thing, and requires the corresponding gate to go red — and only that gate,
where the invariants are independent.

The fakes are deliberately hand-built rather than produced by perturbing a real
rollout: a real rollout that fails one gate usually fails several, which would
not isolate anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from corridor import (  # noqa: E402
    ALCOVE_BY_NAME,
    CLEAR_ABS_Y,
    DESTINATION_X,
    REJOIN_TOLERANCE_M,
    START_X,
)
from encounter import CLEAR_RANGE_M, UNSAFE_PROXIMITY_M  # noqa: E402
from etiquette_metrics import (  # noqa: E402
    FALLEN_Z,
    MIN_OPENING_PROGRESS_M,
    MIN_PULL_OVER_LATERAL_M,
    MIN_RESUME_PROGRESS_M,
    NOMINAL_Z,
    summarize,
)

BAY = ALCOVE_BY_NAME["bay_open"]
FAR = ALCOVE_BY_NAME["bay_far"]


class FakeMachine:
    def __init__(self, cycles, timeouts=None, decisions=None):
        self.cycles = cycles
        self.timeouts = list(timeouts or [])
        self.decisions = list(decisions or [])
        self.no_alcove_events = []


class FakeRollout:
    """A minimal stand-in with exactly the surface ``summarize`` reads."""

    def __init__(self, records, machine, **kwargs):
        self.records = records
        self.machine = machine
        self.seconds = kwargs.get("seconds", 48.0)
        self.dt = 1.0 / 50.0
        self.decimation = 10
        self.duck_radius = 0.1303
        self.duck_exact_radius = 0.0978
        self.duck_lateral_half = 0.0705
        self.adult_lateral_half = 0.1040
        self.adult_exact_radius = 0.1155
        self.path_m = kwargs.get("path_m", 6.0)
        self.forward_progress_m = kwargs.get("forward_progress_m", 4.0)
        self.max_x = max((r["duck_xy"][0] for r in records), default=0.0)
        self.min_wall_clearance = kwargs.get("min_wall_clearance", 0.05)
        self.min_wall_geom = "wall_plus_0"
        self.min_person_clearance = kwargs.get("min_person_clearance", 0.20)
        self.wall_geoms = tuple(f"wall_{i}" for i in range(21))
        self.transitions = []
        self.pull_over_path = kwargs.get("pull_over_path", {1: 1.0, 2: 1.0})
        self.rejoin_path = kwargs.get("rejoin_path", {1: 0.5, 2: 0.5})
        self.yield_tracking = kwargs.get(
            "yield_tracking", {1: [True] * 200, 2: [True] * 200})
        self.yield_tracking_in_view = kwargs.get(
            "yield_tracking_in_view", {1: [True] * 180, 2: [True] * 180})
        self.yield_command_max = kwargs.get(
            "yield_command_max", {1: 0.0, 2: 0.0})
        self.yield_person_side = kwargs.get(
            "yield_person_side", {1: [1.2, -1.2], 2: [1.2, -1.2]})
        self.detect_proximity = kwargs.get(
            "detect_proximity", {1: 1.5, 2: 1.5})


def _record(t, state, x, y, cycle=1, command=(0.0, 0.0, 0.0), **kwargs):
    intrusion = kwargs.get(
        "passage_intrusion_m",
        -0.02 if abs(y) >= CLEAR_ABS_Y else 0.26)
    return {
        "t": t, "state": state, "state_elapsed_s": 0.1, "cycle": cycle,
        "command": list(command),
        "duck_xy": [x, y],
        "duck_yaw_deg": 0.0,
        "trunk_z_m": kwargs.get("trunk_z_m", NOMINAL_Z),
        "min_trunk_z_m": kwargs.get("min_trunk_z_m", 0.110),
        "clears_passage": kwargs.get("clears_passage", abs(y) >= CLEAR_ABS_Y),
        "passage_intrusion_m": intrusion,
        "wall_clearance_geometric_m": 0.05,
        "wall_clearance_m": kwargs.get("wall_clearance_m", 0.05),
        "wall_nearest_geom": "wall_plus_0",
        "at_destination": kwargs.get("at_destination", x >= DESTINATION_X),
        "destination_remaining_m": max(0.0, DESTINATION_X - x),
        "target_alcove": kwargs.get("target_alcove"),
        "target_park_y": None,
        "tracked_person": kwargs.get("tracked_person"),
        "tracked_visible": True,
        "tracked_fraction": 1.0,
        "visible_people": [],
        "view_yaw_deg": 0.0, "gaze_yaw_deg": 0.0,
        "nearest_person": "chen",
        "nearest_clearance_m": kwargs.get("nearest_clearance_m", 0.5),
        "person_clearances": {"chen": 0.5, "diaz": 0.9},
        "person_xy": {"chen": [3.0, 0.02], "diaz": [6.0, -0.02]},
        "person_moving": {"chen": True, "diaz": True},
        "soonest_person": "chen",
        "soonest_time_to_meet_s": 5.0,
        "soonest_range_m": 2.0,
        "soonest_counterfactual_m": -0.15,
        "alcove_scores": [],
        "path_m": 1.0,
        "completed_cycles": cycle - 1,
    }


def _decision(selected="bay_open"):
    """A decision record with two clearance rejections and one viable pick."""
    def entry(name, clears, reachable, behind, viable):
        return {
            "alcove": name, "side": -1, "center_x": 0.0,
            "usable_outer_y": 0.59, "max_trunk_abs_y": 0.46,
            "park_y": -0.397, "clears_passage": clears,
            "clearance_headroom_m": 0.06, "reachable": reachable,
            "behind": behind, "travel_time_s": 5.0,
            "time_available_s": 9.0, "time_margin_s": 2.0,
            "viable": viable, "rejected_because": [],
        }
    return {
        "encounter": {},
        "considered": 4,
        "selected": entry(selected, True, True, False, True),
        "candidates": [
            entry("bay_shallow", False, True, False, False),
            entry("bay_crates", False, True, False, False),
            entry(selected, True, True, False, True),
            entry("bay_far", True, False, False, False),
        ],
        "rejected": [],
        "viable_count": 1,
    }


def _cycle(index=1, person="chen", **overrides):
    entry = {
        "index": index,
        "person": person,
        "detected_at_s": 5.6,
        "detected_x": -1.17,
        "detected_y": 0.0,
        "detect_range_m": 4.8,
        "detect_time_to_meet_s": 8.9,
        "counterfactual_clearance_m": -0.155,
        "head_on": True,
        "adult_direction": -1.0,
        "predicted_meet_x": 0.4,
        "decision": _decision(),
        "selected_alcove": "bay_open",
        "selected_park_y": BAY.park_y,
        "selected_margin_s": 2.0,
        "alcoves_considered": 4,
        "alcoves_rejected": ["bay_shallow", "bay_crates", "bay_far"],
        "alcoves_viable": ["bay_open"],
        "pull_over_started_s": 6.7,
        "pull_over_start_xy": [-1.02, 0.0],
        "pull_over_duration_s": 6.1,
        "park_xy": [BAY.park_x, BAY.park_y],
        "yield_started_s": 12.8,
        "yield_duration_s": 4.3,
        "cleared_at_s": 17.1,
        "clear_range_m": 0.61,
        "rejoin_started_s": 17.6,
        "rejoin_duration_s": 2.4,
        "rejoined_at_s": 20.0,
        "rejoin_xy": [-0.30, -0.05],
        "completed_at_s": 20.0,
    }
    entry.update(overrides)
    return entry


def build(**mutations):
    """A complete, passing two-cycle fake rollout, mutated by keyword."""
    records = []
    t = 0.0

    def add(state, x, y, cycle, command=(0.0, 0.0, 0.0), n=10, **kw):
        nonlocal t
        for _ in range(n):
            records.append(_record(t, state, x, y, cycle, command, **kw))
            t += 0.02

    opening = mutations.get("opening_progress", 0.8)
    add("CRUISE", START_X, 0.0, 1, (0.36, 0.0, 0.0), n=5)
    add("CRUISE", START_X + opening, 0.0, 1, (0.36, 0.0, 0.0), n=5)
    for index, (bay, person) in enumerate(
            ((BAY, "chen"), (FAR, "diaz")), start=1):
        park_y = mutations.get(f"park_y_{index}", bay.park_y)
        add("DETECT", -1.1, 0.0, index, (0.36, 0.0, 0.0),
            target_alcove=None, tracked_person=person,
            nearest_clearance_m=mutations.get(
                "detect_clearance", 1.5))
        add("SELECT_ALCOVE", -1.0, 0.0, index, (0.36, 0.0, 0.0),
            tracked_person=person)
        add("PULL_OVER", bay.park_x, 0.0, index, (0.28, -0.6, 0.0),
            target_alcove=bay.name, tracked_person=person)
        add("PULL_OVER", bay.park_x, park_y, index, (0.0, -0.6, 0.0),
            target_alcove=bay.name, tracked_person=person)
        add("YIELD", bay.park_x, park_y, index,
            mutations.get("yield_command", (0.0, 0.0, 0.0)),
            n=20, target_alcove=bay.name, tracked_person=person,
            wall_clearance_m=mutations.get("wall_clearance", 0.05),
            nearest_clearance_m=mutations.get("person_clearance", 0.4))
        add("CLEAR", bay.park_x, park_y, index, target_alcove=bay.name,
            tracked_person=person)
        # The rejoin starts at the park point and ends on the centreline, so
        # its measured lateral travel is the full depth of the recess.
        add("REJOIN", bay.park_x, park_y, index, (0.0, 0.6, 0.0),
            target_alcove=bay.name, n=2)
        add("REJOIN", bay.park_x, park_y * 0.5, index, (0.0, 0.6, 0.0),
            target_alcove=bay.name)
        add("REJOIN", bay.park_x, mutations.get(f"rejoin_y_{index}", -0.05),
            index, (0.0, 0.6, 0.0), target_alcove=bay.name)
        # RESUME must carry the whole post-rejoin walk, because the gate
        # measures forward progress from the LAST rejoin step to the end of
        # the rollout.
        add("RESUME", bay.park_x + 0.2, -0.02, index, (0.36, 0.0, 0.0))
    resume = mutations.get("resume_progress", 0.7)
    final_x = mutations.get("final_x", DESTINATION_X + 0.01)
    add("RESUME", final_x - resume, -0.02, 3, (0.36, 0.0, 0.0), n=5)
    add("DONE", final_x, -0.02, 3, n=10,
        trunk_z_m=mutations.get("final_z", NOMINAL_Z),
        at_destination=mutations.get("at_destination", final_x >= DESTINATION_X))

    cycles = mutations.get("cycles") or [
        _cycle(1, "chen", **mutations.get("cycle_1", {})),
        _cycle(2, "diaz", selected_alcove="bay_far",
               selected_park_y=FAR.park_y, rejoin_xy=[FAR.park_x, 0.04],
               **mutations.get("cycle_2", {})),
    ]
    machine = FakeMachine(cycles, timeouts=mutations.get("timeouts"))
    kwargs = {k: v for k, v in mutations.items() if k in {
        "yield_tracking", "yield_tracking_in_view", "yield_command_max",
        "yield_person_side", "detect_proximity", "pull_over_path",
        "rejoin_path", "min_wall_clearance", "min_person_clearance"}}
    return FakeRollout(records, machine, **kwargs)


def gates(**mutations):
    return summarize(build(**mutations))["gates"]


# ------------------------------------------------------------- the baseline
def test_the_unmutated_fake_passes_every_gate():
    """Without this, every counterexample below could be vacuous."""
    summary = summarize(build())
    failed = [name for name, ok in summary["gates"].items() if not ok]
    assert not failed, f"the baseline fake must pass everything, failed: {failed}"


# ------------------------------------------------------------ counterexamples
def test_a_missing_state_fails_state_order():
    rollout = build()
    rollout.records = [r for r in rollout.records if r["state"] != "CLEAR"]
    assert not summarize(rollout)["gates"]["state_order"]


def test_a_reordered_state_fails_state_order():
    rollout = build()
    for record in rollout.records:
        if record["state"] == "YIELD":
            record["state"] = "REJOIN"
    assert not summarize(rollout)["gates"]["state_order"]


def test_no_opening_walk_fails_real_forward_start():
    assert not gates(opening_progress=0.05)["real_forward_start"]
    assert gates(opening_progress=MIN_OPENING_PROGRESS_M + 0.1)[
        "real_forward_start"]


def test_late_detection_fails_detects_before_unsafe():
    close = {1: UNSAFE_PROXIMITY_M - 0.1, 2: UNSAFE_PROXIMITY_M - 0.1}
    assert not gates(detect_proximity=close)["detects_before_unsafe"]


def test_a_safe_counterfactual_fails_counterfactual_recorded():
    """Pulling over for a pass that was fine anyway proves nothing."""
    assert not gates(
        cycle_1={"counterfactual_clearance_m": +0.05},
    )["counterfactual_recorded"]


def test_a_missing_counterfactual_fails_counterfactual_recorded():
    assert not gates(
        cycle_1={"counterfactual_clearance_m": None},
    )["counterfactual_recorded"]


def test_scoring_one_alcove_fails_evaluated_enough():
    assert not gates(cycle_1={"alcoves_considered": 1})["evaluated_enough"]


def test_rejections_only_for_distance_fail_rejected_on_clearance():
    """Refusing bays merely out of reach says nothing about judgement."""
    decision = _decision()
    for candidate in decision["candidates"]:
        candidate["clears_passage"] = True
        candidate["reachable"] = candidate["alcove"] == "bay_open"
    assert not gates(
        cycle_1={"decision": decision},
        cycle_2={"decision": decision},
    )["rejected_on_clearance"]


def test_selecting_an_unusable_bay_fails_selection_is_viable():
    decision = _decision()
    decision["selected"]["clears_passage"] = False
    assert not gates(cycle_1={"decision": decision})["selection_is_viable"]


def test_selecting_an_unreachable_bay_fails_selection_is_viable():
    decision = _decision()
    decision["selected"]["reachable"] = False
    assert not gates(cycle_1={"decision": decision})["selection_is_viable"]


def test_a_token_sidestep_fails_pull_over_moved():
    tiny = MIN_PULL_OVER_LATERAL_M - 0.05
    assert not gates(park_y_1=-tiny)["pull_over_moved"]


def test_zero_pull_over_path_fails_pull_over_moved():
    assert not gates(pull_over_path={1: 0.01, 2: 1.0})["pull_over_moved"]


def test_stopping_inside_the_passage_fails_footprint_cleared():
    rollout = build()
    for record in rollout.records:
        if record["state"] == "YIELD":
            record["clears_passage"] = False
            record["passage_intrusion_m"] = +0.04
    assert not summarize(rollout)["gates"]["footprint_cleared"]


def test_a_nonzero_yield_command_fails_yield_command_zero():
    assert not gates(yield_command=(0.0, 0.02, 0.0))["yield_command_zero"]


def test_a_decaying_yield_command_still_fails():
    """A tail is still a command."""
    rollout = build()
    for index, record in enumerate(rollout.records):
        if record["state"] == "YIELD":
            record["command"] = [0.0, 0.30 * (0.9 ** index), 0.0]
    assert not summarize(rollout)["gates"]["yield_command_zero"]


def test_an_adult_that_never_passes_fails_adult_passed():
    """Offsets that never change sign mean the person turned back."""
    assert not gates(
        yield_person_side={1: [1.2, 0.9], 2: [1.2, 0.9]})["adult_passed"]


def test_rejoining_too_early_fails_no_early_rejoin():
    assert not gates(
        cycle_1={"clear_range_m": CLEAR_RANGE_M - 0.1})["no_early_rejoin"]


def test_stopping_short_of_the_centreline_fails_rejoin_centred():
    assert not gates(
        cycle_1={"rejoin_xy": [BAY.park_x, -(REJOIN_TOLERANCE_M + 0.1)]},
    )["rejoin_centred"]


def test_a_token_rejoin_fails_rejoin_centred():
    """Ending near the centreline is not enough: the duck must have moved."""
    rollout = build()
    for record in rollout.records:
        if record["state"] == "REJOIN":
            record["duck_xy"][1] = -0.05
    assert not summarize(rollout)["gates"]["rejoin_centred"]


def test_not_walking_on_afterwards_fails_resumed_forward():
    """Progress is measured from the last REJOIN step to the end."""
    stalled = build(resume_progress=0.0)
    last = next(index for index in range(len(stalled.records) - 1, -1, -1)
                if stalled.records[index]["state"] == "REJOIN")
    frozen_x = stalled.records[last]["duck_xy"][0]
    for record in stalled.records[last:]:
        record["duck_xy"][0] = frozen_x
    assert not summarize(stalled)["gates"]["resumed_forward"]
    assert gates(resume_progress=MIN_RESUME_PROGRESS_M + 0.2)["resumed_forward"]


def test_stopping_short_fails_reached_destination():
    assert not gates(final_x=DESTINATION_X - 0.5,
                     at_destination=False)["reached_destination"]


def test_touching_a_person_fails_person_clearance_and_no_contacts():
    result = gates(person_clearance=-0.01)
    assert not result["person_clearance"]
    assert not result["no_contacts"]


def test_touching_a_wall_fails_wall_clearance_and_no_contacts():
    result = gates(wall_clearance=-0.01)
    assert not result["wall_clearance"]
    assert not result["no_contacts"]


def test_losing_the_adult_in_the_pip_fails_tracking():
    partial = {1: [True] * 100 + [False] * 100, 2: [True] * 180}
    assert not gates(yield_tracking_in_view=partial)["tracking"]


def test_a_command_below_gait_onset_fails_no_decorative_commands():
    """The HUD would show motion the floor never sees."""
    rollout = build()
    for record in rollout.records:
        if record["state"] == "RESUME":
            record["command"] = [0.10, 0.0, 0.0]
    assert not summarize(rollout)["gates"]["no_decorative_commands"]


def test_a_sub_onset_lateral_command_also_fails():
    rollout = build()
    for record in rollout.records:
        if record["state"] == "PULL_OVER":
            record["command"] = [0.0, -0.10, 0.0]
    assert not summarize(rollout)["gates"]["no_decorative_commands"]


def test_a_fall_fails_no_falls_and_min_trunk_z():
    rollout = build()
    rollout.records[40]["trunk_z_m"] = FALLEN_Z - 0.01
    result = summarize(rollout)["gates"]
    assert not result["no_falls"]
    assert not result["min_trunk_z"]


def test_a_low_final_height_fails_final_trunk_z():
    assert not gates(final_z=NOMINAL_Z - 0.05)["final_trunk_z"]


def test_a_timeout_fails_no_timeouts():
    assert not gates(timeouts=["pull_over_timeout"])["no_timeouts"]


def test_a_nonzero_command_while_clear_fails_still_when_still():
    rollout = build()
    for record in rollout.records:
        if record["state"] == "CLEAR":
            record["command"] = [0.30, 0.0, 0.0]
    assert not summarize(rollout)["gates"]["still_when_still"]


def test_no_cycles_at_all_fails_the_evidence_gates():
    rollout = build()
    rollout.machine = FakeMachine([])
    result = summarize(rollout)["gates"]
    for name in ("counterfactual_recorded", "evaluated_enough",
                 "rejected_on_clearance", "selection_is_viable",
                 "pull_over_moved", "footprint_cleared", "adult_passed",
                 "no_early_rejoin", "rejoin_centred", "tracking"):
        assert not result[name], f"{name} must not pass with zero cycles"
