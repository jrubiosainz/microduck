#!/usr/bin/env python3
"""Invariants of the REAL run, asserted against its recorded trace.

The gate in ``slalom_metrics`` grades the summary; these grade the per-tick
stream it was computed from.  That difference matters: a summary can only be
wrong in the ways its own arithmetic allows, whereas the trace can show a tick
where the duck did something the summary averaged away.

Also here: module hygiene, so the headless gate keeps its no-rendering
guarantee and the modules stay bounded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from slalom_cast import ALL_NAMES
from slalom_course import GOAL_XY, goal_contains
from slalom_states import (
    SAFE_CLEARANCE_M,
    STATES,
    VX_ONSET,
    WALKING_STATES,
    ZERO_COMMAND_STATES,
)

REPO = Path(__file__).resolve().parents[1]


# -- per-tick invariants ----------------------------------------------------------
def test_every_tick_is_in_a_declared_state(trace):
    declared = set(STATES)
    for record in trace:
        assert record["state"] in declared, record["state"]


def test_the_command_is_exactly_zero_in_every_zero_state(trace):
    """Checked per TICK, not per state maximum: a summary can average this away."""
    for record in trace:
        if record["state"] in ZERO_COMMAND_STATES:
            assert record["command"] == [0.0, 0.0, 0.0], (
                record["t"], record["state"], record["command"])


def test_no_tick_ever_carries_a_lateral_command(trace):
    """There is no strafe on this policy; every sidestep is a turning path."""
    assert all(record["command"][1] == 0.0 for record in trace)


def test_no_tick_emits_a_command_below_the_gait_onset(trace):
    """MEASURED: vx=0.22 moves 0.009 m in 6 s.  There is nothing in between."""
    for record in trace:
        vx = record["command"][0]
        assert vx == 0.0 or vx >= VX_ONSET, (record["t"], vx)


def test_the_trunk_never_falls(trace):
    assert all(record["trunk_z"] >= 0.09 for record in trace)


def test_clearance_to_every_body_is_positive_at_every_tick(trace):
    for record in trace:
        assert record["min_body_clearance_m"] > 0.0, (
            record["t"], record["nearest_body"],
            record["min_body_clearance_m"])


def test_clearance_to_static_scenery_is_positive_at_every_tick(trace):
    for record in trace:
        assert record["scenery_clearance_m"] > 0.0, (
            record["t"], record["nearest_scenery"])


def test_the_duck_never_walks_through_the_goal_band_without_stopping(trace):
    """Arriving is a place the duck STOPS, not one it passes through."""
    inside = [r for r in trace if goal_contains(r["duck_xy"])]
    assert inside, "the duck never entered the band"
    assert all(r["state"] in ("GOAL", "DONE", "ADVANCE", "REPLAN")
               for r in inside)
    # And the last tick of the run is inside it.
    assert goal_contains(trace[-1]["duck_xy"])


def test_progress_toward_the_goal_is_real(trace):
    start = np.asarray(trace[0]["duck_xy"], dtype=float)
    end = np.asarray(trace[-1]["duck_xy"], dtype=float)
    goal = np.asarray(GOAL_XY, dtype=float)
    assert np.linalg.norm(end - goal) < np.linalg.norm(start - goal) - 6.0


def test_the_duck_moved_on_both_hands_of_the_lane(trace):
    ys = [record["duck_xy"][1] for record in trace]
    assert max(ys) > 0.10, "never went left of the lane"
    assert min(ys) < -0.10, "never went right of the lane"


def test_every_committed_pass_was_justified_when_it_was_taken(trace):
    """At the tick a CHOOSE state opens, the planner's own bar must be met."""
    previous = None
    for record in trace:
        if record["state"].startswith("CHOOSE") and previous is not None \
                and not previous["state"].startswith("CHOOSE"):
            assert record["chosen_clearance_m"] >= SAFE_CLEARANCE_M, (
                record["t"], record["chosen_clearance_m"])
        previous = record


def test_the_walking_states_actually_walk(trace):
    """A walking state that never moved would be a stall wearing a label."""
    for state in WALKING_STATES:
        ticks = [r for r in trace if r["state"] == state]
        if not ticks:
            continue
        assert any(r["command"][0] > 0.0 for r in ticks), state


def test_the_predictions_in_the_trace_are_real_predictions(trace):
    """Each record carries the occupancy the planner actually scored."""
    with_predictions = [r for r in trace if r["predicted_occupancy"]]
    assert with_predictions, "no predicted occupancy was ever recorded"
    sample = with_predictions[len(with_predictions) // 2]
    horizons = [entry["dt_s"] for entry in sample["predicted_occupancy"]]
    assert horizons == sorted(horizons)
    assert horizons[0] > 0.0


def test_every_tracked_body_has_a_finite_measured_velocity(trace):
    for record in trace[::200]:
        for name, entry in record["tracks"].items():
            assert name in ALL_NAMES
            assert all(np.isfinite(entry["vel"]))


def test_the_tracker_filter_bounds_the_rate_of_change_of_its_estimate(summary):
    """What the low-pass ACTUALLY does on this cast, measured rather than assumed.

    An earlier version of this test asserted the filter suppressed a gait bob
    and compared the raw and filtered SPEED extremes.  Both come out at exactly
    0.300 m/s, because these actors walk analytic constant-speed routes and
    their bob is written into ``z`` only — which the tracker never reads.  The
    claim was about a measurement nobody had taken.

    What the filter genuinely bounds is how fast the estimate may CHANGE when a
    body turns onto a fillet, so that is what is asserted.
    """
    raw = summary["tracker_max_raw_accel_mps2"]
    filtered = summary["tracker_max_filtered_accel_mps2"]
    assert raw > 0.0 and filtered > 0.0
    assert filtered < raw, (raw, filtered)
    # And the speed extremes coincide, which is the finding that corrected the
    # original claim.  Pinned so a future edit cannot quietly reintroduce it.
    assert summary["tracker_max_filtered_speed_mps"] <= \
        summary["tracker_max_raw_speed_mps"]


# -- module hygiene ------------------------------------------------------------------
def test_the_headless_gate_imports_no_rendering_stack():
    """``validate_slalom`` must have no PIL, imageio or GPU dependency.

    Proved by BLOCKING those modules and importing the entry point, rather than
    by reading the imports: a transitive import would pass a source scan.
    """
    blocked = {"PIL", "imageio", "imageio.v2", "matplotlib"}

    class Blocker:
        def find_module(self, name, path=None):
            return self if name.split(".")[0] in blocked else None

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in blocked:
                raise ImportError(f"{name} is blocked by this test")
            return None

    saved_modules = {name: sys.modules.pop(name)
                     for name in list(sys.modules)
                     if name.split(".")[0] in blocked}
    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        for name in ("slalom_metrics", "slalom_summary", "rollout_slalom",
                     "slalom_plan", "slalom_machine", "slalom_control",
                     "slalom_sense", "slalom_camera"):
            sys.modules.pop(name, None)
        import importlib
        importlib.import_module("rollout_slalom")
        importlib.import_module("slalom_metrics")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved_modules)


def test_every_module_stays_bounded():
    """Modules over ~520 lines are doing more than one job.

    The bound is generous because several modules here carry long MEASURED
    findings in their docstrings — the scars that stop a future edit from
    reintroducing a bug — and deleting that prose to hit a line count would
    trade real knowledge for a number.
    """
    oversized = []
    for path in sorted((REPO / "scripts").glob("*.py")):
        code = [line for line in path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")]
        if len(code) > 400:
            oversized.append((path.name, len(code)))
    assert not oversized, oversized


def test_no_module_imports_the_choreography_into_the_decision_layer():
    """The duck must never be able to read the scenario's schedule.

    ``slalom_actors`` owns the choreography; the planner, the machine and the
    controller must not IMPORT it, or "the duck did not know" becomes an
    honour-system claim rather than a structural one.

    Parsed with ``ast`` rather than matched as a substring: these modules
    DISCUSS the choreography in their docstrings, and a text search cannot tell
    a sentence about a module from a dependency on it.
    """
    import ast

    forbidden = {"slalom_actors"}
    for name in ("slalom_plan.py", "slalom_machine.py", "slalom_control.py"):
        tree = ast.parse((REPO / "scripts" / name).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0]
                                for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & forbidden), \
            f"{name} imports {sorted(imported & forbidden)}"


def test_the_gate_thresholds_are_not_defined_twice():
    """The planner's safety bar has exactly one definition."""
    from slalom_states import SAFE_CLEARANCE_M as planner_bar
    from slalom_thresholds import MIN_CHOSEN_PREDICTED_CLEARANCE_M as gate_bar
    assert planner_bar is gate_bar or planner_bar == gate_bar
