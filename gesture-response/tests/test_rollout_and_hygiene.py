#!/usr/bin/env python3
"""The real run, the scene, and the structural claims about the source itself.

Three kinds of test live here:

* **Hygiene** - the import graph parsed with ``ast``, so "the duck never read
  the choreography" is enforced by the module structure rather than promised in
  a docstring.
* **The scene** - geometry constants pinned against the compiled model, so a
  scene edit that invalidates a measured constant fails here rather than
  silently changing what the gates mean.
* **The run** - the summary of a REAL validation rollout, graded on the same
  numbers the README quotes.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

from gest_arena import FIXTURES, OCCLUDERS, OCCLUDING_HEIGHT_M, inside_area
from gest_cast import ALL_NAMES, INSTRUCTOR, is_instructor
from gest_gesture import MOTION_WINDOW_S
from gest_script import (
    CUES,
    DISTRACTOR_CUES,
    EXPECTED_COMMANDS,
    INSTRUCTOR_CUES,
    REJECTED_CUES,
    ROUTES,
)
from gest_states import (
    CONFIRM_S,
    DUCK_PLANAR_RADIUS,
    FORBIDDEN_STATES,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
    ZERO_COMMAND_STATES,
)
from policy_runtime import (
    ACTION_SCALE,
    FALLEN_TRUNK_Z,
    GYRO_SENSOR,
    NOMINAL_TRUNK_Z,
    OBS_DIM,
)

# The modules that make DECISIONS.  None of them may reach the scenario.
DECISION_MODULES = (
    "gest_machine", "gest_detect", "gest_acquire", "gest_gesture",
    "gest_templates", "gest_commands", "gest_pose", "gest_control",
    "gest_states", "gest_episode", "gest_gates",
)
# The modules that ARE the scenario.  The duck must not be able to read them.
SCENARIO_MODULES = ("gest_actors", "gest_script")


def imports_of(module: str) -> set[str]:
    tree = ast.parse((SCRIPTS / f"{module}.py").read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


# -- hygiene: the duck never reads the choreography --------------------------
@pytest.mark.parametrize("module", DECISION_MODULES)
def test_decision_modules_cannot_reach_the_scenario(module):
    """Parsed with ``ast``, so this is structural rather than an honour system.

    If any decision module could import ``gest_script`` it could read what the
    instructor is about to do, and every claim about the duck deciding for
    itself would be unfalsifiable.
    """
    forbidden = imports_of(module) & set(SCENARIO_MODULES)
    assert not forbidden, (
        f"{module} imports {forbidden}, so the duck can read the choreography")


def test_the_scenario_modules_really_do_hold_the_answers():
    """The check above is only meaningful if there is something to hide."""
    assert "EXPECTED_COMMANDS" in (SCRIPTS / "gest_script.py").read_text()
    assert len(CUES) >= 9


def test_the_sense_boundary_is_the_only_way_in():
    """The machine sees a Sense and an Interlock, and nothing else.

    Enforced by requiring the machine's imports to name no simulator module:
    no mujoco, no camera, no actors.
    """
    machine = imports_of("gest_machine")
    for banned in ("mujoco", "gest_camera", "gest_actors", "gest_script",
                   "numpy"):
        assert banned not in machine, (
            f"gest_machine imports {banned}; it must decide on a Sense alone")


@pytest.mark.parametrize("module", ("gest_machine", "gest_control",
                                    "gest_gesture", "gest_episode"))
def test_decision_modules_touch_no_physics(module):
    assert "mujoco" not in imports_of(module)


def test_every_module_is_bounded():
    """Bounded modules, because a 900-line module is where invariants hide.

    ``contact_geometry`` is exempt and that exemption is the point: it is
    inherited VERBATIM from the frozen sibling behavior, byte for byte, and
    reformatting it to fit a line budget would fork a file whose whole value is
    that it is the same one that was already validated.
    """
    inherited = {"contact_geometry.py"}
    for path in sorted(SCRIPTS.glob("*.py")):
        if path.name in inherited:
            continue
        lines = len(path.read_text().splitlines())
        assert lines <= 400, f"{path.name} is {lines} lines"


def test_the_inherited_module_is_byte_identical_to_its_source():
    """The exemption above is only honest if the file really is unchanged."""
    sibling = (REPO.parents[0] / "patrol-and-investigate" / "scripts"
               / "contact_geometry.py")
    if not sibling.is_file():
        pytest.skip("sibling behavior not present")
    assert (SCRIPTS / "contact_geometry.py").read_bytes() \
        == sibling.read_bytes(), (
            "contact_geometry has diverged from the frozen sibling it was "
            "inherited from; either re-sync it or stop claiming it is inherited")


def test_no_module_imports_itself_through_a_cycle():
    """The split into smaller modules must not have created an import cycle.

    A cycle would still import successfully in many orders and fail in others,
    which is the kind of bug that appears only on someone else's machine.
    """
    graph = {}
    for path in sorted(SCRIPTS.glob("*.py")):
        local = {m for m in imports_of(path.stem)
                 if (SCRIPTS / f"{m}.py").is_file()}
        graph[path.stem] = local

    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node: str, trail: list[str]) -> None:
        if node in done:
            return
        assert node not in visiting, (
            f"import cycle: {' -> '.join(trail + [node])}")
        visiting.add(node)
        for child in sorted(graph.get(node, ())):
            visit(child, trail + [node])
        visiting.discard(node)
        done.add(node)

    for node in sorted(graph):
        visit(node, [])


# -- the scenario is well-formed ------------------------------------------------
def test_the_required_sequence_is_exactly_the_six_commands():
    assert list(EXPECTED_COMMANDS) == [
        "COME", "STOP", "TURN_LEFT", "TURN_RIGHT", "BACK_UP", "WAVE"]


def test_no_expected_command_is_empty():
    """THE REGRESSION THIS PINS: two of them silently were.

    ``command_for`` takes a TEMPLATE name and the cues carry ANIMATION names, so
    the two pointing gestures resolved to ``""`` and the acceptance gate asked
    for two empty commands - which a duck that executed neither turn would have
    satisfied.
    """
    assert all(EXPECTED_COMMANDS), f"empty command in {EXPECTED_COMMANDS}"


def test_the_instructor_gives_every_accepted_gesture():
    assert all(is_instructor(c.person) for c in INSTRUCTOR_CUES)
    assert all(not is_instructor(c.person) for c in DISTRACTOR_CUES)


def test_the_partial_sits_between_two_real_commands():
    """So a rejection cannot be confused with the session finishing."""
    partial = next(c for c in INSTRUCTOR_CUES if c.expect == "reject_partial")
    before = [c for c in INSTRUCTOR_CUES
              if c.expect == "accept" and c.at_s < partial.at_s]
    after = [c for c in INSTRUCTOR_CUES
             if c.expect == "accept" and c.at_s > partial.at_s]
    assert before and after, "the partial is at one end of the session"


def test_distractor_gestures_are_from_the_same_vocabulary():
    """A distractor gesturing badly would pass the wrong-person gate cheaply."""
    instructor_gestures = {c.gesture for c in INSTRUCTOR_CUES}
    for cue in DISTRACTOR_CUES:
        assert cue.gesture in instructor_gestures, (
            f"{cue.person} gives {cue.gesture}, which the instructor never does")


def test_no_two_cues_from_one_person_overlap():
    for name in ALL_NAMES:
        cues = sorted((c for c in CUES if c.person == name),
                      key=lambda c: c.at_s)
        for first, second in zip(cues, cues[1:]):
            assert first.ends_s <= second.at_s, (
                f"{name}'s {first.gesture} overlaps {second.gesture}")


def test_every_route_stays_inside_the_area():
    for name, route in ROUTES.items():
        for corner in route.corners:
            assert inside_area(corner, 0.10), (
                f"{name}'s route corner {corner} is outside the floor")


def test_at_least_one_fixture_can_really_occlude():
    """The visibility gate must be conditioned on something that can happen."""
    assert OCCLUDERS, "no fixture is tall enough to hide anybody"
    for fixture in OCCLUDERS:
        assert fixture.height_m >= OCCLUDING_HEIGHT_M


# -- the compiled scene ---------------------------------------------------------
def test_scene_carries_the_stock_walking_robot(model):
    assert model.nu == 14
    assert model.nmesh > 0, "meshdir did not resolve"


def test_the_exact_gyro_sensor_exists_and_is_distinct(model):
    from policy_runtime import gyro_address

    address = gyro_address(model)
    assert address >= 0
    with pytest.raises(ValueError):
        gyro_address(model, "imu_gyro")


def test_duck_planar_radius_matches_the_built_model(model, data):
    """The clearance constant is pinned against the model it describes."""
    from contact_geometry import duck_planar_radius

    measured = duck_planar_radius(model, data, model.body("trunk_base").id)
    assert measured == pytest.approx(DUCK_PLANAR_RADIUS, abs=5e-4), (
        f"the model measures {measured:.4f} but the constant says "
        f"{DUCK_PLANAR_RADIUS}")


def test_the_bounding_radius_over_states_the_robot(model, data):
    """Conservative in the safe direction, and that is checked rather than said."""
    from contact_geometry import duck_planar_radius, exact_planar_radius

    trunk = model.body("trunk_base").id
    assert exact_planar_radius(model, data, trunk) < duck_planar_radius(
        model, data, trunk)


def test_every_person_has_real_articulated_arms(model):
    """The gesture must be a kinematic chain, not a decoration."""
    for name in ALL_NAMES:
        for side in ("l", "r"):
            for part in (f"{name}_shoulder_{side}", f"{name}_fore_{side}",
                         f"{name}_hand_{side}"):
                assert model.body(part) is not None


def test_the_pip_camera_exists_and_is_the_one_measured_from(model):
    assert model.camera("gesture_camera") is not None
    assert model.camera("head_camera") is not None


# -- the real run ---------------------------------------------------------------
def test_all_gates_passed(summary):
    failed = [g["name"] for g in summary["gates"] if not g["passed"]]
    assert not failed, f"failing gates: {failed}"


def test_the_run_is_long_enough_to_contain_the_session(summary):
    from gest_script import session_end_s

    assert summary["seconds"] >= session_end_s()


def test_the_accepted_order_is_exact(summary):
    assert summary["sequence"]["accepted"] == list(EXPECTED_COMMANDS)


def test_no_forbidden_state_was_entered(summary):
    entered = {t["to"] for t in summary["transitions"]}
    assert not (entered & set(FORBIDDEN_STATES))


def test_zero_command_states_held_exact_zeros_every_tick(trace):
    """Checked per tick against the trace, not against a summary counter."""
    breaches = [r for r in trace
                if r["state"] in ZERO_COMMAND_STATES
                and r["command_peak"] != 0.0]
    assert not breaches, (
        f"{len(breaches)} ticks emitted a nonzero command in a zero state, "
        f"first at t={breaches[0]['t']} in {breaches[0]['state']}")


def test_no_lateral_command_on_any_tick(trace):
    worst = max(abs(float(r["command"][1])) for r in trace)
    assert worst == 0.0, f"max |vy| was {worst}"


def test_the_duck_never_overlapped_anybody(trace):
    worst = min(float(r["min_clearance_m"]) for r in trace)
    assert worst > 0.0, f"minimum surface clearance was {worst:.4f} m"


def test_the_duck_stayed_on_its_feet(trace):
    worst = min(float(r["trunk_z"]) for r in trace)
    assert worst >= FALLEN_TRUNK_Z, f"trunk fell to {worst:.4f} m"
    assert abs(float(trace[-1]["trunk_z"]) - NOMINAL_TRUNK_Z) <= 0.012


def test_every_acceptance_was_camera_confirmed(summary):
    for episode in summary["episodes"]:
        assert episode["confirm_visible_fraction"] >= 0.95, (
            f"{episode['command']} was confirmed with the instructor visible "
            f"on only {episode['confirm_visible_fraction'] * 100:.0f}% of ticks")
        assert episode["confirm_arm_readable_fraction"] >= 0.95


def test_every_acceptance_was_sustained(summary):
    for episode in summary["episodes"]:
        assert episode["confirm_held_s"] >= CONFIRM_S - 1e-9


def test_the_two_turns_are_really_opposite(summary):
    left = summary["turns"]["TURN_LEFT"]["turned_deg"]
    right = summary["turns"]["TURN_RIGHT"]["turned_deg"]
    assert left > 0.0 > right, (
        f"left turned {left} and right turned {right}: not opposite")
    assert summary["turns"]["TURN_LEFT"]["sign_correct"]
    assert summary["turns"]["TURN_RIGHT"]["sign_correct"]


def test_the_reverse_moved_backward_along_its_own_heading(summary):
    assert summary["back_up"]["back_m"] > 0.0
    assert summary["back_up"]["reached"]


def test_the_stop_interrupted_a_real_command(summary):
    """A STOP that interrupted nothing would prove nothing."""
    assert float(summary["stop"]["command_before_stop"]) > 0.0
    assert summary["stop"]["interrupts_command"] == "COME"
    assert summary["stop"]["ticks_to_zero"] <= 1


def test_a_distractor_was_ignored_while_plainly_readable(summary):
    """The wrong-person claim, as evidence rather than as an absence."""
    ignored = [n for n, e in summary["wrong_person"].items()
               if e["sustained_past_confirm"]]
    assert ignored, (
        "no distractor held a readable command past the confirm window, so "
        "ignoring them proves nothing")
    for name in ignored:
        assert not any(e["person"] == name for e in summary["episodes"])


def test_the_partial_was_refused_while_fully_visible(summary):
    partial = summary["partial"]
    assert partial["accepted"] == 0
    assert partial["visible_fraction"] >= 0.95
    assert partial["readable_fraction"] >= 0.95
    assert partial["logged_rejections"] >= 1


def test_the_policy_is_the_untouched_stock_one(summary):
    policy = summary["policy"]
    assert policy["sha256_matches_stock"]
    assert policy["obs_dim"] == OBS_DIM == 61
    assert policy["action_scale"] == ACTION_SCALE == 0.9
    assert policy["gyro_sensor"] == GYRO_SENSOR == "imu_ang_vel"


def test_the_control_rate_and_decimation_agree(summary):
    assert summary["ctrl_hz"] == 50.0
    assert summary["decimation"] * summary["timestep"] == pytest.approx(
        1.0 / summary["ctrl_hz"], rel=1e-9)


def test_the_headless_gate_imports_no_rendering_stack():
    """Proved by blocking the render modules and importing the entry point."""
    import importlib

    blocked = {"PIL", "imageio", "matplotlib"}

    class Blocker:
        def find_module(self, name, path=None):
            return self if name.split(".")[0] in blocked else None

        def load_module(self, name):
            raise ImportError(f"{name} is blocked for this test")

    for name in list(sys.modules):
        if name.split(".")[0] in blocked:
            del sys.modules[name]
    for name in ("validate_gesture", "rollout_gesture"):
        sys.modules.pop(name, None)

    sys.meta_path.insert(0, Blocker())
    try:
        importlib.import_module("validate_gesture")
    finally:
        sys.meta_path.pop(0)
