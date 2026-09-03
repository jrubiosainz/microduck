#!/usr/bin/env python3
"""The rollout's own invariants: the monotonic cursor, tick order, and the gate.

Three things live in ``rollout_beside`` that nothing else can check:

* **the crossing waypoint cursor is monotonic**, which is a scar from
  ``lost-child-find-person``: a stateless "first point farther than the
  tolerance" selector re-targets an already-passed waypoint as soon as the
  duck's distance to it grows again, producing an endless loop around it;
* **the machine decides on measurements taken BEFORE the physics step**, never
  after, so a decision is never authorised by a world state that only exists
  because of the decision;
* **the clearance gate is non-vacuous**, i.e. it is measured against scenery
  collected from the scene's own naming rather than a hand-written list.

The cursor tests drive ``_advance_cross_cursor`` directly on a rollout built
with the real scene but never stepped, so they are exact and cheap.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from beside_geometry import CROSS_ARRIVE_M  # noqa: E402
from rollout_beside import SCENERY_PREFIXES, scenery_geom_names  # noqa: E402

POLICY = REPO / "onnx" / "alpha_walking.onnx"


class CursorHost:
    """The smallest object ``_advance_cross_cursor`` needs.

    Bound as an unbound method so the cursor logic is tested exactly as the
    rollout runs it, without loading MuJoCo or the policy.
    """

    def __init__(self, waypoints):
        from rollout_beside import BesideRollout

        self._cross_waypoints = [np.asarray(w, dtype=np.float64)
                                 for w in waypoints]
        self._cross_cursor = 0
        self._advance = BesideRollout._advance_cross_cursor.__get__(self)

    def advance(self, duck_xy):
        self._advance(np.asarray(duck_xy, dtype=np.float64))
        return self._cross_cursor


# -- the monotonic cursor -----------------------------------------------------

def test_the_cursor_advances_when_a_waypoint_is_reached():
    host = CursorHost([(0.0, 0.0), (1.0, 0.0)])
    assert host.advance((0.6, 0.0)) == 0, "still outside the arrival tolerance"
    assert host.advance((CROSS_ARRIVE_M - 0.01, 0.0)) == 1


def test_the_cursor_never_moves_backwards_when_the_duck_drifts_away():
    """THE SCAR.  A stateless selector re-targets a passed waypoint as soon as
    the distance to it grows, which is how a sibling behavior produced an
    endless loop around a corner."""
    host = CursorHost([(0.0, 0.0), (1.0, 0.0)])
    assert host.advance((0.05, 0.0)) == 1
    for x in (0.5, 0.2, 0.0, -0.5, -3.0):
        assert host.advance((x, 0.0)) == 1, (
            "the cursor moved back to an already-passed waypoint")


def test_the_cursor_stops_at_the_last_waypoint_rather_than_running_off_the_end():
    host = CursorHost([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    host.advance((0.0, 0.0))
    host.advance((1.0, 0.0))
    assert host._cross_cursor == 2
    for _ in range(5):
        assert host.advance((2.0, 0.0)) == 2


def test_the_cursor_can_skip_a_waypoint_it_is_already_past():
    """Monotonic does not mean one-at-a-time: arriving near the second point
    while the first is also inside tolerance must not strand the duck."""
    host = CursorHost([(0.0, 0.0), (0.05, 0.0), (2.0, 0.0)])
    assert host.advance((0.0, 0.0)) == 2


def test_an_empty_or_single_waypoint_list_is_safe():
    assert CursorHost([]).advance((0.0, 0.0)) == 0
    assert CursorHost([(1.0, 1.0)]).advance((1.0, 1.0)) == 0


def test_only_one_method_is_permitted_to_move_the_cursor():
    """A second writer would reintroduce the loop by the back door."""
    source = (REPO / "scripts" / "rollout_beside.py").read_text()
    tree = ast.parse(source)
    writers = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (isinstance(target, ast.Attribute)
                    and target.attr == "_cross_cursor"):
                writers.add(_enclosing_function(tree, node))
    assert writers <= {"__init__", "_advance_cross_cursor", "_plan_cross"}, (
        f"unexpected writers of the cursor: {sorted(writers)}")
    assert "_advance_cross_cursor" in writers


def _enclosing_function(tree, node) -> str:
    for candidate in ast.walk(tree):
        if isinstance(candidate, ast.FunctionDef):
            for inner in ast.walk(candidate):
                if inner is node:
                    return candidate.name
    return "<module>"


# -- the crossing waypoints ---------------------------------------------------

def test_the_crossing_waypoints_are_planned_astern_and_on_the_target_side():
    """Both offsets are behind her, and the second commits to the far side."""
    from rollout_beside import BesideRollout

    host = type("Host", (), {})()
    BesideRollout._plan_cross(host, None, -1)
    assert host._cross_cursor == 0
    for lateral, longitudinal in host._cross_offsets:
        assert longitudinal < 0.0, "every crossing waypoint is astern of her"
    assert host._cross_offsets[0][0] == 0.0, "the first is on her centreline"
    assert host._cross_offsets[1][0] < 0.0, "the second is on the target side"

    BesideRollout._plan_cross(host, None, +1)
    assert host._cross_offsets[1][0] > 0.0


def test_the_crossing_waypoints_are_reanchored_to_her_live_pose():
    """She keeps walking during the manoeuvre, so a world-frozen waypoint would
    be behind where she WAS rather than behind where she IS."""
    import math

    from beside_geometry import relative
    from rollout_beside import BesideRollout

    host = type("Host", (), {})()
    BesideRollout._plan_cross(host, None, -1)

    class Guardian:
        def __init__(self, pos, yaw):
            self.pos = np.asarray(pos, dtype=np.float64)
            self.yaw = yaw

    early = BesideRollout._cross_targets(host, Guardian((0.0, 0.0), 0.0))
    later = BesideRollout._cross_targets(
        host, Guardian((2.0, 1.0), math.radians(35.0)))
    assert not np.allclose(early[0], later[0]), "the waypoint must move with her"
    # The offsets in HER frame are identical at both instants.
    for point, guardian in ((early[0], Guardian((0.0, 0.0), 0.0)),
                            (later[0], Guardian((2.0, 1.0),
                                                math.radians(35.0)))):
        lateral, longitudinal = relative(point, guardian.pos, guardian.yaw)
        assert longitudinal == pytest.approx(host._cross_offsets[0][1])
        assert lateral == pytest.approx(host._cross_offsets[0][0], abs=1e-12)


# -- the clearance gate's own non-vacuity --------------------------------------

def test_the_scenery_gate_collects_geoms_from_the_scene_not_a_hand_list():
    from policy_runtime import load_scene

    model = load_scene()
    names = scenery_geom_names(model)
    assert names, "the clearance gate would be vacuous"
    assert all(name.startswith(SCENERY_PREFIXES) for name in names)
    # Every obstacle in the layout, and all four walls, must be represented.
    from promenade_layout import OBSTACLES

    for obstacle in OBSTACLES:
        assert any(f"obs_{obstacle.name}" == name for name in names), (
            f"{obstacle.name} is missing from the clearance gate")
    for wall in ("wall_n", "wall_s", "wall_e", "wall_w"):
        assert wall in names


def test_a_scene_with_no_scenery_is_refused_rather_than_silently_passing():
    class Empty:
        ngeom = 0

    import mujoco

    original = mujoco.mj_id2name
    try:
        mujoco.mj_id2name = lambda *a, **k: None
        with pytest.raises(RuntimeError, match="vacuous"):
            scenery_geom_names(Empty())
    finally:
        mujoco.mj_id2name = original


# -- ordering ------------------------------------------------------------------

def test_the_machine_is_advanced_before_the_physics_step():
    """One control tick at 50 Hz is 20 ms, which is honest and is also what a
    real perception pipeline incurs.  Grading a side and acting on that grade
    within the same tick would let a decision be authorised by a world state
    that only exists after the decision was made.
    """
    source = (REPO / "scripts" / "rollout_beside.py").read_text()
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "step")
    lines = {"machine": None, "policy": None, "physics": None,
             "repose": None, "camera": None}
    for node in ast.walk(step):
        if not isinstance(node, ast.Call):
            continue
        text = ast.unparse(node.func)
        if text == "self.machine.update" and lines["machine"] is None:
            lines["machine"] = node.lineno
        elif text == "self.runner.step" and lines["policy"] is None:
            lines["policy"] = node.lineno
        elif text == "mujoco.mj_step" and lines["physics"] is None:
            lines["physics"] = node.lineno
        elif text == "pose_people" and lines["repose"] is None:
            lines["repose"] = node.lineno
        elif text == "self.camera.update" and lines["camera"] is None:
            lines["camera"] = node.lineno
    assert all(value is not None for value in lines.values()), lines
    assert lines["machine"] < lines["policy"] < lines["physics"] \
        < lines["repose"] < lines["camera"], lines


def test_the_verdicts_are_computed_from_the_previous_tick_world_state():
    source = (REPO / "scripts" / "rollout_beside.py").read_text()
    assert "tracks_from_states(self._previous_people" in source, (
        "the side grade must be taken from the world as it was measured, not "
        "from the world the physics step has just produced")


def test_the_camera_never_writes_back_into_the_walking_state():
    """Gaze cannot prop the robot up: the head is posed only in an isolated
    render copy."""
    source = (REPO / "scripts" / "beside_camera.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            text = ast.unparse(target)
            # Subscripted writes into a data object, e.g. ``x.qpos[i] = v``.
            # ``self.qpos_idx`` is an index array, not simulator state.
            if not any(f".{field}[" in text for field in
                       ("qpos", "ctrl", "qvel", "mocap_pos", "mocap_quat")):
                continue
            assert text.startswith("self.render_data"), (
                f"the camera writes {text!r} outside the render copy")


def test_the_headless_gate_imports_no_rendering_stack():
    """A validation run has no PIL, imageio or GPU dependency at all."""
    modules = ("beside_metrics", "rollout_beside", "beside_camera",
               "beside_control", "beside_machine", "side_choice",
               "beside_actors", "beside_route", "promenade_layout",
               "contact_geometry", "policy_runtime", "validate_beside")
    forbidden = ("PIL", "imageio", "matplotlib", "cv2", "moviepy")
    for name in modules:
        source = (REPO / "scripts" / f"{name}.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for imported in names:
                root = imported.split(".")[0]
                assert root not in forbidden, (
                    f"{name} imports the rendering stack: {imported}")
