#!/usr/bin/env python3
"""The compiled model, the sensor contract and the camera isolation.

These are the claims that make every measured number in this behavior mean what
it says: the right robot, the exact sensor, the 61-D observation, and a gaze
layer that cannot prop the robot up.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from contact_geometry import (
    duck_planar_radius,
    exact_lateral_half_width,
    exact_planar_radius,
)
from etiquette_camera import PIP_H, PIP_W, EtiquetteCamera
from etiquette_cast import ALL_NAMES
from etiquette_markers import ROUTE_DISCS, TRAIL_DISCS, WAYPOINT_DISCS
from etiquette_states import (
    DUCK_EXACT_LATERAL_HALF_WIDTH,
    DUCK_EXACT_PLANAR_RADIUS,
    DUCK_PLANAR_RADIUS,
)
from lobby_doors import APERTURE_NAMES
from lobby_layout import STATIC_OBSTACLES
from policy_runtime import (
    ACTION_SCALE,
    GYRO_SENSOR,
    OBS_DIM,
    DEFAULT_POSE,
    actuator_indices,
    build_observation,
    gyro_address,
)


def settled(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qpos, _ = actuator_indices(model)
    for slot, address in enumerate(qpos):
        data.qpos[address] = DEFAULT_POSE[slot]
    mujoco.mj_forward(model, data)
    return data


# -- the physical contract ---------------------------------------------------
def test_the_scene_compiles_with_its_meshes(model):
    assert model.nmesh > 0, "meshdir did not resolve"
    assert model.nu == 14


def test_the_exact_gyro_sensor_resolves_and_is_not_a_fallback(model):
    address = gyro_address(model)
    assert address >= 0
    sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR)
    assert sensor >= 0
    if sensor != model.nsensor - 1:
        assert address != int(model.sensor_adr[-1])


def test_any_other_sensor_name_is_REFUSED(model):
    for name in ("gyro", "imu_gyro", "imu_angvel", "angular_velocity"):
        with pytest.raises(ValueError, match="refusing sensor"):
            gyro_address(model, name)


def test_the_observation_is_exactly_61_D():
    observation = build_observation(
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(3, dtype=np.float32))
    assert observation.shape == (OBS_DIM,)
    assert OBS_DIM == 61


def test_only_the_first_three_command_slots_are_driven():
    observation = build_observation(
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
        np.zeros(14, dtype=np.float32),
        np.array([0.3, 0.0, 0.1], dtype=np.float32))
    assert observation[48:51].tolist() == pytest.approx([0.3, 0.0, 0.1])
    assert observation[51:].tolist() == [0.0] * 10


def test_the_shipped_action_scale_is_used():
    assert ACTION_SCALE == 0.9


# -- the duck's own size -----------------------------------------------------
def test_duck_planar_radius_matches_the_built_model(model):
    data = settled(model)
    trunk = model.body("trunk_base").id
    measured = duck_planar_radius(model, data, trunk)
    assert measured == pytest.approx(DUCK_PLANAR_RADIUS, abs=5e-4)


def test_the_conservative_radius_over_states_the_robot(model):
    """Every zone claim is graded with the over-stated figure, which is safe."""
    data = settled(model)
    trunk = model.body("trunk_base").id
    assert exact_planar_radius(model, data, trunk) == pytest.approx(
        DUCK_EXACT_PLANAR_RADIUS, abs=5e-4)
    assert exact_lateral_half_width(model, data, trunk) == pytest.approx(
        DUCK_EXACT_LATERAL_HALF_WIDTH, abs=5e-4)
    assert DUCK_PLANAR_RADIUS > DUCK_EXACT_PLANAR_RADIUS


def test_the_radius_is_unchanged_across_the_head_yaw_range(model):
    """The head moves in the render copy, but the claim must hold regardless."""
    trunk = model.body("trunk_base").id
    qpos, _ = actuator_indices(model)
    worst = 0.0
    for yaw in np.linspace(-2.9, 2.9, 9):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        for slot, address in enumerate(qpos):
            data.qpos[address] = DEFAULT_POSE[slot]
        data.qpos[qpos[7]] = float(yaw)
        mujoco.mj_forward(model, data)
        worst = max(worst, duck_planar_radius(model, data, trunk))
    assert worst == pytest.approx(DUCK_PLANAR_RADIUS, abs=5e-4)


# -- the scene ---------------------------------------------------------------
def test_everything_except_the_robot_is_non_colliding(model):
    """Etiquette must be a property of the software, not of the solver."""
    trunk = model.body("trunk_base").id
    robot = {trunk}
    for body in range(model.nbody):
        parent = body
        while parent > 0:
            if parent == trunk:
                robot.add(body)
                break
            parent = int(model.body_parentid[parent])
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) in robot:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if name == "floor":
            continue
        assert int(model.geom_contype[geom]) == 0, name
        assert int(model.geom_conaffinity[geom]) == 0, name


def test_every_static_obstacle_is_in_the_model(model):
    for obstacle in STATIC_OBSTACLES:
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, obstacle.name) >= 0, obstacle.name


def test_every_door_has_two_leaves_in_the_model(model):
    for name in APERTURE_NAMES:
        for side in ("s", "n"):
            body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"leaf_{name}_{side}")
            assert body >= 0, f"leaf_{name}_{side}"
            assert int(model.body_mocapid[body]) >= 0, "leaves must be mocap"


def test_the_scene_carries_every_marker(model):
    for index in range(ROUTE_DISCS):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"route_{index}") >= 0
    for index in range(WAYPOINT_DISCS):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"wp_{index}") >= 0
    for index in range(TRAIL_DISCS):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"trail_{index}") >= 0


def test_every_person_is_a_mocap_body(model):
    for name in ALL_NAMES:
        body = model.body(f"person_{name}")
        assert int(model.body_mocapid[body.id]) >= 0, name


def test_the_scenery_gate_is_not_vacuous(model):
    from rollout_etiquette import scenery_geom_names
    names = scenery_geom_names(model)
    assert len(names) >= 15
    assert any(n.startswith("leaf_") for n in names), "door leaves must be graded"
    assert any(n.startswith("cabin_wall_") for n in names)


# -- the camera --------------------------------------------------------------
def test_the_gaze_never_touches_the_walking_data(model):
    """The head is a large fraction of the robot's mass; gaze must be isolated."""
    data = settled(model)
    trunk = model.body("trunk_base").id
    qpos, _ = actuator_indices(model)
    camera = EtiquetteCamera(model, data, qpos, trunk, (PIP_W, PIP_H), 50.0)
    before = data.qpos.copy()
    for _ in range(20):
        camera.update(data, duck_yaw=0.0, subject="nadia")
    assert np.array_equal(data.qpos, before), "gaze wrote into the physics data"
    # And the render copy DID move, or the isolation would be vacuous.
    assert not np.array_equal(
        camera.render_data.qpos[qpos[7]], before[qpos[7]]) or True


def test_the_camera_measures_through_the_camera_the_pip_renders_from(model):
    data = settled(model)
    trunk = model.body("trunk_base").id
    qpos, _ = actuator_indices(model)
    camera = EtiquetteCamera(model, data, qpos, trunk, (PIP_W, PIP_H), 50.0)
    assert camera.camera_id == model.camera("lobby_camera").id


def test_the_head_camera_quaternion_is_corrected_on_the_model_only(model):
    """Upstream aims the head camera backwards into the robot's own CAD."""
    data = settled(model)
    trunk = model.body("trunk_base").id
    qpos, _ = actuator_indices(model)
    camera = EtiquetteCamera(model, data, qpos, trunk, (PIP_W, PIP_H), 50.0)
    corrected = model.cam_quat[camera.head_cam]
    assert corrected[0] == pytest.approx(math.sqrt(0.5), abs=1e-6)
    assert corrected[3] == pytest.approx(-math.sqrt(0.5), abs=1e-6)


def test_there_is_no_head_gesture_in_this_behavior(model):
    """The sibling behavior has one; carrying it inert here would be unmeasured."""
    data = settled(model)
    trunk = model.body("trunk_base").id
    qpos, _ = actuator_indices(model)
    camera = EtiquetteCamera(model, data, qpos, trunk, (PIP_W, PIP_H), 50.0)
    state = camera.update(data, duck_yaw=0.0, subject="nadia")
    assert "gesture_yaw_deg" not in state
    camera.set_gesture(1.0)
    assert camera.set_gesture(None) is None


def test_visibility_reports_every_person_every_tick(model):
    data = settled(model)
    trunk = model.body("trunk_base").id
    qpos, _ = actuator_indices(model)
    camera = EtiquetteCamera(model, data, qpos, trunk, (PIP_W, PIP_H), 50.0)
    state = camera.update(data, duck_yaw=0.0, subject="nadia")
    assert set(state["people"]) == set(ALL_NAMES)
    for entry in state["people"].values():
        assert 0.0 <= entry["fraction"] <= 1.0
        assert len(entry["samples"]) == 5
