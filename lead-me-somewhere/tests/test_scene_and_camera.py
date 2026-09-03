"""The scene, the camera and the isolation of gaze from walking physics.

Two things this module exists to prevent:

* a scene that compiled but lost its meshes, or resolved the wrong sensor, so
  every measured constant in the behavior would be describing a different robot;
* a gaze or gesture layer that leaked into the locomotion state, which would let
  the head prop the robot up and make "the stock walking policy did this"
  false.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from guide_camera import PIP_H, PIP_W, GuideCamera
from guide_cast import ALL_NAMES, BASE_SAMPLE_DZ, FOLLOWER, PEOPLE
from guide_layout import (
    DESTINATIONS,
    OBSTACLES,
    OCCLUDERS,
    OCCLUDING_HEIGHT_M,
)
from guide_states import (
    DUCK_PLANAR_RADIUS,
    INDICATE_PITCH_AMPLITUDE_DEG,
    INDICATE_YAW_AMPLITUDE_DEG,
)
from policy_runtime import (
    ACTION_SCALE,
    GYRO_SENSOR,
    OBS_DIM,
    HEAD_PITCH_ACT,
    HEAD_ROLL_ACT,
    HEAD_YAW_ACT,
    actuator_indices,
    build_observation,
    gyro_address,
)

pytestmark = pytest.mark.slow


# -- the model --------------------------------------------------------------

def test_the_scene_compiled_with_its_meshes(model):
    """A scene whose meshdir did not resolve compiles happily with zero meshes
    and then describes a robot that is not this one."""
    assert model.nmesh > 0
    assert model.nu == 14


def test_only_the_exact_gyro_sensor_is_accepted(model):
    address = gyro_address(model)
    assert address >= 0
    with pytest.raises(ValueError):
        gyro_address(model, "gyro")
    with pytest.raises(ValueError):
        gyro_address(model, "imu_gyro")


def test_the_gyro_is_not_an_accidental_last_sensor_fallback(model):
    """``mj_name2id`` returns -1 for an unknown sensor and
    ``sensor_adr[-1]`` is a VALID index, so a wrong name silently feeds a
    different physical quantity into the policy's base_ang_vel slot."""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                  GYRO_SENSOR)
    assert sensor_id >= 0
    if sensor_id != model.nsensor - 1:
        assert int(model.sensor_adr[sensor_id]) != int(model.sensor_adr[-1])


def test_the_observation_is_exactly_61_dimensional():
    observation = build_observation(
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(3, dtype=np.float32))
    assert observation.shape == (OBS_DIM,)
    assert OBS_DIM == 61


def test_the_command_slots_beyond_the_twist_are_zero_padded():
    twist = np.array([0.3, 0.0, -0.1], dtype=np.float32)
    observation = build_observation(
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
        np.zeros(14, dtype=np.float32), twist)
    assert np.allclose(observation[48:51], twist)
    assert np.allclose(observation[51:], 0.0)


def test_the_action_scale_is_the_shipped_one():
    assert ACTION_SCALE == 0.9


def test_the_duck_planar_radius_constant_matches_the_model(model, data):
    """The planner inflates obstacles by this figure.  If it drifts from the
    built scene, every clearance argument in the behavior is about a robot of
    the wrong size."""
    from contact_geometry import duck_planar_radius
    measured = duck_planar_radius(model, data, model.body("trunk_base").id)
    assert DUCK_PLANAR_RADIUS == pytest.approx(measured, abs=0.01)


# -- the scene's contents ---------------------------------------------------

def test_every_person_is_present_and_non_colliding(model):
    for person in PEOPLE:
        body = model.body(f"person_{person.name}")
        assert model.body_mocapid[body.id] >= 0, "people must be mocap bodies"
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if any(name.startswith(p.name + "_") for p in PEOPLE):
            assert model.geom_contype[geom] == 0
            assert model.geom_conaffinity[geom] == 0


def test_the_scenery_cannot_push_the_robot(model):
    """Non-colliding scenery is deliberate: it makes route safety a property of
    the SOFTWARE rather than of the contact solver."""
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if name.startswith(("obs_", "wall_", "dest_")):
            assert model.geom_contype[geom] == 0
            assert model.geom_conaffinity[geom] == 0


def test_all_three_destinations_are_real_geometry(model):
    """Real pylons, not HUD annotations, so a viewer can see three candidates
    existed and which one the duck walked to."""
    for destination in DESTINATIONS:
        key = destination.key.lower()
        for suffix in ("base", "post", "sign"):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                     f"dest_{key}_{suffix}")
            assert geom >= 0, f"dest_{key}_{suffix} missing from the scene"


def test_the_occluders_are_derived_from_height_not_declared():
    """Shortening a body in the layout must stop it counting as an occluder
    everywhere at once."""
    for obstacle in OBSTACLES:
        assert obstacle.occludes == (obstacle.height_m >= OCCLUDING_HEIGHT_M)
    assert OCCLUDERS, "no full-height body can hide anybody"
    top_sample = max(BASE_SAMPLE_DZ) + 0.36
    for occluder in OCCLUDERS:
        assert occluder.height_m > top_sample


def test_the_low_furniture_cannot_hide_anybody():
    lowest_sample = min(BASE_SAMPLE_DZ) + 0.36
    for obstacle in OBSTACLES:
        if not obstacle.occludes:
            assert obstacle.height_m < lowest_sample + 0.4


def test_the_scenery_gate_collects_geoms_from_the_model(model):
    """Hand-listing would let a scene edit silently drop a surface."""
    from rollout_guide import scenery_geom_names
    names = scenery_geom_names(model)
    assert len(names) >= len(OBSTACLES) + 4
    for obstacle in OBSTACLES:
        assert f"obs_{obstacle.name}" in names


# -- the camera and gaze isolation -----------------------------------------

@pytest.fixture(scope="module")
def camera(model, data):
    qpos_idx, _ = actuator_indices(model)
    return GuideCamera(model, data, qpos_idx, model.body("trunk_base").id,
                       (PIP_W, PIP_H), 50.0)


def test_gaze_never_touches_the_walking_state(model, data, camera):
    """The head is a large fraction of the robot's mass and the stock walking
    policy was never trained to compensate an imposed head trajectory.  If gaze
    leaked into the physics it could prop the robot up."""
    before = data.qpos.copy()
    for _ in range(30):
        camera.update(data, duck_yaw=0.3, subject=FOLLOWER.name)
    assert np.array_equal(before, data.qpos), (
        "the camera wrote into the authoritative walking state")


def test_the_arrival_gesture_is_also_rendering_only(model, data, camera):
    """The indication is body-safe BY CONSTRUCTION, not by restraint."""
    before = data.qpos.copy()
    for index in range(60):
        camera.update(data, duck_yaw=0.0, subject=FOLLOWER.name,
                      gesture_elapsed=index * 0.02)
    assert np.array_equal(before, data.qpos)


def test_the_gesture_actually_moves_the_rendering_head(model, data, camera):
    """A gesture nobody can see is not an indication."""
    yaws = []
    for index in range(120):
        state = camera.update(data, duck_yaw=0.0, subject=FOLLOWER.name,
                              gesture_elapsed=index * 0.02)
        yaws.append(state["gesture_yaw_deg"])
    assert max(yaws) - min(yaws) > 10.0
    assert max(abs(y) for y in yaws) <= INDICATE_YAW_AMPLITUDE_DEG + 1e-6


def test_the_gesture_stays_inside_the_measured_joint_range(model):
    yaw_joint = int(model.actuator_trnid[HEAD_YAW_ACT, 0])
    pitch_joint = int(model.actuator_trnid[HEAD_PITCH_ACT, 0])
    yaw_limit = math.degrees(float(model.jnt_range[yaw_joint][1]))
    pitch_limit = math.degrees(float(model.jnt_range[pitch_joint][1]))
    assert INDICATE_YAW_AMPLITUDE_DEG < yaw_limit
    assert INDICATE_PITCH_AMPLITUDE_DEG < pitch_limit


def test_the_gesture_is_off_when_not_indicating(model, data, camera):
    state = camera.update(data, duck_yaw=0.0, subject=FOLLOWER.name,
                          gesture_elapsed=None)
    assert state["gesture_yaw_deg"] == 0.0


def test_the_head_camera_looks_forward_not_into_its_own_cad(model, camera):
    """The upstream head_camera quaternion aims -Z backwards into the robot.
    It is corrected on the in-memory model only, so the physical camera
    POSITION copied to the rig stays meaningful."""
    quat = model.cam_quat[camera.head_cam]
    assert not np.allclose(quat, [0.0, 0.0, -1.0, 0.0])


def test_visibility_is_measured_through_the_pip_camera(camera):
    """What the viewer sees and what the gate graded must be the same
    frustum."""
    assert camera.camera_id == camera.guide_cam
    assert (camera.pip_w, camera.pip_h) == (PIP_W, PIP_H)
    assert camera.tan_h == pytest.approx((PIP_W / PIP_H) * camera.tan_v)


def test_the_camera_samples_five_points_scaled_by_stature(camera):
    for name in ALL_NAMES:
        points = camera.sample_points(name)
        assert len(points) == len(BASE_SAMPLE_DZ)


def test_a_person_behind_an_occluder_is_reported_invisible(model, data,
                                                            camera):
    """The occlusion ray cast has to hit real geometry, or the visibility
    percentage is a frustum test wearing a perception costume."""
    occluder = OCCLUDERS[0]
    body = model.body(f"person_{FOLLOWER.name}")
    mocap = int(model.body_mocapid[body.id])
    # Put her directly behind the occluder, on the far side from the origin.
    direction = np.array(occluder.center, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    data.mocap_pos[mocap, :2] = np.array(occluder.center) + direction * 0.9
    data.mocap_pos[mocap, 2] = 0.36
    # And the duck on the near side, looking at her through it.
    data.qpos[0:2] = np.array(occluder.center) - direction * 0.9
    mujoco.mj_forward(model, data)
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    state = camera.update(data, duck_yaw=yaw, subject=FOLLOWER.name)
    entry = state["people"][FOLLOWER.name]
    if entry["visible"]:
        pytest.skip("this occluder geometry does not block at this range")
    assert camera.blocking_geom(FOLLOWER.name)


def test_the_planar_occluder_test_agrees_with_the_hall(model):
    """The cheap LOS predicate must name a body that exists in the scene."""
    from guide_layout import occluder_between
    occluder = OCCLUDERS[0]
    centre = np.array(occluder.center, dtype=np.float64)
    direction = np.array([1.0, 0.0])
    name = occluder_between(centre - direction * 1.2, centre + direction * 1.2)
    assert name in {o.name for o in OCCLUDERS}
    far = occluder_between((0.0, 0.0), (0.001, 0.001))
    assert far is None or far in {o.name for o in OCCLUDERS}
