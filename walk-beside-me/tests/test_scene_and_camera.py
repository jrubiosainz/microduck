#!/usr/bin/env python3
"""Scene, sensor and camera against the real MuJoCo model.

These load the actual promenade scene and the stock ONNX policy, so they are
slower than the pure-logic suite and are marked ``slow``.  They catch the class
of defect a pure unit test cannot: a constant that has drifted from the geometry
it describes, a scene whose obstacles no longer match the layout the chooser
measures against, a sensor name that silently resolves to a different physical
quantity, or a gaze layer that writes back into the walking state.

THE SENSOR TRAP IS THE DANGEROUS ONE.  ``mj_name2id`` returns -1 for an unknown
sensor and ``model.sensor_adr[-1]`` is a VALID index, so a wrong name feeds a
different quantity into the policy's ``base_ang_vel`` slot.  The observation
still looks plausible and the robot still walks, which is exactly what makes it
dangerous, and it would invalidate every measured constant in this behavior.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tools"))

from beside_actors import people_at, pose_people  # noqa: E402
from beside_camera import PIP_H, PIP_W, BesideCamera  # noqa: E402
from beside_cast import (  # noqa: E402
    ALL_NAMES,
    BASE_ORIGIN_Z,
    BASE_SAMPLE_DZ,
    BY_NAME,
    GUARDIAN,
    PEOPLE,
)
from beside_constants import (  # noqa: E402
    TRACK_PITCH_DEG,
    TRACK_PITCH_RATE_DPS,
    TRACK_YAW_RATE_DPS,
)
from policy_runtime import (  # noqa: E402
    ACTION_SCALE,
    COMMAND_DIM,
    CTRL_HZ,
    DEFAULT_POSE,
    GYRO_SENSOR,
    HEAD_PITCH_ACT,
    HEAD_ROLL_ACT,
    HEAD_YAW_ACT,
    NOMINAL_TRUNK_Z,
    OBS_DIM,
    PolicyRunner,
    actuator_indices,
    build_observation,
    gyro_address,
    load_scene,
    quat_rotate_inverse,
    wrap_angle,
)
from promenade_layout import (  # noqa: E402
    OBSTACLES,
    OCCLUDERS,
    OCCLUDING_HEIGHT_M,
    FLOOR_HALF,
)

pytestmark = pytest.mark.slow

POLICY = REPO / "onnx" / "alpha_walking.onnx"
POLICY_SHA256 = (
    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c")


@pytest.fixture(scope="module")
def model():
    return load_scene()


@pytest.fixture(scope="module")
def data(model):
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    return state


# -- the policy contract -------------------------------------------------------

def test_the_policy_is_the_byte_identical_stock_walking_policy():
    digest = hashlib.sha256(POLICY.read_bytes()).hexdigest()
    assert digest == POLICY_SHA256


def test_the_policy_expects_the_sixty_one_dimensional_observation():
    runner = PolicyRunner(POLICY)
    assert int(runner.session.get_inputs()[0].shape[1]) == OBS_DIM


def test_a_policy_of_the_wrong_width_is_refused_rather_than_adapted(tmp_path):
    missing = tmp_path / "nope.onnx"
    with pytest.raises(SystemExit, match="policy not found"):
        PolicyRunner(missing)


def test_the_observation_is_assembled_in_the_documented_order():
    gyro = np.arange(3, dtype=np.float32)
    gravity = np.arange(3, dtype=np.float32) + 10.0
    joint_pos = np.arange(14, dtype=np.float32) + 100.0
    joint_vel = np.arange(14, dtype=np.float32) + 200.0
    last_action = np.arange(14, dtype=np.float32) + 300.0
    twist = np.array([0.3, 0.0, -0.1], dtype=np.float32)
    observation = build_observation(gyro, gravity, joint_pos, joint_vel,
                                    last_action, twist)
    assert observation.shape == (OBS_DIM,)
    assert np.allclose(observation[0:3], gyro)
    assert np.allclose(observation[3:6], gravity)
    assert np.allclose(observation[6:20], joint_pos)
    assert np.allclose(observation[20:34], joint_vel)
    assert np.allclose(observation[34:48], last_action)
    assert np.allclose(observation[48:51], twist)
    assert np.allclose(observation[51:], 0.0), (
        "the unused command slots are zero-padded, which is the documented "
        "convention for a task that does not drive them")
    assert observation.shape[0] == 48 + COMMAND_DIM


def test_a_malformed_observation_is_refused():
    with pytest.raises(RuntimeError, match="61-D observation"):
        build_observation(np.zeros(3, dtype=np.float32),
                          np.zeros(3, dtype=np.float32),
                          np.zeros(13, dtype=np.float32),
                          np.zeros(14, dtype=np.float32),
                          np.zeros(14, dtype=np.float32),
                          np.zeros(3, dtype=np.float32))


def test_the_action_scale_and_default_pose_are_the_shipped_ones():
    assert ACTION_SCALE == 0.9
    assert DEFAULT_POSE.shape == (14,)
    assert CTRL_HZ == 50.0


# -- the sensor trap -----------------------------------------------------------

def test_the_gyro_resolves_to_the_exact_declared_sensor(model):
    address = gyro_address(model)
    sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR)
    assert sensor >= 0
    assert address == int(model.sensor_adr[sensor])


def test_any_other_sensor_name_is_refused_with_no_fallback_list(model):
    """A different quantity in the base_ang_vel slot invalidates every measured
    constant in this behavior, and the robot would still walk."""
    for name in ("angular-velocity", "imu_lin_vel", "orientation",
                 "root_angmom", "", "gyro"):
        with pytest.raises(ValueError, match="refusing sensor"):
            gyro_address(model, name)


def test_the_scene_actually_contains_a_distinct_imu_ang_vel_sensor(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
             for i in range(model.nsensor)]
    assert GYRO_SENSOR in names
    assert names.count(GYRO_SENSOR) == 1
    # There is a similarly-named sensor in this model, which is exactly why the
    # resolver refuses anything but the declared name.
    assert "angular-velocity" in names


def test_the_scene_load_asserts_the_stock_walking_robot(model):
    assert model.nu == 14
    assert model.nmesh > 0, "meshdir did not resolve"


def test_actuator_indices_are_consistent_with_the_model(model):
    qpos_idx, qvel_idx = actuator_indices(model)
    assert len(qpos_idx) == model.nu == len(qvel_idx)
    assert len(set(qpos_idx.tolist())) == model.nu
    assert all(0 <= int(i) < model.nq for i in qpos_idx)
    assert all(0 <= int(i) < model.nv for i in qvel_idx)


def test_the_gravity_projection_is_a_unit_vector_in_the_body_frame():
    for angle in (0.0, 0.7, -1.3):
        quat = np.array([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0],
                        dtype=np.float32)
        projected = quat_rotate_inverse(
            quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        assert float(np.linalg.norm(projected)) == pytest.approx(1.0, abs=1e-6)


def test_wrap_angle_maps_into_a_single_turn():
    for value in (0.0, math.pi - 1e-9, -math.pi, 3.0 * math.pi, -7.5):
        wrapped = wrap_angle(value)
        assert -math.pi <= wrapped < math.pi
        assert math.isclose(math.cos(wrapped), math.cos(value), abs_tol=1e-9)
        assert math.isclose(math.sin(wrapped), math.sin(value), abs_tol=1e-9)


# -- the scene matches the layout ---------------------------------------------

def test_every_obstacle_in_the_layout_is_painted_into_the_scene(model):
    """A scene edit that moves the kiosk must move it everywhere at once."""
    for obstacle in OBSTACLES:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                 f"obs_{obstacle.name}")
        assert geom >= 0, f"{obstacle.name} is missing from the scene"
        position = model.geom_pos[geom]
        assert float(position[0]) == pytest.approx(obstacle.center[0], abs=1e-3)
        assert float(position[1]) == pytest.approx(obstacle.center[1], abs=1e-3)


def test_the_obstacle_heights_in_the_scene_match_the_layout(model):
    for obstacle in OBSTACLES:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                 f"obs_{obstacle.name}")
        half_height = float(model.geom_size[geom][2]) \
            if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_BOX) \
            else float(model.geom_size[geom][1])
        assert half_height * 2.0 == pytest.approx(obstacle.height_m, abs=1e-3)


def test_occlusion_is_derived_from_height_rather_than_hand_labelled():
    """Shortening a body in the layout must also stop it counting as an
    occluder, everywhere at once."""
    for obstacle in OBSTACLES:
        assert obstacle.occludes == (obstacle.height_m >= OCCLUDING_HEIGHT_M)
    assert {o.name for o in OCCLUDERS} == {
        o.name for o in OBSTACLES if o.height_m >= OCCLUDING_HEIGHT_M}


def test_the_hedge_obstructs_the_robot_without_ever_hiding_a_person():
    """The occlusion arithmetic, checked as numbers against the real samples.

    MEASURED: the five camera samples on an adult sit at z = 0.26, 0.38, 0.52,
    0.64 and 0.70.  ``hedge_s`` is 0.45 m, so it stands above the two LOWEST
    samples and below the top three.  A body behind it therefore loses at most
    two of five samples and remains visible, which is what "it never hides a
    person" means here — visibility is ``sample_count > 0``.

    NOTE: the layout docstring's "below the 0.66 m lowest-but-one camera
    sample" was measured wrong (the lowest-but-one sample is 0.38 m and no
    sample sits at 0.66 m) and has been corrected to match these numbers.  The
    CONCLUSION was always right and is pinned here.
    """
    from promenade_layout import HEDGE_S

    samples = sorted(BASE_ORIGIN_Z + dz for dz in BASE_SAMPLE_DZ)
    assert samples == pytest.approx([0.26, 0.38, 0.52, 0.64, 0.70], abs=1e-9)

    above = [z for z in samples if z > HEDGE_S.height_m]
    assert len(above) >= 3, (
        "a body behind the hedge must keep a majority of its samples")
    assert HEDGE_S.height_m < OCCLUDING_HEIGHT_M
    assert not HEDGE_S.occludes, (
        "the hedge must never count as a full-height occluder")
    assert HEDGE_S.height_m > 0.19, "but it is still well above the duck's eye"


def test_the_kiosk_and_columns_do_occlude():
    from promenade_layout import COLUMN_N, COLUMN_W, KIOSK

    topmost_person_sample = BASE_ORIGIN_Z + max(BASE_SAMPLE_DZ)
    for obstacle in (KIOSK, COLUMN_N, COLUMN_W):
        assert obstacle.occludes
        assert obstacle.height_m > topmost_person_sample


def test_the_scene_is_entirely_non_colliding_except_the_robot(model):
    """A step the duck takes is never the result of somebody nudging it."""
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if not name:
            continue
        if name.startswith(("obs_", "wall_", "hall_", "slot_", "cross_",
                            "trail_", "blocked_")) or name == "floor":
            if name == "floor":
                continue
            assert int(model.geom_contype[geom]) == 0, name
            assert int(model.geom_conaffinity[geom]) == 0, name


def test_every_person_is_a_non_colliding_mocap_body(model):
    for name in ALL_NAMES:
        body = model.body(f"person_{name}")
        assert int(model.body_mocapid[body.id]) >= 0, (
            f"{name} is not a mocap body and would add degrees of freedom")


def test_person_geometry_is_scaled_by_stature(model):
    """A shorter adult genuinely has a lower head, so the camera's topmost
    sample of them is genuinely nearer the ground."""
    heights = {}
    for person in PEOPLE:
        body = model.body(f"person_{person.name}")
        geoms = [g for g in range(model.ngeom)
                 if int(model.geom_bodyid[g]) == body.id]
        heights[person.name] = max(float(model.geom_pos[g][2]) for g in geoms)
    tallest = max(PEOPLE, key=lambda p: p.stature)
    shortest = min(PEOPLE, key=lambda p: p.stature)
    assert heights[tallest.name] > heights[shortest.name]
    assert tallest.height_m > shortest.height_m


def test_the_promenade_holds_every_actor_inside_its_walls():
    for step in range(0, 4301, 25):
        for name, state in people_at(step * 0.02).items():
            assert abs(float(state.pos[0])) <= FLOOR_HALF[0], name
            assert abs(float(state.pos[1])) <= FLOOR_HALF[1], name


# -- the camera ----------------------------------------------------------------

@pytest.fixture(scope="module")
def camera(model, data):
    qpos_idx, _ = actuator_indices(model)
    trunk = model.body("trunk_base").id
    pose_people(model, data, people_at(0.0), 0.0)
    mujoco.mj_forward(model, data)
    return BesideCamera(model, data, qpos_idx, trunk, (PIP_W, PIP_H), CTRL_HZ)


def test_the_visibility_frustum_is_the_one_the_pip_would_render_through(
        model, camera):
    """The headless gate and any future render must measure through the same
    frustum, or the reported percentages and the picture disagree."""
    fovy = float(model.cam_fovy[camera.beside_cam])
    assert camera.half_v_deg == pytest.approx(fovy / 2.0, abs=1e-9)
    expected_h = math.degrees(math.atan(
        (PIP_W / PIP_H) * math.tan(math.radians(fovy / 2.0))))
    assert camera.half_h_deg == pytest.approx(expected_h, abs=1e-9)
    assert camera.pip_w == PIP_W and camera.pip_h == PIP_H


def test_the_head_camera_quaternion_is_corrected_on_the_model(model, camera):
    """The upstream quaternion aims -Z backwards into the robot's own CAD."""
    expected = np.array([math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)])
    assert np.allclose(model.cam_quat[camera.head_cam], expected, atol=1e-9)


def test_the_camera_poses_the_head_only_in_the_isolated_render_data(
        model, data, camera):
    """THE ISOLATION CLAIM.  The head is a large fraction of the robot's mass
    and the stock walking policy was never trained to compensate an imposed head
    trajectory, so gaze must not be able to prop the robot up."""
    qpos_idx, _ = actuator_indices(model)
    before = data.qpos.copy()
    before_ctrl = data.ctrl.copy()

    camera.update(data, duck_yaw=0.6, subject=GUARDIAN.name)

    assert np.array_equal(data.qpos, before), (
        "the camera wrote into the authoritative walking state")
    assert np.array_equal(data.ctrl, before_ctrl)

    # ... and it DID pose the head, in the copy, so the test is not vacuous.
    head_slots = [int(qpos_idx[HEAD_YAW_ACT]), int(qpos_idx[HEAD_PITCH_ACT]),
                  int(qpos_idx[HEAD_ROLL_ACT])]
    moved = any(camera.render_data.qpos[slot] != before[slot]
                for slot in head_slots)
    assert moved, "the camera did not actually aim the head"


def test_the_render_data_is_a_separate_object_from_the_walking_data(
        model, data, camera):
    assert camera.render_data is not data


def test_the_gaze_joints_stay_inside_their_declared_ranges(model, data, camera):
    qpos_idx, _ = actuator_indices(model)
    for yaw in (-3.0, -1.0, 0.0, 1.0, 3.0):
        camera.update(data, duck_yaw=yaw, subject=GUARDIAN.name)
        yaw_lo, yaw_hi = model.jnt_range[camera.head_yaw_joint]
        pitch_lo, pitch_hi = model.jnt_range[camera.head_pitch_joint]
        assert yaw_lo <= camera.gaze_yaw <= yaw_hi
        assert pitch_lo <= camera.gaze_pitch <= pitch_hi


def test_the_head_slews_at_a_bounded_rate_rather_than_snapping(
        model, data, camera):
    """Rates slow enough that a PiP is readable, and fast enough to hold a
    companion at arm's length."""
    camera.view_yaw = 0.0
    before = camera.view_yaw
    camera._aim_at(np.array([0.0, 5.0, 0.4]))
    step = abs(wrap_angle(camera.view_yaw - before))
    limit = math.radians(TRACK_YAW_RATE_DPS) * (1.0 / CTRL_HZ) / 0.02
    assert step <= limit + 1e-9
    assert step > 0.0, "the head must actually move toward the target"


def test_the_pitch_slews_more_slowly_than_the_yaw():
    assert TRACK_PITCH_RATE_DPS < TRACK_YAW_RATE_DPS
    assert TRACK_PITCH_DEG > 0.0


def test_each_person_is_sampled_at_five_points_scaled_by_their_stature(camera):
    for name in ALL_NAMES:
        points = camera.sample_points(name)
        assert len(points) == len(BASE_SAMPLE_DZ) == 5
        spread = float(points[-1][2] - points[0][2])
        expected = (max(BASE_SAMPLE_DZ) - min(BASE_SAMPLE_DZ)) \
            * BY_NAME[name].stature
        assert spread == pytest.approx(expected, abs=1e-9)


def test_the_guardian_is_visible_from_the_duck_real_start_pose(model, camera):
    """Measured from the pose the rollout actually starts in.

    The module-scope ``data`` fixture leaves the duck at the origin, which is
    not where this behavior begins; the start pose is behind her and off to her
    right, and that is the geometry the visibility gate opens on.
    """
    from beside_geometry import DUCK_START_XY, DUCK_START_YAW_DEG

    state = mujoco.MjData(model)
    mujoco.mj_resetData(model, state)
    state.qpos[0], state.qpos[1] = DUCK_START_XY
    half = math.radians(DUCK_START_YAW_DEG) * 0.5
    state.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
    pose_people(model, state, people_at(0.0), 0.0)
    mujoco.mj_forward(model, state)

    result = camera.update(state, duck_yaw=math.radians(DUCK_START_YAW_DEG),
                           subject=GUARDIAN.name)
    entry = result["people"][GUARDIAN.name]
    assert entry["visible"]
    assert 0 < entry["sample_count"] <= 5
    assert entry["range_m"] > 0.0
    assert GUARDIAN.name in result["visible_people"]


def test_a_person_behind_the_camera_is_not_reported_visible(
        model, data, camera):
    """The frustum test must have a depth term, or everything is 'visible'."""
    body = model.body(f"person_{GUARDIAN.name}")
    mocap = int(model.body_mocapid[body.id])
    original = data.mocap_pos[mocap].copy()
    try:
        eye = camera.render_data.cam_xpos[camera.beside_cam].copy()
        data.mocap_pos[mocap][:2] = (float(eye[0]) - 3.0, float(eye[1]))
        mujoco.mj_forward(model, data)
        state = camera.update(data, duck_yaw=0.0, subject="tomas")
        assert not state["people"][GUARDIAN.name]["visible"]
    finally:
        data.mocap_pos[mocap] = original
        mujoco.mj_forward(model, data)


def test_the_off_axis_angle_is_only_reported_for_samples_actually_seen(
        model, data, camera):
    """Reporting the off-axis angle of a sample the camera cannot see would let
    a gate open on somebody standing behind a column."""
    state = camera.update(data, duck_yaw=math.radians(24.0),
                          subject=GUARDIAN.name)
    for name, entry in state["people"].items():
        if entry["sample_count"] == 0:
            assert entry["off_axis_deg"] == pytest.approx(180.0, abs=1e-6)
        else:
            assert entry["off_axis_deg"] <= 180.0


def test_the_camera_reports_a_fraction_consistent_with_its_sample_count(
        model, data, camera):
    state = camera.update(data, duck_yaw=0.4, subject=GUARDIAN.name)
    for entry in state["people"].values():
        assert entry["fraction"] == pytest.approx(entry["sample_count"] / 5.0)
        assert entry["visible"] == (entry["sample_count"] > 0)
        assert len(entry["samples"]) == 5


def test_the_nominal_trunk_height_matches_the_gate_the_metrics_apply():
    """The final-height gate allows 0.012 m about this figure."""
    assert NOMINAL_TRUNK_Z == pytest.approx(0.116, abs=1e-9)
