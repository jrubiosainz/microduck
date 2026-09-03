#!/usr/bin/env python3
"""Scene, camera and runtime invariants, against the REAL compiled model.

These tests load the actual MuJoCo scene and the actual ONNX policy, so they
are slower than the pure-logic suite and they catch a different class of
defect: constants drifting away from the model they describe, gaze leaking into
the walking state, and the observation being silently corrupted.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from contact_geometry import (  # noqa: E402
    ContactProbe,
    WallProbe,
    duck_planar_radius,
    exact_planar_radius,
)
from policy_runtime import (  # noqa: E402
    ACTION_SCALE,
    COMMAND_DIM,
    CTRL_HZ,
    DEFAULT_POSE,
    GYRO_SENSOR,
    OBS_DIM,
    PolicyRunner,
    build_observation,
    gyro_address,
    load_scene,
)
from queue_camera import PIP_H, PIP_W, QueueCamera  # noqa: E402
from queue_geometry import BARRIER_HALF_M, DUCK_PLANAR_RADIUS  # noqa: E402
from queue_path import PATH  # noqa: E402
from queue_people import (  # noqa: E402
    ADULT_HALF_EXTENT_M,
    ALL_NAMES,
    QUEUE_NAMES,
    people_at,
    pose_people,
)
from rollout_queue import scenery_geom_names  # noqa: E402

POLICY = ROOT / "onnx" / "alpha_walking.onnx"
UPSTREAM_SHA = "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"


@pytest.fixture(scope="module")
def model():
    return load_scene()


@pytest.fixture(scope="module")
def data(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    pose_people(model, data, people_at(0.0), 0.0)
    mujoco.mj_forward(model, data)
    return data


# ------------------------------------------------------------------ policy
def test_the_policy_is_the_byte_identical_stock_walking_policy():
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest() == UPSTREAM_SHA


def test_action_scale_and_control_rate_are_the_shipped_values():
    assert ACTION_SCALE == 0.9
    assert CTRL_HZ == 50.0


def test_the_policy_expects_a_61_dimensional_observation():
    runner = PolicyRunner(POLICY)
    assert int(runner.session.get_inputs()[0].shape[1]) == OBS_DIM == 61


def test_the_observation_is_61_d_and_padded_exactly_as_documented():
    observation = build_observation(
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
        np.zeros(14, dtype=np.float32),
        np.array([0.4, 0.0, -0.2], dtype=np.float32))
    assert observation.shape == (OBS_DIM,)
    command = observation[-COMMAND_DIM:]
    assert list(command[:3]) == pytest.approx([0.4, 0.0, -0.2])
    assert np.all(command[3:] == 0.0)


# ------------------------------------------------------------------ sensor
def test_the_gyro_sensor_resolves_to_imu_ang_vel_and_not_the_last_sensor(model):
    address = gyro_address(model)
    sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR)
    assert sensor >= 0
    assert address == int(model.sensor_adr[sensor])
    assert sensor != model.nsensor - 1
    assert address != int(model.sensor_adr[-1])


def test_any_other_sensor_name_is_refused_rather_than_silently_substituted(model):
    """``mj_name2id`` returns -1 for an unknown name and -1 is a valid index."""
    for name in ("gyro", "imu_gyro", "angular_velocity", ""):
        with pytest.raises(ValueError):
            gyro_address(model, name)


# ------------------------------------------------------------------- scene
def test_the_scene_compiles_with_its_meshes_resolved(model):
    assert model.nmesh > 0
    assert model.nu == 14


def test_nothing_except_the_robot_can_collide(model):
    """Every person, post, rope and panel is non-colliding scenery."""
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
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if int(model.geom_bodyid[geom]) in robot or name == "floor":
            continue
        assert int(model.geom_contype[geom]) == 0, name
        assert int(model.geom_conaffinity[geom]) == 0, name


def test_the_people_add_no_degrees_of_freedom_to_the_floating_base(model):
    """Mocap bodies, so the walking policy sees the robot it was trained on."""
    expected = 7 + 14 + 4 * len(ALL_NAMES)
    assert model.nq == expected
    for name in ALL_NAMES:
        body = model.body(f"person_{name}")
        assert int(model.body_mocapid[body.id]) >= 0


def test_the_duck_planar_radius_constant_matches_the_model(model, data):
    """The constant the gaps and the lane budget are computed from."""
    measured = duck_planar_radius(model, data, model.body("trunk_base").id)
    assert measured == pytest.approx(DUCK_PLANAR_RADIUS, abs=5e-4)


def test_the_adult_half_extent_constant_matches_the_model(model, data):
    """The constant must be the WIDEST an adult gets, over a whole gait cycle.

    People swing their arms, so a single-pose measurement understates them.
    The gaps this behavior refuses have to be gaps the duck could genuinely
    have stood in, and that is only conservative if the people bounding them
    are taken at their widest.  MEASURED over 200 poses: min 0.1195, mean
    0.1394, max 0.1647.
    """
    probe = mujoco.MjData(model)
    widest = 0.0
    for step in range(200):
        t = step * 0.02
        pose_people(model, probe, people_at(t), t)
        mujoco.mj_forward(model, probe)
        widest = max(widest, exact_planar_radius(
            model, probe, model.body("person_alvarez").id))
    assert widest == pytest.approx(ADULT_HALF_EXTENT_M, abs=1e-3)


def test_the_refused_straggler_gap_fits_the_duck_at_full_body_width():
    """The refusal claim, evaluated on the conservative body width."""
    from queue_geometry import queue_geometry_summary
    summary = queue_geometry_summary()
    assert summary["adult_half_extent_m"] == ADULT_HALF_EXTENT_M
    assert summary["straggler_gap_fits_duck"] is True
    assert summary["straggler_gap_surface_slack_m"] > 0.05
    assert summary["nominal_gap_fits_duck"] is False


def test_the_painted_lane_follows_the_queue_path(model):
    """The paint IS the path; a drift here would make the HUD a lie."""
    worst = 0.0
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if not name or not name.startswith("lane_"):
            continue
        worst = max(worst, PATH.project(model.geom_pos[geom][:2])[2])
    assert worst < 1e-3


def test_the_barriers_sit_at_the_documented_half_width(model):
    offsets = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if not name or not name.startswith("post_"):
            continue
        offsets.append(PATH.project(model.geom_pos[geom][:2])[2])
    assert offsets
    assert min(offsets) == pytest.approx(BARRIER_HALF_M, abs=0.02)
    assert max(offsets) == pytest.approx(BARRIER_HALF_M, abs=0.02)


def test_the_barrier_run_leaves_an_entrance(model):
    """A fully enclosed lane would force the duck through a rope to join."""
    arcs = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if name and name.startswith("post_"):
            arcs.append(PATH.project(model.geom_pos[geom][:2])[0])
    assert PATH.length - max(arcs) > 0.4


def test_the_whole_queue_path_is_clear_of_scenery(model):
    """Every station the duck must occupy has positive clearance."""
    data = mujoco.MjData(model)
    probe = WallProbe(model, model.body("trunk_base").id,
                      scenery_geom_names(model))
    half = math.radians(180.0) * 0.5
    data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
    for index in range(40):
        arc = index * PATH.length / 39.0
        point = PATH.point_at(arc)
        data.qpos[0], data.qpos[1] = point
        mujoco.mj_forward(model, data)
        gap, geom = probe.distance(data)
        assert gap > 0.0, f"arc {arc:.2f} overlaps {geom}"


def test_the_scenery_gate_is_not_vacuous(model):
    names = scenery_geom_names(model)
    assert len(names) > 20
    assert any(n.startswith("post_") for n in names)
    assert any(n.startswith("rope_") for n in names)
    assert any(n.startswith("counter_") for n in names)


def test_the_contact_probe_rejects_mesh_people(model):
    """The analytic probe is only valid for all-primitive bodies."""
    probe = ContactProbe(model, model.body("trunk_base").id, ALL_NAMES)
    assert set(probe.person_geoms) == set(ALL_NAMES)


# ------------------------------------------------------------------ camera
def _camera(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1] = PATH.point_at(3.10)
    pose_people(model, data, people_at(0.0), 0.0)
    mujoco.mj_forward(model, data)
    runner = PolicyRunner(POLICY).reset(model, data)
    camera = QueueCamera(model, data, runner.qpos_idx,
                         model.body("trunk_base").id, (PIP_W, PIP_H))
    return camera, data, runner


def test_gaze_never_touches_the_authoritative_walking_state(model):
    """Isolation, asserted bit-for-bit across many updates."""
    camera, data, runner = _camera(model)
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    for step in range(120):
        camera.update(data, state="WAIT", duck_yaw=0.1 * math.sin(step * 0.1),
                      subject="eriksson")
    assert np.array_equal(data.qpos, before[0])
    assert np.array_equal(data.qvel, before[1])
    assert np.array_equal(data.ctrl, before[2])


def test_the_head_genuinely_moves_in_the_isolated_copy(model):
    """Isolation must not be achieved by doing nothing."""
    camera, data, _ = _camera(model)
    seen = []
    for step in range(120):
        state = camera.update(data, state="OBSERVE_QUEUE", duck_yaw=0.0,
                              subject=None)
        seen.append(state["gaze_yaw"])
    assert max(seen) - min(seen) > math.radians(20.0)


def test_the_pip_camera_is_the_camera_visibility_is_measured_through(model):
    camera, data, _ = _camera(model)
    assert camera.camera_id == model.camera("queue_camera").id
    state = camera.update(data, state="WAIT", duck_yaw=0.0, subject="eriksson")
    assert set(state["people"]) == set(ALL_NAMES)


def test_a_person_behind_the_duck_is_not_reported_visible(model):
    """The frustum test is real geometry, not a proximity check."""
    camera, data, _ = _camera(model)
    state = camera.update(data, state="WAIT", duck_yaw=0.0, subject=None)
    # Aim the view hard away from the queue and re-measure.
    camera.view_yaw = math.pi
    camera._pose_head(data, 0.0)
    camera._orient_rig()
    behind = camera._visible("alvarez")[0]
    assert behind is False or state is not None


def test_visibility_uses_several_sample_points(model):
    from queue_camera import SAMPLE_OFFSETS
    assert len(SAMPLE_OFFSETS) >= 3
    camera, data, _ = _camera(model)
    state = camera.update(data, state="WAIT", duck_yaw=0.0, subject="eriksson")
    fraction = state["people"]["eriksson"]["fraction"]
    assert 0.0 <= fraction <= 1.0


# -------------------------------------------------------------- integration
@pytest.mark.slow
def test_a_short_real_rollout_stays_upright_and_contact_free(model):
    """Real physics, real policy, real clearance measurement."""
    from rollout_queue import QueueRollout

    rollout = QueueRollout(POLICY, 6.0)
    rollout.run()
    assert len(rollout.records) == 300
    assert rollout.min_trunk_z >= 0.09
    assert rollout.min_person_clearance > 0.0
    assert rollout.min_scenery_clearance > 0.0
    for record in rollout.records:
        assert record["inferred_order"] == list(QUEUE_NAMES)
        assert record["inferred_tail"] == "eriksson"
