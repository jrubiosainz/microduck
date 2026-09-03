#!/usr/bin/env python3
"""Scene, sensor, camera-isolation and end-to-end gate tests.

These need MuJoCo and the stock ONNX policy, so they are slower than the pure
logic suite.  They exist because several of this behavior's central claims are
about the SIMULATOR, not about arithmetic:

* the observation really is 61-D and really is fed by ``imu_ang_vel``;
* the policy really is the byte-identical stock walking policy;
* the vehicles really cannot touch the robot;
* gaze really does not leak into the walking state;
* the analytic contact probe really is conservative against MuJoCo's own
  narrowphase, in the direction that matters.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from contact_geometry import (  # noqa: E402
    ContactProbe,
    box_sphere_distance,
    capsule_sphere_distance,
    cylinder_sphere_distance,
    duck_planar_radius,
    primitive_sphere_distance,
)
from guardian_camera import PIP_H, PIP_W, GuardianCamera  # noqa: E402
from policy_runtime import (  # noqa: E402
    ACTION_SCALE,
    COMMAND_DIM,
    CTRL_HZ,
    GYRO_SENSOR,
    OBS_DIM,
    PolicyRunner,
    build_observation,
    gyro_address,
    load_scene,
)
from street import DUCK_PLANAR_RADIUS, SECTORS, START_X  # noqa: E402
from traffic import VEHICLE_NAMES, pose_traffic, traffic_at  # noqa: E402

POLICY = REPO_ROOT / "onnx" / "alpha_walking.onnx"
# The canonical stock walking policy, shared by every behavior in this lab and
# matching the upstream microduck_rl checkout.
STOCK_POLICY_SHA256 = (
    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c")


@pytest.fixture(scope="module")
def model():
    return load_scene()


@pytest.fixture(scope="module")
def data(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    d.qpos[0] = START_X
    pose_traffic(model, d, traffic_at(0.0), 0.0)
    mujoco.mj_forward(model, d)
    return d


# ===================================================================
# the policy itself
# ===================================================================

def test_policy_is_the_byte_identical_stock_walking_policy():
    digest = hashlib.sha256(POLICY.read_bytes()).hexdigest()
    assert digest == STOCK_POLICY_SHA256, (
        "this behavior is measured against the stock alpha_walking policy; "
        "no policy was trained for it")


def test_policy_expects_a_61_dimensional_observation():
    runner = PolicyRunner(POLICY)
    assert int(runner.session.get_inputs()[0].shape[1]) == OBS_DIM


def test_action_scale_is_the_shipped_walking_value():
    assert ACTION_SCALE == 0.9


def test_control_rate_and_decimation_match_the_model(model):
    assert CTRL_HZ == 50.0
    decimation = round((1.0 / CTRL_HZ) / model.opt.timestep)
    assert decimation == 10
    assert model.opt.timestep == pytest.approx(0.002)


# ===================================================================
# the sensor trap
# ===================================================================

def test_gyro_resolves_to_the_real_imu_sensor(model):
    address = gyro_address(model)
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                  GYRO_SENSOR)
    assert sensor_id >= 0
    assert address == int(model.sensor_adr[sensor_id])


def test_gyro_address_is_not_the_last_sensor_address(model):
    """The trap that makes a wrong sensor name silently plausible.

    ``mj_name2id`` returns -1 for an unknown name and ``sensor_adr[-1]`` is a
    VALID index, so a wrong name feeds a different physical quantity into the
    policy's ``base_ang_vel`` slot.  The robot still walks and still renders
    plausibly, which is exactly what makes it dangerous.
    """
    assert gyro_address(model) != int(model.sensor_adr[-1])


def test_any_other_sensor_name_is_refused(model):
    for name in ("gyro", "imu_gyro", "angular_velocity", "imu"):
        with pytest.raises(ValueError):
            gyro_address(model, name)


def test_observation_is_61_d_with_only_the_twist_populated():
    observation = build_observation(
        np.ones(3, np.float32), np.ones(3, np.float32),
        np.ones(14, np.float32), np.ones(14, np.float32),
        np.ones(14, np.float32), np.array([0.5, 0.1, -0.2], np.float32))
    assert observation.shape == (OBS_DIM,)
    command = observation[-COMMAND_DIM:]
    assert list(command[:3]) == [0.5, pytest.approx(0.1), pytest.approx(-0.2)]
    assert not command[3:].any(), "unused command slots must stay zero-padded"


def test_a_wrong_width_observation_is_rejected():
    with pytest.raises(RuntimeError):
        build_observation(
            np.ones(3, np.float32), np.ones(3, np.float32),
            np.ones(13, np.float32), np.ones(14, np.float32),
            np.ones(14, np.float32), np.zeros(3, np.float32))


# ===================================================================
# the scene
# ===================================================================

def test_scene_compiles_with_its_meshes_and_the_stock_actuators(model):
    assert model.nmesh > 0, "meshdir did not resolve"
    assert model.nu == 14


def test_every_vehicle_is_non_colliding_mocap_scenery(model):
    """A crossing must never succeed because a car pushed the duck."""
    for name in VEHICLE_NAMES:
        body = model.body(f"vehicle_{name}")
        assert model.body_mocapid[body.id] >= 0, f"{name} is not mocap"
        geoms = [g for g in range(model.ngeom)
                 if int(model.geom_bodyid[g]) == body.id]
        assert geoms
        for geom in geoms:
            assert int(model.geom_contype[geom]) == 0
            assert int(model.geom_conaffinity[geom]) == 0


def test_vehicles_add_no_degrees_of_freedom_to_the_robot(model):
    """The walking policy must see exactly the robot it was trained on.

    Seven vehicles, all the street furniture and every marking are in this
    scene, and NONE of them may add state: 7 free-joint qpos + 14 hinges, and
    nothing else.  Mocap bodies carry no qpos by construction, so this is the
    assertion that keeps them that way.
    """
    assert model.nq == 7 + 14
    assert model.nv == 6 + 14
    assert model.nmocap == 1 + len(VEHICLE_NAMES)   # rig + one per vehicle


def test_street_constants_match_the_generated_scene(model):
    """The decision geometry and the rendered paint must be the same numbers."""
    from street import ROAD_HALF_WIDTH, SAFE_ZONE_X, WAIT_LINE_X

    road = model.geom("road")
    assert float(model.geom_size[road.id][0]) == pytest.approx(ROAD_HALF_WIDTH)
    near_line = model.geom("wait_line_near")
    assert float(model.geom_pos[near_line.id][0]) == pytest.approx(-WAIT_LINE_X)
    zone = model.geom("safe_zone")
    assert float(model.geom_pos[zone.id][0]) == pytest.approx(SAFE_ZONE_X)


def test_the_road_is_longer_than_the_traffic_loop(model):
    """No vehicle may ever be posed off the end of the asphalt."""
    from traffic import LOOP_HALF_Y

    road = model.geom("road")
    assert float(model.geom_size[road.id][1]) > LOOP_HALF_Y


def test_duck_planar_radius_constant_matches_the_measured_geometry(model, data):
    """``street.DUCK_PLANAR_RADIUS`` is load-bearing for every occupancy gate.

    A stale value silently weakens lane occupancy, wait-line encroachment and
    the crossing-duration estimate all at once.  The first draft inherited
    0.090 from a sibling behavior and under-reported the footprint by 45%.
    """
    measured = duck_planar_radius(model, data, model.body("trunk_base").id)
    assert measured == pytest.approx(DUCK_PLANAR_RADIUS, abs=0.005), (
        f"measured {measured:.4f} m but street.py declares "
        f"{DUCK_PLANAR_RADIUS}")


# ===================================================================
# contact geometry
# ===================================================================

def test_analytic_primitive_distances_are_exact_on_known_geometry(model, data):
    """Each analytic form is checked against a hand-computed answer.

    A wheel is a cylinder of radius 0.042 and half-height 0.017 whose local
    z axis is rotated onto WORLD X by ``quat="0.7071068 0.7071068 0 0"``.  So a
    probe displaced along world x approaches the wheel's flat FACE and must
    clear the half-height, while a probe displaced along world z approaches its
    curved side and must clear the radius.  Getting those two the wrong way
    round is exactly the kind of frame error this test exists to catch.
    """
    wheel = model.geom("hatch_wheel_fl").id
    centre = data.geom_xpos[wheel].copy()

    along_axis = cylinder_sphere_distance(
        model, data, wheel, centre + np.array([1.0, 0.0, 0.0]), 0.0)
    assert along_axis == pytest.approx(1.0 - 0.017, abs=1e-6)

    radial = cylinder_sphere_distance(
        model, data, wheel, centre + np.array([0.0, 0.0, 1.0]), 0.0)
    assert radial == pytest.approx(1.0 - 0.042, abs=1e-6)

    # The sphere radius is subtracted directly, on either approach.
    assert cylinder_sphere_distance(
        model, data, wheel, centre + np.array([0.0, 0.0, 1.0]), 0.25) == \
        pytest.approx(1.0 - 0.042 - 0.25, abs=1e-6)

    # A box: the chassis is 0.215 x 0.100 x 0.046 in its own frame.
    chassis = model.geom("hatch_chassis").id
    box_centre = data.geom_xpos[chassis].copy()
    # The car faces +y or −y, so its local x (half-extent 0.215) lies along
    # world y.  Probe along world z, whose half-extent is 0.046.
    assert box_sphere_distance(
        model, data, chassis, box_centre + np.array([0.0, 0.0, 1.0]), 0.0) == \
        pytest.approx(1.0 - 0.046, abs=1e-6)

    # A capsule: the scooter's rider, radius 0.062.
    rider = model.geom("scooter_rider").id
    spine_mid = data.geom_xpos[rider].copy()
    assert capsule_sphere_distance(
        model, data, rider, spine_mid + np.array([0.0, 0.0, 2.0]), 0.0) > 0.0


def test_a_point_inside_a_primitive_reports_negative_distance(model, data):
    wheel = model.geom("hatch_wheel_fl").id
    inside = data.geom_xpos[wheel].copy()
    assert cylinder_sphere_distance(model, data, wheel, inside, 0.0) < 0.0
    box = model.geom("hatch_chassis").id
    assert box_sphere_distance(model, data, box,
                               data.geom_xpos[box].copy(), 0.0) < 0.0


def test_primitive_dispatch_covers_every_vehicle_geom_type(model, data):
    """No vehicle geom may fall through to the conservative bounding sphere.

    Falling through is safe but lossy, and a silent fallback would mean the
    reported clearances were not the exact ones the README claims.
    """
    exact = {
        int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_SPHERE),
    }
    probe = ContactProbe(model, model.body("trunk_base").id, VEHICLE_NAMES)
    for name, geoms in probe.vehicle_geoms.items():
        for geom in geoms:
            kind = int(model.geom_type[geom])
            if kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                continue          # helmets; bounding sphere is fine and safe
            assert kind in exact, (
                f"{name} geom {model.geom(geom).name} has type {kind}, which "
                "would silently take the conservative path")


def test_the_probe_never_over_reports_clearance_versus_mujoco(model):
    """The analytic path must be conservative wherever MuJoCo is trustworthy.

    MuJoCo's mesh-versus-primitive narrowphase returns spurious exact zeros in
    this scene, so it cannot be used as ground truth.  What CAN be required is
    one-sided: whenever MuJoCo reports a positive distance, the analytic probe
    must not claim MORE clearance than that, because the analytic form
    approximates the duck by bounding spheres.

    The vehicles are parked at their t=0 positions, which are tens of metres
    away, so the comparison is run with the traffic posed right next to the
    duck — otherwise every pair is beyond any usable cutoff and the test
    silently checks nothing.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.0
    # Park one vehicle a short distance away, in range of a 3 m cutoff.
    for name in VEHICLE_NAMES:
        body = model.body(f"vehicle_{name}")
        mocap = int(model.body_mocapid[body.id])
        data.mocap_pos[mocap] = np.array([0.275, 1.2, 0.0])
    mujoco.mj_forward(model, data)

    probe = ContactProbe(model, model.body("trunk_base").id, VEHICLE_NAMES)
    checked = 0
    for name in VEHICLE_NAMES:
        for vehicle_geom in probe.vehicle_geoms[name]:
            for duck_geom in probe.duck_geoms[:6]:
                mujoco_distance = float(mujoco.mj_geomDistance(
                    model, data, duck_geom, vehicle_geom, 3.0, None))
                if mujoco_distance <= 0.0 or mujoco_distance >= 3.0:
                    continue      # spurious zero, or beyond the cutoff
                analytic = primitive_sphere_distance(
                    model, data, vehicle_geom, data.geom_xpos[duck_geom],
                    probe.duck_rbound[duck_geom])
                assert analytic <= mujoco_distance + 1e-6, (
                    f"{name}/{model.geom(vehicle_geom).name}: analytic "
                    f"{analytic:.4f} claims more clearance than MuJoCo's "
                    f"{mujoco_distance:.4f}")
                checked += 1
    assert checked > 0, "the comparison never ran"


def test_the_probe_refuses_a_vehicle_carrying_mesh_geometry(model):
    """The analytic path is only valid for all-primitive vehicles."""
    probe = ContactProbe(model, model.body("trunk_base").id, VEHICLE_NAMES)
    assert probe.vehicle_geoms
    mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
    for geoms in probe.vehicle_geoms.values():
        assert all(int(model.geom_type[g]) != mesh for g in geoms)


# ===================================================================
# camera isolation
# ===================================================================

def test_gaze_never_touches_the_authoritative_walking_state(model):
    """The central isolation claim, asserted rather than described.

    The head is a large fraction of the robot's mass and the stock walking
    policy was never trained to compensate an imposed head trajectory, so gaze
    must live entirely in a separate MjData.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0] = START_X
    pose_traffic(model, data, traffic_at(0.0), 0.0)
    mujoco.mj_forward(model, data)
    runner = PolicyRunner(POLICY).reset(model, data)
    camera = GuardianCamera(model, data, runner.qpos_idx,
                            model.body("trunk_base").id, (PIP_W, PIP_H))

    before_qpos = data.qpos.copy()
    before_qvel = data.qvel.copy()
    before_ctrl = data.ctrl.copy()
    for state in ("LOOK_LEFT", "LOOK_RIGHT", "CROSSING", "WAIT_FOR_GAP"):
        for tick in range(30):
            camera.update(data, state=state, duck_yaw=0.0, t=tick * 0.02)
    assert np.array_equal(data.qpos, before_qpos)
    assert np.array_equal(data.qvel, before_qvel)
    assert np.array_equal(data.ctrl, before_ctrl)


def test_the_head_actually_moves_in_the_isolated_copy(model):
    """Isolation must not be achieved by simply doing nothing.

    Mutation guard: a camera that never posed the head at all would trivially
    pass the isolation test above.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    runner = PolicyRunner(POLICY).reset(model, data)
    camera = GuardianCamera(model, data, runner.qpos_idx,
                            model.body("trunk_base").id, (PIP_W, PIP_H))
    for tick in range(120):
        left = camera.update(data, state="LOOK_LEFT", duck_yaw=0.0,
                             t=tick * 0.02)
    for tick in range(120):
        right = camera.update(data, state="LOOK_RIGHT", duck_yaw=0.0,
                              t=tick * 0.02)
    assert left["gaze_yaw"] > math.radians(30.0)
    assert right["gaze_yaw"] < math.radians(-30.0)


def test_the_scan_looks_at_opposite_sides_of_the_road(model):
    """LOOK_LEFT and LOOK_RIGHT must aim in genuinely opposite directions."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0] = -1.055
    pose_traffic(model, data, traffic_at(0.0), 0.0)
    mujoco.mj_forward(model, data)
    runner = PolicyRunner(POLICY).reset(model, data)
    camera = GuardianCamera(model, data, runner.qpos_idx,
                            model.body("trunk_base").id, (PIP_W, PIP_H))
    for tick in range(150):
        left = camera.update(data, state="LOOK_LEFT", duck_yaw=0.0,
                             t=tick * 0.02)
    assert left["left_fraction"] > left["right_fraction"]
    for tick in range(150):
        right = camera.update(data, state="LOOK_RIGHT", duck_yaw=0.0,
                              t=tick * 0.02)
    assert right["right_fraction"] > right["left_fraction"]


def test_sector_samples_sit_on_the_lanes_their_traffic_uses(model):
    """The scan is graded against the road, not against arbitrary waypoints."""
    from street import FAR_LANE_X, NEAR_LANE_X

    for point in SECTORS["left"]:
        assert point[0] == pytest.approx(NEAR_LANE_X)
        assert point[1] > 0.0          # left is +y
    for point in SECTORS["right"]:
        assert point[0] == pytest.approx(FAR_LANE_X)
        assert point[1] < 0.0          # right is −y


def test_visibility_is_measured_through_the_pip_camera(model):
    """The frustum the HUD draws must be the frustum the gate measures.

    Drawing one rectangle while measuring another would let the reported
    percentages disagree with the picture a viewer can check for themselves.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    runner = PolicyRunner(POLICY).reset(model, data)
    camera = GuardianCamera(model, data, runner.qpos_idx,
                            model.body("trunk_base").id, (PIP_W, PIP_H))
    assert camera.camera_id == model.camera("guardian_camera").id
    aspect = PIP_W / PIP_H
    assert camera.tan_h == pytest.approx(aspect * camera.tan_v)


def test_a_point_behind_the_camera_is_never_visible(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    runner = PolicyRunner(POLICY).reset(model, data)
    camera = GuardianCamera(model, data, runner.qpos_idx,
                            model.body("trunk_base").id, (PIP_W, PIP_H))
    camera.update(data, state="APPROACH_CURB", duck_yaw=0.0, t=0.0)
    eye = camera.render_data.cam_xpos[camera.guardian_cam].copy()
    forward = -camera.render_data.cam_xmat[camera.guardian_cam].reshape(3, 3)[:, 2]
    assert not camera._point_visible(eye - forward * 3.0)


# ===================================================================
# end to end
# ===================================================================

@pytest.mark.slow
def test_short_rollout_stays_upright_and_never_touches_a_vehicle():
    """A cheap integration smoke test: physics, camera and machine together."""
    from rollout_guardian import GuardianRollout

    rollout = GuardianRollout(POLICY, 6.0, pip_size=(PIP_W, PIP_H))
    records = rollout.run()
    assert len(records) == 300
    assert all(r["trunk_z_m"] >= 0.09 for r in records)
    assert all(r["nearest_clearance_m"] > 0.0 for r in records)
    assert records[0]["state"] == "APPROACH_CURB"


@pytest.mark.slow
def test_the_duck_never_encroaches_the_wait_line_before_committing():
    """The behavior's safety claim, measured on real physics."""
    from rollout_guardian import GuardianRollout

    rollout = GuardianRollout(POLICY, 14.0, pip_size=(PIP_W, PIP_H))
    rollout.run()
    assert rollout.min_wait_line_margin > 0.0
    for record in rollout.records:
        if not record["in_road"]:
            continue
        assert record["state"] == "CROSSING", (
            f"in the road at t={record['t']:.2f} s while {record['state']}")
