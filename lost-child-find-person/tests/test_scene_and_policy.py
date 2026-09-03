#!/usr/bin/env python3
"""Scene, sensor and policy invariants, against the REAL compiled model.

These tests load the actual MuJoCo scene and the actual ONNX policy, so they are
slower than the pure-logic suite and they catch a different class of defect:
the observation being silently corrupted, a wrong sensor feeding the policy's
``base_ang_vel`` slot, and layout constants drifting away from the geometry they
claim to describe.

The head camera and the per-sample readability rules are graded in
``test_camera_and_constants``.
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
from lost_cast import ALL_NAMES, GUARDIAN  # noqa: E402
from lost_geometry import DUCK_PLANAR_RADIUS, DUCK_START_XY, DUCK_START_YAW_DEG  # noqa: E402
from lost_people import (  # noqa: E402
    GUARDIAN_ROUTE,
    LOOKALIKE_ROUTES,
    WALKERS,
    max_visible_jump,
    moving_fraction,
    people_at,
    pose_people,
)
from plaza_layout import OBSTACLES, clear_of_obstacles  # noqa: E402
from policy_runtime import (  # noqa: E402
    ACTION_SCALE,
    COMMAND_DIM,
    CTRL_HZ,
    DEFAULT_POSE,
    GYRO_SENSOR,
    NOMINAL_TRUNK_Z,
    OBS_DIM,
    PolicyRunner,
    build_observation,
    gyro_address,
    load_scene,
)
from rollout_lost import scenery_geom_names  # noqa: E402

POLICY = ROOT / "onnx" / "alpha_walking.onnx"
SCENE = ROOT / "assets" / "scene_lost_child.xml"
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


# ------------------------------------------------------------------- policy
def test_the_policy_is_the_byte_identical_stock_walking_policy():
    """Nothing about this behavior is allowed to retrain or fine-tune it."""
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest() == UPSTREAM_SHA


def test_the_action_scale_and_control_rate_are_the_shipped_values():
    assert ACTION_SCALE == 0.9
    assert CTRL_HZ == 50.0
    assert NOMINAL_TRUNK_Z == 0.116


def test_the_policy_expects_a_61_dimensional_observation():
    runner = PolicyRunner(POLICY)
    assert int(runner.session.get_inputs()[0].shape[1]) == OBS_DIM == 61


def test_the_observation_is_61_d_and_padded_exactly_as_documented():
    """ang_vel 3 + gravity 3 + joint_pos 14 + joint_vel 14 + action 14 + cmd 13."""
    observation = build_observation(
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
        np.zeros(14, dtype=np.float32),
        np.array([0.42, 0.0, -0.16], dtype=np.float32))
    assert observation.shape == (OBS_DIM,)
    assert 3 + 3 + 14 + 14 + 14 + COMMAND_DIM == OBS_DIM
    command = observation[-COMMAND_DIM:]
    assert list(command[:3]) == pytest.approx([0.42, 0.0, -0.16])
    assert np.all(command[3:] == 0.0), "the unused command slots must be zero"


def test_a_wrong_width_observation_is_refused_rather_than_reshaped():
    with pytest.raises(RuntimeError):
        build_observation(
            np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
            np.zeros(13, dtype=np.float32), np.zeros(14, dtype=np.float32),
            np.zeros(14, dtype=np.float32),
            np.zeros(3, dtype=np.float32))


def test_the_default_pose_is_the_fourteen_joint_stand_keyframe():
    assert DEFAULT_POSE.shape == (14,)
    assert DEFAULT_POSE.dtype == np.float32


# ------------------------------------------------------------------- sensor
def test_the_gyro_sensor_resolves_to_imu_ang_vel_and_not_the_last_sensor(model):
    """``mj_name2id`` returns -1 for an unknown name and -1 is a VALID index."""
    address = gyro_address(model)
    sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR)
    assert sensor >= 0
    assert address == int(model.sensor_adr[sensor])
    assert GYRO_SENSOR == "imu_ang_vel"


def test_any_other_sensor_name_is_refused_rather_than_silently_substituted(model):
    """A different quantity in the base_ang_vel slot invalidates every constant."""
    for name in ("gyro", "imu_gyro", "angular_velocity", "framequat", ""):
        with pytest.raises(ValueError):
            gyro_address(model, name)


def test_the_gyro_slot_carries_three_components(model):
    address = gyro_address(model)
    assert address + 3 <= model.nsensordata


# -------------------------------------------------------------------- scene
def test_the_scene_compiles_with_its_meshes_resolved(model):
    assert model.nmesh > 0
    assert model.nu == 14


def test_the_scene_xml_is_well_formed():
    import xml.dom.minidom
    document = xml.dom.minidom.parse(str(SCENE))
    assert document.documentElement.tagName == "mujoco"


def test_nothing_except_the_robot_can_collide(model):
    """Every wall, kiosk, column and person is kinematic scenery.

    Going round an occluder is therefore a property of the CONTROLLER, graded by
    measured surface clearance, not enforced by the contact solver.
    """
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
    assert model.nq == 7 + 14 + 4 * len(ALL_NAMES)
    for name in ALL_NAMES:
        assert int(model.body_mocapid[model.body(f"person_{name}").id]) >= 0


def test_the_clearance_gate_is_not_vacuous(model):
    """The gate collects scenery from the scene's own naming, not a hand list."""
    names = scenery_geom_names(model)
    assert len(names) >= 10
    assert any(n.startswith("wall_") for n in names)
    assert any(n.startswith("obs_") for n in names)
    assert "obs_kiosk" in names


def test_every_authored_obstacle_appears_in_the_compiled_scene(model):
    """A shape in the layout with no geom would make its gate unmeasurable."""
    names = set(scenery_geom_names(model))
    for obstacle in OBSTACLES:
        assert f"obs_{obstacle.name}" in names, obstacle.name


def test_the_duck_planar_radius_constant_matches_the_model(model, data):
    measured = duck_planar_radius(model, data, model.body("trunk_base").id)
    assert measured == pytest.approx(DUCK_PLANAR_RADIUS, abs=5e-4)


def test_the_bounding_sphere_radius_over_states_the_robot(model, data):
    """That is the safe direction for every clearance gate built on it."""
    trunk = model.body("trunk_base").id
    assert duck_planar_radius(model, data, trunk) > \
        exact_planar_radius(model, data, trunk)


def test_the_duck_start_pose_is_clear_of_every_obstacle():
    assert clear_of_obstacles(DUCK_START_XY, 0.20) is True
    assert 0.0 <= DUCK_START_YAW_DEG <= 360.0


def test_the_contact_probe_refuses_mesh_people(model):
    """The analytic probe is only valid for all-primitive bodies."""
    probe = ContactProbe(model, model.body("trunk_base").id, ALL_NAMES)
    assert set(probe.person_geoms) == set(ALL_NAMES)
    assert all(probe.person_geoms[name] for name in ALL_NAMES)


def test_the_wall_probe_refuses_mesh_scenery(model):
    probe = WallProbe(model, model.body("trunk_base").id,
                      scenery_geom_names(model))
    assert probe.wall_geoms
    mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
    assert all(int(model.geom_type[g]) != mesh for g in probe.wall_geoms)


def test_an_unknown_wall_geom_is_refused_rather_than_skipped(model):
    with pytest.raises(RuntimeError, match="not found"):
        WallProbe(model, model.body("trunk_base").id, ("no_such_geom",))


# ---------------------------------------------------------------- the crowd
def test_nobody_teleports_between_control_ticks():
    """Smootherstep legs, so a jump stays below one tick of ordinary walking."""
    jump, name, when = max_visible_jump(60.0)
    assert jump < 0.05, f"{name} jumped {jump:.4f} m at t={when}"


def test_every_adult_actually_walks_for_a_real_part_of_the_rollout():
    """A crowd that stops is not an occlusion problem; it is a diagram."""
    fractions = moving_fraction(60.0)
    assert set(fractions) == set(ALL_NAMES)
    for name in ALL_NAMES:
        if name == GUARDIAN.name:
            continue
        assert fractions[name] > 0.40, f"{name} barely moves"


def test_the_guardian_route_is_timed_across_the_whole_rollout():
    assert GUARDIAN_ROUTE[0].t == 0.0
    assert GUARDIAN_ROUTE[-1].t >= 60.0
    times = [waypoint.t for waypoint in GUARDIAN_ROUTE]
    assert times == sorted(times)


def test_the_guardian_route_passes_behind_the_kiosk(model):
    """The loss must be geometric: a solid body genuinely in the sightline."""
    from plaza_layout import KIOSK
    behind = [t * 0.1 for t in range(0, 600)
              if KIOSK.segment_hits((2.05, -1.05), WALKERS["priya"].pos_at(t * 0.1))]
    assert len(behind) > 20


def test_both_authored_look_alikes_have_their_own_timed_route():
    assert set(LOOKALIKE_ROUTES) == {"mira", "sofia"}
    for name, route in LOOKALIKE_ROUTES.items():
        assert route[0].t == 0.0
        assert route[-1].t >= 60.0


def test_every_person_in_the_cast_has_a_walker():
    assert set(WALKERS) == set(ALL_NAMES)


def test_nobody_walks_out_of_the_hall():
    from plaza_layout import FLOOR_HALF
    for step in range(0, 601):
        for name, state in people_at(step * 0.1).items():
            assert abs(float(state.pos[0])) <= FLOOR_HALF[0], name
            assert abs(float(state.pos[1])) <= FLOOR_HALF[1], name


