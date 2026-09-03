#!/usr/bin/env python3
"""Scene, camera and gate-summary tests against the real MuJoCo model.

These load the actual corridor scene and the stock ONNX policy, so they are
slower than the pure-logic suite and are marked ``slow``.  They exist to catch
the class of defect a pure unit test cannot: a constant that has drifted away
from the geometry it claims to describe, a camera that measures through a
different frustum from the one it renders, or a gaze layer that quietly writes
back into the walking state.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tools"))

from contact_geometry import (  # noqa: E402
    ContactProbe,
    WallProbe,
    box_sphere_distance,
    duck_planar_radius,
    exact_lateral_half_width,
    exact_planar_radius,
)
from corridor import (  # noqa: E402
    ADULT_LATERAL_HALF,
    ALCOVES,
    ALCOVE_BY_NAME,
    CENTER_PASSAGE_HALF,
    CLEAR_ABS_Y,
    CORRIDOR_HALF_WIDTH,
    CORRIDOR_X_MAX,
    CORRIDOR_X_MIN,
    DESTINATION_X,
    DUCK_LATERAL_HALF,
    DUCK_PLANAR_RADIUS,
    SAFE_PASSING_GAP_M,
    START_X,
)
from encounter import CLEAR_RANGE_M, VY_SPEED_MPS  # noqa: E402
from etiquette_camera import (  # noqa: E402
    PERSON_SAMPLE_Z,
    PIP_H,
    PIP_W,
    EtiquetteCamera,
)
from people import PERSON_NAMES, people_at, pose_people  # noqa: E402
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
from rollout_etiquette import wall_geom_names  # noqa: E402

POLICY = REPO / "onnx" / "alpha_walking.onnx"
# The stock walking policy from microduck_rl, byte-identical.
POLICY_SHA256 = (
    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    return load_scene()


@pytest.fixture(scope="module")
def data(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1] = START_X, 0.0
    pose_people(model, data, people_at(0.0), 0.0)
    mujoco.mj_forward(model, data)
    return data


# ------------------------------------------------------------------- policy
class TestPolicyIdentity:
    def test_the_policy_is_the_stock_walking_network(self):
        import hashlib

        digest = hashlib.sha256(POLICY.read_bytes()).hexdigest()
        assert digest == POLICY_SHA256, (
            "every measured constant in this behavior was taken against the "
            "stock policy; a different network invalidates all of them")

    def test_the_shipped_action_scale_is_used(self):
        assert ACTION_SCALE == 0.9

    def test_the_control_rate_is_fifty_hertz(self):
        assert CTRL_HZ == 50.0

    def test_the_observation_is_sixty_one_dimensional(self):
        observation = build_observation(
            np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
            np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
            np.zeros(14, dtype=np.float32), np.zeros(3, dtype=np.float32))
        assert observation.shape == (OBS_DIM,)
        assert OBS_DIM == 3 + 3 + 14 + 14 + 14 + COMMAND_DIM

    def test_the_policy_accepts_that_observation(self):
        runner = PolicyRunner(POLICY)
        width = int(runner.session.get_inputs()[0].shape[1])
        assert width == OBS_DIM

    def test_only_the_real_gyro_sensor_is_accepted(self, model):
        with pytest.raises(ValueError):
            gyro_address(model, "accelerometer")
        with pytest.raises(ValueError):
            gyro_address(model, "gyro")

    def test_the_gyro_resolves_to_a_real_distinct_address(self, model):
        address = gyro_address(model)
        assert address >= 0
        sensor = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR)
        assert sensor >= 0
        assert int(model.sensor_adr[sensor]) == address


# -------------------------------------------------------------------- scene
class TestScene:
    def test_the_scene_compiles_with_meshes_and_actuators(self, model):
        assert model.nmesh > 0, "meshdir did not resolve"
        assert model.nu == 14

    def test_the_robot_is_the_only_dynamic_body(self, model):
        """7 free-joint qpos + 14 hinges + 4 hinges per scripted adult."""
        assert model.nq == 7 + 14 + 4 * len(PERSON_NAMES)

    def test_every_person_is_mocap_and_non_colliding(self, model):
        for name in PERSON_NAMES:
            body = model.body(f"person_{name}")
            assert int(model.body_mocapid[body.id]) >= 0
            for geom in range(model.ngeom):
                if int(model.geom_bodyid[geom]) == body.id:
                    assert int(model.geom_contype[geom]) == 0
                    assert int(model.geom_conaffinity[geom]) == 0

    def test_the_walls_are_non_colliding_by_design(self, model):
        """Staying inside the corridor must be the controller's achievement."""
        for name in wall_geom_names(model):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert int(model.geom_contype[geom]) == 0
            assert int(model.geom_conaffinity[geom]) == 0

    def test_the_clearance_gate_covers_real_surfaces(self, model):
        names = wall_geom_names(model)
        assert len(names) >= 12
        # every alcove contributes a back and two cheeks
        for alcove in ALCOVES:
            assert f"{alcove.name}_back" in names
            assert f"{alcove.name}_cheek_lo" in names
            assert f"{alcove.name}_cheek_hi" in names

    def test_the_crates_are_in_the_clearance_gate(self, model):
        names = wall_geom_names(model)
        assert any("crate" in name for name in names), (
            "the obstruction must be a surface the duck can hit, or the "
            "blocked bay is only nominally blocked")

    def test_the_corridor_walls_sit_where_the_geometry_says(self, model, data):
        """The scene and corridor.py must not drift apart."""
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_plus_0")
        assert geom >= 0
        face = abs(float(data.geom_xpos[geom][1])) - float(
            model.geom_size[geom][1])
        assert face == pytest.approx(CORRIDOR_HALF_WIDTH, abs=1e-6)

    def test_alcove_backs_sit_at_their_declared_depth(self, model, data):
        for alcove in ALCOVES:
            geom = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{alcove.name}_back")
            face = abs(float(data.geom_xpos[geom][1])) - float(
                model.geom_size[geom][1])
            assert face == pytest.approx(alcove.outer_y, abs=1e-6)

    def test_the_crates_begin_where_the_scorer_thinks(self, model, data):
        alcove = ALCOVE_BY_NAME["bay_crates"]
        geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{alcove.name}_crate_0")
        assert geom >= 0
        inner = abs(float(data.geom_xpos[geom][1])) - float(
            model.geom_size[geom][1])
        assert inner == pytest.approx(alcove.blocked_from, abs=1e-6)

    def test_the_duck_starts_on_the_centreline_in_the_corridor(self, data, model):
        trunk = model.body("trunk_base").id
        assert float(data.xpos[trunk][0]) == pytest.approx(START_X, abs=1e-6)
        assert abs(float(data.xpos[trunk][1])) < 1e-6
        assert CORRIDOR_X_MIN < START_X < CORRIDOR_X_MAX

    def test_no_scenery_obstructs_the_corridor_itself(self, model, data):
        """The pipes-across-the-shot defect, as a check rather than a memory.

        Nothing that is not a wall, a person or the robot may occupy the volume
        above the corridor floor between the wall faces.
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
        people_bodies = set()
        for name in PERSON_NAMES:
            root = model.body(f"person_{name}").id
            for body in range(model.nbody):
                parent = body
                while parent > 0:
                    if parent == root:
                        people_bodies.add(body)
                        break
                    parent = int(model.body_parentid[parent])
            people_bodies.add(root)

        offenders = []
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            if body in robot or body in people_bodies:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if name.startswith(("bay_", "centreline", "passage_", "destination",
                                "corridor_floor")):
                continue
            pos = data.geom_xpos[geom]
            if not (CORRIDOR_X_MIN <= float(pos[0]) <= CORRIDOR_X_MAX):
                continue
            half_y = float(model.geom_size[geom][1])
            if abs(float(pos[1])) - half_y < CORRIDOR_HALF_WIDTH - 0.02 and \
                    float(pos[2]) > 0.02:
                offenders.append(name)
        assert not offenders, (
            f"these geoms sit inside the corridor volume and will occlude the "
            f"subject: {offenders}")


# ------------------------------------------------------------------- radii
class TestMeasuredRadii:
    def test_the_duck_radius_constant_matches_the_model(self, model, data):
        """The constant must never UNDER-state the robot.

        Under-stating a footprint radius weakens corridor clearance, passage
        intrusion and alcove fit all at once, so the test is one-sided: the
        constant must be at least what the model reports, and close enough that
        it is still describing this robot.
        """
        trunk = model.body("trunk_base").id
        measured = duck_planar_radius(model, data, trunk)
        assert DUCK_PLANAR_RADIUS >= measured - 1e-4, (
            "a radius below the measured envelope silently weakens every "
            "footprint gate")
        assert DUCK_PLANAR_RADIUS - measured < 0.03, (
            "and one far above it is no longer describing this robot")

    def test_the_duck_lateral_half_matches_the_model(self, model, data):
        trunk = model.body("trunk_base").id
        measured = exact_lateral_half_width(model, data, trunk)
        assert DUCK_LATERAL_HALF == pytest.approx(measured, abs=0.006), (
            "the passing premise is graded on this number")

    def test_the_adult_lateral_half_matches_the_model(self, model, data):
        body = model.body(f"person_{PERSON_NAMES[0]}").id
        mocap = int(model.body_mocapid[body])
        data.mocap_pos[mocap] = [0.0, 0.0, 0.36]
        mujoco.mj_forward(model, data)
        measured = exact_lateral_half_width(model, data, body)
        assert ADULT_LATERAL_HALF == pytest.approx(measured, abs=0.005)

    def test_the_bounding_radius_over_states_the_robot(self, model, data):
        """Which is the safe direction for every footprint gate."""
        trunk = model.body("trunk_base").id
        assert duck_planar_radius(model, data, trunk) > exact_planar_radius(
            model, data, trunk)

    def test_the_exact_lateral_half_is_the_narrowest_measure(self, model, data):
        trunk = model.body("trunk_base").id
        assert exact_lateral_half_width(model, data, trunk) <= (
            exact_planar_radius(model, data, trunk) + 1e-9)


# ------------------------------------------------------------------ probes
class TestClearanceProbes:
    def test_the_wall_probe_finds_a_real_overlap(self, model, data):
        """Push the duck into a wall and require the probe to report it."""
        trunk = model.body("trunk_base").id
        probe = WallProbe(model, trunk, wall_geom_names(model))
        clear, _ = probe.distance(data)
        assert clear > 0.0
        data.qpos[1] = CORRIDOR_HALF_WIDTH
        mujoco.mj_forward(model, data)
        overlapped, geom = probe.distance(data)
        assert overlapped < 0.0
        assert geom
        data.qpos[1] = 0.0
        mujoco.mj_forward(model, data)

    def test_the_person_probe_finds_a_real_overlap(self, model, data):
        trunk = model.body("trunk_base").id
        probe = ContactProbe(model, trunk, PERSON_NAMES)
        name = PERSON_NAMES[0]
        mocap = int(model.body_mocapid[model.body(f"person_{name}").id])
        far = data.mocap_pos[mocap].copy()
        assert probe.distance(data, name) > 0.0
        data.mocap_pos[mocap] = [float(data.xpos[trunk][0]),
                                 float(data.xpos[trunk][1]), 0.36]
        mujoco.mj_forward(model, data)
        assert probe.distance(data, name) < 0.0
        data.mocap_pos[mocap] = far
        mujoco.mj_forward(model, data)

    def test_the_probes_refuse_mesh_scenery(self, model):
        """The analytic path is only valid for primitives."""
        trunk = model.body("trunk_base").id
        with pytest.raises(RuntimeError):
            WallProbe(model, trunk, ("trunk_base",))

    def test_box_distance_is_exact_for_a_known_pair(self, model, data):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_plus_0")
        centre = data.geom_xpos[geom].copy()
        half_y = float(model.geom_size[geom][1])
        probe_point = centre + np.array([0.0, half_y + 0.25, 0.0])
        distance = box_sphere_distance(model, data, geom, probe_point, 0.05)
        assert distance == pytest.approx(0.20, abs=1e-6)


# ------------------------------------------------------------------ camera
class TestCamera:
    @pytest.fixture
    def camera(self, model, data):
        runner = PolicyRunner(POLICY).reset(model, mujoco.MjData(model))
        return EtiquetteCamera(model, data, runner.qpos_idx,
                               model.body("trunk_base").id, (PIP_W, PIP_H))

    def test_the_pip_frustum_is_the_one_visibility_is_measured_through(
            self, camera, model):
        assert camera.camera_id == model.camera("etiquette_camera").id
        assert camera.pip_w == PIP_W and camera.pip_h == PIP_H

    def test_gaze_never_touches_the_authoritative_walking_state(
            self, model, data, camera):
        """The single most important isolation claim in this behavior."""
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        ctrl = data.ctrl.copy()
        for step in range(120):
            camera.update(data, state="YIELD", duck_yaw=0.1 * math.sin(step),
                          duck_pos=data.xpos[model.body("trunk_base").id],
                          tracked=np.array([1.0, 0.0]), t=step * 0.02)
        assert np.array_equal(qpos, data.qpos)
        assert np.array_equal(qvel, data.qvel)
        assert np.array_equal(ctrl, data.ctrl)

    def test_the_head_really_moves_in_the_isolated_copy(
            self, model, data, camera):
        """Isolation must not be achieved by doing nothing."""
        from policy_runtime import HEAD_YAW_ACT

        index = camera.qpos_idx[HEAD_YAW_ACT]
        trunk = model.body("trunk_base").id
        for step in range(60):
            camera.update(data, state="YIELD", duck_yaw=0.0,
                          duck_pos=data.xpos[trunk],
                          tracked=np.array([1.0, 2.0]), t=step * 0.02)
        left = float(camera.render_data.qpos[index])
        for step in range(60):
            camera.update(data, state="YIELD", duck_yaw=0.0,
                          duck_pos=data.xpos[trunk],
                          tracked=np.array([1.0, -2.0]), t=step * 0.02)
        right = float(camera.render_data.qpos[index])
        assert abs(left - right) > 0.5

    def test_a_person_straight_ahead_is_visible(self, model, data, camera):
        trunk = model.body("trunk_base").id
        name = PERSON_NAMES[0]
        mocap = int(model.body_mocapid[model.body(f"person_{name}").id])
        original = data.mocap_pos[mocap].copy()
        duck = data.xpos[trunk].copy()
        data.mocap_pos[mocap] = [float(duck[0]) + 1.0, float(duck[1]), 0.36]
        mujoco.mj_forward(model, data)
        for step in range(80):
            state = camera.update(
                data, state="YIELD", duck_yaw=0.0, duck_pos=duck,
                tracked=data.mocap_pos[mocap][:2], t=step * 0.02)
        assert state["people"][name]["visible"], (
            "an unobstructed adult one metre ahead must be visible, or the "
            "gate is measuring self-occlusion rather than sight")
        data.mocap_pos[mocap] = original
        mujoco.mj_forward(model, data)

    def test_a_person_behind_the_duck_is_not_visible(self, model, data, camera):
        trunk = model.body("trunk_base").id
        name = PERSON_NAMES[0]
        mocap = int(model.body_mocapid[model.body(f"person_{name}").id])
        original = data.mocap_pos[mocap].copy()
        duck = data.xpos[trunk].copy()
        data.mocap_pos[mocap] = [float(duck[0]) + 1.0, float(duck[1]), 0.36]
        mujoco.mj_forward(model, data)
        # aim the head firmly the other way
        for step in range(120):
            state = camera.update(
                data, state="CRUISE", duck_yaw=0.0, duck_pos=duck,
                tracked=np.array([float(duck[0]) - 3.0, float(duck[1])]),
                t=step * 0.02)
        camera.view_yaw = math.pi
        camera._pose_head(data, 0.0)
        camera._orient_rig()
        assert not camera.person_visibility()[name]["visible"]
        data.mocap_pos[mocap] = original
        mujoco.mj_forward(model, data)

    def test_visibility_samples_the_whole_body(self):
        assert len(PERSON_SAMPLE_Z) >= 3
        assert min(PERSON_SAMPLE_Z) < 0.2 < max(PERSON_SAMPLE_Z)


# ------------------------------------------------------------ justifications
class TestConstantsAreJustified:
    def test_the_safe_passing_gap_exceeds_measured_tracking_error(self):
        """MEASURED closed-loop cruise excursion: 0.0634 m over 12 s."""
        assert SAFE_PASSING_GAP_M >= 0.0634

    def test_the_clear_range_covers_the_measured_rejoin(self):
        """The adult must not be close while the duck crosses the passage."""
        deepest = max(abs(a.park_y) for a in ALCOVES if a.clears_passage)
        rejoin_time = deepest / VY_SPEED_MPS
        adult_travel = 0.42 * rejoin_time
        assert adult_travel > CLEAR_RANGE_M, (
            "an adult receding at walking pace must open far more range than "
            "the release threshold during the duck's own rejoin")

    def test_the_clear_abs_y_is_the_sum_it_claims_to_be(self):
        assert CLEAR_ABS_Y == pytest.approx(
            CENTER_PASSAGE_HALF + DUCK_PLANAR_RADIUS)

    def test_the_destination_is_beyond_every_alcove(self):
        """So reaching it always requires resuming, never just parking."""
        assert DESTINATION_X > max(a.x_span[1] for a in ALCOVES)
