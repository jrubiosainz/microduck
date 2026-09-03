#!/usr/bin/env python3
"""Real-MuJoCo tests: scene invariants, sensor identity, camera gate, metrics.

These load the actual compiled scene, so they are slower than the pure logic
tests and they catch a different class of defect: a scene that compiles without
meshes, a sensor name that silently resolves to another quantity, an
acquisition gate that would open on somebody the camera cannot see, and a
metrics gate that passes a rollout it should reject.

MUTATION DISCIPLINE
-------------------
The metrics gates are exercised against SYNTHETIC counterexamples built from a
passing summary: each ``test_mutation_*`` corrupts exactly one property and
requires the corresponding gate to flip to False.  A gate that cannot fail is
not a gate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

mujoco = pytest.importorskip("mujoco")

from attention_camera import PIP_H, PIP_W, AttentionCamera  # noqa: E402
from people_routes import ADULT_NAMES, crowd_at, pose_crowd  # noqa: E402
from policy_runtime import (  # noqa: E402
    ACTION_SCALE,
    COMMAND_DIM,
    CTRL_HZ,
    DEFAULT_POSE,
    GYRO_SENSOR,
    OBS_DIM,
    build_observation,
    gyro_address,
    load_scene,
)
from recall_metrics import summarize  # noqa: E402
from recall_model import ACQUIRE_CONE_DEG, STANDOFF_MAX, STANDOFF_MIN  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return load_scene()


@pytest.fixture(scope="module")
def data(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    for slot, address in enumerate(
        [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    ):
        d.qpos[address] = DEFAULT_POSE[slot]
    mujoco.mj_forward(model, d)
    return d


# --------------------------------------------------------------------------
# scene structure
# --------------------------------------------------------------------------
def test_scene_compiles_with_meshes(model):
    """Zero meshes means meshdir did not resolve and the robot is invisible."""
    assert model.nmesh > 0


def test_scene_uses_the_stock_walking_robot(model):
    assert model.nu == 14, "the walking policy drives exactly 14 actuators"


def test_scene_contains_at_least_four_adults(model):
    present = [
        name for name in ADULT_NAMES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"person_{name}") >= 0
    ]
    assert len(present) >= 4
    assert len(present) == len(ADULT_NAMES)


def test_every_adult_is_kinematic_and_non_colliding(model):
    """An adult that can push the duck would invalidate every physics claim."""
    for name in ADULT_NAMES:
        body = model.body(f"person_{name}")
        assert int(model.body_mocapid[body.id]) >= 0, f"{name} is not mocap"
        for geom in range(model.ngeom):
            parent = int(model.geom_bodyid[geom])
            root = parent
            while root > 0 and root != body.id:
                root = int(model.body_parentid[root])
            if root == body.id:
                assert int(model.geom_contype[geom]) == 0
                assert int(model.geom_conaffinity[geom]) == 0


def test_adults_add_no_degrees_of_freedom_to_the_robot(model):
    """Mocap bodies must not change what the walking policy sees.

    The robot is a free joint (7 qpos) plus 14 actuated joints.  The adults'
    hinges exist for animation only and are never actuated.
    """
    actuated = {int(model.actuator_trnid[i, 0]) for i in range(model.nu)}
    for name in ADULT_NAMES:
        for side in ("l", "r"):
            for joint in (f"{name}_hip_{side}", f"{name}_shoulder_{side}"):
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                assert joint_id >= 0, f"{joint} missing"
                assert joint_id not in actuated, f"{joint} is actuated"


def test_caller_arms_can_reach_overhead(model):
    """The wave is only unmistakable if the shoulder hinge passes vertical."""
    from people_routes import WAVE_AMPLITUDE_DEG, WAVE_CENTER_DEG

    for name in ADULT_NAMES:
        joint = model.joint(f"{name}_shoulder_r")
        low, high = np.degrees(model.jnt_range[joint.id])
        assert low <= WAVE_CENTER_DEG - WAVE_AMPLITUDE_DEG
        assert high >= WAVE_CENTER_DEG + WAVE_AMPLITUDE_DEG
        assert low <= -180.0, "the arm cannot reach past vertical"


def test_required_cameras_and_markers_exist(model):
    for camera in ("head_camera", "attention_camera"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) >= 0
    for body in ("attention_rig", "call_ring", "goal_marker", "call_beacon"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0


# --------------------------------------------------------------------------
# sensor identity and observation width  (the PR #22 correction)
# --------------------------------------------------------------------------
def test_gyro_resolves_to_imu_ang_vel(model):
    address = gyro_address(model)
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR)
    assert sensor_id >= 0
    assert address == int(model.sensor_adr[sensor_id])


def test_gyro_address_is_not_the_silent_last_sensor_fallback(model):
    """``mj_name2id`` returns -1 for an unknown name and ``sensor_adr[-1]`` is a
    VALID index, so a wrong name silently feeds another quantity into
    ``base_ang_vel``.  The observation still looks plausible and the robot still
    walks, which is exactly what makes it dangerous."""
    assert gyro_address(model) != int(model.sensor_adr[-1])


def test_any_other_sensor_name_is_refused(model):
    for name in ("imu_gyro", "gyro", "angular-velocity", "root_angmom"):
        with pytest.raises(ValueError):
            gyro_address(model, name)


def test_root_angmom_exists_and_is_a_different_quantity(model):
    """The corrupted path read this instead. It must be present and distinct,
    otherwise the regression this guards against cannot be reproduced."""
    wrong = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "root_angmom")
    assert wrong >= 0
    assert int(model.sensor_adr[wrong]) != gyro_address(model)


def test_observation_is_exactly_61_dimensional():
    observation = build_observation(
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(3, dtype=np.float32),
    )
    assert observation.shape == (OBS_DIM,)
    assert OBS_DIM == 3 + 3 + 14 + 14 + 14 + COMMAND_DIM


def test_observation_places_the_twist_in_the_first_command_slots():
    twist = np.array([0.31, -0.12, 0.44], dtype=np.float32)
    observation = build_observation(
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
        np.zeros(14, dtype=np.float32), twist,
    )
    command = observation[-COMMAND_DIM:]
    assert np.allclose(command[:3], twist)
    assert np.allclose(command[3:], 0.0), "unused command slots must be zero-padded"


def test_action_scale_and_rate_are_the_shipped_values():
    assert ACTION_SCALE == pytest.approx(0.9)
    assert CTRL_HZ == pytest.approx(50.0)


def test_policy_is_the_stock_walking_network():
    """Byte-identical to the upstream checkout, not a retrained variant."""
    import hashlib

    policy = REPO_ROOT / "onnx" / "alpha_walking.onnx"
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    assert digest == (
        "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c")


def test_decimation_is_derived_from_the_model_timestep(model):
    assert int(round((1.0 / CTRL_HZ) / model.opt.timestep)) == 10


# --------------------------------------------------------------------------
# camera and the acquisition gate
# --------------------------------------------------------------------------
def test_camera_sees_a_person_it_is_aimed_at(model, data):
    crowd = crowd_at(3.0)
    pose_crowd(model, data, crowd, 3.0)
    mujoco.mj_forward(model, data)
    qpos_idx = np.array(
        [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)])
    camera = AttentionCamera(
        model, data, qpos_idx, model.body("trunk_base").id, (PIP_W, PIP_H))
    target = crowd["red"].pos - data.xpos[model.body("trunk_base").id][:2]
    camera.view_yaw = math.atan2(float(target[1]), float(target[0]))
    camera.view_pitch = math.radians(6.0)
    camera._pose_head(data, 0.0)
    camera._orient_rig()
    visible, off_axis, _ = camera._visible("red")
    assert visible
    assert math.degrees(off_axis) < ACQUIRE_CONE_DEG


def test_camera_does_not_see_a_person_behind_it(model, data):
    """A caller outside the frustum must NOT satisfy the gate, however close."""
    crowd = crowd_at(3.0)
    pose_crowd(model, data, crowd, 3.0)
    mujoco.mj_forward(model, data)
    qpos_idx = np.array(
        [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)])
    camera = AttentionCamera(
        model, data, qpos_idx, model.body("trunk_base").id, (PIP_W, PIP_H))
    target = crowd["red"].pos - data.xpos[model.body("trunk_base").id][:2]
    # Aim 180 deg away from red.
    camera.view_yaw = math.atan2(float(target[1]), float(target[0])) + math.pi
    camera.view_pitch = math.radians(6.0)
    camera._pose_head(data, 0.0)
    camera._orient_rig()
    visible, _, _ = camera._visible("red")
    assert not visible


def test_acquisition_cone_is_inside_the_measured_frustum(model, data):
    qpos_idx = np.array(
        [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)])
    camera = AttentionCamera(
        model, data, qpos_idx, model.body("trunk_base").id, (PIP_W, PIP_H))
    assert ACQUIRE_CONE_DEG < camera.half_v_deg
    assert ACQUIRE_CONE_DEG < camera.half_h_deg


def test_gaze_never_touches_the_authoritative_state(model, data):
    """Head posing happens in an isolated MjData copy.

    If gaze leaked into the walking state it could prop the robot up, and every
    stability number in the README would be meaningless.
    """
    crowd = crowd_at(5.0)
    pose_crowd(model, data, crowd, 5.0)
    mujoco.mj_forward(model, data)
    qpos_idx = np.array(
        [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)])
    camera = AttentionCamera(
        model, data, qpos_idx, model.body("trunk_base").id, (PIP_W, PIP_H))
    before = data.qpos.copy()
    camera.update(data, state="SEARCH", state_elapsed=1.0, duck_yaw=0.0,
                  caller="red", locked=None)
    assert np.array_equal(data.qpos, before), "gaze modified the physical state"
    # And the isolated copy DID move, so the test is not vacuous.
    assert not np.array_equal(
        camera.render_data.qpos[qpos_idx[7]], before[qpos_idx[7]]
    ) or camera.gaze_yaw != 0.0


# --------------------------------------------------------------------------
# metrics gates: synthetic counterexamples
# --------------------------------------------------------------------------
class _FakeRollout:
    """Minimal stand-in carrying exactly what ``summarize`` reads."""

    def __init__(self, cycles, records, refused=None):
        self.seconds = 54.0
        self.dt = 1.0 / 50.0
        self.decimation = 10
        self.duck_radius = 0.09
        self.records = records
        self.transitions = []

        class _Machine:
            pass

        self.machine = _Machine()
        self.machine.cycles = cycles
        self.machine.refused_calls = refused if refused is not None else [
            {"caller": "blue", "call_start_s": 22.0, "refused_at_s": 22.0,
             "busy_with": "yellow", "state": "APPROACH"}
        ]


def _good_cycle(index, caller, bearing):
    return {
        "cycle": index, "caller": caller, "call_start_s": 1.5,
        "search_start_s": 1.5, "search_duration_s": 2.0, "lock_s": 4.0,
        "lock_range_m": 1.9, "lock_off_axis_deg": 1.2, "lock_gate_open": True,
        "lock_caller_visible": True, "lock_is_active_caller": True,
        "call_bearing_deg": bearing, "approach_start_s": 5.0,
        "approach_end_s": 11.0, "approach_duration_s": 6.0,
        "approach_timeout": False, "approach_start_range_m": 1.90,
        "approach_min_range_m": 0.61, "arrival_range_m": 0.61,
        "final_range_m": 0.60, "approach_path_m": 1.25, "approach_net_m": 1.03,
        "approach_steps": 300, "approach_visible_steps": 300,
        "arrived_steps": 100, "arrived_visible_steps": 100,
        "min_caller_clearance_m": 0.31, "final_facing_error_deg": 10.0,
        "caller_changed": False,
    }


def _records(n=2700, clearance=0.16, command_outside=0.0):
    out = []
    for index in range(n):
        state = "APPROACH" if index % 3 == 0 else "LISTEN"
        out.append({
            "state": state,
            "command": [0.46, 0.0, 0.0] if state == "APPROACH"
            else [command_outside, 0.0, 0.0],
            "trunk_z_m": 0.116, "nearest_clearance_m": clearance,
        })
    out[-1]["trunk_z_m"] = 0.1163
    return out


def _summary(cycles=None, records=None, refused=None):
    cycles = cycles if cycles is not None else [
        _good_cycle(1, "red", -1.8),
        _good_cycle(2, "yellow", -120.7),
        _good_cycle(3, "green", 114.1),
    ]
    return summarize(
        _FakeRollout(cycles, records if records is not None else _records(),
                     refused),
        expected_order=("red", "yellow", "green"),
        standoff_min=STANDOFF_MIN, standoff_max=STANDOFF_MAX,
    )


def test_synthetic_passing_summary_passes_every_gate():
    """The baseline the mutations are measured against must itself pass."""
    summary = _summary()
    failed = [name for name, ok in summary["gates"].items() if not ok]
    assert failed == [], f"baseline fixture fails: {failed}"


def test_mutation_two_recalls_fails_the_recall_count():
    summary = _summary(cycles=[
        _good_cycle(1, "red", -1.8), _good_cycle(2, "yellow", -120.7)])
    assert summary["gates"]["recalls"] is False


def test_mutation_repeated_caller_fails_distinct_and_order():
    summary = _summary(cycles=[
        _good_cycle(1, "red", -1.8),
        _good_cycle(2, "yellow", -120.7),
        _good_cycle(3, "yellow", 114.1),
    ])
    assert summary["gates"]["distinct_callers"] is False
    assert summary["gates"]["caller_order"] is False


def test_mutation_wrong_order_fails_the_order_gate():
    summary = _summary(cycles=[
        _good_cycle(1, "yellow", -1.8),
        _good_cycle(2, "red", -120.7),
        _good_cycle(3, "green", 114.1),
    ])
    assert summary["gates"]["caller_order"] is False


def test_mutation_clustered_bearings_fail_the_bearing_gate():
    summary = _summary(cycles=[
        _good_cycle(1, "red", 10.0),
        _good_cycle(2, "yellow", 20.0),
        _good_cycle(3, "green", 30.0),
    ])
    assert summary["gates"]["distinct_bearings"] is False


def test_mutation_lock_on_the_wrong_person_fails_the_lock_gate():
    cycles = [_good_cycle(1, "red", -1.8), _good_cycle(2, "yellow", -120.7),
              _good_cycle(3, "green", 114.1)]
    cycles[1]["lock_is_active_caller"] = False
    summary = _summary(cycles=cycles)
    assert summary["gates"]["no_wrong_locks"] is False


def test_mutation_lock_without_the_camera_gate_fails():
    """The central claim: a lock justified by world geometry alone is invalid."""
    cycles = [_good_cycle(1, "red", -1.8), _good_cycle(2, "yellow", -120.7),
              _good_cycle(3, "green", 114.1)]
    cycles[2]["lock_gate_open"] = False
    summary = _summary(cycles=cycles)
    assert summary["gates"]["locks_were_seen"] is False

    cycles[2]["lock_gate_open"] = True
    cycles[2]["lock_caller_visible"] = False
    summary = _summary(cycles=cycles)
    assert summary["gates"]["locks_were_seen"] is False


def test_mutation_poor_approach_visibility_fails():
    cycles = [_good_cycle(1, "red", -1.8), _good_cycle(2, "yellow", -120.7),
              _good_cycle(3, "green", 114.1)]
    cycles[0]["approach_visible_steps"] = 280   # 93.3% < 95%
    summary = _summary(cycles=cycles)
    assert summary["gates"]["approach_visibility"] is False


def test_mutation_caller_not_visible_at_arrival_fails():
    cycles = [_good_cycle(1, "red", -1.8), _good_cycle(2, "yellow", -120.7),
              _good_cycle(3, "green", 114.1)]
    cycles[1]["arrived_visible_steps"] = 99     # 99% < 100%
    summary = _summary(cycles=cycles)
    assert summary["gates"]["arrived_visibility"] is False


def test_mutation_decorative_approach_that_never_moved_fails():
    """A nonzero command alone does not prove the policy crossed gait onset."""
    cycles = [_good_cycle(1, "red", -1.8), _good_cycle(2, "yellow", -120.7),
              _good_cycle(3, "green", 114.1)]
    cycles[0]["approach_path_m"] = 0.05
    cycles[0]["approach_net_m"] = 0.02
    summary = _summary(cycles=cycles)
    assert summary["gates"]["approach_moved"] is False


def test_mutation_approach_that_did_not_close_the_range_fails():
    cycles = [_good_cycle(1, "red", -1.8), _good_cycle(2, "yellow", -120.7),
              _good_cycle(3, "green", 114.1)]
    cycles[2]["approach_start_range_m"] = 0.75
    cycles[2]["approach_min_range_m"] = 0.61
    summary = _summary(cycles=cycles)
    assert summary["gates"]["approach_closed"] is False


def test_mutation_overshooting_the_standoff_band_fails():
    cycles = [_good_cycle(1, "red", -1.8), _good_cycle(2, "yellow", -120.7),
              _good_cycle(3, "green", 114.1)]
    cycles[0]["final_range_m"] = 0.28          # too close
    summary = _summary(cycles=cycles)
    assert summary["gates"]["standoff_band"] is False

    cycles[0]["final_range_m"] = 1.10          # stopped far too early
    summary = _summary(cycles=cycles)
    assert summary["gates"]["standoff_band"] is False


def test_mutation_facing_away_from_the_caller_fails():
    """Arriving at the right distance while pointed elsewhere is not a recall."""
    from recall_metrics import MAX_FACING_ERROR_DEG

    cycles = [_good_cycle(1, "red", -1.8), _good_cycle(2, "yellow", -120.7),
              _good_cycle(3, "green", 114.1)]
    cycles[2]["final_facing_error_deg"] = MAX_FACING_ERROR_DEG + 15.0
    summary = _summary(cycles=cycles)
    assert summary["gates"]["faces_caller"] is False
    # And the sign is not what matters: turned the other way fails too.
    cycles[2]["final_facing_error_deg"] = -(MAX_FACING_ERROR_DEG + 15.0)
    assert _summary(cycles=cycles)["gates"]["faces_caller"] is False


def test_mutation_any_command_while_stationary_fails():
    summary = _summary(records=_records(command_outside=1e-6))
    assert summary["gates"]["still_when_still"] is False


def test_mutation_caller_change_mid_cycle_fails():
    cycles = [_good_cycle(1, "red", -1.8), _good_cycle(2, "yellow", -120.7),
              _good_cycle(3, "green", 114.1)]
    cycles[1]["caller_changed"] = True
    summary = _summary(cycles=cycles)
    assert summary["gates"]["no_caller_change"] is False


def test_mutation_person_contact_fails():
    summary = _summary(records=_records(clearance=-0.001))
    assert summary["gates"]["no_person_contact"] is False


def test_mutation_a_fall_fails_the_stability_gates():
    records = _records()
    records[900]["trunk_z_m"] = 0.085
    summary = _summary(records=records)
    assert summary["gates"]["no_falls"] is False
    assert summary["gates"]["min_trunk_z"] is False


def test_mutation_not_recovering_nominal_height_fails():
    records = _records()
    records[-1]["trunk_z_m"] = 0.140
    summary = _summary(records=records)
    assert summary["gates"]["final_trunk_z"] is False


def test_mutation_obeying_the_interrupting_call_fails_the_refusal_gate():
    summary = _summary(refused=[])
    assert summary["gates"]["refused_interrupt"] is False


def test_standoff_band_matches_the_reported_band():
    summary = _summary()
    assert summary["standoff_band_m"] == [STANDOFF_MIN, STANDOFF_MAX]
