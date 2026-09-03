#!/usr/bin/env python3
"""Authoritative locomotion layer: scene loading, 61-D observation, ONNX rollout.

This module owns everything the walking policy touches.  Nothing here knows
about threats, gaze or rendering — the physical locomotion state produced by
:class:`PolicyRunner` is the single source of truth for the whole behavior.

Two failure modes are asserted rather than trusted:

* **Sensor identity.**  ``mj_name2id`` returns ``-1`` for an unknown sensor and
  ``model.sensor_adr[-1]`` is a *valid* index, so a wrong name silently feeds a
  different physical quantity into the policy's ``base_ang_vel`` slot.  The
  observation still looks plausible and the robot still walks, which is exactly
  what makes it dangerous.  :func:`gyro_address` raises on any name other than
  ``imu_ang_vel`` and refuses the last-sensor address.
* **Observation width.**  The exported walking policy is 61-D:
  ``ang_vel(3) + gravity(3) + joint_pos(14) + joint_vel(14) + last_action(14)
  + command(13)``.  Only the first three command entries (the twist) are used;
  the remaining ten are zero-padded, which is the documented convention for a
  task that does not drive those slots.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

CTRL_HZ = 50.0
# Shipped walking action scale.  Actions are offsets from DEFAULT_POSE.
ACTION_SCALE = 0.9
# The ONLY acceptable angular-velocity sensor.  No fallback list, ever.
GYRO_SENSOR = "imu_ang_vel"
OBS_DIM = 61
COMMAND_DIM = 13
NOMINAL_TRUNK_Z = 0.116
FALLEN_TRUNK_Z = 0.09

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = REPO_ROOT / "assets" / "scene_door_lift.xml"

# The scene does `<include file="robot_walk.xml"/>` with a bare filename, and
# `robot_walk.xml` declares `<compiler meshdir="assets"/>`.  MuJoCo resolves both
# the include and meshdir against the TOP-LEVEL file's directory, so the scene
# must be loaded from inside the robot directory or the meshes silently fail to
# resolve.  Rather than hardcode a machine path, the default is derived from this
# repository's own location (.../projects/microduck-lab/door-elevator-etiquette
# -> .../projects/microduck_rl/src/mjlab_microduck/robot/microduck) and can be
# overridden with MICRODUCK_RL_ROBOT_DIR or --robot-dir.
# (.../projects/microduck-lab/door-elevator-etiquette → …/projects/microduck_rl/…)
DEFAULT_ROBOT_DIR = (
    REPO_ROOT.parents[1]
    / "microduck_rl"
    / "src"
    / "mjlab_microduck"
    / "robot"
    / "microduck"
)

# STAND2 pose (HOME_FRAME), identical to the STAND keyframe in the scenes.
DEFAULT_POSE = np.array(
    [
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,
        0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,
    ],
    dtype=np.float32,
)

HEAD_PITCH_ACT = 6
HEAD_YAW_ACT = 7
HEAD_ROLL_ACT = 8


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def gyro_address(model: mujoco.MjModel, name: str = GYRO_SENSOR) -> int:
    """Resolve the angular-velocity sensor address, refusing a silent fallback."""
    if name != GYRO_SENSOR:
        raise ValueError(
            f"refusing sensor {name!r}: this behavior is measured against "
            f"{GYRO_SENSOR!r} only, and a different quantity in the "
            "base_ang_vel slot invalidates every measured constant"
        )
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"sensor {name!r} not found in the model")
    address = int(model.sensor_adr[sensor_id])
    if address == int(model.sensor_adr[-1]) and sensor_id != model.nsensor - 1:
        raise ValueError(f"sensor {name!r} resolved to the last-sensor address")
    return address


def actuator_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """qpos and qvel indices of every actuated joint, in actuator order."""
    qpos = np.array(
        [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    )
    qvel = np.array(
        [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    )
    return qpos, qvel


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the inverse of quaternion ``[w, x, y, z]``."""
    w = quat[0]
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


def build_observation(
    gyro: np.ndarray,
    gravity: np.ndarray,
    joint_pos_rel: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray,
    twist: np.ndarray,
    command_dim: int = COMMAND_DIM,
) -> np.ndarray:
    """Assemble the 61-D actor observation, zero-padding the unused command slots."""
    command = np.zeros(command_dim, dtype=np.float32)
    command[:3] = twist
    observation = np.concatenate(
        [gyro, gravity, joint_pos_rel, joint_vel, last_action, command]
    ).astype(np.float32)
    if observation.shape != (OBS_DIM,):
        raise RuntimeError(f"expected {OBS_DIM}-D observation, got {observation.shape}")
    return observation


def robot_dir(explicit: str | Path | None = None) -> Path:
    """Directory holding ``robot_walk.xml`` and its ``assets/`` mesh folder."""
    import os

    candidate = explicit or os.environ.get("MICRODUCK_RL_ROBOT_DIR") or DEFAULT_ROBOT_DIR
    directory = Path(candidate).expanduser().resolve()
    if not (directory / "robot_walk.xml").is_file():
        raise SystemExit(
            f"robot_walk.xml not found in {directory}; pass --robot-dir or set "
            "MICRODUCK_RL_ROBOT_DIR to a microduck_rl robot directory"
        )
    return directory


def install_scene(explicit_robot_dir: str | Path | None = None) -> Path:
    """Copy the committed scene next to ``robot_walk.xml`` and return that path.

    Copying is what makes `<include file="robot_walk.xml">` and the robot's
    `meshdir="assets"` resolve.  The committed asset stays the single source of
    truth; this only materializes it where MuJoCo can compile it.
    """
    import shutil

    if not SCENE_XML.is_file():
        raise SystemExit(f"scene not found: {SCENE_XML}; run tools/build_scene.py")
    target = robot_dir(explicit_robot_dir) / SCENE_XML.name
    if not target.is_file() or target.read_bytes() != SCENE_XML.read_bytes():
        shutil.copyfile(SCENE_XML, target)
    return target


def load_scene(
    path: str | Path | None = None, explicit_robot_dir: str | Path | None = None
) -> mujoco.MjModel:
    """Load the lobby scene and assert the robot is the stock walking robot."""
    scene = Path(path) if path is not None else install_scene(explicit_robot_dir)
    if not scene.is_file():
        raise SystemExit(f"scene not found: {scene}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    if model.nmesh == 0:
        raise RuntimeError(
            f"{scene} compiled with zero meshes: meshdir did not resolve"
        )
    if model.nu != 14:
        raise RuntimeError(f"expected 14 policy actuators, got {model.nu}")
    gyro_address(model)  # raises unless imu_ang_vel exists and is distinct
    return model


class _Rollout:
    """One bound (model, data) rollout: observation assembly and ONNX stepping."""

    def __init__(self, session, model: mujoco.MjModel, data: mujoco.MjData):
        self.session = session
        self.model = model
        self.data = data
        self.qpos_idx, self.qvel_idx = actuator_indices(model)
        self.gyro_adr = gyro_address(model)
        self.trunk = model.body("trunk_base").id
        self.last_action = np.zeros(model.nu, dtype=np.float32)
        for slot, address in enumerate(self.qpos_idx):
            data.qpos[address] = DEFAULT_POSE[slot]
        data.ctrl[:] = DEFAULT_POSE
        mujoco.mj_forward(model, data)

    def yaw(self, data: mujoco.MjData | None = None) -> float:
        data = self.data if data is None else data
        forward = data.xmat[self.trunk].reshape(3, 3)[:, 0]
        return math.atan2(float(forward[1]), float(forward[0]))

    def observe(self, data: mujoco.MjData, twist: np.ndarray) -> np.ndarray:
        quat = data.xquat[self.trunk].astype(np.float32)
        gravity = quat_rotate_inverse(
            quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        gyro = data.sensordata[self.gyro_adr : self.gyro_adr + 3].astype(np.float32)
        joint_pos = data.qpos[self.qpos_idx].astype(np.float32) - DEFAULT_POSE
        joint_vel = data.qvel[self.qvel_idx].astype(np.float32)
        return build_observation(
            gyro, gravity, joint_pos, joint_vel, self.last_action, twist
        )

    def step(self, data: mujoco.MjData, twist: np.ndarray) -> np.ndarray:
        """Run one policy inference and write ``data.ctrl``. Physics is the caller's."""
        observation = self.observe(data, twist)
        action = self.session.run(
            [self.output_name], {self.input_name: observation.reshape(1, -1)}
        )[0].squeeze(0).astype(np.float32)
        self.last_action = action.copy()
        data.ctrl[:] = DEFAULT_POSE + ACTION_SCALE * action
        return action

    @property
    def input_name(self) -> str:
        return self.session.get_inputs()[0].name

    @property
    def output_name(self) -> str:
        return self.session.get_outputs()[0].name


class PolicyRunner:
    """Loads the stock walking ONNX once and binds it to rollouts."""

    def __init__(self, policy_path: str | Path):
        import onnxruntime as ort

        path = Path(policy_path)
        if not path.is_file():
            raise SystemExit(f"policy not found: {path}")
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        observation_dim = int(self.session.get_inputs()[0].shape[1])
        if observation_dim != OBS_DIM:
            raise RuntimeError(
                f"policy expects a {observation_dim}-D observation; this behavior "
                f"is measured against {OBS_DIM}-D only"
            )
        self.path = path

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> _Rollout:
        return _Rollout(self.session, model, data)
