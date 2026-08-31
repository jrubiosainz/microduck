#!/usr/bin/env python3
"""Attention camera: scanning sweep, threat lock, visibility measurement.

The camera pose lives in an **isolated rendering ``MjData``** that is a copy of
the authoritative walking state.  Head yaw/pitch and the stabilized rig are
posed only in that copy; they are never written back into the locomotion data,
because the head is a large fraction of the robot's mass and the stock walking
policy was never trained to compensate an imposed head trajectory.

Visibility is measured from the EXACT camera the PiP renders from
(``attention_camera``), so the reported percentage and the picture agree.  The
test is real camera geometry — frustum containment plus an occlusion ray cast
through the actual scene — but the *identity* of each adult comes from the
simulator.  That is a semantic proxy for pedestrian detection, not RGB
recognition.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from crowd_routes import ADULT_NAMES
from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT, wrap_angle

# Sample points on an adult, relative to their mocap origin (z is up).  A single
# torso-centre ray is not enough in a crowd: another adult can cover that one
# pixel while the head and legs stay plainly visible.
SAMPLE_OFFSETS: tuple[float, ...] = (-0.06, 0.06, 0.20, 0.26)
# PiP pixel geometry lives HERE, not in the overlay, because it sets the
# camera's horizontal FOV and therefore every visibility measurement.  The
# no-render gate and the final render must measure through the same frustum,
# so the overlay imports these rather than defining its own.
PIP_W, PIP_H = 300, 220
SCAN_PERIOD_S = 4.2
SCAN_AMPLITUDE = math.radians(105.0)
SCAN_PITCH = math.radians(7.0)
LOCK_YAW_RATE = math.radians(9.0)
LOCK_PITCH_RATE = math.radians(4.0)


class AttentionCamera:
    """Scan the crowd, then lock and track the selected threat."""

    def __init__(self, model, data, qpos_idx, trunk_id, pip_size=(PIP_W, PIP_H)):
        self.model = model
        self.render_data = mujoco.MjData(model)
        self.qpos_idx = qpos_idx
        self.trunk_id = trunk_id
        self.pip_w, self.pip_h = pip_size

        self.head_pitch_joint = int(model.actuator_trnid[HEAD_PITCH_ACT, 0])
        self.head_yaw_joint = int(model.actuator_trnid[HEAD_YAW_ACT, 0])
        self.head_cam = model.camera("head_camera").id
        self.attention_cam = model.camera("attention_camera").id
        rig = model.body("attention_rig")
        self.rig_mocap = int(model.body_mocapid[rig.id])

        self.people = {}
        self.person_bodies = {}
        for name in ADULT_NAMES:
            body = model.body(f"person_{name}")
            self.people[name] = int(model.body_mocapid[body.id])
            self.person_bodies[name] = self._descendants(body.id)
        self.self_bodies = self._descendants(trunk_id) | {trunk_id}

        # MuJoCo cameras look down local -Z with +Y as image up.  The upstream
        # head_camera quaternion is [0 0 -1 0], which aims -Z backwards into the
        # robot's own CAD.  Correct it on the in-memory model only, so the
        # physical head-camera POSITION we copy to the rig is meaningful.
        model.cam_quat[self.head_cam] = np.array(
            [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)], dtype=np.float64
        )
        mujoco.mj_forward(model, data)

        self.gaze_yaw = float(data.qpos[qpos_idx[HEAD_YAW_ACT]])
        self.gaze_pitch = float(data.qpos[qpos_idx[HEAD_PITCH_ACT]])
        self.view_yaw = 0.0
        self.view_pitch = SCAN_PITCH

        half_v = math.radians(float(model.cam_fovy[self.attention_cam])) * 0.5
        self.tan_v = math.tan(half_v)
        self.tan_h = (self.pip_w / self.pip_h) * self.tan_v

    # -- model helpers --------------------------------------------------
    def _descendants(self, root: int) -> set[int]:
        bodies = {root}
        for body in range(self.model.nbody):
            parent = body
            while parent > 0:
                if parent == root:
                    bodies.add(body)
                    break
                parent = int(self.model.body_parentid[parent])
        return bodies

    def _pose_head(self, data, duck_yaw: float) -> None:
        """Copy physics into the render data and pose the head there ONLY."""
        mujoco.mj_copyData(self.render_data, self.model, data)
        yaw_lo, yaw_hi = self.model.jnt_range[self.head_yaw_joint]
        pitch_lo, pitch_hi = self.model.jnt_range[self.head_pitch_joint]
        self.gaze_yaw = float(
            np.clip(wrap_angle(self.view_yaw - duck_yaw), yaw_lo + 0.03, yaw_hi - 0.03)
        )
        self.gaze_pitch = float(
            np.clip(-self.view_pitch, pitch_lo + 0.04, pitch_hi - 0.04)
        )
        self.render_data.qpos[self.qpos_idx[HEAD_YAW_ACT]] = self.gaze_yaw
        self.render_data.qpos[self.qpos_idx[HEAD_PITCH_ACT]] = self.gaze_pitch
        self.render_data.qpos[self.qpos_idx[HEAD_ROLL_ACT]] = 0.0
        mujoco.mj_forward(self.model, self.render_data)

    def _orient_rig(self) -> None:
        """Place the attention camera at the head camera, with a world-up roll.

        The rig is electronically stabilized: it sits exactly where the physical
        head camera sits, but its horizon is held level so a human can read the
        PiP while the duck's trunk pitches through its gait.
        """
        eye = self.render_data.cam_xpos[self.head_cam].copy()
        cos_p = math.cos(self.view_pitch)
        forward = np.array(
            [
                cos_p * math.cos(self.view_yaw),
                cos_p * math.sin(self.view_yaw),
                math.sin(self.view_pitch),
            ]
        )
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= max(float(np.linalg.norm(right)), 1e-9)
        up = np.cross(right, forward)
        up /= max(float(np.linalg.norm(up)), 1e-9)
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, np.column_stack((right, up, -forward)).ravel())
        self.render_data.mocap_pos[self.rig_mocap] = eye
        self.render_data.mocap_quat[self.rig_mocap] = quaternion
        mujoco.mj_forward(self.model, self.render_data)

    # -- aiming ---------------------------------------------------------
    def _aim_at(self, target: np.ndarray) -> None:
        eye = self.render_data.cam_xpos[self.head_cam]
        delta = np.asarray(target, dtype=np.float64) - eye
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
        desired_pitch = math.atan2(
            float(delta[2]), float(np.linalg.norm(delta[:2]))
        )
        self.view_yaw = wrap_angle(
            self.view_yaw
            + float(
                np.clip(
                    wrap_angle(desired_yaw - self.view_yaw),
                    -LOCK_YAW_RATE,
                    LOCK_YAW_RATE,
                )
            )
        )
        self.view_pitch += float(
            np.clip(desired_pitch - self.view_pitch, -LOCK_PITCH_RATE, LOCK_PITCH_RATE)
        )

    def _scan(self, duck_yaw: float, elapsed: float) -> None:
        """Sweep the head left/right across the plaza while standing still."""
        self.view_yaw = wrap_angle(
            duck_yaw + SCAN_AMPLITUDE * math.sin(2.0 * math.pi * elapsed / SCAN_PERIOD_S)
        )
        self.view_pitch = SCAN_PITCH

    # -- visibility -----------------------------------------------------
    def _adult_center(self, name: str) -> np.ndarray:
        return self.render_data.mocap_pos[self.people[name]].copy()

    def _visible(self, name: str) -> tuple[bool, float, float]:
        """(visible, smallest off-axis angle, range) from the attention camera."""
        center = self._adult_center(name)
        eye = self.render_data.cam_xpos[self.attention_cam].copy()
        rotation = self.render_data.cam_xmat[self.attention_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        distance_to_center = float(np.linalg.norm(center - eye))
        visible = False
        best_off_axis = math.pi
        for z_offset in SAMPLE_OFFSETS:
            target = center + np.array([0.0, 0.0, z_offset])
            delta = target - eye
            distance = float(np.linalg.norm(delta))
            unit = delta / max(distance, 1e-9)
            depth = float(delta @ forward)
            in_fov = (
                depth > 0.0
                and abs(float(delta @ right)) <= depth * self.tan_h
                and abs(float(delta @ up)) <= depth * self.tan_v
            )
            off_axis = math.acos(float(np.clip(unit @ forward, -1.0, 1.0)))
            if in_fov:
                best_off_axis = min(best_off_axis, off_axis)
                if not self._occluded(eye, unit, distance, name):
                    visible = True
        return visible, best_off_axis, distance_to_center

    def _occluded(self, eye, direction, distance, name) -> bool:
        travelled = 0.02
        geom_id = np.zeros(1, dtype=np.int32)
        for _ in range(12):
            origin = eye + direction * travelled
            hit = mujoco.mj_ray(
                self.model, self.render_data, origin, direction, None, 1, -1, geom_id
            )
            if geom_id[0] < 0 or hit < 0.0:
                return False
            body = int(self.model.geom_bodyid[int(geom_id[0])])
            if body in self.person_bodies[name]:
                return False
            if body in self.self_bodies:
                travelled += hit + 0.005
                if travelled >= distance:
                    return False
                continue
            return travelled + hit < distance - 0.02
        return False

    # -- public ---------------------------------------------------------
    def update(self, data, *, state: str, state_elapsed: float, duck_yaw: float,
               locked: str | None) -> dict:
        """Pose the camera for this control tick and measure what it sees."""
        self._pose_head(data, duck_yaw)
        if state == "SCANNING" or locked is None:
            self._scan(duck_yaw, state_elapsed)
        else:
            target = self._adult_center(locked)
            target[2] += 0.08
            self._aim_at(target)
        self._pose_head(data, duck_yaw)
        self._orient_rig()

        visible_names: list[str] = []
        for name in ADULT_NAMES:
            seen, _, _ = self._visible(name)
            if seen:
                visible_names.append(name)
        locked_visible = False
        locked_off_axis = math.pi
        locked_range = float("nan")
        if locked is not None:
            locked_visible, locked_off_axis, locked_range = self._visible(locked)
        return {
            "visible": visible_names,
            "locked": locked,
            "locked_visible": locked_visible,
            "locked_off_axis": locked_off_axis,
            "locked_range": locked_range,
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
        }

    @property
    def camera_id(self) -> int:
        return self.attention_cam
