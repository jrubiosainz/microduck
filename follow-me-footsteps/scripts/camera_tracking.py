#!/usr/bin/env python3
"""Independent head-camera gaze and visibility measurement for follow-me."""
import math

import mujoco
import numpy as np


class HeadCameraTracker:
    """Aim a rendering-only head pose at the person without altering locomotion."""

    def __init__(self, model, data, qpos_idx, trunk_id, pip_size=(225, 165)):
        self.model = model
        self.gaze_data = mujoco.MjData(model)
        self.qpos_idx = qpos_idx
        self.trunk_id = trunk_id
        self.pip_w, self.pip_h = pip_size
        self.head_pitch_act = 6
        self.head_yaw_act = 7
        self.head_roll_act = 8
        self.head_pitch_joint = int(model.actuator_trnid[self.head_pitch_act, 0])
        self.head_yaw_joint = int(model.actuator_trnid[self.head_yaw_act, 0])
        self.head_cam = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        self.follow_cam = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "follow_camera")
        follow_rig = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "follow_camera_rig")
        self.follow_mocap = int(model.body_mocapid[follow_rig])
        self.person_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "person")
        self.person_mocap = int(model.body_mocapid[self.person_id])

        # Correct the upstream optical frame: MuJoCo cameras look along -Z.
        model.cam_quat[self.head_cam] = np.array(
            [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)], dtype=np.float64)
        mujoco.mj_forward(model, data)

        self.gaze_pitch = float(data.qpos[qpos_idx[self.head_pitch_act]])
        self.gaze_yaw = float(data.qpos[qpos_idx[self.head_yaw_act]])
        vertical_half = math.radians(float(model.cam_fovy[self.follow_cam])) * 0.5
        self.tan_v = math.tan(vertical_half)
        self.tan_h = (self.pip_w / self.pip_h) * self.tan_v
        self.person_bodies = self._descendants(self.person_id)
        self.self_bodies = self._descendants(trunk_id)
        self.self_bodies.add(trunk_id)
        self.lost_steps = 0
        self.max_off_axis = 0.0
        self.off_axis_sq_sum = 0.0
        self.samples = 0
        self.visible = False
        self.off_axis = 0.0

    def _descendants(self, root):
        bodies = {root}
        for body in range(self.model.nbody):
            parent = body
            while parent > 0:
                if parent == root:
                    bodies.add(body)
                    break
                parent = int(self.model.body_parentid[parent])
        return bodies

    def _pose_gaze_data(self, data):
        mujoco.mj_copyData(self.gaze_data, self.model, data)
        self.gaze_data.qpos[self.qpos_idx[self.head_pitch_act]] = self.gaze_pitch
        self.gaze_data.qpos[self.qpos_idx[self.head_yaw_act]] = self.gaze_yaw
        # Keep the rendered optical horizon level. The walking policy's head
        # roll is a balance output, not a gaze request.
        self.gaze_data.qpos[self.qpos_idx[self.head_roll_act]] = 0.0
        mujoco.mj_forward(self.model, self.gaze_data)

    def _stabilize_camera(self, target):
        """Use the physical camera position with an electronically level view."""
        eye = self.gaze_data.cam_xpos[self.head_cam].copy()
        forward = target - eye
        forward /= max(float(np.linalg.norm(forward)), 1e-9)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        rotation = np.column_stack((right, up, -forward))
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        self.gaze_data.mocap_pos[self.follow_mocap] = eye
        self.gaze_data.mocap_quat[self.follow_mocap] = quaternion
        mujoco.mj_forward(self.model, self.gaze_data)

    def update(self, data):
        self._pose_gaze_data(data)
        cam_rot = self.gaze_data.cam_xmat[self.head_cam].reshape(3, 3)
        forward = -cam_rot[:, 2]
        left = -cam_rot[:, 0]
        up = cam_rot[:, 1]
        target = self.gaze_data.mocap_pos[self.person_mocap].copy()
        target[2] += 0.02
        to_person = target - self.gaze_data.cam_xpos[self.head_cam]
        unit = to_person / max(float(np.linalg.norm(to_person)), 1e-9)
        bearing = math.atan2(float(np.dot(unit, left)), float(np.dot(unit, forward)))
        elevation = math.atan2(float(np.dot(unit, up)), float(np.dot(unit, forward)))

        yaw_lo, yaw_hi = self.model.jnt_range[self.head_yaw_joint]
        pitch_lo, pitch_hi = self.model.jnt_range[self.head_pitch_joint]
        yaw_margin = math.radians(3.0)
        pitch_margin = math.radians(5.0)
        self.gaze_yaw = float(np.clip(
            self.gaze_yaw + bearing, yaw_lo + yaw_margin, yaw_hi - yaw_margin))
        # Positive physical pitch points the camera down.
        self.gaze_pitch = float(np.clip(
            self.gaze_pitch - elevation,
            pitch_lo + pitch_margin, pitch_hi - pitch_margin))
        self._pose_gaze_data(data)
        self._stabilize_camera(target)

        eye = self.gaze_data.cam_xpos[self.follow_cam].copy()
        cam_rot = self.gaze_data.cam_xmat[self.follow_cam].reshape(3, 3)
        right = cam_rot[:, 0]
        up = cam_rot[:, 1]
        forward = -cam_rot[:, 2]
        to_person = target - eye
        distance = float(np.linalg.norm(to_person))
        unit = to_person / max(distance, 1e-9)
        depth = float(np.dot(to_person, forward))
        image_x = float(np.dot(to_person, right))
        image_y = float(np.dot(to_person, up))
        in_fov = (depth > 0.0
                  and abs(image_x) <= depth * self.tan_h
                  and abs(image_y) <= depth * self.tan_v)
        self.off_axis = math.acos(float(np.clip(np.dot(unit, forward), -1.0, 1.0)))
        occluded = self._occluded(eye, unit, distance) if in_fov else False
        self.visible = in_fov and not occluded

        self.samples += 1
        self.max_off_axis = max(self.max_off_axis, self.off_axis)
        self.off_axis_sq_sum += self.off_axis * self.off_axis
        if not self.visible:
            self.lost_steps += 1
        return {
            "visible": self.visible,
            "off_axis": self.off_axis,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
        }

    def _occluded(self, eye, direction, target_distance):
        travelled = 0.02
        geom_id = np.zeros(1, dtype=np.int32)
        for _ in range(10):
            origin = eye + direction * travelled
            hit = mujoco.mj_ray(
                self.model, self.gaze_data, origin, direction,
                None, 1, -1, geom_id)
            if geom_id[0] < 0 or hit < 0.0:
                return False
            body = int(self.model.geom_bodyid[int(geom_id[0])])
            if body in self.person_bodies:
                return False
            if body in self.self_bodies:
                travelled += hit + 0.005
                if travelled >= target_distance:
                    return False
                continue
            return travelled + hit < target_distance - 0.02
        return False

    @property
    def camera_id(self):
        return self.follow_cam

    @property
    def rms_off_axis(self):
        if not self.samples:
            return 0.0
        return math.sqrt(self.off_axis_sq_sum / self.samples)
