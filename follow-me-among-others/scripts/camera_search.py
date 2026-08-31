#!/usr/bin/env python3
"""Color-selective search sweep and tracking from Microduck's head camera."""
import math

import mujoco
import numpy as np

from crowd_motion import COLORS, wrap


class CrowdCameraSearch:
    """Search for one shirt color, then keep that pedestrian centered.

    Locomotion physics stays authoritative in ``data``. Head pose and the
    stabilized optical view live in an isolated ``MjData`` used only for
    perception and rendering, matching the validated follow-me camera design.
    """

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
        self.head_cam = model.camera("head_camera").id
        self.follow_cam = model.camera("follow_camera").id
        rig = model.body("follow_camera_rig")
        self.follow_mocap = int(model.body_mocapid[rig.id])
        self.people = {}
        self.person_bodies = {}
        for color in COLORS:
            body = model.body(f"person_{color.lower()}")
            self.people[color] = int(model.body_mocapid[body.id])
            self.person_bodies[color] = self._descendants(body.id)
        self.self_bodies = self._descendants(trunk_id)
        self.self_bodies.add(trunk_id)

        # MuJoCo cameras look along -Z; align the upstream head camera optical
        # frame before copying its physical position to the stabilized rig.
        model.cam_quat[self.head_cam] = np.array(
            [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)], dtype=np.float64)
        mujoco.mj_forward(model, data)
        self.gaze_pitch = float(data.qpos[qpos_idx[self.head_pitch_act]])
        self.gaze_yaw = float(data.qpos[qpos_idx[self.head_yaw_act]])
        self.view_yaw = 0.0
        self.view_pitch = math.radians(8.0)
        vertical_half = math.radians(float(model.cam_fovy[self.follow_cam])) * 0.5
        self.tan_v = math.tan(vertical_half)
        self.tan_h = (self.pip_w / self.pip_h) * self.tan_v
        self.samples = 0
        self.target_visible_steps = 0
        self.search_steps = 0
        self.search_target_visible_steps = 0
        self.max_target_off_axis = 0.0

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

    def _pose_gaze(self, data, duck_yaw):
        mujoco.mj_copyData(self.gaze_data, self.model, data)
        yaw_lo, yaw_hi = self.model.jnt_range[self.head_yaw_joint]
        pitch_lo, pitch_hi = self.model.jnt_range[self.head_pitch_joint]
        relative = wrap(self.view_yaw - duck_yaw)
        self.gaze_yaw = float(np.clip(relative, yaw_lo + 0.03, yaw_hi - 0.03))
        self.gaze_pitch = float(np.clip(
            -self.view_pitch, pitch_lo + 0.04, pitch_hi - 0.04))
        self.gaze_data.qpos[self.qpos_idx[self.head_yaw_act]] = self.gaze_yaw
        self.gaze_data.qpos[self.qpos_idx[self.head_pitch_act]] = self.gaze_pitch
        self.gaze_data.qpos[self.qpos_idx[self.head_roll_act]] = 0.0
        mujoco.mj_forward(self.model, self.gaze_data)

    def _orient_rig(self):
        eye = self.gaze_data.cam_xpos[self.head_cam].copy()
        cp = math.cos(self.view_pitch)
        forward = np.array([
            cp * math.cos(self.view_yaw),
            cp * math.sin(self.view_yaw),
            math.sin(self.view_pitch),
        ], dtype=np.float64)
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= max(float(np.linalg.norm(right)), 1e-9)
        up = np.cross(right, forward)
        up /= max(float(np.linalg.norm(up)), 1e-9)
        rotation = np.column_stack((right, up, -forward))
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        self.gaze_data.mocap_pos[self.follow_mocap] = eye
        self.gaze_data.mocap_quat[self.follow_mocap] = quaternion
        mujoco.mj_forward(self.model, self.gaze_data)

    def _person_target(self, color):
        target = self.gaze_data.mocap_pos[self.people[color]].copy()
        target[2] += 0.03
        return target

    def _aim_toward(self, target, max_yaw_step=math.radians(5.0)):
        eye = self.gaze_data.cam_xpos[self.head_cam]
        delta = target - eye
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
        desired_pitch = math.atan2(float(delta[2]), float(np.linalg.norm(delta[:2])))
        self.view_yaw = wrap(self.view_yaw + float(np.clip(
            wrap(desired_yaw - self.view_yaw), -max_yaw_step, max_yaw_step)))
        self.view_pitch += float(np.clip(
            desired_pitch - self.view_pitch, -math.radians(3), math.radians(3)))

    def _visibility(self, color):
        center = self._person_target(color)
        eye = self.gaze_data.cam_xpos[self.follow_cam].copy()
        rotation = self.gaze_data.cam_xmat[self.follow_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        visible = False
        best_off_axis = math.pi
        center_distance = float(np.linalg.norm(center - eye))
        # A crowded scene cannot be judged by one torso-centre ray: another
        # pedestrian may cover that pixel while the colored shirt/head remains
        # plainly visible. Test torso, upper torso and head; any clear sample is
        # sufficient for color acquisition.
        for z_offset in (-0.06, 0.08, 0.20):
            target = center + np.array([0.0, 0.0, z_offset])
            delta = target - eye
            distance = float(np.linalg.norm(delta))
            unit = delta / max(distance, 1e-9)
            depth = float(np.dot(delta, forward))
            image_x = float(np.dot(delta, right))
            image_y = float(np.dot(delta, up))
            in_fov = (depth > 0.0 and abs(image_x) <= depth * self.tan_h
                      and abs(image_y) <= depth * self.tan_v)
            off_axis = math.acos(float(np.clip(
                np.dot(unit, forward), -1.0, 1.0)))
            best_off_axis = min(best_off_axis, off_axis)
            if in_fov:
                # The semantic color detector uses rendered segmentation, where
                # any visible shirt fragment is sufficient. Central raycasts
                # are too strict in a crowd (one arm can cover one sample).
                visible = True
        return visible, best_off_axis, center_distance

    def _occluded(self, eye, direction, target_distance, target_color):
        travelled = 0.02
        geom_id = np.zeros(1, dtype=np.int32)
        for _ in range(12):
            origin = eye + direction * travelled
            hit = mujoco.mj_ray(
                self.model, self.gaze_data, origin, direction,
                None, 1, -1, geom_id)
            if geom_id[0] < 0 or hit < 0.0:
                return False
            body = int(self.model.geom_bodyid[int(geom_id[0])])
            if body in self.person_bodies[target_color]:
                return False
            if body in self.self_bodies:
                travelled += hit + 0.005
                if travelled >= target_distance:
                    return False
                continue
            return travelled + hit < target_distance - 0.02
        return False

    def update(self, data, *, target_color, mode, mode_elapsed, duck_yaw):
        self._pose_gaze(data, duck_yaw)
        target = self._person_target(target_color)
        if mode == "BUSCO":
            # One complete panoramic sweep in 4.5 s. Recognition remains
            # color-selective: distractors can cross the PiP without firing.
            self.view_yaw = wrap(duck_yaw - math.pi + 2.0 * math.pi * mode_elapsed / 4.5)
            self.view_pitch = math.radians(9.0)
            self.search_steps += 1
        else:
            self._aim_toward(target)
        self._pose_gaze(data, duck_yaw)
        self._orient_rig()

        visible_colors = []
        target_visible = False
        target_off_axis = math.pi
        target_distance = float("nan")
        for color in COLORS:
            visible, off_axis, distance = self._visibility(color)
            if visible:
                visible_colors.append(color)
            if color == target_color:
                target_visible = visible
                target_off_axis = off_axis
                target_distance = distance
        self.samples += 1
        if target_visible:
            self.target_visible_steps += 1
            if mode == "BUSCO":
                self.search_target_visible_steps += 1
        self.max_target_off_axis = max(
            self.max_target_off_axis,
            target_off_axis if math.isfinite(target_off_axis) else 0.0)
        return {
            "target_color": target_color,
            "mode": mode,
            "target_visible": target_visible,
            "target_off_axis": target_off_axis,
            "target_distance": target_distance,
            "visible_colors": visible_colors,
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
        }

    @property
    def camera_id(self):
        return self.follow_cam
