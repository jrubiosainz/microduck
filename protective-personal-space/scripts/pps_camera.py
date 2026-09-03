#!/usr/bin/env python3
"""Rate-limited, render-only head camera for Protective Personal Space."""
from __future__ import annotations
import math
import mujoco
import numpy as np
from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT
from pps_cast import ALL_NAMES
from pps_states import TRACK_PITCH_DEG, TRACK_PITCH_RATE_DPS, TRACK_YAW_RATE_DPS
from pps_visibility import CameraVisibility

PIP_W, PIP_H = 300, 216


def wrap(angle):
    return (angle + math.pi) % (2*math.pi) - math.pi


class PpsCamera(CameraVisibility):
    def __init__(self, model, data, qpos_idx, trunk_id,
                 pip_size=(PIP_W, PIP_H), ctrl_hz=50.0):
        self.model, self.render_data = model, mujoco.MjData(model)
        self.qpos_idx, self.trunk_id = qpos_idx, trunk_id
        self.dt = 1.0/ctrl_hz
        self.head_pitch_joint = int(model.actuator_trnid[HEAD_PITCH_ACT, 0])
        self.head_yaw_joint = int(model.actuator_trnid[HEAD_YAW_ACT, 0])
        self.head_cam = model.camera("head_camera").id
        self.view_cam = model.camera("pps_camera").id
        self.rig_mocap = int(model.body_mocapid[model.body("pps_rig").id])
        self.bodies, self.body_geoms = {}, {}
        for name in ALL_NAMES:
            body = model.body(f"actor_{name}")
            self.bodies[name] = int(model.body_mocapid[body.id])
            self.body_geoms[name] = self._descendants(body.id)
        self.self_bodies = self._descendants(trunk_id)
        model.cam_quat[self.head_cam] = np.array(
            [math.sqrt(.5), 0, 0, -math.sqrt(.5)], dtype=np.float64)
        mujoco.mj_forward(model, data)
        self.view_yaw, self.view_pitch = 0.0, math.radians(TRACK_PITCH_DEG)
        self.gaze_yaw = self.gaze_pitch = 0.0
        self._initialized = False
        half_v = math.radians(float(model.cam_fovy[self.view_cam]))*.5
        self.tan_v = math.tan(half_v)
        self.tan_h = (pip_size[0]/pip_size[1])*self.tan_v

    def _descendants(self, root):
        found = {root}
        for body in range(self.model.nbody):
            parent = body
            while parent > 0:
                if parent == root:
                    found.add(body); break
                parent = int(self.model.body_parentid[parent])
        return found

    def _copy_and_pose_head(self, data, duck_yaw):
        mujoco.mj_copyData(self.render_data, self.model, data)
        yaw_lo, yaw_hi = self.model.jnt_range[self.head_yaw_joint]
        pitch_lo, pitch_hi = self.model.jnt_range[self.head_pitch_joint]
        self.gaze_yaw = float(np.clip(wrap(self.view_yaw-duck_yaw), yaw_lo+.03, yaw_hi-.03))
        self.gaze_pitch = float(np.clip(-self.view_pitch, pitch_lo+.04, pitch_hi-.04))
        self.render_data.qpos[self.qpos_idx[HEAD_YAW_ACT]] = self.gaze_yaw
        self.render_data.qpos[self.qpos_idx[HEAD_PITCH_ACT]] = self.gaze_pitch
        self.render_data.qpos[self.qpos_idx[HEAD_ROLL_ACT]] = 0.0
        mujoco.mj_forward(self.model, self.render_data)

    def _aim(self, target):
        eye = self.render_data.cam_xpos[self.head_cam]
        delta = np.asarray(target)-eye
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
        desired_pitch = math.atan2(float(delta[2]), float(np.linalg.norm(delta[:2])))
        if not self._initialized:
            # Initial acquisition is not a claimed search phase. Start on the
            # protected person; all subsequent motion remains rate-limited.
            self.view_yaw, self.view_pitch = desired_yaw, desired_pitch
            self._initialized = True
            return
        yr, pr = math.radians(TRACK_YAW_RATE_DPS)*self.dt, math.radians(TRACK_PITCH_RATE_DPS)*self.dt
        self.view_yaw = wrap(self.view_yaw + float(np.clip(wrap(desired_yaw-self.view_yaw), -yr, yr)))
        self.view_pitch += float(np.clip(desired_pitch-self.view_pitch, -pr, pr))

    def _orient_rig(self):
        eye = self.render_data.cam_xpos[self.head_cam].copy()
        cp = math.cos(self.view_pitch)
        forward = np.array([cp*math.cos(self.view_yaw), cp*math.sin(self.view_yaw), math.sin(self.view_pitch)])
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0,0,1]); right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, forward)
        quat = np.empty(4); mujoco.mju_mat2Quat(quat, np.column_stack((right,up,-forward)).ravel())
        self.render_data.mocap_pos[self.rig_mocap] = eye
        self.render_data.mocap_quat[self.rig_mocap] = quat
        mujoco.mj_forward(self.model, self.render_data)

    def update(self, data, duck_yaw: float, subject: str, secondary: str | None = None):
        self._copy_and_pose_head(data, duck_yaw)
        target = self.body_origin(subject)
        if secondary is not None:
            # Keep the protected person central while preserving the active
            # intruder in the deliberately wide, measured PiP frustum.
            target = 0.70 * target + 0.30 * self.body_origin(secondary)
        target[2] += .16
        self._aim(target); self._copy_and_pose_head(data, duck_yaw); self._orient_rig()
        people, visible = {}, []
        for name in ALL_NAMES:
            seen, off_axis, distance = self.visible_samples(name)
            entry = {"visible": any(seen), "samples": seen, "fraction": sum(seen)/len(seen),
                     "range_m": distance, "off_axis_deg": math.degrees(off_axis)}
            people[name] = entry
            if entry["visible"]: visible.append(name)
        return {"people": people, "visible_people": visible, "subject": subject,
                "view_yaw": self.view_yaw, "view_pitch": self.view_pitch,
                "gaze_yaw": self.gaze_yaw, "gaze_pitch": self.gaze_pitch}

    @property
    def camera_id(self): return self.view_cam
