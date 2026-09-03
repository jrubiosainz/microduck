#!/usr/bin/env python3
"""What the duck can actually see, where it is looking, and whether it can read
an arm.

The camera pose lives in an **isolated rendering ``MjData``** that is a copy of
the authoritative walking state.  Head yaw/pitch and the stabilized rig are posed
only in that copy and never written back into the locomotion data, because the
head is a large fraction of the robot's mass and the stock walking policy was
never trained to compensate an imposed head trajectory.  Gaze therefore cannot
prop the robot up, and the physical locomotion state remains the single source
of truth.

Visibility is measured from the EXACT camera the PiP is rendered from, so the
reported percentages and the picture agree.

THE ARM GATE IS WHAT MAKES THIS CAMERA DIFFERENT FROM ITS SIBLINGS
--------------------------------------------------------------------
A sibling behavior only had to answer "can the duck see this body".  Here that
is not enough: a gesture may be accepted only if the duck could read the ARM it
was made with.  :meth:`arm_readable` therefore tests the three keypoints of one
arm - shoulder, elbow and hand - individually, each with frustum containment and
its own occlusion ray cast, and requires all three.

That distinction is load-bearing rather than pedantic.  A person can be
comfortably in frame with their raised hand outside it: the duck is 0.20 m tall
and stands close enough to read a gesture, so an adult's raised arm sits near
the top of the frustum precisely when the torso is centred.  MEASURED on the
real run, the arm gate is strictly harder than the body gate - it fails on ticks
where the body gate passes - which is what stops "the duck saw the person" from
standing in for "the duck read the gesture".

THE HEAD SEARCH IS A SWEEP, BECAUSE THE BODY CANNOT TURN
----------------------------------------------------------
Turning in place is MEASURED to be unavailable on this policy - at most
1.6 deg/s at ``vx = 0`` - so a stopped robot cannot sweep its body across the
area.  :meth:`search_target` drives a triangle wave about the duck's own
heading at the MEASURED 26 deg/s head rate, covering +/-52 deg.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT, wrap_angle
from gest_cast import ALL_NAMES
from gest_visibility import ARM_JOINTS, CameraVisibility  # noqa: F401
from gest_states import (
    SEARCH_SWEEP_DEG,
    TRACK_PITCH_DEG,
    TRACK_PITCH_RATE_DPS,
    TRACK_YAW_RATE_DPS,
)

# PiP pixel geometry lives HERE, not in an overlay, because it sets the camera's
# horizontal FOV and therefore every visibility measurement.  The headless gate
# and any render must measure through the same frustum.
PIP_W, PIP_H = 300, 216

class GestureCamera(CameraVisibility):
    """Head camera: pose the head, then report what it could actually read.

    The POSING lives here; the measuring - frustum containment, occlusion ray
    casts and the arm-readability gate - lives in
    :class:`gest_visibility.CameraVisibility`, which this inherits.
    """

    def __init__(self, model, data, qpos_idx, trunk_id,
                 pip_size=(PIP_W, PIP_H), ctrl_hz: float = 50.0):
        self.model = model
        self.render_data = mujoco.MjData(model)
        self.qpos_idx = qpos_idx
        self.trunk_id = trunk_id
        self.pip_w, self.pip_h = pip_size
        self.dt = 1.0 / ctrl_hz

        self.head_pitch_joint = int(model.actuator_trnid[HEAD_PITCH_ACT, 0])
        self.head_yaw_joint = int(model.actuator_trnid[HEAD_YAW_ACT, 0])
        self.head_cam = model.camera("head_camera").id
        self.view_cam = model.camera("gesture_camera").id
        rig = model.body("gesture_rig")
        self.rig_mocap = int(model.body_mocapid[rig.id])

        self.bodies: dict[str, int] = {}
        self.body_geoms: dict[str, set[int]] = {}
        # The six arm keypoint BODY ids per person.  Read from the model by
        # name, so the positions the gate tests are the ones MuJoCo computed
        # from the real joint angles.
        self.arm_bodies: dict[str, dict[str, int]] = {}
        for name in ALL_NAMES:
            body = model.body(f"actor_{name}")
            self.bodies[name] = int(model.body_mocapid[body.id])
            self.body_geoms[name] = self._descendants(body.id)
            self.arm_bodies[name] = {}
            for side in ("l", "r"):
                self.arm_bodies[name][f"{side}_shoulder"] = int(
                    model.body(f"{name}_shoulder_{side}").id)
                self.arm_bodies[name][f"{side}_elbow"] = int(
                    model.body(f"{name}_fore_{side}").id)
                self.arm_bodies[name][f"{side}_hand"] = int(
                    model.body(f"{name}_hand_{side}").id)
        self.self_bodies = self._descendants(trunk_id) | {trunk_id}

        # MuJoCo cameras look down local -Z with +Y as image up.  The upstream
        # head_camera quaternion is [0 0 -1 0], which aims -Z backwards into the
        # robot's own CAD.  Correct it on the in-memory model only, so the
        # physical head-camera POSITION copied to the rig stays meaningful.
        model.cam_quat[self.head_cam] = np.array(
            [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)], dtype=np.float64)
        mujoco.mj_forward(model, data)

        self.gaze_yaw = float(data.qpos[qpos_idx[HEAD_YAW_ACT]])
        self.gaze_pitch = float(data.qpos[qpos_idx[HEAD_PITCH_ACT]])
        self.view_yaw = 0.0
        self.view_pitch = math.radians(TRACK_PITCH_DEG)

        half_v = math.radians(float(model.cam_fovy[self.view_cam])) * 0.5
        self.tan_v = math.tan(half_v)
        self.tan_h = (self.pip_w / self.pip_h) * self.tan_v
        self.half_v_deg = math.degrees(half_v)
        self.half_h_deg = math.degrees(math.atan(self.tan_h))

        self._search_t = 0.0

    # -- model helpers ---------------------------------------------------
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
        """Copy physics into the render data and pose the head THERE ONLY."""
        mujoco.mj_copyData(self.render_data, self.model, data)
        yaw_lo, yaw_hi = self.model.jnt_range[self.head_yaw_joint]
        pitch_lo, pitch_hi = self.model.jnt_range[self.head_pitch_joint]
        self.gaze_yaw = float(np.clip(
            wrap_angle(self.view_yaw - duck_yaw), yaw_lo + 0.03, yaw_hi - 0.03))
        self.gaze_pitch = float(np.clip(
            -self.view_pitch, pitch_lo + 0.04, pitch_hi - 0.04))
        self.render_data.qpos[self.qpos_idx[HEAD_YAW_ACT]] = self.gaze_yaw
        self.render_data.qpos[self.qpos_idx[HEAD_PITCH_ACT]] = self.gaze_pitch
        self.render_data.qpos[self.qpos_idx[HEAD_ROLL_ACT]] = 0.0
        mujoco.mj_forward(self.model, self.render_data)

    def _orient_rig(self) -> None:
        """Place the PiP camera at the head camera, with a world-up roll.

        The rig is electronically stabilized: it sits exactly where the physical
        head camera sits, but its horizon is held level so a human can read the
        PiP while the duck's trunk pitches through its gait.  Any render must
        label it as stabilized.
        """
        eye = self.render_data.cam_xpos[self.head_cam].copy()
        cos_p = math.cos(self.view_pitch)
        forward = np.array([cos_p * math.cos(self.view_yaw),
                            cos_p * math.sin(self.view_yaw),
                            math.sin(self.view_pitch)])
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= max(float(np.linalg.norm(right)), 1e-9)
        up = np.cross(right, forward)
        up /= max(float(np.linalg.norm(up)), 1e-9)
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            quaternion, np.column_stack((right, up, -forward)).ravel())
        self.render_data.mocap_pos[self.rig_mocap] = eye
        self.render_data.mocap_quat[self.rig_mocap] = quaternion
        mujoco.mj_forward(self.model, self.render_data)

    # -- aiming ----------------------------------------------------------
    def _aim_at(self, target: np.ndarray) -> None:
        eye = self.render_data.cam_xpos[self.head_cam]
        delta = np.asarray(target, dtype=np.float64) - eye
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
        desired_pitch = math.atan2(
            float(delta[2]), float(np.linalg.norm(delta[:2])))
        yaw_rate = math.radians(TRACK_YAW_RATE_DPS) * self.dt / 0.02
        pitch_rate = math.radians(TRACK_PITCH_RATE_DPS) * self.dt / 0.02
        self.view_yaw = wrap_angle(self.view_yaw + float(np.clip(
            wrap_angle(desired_yaw - self.view_yaw), -yaw_rate, yaw_rate)))
        self.view_pitch += float(np.clip(
            desired_pitch - self.view_pitch, -pitch_rate, pitch_rate))

    def begin_search(self) -> None:
        self._search_t = 0.0

    def search_target(self, duck_xy, duck_yaw: float) -> np.ndarray:
        """The world point the head sweeps to next while looking for somebody.

        A TRIANGLE WAVE about the duck's own heading, at the MEASURED head rate.
        Returning a POINT rather than a yaw keeps a single aiming path through
        :meth:`_aim_at`, so the sweep is rate-limited by the same measured
        constant that limits target tracking.
        """
        span = math.radians(SEARCH_SWEEP_DEG)
        rate = math.radians(TRACK_YAW_RATE_DPS)
        period = 4.0 * span / rate
        phase = (self._search_t % period) / period
        if phase < 0.25:
            offset = span * (phase / 0.25)
        elif phase < 0.75:
            offset = span * (1.0 - 2.0 * (phase - 0.25) / 0.5)
        else:
            offset = span * (-1.0 + (phase - 0.75) / 0.25)
        self._search_t += self.dt
        yaw = duck_yaw + offset
        duck = np.asarray(duck_xy, dtype=np.float64)[:2]
        # A point 2.4 m out at chest height: far enough that the aim is a
        # BEARING rather than a nearby point the head would converge on.
        return np.array([duck[0] + 2.4 * math.cos(yaw),
                         duck[1] + 2.4 * math.sin(yaw), 0.42])

    # -- public ----------------------------------------------------------
    def update(self, data, *, duck_yaw: float, subject: str, look_at,
               present: dict[str, bool]) -> dict:
        """Pose the camera for this tick and measure what it actually sees.

        Returns per-person body visibility AND per-arm readability, because the
        gesture gate needs the second and the acquisition gate needs the first.
        """
        self._pose_head(data, duck_yaw)
        self._aim_at(np.asarray(look_at, dtype=np.float64))
        self._pose_head(data, duck_yaw)
        self._orient_rig()

        bodies: dict[str, dict] = {}
        visible: list[str] = []
        keypoints: dict[str, dict] = {}
        for name in ALL_NAMES:
            if not present.get(name, True):
                bodies[name] = {"visible": False, "samples": [False] * 5,
                                "sample_count": 0, "fraction": 0.0,
                                "off_axis_deg": 180.0, "range_m": 99.0,
                                "present": False,
                                "arm_readable": {"l": False, "r": False}}
                keypoints[name] = {}
                continue
            seen, off_axis, range_m = self._visible_samples(
                self.sample_points(name), self.body_geoms[name])
            count = sum(seen)
            points = self.arm_keypoints(name)
            keypoints[name] = points
            bodies[name] = {
                "visible": count > 0,
                "samples": seen,
                "sample_count": count,
                "fraction": count / len(seen),
                "off_axis_deg": math.degrees(off_axis),
                "range_m": range_m,
                "present": True,
                # THE ARM GATE.  Strictly harder than the body gate above, and
                # measured separately so a metric can show that it is.
                "arm_readable": {
                    side: self.arm_readable(name, side, points)
                    for side in ("l", "r")},
            }
            if count > 0:
                visible.append(name)

        return {
            "bodies": bodies,
            "keypoints": keypoints,
            "visible_bodies": visible,
            "subject": subject,
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
            "aim_in_frustum": self.point_in_frustum(look_at),
        }

    @property
    def camera_id(self) -> int:
        return self.view_cam
