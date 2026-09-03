#!/usr/bin/env python3
"""Isolated scan gaze and the road-sector visibility gate.

The camera pose lives in an **isolated rendering ``MjData``** that is a copy of
the authoritative walking state.  Head yaw/pitch and the stabilized rig are
posed only in that copy and are never written back into the locomotion data,
because the head is a large fraction of the robot's mass and the stock walking
policy was never trained to compensate an imposed head trajectory.  Gaze
therefore cannot prop the robot up, and the robot's real head actuators are not
exercised here.

Why the sector gate matters
---------------------------
This behavior's central claim is that the duck **looks before it crosses**.
A head joint reaching an angle proves nothing: the claim is only meaningful if
the road the duck is checking was genuinely inside its camera.  So each LOOK
phase is graded against sample points placed on the LANE that phase is about
(``street.SECTORS``), tested through the frustum of the exact camera the PiP
renders from, with occlusion ray casts against the actual scene geometry.

The duck cannot turn in place with this policy (MEASURED on this scene:
``wz=±0.85`` at ``vx=0`` moves the trunk under 10 deg in SIX seconds), and it
must not step into the road to see up it.  The head yaw joint spans ±170 deg,
which is what makes a genuine left/right scan possible from a dead stop — and
that is exactly why the LOOK states can hold the locomotion command at exactly
zero and still be a real scan.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT, wrap_angle
from street import SECTOR_VISIBLE_MIN_FRACTION, SECTORS
from traffic import VEHICLE_NAMES

# PiP pixel geometry lives HERE, not in the overlay, because it sets the
# camera's horizontal FOV and therefore every visibility measurement.  The
# no-render gate and the final render must measure through the same frustum, so
# the overlay imports these rather than defining its own.
PIP_W, PIP_H = 300, 220

# Where the head points in each phase, as a world yaw offset from +x (the
# direction the duck faces).  Left is +y, right is −y.
#
# The magnitudes are chosen against the camera's own geometry rather than for
# looks: the guardian camera is 58 deg vertical on a 300x220 PiP, so its
# horizontal half-angle is 37.1 deg.  Aiming 74 deg off-axis therefore puts the
# road's vanishing direction near the far edge of frame while keeping the
# nearest sector sample — the one that matters most, 1.30 m up the lane —
# comfortably inside it.  Aiming a full 90 deg would centre the far samples and
# push the near one out of frame entirely.
SCAN_YAW: dict[str, float] = {
    "left": math.radians(74.0),
    "right": math.radians(-74.0),
}
# Ahead-and-slightly-down while approaching, waiting and safe.
IDLE_YAW: float = 0.0
SCAN_PITCH: float = math.radians(2.0)
IDLE_PITCH: float = math.radians(8.0)
# While CROSSING the head looks DOWN at the road it is walking over.
#
# MEASURED FROM THE PREVIEW: a level gaze along +x put the far pavement's
# buildings dead centre and left the crossing entirely below frame, so the PiP
# during the most important phase of the behavior showed a grey wall.
#
# NOTE THE SIGN.  ``view_pitch`` feeds the rig's forward vector as
# ``forward.z = sin(view_pitch)``, so POSITIVE pitch aims the camera UP.  The
# first attempt at this fix used +20 deg and framed even more sky.  Looking
# down at the road requires a NEGATIVE pitch; the head camera sits about 0.23 m
# up, so -20 deg places the road surface 0.6-1.5 m ahead in the middle of the
# frame — the zebra the duck is actually crossing, with the far kerb and the
# safe zone arriving from the top.
CROSSING_PITCH: float = math.radians(-20.0)
# While waiting, the duck keeps checking both ways rather than staring ahead.
# A pedestrian who has finished the formal scan does not stop watching, and a
# camera parked forward would make the WAIT phase visually dead.
WAIT_SWEEP_AMPLITUDE: float = math.radians(66.0)
WAIT_SWEEP_HZ: float = 0.16
# Head slew rate, per control tick.  A head that snaps between phases looks
# like a teleport and gives the sector gate no chance to accumulate confirmed
# visibility, so the slew is rate-limited and the gate measures what the head
# ACTUALLY reached.
SLEW_YAW_RATE = math.radians(3.6)      # per tick at 50 Hz ≈ 180 deg/s
SLEW_PITCH_RATE = math.radians(1.6)


class GuardianCamera:
    """Isolated head gaze, the PiP camera, and the road-sector visibility gate."""

    def __init__(self, model, data, qpos_idx, trunk_id, pip_size=(PIP_W, PIP_H)):
        self.model = model
        self.render_data = mujoco.MjData(model)
        self.qpos_idx = qpos_idx
        self.trunk_id = trunk_id
        self.pip_w, self.pip_h = pip_size

        self.head_pitch_joint = int(model.actuator_trnid[HEAD_PITCH_ACT, 0])
        self.head_yaw_joint = int(model.actuator_trnid[HEAD_YAW_ACT, 0])
        self.head_cam = model.camera("head_camera").id
        self.guardian_cam = model.camera("guardian_camera").id
        rig = model.body("guardian_rig")
        self.rig_mocap = int(model.body_mocapid[rig.id])

        self.vehicle_bodies = {}
        for name in VEHICLE_NAMES:
            body = model.body(f"vehicle_{name}")
            self.vehicle_bodies[name] = self._descendants(body.id)
        self.self_bodies = self._descendants(trunk_id) | {trunk_id}

        # MuJoCo cameras look down local −Z with +Y as image up.  The upstream
        # head_camera quaternion is [0 0 -1 0], which aims −Z backwards into the
        # robot's own CAD.  Correct it on the in-memory model only, so the
        # physical head-camera POSITION we copy to the rig is meaningful.
        model.cam_quat[self.head_cam] = np.array(
            [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)], dtype=np.float64)
        mujoco.mj_forward(model, data)

        self.gaze_yaw = float(data.qpos[qpos_idx[HEAD_YAW_ACT]])
        self.gaze_pitch = float(data.qpos[qpos_idx[HEAD_PITCH_ACT]])
        self.view_yaw = 0.0
        self.view_pitch = IDLE_PITCH

        half_v = math.radians(float(model.cam_fovy[self.guardian_cam])) * 0.5
        self.tan_v = math.tan(half_v)
        self.tan_h = (self.pip_w / self.pip_h) * self.tan_v
        self.half_v_deg = math.degrees(half_v)
        self.half_h_deg = math.degrees(math.atan(self.tan_h))

        self.dt = 1.0 / 50.0

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
        PiP while the duck's trunk pitches through its gait.  The PiP is
        labelled as stabilized for exactly this reason.
        """
        eye = self.render_data.cam_xpos[self.head_cam].copy()
        cos_p = math.cos(self.view_pitch)
        forward = np.array([
            cos_p * math.cos(self.view_yaw),
            cos_p * math.sin(self.view_yaw),
            math.sin(self.view_pitch),
        ])
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

    # -- aiming ---------------------------------------------------------
    def _slew_to(self, target_yaw: float, target_pitch: float) -> None:
        """Rate-limited head slew, so phase changes are motion, not teleports."""
        self.view_yaw = wrap_angle(self.view_yaw + float(np.clip(
            wrap_angle(target_yaw - self.view_yaw),
            -SLEW_YAW_RATE, SLEW_YAW_RATE)))
        self.view_pitch += float(np.clip(
            target_pitch - self.view_pitch, -SLEW_PITCH_RATE, SLEW_PITCH_RATE))

    def _target_for(self, state: str, duck_yaw: float, t: float) -> tuple[float, float]:
        """World yaw and pitch the head should be reaching for in this state."""
        if state == "LOOK_LEFT" or state == "LOOK_LEFT_AGAIN":
            return wrap_angle(duck_yaw + SCAN_YAW["left"]), SCAN_PITCH
        if state == "LOOK_RIGHT":
            return wrap_angle(duck_yaw + SCAN_YAW["right"]), SCAN_PITCH
        if state == "WAIT_FOR_GAP":
            # Keep watching both ways while waiting.
            offset = WAIT_SWEEP_AMPLITUDE * math.sin(
                2.0 * math.pi * WAIT_SWEEP_HZ * t)
            return wrap_angle(duck_yaw + offset), SCAN_PITCH
        if state == "CROSSING":
            # Eyes on where you are going.  A duck that keeps scanning while
            # crossing would be pretending the decision is still open; it is
            # not, and the machine will not act on it.  Pitched down onto the
            # crossing itself rather than level at the far buildings.
            return wrap_angle(duck_yaw + IDLE_YAW), CROSSING_PITCH
        return wrap_angle(duck_yaw + IDLE_YAW), IDLE_PITCH

    # -- visibility -----------------------------------------------------
    def _point_visible(self, point: np.ndarray) -> bool:
        """Is this world point inside the PiP camera's frustum and unoccluded?"""
        eye = self.render_data.cam_xpos[self.guardian_cam].copy()
        rotation = self.render_data.cam_xmat[self.guardian_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        delta = np.asarray(point, dtype=np.float64) - eye
        distance = float(np.linalg.norm(delta))
        depth = float(delta @ forward)
        if depth <= 0.0:
            return False
        if abs(float(delta @ right)) > depth * self.tan_h:
            return False
        if abs(float(delta @ up)) > depth * self.tan_v:
            return False
        return not self._occluded(eye, delta / max(distance, 1e-9), distance)

    def _occluded(self, eye, direction, distance) -> bool:
        """Ray cast from the camera, stepping through the duck's own geometry.

        The robot's own body is in front of its head camera, so a naive cast
        reports every sample as occluded.  Self-hits are stepped past; a
        vehicle in the way is a genuine occlusion and is reported as one,
        because a car blocking the view up the road is exactly the situation a
        pedestrian must not treat as "I looked".
        """
        travelled = 0.02
        geom_id = np.zeros(1, dtype=np.int32)
        for _ in range(16):
            origin = eye + direction * travelled
            hit = mujoco.mj_ray(
                self.model, self.render_data, origin, direction, None,
                1, -1, geom_id)
            if geom_id[0] < 0 or hit < 0.0:
                return False
            body = int(self.model.geom_bodyid[int(geom_id[0])])
            if body in self.self_bodies:
                travelled += hit + 0.005
                if travelled >= distance:
                    return False
                continue
            return travelled + hit < distance - 0.02
        return False

    def sector_visibility(self) -> dict[str, dict]:
        """Per-sector fraction of sample points visible through the PiP camera."""
        result: dict[str, dict] = {}
        for sector, points in SECTORS.items():
            flags = [self._point_visible(np.asarray(p, dtype=np.float64))
                     for p in points]
            fraction = sum(flags) / len(flags)
            result[sector] = {
                "fraction": fraction,
                "visible": fraction >= SECTOR_VISIBLE_MIN_FRACTION,
                "points": flags,
            }
        return result

    def visible_vehicles(self) -> list[str]:
        """Which road users are currently inside the PiP camera's frustum.

        Sampled at the vehicle's cabin height and at both ends of its body, so
        a long vehicle entering frame counts before its centre point does.
        """
        seen: list[str] = []
        for name in VEHICLE_NAMES:
            body = self.model.body(f"vehicle_{name}")
            mocap = int(self.model.body_mocapid[body.id])
            centre = self.render_data.mocap_pos[mocap].copy()
            for offset in (-0.18, 0.0, 0.18):
                point = centre + np.array([0.0, offset, 0.15])
                if self._point_visible(point):
                    seen.append(name)
                    break
        return seen

    # -- public ---------------------------------------------------------
    def update(self, data, *, state: str, duck_yaw: float, t: float) -> dict:
        """Pose the camera for this control tick and measure what it sees.

        Order matters: the head is posed, the rig is placed at the resulting
        physical head-camera position, and only THEN is visibility measured —
        so every reported fraction comes from the same camera pose the PiP is
        rendered from on this frame.
        """
        target_yaw, target_pitch = self._target_for(state, duck_yaw, t)
        self._slew_to(target_yaw, target_pitch)
        self._pose_head(data, duck_yaw)
        self._orient_rig()

        sectors = self.sector_visibility()
        return {
            "sectors": sectors,
            "left_visible": sectors["left"]["visible"],
            "right_visible": sectors["right"]["visible"],
            "left_fraction": sectors["left"]["fraction"],
            "right_fraction": sectors["right"]["fraction"],
            "visible_vehicles": self.visible_vehicles(),
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
            "target_yaw": target_yaw,
        }

    @property
    def camera_id(self) -> int:
        return self.guardian_cam
