#!/usr/bin/env python3
"""Attention camera: search sweep, caller lock, and the acquisition gate.

The camera pose lives in an **isolated rendering ``MjData``** that is a copy of
the authoritative walking state.  Head yaw/pitch and the stabilized rig are
posed only in that copy; they are never written back into the locomotion data,
because the head is a large fraction of the robot's mass and the stock walking
policy was never trained to compensate an imposed head trajectory.

Visibility and the acquisition gate are measured from the EXACT camera the PiP
renders from (``attention_camera``), so the reported percentages and the
picture agree.  The test is real camera geometry — frustum containment plus an
occlusion ray cast through the actual scene — but the *identity* of each adult
comes from the simulator.  That is a semantic proxy for person recognition,
not RGB classification.

Why the gate matters here
-------------------------
This behavior's central claim is that the duck only goes to someone it has
actually FOUND.  A call can arrive from directly behind, and the duck cannot
turn in place (measured: ``wz=+/-0.85`` at ``vx=0`` moves the trunk under
10 deg in six seconds), so a caller behind the robot can only be acquired by
sweeping the HEAD.  The head yaw joint spans +/-170 deg, which makes a genuine
rear search possible without taking a single step — and that is exactly why
``SEARCH`` can hold the locomotion command at exactly zero.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from people_routes import ADULT_NAMES
from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT, wrap_angle

# Sample points on an adult, relative to their mocap origin (z is up): knees,
# torso, upper torso and cap.  A single torso-centre ray is not enough - another
# adult can cover that one point while the head and legs stay plainly visible.
SAMPLE_OFFSETS: tuple[float, ...] = (-0.06, 0.06, 0.20, 0.30)
# PiP pixel geometry lives HERE, not in the overlay, because it sets the
# camera's horizontal FOV and therefore every visibility measurement.  The
# no-render gate and the final render must measure through the same frustum, so
# the overlay imports these rather than defining its own.
PIP_W, PIP_H = 300, 220
# Head sweep.  The scan is INCREMENTAL (an offset advanced each tick and
# reversed at the limits) rather than a closed-form function of elapsed time,
# so the head can break off to fixate a glimpsed caller and later resume
# sweeping from wherever it is, instead of snapping back to a phase position.
#
# MEASURED CONSTRAINT ON THE RATE: acquisition needs the caller to stay inside
# the 12 deg cone for ACQUIRE_CONFIRM_S = 0.24 s.  A sweep that crosses the
# cone faster than that can NEVER acquire anybody - the first version used a
# 5 s triangle over +/-150 deg, i.e. 120 deg/s, which spends 24/120 = 0.20 s
# inside the cone and locked nobody in a 46 s rollout despite the caller being
# plainly visible.  The fix is not merely a slower sweep, though: see
# ``_fixating`` below.
SEARCH_AMPLITUDE = math.radians(150.0)
SEARCH_RATE = math.radians(90.0)      # per second, while sweeping
SEARCH_PITCH = math.radians(6.0)
# While listening the duck keeps a slow, narrow idle scan: it is awake, not
# staring at a wall.
IDLE_AMPLITUDE = math.radians(55.0)
IDLE_RATE = math.radians(28.0)
# Tracking rates, per control tick, once a caller is glimpsed or locked.
LOCK_YAW_RATE = math.radians(9.0)
LOCK_PITCH_RATE = math.radians(4.0)


class AttentionCamera:
    """Sweep for the caller, then lock and keep them centred."""

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
        self.view_pitch = SEARCH_PITCH

        half_v = math.radians(float(model.cam_fovy[self.attention_cam])) * 0.5
        self.tan_v = math.tan(half_v)
        self.tan_h = (self.pip_w / self.pip_h) * self.tan_v
        self.half_v_deg = math.degrees(half_v)
        self.half_h_deg = math.degrees(math.atan(self.tan_h))

        # Incremental sweep state, and whether the head has broken off the
        # sweep to fixate a caller it has actually glimpsed.
        self._sweep_offset = 0.0
        self._sweep_dir = 1.0
        self._fixating = False
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
        mujoco.mju_mat2Quat(quaternion, np.column_stack((right, up, -forward)).ravel())
        self.render_data.mocap_pos[self.rig_mocap] = eye
        self.render_data.mocap_quat[self.rig_mocap] = quaternion
        mujoco.mj_forward(self.model, self.render_data)

    # -- aiming ---------------------------------------------------------
    def _aim_at(self, target: np.ndarray) -> None:
        eye = self.render_data.cam_xpos[self.head_cam]
        delta = np.asarray(target, dtype=np.float64) - eye
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
        desired_pitch = math.atan2(float(delta[2]), float(np.linalg.norm(delta[:2])))
        self.view_yaw = wrap_angle(
            self.view_yaw
            + float(np.clip(
                wrap_angle(desired_yaw - self.view_yaw),
                -LOCK_YAW_RATE, LOCK_YAW_RATE,
            ))
        )
        self.view_pitch += float(
            np.clip(desired_pitch - self.view_pitch, -LOCK_PITCH_RATE, LOCK_PITCH_RATE)
        )

    def _sweep(self, duck_yaw: float, *, idle: bool) -> None:
        """Swing the head across the plaza while the body stays still.

        Advances an offset relative to the duck's own heading and reverses at
        the limits.  Because the offset is a stored quantity rather than a
        function of elapsed time, breaking off to fixate and later resuming
        does not teleport the head.
        """
        amplitude = IDLE_AMPLITUDE if idle else SEARCH_AMPLITUDE
        rate = (IDLE_RATE if idle else SEARCH_RATE) * self.dt
        self._sweep_offset += self._sweep_dir * rate
        if self._sweep_offset >= amplitude:
            self._sweep_offset = amplitude
            self._sweep_dir = -1.0
        elif self._sweep_offset <= -amplitude:
            self._sweep_offset = -amplitude
            self._sweep_dir = 1.0
        self.view_yaw = wrap_angle(duck_yaw + self._sweep_offset)
        self.view_pitch = SEARCH_PITCH

    def _resume_sweep_from_view(self, duck_yaw: float) -> None:
        """Re-seat the sweep offset at the current view, so resuming is smooth."""
        self._sweep_offset = float(
            np.clip(wrap_angle(self.view_yaw - duck_yaw),
                    -SEARCH_AMPLITUDE, SEARCH_AMPLITUDE)
        )

    # -- visibility -----------------------------------------------------
    def _adult_center(self, name: str) -> np.ndarray:
        return self.render_data.mocap_pos[self.people[name]].copy()

    def _visible(self, name: str) -> tuple[bool, float, float]:
        """(visible, smallest off-axis angle, range) from the attention camera.

        ``off_axis`` is reported for the sample points that are actually INSIDE
        the frustum and unoccluded.  Reporting the off-axis angle of a sample
        the camera cannot see would let the acquisition gate open on a caller
        hidden behind somebody else.
        """
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
            if not in_fov:
                continue
            if self._occluded(eye, unit, distance, name):
                continue
            visible = True
            best_off_axis = min(
                best_off_axis,
                math.acos(float(np.clip(unit @ forward, -1.0, 1.0))),
            )
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
               caller: str | None, locked: str | None) -> dict:
        """Pose the camera for this control tick and measure what it sees.

        ``caller`` is who is calling right now (may be unfound); ``locked`` is
        who the machine has committed to.

        SEARCH IS A TWO-STAGE PIPELINE, which is both more realistic and the
        only way acquisition is achievable at all:

        1. **Sweep** until the caller falls anywhere inside the camera frustum
           (74 deg x 58 deg).  That is a genuine glimpse, measured through the
           real frustum with occlusion, not a world-frame proximity test.
        2. **Fixate** the glimpsed caller, bringing them toward the optical
           axis so the 12 deg acquisition cone can be satisfied for a
           continuous 0.24 s.

        Without stage 2 the sweep rate and the confirmation window fight each
        other directly: any sweep fast enough to scan 300 deg in a few seconds
        crosses a 12 deg cone in under 0.24 s, so no lock is ever possible.
        Slowing the sweep enough to fix that alone would make every search take
        the better part of ten seconds.  Glimpse-then-fixate is what a real
        active-vision system does, and it makes the cone a statement about
        *how well centred* the caller is at lock time rather than a lottery on
        sweep phase.
        """
        self._pose_head(data, duck_yaw)
        if locked is not None:
            target = self._adult_center(locked)
            target[2] += 0.10
            self._aim_at(target)
            self._fixating = True
        elif self._fixating and caller is not None:
            target = self._adult_center(caller)
            target[2] += 0.10
            self._aim_at(target)
        else:
            self._sweep(duck_yaw, idle=(state == "LISTEN"))
        self._pose_head(data, duck_yaw)
        self._orient_rig()

        visible_names: list[str] = []
        off_axis_by_name: dict[str, float] = {}
        for name in ADULT_NAMES:
            seen, off_axis, _ = self._visible(name)
            off_axis_by_name[name] = off_axis
            if seen:
                visible_names.append(name)

        subject = locked or caller
        subject_visible = False
        subject_off_axis = math.pi
        subject_range = float("nan")
        if subject is not None:
            subject_visible, subject_off_axis, subject_range = self._visible(subject)

        # Decide fixation for the NEXT tick from what this camera pose ACTUALLY
        # saw.  Causality matters: fixating on a caller the camera has not yet
        # seen would make the search decorative.
        if locked is None:
            if state in ("SEARCH", "LISTEN") and subject_visible:
                self._fixating = True
            elif not subject_visible:
                if self._fixating:
                    self._resume_sweep_from_view(duck_yaw)
                self._fixating = False

        # THE ACQUISITION GATE.  Open only when the CALLER is geometrically
        # visible through this exact camera AND within the acquisition cone of
        # its optical axis.  Nothing else may open it: not world-frame
        # geometry, not proximity, not the fact that the caller is calling.
        from recall_model import ACQUIRE_CONE_DEG

        gate_open = bool(
            subject is not None
            and subject_visible
            and math.degrees(subject_off_axis) <= ACQUIRE_CONE_DEG
        )
        return {
            "visible": visible_names,
            "off_axis_deg": {
                name: math.degrees(value) for name, value in off_axis_by_name.items()
            },
            "subject": subject,
            "subject_visible": subject_visible,
            "subject_off_axis_deg": math.degrees(subject_off_axis),
            "subject_range_m": subject_range,
            "gate_open": gate_open,
            "fixating": bool(self._fixating),
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
        }

    @property
    def camera_id(self) -> int:
        return self.attention_cam
