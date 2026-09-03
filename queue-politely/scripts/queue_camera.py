#!/usr/bin/env python3
"""Attention camera: watching the person ahead, and the acquisition gate.

The camera pose lives in an **isolated rendering ``MjData``** that is a copy of
the authoritative walking state.  Head yaw/pitch and the stabilized rig are
posed only in that copy and never written back into the locomotion data,
because the head is a large fraction of the robot's mass and the stock walking
policy was never trained to compensate an imposed head trajectory.

Visibility is measured from the EXACT camera the PiP renders from
(``queue_camera``), so the reported percentages and the picture agree.  The test
is real camera geometry - frustum containment plus an occlusion ray cast through
the actual scene - but the IDENTITY of each person comes from the simulator.
That is a semantic proxy for person recognition, not RGB classification.

WHAT THIS BEHAVIOR ASKS OF THE CAMERA, AND WHY IT IS HARDER THAN IT LOOKS
-------------------------------------------------------------------------
The gate is "the person ahead was visible for at least 95 % of every advance".
In a queue that bends through 180 deg that is a genuine constraint rather than
a formality:

* while the duck is on the return leg its predecessor is on the OTHER side of
  the fold, roughly 130 deg off the duck's own heading, so the head has to be
  turned hard to keep them in frame;
* the people between them are opaque bodies standing directly on the sightline,
  so occlusion is real and is measured against actual scene geometry;
* the head yaw joint spans +/-170 deg, which is what makes this achievable
  without the body turning - and is exactly why WAIT can hold the locomotion
  command at exactly zero while still watching the queue.

During the stationary observation phases the head performs a genuine SWEEP of
the queue, incremental rather than a closed-form function of time, so breaking
off to fixate somebody and later resuming does not teleport the head.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT, wrap_angle
from queue_people import ALL_NAMES, CLERK
from queue_vision import SAMPLE_OFFSETS, VisibilityMixin  # noqa: F401

# PiP pixel geometry lives HERE, not in the overlay, because it sets the
# camera's horizontal FOV and therefore every visibility measurement.  The
# no-render gate and the final render must measure through the same frustum.
PIP_W, PIP_H = 300, 220
# Sweep while observing the queue.
SWEEP_AMPLITUDE = math.radians(120.0)
SWEEP_RATE = math.radians(52.0)
SWEEP_PITCH = math.radians(7.0)
# Tracking rates per control tick once a subject is being followed.
LOCK_YAW_RATE = math.radians(9.0)
LOCK_PITCH_RATE = math.radians(4.0)
# How well centred a subject must be to count as ACQUIRED during the
# identification phase.
ACQUIRE_CONE_DEG = 14.0


class QueueCamera(VisibilityMixin):
    """Sweep the queue, then watch whoever the duck is standing behind."""

    def __init__(self, model, data, qpos_idx, trunk_id,
                 pip_size=(PIP_W, PIP_H)):
        self.model = model
        self.render_data = mujoco.MjData(model)
        self.qpos_idx = qpos_idx
        self.trunk_id = trunk_id
        self.pip_w, self.pip_h = pip_size

        self.head_pitch_joint = int(model.actuator_trnid[HEAD_PITCH_ACT, 0])
        self.head_yaw_joint = int(model.actuator_trnid[HEAD_YAW_ACT, 0])
        self.head_cam = model.camera("head_camera").id
        self.queue_cam = model.camera("queue_camera").id
        rig = model.body("queue_rig")
        self.rig_mocap = int(model.body_mocapid[rig.id])

        self.people = {}
        self.person_bodies = {}
        for name in ALL_NAMES:
            body = model.body(f"person_{name}")
            self.people[name] = int(model.body_mocapid[body.id])
            self.person_bodies[name] = self._descendants(body.id)
        self.self_bodies = self._descendants(trunk_id) | {trunk_id}

        # MuJoCo cameras look down local -Z with +Y as image up.  The upstream
        # head_camera quaternion is [0 0 -1 0], which aims -Z backwards into the
        # robot's own CAD.  Correct it on the in-memory model only, so the
        # physical head-camera POSITION we copy to the rig is meaningful.
        model.cam_quat[self.head_cam] = np.array(
            [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)], dtype=np.float64)
        mujoco.mj_forward(model, data)

        self.gaze_yaw = float(data.qpos[qpos_idx[HEAD_YAW_ACT]])
        self.gaze_pitch = float(data.qpos[qpos_idx[HEAD_PITCH_ACT]])
        self.view_yaw = 0.0
        self.view_pitch = SWEEP_PITCH

        half_v = math.radians(float(model.cam_fovy[self.queue_cam])) * 0.5
        self.tan_v = math.tan(half_v)
        self.tan_h = (self.pip_w / self.pip_h) * self.tan_v
        self.half_v_deg = math.degrees(half_v)
        self.half_h_deg = math.degrees(math.atan(self.tan_h))

        self._sweep_offset = 0.0
        self._sweep_dir = -1.0
        self.dt = 1.0 / 50.0

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
            wrap_angle(self.view_yaw - duck_yaw),
            yaw_lo + 0.03, yaw_hi - 0.03))
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
        PiP while the duck's trunk pitches through its gait.
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
        self.view_yaw = wrap_angle(self.view_yaw + float(np.clip(
            wrap_angle(desired_yaw - self.view_yaw),
            -LOCK_YAW_RATE, LOCK_YAW_RATE)))
        self.view_pitch += float(np.clip(
            desired_pitch - self.view_pitch, -LOCK_PITCH_RATE, LOCK_PITCH_RATE))

    def _sweep(self, duck_yaw: float) -> None:
        """Swing the head across the queue while the body stays still.

        Advances a stored offset and reverses at the limits, rather than being a
        function of elapsed time, so breaking off to fixate and later resuming
        does not teleport the head.
        """
        self._sweep_offset += self._sweep_dir * SWEEP_RATE * self.dt
        if self._sweep_offset >= SWEEP_AMPLITUDE:
            self._sweep_offset = SWEEP_AMPLITUDE
            self._sweep_dir = -1.0
        elif self._sweep_offset <= -SWEEP_AMPLITUDE:
            self._sweep_offset = -SWEEP_AMPLITUDE
            self._sweep_dir = 1.0
        self.view_yaw = wrap_angle(duck_yaw + self._sweep_offset)
        self.view_pitch = SWEEP_PITCH

    def _resume_sweep_from_view(self, duck_yaw: float) -> None:
        self._sweep_offset = float(np.clip(
            wrap_angle(self.view_yaw - duck_yaw),
            -SWEEP_AMPLITUDE, SWEEP_AMPLITUDE))

    # -- public ----------------------------------------------------------
    def update(self, data, *, state: str, duck_yaw: float,
               subject: str | None) -> dict:
        """Pose the camera for this tick and measure what it actually sees.

        ``subject`` is the person the duck is standing behind, once it has one.
        While the duck is still reading the queue there is no subject and the
        head sweeps; the sweep is what makes OBSERVE_QUEUE and IDENTIFY_TAIL a
        genuine look rather than a pause.

        AT THE COUNTER THERE IS NO SUBJECT AND NO SWEEP.  The duck has arrived
        and is being served, so the head is aimed at the clerk: a sweep there
        pans across a blank counter face, and the first preview showed exactly
        that - the last ten seconds of PiP were an empty green sign board.
        """
        self._pose_head(data, duck_yaw)
        aim_at = subject
        if aim_at is None and state in ("AT_COUNTER", "DONE"):
            aim_at = CLERK.name
        if aim_at is not None:
            target = self._person_center(aim_at)
            target[2] += 0.10
            self._aim_at(target)
        else:
            self._sweep(duck_yaw)
        self._pose_head(data, duck_yaw)
        self._orient_rig()

        people: dict[str, dict] = {}
        visible_names: list[str] = []
        for name in ALL_NAMES:
            seen, fraction, off_axis, distance = self._visible(name)
            people[name] = {
                "visible": bool(seen), "fraction": float(fraction),
                "off_axis_deg": math.degrees(off_axis),
                "range_m": float(distance),
            }
            if seen:
                visible_names.append(name)

        subject_entry = people.get(subject) if subject else None
        acquired = bool(
            subject_entry is not None and subject_entry["visible"]
            and subject_entry["off_axis_deg"] <= ACQUIRE_CONE_DEG)
        if subject is None:
            pass
        elif subject_entry is not None and not subject_entry["visible"]:
            self._resume_sweep_from_view(duck_yaw)

        return {
            "people": people,
            "visible_people": visible_names,
            "subject": subject,
            "subject_visible": bool(
                subject_entry["visible"]) if subject_entry else False,
            "subject_fraction": float(
                subject_entry["fraction"]) if subject_entry else 0.0,
            "subject_off_axis_deg": float(
                subject_entry["off_axis_deg"]) if subject_entry else 180.0,
            "acquired": acquired,
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
        }

    @property
    def camera_id(self) -> int:
        return self.queue_cam
