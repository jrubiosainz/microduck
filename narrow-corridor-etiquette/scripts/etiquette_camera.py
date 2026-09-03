#!/usr/bin/env python3
"""Isolated gaze, the PiP camera, and the adult-tracking visibility gate.

The camera pose lives in an **isolated rendering ``MjData``** that is a copy of
the authoritative walking state.  Head yaw/pitch and the stabilized rig are
posed only in that copy and are never written back into the locomotion data,
because the head is a large fraction of the robot's mass and the stock walking
policy was never trained to compensate an imposed head trajectory.  Gaze
therefore cannot prop the robot up, and the robot's real head actuators are not
exercised here.

Why the tracking gate matters
-----------------------------
This behavior's central social claim is that the duck **watches the person go
past** rather than merely standing still while they do.  A head joint reaching
an angle proves nothing: the claim is only meaningful if the adult was
genuinely inside the duck's camera.  So during YIELD the adult's body is
sampled at head, chest and knee height and tested through the frustum of the
exact camera the PiP renders from, with occlusion ray casts against the actual
scene geometry — including the alcove's own cheek, which is a real obstruction
when the person is nearly abeam.

The duck cannot turn in place with this policy (MEASURED on this scene:
``wz=±0.85`` at ``vx=0`` moves the trunk under 10° in SIX seconds), and inside
an alcove it must not step back out to look.  The head yaw joint spans ±170°,
which is what makes tracking a person from a dead stop possible at all — and
that is exactly why YIELD can hold the locomotion command at exactly zero and
still be a real act of watching.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from corridor import ALCOVE_BY_NAME, CORRIDOR_X_MAX, DESTINATION_X
from people import PERSON_NAMES
from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT, wrap_angle

# PiP pixel geometry lives HERE, not in the overlay, because it sets the
# camera's horizontal FOV and therefore every visibility measurement.  The
# no-render gate and the final render must measure through the same frustum, so
# the overlay imports these rather than defining its own.
PIP_W, PIP_H = 300, 220

# Where the head points when nothing in particular demands attention: ahead and
# slightly down, along the corridor the duck is walking.
IDLE_YAW: float = 0.0
IDLE_PITCH: float = math.radians(4.0)
# While tracking a person the gaze is aimed at their chest, which sits well
# above the duck's own camera height, so the pitch is computed per tick from
# the geometry rather than fixed.
TRACK_TARGET_Z: float = 0.30
# Where the head looks once the duck has arrived: out into the lobby beyond the
# corridor's far doorway, rather than down at the threshold it is standing on.
# MEASURED FROM THE PREVIEW: aiming at the destination after reaching it framed
# a blank patch of wall for the whole four-second DONE tail.
LOBBY_LOOK_X: float = CORRIDOR_X_MAX + 2.2
# The duck's head camera sits about 0.23 m up.
HEAD_CAMERA_Z: float = 0.23
# Head slew rate, per control tick.  A head that snaps between targets looks
# like a teleport and gives the tracking gate no chance to accumulate confirmed
# visibility, so the slew is rate-limited and the gate measures what the head
# ACTUALLY reached.
SLEW_YAW_RATE = math.radians(4.2)      # per tick at 50 Hz ≈ 210 deg/s
SLEW_PITCH_RATE = math.radians(2.0)
# Fraction of an adult's sample points that must be inside the frustum for the
# person to count as tracked.  One of three is a genuine sighting of a body
# that is 0.7 m tall in a 58° camera at close range; demanding all three would
# fail whenever a shoulder leaves frame.
PERSON_VISIBLE_MIN_FRACTION: float = 1.0 / 3.0
# Where on the body the tracking gate samples: knee, chest and head.
PERSON_SAMPLE_Z: tuple[float, ...] = (0.10, 0.30, 0.62)


class EtiquetteCamera:
    """Isolated head gaze, the PiP camera, and the person-tracking gate."""

    def __init__(self, model, data, qpos_idx, trunk_id, pip_size=(PIP_W, PIP_H)):
        self.model = model
        self.render_data = mujoco.MjData(model)
        self.qpos_idx = qpos_idx
        self.trunk_id = trunk_id
        self.pip_w, self.pip_h = pip_size

        self.head_pitch_joint = int(model.actuator_trnid[HEAD_PITCH_ACT, 0])
        self.head_yaw_joint = int(model.actuator_trnid[HEAD_YAW_ACT, 0])
        self.head_cam = model.camera("head_camera").id
        self.pip_cam = model.camera("etiquette_camera").id
        rig = model.body("etiquette_rig")
        self.rig_mocap = int(model.body_mocapid[rig.id])

        self.person_bodies = {}
        self.person_subtrees = {}
        for name in PERSON_NAMES:
            body = model.body(f"person_{name}")
            self.person_bodies[name] = int(model.body_mocapid[body.id])
            self.person_subtrees[name] = self._descendants(body.id)
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

        half_v = math.radians(float(model.cam_fovy[self.pip_cam])) * 0.5
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
        """Rate-limited head slew, so target changes are motion, not teleports."""
        self.view_yaw = wrap_angle(self.view_yaw + float(np.clip(
            wrap_angle(target_yaw - self.view_yaw),
            -SLEW_YAW_RATE, SLEW_YAW_RATE)))
        self.view_pitch += float(np.clip(
            target_pitch - self.view_pitch, -SLEW_PITCH_RATE, SLEW_PITCH_RATE))

    def _aim_at(self, eye, point) -> tuple[float, float]:
        """World yaw and pitch that put ``point`` on the camera's axis."""
        delta = np.asarray(point, dtype=np.float64) - np.asarray(
            eye, dtype=np.float64)
        planar = math.hypot(float(delta[0]), float(delta[1]))
        return (math.atan2(float(delta[1]), float(delta[0])),
                math.atan2(float(delta[2]), max(planar, 1e-6)))

    def _target_for(self, state: str, duck_yaw: float, duck_pos,
                    tracked=None) -> tuple[float, float]:
        """World yaw and pitch the head should be reaching for in this state.

        Whenever there is a person to attend to — from the moment the encounter
        is detected until the yield is released — the head aims at THEM, which
        is what makes "looks at the passing adult" a measured claim rather than
        a caption.  Otherwise it looks where the duck is going.
        """
        eye = np.array([float(duck_pos[0]), float(duck_pos[1]),
                        HEAD_CAMERA_Z], dtype=np.float64)
        if tracked is not None and state in (
                "DETECT", "SELECT_ALCOVE", "PULL_OVER", "YIELD", "CLEAR"):
            point = np.array([float(tracked[0]), float(tracked[1]),
                              TRACK_TARGET_Z], dtype=np.float64)
            return self._aim_at(eye, point)
        if state in ("REJOIN", "RESUME", "CRUISE"):
            # Eyes back down the corridor, on the destination.
            point = np.array([DESTINATION_X, 0.0, 0.20], dtype=np.float64)
            yaw, pitch = self._aim_at(eye, point)
            return yaw, max(pitch, -IDLE_PITCH)
        if state == "DONE":
            # Arrived.  Aiming at the destination now would be aiming at the
            # duck's own feet, and the preview framed a blank patch of lobby
            # wall for the whole of the DONE tail.  Look out into the room
            # instead, which is where a robot that has just arrived would look.
            point = np.array([LOBBY_LOOK_X, 0.0, 0.34], dtype=np.float64)
            return self._aim_at(eye, point)
        return wrap_angle(duck_yaw + IDLE_YAW), IDLE_PITCH

    # -- visibility -----------------------------------------------------
    def _point_visible(self, point: np.ndarray,
                       owner_bodies: set[int] | None = None) -> bool:
        """Is this world point inside the PiP camera's frustum and unoccluded?

        ``owner_bodies`` is the set of bodies the point belongs to.  A ray cast
        at a point on a person's own centreline necessarily strikes that
        person's torso before reaching the centreline itself, so without this
        the gate reports every adult as occluded by themselves.  MEASURED: the
        torso capsule has a 0.078 m radius and the occlusion test tolerates
        0.02 m, so a perfectly centred, wholly unobstructed adult scored 0.00
        visibility.  Hitting the target's own geometry is what "seeing them"
        means, so those hits end the cast successfully.
        """
        eye = self.render_data.cam_xpos[self.pip_cam].copy()
        rotation = self.render_data.cam_xmat[self.pip_cam].reshape(3, 3)
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
        return not self._occluded(
            eye, delta / max(distance, 1e-9), distance, owner_bodies)

    def _occluded(self, eye, direction, distance,
                  owner_bodies: set[int] | None = None) -> bool:
        """Ray cast from the camera, stepping through the duck's own geometry.

        The robot's own body is in front of its head camera, so a naive cast
        reports every sample as occluded.  Self-hits are stepped past.  A hit on
        the TARGET's own geometry ends the cast successfully — that is what
        seeing them means.  Anything else in the way, and in this scene that
        means a wall or an alcove cheek, is a genuine occlusion and is reported
        as one, because a cheek between the duck and the person is exactly the
        situation that would make "the duck watched them pass" untrue.
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
            if owner_bodies is not None and body in owner_bodies:
                return False
            if body in self.self_bodies:
                travelled += hit + 0.005
                if travelled >= distance:
                    return False
                continue
            return travelled + hit < distance - 0.02
        return False

    def person_visibility(self) -> dict[str, dict]:
        """Per-person fraction of body sample points visible through the PiP."""
        result: dict[str, dict] = {}
        for name, mocap in self.person_bodies.items():
            centre = self.render_data.mocap_pos[mocap].copy()
            owner = self.person_subtrees[name]
            flags = [
                self._point_visible(
                    np.array([centre[0], centre[1], z], dtype=np.float64),
                    owner)
                for z in PERSON_SAMPLE_Z
            ]
            fraction = sum(flags) / len(flags)
            result[name] = {
                "fraction": fraction,
                "visible": fraction >= PERSON_VISIBLE_MIN_FRACTION,
                "points": flags,
            }
        return result

    # -- public ---------------------------------------------------------
    def update(self, data, *, state: str, duck_yaw: float, duck_pos,
               tracked=None, t: float = 0.0) -> dict:
        """Pose the camera for this control tick and measure what it sees.

        Order matters: the head is posed, the rig is placed at the resulting
        physical head-camera position, and only THEN is visibility measured —
        so every reported fraction comes from the same camera pose the PiP is
        rendered from on this frame.
        """
        target_yaw, target_pitch = self._target_for(
            state, duck_yaw, duck_pos, tracked)
        self._slew_to(target_yaw, target_pitch)
        self._pose_head(data, duck_yaw)
        self._orient_rig()

        people = self.person_visibility()
        return {
            "people": people,
            "visible_people": [n for n, v in people.items() if v["visible"]],
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
            "target_yaw": target_yaw,
            "target_pitch": target_pitch,
        }

    @property
    def camera_id(self) -> int:
        return self.pip_cam
