#!/usr/bin/env python3
"""What the duck can actually see of the traffic it is negotiating with, and of
the goal it is walking to.

The camera pose lives in an **isolated rendering ``MjData``** that is a copy of
the authoritative walking state.  Head yaw/pitch and the stabilized rig are
posed only in that copy and never written back into the locomotion data, because
the head is a large fraction of the robot's mass and the stock walking policy was
never trained to compensate an imposed head trajectory.  Gaze therefore cannot
prop the robot up, and the physical locomotion state remains the single source
of truth.

Visibility is measured from the EXACT camera the PiP is rendered from, so the
reported percentages and the picture agree.  The test is real camera geometry —
frustum containment plus an occlusion ray cast through actual scene geometry —
but the IDENTITY of each body comes from the simulator.  That is a **semantic
proxy** for object recognition, not RGB classification, and it is labelled as
such wherever it surfaces.

WHY VISIBILITY MATTERS FOR THIS BEHAVIOR SPECIFICALLY
------------------------------------------------------
Every decision this robot makes is a decision about somebody else's predicted
position: to pass a pedestrian on the right, to thread a cone on the left, to
stop because two bodies are converging.  A robot that made those decisions
without being able to SEE the bodies concerned would be guessing, however
correct the outcome looked.  So the gate requires the NEGOTIATED body to be
visible in almost every tick where line of sight geometrically EXISTS.

Conditioning on line of sight separates two very different failures: a camera
pointing the wrong way, which is the duck's fault, and a body genuinely behind
another body, which is not.

THE GOAL IS MEASURED THROUGH THE SAME CAMERA
----------------------------------------------
:meth:`SlalomCamera.goal_visibility` samples the beacon post at five heights
through the identical frustum and the identical ray cast.  "It could see where
it was going" is therefore the same kind of measurement as "it could see the
cart", taken with the same instrument, rather than a claim about the plan view.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT, wrap_angle
from slalom_cast import ALL_NAMES, BY_NAME
from slalom_course import goal_sample_points
from slalom_states import (
    TRACK_PITCH_DEG,
    TRACK_PITCH_RATE_DPS,
    TRACK_YAW_RATE_DPS,
)

# PiP pixel geometry lives HERE, not in an overlay, because it sets the camera's
# horizontal FOV and therefore every visibility measurement.  The headless gate
# and any render must measure through the same frustum.
PIP_W, PIP_H = 300, 216


class SlalomCamera:
    """Head camera: track the negotiated body or the goal, and report what it saw."""

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
        self.view_cam = model.camera("slalom_camera").id
        rig = model.body("slalom_rig")
        self.rig_mocap = int(model.body_mocapid[rig.id])

        self.bodies: dict[str, int] = {}
        self.body_geoms: dict[str, set[int]] = {}
        for name in ALL_NAMES:
            body = model.body(f"actor_{name}")
            self.bodies[name] = int(model.body_mocapid[body.id])
            self.body_geoms[name] = self._descendants(body.id)
        self.self_bodies = self._descendants(trunk_id) | {trunk_id}
        self.goal_geoms = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("goal_beacon", "goal_band", "goal_pylon_n",
                         "goal_pylon_s")
        }
        self.goal_geoms.discard(-1)

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

    def set_gesture(self, elapsed: float | None) -> None:
        """There is no head gesture in this behavior.  Always a no-op.

        Kept as a named function so the ABSENCE is discoverable rather than
        silent: a sibling behavior has an arrival gesture, and carrying its code
        here unused — or worse, wiring it in without measuring it on this scene —
        would be shipping unmeasured behavior.
        """
        return

    # -- visibility ------------------------------------------------------
    def _body_origin(self, name: str) -> np.ndarray:
        return self.render_data.mocap_pos[self.bodies[name]].copy()

    def sample_points(self, name: str) -> list[np.ndarray]:
        """The five world points the camera tests on this body.

        Offsets are scaled by the body's own stature, so a shorter one is
        genuinely sampled lower.
        """
        origin = self._body_origin(name)
        return [origin + np.array([0.0, 0.0, dz])
                for dz in BY_NAME[name].sample_dz]

    def _visible_samples(self, points, own_bodies: set[int],
                         own_geoms: set[int] | None = None
                         ) -> tuple[list[bool], float, float]:
        """Which of a set of world points the camera can see.

        ``off_axis`` is reported only over samples that are actually INSIDE the
        frustum and unoccluded.  Reporting the off-axis angle of a sample the
        camera cannot see would let a gate open on something behind a crate.
        """
        eye = self.render_data.cam_xpos[self.view_cam].copy()
        rotation = self.render_data.cam_xmat[self.view_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        centre = np.mean(np.asarray(points, dtype=np.float64), axis=0)
        range_m = float(np.linalg.norm(centre - eye))

        seen: list[bool] = []
        best_off_axis = math.pi
        for target in points:
            delta = np.asarray(target, dtype=np.float64) - eye
            distance = float(np.linalg.norm(delta))
            unit = delta / max(distance, 1e-9)
            depth = float(delta @ forward)
            in_fov = (depth > 0.0
                      and abs(float(delta @ right)) <= depth * self.tan_h
                      and abs(float(delta @ up)) <= depth * self.tan_v)
            if not in_fov or self._occluded(eye, unit, distance, own_bodies,
                                            own_geoms):
                seen.append(False)
                continue
            seen.append(True)
            best_off_axis = min(best_off_axis, math.acos(
                float(np.clip(unit @ forward, -1.0, 1.0))))
        return seen, best_off_axis, range_m

    def _occluded(self, eye, direction, distance, own_bodies: set[int],
                  own_geoms: set[int] | None = None) -> bool:
        """Ray cast through real scene geometry.

        Hitting the TARGET'S OWN body ends the cast successfully — that is what
        seeing it means, and a ray to a point on a body's centreline
        necessarily strikes that body first.  Hitting the duck's own geometry
        advances the ray past it rather than reporting occlusion.
        """
        travelled = 0.02
        geom_id = np.zeros(1, dtype=np.int32)
        for _ in range(12):
            origin = eye + direction * travelled
            hit = mujoco.mj_ray(self.model, self.render_data, origin, direction,
                                None, 1, -1, geom_id)
            if geom_id[0] < 0 or hit < 0.0:
                return False
            geom = int(geom_id[0])
            body = int(self.model.geom_bodyid[geom])
            if body in own_bodies or (own_geoms and geom in own_geoms):
                return False
            if body in self.self_bodies:
                travelled += hit + 0.005
                if travelled >= distance:
                    return False
                continue
            return travelled + hit < distance - 0.02
        return False

    def blocking_geom(self, name: str) -> str:
        """Name of the geom that blocks the CHEST sample of ``name``, if any."""
        eye = self.render_data.cam_xpos[self.view_cam].copy()
        target = self.sample_points(name)[2]
        delta = target - eye
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-9)
        travelled = 0.02
        geom_id = np.zeros(1, dtype=np.int32)
        for _ in range(12):
            origin = eye + direction * travelled
            hit = mujoco.mj_ray(self.model, self.render_data, origin, direction,
                                None, 1, -1, geom_id)
            if geom_id[0] < 0 or hit < 0.0:
                return ""
            body = int(self.model.geom_bodyid[int(geom_id[0])])
            if body in self.body_geoms[name]:
                return ""
            if body in self.self_bodies:
                travelled += hit + 0.005
                if travelled >= distance:
                    return ""
                continue
            if travelled + hit < distance - 0.02:
                return mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0])) or ""
            return ""
        return ""

    def goal_visibility(self) -> dict:
        """What the head camera can see of the destination beacon.

        Sampled through the IDENTICAL frustum and ray cast used for every body,
        so "it could see where it was going" is the same kind of measurement as
        "it could see the cart".
        """
        seen, off_axis, range_m = self._visible_samples(
            goal_sample_points(), set(), self.goal_geoms)
        count = sum(seen)
        return {
            "visible": count > 0,
            "samples": seen,
            "sample_count": count,
            "fraction": count / len(seen),
            "off_axis_deg": math.degrees(off_axis),
            "range_m": range_m,
        }

    # -- public ----------------------------------------------------------
    def update(self, data, *, duck_yaw: float, subject: str,
               look_at=None) -> dict:
        """Pose the camera for this tick and measure what it actually sees.

        ``subject`` is the body the head tracks, chosen by ``slalom_aim`` from
        the current state: whoever the duck is negotiating with right now, or
        ``""`` when it is looking at the goal.
        """
        self._pose_head(data, duck_yaw)
        target = (np.asarray(look_at, dtype=np.float64) if look_at is not None
                  else self._body_origin(subject) if subject
                  else np.array([0.0, 0.0, 0.3]))
        self._aim_at(target)
        self._pose_head(data, duck_yaw)
        self._orient_rig()

        bodies: dict[str, dict] = {}
        visible: list[str] = []
        for name in ALL_NAMES:
            seen, off_axis, range_m = self._visible_samples(
                self.sample_points(name), self.body_geoms[name])
            count = sum(seen)
            bodies[name] = {
                "visible": count > 0,
                "samples": seen,
                "sample_count": count,
                "fraction": count / len(seen),
                "off_axis_deg": math.degrees(off_axis),
                "range_m": range_m,
            }
            if count > 0:
                visible.append(name)

        return {
            "bodies": bodies,
            "visible_bodies": visible,
            "goal": self.goal_visibility(),
            "subject": subject,
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
        }

    @property
    def camera_id(self) -> int:
        return self.view_cam
