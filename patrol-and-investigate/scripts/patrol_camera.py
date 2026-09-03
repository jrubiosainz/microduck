#!/usr/bin/env python3
"""What the duck can actually see, and where it is looking.

The camera pose lives in an **isolated rendering ``MjData``** that is a copy of
the authoritative walking state.  Head yaw/pitch and the stabilized rig are posed
only in that copy and never written back into the locomotion data, because the
head is a large fraction of the robot's mass and the stock walking policy was
never trained to compensate an imposed head trajectory.  Gaze therefore cannot
prop the robot up, and the physical locomotion state remains the single source
of truth.

Visibility is measured from the EXACT camera the PiP is rendered from, so the
reported percentages and the picture agree.  The test is real camera geometry -
frustum containment plus an occlusion ray cast through actual scene geometry -
but the IDENTITY of each body comes from the simulator.  That is a **semantic
proxy** for object recognition, not RGB classification, and it is labelled as
such wherever it surfaces.

THE SCAN IS WHY THIS CAMERA IS DIFFERENT FROM ITS SIBLINGS
------------------------------------------------------------
Turning in place is MEASURED to be unavailable on this policy - at most
1.6 deg/s at ``vx = 0`` - so a stopped robot cannot sweep its body across a
room.  Every checkpoint scan is therefore a HEAD sweep, and
:meth:`PatrolCamera.scan_yaw` drives it: a triangle wave about the checkpoint's
own outward watch bearing, at the MEASURED 26 deg/s head rate, covering
+/-52 deg.  The arc the head actually travelled is accumulated from the pose it
actually reached, so "it swept 104 deg" is a measurement rather than the
commanded amplitude.

THE CAMERA GATE THE DETECTOR USES IS THIS ONE
-----------------------------------------------
``update`` returns a per-body visibility dictionary, and that dictionary is the
ONLY way a body can become a detection candidate.  A body outside the frustum,
behind the central rack, or behind another body simply is not in it.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT, wrap_angle
from patrol_cast import ALL_NAMES, BY_NAME
from patrol_states import (
    SCAN_SWEEP_DEG,
    TRACK_PITCH_DEG,
    TRACK_PITCH_RATE_DPS,
    TRACK_YAW_RATE_DPS,
)

# PiP pixel geometry lives HERE, not in an overlay, because it sets the camera's
# horizontal FOV and therefore every visibility measurement.  The headless gate
# and any render must measure through the same frustum.
PIP_W, PIP_H = 300, 216


class PatrolCamera:
    """Head camera: sweep, track, and report what it actually saw."""

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
        self.view_cam = model.camera("patrol_camera").id
        rig = model.body("patrol_rig")
        self.rig_mocap = int(model.body_mocapid[rig.id])

        self.bodies: dict[str, int] = {}
        self.body_geoms: dict[str, set[int]] = {}
        for name in ALL_NAMES:
            body = model.body(f"actor_{name}")
            self.bodies[name] = int(model.body_mocapid[body.id])
            self.body_geoms[name] = self._descendants(body.id)
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

        # Scan bookkeeping.  ``scan_arc_deg`` accumulates the arc the head
        # ACTUALLY travelled, from the pose it actually reached each tick, so a
        # sweep that was cut short reports the shorter arc.
        self._scan_t = 0.0
        self._scan_last_yaw: float | None = None
        self.scan_arc_deg = 0.0

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

    def begin_scan(self) -> None:
        """Reset the sweep.  Called when a checkpoint scan starts."""
        self._scan_t = 0.0
        self._scan_last_yaw = None
        self.scan_arc_deg = 0.0

    def scan_target(self, duck_xy, watch_deg: float) -> np.ndarray:
        """The world point the head sweeps to next, for a checkpoint scan.

        A TRIANGLE WAVE about the checkpoint's own outward watch bearing, at the
        MEASURED head rate.  Returning a POINT rather than a yaw keeps a single
        aiming path through :meth:`_aim_at`, so the sweep is rate-limited by the
        same measured constant that limits target tracking - which is what makes
        the swept arc a physical quantity rather than a commanded one.
        """
        span = math.radians(SCAN_SWEEP_DEG)
        rate = math.radians(TRACK_YAW_RATE_DPS)
        # Triangle wave in yaw: out to +span, back through -span, and return.
        period = 4.0 * span / rate
        phase = (self._scan_t % period) / period
        if phase < 0.25:
            offset = span * (phase / 0.25)
        elif phase < 0.75:
            offset = span * (1.0 - 2.0 * (phase - 0.25) / 0.5)
        else:
            offset = span * (-1.0 + (phase - 0.75) / 0.25)
        self._scan_t += self.dt
        yaw = math.radians(watch_deg) + offset
        duck = np.asarray(duck_xy, dtype=np.float64)[:2]
        # A point 2.2 m out along the swept bearing, at chest height: far enough
        # that the aim is a bearing rather than a nearby point the head would
        # converge on and stop.
        return np.array([duck[0] + 2.2 * math.cos(yaw),
                         duck[1] + 2.2 * math.sin(yaw), 0.30])

    def note_scan_arc(self) -> None:
        """Accumulate the arc the head ACTUALLY travelled this tick."""
        if self._scan_last_yaw is not None:
            self.scan_arc_deg += abs(math.degrees(
                wrap_angle(self.view_yaw - self._scan_last_yaw)))
        self._scan_last_yaw = self.view_yaw

    # -- visibility ------------------------------------------------------
    def _body_origin(self, name: str) -> np.ndarray:
        return self.render_data.mocap_pos[self.bodies[name]].copy()

    def sample_points(self, name: str) -> list[np.ndarray]:
        """The five world points the camera tests on this body.

        Offsets are scaled by the body's own stature for a person and spread
        over its own box for an object, so each is genuinely sampled over the
        solid the viewer sees.
        """
        origin = self._body_origin(name)
        return [origin + np.array([0.0, 0.0, dz])
                for dz in BY_NAME[name].sample_dz]

    def _visible_samples(self, points, own_bodies: set[int]
                         ) -> tuple[list[bool], float, float]:
        """Which of a set of world points the camera can see.

        ``off_axis`` is reported only over samples that are actually INSIDE the
        frustum and unoccluded.  Reporting the off-axis angle of a sample the
        camera cannot see would let a gate open on something behind the rack.
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
            if not in_fov or self._occluded(eye, unit, distance, own_bodies):
                seen.append(False)
                continue
            seen.append(True)
            best_off_axis = min(best_off_axis, math.acos(
                float(np.clip(unit @ forward, -1.0, 1.0))))
        return seen, best_off_axis, range_m

    def _occluded(self, eye, direction, distance, own_bodies: set[int]) -> bool:
        """Ray cast through real scene geometry.

        Hitting the TARGET'S OWN body ends the cast successfully - that is what
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
            body = int(self.model.geom_bodyid[int(geom_id[0])])
            if body in own_bodies:
                return False
            if body in self.self_bodies:
                travelled += hit + 0.005
                if travelled >= distance:
                    return False
                continue
            return travelled + hit < distance - 0.02
        return False

    def blocking_geom(self, name: str) -> str:
        """Name of the geom that blocks the MIDDLE sample of ``name``, if any."""
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

    def point_in_frustum(self, point) -> bool:
        """Is a world point inside the camera's own frustum right now?

        This is what "the camera was ACTIVE on its target" is measured with, and
        it is deliberately a property the head can FAIL to satisfy.  The gaze is
        rate-limited to the MEASURED 26 deg/s, so whenever the aim point moves
        faster than that - the duck turning a 60 deg corner, or a sweep
        reversing - the head genuinely lags and its target falls outside the
        frustum for a few ticks.  Asking instead whether the camera was "aimed
        at" its target would be vacuous, because by construction it always is.
        """
        eye = self.render_data.cam_xpos[self.view_cam]
        rotation = self.render_data.cam_xmat[self.view_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        delta = np.asarray(point, dtype=np.float64) - eye
        depth = float(delta @ forward)
        return bool(depth > 0.0
                    and abs(float(delta @ right)) <= depth * self.tan_h
                    and abs(float(delta @ up)) <= depth * self.tan_v)

    # -- public ----------------------------------------------------------
    def update(self, data, *, duck_yaw: float, subject: str, look_at,
               present: dict[str, bool]) -> dict:
        """Pose the camera for this tick and measure what it actually sees.

        ``present`` marks which bodies physically exist yet.  One that does not
        is reported as not visible without a ray cast: it is parked below the
        floor, so the cast would answer the same thing more slowly.
        """
        self._pose_head(data, duck_yaw)
        self._aim_at(np.asarray(look_at, dtype=np.float64))
        self._pose_head(data, duck_yaw)
        self._orient_rig()

        bodies: dict[str, dict] = {}
        visible: list[str] = []
        for name in ALL_NAMES:
            if not present.get(name, True):
                bodies[name] = {"visible": False, "samples": [False] * 5,
                                "sample_count": 0, "fraction": 0.0,
                                "off_axis_deg": 180.0, "range_m": 99.0,
                                "present": False}
                continue
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
                "present": True,
            }
            if count > 0:
                visible.append(name)

        return {
            "bodies": bodies,
            "visible_bodies": visible,
            "subject": subject,
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
            "scan_arc_deg": self.scan_arc_deg,
            # Whether the point the head was TOLD to look at is actually in
            # frame.  See :meth:`point_in_frustum`.
            "aim_in_frustum": self.point_in_frustum(look_at),
        }

    @property
    def camera_id(self) -> int:
        return self.view_cam
