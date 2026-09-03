#!/usr/bin/env python3
"""What the duck can actually see, and what it can read off what it sees.

The camera pose lives in an **isolated rendering ``MjData``** that is a copy of
the authoritative walking state.  Head yaw/pitch and the stabilized rig are
posed only in that copy and never written back into the locomotion data, because
the head is a large fraction of the robot's mass and the stock walking policy was
never trained to compensate an imposed head trajectory.

Visibility is measured from the EXACT camera the PiP renders from
(``lost_camera``), so the reported percentages and the picture agree.  The test
is real camera geometry — frustum containment plus an occlusion ray cast through
actual scene geometry — but the IDENTITY of each body comes from the simulator.
That is a semantic proxy for person re-identification, not RGB classification,
and it is labelled as such wherever it surfaces.

PER-SAMPLE VISIBILITY IS THE WHOLE DESIGN
------------------------------------------
Each person is sampled at five heights — knees, waist, chest, head, crown —
scaled by their own stature.  The camera reports which of those five it can see,
not merely whether it can see "them", and the identity layer consumes exactly
that:

* the shirt and satchel are readable only if the torso samples are visible;
* the cap is readable only if the head samples are visible;
* stature is readable only if the knees AND the head are visible, because you
  cannot judge somebody's height from their shoulders up.

So a candidate standing half behind a column yields an INCOMPLETE descriptor and
can never be confirmed, however well the visible half matches.  That is what
makes the confirmation gate a duration rather than an instant, and it is why the
duck cannot be fooled by a glimpse.

THE SEARCH SWEEP IS DONE WITH THE HEAD, AND THAT IS A MEASUREMENT
------------------------------------------------------------------
MEASURED on this model: at ``vx = 0`` even ``wz = ±0.55`` turns the trunk about
1 deg/s, so a body scan is physically impossible.  The head yaw joint spans a
measured ±170 deg.  The only search this robot can perform is a head sweep at an
exactly-zero locomotion command — which is also the only one the acceptance gate
permits.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from lost_cast import ALL_NAMES, BY_NAME, FEATURE_SAMPLES
from lost_constants import (
    SCAN_AMPLITUDE_DEG,
    SCAN_PITCH_DEG,
    SCAN_RATE_DPS,
    TRACK_PITCH_RATE_DPS,
    TRACK_YAW_RATE_DPS,
)
from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT, wrap_angle

# PiP pixel geometry lives HERE, not in the overlay, because it sets the
# camera's horizontal FOV and therefore every visibility measurement.  The
# no-render gate and the final render must measure through the same frustum.
PIP_W, PIP_H = 300, 216


class LostCamera:
    """Head camera: sweep, track, and report per-sample visibility."""

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
        self.lost_cam = model.camera("lost_camera").id
        rig = model.body("lost_rig")
        self.rig_mocap = int(model.body_mocapid[rig.id])

        self.people: dict[str, int] = {}
        self.person_bodies: dict[str, set[int]] = {}
        for name in ALL_NAMES:
            body = model.body(f"person_{name}")
            self.people[name] = int(model.body_mocapid[body.id])
            self.person_bodies[name] = self._descendants(body.id)
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
        self.view_pitch = math.radians(SCAN_PITCH_DEG)

        half_v = math.radians(float(model.cam_fovy[self.lost_cam])) * 0.5
        self.tan_v = math.tan(half_v)
        self.tan_h = (self.pip_w / self.pip_h) * self.tan_v
        self.half_v_deg = math.degrees(half_v)
        self.half_h_deg = math.degrees(math.atan(self.tan_h))

        self._scan_offset = 0.0
        self._scan_dir = 1.0
        self.scan_reversals = 0

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
        PiP while the duck's trunk pitches through its gait.  It is labelled as
        stabilized in the overlay.
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
        yaw_rate = math.radians(TRACK_YAW_RATE_DPS)
        pitch_rate = math.radians(TRACK_PITCH_RATE_DPS)
        self.view_yaw = wrap_angle(self.view_yaw + float(np.clip(
            wrap_angle(desired_yaw - self.view_yaw), -yaw_rate, yaw_rate)))
        self.view_pitch += float(np.clip(
            desired_pitch - self.view_pitch, -pitch_rate, pitch_rate))

    def _scan(self, duck_yaw: float) -> None:
        """Sweep the head across the hall while the body stays exactly still.

        Advances a stored offset and reverses at the limits rather than being a
        closed-form function of time, so breaking off to evaluate a candidate
        and later resuming does not teleport the head.  ``scan_reversals``
        counts the turning points, which is how the metrics can state that the
        search was a real sweep rather than a stare.
        """
        amplitude = math.radians(SCAN_AMPLITUDE_DEG)
        self._scan_offset += self._scan_dir * math.radians(SCAN_RATE_DPS) * self.dt
        if self._scan_offset >= amplitude:
            self._scan_offset = amplitude
            self._scan_dir = -1.0
            self.scan_reversals += 1
        elif self._scan_offset <= -amplitude:
            self._scan_offset = -amplitude
            self._scan_dir = 1.0
            self.scan_reversals += 1
        self.view_yaw = wrap_angle(duck_yaw + self._scan_offset)
        self.view_pitch = math.radians(SCAN_PITCH_DEG)

    def resume_scan_from_view(self, duck_yaw: float) -> None:
        amplitude = math.radians(SCAN_AMPLITUDE_DEG)
        self._scan_offset = float(np.clip(
            wrap_angle(self.view_yaw - duck_yaw), -amplitude, amplitude))

    # -- visibility ------------------------------------------------------
    def _person_origin(self, name: str) -> np.ndarray:
        return self.render_data.mocap_pos[self.people[name]].copy()

    def sample_points(self, name: str) -> list[np.ndarray]:
        """The five world points the camera tests on this person.

        Offsets are scaled by the person's own stature, so a shorter adult is
        genuinely sampled lower and the stature feature is geometry rather than
        a label attached to a body of identical size.
        """
        origin = self._person_origin(name)
        return [origin + np.array([0.0, 0.0, dz])
                for dz in BY_NAME[name].sample_dz]

    def _visible_samples(self, name: str) -> tuple[list[bool], float, float]:
        """Which of the five samples the camera can see, plus off-axis and range.

        ``off_axis`` is reported only over samples that are actually INSIDE the
        frustum and unoccluded.  Reporting the off-axis angle of a sample the
        camera cannot see would let a gate open on somebody standing behind a
        column — which, in this hall, is the normal case.
        """
        eye = self.render_data.cam_xpos[self.lost_cam].copy()
        rotation = self.render_data.cam_xmat[self.lost_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        origin = self._person_origin(name)
        range_m = float(np.linalg.norm(origin - eye))

        seen: list[bool] = []
        best_off_axis = math.pi
        for target in self.sample_points(name):
            delta = target - eye
            distance = float(np.linalg.norm(delta))
            unit = delta / max(distance, 1e-9)
            depth = float(delta @ forward)
            in_fov = (depth > 0.0
                      and abs(float(delta @ right)) <= depth * self.tan_h
                      and abs(float(delta @ up)) <= depth * self.tan_v)
            if not in_fov or self._occluded(eye, unit, distance, name):
                seen.append(False)
                continue
            seen.append(True)
            best_off_axis = min(best_off_axis, math.acos(
                float(np.clip(unit @ forward, -1.0, 1.0))))
        return seen, best_off_axis, range_m

    def _occluded(self, eye, direction, distance, name) -> bool:
        """Ray cast through real scene geometry.

        Hitting the TARGET'S OWN body ends the cast successfully — that is what
        seeing them means, and a ray to a point on somebody's centreline
        necessarily strikes their own torso first.  Hitting the duck's own
        geometry advances the ray past it rather than reporting occlusion.
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
            if body in self.person_bodies[name]:
                return False
            if body in self.self_bodies:
                travelled += hit + 0.005
                if travelled >= distance:
                    return False
                continue
            return travelled + hit < distance - 0.02
        return False

    def blocking_geom(self, name: str) -> str:
        """Name of the geom that blocks the CHEST sample of ``name``, if any.

        Reported for the HUD and the metrics so an occlusion can be attributed
        to the kiosk, a column or another person by name, rather than merely
        being counted.
        """
        eye = self.render_data.cam_xpos[self.lost_cam].copy()
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
            if body in self.person_bodies[name]:
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

    @staticmethod
    def readable_features(seen: list[bool]) -> set[str]:
        """Which appearance features these visible samples permit reading."""
        return {feature for feature, indices in FEATURE_SAMPLES.items()
                if all(seen[i] for i in indices)}

    def observe(self, name: str, seen: list[bool]) -> dict:
        """The appearance descriptor readable from this set of visible samples.

        SEMANTIC PROXY.  The values come from the simulator's own description of
        the body; the camera decides only WHICH of them are readable.  A real
        system would run a re-identification network over the visible pixels;
        the visibility logic — and therefore every gate about occlusion,
        confirmation duration and false candidates — would be the same.
        """
        person = BY_NAME[name]
        readable = self.readable_features(seen)
        observed: dict = {}
        if "shirt" in readable:
            observed["shirt"] = person.shirt
        if "stature" in readable:
            observed["stature"] = person.height_m
        if "cap" in readable:
            observed["cap"] = person.cap
        if "satchel" in readable:
            observed["satchel"] = person.satchel
        return observed

    # -- public ----------------------------------------------------------
    def update(self, data, *, duck_yaw: float, subject: str | None,
               scanning: bool) -> dict:
        """Pose the camera for this tick and measure what it actually sees.

        ``subject`` is the body the head tracks — the guardian while following,
        or a candidate while it is being evaluated.  ``scanning`` forces the
        sweep even when a subject exists, which is what SEARCH_SWEEP does.
        """
        self._pose_head(data, duck_yaw)
        if scanning or subject is None:
            self._scan(duck_yaw)
        else:
            target = self._person_origin(subject)
            target[2] += 0.12 * BY_NAME[subject].stature
            self._aim_at(target)
        self._pose_head(data, duck_yaw)
        self._orient_rig()

        people: dict[str, dict] = {}
        visible: list[str] = []
        for name in ALL_NAMES:
            seen, off_axis, range_m = self._visible_samples(name)
            count = sum(seen)
            entry = {
                "visible": count > 0,
                "samples": seen,
                "sample_count": count,
                "fraction": count / len(seen),
                "off_axis_deg": math.degrees(off_axis),
                "range_m": range_m,
                "readable": sorted(self.readable_features(seen)),
                "observed": self.observe(name, seen),
            }
            people[name] = entry
            if count > 0:
                visible.append(name)

        return {
            "people": people,
            "visible_people": visible,
            "subject": subject,
            "scanning": bool(scanning),
            "view_yaw": self.view_yaw,
            "view_pitch": self.view_pitch,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
            "scan_reversals": self.scan_reversals,
        }

    @property
    def camera_id(self) -> int:
        return self.lost_cam
