#!/usr/bin/env python3
"""WHAT THE CAMERA CAN ACTUALLY SEE: frustum containment and real ray casts.

Split out of :mod:`gest_camera` so that module stays about POSING the head and
this stays about MEASURING what the posed head can resolve.  Everything here
reads ``self.render_data`` - the ISOLATED copy the head was posed in - so no
measurement can be taken from a camera the physics never had.

THE ARM GATE IS WHAT MAKES THIS CAMERA DIFFERENT FROM ITS SIBLINGS
--------------------------------------------------------------------
A sibling behavior only had to answer "can the duck see this body".  Here that
is not enough: a gesture may be accepted only if the duck could read the ARM it
was made with.  :meth:`arm_readable` therefore tests the three keypoints of one
arm - shoulder, elbow and hand - individually, each with frustum containment and
its own occlusion ray cast, and requires all three.

That distinction is load-bearing rather than pedantic.  A person can be
comfortably in frame with their raised hand outside it: the duck is 0.20 m tall
and stands close enough to read a gesture, so an adult's raised arm sits near the
top of the frustum precisely when the torso is centred.  MEASURED on the real
run, the arm gate is strictly harder than the body gate - it fails on ticks
where the body gate passes - which is what stops "the duck saw the person" from
standing in for "the duck read the gesture".
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from gest_cast import BY_NAME

# The three keypoints of one arm, in the order the readability gate reports them.
ARM_JOINTS = ("shoulder", "elbow", "hand")


class CameraVisibility:
    """Frustum and occlusion measurement for a posed :class:`GestureCamera`."""

    # -- visibility ------------------------------------------------------
    def _body_origin(self, name: str) -> np.ndarray:
        return self.render_data.mocap_pos[self.bodies[name]].copy()

    def sample_points(self, name: str) -> list[np.ndarray]:
        """The five world points the camera tests on this person's body."""
        origin = self._body_origin(name)
        return [origin + np.array([0.0, 0.0, dz])
                for dz in BY_NAME[name].sample_dz]

    def arm_keypoints(self, name: str) -> dict[str, np.ndarray]:
        """The six arm keypoints, as MuJoCo placed them this tick.

        These are the SAME positions :mod:`gest_pose` measures its features
        from and the SAME positions :meth:`arm_readable` ray-casts to, which is
        what makes "the duck could read the arm it classified" one claim rather
        than two that happen to agree.
        """
        return {key: self.render_data.xpos[body].copy()
                for key, body in self.arm_bodies[name].items()}

    def _point_visible(self, point, own_bodies: set[int]) -> bool:
        """Frustum containment plus a real occlusion ray cast, for one point."""
        eye = self.render_data.cam_xpos[self.view_cam].copy()
        rotation = self.render_data.cam_xmat[self.view_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        delta = np.asarray(point, dtype=np.float64) - eye
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return True
        unit = delta / distance
        depth = float(delta @ forward)
        in_fov = (depth > 0.0
                  and abs(float(delta @ right)) <= depth * self.tan_h
                  and abs(float(delta @ up)) <= depth * self.tan_v)
        if not in_fov:
            return False
        return not self._occluded(eye, unit, distance, own_bodies)

    def arm_readable(self, name: str, side: str,
                     keypoints: dict[str, np.ndarray] | None = None) -> bool:
        """Could the camera read ALL THREE keypoints of one arm?

        Every keypoint individually, each with its own frustum test and its own
        occlusion ray cast.  Requiring all three is the whole point: an arm
        whose hand is outside the frame has not been read, however clearly its
        shoulder is in shot.
        """
        points = keypoints if keypoints is not None else self.arm_keypoints(name)
        own = self.body_geoms[name]
        return all(self._point_visible(points[f"{side}_{joint}"], own)
                   for joint in ARM_JOINTS)

    def _visible_samples(self, points, own_bodies: set[int]
                         ) -> tuple[list[bool], float, float]:
        """Which of a set of world points the camera can see.

        ``off_axis`` is reported only over samples that are actually INSIDE the
        frustum and unoccluded.  Reporting the off-axis angle of a sample the
        camera cannot see would let a gate open on somebody behind a rack.
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
        seeing it means, and a ray to a point on a body necessarily strikes that
        body first.  Hitting the duck's own geometry advances the ray past it
        rather than reporting occlusion.
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
        faster than that - the duck carving a turn arc, or a sweep reversing -
        the head genuinely lags and its target falls outside the frustum for a
        few ticks.  Asking instead whether the camera was "aimed at" its target
        would be vacuous, because by construction it always is.
        """
        eye = self.render_data.cam_xpos[self.view_cam]
        rotation = self.render_data.cam_xmat[self.view_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        delta = np.asarray(point, dtype=np.float64) - eye
        depth = float(delta @ forward)
        return bool(depth > 0.0
                    and abs(float(delta @ right)) <= depth * self.tan_h
                    and abs(float(delta @ up)) <= depth * self.tan_v)
