#!/usr/bin/env python3
"""Exact PiP visibility for Protective Personal Space.

This mixin reads only an isolated render ``MjData`` supplied by ``PpsCamera``.
Identity remains a simulator body-id proxy; visibility is real frustum geometry
plus MuJoCo ray casting through the scene actually rendered in the PiP.
"""
from __future__ import annotations

import math
import mujoco
import numpy as np

from pps_cast import BY_NAME


class CameraVisibility:
    """Body-sample visibility mixed into a camera owning the documented fields."""

    def body_origin(self, name: str) -> np.ndarray:
        return self.render_data.mocap_pos[self.bodies[name]].copy()

    def sample_points(self, name: str) -> list[np.ndarray]:
        origin = self.body_origin(name)
        return [origin + np.array([0.0, 0.0, dz])
                for dz in BY_NAME[name].sample_dz]

    def _point_visible(self, target, own_bodies: set[int]) -> tuple[bool, float]:
        eye = self.render_data.cam_xpos[self.view_cam].copy()
        rotation = self.render_data.cam_xmat[self.view_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        delta = np.asarray(target, dtype=np.float64) - eye
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return True, 0.0
        unit = delta / distance
        depth = float(delta @ forward)
        inside = (depth > 0.0
                  and abs(float(delta @ right)) <= depth * self.tan_h
                  and abs(float(delta @ up)) <= depth * self.tan_v)
        if not inside or self._occluded(eye, unit, distance, own_bodies):
            return False, math.pi
        return True, math.acos(float(np.clip(unit @ forward, -1.0, 1.0)))

    def has_line_of_sight(self, name: str) -> bool:
        """True when at least one body sample is not occluded, independent of FOV."""
        eye = self.render_data.cam_xpos[self.view_cam].copy()
        own = self.body_geoms[name]
        for point in self.sample_points(name):
            delta = np.asarray(point, dtype=np.float64) - eye
            distance = float(np.linalg.norm(delta))
            if distance < 1e-9:
                return True
            if not self._occluded(eye, delta / distance, distance, own):
                return True
        return False

    def visible_samples(self, name: str) -> tuple[list[bool], float, float]:
        origin = self.body_origin(name)
        eye = self.render_data.cam_xpos[self.view_cam]
        seen, best = [], math.pi
        for point in self.sample_points(name):
            visible, off_axis = self._point_visible(point, self.body_geoms[name])
            seen.append(visible)
            if visible:
                best = min(best, off_axis)
        return seen, best, float(np.linalg.norm(origin - eye))

    def _occluded(self, eye, direction, distance, own_bodies: set[int]) -> bool:
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
        eye = self.render_data.cam_xpos[self.view_cam].copy()
        target = self.sample_points(name)[2]
        delta = target - eye
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-9)
        travelled = 0.02
        geom_id = np.zeros(1, dtype=np.int32)
        for _ in range(12):
            hit = mujoco.mj_ray(
                self.model, self.render_data, eye + direction * travelled,
                direction, None, 1, -1, geom_id)
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
                return (mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0])) or "")
            return ""
        return ""

    def point_in_frustum(self, point) -> bool:
        visible, _ = self._point_visible(point, set())
        return visible
