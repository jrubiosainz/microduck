#!/usr/bin/env python3
"""Visibility measurement through the REAL PiP camera.

Frustum containment plus an occlusion ray cast against actual scene geometry.
The IDENTITY of each person comes from the simulator, which is a semantic proxy
for person recognition and not RGB classification; the CAMERA GEOMETRY is real,
and it is the exact camera the PiP renders from, so the reported percentages and
the picture always agree.

WHY OCCLUSION IS THE HARD PART IN A QUEUE
-----------------------------------------
In a queue somebody is *always* in front of somebody else, so occlusion is the
normal case rather than an edge case, and two rules matter:

* **Hitting the target's own body ends the cast successfully.**  A ray toward a
  point on somebody's centreline necessarily strikes their own torso first -
  a 0.078 m capsule against a 0.02 m tolerance - so treating that as occlusion
  scores a perfectly visible person at zero.
* **Hitting the duck's own geometry advances the ray** past itself rather than
  reporting a blocked view.

Several sample points per person (knees, torso, upper torso, head) are used
because the person in front covers the torso centre while head and legs stay
plainly visible.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

# Sample points on a person, relative to their mocap origin (z up).
SAMPLE_OFFSETS: tuple[float, ...] = (-0.06, 0.06, 0.20, 0.30)


class VisibilityMixin:
    """Measures what the PiP camera can actually see.

    Expects the host to provide ``model``, ``render_data``, ``queue_cam``,
    ``people``, ``person_bodies``, ``self_bodies``, ``tan_h`` and ``tan_v``.
    """

    def _person_center(self, name: str) -> np.ndarray:
        return self.render_data.mocap_pos[self.people[name]].copy()

    def _visible(self, name: str) -> tuple[bool, float, float, float]:
        """(visible, fraction of samples seen, off-axis angle, range).

        ``off_axis`` is reported only for samples that are actually INSIDE the
        frustum and unoccluded.  Reporting the off-axis angle of a sample the
        camera cannot see would let a gate open on somebody hidden behind
        another person - which, in a queue, is the normal case.
        """
        center = self._person_center(name)
        eye = self.render_data.cam_xpos[self.queue_cam].copy()
        rotation = self.render_data.cam_xmat[self.queue_cam].reshape(3, 3)
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        distance_to_center = float(np.linalg.norm(center - eye))
        seen = 0
        best_off_axis = math.pi
        for z_offset in SAMPLE_OFFSETS:
            target = center + np.array([0.0, 0.0, z_offset])
            delta = target - eye
            distance = float(np.linalg.norm(delta))
            unit = delta / max(distance, 1e-9)
            depth = float(delta @ forward)
            in_fov = (depth > 0.0
                      and abs(float(delta @ right)) <= depth * self.tan_h
                      and abs(float(delta @ up)) <= depth * self.tan_v)
            if not in_fov or self._occluded(eye, unit, distance, name):
                continue
            seen += 1
            best_off_axis = min(best_off_axis, math.acos(
                float(np.clip(unit @ forward, -1.0, 1.0))))
        return (seen > 0, seen / len(SAMPLE_OFFSETS), best_off_axis,
                distance_to_center)

    def _occluded(self, eye, direction, distance, name) -> bool:
        """Ray cast through real scene geometry.

        Hitting the TARGET'S OWN body ends the cast successfully - that is what
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
