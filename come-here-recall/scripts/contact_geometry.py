#!/usr/bin/env python3
"""Exact duck-versus-person surface distance, and the traps that faked it.

Split out of ``rollout_recall`` because it is self-contained geometry with no
knowledge of calls, states or cameras — and because the two MuJoCo pitfalls it
documents are the kind of thing that must stay findable.

TWO MEASURED TRAPS
------------------
1. **``mj_geomDistance`` returns the CUTOFF ITSELF**, not the true distance,
   for a pair farther apart than the cutoff.  Feeding the running minimum back
   in as the next cutoff collapses the scan: once any pair returns ``x`` every
   later pair is clamped to ``x``, so a single ``0.0`` reports contact for the
   whole frame.  The cutoff is therefore held FIXED and the minimum accumulated
   separately.

2. **MuJoCo's MESH-vs-BOX narrowphase returns exactly ``0.0``** for pairs that
   are plainly apart.  ``move-away-crowd`` measured 65 spurious zeros in
   264,000 samples against carried boxes.  The same artifact appeared in this
   behavior at ``t=28.02 s``: one step reported clearance ``0.00000`` against
   yellow while the steps on either side reported 0.21-0.23 m, and the
   offending pair was a duck mesh against ``yellow_brim`` whose geom centres
   were **0.5526 m** apart.

   The robot is all meshes and the only boxes on an adult are the cap brims, so
   box pairs are handled analytically instead: :func:`box_sphere_distance`
   against each robot geom's bounding sphere.  That is exact for the box and
   conservative for the mesh, so it can only UNDER-report clearance — it cannot
   hide a real contact.
"""

from __future__ import annotations

import mujoco
import numpy as np


def box_sphere_distance(model: mujoco.MjModel, data: mujoco.MjData,
                        box_geom: int, center: np.ndarray, radius: float) -> float:
    """Exact distance from an oriented box's surface to a sphere.

    Transform the sphere centre into the box frame, clamp per axis, take the
    norm of the outside part, and subtract the radius.  Negative means overlap.
    """
    box_pos = data.geom_xpos[box_geom]
    rotation = data.geom_xmat[box_geom].reshape(3, 3)
    half = model.geom_size[box_geom]
    local = rotation.T @ (np.asarray(center, dtype=np.float64) - box_pos)
    outside = np.maximum(np.abs(local) - half, 0.0)
    outside_distance = float(np.linalg.norm(outside))
    if outside_distance == 0.0:
        # Centre is inside the box: signed depth to the nearest face.
        return -float(np.min(half - np.abs(local))) - radius
    return outside_distance - radius


def body_subtree(model: mujoco.MjModel, root: int) -> set[int]:
    """``root`` and every body beneath it."""
    bodies = {root}
    for body in range(model.nbody):
        parent = body
        while parent > 0:
            if parent == root:
                bodies.add(body)
                break
            parent = int(model.body_parentid[parent])
    return bodies


def geoms_of(model: mujoco.MjModel, bodies: set[int]) -> list[int]:
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]


def duck_planar_radius(model: mujoco.MjModel, data: mujoco.MjData,
                       trunk_id: int) -> float:
    """Largest planar distance from the trunk origin to any robot geom surface.

    Reported for context and used to justify the standoff band; the contact
    gate itself uses exact geom-pair distances.
    """
    center = data.xpos[trunk_id][:2]
    radius = 0.0
    for geom in geoms_of(model, body_subtree(model, trunk_id)):
        offset = float(np.linalg.norm(data.geom_xpos[geom][:2] - center))
        radius = max(radius, offset + float(model.geom_rbound[geom]))
    return radius


class ContactProbe:
    """Exact surface-to-surface distance between the duck and each adult."""

    def __init__(self, model: mujoco.MjModel, trunk_id: int,
                 adult_names: tuple[str, ...]):
        self.model = model
        self.duck_geoms = geoms_of(model, body_subtree(model, trunk_id))
        self.adult_geoms: dict[str, list[int]] = {}
        self.adult_boxes: dict[str, list[int]] = {}
        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        for name in adult_names:
            geoms = geoms_of(
                model, body_subtree(model, model.body(f"person_{name}").id))
            self.adult_boxes[name] = [
                g for g in geoms if int(model.geom_type[g]) == box_type]
            self.adult_geoms[name] = [
                g for g in geoms if int(model.geom_type[g]) != box_type]
        self.duck_rbound = {
            g: float(model.geom_rbound[g]) for g in self.duck_geoms}

    def distance(self, data: mujoco.MjData, name: str,
                 cutoff: float = 1.2) -> float:
        """Smallest surface separation between the duck and one adult.

        Negative means real geometric overlap.  Values beyond ``cutoff`` are
        reported as ``cutoff``: the exact number only matters near contact.
        """
        best = cutoff
        for adult_geom in self.adult_geoms[name]:
            for duck_geom in self.duck_geoms:
                distance = float(mujoco.mj_geomDistance(
                    self.model, data, duck_geom, adult_geom, cutoff, None))
                if distance < best:
                    best = distance
        for box_geom in self.adult_boxes[name]:
            for duck_geom in self.duck_geoms:
                distance = box_sphere_distance(
                    self.model, data, box_geom,
                    data.geom_xpos[duck_geom], self.duck_rbound[duck_geom])
                if distance < best:
                    best = distance
        return best
