#!/usr/bin/env python3
"""Exact duck-versus-vehicle surface distance, and the traps that faked it.

Self-contained geometry with no knowledge of states, traffic schedules or
cameras — and the two MuJoCo pitfalls it documents must stay findable.

Why this matters for THIS behavior
----------------------------------
Vehicles are non-colliding mocap scenery, so MuJoCo will happily let a car pass
straight through the duck without registering a contact.  "Zero contacts"
from the physics engine is therefore vacuous here.  The honest gate is a
GEOMETRIC one: the smallest surface-to-surface separation between the duck and
every road user, measured every control tick and required to stay positive.

TWO MEASURED TRAPS
------------------
1. **``mj_geomDistance`` returns the CUTOFF ITSELF**, not the true distance,
   for a pair farther apart than the cutoff.  Feeding the running minimum back
   in as the next cutoff collapses the scan: once any pair returns ``x`` every
   later pair is clamped to ``x``, so a single ``0.0`` reports contact for the
   whole frame.  The cutoff is therefore held FIXED and the minimum accumulated
   separately.

2. **MuJoCo's narrowphase returns exactly ``0.0`` for pairs that are plainly
   apart**, whenever a MESH is paired with a primitive.  ``move-away-crowd``
   measured 65 spurious zeros in 264,000 samples against carried boxes, and
   ``come-here-recall`` hit it against a cap brim whose geom centres were
   0.5526 m apart.

   THIS BEHAVIOR HIT A NEW VARIANT: **mesh-versus-CYLINDER**.  Measured inside
   the real rollout, ``mj_geomDistance`` returned ``0.000000`` for

   * a duck mesh against ``sedan_wheel_fl`` at t=7.12 s, geom centres
     **1.0629 m** apart;
   * a duck mesh against ``scooter_wheel_f`` at t=16.14 s, centres **1.1817 m**;
   * a duck mesh against ``scooter_wheel_r`` at t=16.40 s, centres **1.1602 m**.

   Seven of 2300 control steps reported exact-zero clearance while the nearest
   vehicle was more than a metre away.  Restricting the analytic fallback to
   boxes — which is what the two earlier behaviors needed — would have left
   every wheel on every vehicle able to fake a contact, and the contact gate
   would have failed for a reason that has nothing to do with the robot.

   Worse, the artifact is **state-dependent**: reconstructing the same duck and
   vehicle poses in a fresh ``MjData`` and re-running the same query returns
   the correct distance and no spurious pair at all.  It cannot be reproduced
   outside the rollout, so it cannot be worked around by screening poses.

   The robot is all meshes and every vehicle is built entirely from primitives,
   so **no vehicle geom goes through ``mj_geomDistance`` at all**.  Every one is
   handled by :func:`primitive_sphere_distance` against each robot geom's
   bounding sphere: exact for the primitive, conservative for the mesh.  It can
   only UNDER-report clearance — it cannot hide a real contact.
"""

from __future__ import annotations

import math

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


def cylinder_sphere_distance(model: mujoco.MjModel, data: mujoco.MjData,
                             geom: int, center: np.ndarray,
                             radius: float) -> float:
    """Exact distance from an oriented cylinder's surface to a sphere.

    In the cylinder's local frame the surface is the set ``{|r| <= R,
    |z| <= h}``.  Clamping the sphere centre to that set and measuring the
    residual gives the exact surface distance, positive outside and negative
    inside.
    """
    pos = data.geom_xpos[geom]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    cyl_radius = float(model.geom_size[geom][0])
    half_height = float(model.geom_size[geom][1])
    local = rotation.T @ (np.asarray(center, dtype=np.float64) - pos)
    radial = float(np.linalg.norm(local[:2]))
    outside_radial = max(radial - cyl_radius, 0.0)
    outside_axial = max(abs(float(local[2])) - half_height, 0.0)
    if outside_radial == 0.0 and outside_axial == 0.0:
        # Centre is inside: signed depth to the nearest surface.
        return -min(cyl_radius - radial, half_height - abs(float(local[2]))) - radius
    return math.hypot(outside_radial, outside_axial) - radius


def capsule_sphere_distance(model: mujoco.MjModel, data: mujoco.MjData,
                            geom: int, center: np.ndarray,
                            radius: float) -> float:
    """Exact distance from a capsule's surface to a sphere."""
    pos = data.geom_xpos[geom]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    cap_radius = float(model.geom_size[geom][0])
    half_height = float(model.geom_size[geom][1])
    local = rotation.T @ (np.asarray(center, dtype=np.float64) - pos)
    # Closest point on the capsule's spine, then treat it as a sphere.
    clamped = np.array(
        [0.0, 0.0, float(np.clip(local[2], -half_height, half_height))])
    return float(np.linalg.norm(local - clamped)) - cap_radius - radius


def sphere_sphere_distance(model: mujoco.MjModel, data: mujoco.MjData,
                           geom: int, center: np.ndarray,
                           radius: float) -> float:
    """Exact distance between two spheres' surfaces."""
    delta = np.asarray(center, dtype=np.float64) - data.geom_xpos[geom]
    return float(np.linalg.norm(delta)) - float(model.geom_size[geom][0]) - radius


def bounding_sphere_distance(model: mujoco.MjModel, data: mujoco.MjData,
                             geom: int, center: np.ndarray,
                             radius: float) -> float:
    """Conservative distance using the geom's own bounding sphere.

    Used for shapes with no cheap exact form (an ellipsoid, say).  It always
    UNDER-reports the true clearance, which is the safe direction: it can raise
    a false alarm but it cannot hide a contact.
    """
    delta = np.asarray(center, dtype=np.float64) - data.geom_xpos[geom]
    return float(np.linalg.norm(delta)) - float(model.geom_rbound[geom]) - radius


def primitive_sphere_distance(model: mujoco.MjModel, data: mujoco.MjData,
                              geom: int, center: np.ndarray,
                              radius: float) -> float:
    """Distance from any non-mesh geom's surface to a sphere.

    Exact for boxes, cylinders, capsules and spheres; conservative (bounding
    sphere) for everything else.  This is the ONLY path vehicle geometry takes,
    because ``mj_geomDistance`` cannot be trusted for mesh-versus-primitive
    pairs in this scene — see the module docstring.
    """
    kind = int(model.geom_type[geom])
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return box_sphere_distance(model, data, geom, center, radius)
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return cylinder_sphere_distance(model, data, geom, center, radius)
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return capsule_sphere_distance(model, data, geom, center, radius)
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return sphere_sphere_distance(model, data, geom, center, radius)
    return bounding_sphere_distance(model, data, geom, center, radius)


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

    This is the number ``street.DUCK_PLANAR_RADIUS`` must match: lane occupancy,
    wait-line encroachment and the crossing-duration estimate are all graded on
    the trunk centre inflated by it, so a stale value would quietly weaken every
    one of those gates.  A test pins the two together.
    """
    center = data.xpos[trunk_id][:2]
    radius = 0.0
    for geom in geoms_of(model, body_subtree(model, trunk_id)):
        offset = float(np.linalg.norm(data.geom_xpos[geom][:2] - center))
        radius = max(radius, offset + float(model.geom_rbound[geom]))
    return radius


class ContactProbe:
    """Exact surface-to-surface distance between the duck and each road user.

    EVERY vehicle geom is handled analytically.  ``mj_geomDistance`` is not
    called at all on this scene, because the mesh-versus-primitive narrowphase
    was measured returning exact zeros for pairs over a metre apart (see the
    module docstring).  The duck side is approximated by each robot geom's
    bounding sphere, which is conservative.
    """

    def __init__(self, model: mujoco.MjModel, trunk_id: int,
                 vehicle_names: tuple[str, ...]):
        self.model = model
        self.duck_geoms = geoms_of(model, body_subtree(model, trunk_id))
        self.vehicle_geoms: dict[str, list[int]] = {}
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        for name in vehicle_names:
            geoms = geoms_of(
                model, body_subtree(model, model.body(f"vehicle_{name}").id))
            meshes = [g for g in geoms if int(model.geom_type[g]) == mesh_type]
            if meshes:
                raise RuntimeError(
                    f"vehicle {name!r} carries mesh geoms {meshes}; this probe "
                    "is only valid for all-primitive vehicles")
            self.vehicle_geoms[name] = geoms
        self.duck_rbound = {
            g: float(model.geom_rbound[g]) for g in self.duck_geoms}

    def distance(self, data: mujoco.MjData, name: str,
                 cutoff: float = 1.5) -> float:
        """Smallest surface separation between the duck and one road user.

        Negative means real geometric overlap.  Values beyond ``cutoff`` are
        reported as ``cutoff``: the exact number only matters near contact.
        """
        best = cutoff
        for vehicle_geom in self.vehicle_geoms[name]:
            for duck_geom in self.duck_geoms:
                distance = primitive_sphere_distance(
                    self.model, data, vehicle_geom,
                    data.geom_xpos[duck_geom], self.duck_rbound[duck_geom])
                if distance < best:
                    best = distance
        return best
