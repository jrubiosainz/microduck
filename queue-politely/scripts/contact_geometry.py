#!/usr/bin/env python3
"""Exact duck-versus-scenery surface distance, and the traps that faked it.

Self-contained geometry with no knowledge of states, pedestrian schedules or
cameras — and the two MuJoCo pitfalls it documents must stay findable.

Why this matters for THIS behavior
----------------------------------
Both the adults and the corridor walls are non-colliding scenery, so MuJoCo
will happily let a person walk straight through the duck, or the duck press
through a wall, without registering a contact.  "Zero contacts" from the
physics engine is therefore vacuous here.  The honest gate is a GEOMETRIC one:
the smallest surface-to-surface separation between the duck and every adult,
and between the duck and every wall, measured every control tick and required
to stay positive.

The walls being non-colliding is deliberate.  If they collided, "the duck
stayed inside the corridor" would be enforced by the contact solver rather than
demonstrated by the controller, and a duck that scraped along a wall for ten
seconds would still pass.

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

   THE CROSSWALK BEHAVIOR HIT A NEW VARIANT: **mesh-versus-CYLINDER**.
   Measured inside its real rollout, ``mj_geomDistance`` returned ``0.000000``
   for

   * a duck mesh against ``sedan_wheel_fl`` with geom centres **1.0629 m**
     apart;
   * a duck mesh against ``scooter_wheel_f``/``_r`` at **1.1817 m** and
     **1.1602 m**.

   Seven of 2300 control steps reported exact-zero clearance while the nearest
   body was more than a metre away.  Worse, the artifact is
   **state-dependent**: reconstructing the same poses in a fresh ``MjData`` and
   re-running the same query returns the correct distance and no spurious pair
   at all.  It cannot be reproduced outside the rollout, so it cannot be worked
   around by screening poses.

   THIS BEHAVIOR IS THE WORST CASE FOR THAT TRAP.  The robot is all meshes, the
   corridor is entirely primitives, and the walls sit about 0.14 m from the
   robot's surface for most of the rollout — far closer, and for far longer,
   than any vehicle ever came in the crosswalk scene.  So **no scenery geom
   goes through ``mj_geomDistance`` at all**.  Every one is handled by
   :func:`primitive_sphere_distance` against each robot geom's bounding sphere:
   exact for the primitive, conservative for the mesh.  It can only UNDER-report
   clearance — it cannot hide a real contact.
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

    Computed from each geom's BOUNDING SPHERE, so it over-states the robot: a
    tall thin geom's bounding sphere is much wider than the geom itself.  That
    is the safe direction for every gate about the duck's own footprint —
    corridor clearance, centre-passage intrusion, alcove fit — because an
    over-wide robot makes each of those harder to satisfy.

    This is the number ``corridor.DUCK_PLANAR_RADIUS`` must match, and a test
    pins the two together.  Use :func:`exact_planar_radius` instead for the one
    claim where a fatter robot would flatter the scenario rather than test it.
    """
    center = data.xpos[trunk_id][:2]
    radius = 0.0
    for geom in geoms_of(model, body_subtree(model, trunk_id)):
        offset = float(np.linalg.norm(data.geom_xpos[geom][:2] - center))
        radius = max(radius, offset + float(model.geom_rbound[geom]))
    return radius


def _geom_planar_support(model: mujoco.MjModel, data: mujoco.MjData,
                         geom: int, center: np.ndarray) -> float:
    """Largest planar distance from ``center`` to one geom's actual surface.

    Exact for meshes (every vertex), spheres, capsules, cylinders and boxes;
    falls back to the bounding sphere for anything else, which is conservative.
    """
    kind = int(model.geom_type[geom])
    pos = np.asarray(data.geom_xpos[geom], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom], dtype=np.float64)

    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[geom])
        first = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        local = np.asarray(
            model.mesh_vert[first:first + count], dtype=np.float64)
        world = local @ rotation.T + pos
        return float(np.max(np.linalg.norm(world[:, :2] - center, axis=1)))

    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(np.linalg.norm(pos[:2] - center)) + float(size[0])

    if kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        # Both are a segment swept by a disc, so the planar support is reached
        # at one of the two end points, inflated by the radius.
        axis = rotation[:, 2] * float(size[1])
        ends = np.stack((pos + axis, pos - axis))
        return float(np.max(np.linalg.norm(ends[:, :2] - center, axis=1))
                     + float(size[0]))

    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        signs = np.array([[sx, sy, sz]
                          for sx in (-1.0, 1.0)
                          for sy in (-1.0, 1.0)
                          for sz in (-1.0, 1.0)])
        corners = (signs * size) @ rotation.T + pos
        return float(np.max(np.linalg.norm(corners[:, :2] - center, axis=1)))

    offset = float(np.linalg.norm(pos[:2] - center))
    return offset + float(model.geom_rbound[geom])


def exact_planar_radius(model: mujoco.MjModel, data: mujoco.MjData,
                        body_id: int) -> float:
    """Exact rotation-invariant planar half-extent of a body and its subtree.

    Unlike :func:`duck_planar_radius` this does NOT inflate a geom to its
    bounding sphere, so it is the honest width to use when the question is
    whether two bodies could have squeezed past each other side by side.
    Over-stating either body there would make the corridor look narrower than it
    is, which is the one direction in which conservatism would flatter this
    behavior instead of testing it.
    """
    center = np.asarray(data.xpos[body_id][:2], dtype=np.float64)
    radius = 0.0
    for geom in geoms_of(model, body_subtree(model, body_id)):
        radius = max(radius, _geom_planar_support(model, data, geom, center))
    return radius


def exact_lateral_half_width(model: mujoco.MjModel, data: mujoco.MjData,
                             body_id: int) -> float:
    """Largest |y| offset of a body's surface from its own origin.

    THIS, not the rotation-invariant radius, is the width that matters when two
    bodies pass in a corridor without either of them turning — which is exactly
    what happens here, since the duck cannot turn in place and the adults walk
    straight through.  A body that is long in x and narrow in y has a
    rotation-invariant radius much larger than its lateral half-width, and using
    the larger number would make the corridor look narrower than it is.  That is
    the one direction in which conservatism would flatter this behavior rather
    than test it, so the passing claim is graded on this.
    """
    center_y = float(data.xpos[body_id][1])
    half = 0.0
    for geom in geoms_of(model, body_subtree(model, body_id)):
        kind = int(model.geom_type[geom])
        pos = np.asarray(data.geom_xpos[geom], dtype=np.float64)
        rotation = np.asarray(
            data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
        size = np.asarray(model.geom_size[geom], dtype=np.float64)
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[geom])
            first = int(model.mesh_vertadr[mesh])
            count = int(model.mesh_vertnum[mesh])
            local = np.asarray(
                model.mesh_vert[first:first + count], dtype=np.float64)
            world = local @ rotation.T + pos
            half = max(half, float(np.max(np.abs(world[:, 1] - center_y))))
        elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            half = max(half, abs(float(pos[1]) - center_y) + float(size[0]))
        elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                      int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            axis = rotation[:, 2] * float(size[1])
            ends = np.stack((pos + axis, pos - axis))
            half = max(half, float(np.max(np.abs(ends[:, 1] - center_y)))
                       + float(size[0]))
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            signs = np.array([[sx, sy, sz]
                              for sx in (-1.0, 1.0)
                              for sy in (-1.0, 1.0)
                              for sz in (-1.0, 1.0)])
            corners = (signs * size) @ rotation.T + pos
            half = max(half, float(np.max(np.abs(corners[:, 1] - center_y))))
        else:
            half = max(half, abs(float(pos[1]) - center_y)
                       + float(model.geom_rbound[geom]))
    return half


class WallProbe:
    """Exact surface distance between the duck and the corridor's own geometry.

    The walls are non-colliding scenery — deliberately, so that staying inside
    the corridor is a property of the CONTROLLER rather than of MuJoCo's
    contact solver.  MuJoCo therefore reports no contact however hard the duck
    presses into a wall, and the honest gate is again a geometric one.

    Every wall, cheek, crate and door geom is a box, so all of them go through
    :func:`box_sphere_distance` exactly.  ``mj_geomDistance`` is not called: the
    mesh-versus-primitive narrowphase in this simulator has been measured
    returning exact zeros for pairs more than a metre apart (see the module
    docstring), and a corridor is precisely a scene full of primitives near a
    mesh robot.
    """

    def __init__(self, model: mujoco.MjModel, trunk_id: int,
                 geom_names: tuple[str, ...]):
        self.model = model
        self.duck_geoms = geoms_of(model, body_subtree(model, trunk_id))
        self.duck_rbound = {
            g: float(model.geom_rbound[g]) for g in self.duck_geoms}
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        self.wall_geoms: list[int] = []
        for name in geom_names:
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom < 0:
                raise RuntimeError(f"wall geom {name!r} not found in the model")
            if int(model.geom_type[geom]) == mesh_type:
                raise RuntimeError(
                    f"wall geom {name!r} is a mesh; this probe is only valid "
                    "for primitive scenery")
            self.wall_geoms.append(geom)
        self.geom_names = tuple(geom_names)

    def distance(self, data: mujoco.MjData, cutoff: float = 1.0
                 ) -> tuple[float, str]:
        """Smallest surface separation between the duck and any wall geom.

        Negative means real geometric overlap.  Values beyond ``cutoff`` are
        reported as ``cutoff``, because the exact number only matters near
        contact.  Returns the distance and the name of the limiting geom, so a
        failure names the wall it happened against.
        """
        best = cutoff
        limiting = ""
        for index, wall_geom in enumerate(self.wall_geoms):
            for duck_geom in self.duck_geoms:
                distance = primitive_sphere_distance(
                    self.model, data, wall_geom,
                    data.geom_xpos[duck_geom], self.duck_rbound[duck_geom])
                if distance < best:
                    best = distance
                    limiting = self.geom_names[index]
        return best, limiting


class ContactProbe:
    """Exact surface-to-surface distance between the duck and each adult.

    EVERY pedestrian geom is handled analytically.  ``mj_geomDistance`` is not
    called at all on this scene (see the module docstring).  The duck side is
    approximated by each robot geom's bounding sphere, which is conservative.
    """

    def __init__(self, model: mujoco.MjModel, trunk_id: int,
                 person_names: tuple[str, ...]):
        self.model = model
        self.duck_geoms = geoms_of(model, body_subtree(model, trunk_id))
        self.person_geoms: dict[str, list[int]] = {}
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        for name in person_names:
            geoms = geoms_of(
                model, body_subtree(model, model.body(f"person_{name}").id))
            meshes = [g for g in geoms if int(model.geom_type[g]) == mesh_type]
            if meshes:
                raise RuntimeError(
                    f"person {name!r} carries mesh geoms {meshes}; this probe "
                    "is only valid for all-primitive bodies")
            self.person_geoms[name] = geoms
        self.duck_rbound = {
            g: float(model.geom_rbound[g]) for g in self.duck_geoms}

    def distance(self, data: mujoco.MjData, name: str,
                 cutoff: float = 1.5) -> float:
        """Smallest surface separation between the duck and one adult.

        Negative means real geometric overlap.  Values beyond ``cutoff`` are
        reported as ``cutoff``: the exact number only matters near contact.
        """
        best = cutoff
        for person_geom in self.person_geoms[name]:
            for duck_geom in self.duck_geoms:
                distance = primitive_sphere_distance(
                    self.model, data, person_geom,
                    data.geom_xpos[duck_geom], self.duck_rbound[duck_geom])
                if distance < best:
                    best = distance
        return best
