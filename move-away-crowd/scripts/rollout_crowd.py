#!/usr/bin/env python3
"""Integration layer: crowd routes + threat model + attention camera + policy.

This is the only module that owns all four concerns at once, and it owns them
in a strict order every control tick:

1. the crowd is posed analytically at ``t`` (scripted, non-colliding mocap);
2. the threat model ranks every adult against the duck's CURRENT position;
3. the avoidance machine advances and the evade controller emits a command;
4. the walking policy consumes that command and physics is stepped;
5. the crowd is re-posed at ``t + dt`` and the attention camera measures what
   it actually sees from the same camera the PiP renders from.

Steps 3 and 4 are the only ones that touch locomotion.  The camera work in
step 5 happens in an isolated ``MjData`` inside :class:`AttentionCamera` and is
never written back, so gaze cannot prop the robot up.

Per-cycle evidence (counterfactual clearance, realized clearance, evade path
and camera visibility) is written straight into ``machine.current``, which the
state machine copies into ``machine.cycles`` when the cycle closes.  The
metrics module then grades those cycles without re-deriving anything.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from attention_camera import AttentionCamera
from crowd_routes import ADULT_NAMES, CARRYING_BOX, crowd_at, pose_crowd
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene, wrap_angle
from threat_model import (
    AvoidanceMachine,
    EvadeController,
    escape_heading,
    predict_approach,
    rank_threats,
)

# Markers are cosmetic mocap discs; parking them below the floor hides them.
HIDDEN_Z = -0.30
EVADE_MARKER_RANGE = 0.55


def box_sphere_distance(model: mujoco.MjModel, data: mujoco.MjData,
                        box_geom: int, center: np.ndarray, radius: float) -> float:
    """Exact distance from an oriented box's surface to a sphere.

    Written by hand because ``mj_geomDistance`` is UNRELIABLE for the
    mesh-vs-box pair this scene needs (see :meth:`CrowdRollout.min_surface_distance`).
    Both primitives are trivial, so the analytic form is exact and cheap:
    transform the sphere centre into the box frame, clamp per axis, take the
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

    Reported for context only.  The contact gate uses exact geom-pair distances
    (see :meth:`CrowdRollout.min_surface_distance`), because a disc of this
    radius around the trunk is far more pessimistic than the robot's real
    silhouette and would report a collision for a clean pass.
    """
    bodies = body_subtree(model, trunk_id)
    center = data.xpos[trunk_id][:2]
    radius = 0.0
    for geom in geoms_of(model, bodies):
        offset = float(np.linalg.norm(data.geom_xpos[geom][:2] - center))
        radius = max(radius, offset + float(model.geom_rbound[geom]))
    return radius


def bearing_sector(bearing_rad: float) -> str:
    """Compass sector of an approach bearing, used to prove varied directions."""
    names = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
    index = int(round(math.degrees(bearing_rad) % 360.0 / 45.0)) % 8
    return names[index]


class CrowdRollout:
    """One deterministic move-away-crowd rollout, with or without rendering."""

    def __init__(self, policy_path: str | Path, seconds: float, *,
                 scene: str | Path | None = None,
                 robot_dir: str | Path | None = None,
                 pip_size: tuple[int, int] = (300, 220)):
        self.model = load_scene(scene, robot_dir)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.policy = PolicyRunner(policy_path)
        self.runner = self.policy.reset(self.model, self.data)
        self.trunk = self.model.body("trunk_base").id
        self.decimation = max(
            1, int(round((1.0 / CTRL_HZ) / self.model.opt.timestep))
        )
        self.seconds = float(seconds)
        self.total_steps = int(self.seconds * CTRL_HZ)
        self.dt = 1.0 / CTRL_HZ

        pose_crowd(self.model, self.data, crowd_at(0.0), 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.duck_radius = duck_planar_radius(self.model, self.data, self.trunk)
        # Exact contact geometry: every robot geom against every geom of each
        # adult (torso, head, face, legs, shoes, arms and the carried box+tape).
        # `mj_geomDistance` returns the true separation between the two
        # surfaces, so "no person/box contact" is decided on the shapes the
        # scene actually draws rather than on a bounding disc.
        self._duck_geoms = geoms_of(self.model, body_subtree(self.model, self.trunk))
        self._adult_geoms = {}
        self._adult_boxes = {}
        for name in ADULT_NAMES:
            geoms = geoms_of(
                self.model,
                body_subtree(self.model, self.model.body(f"person_{name}").id),
            )
            box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
            self._adult_boxes[name] = [
                g for g in geoms if int(self.model.geom_type[g]) == box_type
            ]
            self._adult_geoms[name] = [
                g for g in geoms if int(self.model.geom_type[g]) != box_type
            ]
        # Bounding-sphere radius of each robot geom, used for the analytic box
        # test.  Every mesh lies inside its own bounding sphere, so the result
        # is conservative: it can only UNDER-report clearance, never invent it.
        self._duck_rbound = {
            g: float(self.model.geom_rbound[g]) for g in self._duck_geoms
        }

        self.camera = AttentionCamera(
            self.model, self.data, self.runner.qpos_idx, self.trunk, pip_size
        )
        self.machine = AvoidanceMachine(ctrl_hz=CTRL_HZ)
        self.controller = EvadeController(ctrl_hz=CTRL_HZ)
        self.markers = {
            name: int(self.model.body_mocapid[self.model.body(name).id])
            for name in ("threat_marker", "predict_marker", "evade_marker")
        }
        self.records: list[dict] = []
        self.transitions: list[dict] = []
        self.min_trunk_z = float(self.data.xpos[self.trunk][2])
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()
        # Ranking observed when the machine last began confirming a candidate.
        self._ranking_snapshot: list | None = None

    # -- exact contact ---------------------------------------------------
    def min_surface_distance(self, name: str, cutoff: float = 1.2) -> float:
        """Smallest surface-to-surface distance between the duck and one adult.

        Negative means real geometric overlap.  Values beyond ``cutoff`` are
        reported as ``cutoff``: the exact number only matters near contact.

        TWO MEASURED TRAPS, both of which faked contact in earlier runs:

        1. ``mj_geomDistance`` returns the CUTOFF ITSELF, not the true
           distance, for a pair farther apart than the cutoff.  Feeding the
           running minimum back in as the next cutoff collapses the scan: once
           any pair returns ``x`` every later pair is clamped to ``x``, so a
           single ``0.0`` reports contact for the whole frame.  The cutoff is
           therefore held FIXED and the minimum accumulated separately.

        2. Even with a fixed cutoff, MuJoCo's MESH-vs-BOX narrowphase returns
           exactly ``0.0`` for pairs that are plainly apart.  Measured by
           sweeping one adult around the standing robot: 65 spurious zeros in
           264,000 samples, every one of them against ``*_box`` or
           ``*_box_tape``, at separations up to 0.85 m.  All 15 "contacts" in
           run 4 were this artifact — three different adults reported contact
           at the same duck position while the duck stood still, with true
           centre distances of 0.43-0.89 m.

           The robot is 75 mesh geoms and the carried boxes are the only boxes
           in the scene, so box pairs are handled analytically instead:
           :func:`box_sphere_distance` against each robot geom's bounding
           sphere.  That is exact for the box and conservative for the mesh, so
           it can only under-report clearance — it cannot hide a real contact.
        """
        best = cutoff
        for adult_geom in self._adult_geoms[name]:
            for duck_geom in self._duck_geoms:
                distance = float(mujoco.mj_geomDistance(
                    self.model, self.data, duck_geom, adult_geom, cutoff, None
                ))
                if distance < best:
                    best = distance
        for box_geom in self._adult_boxes[name]:
            for duck_geom in self._duck_geoms:
                distance = box_sphere_distance(
                    self.model, self.data, box_geom,
                    self.data.geom_xpos[duck_geom], self._duck_rbound[duck_geom],
                )
                if distance < best:
                    best = distance
        return best

    # -- markers ---------------------------------------------------------
    def _hide_markers(self) -> None:
        for mocap in self.markers.values():
            self.data.mocap_pos[mocap] = np.array([0.0, 0.0, HIDDEN_Z])

    def _show_markers(self, adult_xy, closest_xy, evade_xy) -> None:
        self.data.mocap_pos[self.markers["threat_marker"]] = np.array(
            [adult_xy[0], adult_xy[1], 0.010])
        self.data.mocap_pos[self.markers["predict_marker"]] = np.array(
            [closest_xy[0], closest_xy[1], 0.011])
        self.data.mocap_pos[self.markers["evade_marker"]] = np.array(
            [evade_xy[0], evade_xy[1], 0.012])

    # -- per-cycle evidence ---------------------------------------------
    def _open_evasion(self, t: float, crowd: dict, duck_xy: np.ndarray) -> None:
        """Record the counterfactual the evasion will be graded against."""
        locked = self.machine.locked
        adult = crowd[locked]
        stay = predict_approach(adult.pos, adult.vel, duck_xy, name=locked)
        current = self.machine.current
        current["evade_start_xy"] = [float(v) for v in duck_xy]
        current["counterfactual_clearance_m"] = stay.min_clearance
        current["counterfactual_ttc_s"] = stay.time_to_closest
        current["approach_sector"] = bearing_sector(
            current.get("lock_bearing_deg", 0.0) * math.pi / 180.0)
        current["evade_path_m"] = 0.0
        current["carries_box"] = locked in CARRYING_BOX

    def _accumulate(self, t: float, state: str, crowd: dict,
                    duck_xy: np.ndarray, camera_state: dict) -> None:
        """Fold this tick's evidence into the open cycle."""
        current = self.machine.current
        locked = self.machine.locked
        if not current or locked is None:
            return
        clearance = self.min_surface_distance(locked)
        current["actual_min_clearance_m"] = min(
            current.get("actual_min_clearance_m", float("inf")), clearance)
        if state in ("THREAT_LOCK", "EVADING"):
            current["lock_steps"] = current.get("lock_steps", 0) + 1
            current["lock_visible_steps"] = current.get(
                "lock_visible_steps", 0) + int(camera_state["locked_visible"])
        if state == "EVADING":
            current["evade_steps"] = current.get("evade_steps", 0) + 1
            current["evade_visible_steps"] = current.get(
                "evade_visible_steps", 0) + int(camera_state["locked_visible"])
            current["evade_path_m"] = current.get("evade_path_m", 0.0) + float(
                np.linalg.norm(duck_xy - self._previous_xy))
            current["evade_end_xy"] = [float(v) for v in duck_xy]
            start = np.asarray(current["evade_start_xy"], dtype=np.float64)
            current["evade_net_m"] = float(np.linalg.norm(duck_xy - start))
            adult = crowd[locked]
            here = predict_approach(adult.pos, adult.vel, duck_xy, name=locked)
            current["end_predicted_clearance_m"] = here.min_clearance

    # -- one control tick -----------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        crowd = crowd_at(t)
        pose_crowd(self.model, self.data, crowd, t)
        mujoco.mj_forward(self.model, self.data)

        duck_xy = self.data.xpos[self.trunk][:2].copy()
        duck_yaw = self.runner.yaw(self.data)
        ranking = rank_threats(crowd, duck_xy)
        threat = next((a for a in ranking if a.is_threat), None)

        previous_state = self.machine.state
        # The machine locks onto the candidate it has been CONFIRMING, chosen
        # LOCK_CONFIRM_S earlier.  Grading that decision against the ranking at
        # the transition tick asks whether a past choice is optimal for a future
        # it could not see: in run 1 cycle 3 the lock was the tightest threat
        # when taken, and a different adult became marginally tighter 0.3 s
        # later, which scored as a wrong lock.  Grade the decision instead.
        confirming_before = self.machine.confirming
        decision_ranking = self._ranking_snapshot
        locked_view = None
        if self.machine.state == "THREAT_LOCK" and self.machine.locked is not None:
            locked_view = {
                "surface_clearance_m": self.min_surface_distance(
                    self.machine.locked
                )
            }
        state, changed = self.machine.update(t, threat, locked_view)
        locked = self.machine.locked
        if changed and state == "THREAT_LOCK":
            graded = ranking
            at_confirm = False
            if decision_ranking is not None and confirming_before == locked:
                graded = decision_ranking
                at_confirm = True
            genuine = [a for a in graded if a.is_threat]
            tightest = min(genuine, key=lambda a: a.min_clearance).name
            self.machine.current["lock_is_threat"] = bool(
                any(a.name == locked and a.is_threat for a in graded))
            self.machine.current["lock_top_ranked"] = bool(graded[0].name == locked)
            self.machine.current["lock_tightest_clearance"] = bool(tightest == locked)
            self.machine.current["lock_graded_at_confirm"] = at_confirm
        # Snapshot the ranking a NEW confirmation was started from.
        if self.machine.confirming != confirming_before:
            self._ranking_snapshot = ranking
        if changed and state == "EVADING":
            # A fresh maneuver: let the controller re-choose forward vs backward.
            self.controller.reset()
            self._open_evasion(t, crowd, duck_xy)

        heading_error = 0.0
        if locked is not None and locked in crowd:
            adult = crowd[locked]
            approach = predict_approach(
                adult.pos, adult.vel, duck_xy, name=locked)
            heading = escape_heading(approach, duck_xy, adult.vel)
            heading_error = wrap_angle(heading - duck_yaw)
            goal = duck_xy + EVADE_MARKER_RANGE * np.array(
                [math.cos(heading), math.sin(heading)])
            self._show_markers(adult.pos, approach.closest_point, goal)
        else:
            self._hide_markers()

        command = self.controller.update(state, heading_error)
        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        display_t = min(t + self.dt, self.seconds)
        display_crowd = crowd_at(display_t)
        pose_crowd(self.model, self.data, display_crowd, display_t)
        mujoco.mj_forward(self.model, self.data)

        duck_pos = self.data.xpos[self.trunk].copy()
        duck_yaw_after = self.runner.yaw(self.data)
        self.min_trunk_z = min(self.min_trunk_z, float(duck_pos[2]))
        camera_state = self.camera.update(
            self.data, state=state, state_elapsed=t - self.machine.state_since,
            duck_yaw=duck_yaw_after, locked=locked)

        clearances = {
            name: self.min_surface_distance(name) for name in ADULT_NAMES
        }
        nearest = min(clearances, key=clearances.get)
        self._accumulate(display_t, state, display_crowd, duck_pos[:2],
                         camera_state)
        self._previous_xy = duck_pos[:2].copy()

        if changed:
            event = {
                "t": t, "from": previous_state, "to": state,
                "locked": locked,
                "cycle": self.machine.current.get(
                    "cycle", len(self.machine.cycles) + 1),
            }
            self.transitions.append(event)

        record = {
            "t": display_t,
            "state": state,
            "state_elapsed_s": t - self.machine.state_since,
            "locked": locked,
            "cycle": len(self.machine.cycles) + (1 if self.machine.current else 0),
            "threat": threat.name if threat is not None else None,
            "threat_clearance_m": (
                threat.min_clearance if threat is not None else None),
            "threat_ttc_s": threat.time_to_closest if threat is not None else None,
            "threat_bearing_deg": (
                math.degrees(threat.bearing) if threat is not None else None),
            "locked_visible": bool(camera_state["locked_visible"]),
            "locked_off_axis_deg": math.degrees(camera_state["locked_off_axis"]),
            "locked_range_m": float(camera_state["locked_range"]),
            "visible": list(camera_state["visible"]),
            "view_yaw_deg": math.degrees(camera_state["view_yaw"]),
            "gaze_yaw_deg": math.degrees(camera_state["gaze_yaw"]),
            "heading_error_deg": math.degrees(heading_error),
            "command": [float(v) for v in command],
            "duck_xy": [float(v) for v in duck_pos[:2]],
            "duck_yaw_deg": math.degrees(duck_yaw_after),
            "trunk_z_m": float(duck_pos[2]),
            "nearest_adult": nearest,
            "nearest_clearance_m": float(clearances[nearest]),
            "min_trunk_z_m": self.min_trunk_z,
        }
        self.records.append(record)
        return record

    def run(self, on_frame=None, progress=None) -> list[dict]:
        for index in range(self.total_steps):
            record = self.step(index)
            if progress is not None:
                progress(index, record)
            if on_frame is not None:
                on_frame(index, record)
        return self.records
