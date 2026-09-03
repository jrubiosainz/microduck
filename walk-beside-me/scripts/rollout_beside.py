#!/usr/bin/env python3
"""Integration: actors + camera + side choice + machine + controller + physics.

The only module that owns all six concerns at once, and it owns them in a strict
order every control tick:

1. the people are posed analytically at ``t`` (scripted, non-colliding);
2. both sides are graded from the PREVIOUS tick's measured world state;
3. the machine advances on those verdicts;
4. the controller emits a command from the state and the current target;
5. the walking policy consumes that command and physics is stepped;
6. the people are re-posed at ``t + dt`` and the camera measures what it
   actually sees, from the same camera a PiP would render from.

Steps 4 and 5 are the only ones that touch locomotion.  The camera work in step
6 happens in an isolated ``MjData`` inside :class:`BesideCamera` and is never
written back, so gaze cannot prop the robot up.

ORDERING NOTE, and it is the subtle one: the machine decides on measurements
taken BEFORE the physics step, never after.  Grading a side and then acting on
that grade within the same tick would let a decision be authorised by a world
state that only exists after the decision was made.  One control tick at 50 Hz
is 20 ms, which is honest and is also what a real perception pipeline incurs.

THE WAYPOINT CURSOR IS MONOTONIC, AND THAT IS A SCAR
------------------------------------------------------
``lost-child-find-person`` learned this the hard way: a stateless "first point
farther than the tolerance" selector starts chasing an already-passed waypoint
again as soon as its distance grows, producing an endless loop around it.  The
crossing here advances through its waypoints with a cursor that only ever moves
forward, and :meth:`_advance_cross_cursor` is the only thing permitted to move
it.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np

from beside_actors import ROUTES, people_at, pose_people
from beside_camera import BesideCamera
from beside_cast import ALL_NAMES, GUARDIAN
from beside_constants import BESIDE_STATES
from beside_control import BesideController, LEAD_TIME_S
from beside_geometry import (
    BESIDE_TARGET_M,
    CROSS_ARRIVE_M,
    DUCK_START_XY,
    DUCK_START_YAW_DEG,
    cross_point,
    formation_ok as formation_ok_fn,
    relative,
    slot_point,
)
from beside_machine import BesideMachine
from beside_record import build_record
from contact_geometry import (
    ContactProbe,
    WallProbe,
    duck_planar_radius,
    exact_planar_radius,
)
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene
from promenade_layout import occluder_between
from side_choice import evaluate_both, prefer_side, tracks_from_states

# Every scenery geom the clearance gate is measured against, collected from the
# scene's own naming rather than hand-listed, so a scene change cannot silently
# drop a surface from the gate.
SCENERY_PREFIXES = ("obs_", "wall_")


def scenery_geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = [
        name for geom in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
        and name.startswith(SCENERY_PREFIXES)
    ]
    if not names:
        raise RuntimeError("no scenery geoms found; the clearance gate is vacuous")
    return tuple(names)


class BesideRollout:
    """One deterministic walk-beside-me rollout, with or without rendering."""

    def __init__(self, policy_path: str | Path, seconds: float, *,
                 scene: str | Path | None = None,
                 robot_dir: str | Path | None = None,
                 pip_size: tuple[int, int] = (300, 216)):
        self.model = load_scene(scene, robot_dir)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qpos[1] = DUCK_START_XY
        half = math.radians(DUCK_START_YAW_DEG) * 0.5
        self.data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]

        self.policy = PolicyRunner(policy_path)
        self.policy_sha256 = hashlib.sha256(
            Path(policy_path).read_bytes()).hexdigest()
        self.runner = self.policy.reset(self.model, self.data)
        self.trunk = self.model.body("trunk_base").id
        self.decimation = max(
            1, int(round((1.0 / CTRL_HZ) / self.model.opt.timestep)))
        self.seconds = float(seconds)
        self.total_steps = int(self.seconds * CTRL_HZ)
        self.dt = 1.0 / CTRL_HZ

        pose_people(self.model, self.data, people_at(0.0), 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.duck_radius = duck_planar_radius(self.model, self.data, self.trunk)
        self.duck_exact_radius = exact_planar_radius(
            self.model, self.data, self.trunk)
        # POSE-ZERO SAMPLE, not a gait maximum: measured once, here, with the
        # people posed at t=0 and their arms down.  Reported for context only.
        # No gate consumes it; clearance is measured every tick by ContactProbe
        # against the real geoms at the real pose.
        self.adult_half_extent = exact_planar_radius(
            self.model, self.data, self.model.body(f"person_{GUARDIAN.name}").id)

        self.contacts = ContactProbe(self.model, self.trunk, ALL_NAMES)
        self.scenery_geoms = scenery_geom_names(self.model)
        self.scenery = WallProbe(self.model, self.trunk, self.scenery_geoms)

        self.camera = BesideCamera(self.model, self.data, self.runner.qpos_idx,
                                   self.trunk, pip_size, CTRL_HZ)
        self.machine = BesideMachine(ctrl_hz=CTRL_HZ)
        self.machine.set_guardian(GUARDIAN.name)
        self.controller = BesideController(ctrl_hz=CTRL_HZ)

        self.records: list[dict] = []
        self.min_trunk_z = float(self.data.xpos[self.trunk][2])
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()
        self.path_m = 0.0
        self.fallen_steps = 0
        self.contact_steps = 0
        self.min_person_clearance = float("inf")
        self.min_person_name = ""
        self.min_scenery_clearance = float("inf")
        self.min_scenery_geom = ""

        # Per-state accumulators.
        self.state_steps: dict[str, int] = {}
        self.state_command_max: dict[str, float] = {}
        # Formation bookkeeping.
        self.beside_steps = 0
        self.beside_path_m = 0.0
        self.beside_lateral: list[float] = []
        self.beside_longitudinal: list[float] = []
        self.beside_side_steps: dict[str, int] = {}
        self.formation_steps = 0
        self.max_forward_longitudinal = -float("inf")
        self.max_forward_during_switch = -float("inf")
        self.min_guardian_clearance = float("inf")
        # Visibility, conditioned on line of sight existing at all.
        self.visible_steps = 0
        self.los_steps = 0
        self.visible_with_los = 0
        self.blocked_by: dict[str, int] = {}
        # Per-switch path bookkeeping.
        self.switch_path: dict[int, float] = {}
        self.switch_start_xy: dict[int, np.ndarray] = {}
        self.switch_end_xy: dict[int, np.ndarray] = {}
        self.switch_lateral_start: dict[int, float] = {}
        self.switch_lateral_end: dict[int, float] = {}
        self.switch_min_longitudinal: dict[int, float] = {}
        self.switch_max_longitudinal: dict[int, float] = {}
        self.switch_min_clearance: dict[int, float] = {}
        # The crossing waypoint cursor: monotonic, see the module docstring.
        self._cross_cursor = 0
        self._cross_waypoints: list[np.ndarray] = []

        self._previous_people = people_at(0.0)
        self._camera_state = self.camera.update(
            self.data, duck_yaw=self.runner.yaw(self.data),
            subject=GUARDIAN.name)

    # -- probes ----------------------------------------------------------
    def person_distance(self, name: str, cutoff: float = 1.5) -> float:
        return self.contacts.distance(self.data, name, cutoff)

    def scenery_distance(self, cutoff: float = 1.0) -> tuple[float, str]:
        return self.scenery.distance(self.data, cutoff)

    # -- markers ---------------------------------------------------------
    def _place(self, body_name: str, xy, z: float) -> None:
        body = self.model.body(body_name)
        mocap = int(self.model.body_mocapid[body.id])
        if xy is None:
            self.data.mocap_pos[mocap] = (0.0, 0.0, -3.0)
            return
        self.data.mocap_pos[mocap] = (float(xy[0]), float(xy[1]), z)

    def _pose_markers(self, guardian, verdicts, target_xy) -> None:
        """Show both candidate slots, the one in use, and the crossing path."""
        left = slot_point(guardian.pos, guardian.yaw, 1)
        right = slot_point(guardian.pos, guardian.yaw, -1)
        self._place("slot_left", left, 0.009)
        self._place("slot_right", right, 0.009)
        taken = (slot_point(guardian.pos, guardian.yaw, self.machine.side)
                 if self.machine.side is not None else None)
        self._place("slot_taken", taken, 0.011)
        blocked = next((slot_point(guardian.pos, guardian.yaw, side)
                        for side in (1, -1) if not verdicts[side].usable), None)
        self._place("blocked_marker", blocked, 0.010)
        for index in range(4):
            point = (self._cross_waypoints[index]
                     if index < len(self._cross_waypoints) else None)
            self._place(f"cross_{index}", point, 0.008)
        trail = [r["guardian_xy"] for r in self.records[::-25]][:12]
        for index in range(12):
            self._place(f"trail_{index}",
                        trail[index] if index < len(trail) else None, 0.006)

    # -- crossing waypoints ----------------------------------------------
    def _plan_cross(self, guardian, target_side: int) -> None:
        """Two rear waypoints: astern of her, then astern on the far side.

        Planned when FALL_BACK is entered, in her frame at that instant, and
        then re-anchored to her live pose each tick — she keeps walking during
        the manoeuvre, so a world-frozen waypoint would be behind where she WAS
        rather than behind where she IS.
        """
        self._cross_cursor = 0
        self._cross_offsets = [(0.0, -0.86), (target_side * 0.55, -0.80)]

    def _cross_targets(self, guardian) -> list[np.ndarray]:
        """The crossing waypoints, re-anchored to her live pose."""
        forward = np.array([math.cos(guardian.yaw), math.sin(guardian.yaw)])
        left = np.array([-math.sin(guardian.yaw), math.cos(guardian.yaw)])
        return [np.asarray(guardian.pos, dtype=np.float64)
                + left * lat + forward * lon
                for lat, lon in getattr(self, "_cross_offsets", [])]

    def _advance_cross_cursor(self, duck_xy) -> None:
        """Move the MONOTONIC cursor forward.  The only thing allowed to.

        A stateless nearest-unreached selector re-targets a waypoint the duck has
        already passed as soon as the duck's distance to it grows again, which is
        how ``lost-child-find-person`` produced an endless loop around a corner.
        The cursor therefore never decreases.
        """
        while (self._cross_cursor < len(self._cross_waypoints) - 1
               and float(np.linalg.norm(
                   self._cross_waypoints[self._cross_cursor] - duck_xy))
               <= CROSS_ARRIVE_M):
            self._cross_cursor += 1

    # -- one control tick -------------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        pos_before = self.data.xpos[self.trunk].copy()
        duck_xy_before = np.array([float(pos_before[0]), float(pos_before[1])])

        guardian = self._previous_people[GUARDIAN.name]
        lateral, longitudinal = relative(
            duck_xy_before, guardian.pos, guardian.yaw)

        # -- grade both sides on the PREVIOUS tick's measured world --------
        tracks = tracks_from_states(self._previous_people, GUARDIAN.name)
        verdicts = evaluate_both(
            guardian.pos, guardian.yaw, guardian.velocity, tracks)
        preferred, preference_reason = prefer_side(verdicts, self.machine.side)

        formation = (self.machine.side is not None
                     and formation_ok_fn(lateral, longitudinal,
                                         self.machine.side))

        state_before = self.machine.state
        state, changed = self.machine.update(
            t, formation_ok=formation, lateral=lateral,
            longitudinal=longitudinal, verdicts=verdicts, preferred=preferred,
            preference_reason=preference_reason)

        if changed and state == "FALL_BACK":
            self._plan_cross(guardian, self.machine.target_side)
            self.controller.reset()
            cycle = len(self.machine.switches)
            self.switch_start_xy[cycle] = duck_xy_before.copy()
            self.switch_lateral_start[cycle] = lateral
        if changed and state_before in ("CROSS_BEHIND", "JOIN_OTHER_SIDE") \
                and state in BESIDE_STATES:
            self._cross_waypoints = []

        # -- target selection ---------------------------------------------
        self._cross_waypoints = (
            self._cross_targets(guardian)
            if state in ("FALL_BACK", "CROSS_BEHIND") else [])
        target_xy = None
        target_kind = ""
        urgent = False
        settle = False
        if state == "ACQUIRE":
            # Aim at the slot the chooser will pick, or at a point astern of her
            # while neither side is usable, so the duck closes rather than
            # loitering out of formation.
            side = preferred if preferred is not None else -1
            target_xy = slot_point(guardian.pos, guardian.yaw, side)
            target_kind = "acquire_slot"
        elif state in ("JOIN_SIDE", "JOIN_OTHER_SIDE"):
            target_xy = slot_point(guardian.pos, guardian.yaw, self.machine.side)
            target_kind = "join_slot"
        elif state in BESIDE_STATES or state == "SIDE_BLOCKED":
            # Pursue where the slot WILL be, not where it is.  On the outside of
            # a bend the slot moves faster than she does, and a controller
            # driving at its present position is permanently late through every
            # corner — which is exactly where the formation is most visible.
            lead_xy = (np.asarray(guardian.pos, dtype=np.float64)
                       + guardian.velocity * LEAD_TIME_S)
            target_xy = slot_point(lead_xy, guardian.yaw, self.machine.side)
            target_kind = "beside_lead"
        elif state == "FALL_BACK":
            target_xy = cross_point(guardian.pos, guardian.yaw,
                                    self.machine.side or 1)
            target_kind = "fall_back_point"
        elif state == "CROSS_BEHIND":
            self._advance_cross_cursor(duck_xy_before)
            if self._cross_waypoints:
                target_xy = self._cross_waypoints[
                    min(self._cross_cursor, len(self._cross_waypoints) - 1)]
            else:
                target_xy = cross_point(guardian.pos, guardian.yaw,
                                        self.machine.target_side or 1)
            target_kind = "cross_waypoint"
            urgent = True

        duck_yaw = self.runner.yaw(self.data)
        command = self.controller.update(
            state, duck_xy_before, duck_yaw, target_xy=target_xy,
            longitudinal=longitudinal, urgent=urgent, settle=settle)
        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        # -- re-pose the world and measure --------------------------------
        display_t = min(t + self.dt, self.seconds)
        people = people_at(display_t)
        pose_people(self.model, self.data, people, display_t)

        duck_pos = self.data.xpos[self.trunk].copy()
        duck_xy = np.array([float(duck_pos[0]), float(duck_pos[1])])
        duck_yaw_after = self.runner.yaw(self.data)
        self.min_trunk_z = min(self.min_trunk_z, float(duck_pos[2]))
        if float(duck_pos[2]) < 0.09:
            self.fallen_steps += 1
        travelled = float(np.linalg.norm(duck_xy - self._previous_xy))
        self.path_m += travelled

        self._pose_markers(people[GUARDIAN.name], verdicts, target_xy)
        mujoco.mj_forward(self.model, self.data)

        camera_state = self.camera.update(
            self.data, duck_yaw=duck_yaw_after, subject=GUARDIAN.name)
        self._camera_state = camera_state

        entry = camera_state["people"][GUARDIAN.name]
        now_visible = bool(entry["visible"])
        blocker = "" if now_visible else self.camera.blocking_geom(GUARDIAN.name)

        # -- safety --------------------------------------------------------
        clearances = {name: self.person_distance(name) for name in ALL_NAMES}
        nearest = min(clearances, key=clearances.get)
        if clearances[nearest] < self.min_person_clearance:
            self.min_person_clearance = clearances[nearest]
            self.min_person_name = nearest
        if clearances[nearest] <= 0.0:
            self.contact_steps += 1
        self.min_guardian_clearance = min(
            self.min_guardian_clearance, clearances[GUARDIAN.name])
        scenery_gap, scenery_geom = self.scenery_distance()
        if scenery_gap < self.min_scenery_clearance:
            self.min_scenery_clearance = scenery_gap
            self.min_scenery_geom = scenery_geom

        # -- formation bookkeeping, measured AFTER the step ----------------
        guardian_now = people[GUARDIAN.name]
        lateral_after, longitudinal_after = relative(
            duck_xy, guardian_now.pos, guardian_now.yaw)
        guardian_range = float(np.linalg.norm(guardian_now.pos - duck_xy))
        formation_after = (
            self.machine.side is not None
            and formation_ok_fn(lateral_after, longitudinal_after,
                                self.machine.side))

        peak = float(np.max(np.abs(command)))
        self.state_command_max[state] = max(
            self.state_command_max.get(state, 0.0), peak)
        self.state_steps[state] = self.state_steps.get(state, 0) + 1

        if state in BESIDE_STATES:
            self.beside_steps += 1
            self.beside_path_m += travelled
            self.beside_lateral.append(abs(lateral_after))
            self.beside_longitudinal.append(longitudinal_after)
            self.beside_side_steps[state] = (
                self.beside_side_steps.get(state, 0) + 1)
        if formation_after:
            self.formation_steps += 1

        # The forward half-plane claim: measured EVERY tick, not only during a
        # crossing, because cutting in front is the worst thing this behavior
        # could do at any moment.
        self.max_forward_longitudinal = max(
            self.max_forward_longitudinal, longitudinal_after)
        if state in ("SIDE_BLOCKED", "FALL_BACK", "CROSS_BEHIND",
                     "JOIN_OTHER_SIDE"):
            self.max_forward_during_switch = max(
                self.max_forward_during_switch, longitudinal_after)
            cycle = len(self.machine.switches)
            self.switch_path[cycle] = self.switch_path.get(cycle, 0.0) + travelled
            self.switch_end_xy[cycle] = duck_xy.copy()
            self.switch_lateral_end[cycle] = lateral_after
            self.switch_min_longitudinal[cycle] = min(
                self.switch_min_longitudinal.get(cycle, float("inf")),
                longitudinal_after)
            self.switch_max_longitudinal[cycle] = max(
                self.switch_max_longitudinal.get(cycle, -float("inf")),
                longitudinal_after)
            self.switch_min_clearance[cycle] = min(
                self.switch_min_clearance.get(cycle, float("inf")),
                clearances[GUARDIAN.name])

        # -- visibility, conditioned on line of sight ----------------------
        eye_xy = self.camera.render_data.cam_xpos[self.camera.camera_id][:2]
        los_blocker = occluder_between(eye_xy, guardian_now.pos)
        los_ok = los_blocker is None
        if now_visible:
            self.visible_steps += 1
        if los_ok:
            self.los_steps += 1
            if now_visible:
                self.visible_with_los += 1
        if not now_visible:
            key = blocker or "out_of_frustum"
            self.blocked_by[key] = self.blocked_by.get(key, 0) + 1

        self._previous_people = people
        self._previous_xy = duck_xy.copy()

        record = build_record(
            display_t=display_t, state=state, machine=self.machine,
            command=command, duck_xy=duck_xy, duck_yaw_after=duck_yaw_after,
            duck_pos=duck_pos, min_trunk_z=self.min_trunk_z,
            camera_state=camera_state, clearances=clearances, nearest=nearest,
            scenery_gap=scenery_gap, scenery_geom=scenery_geom, people=people,
            guardian_range=guardian_range, guardian_visible=now_visible,
            guardian_blocker=blocker, lateral=lateral_after,
            longitudinal=longitudinal_after, verdicts=verdicts,
            preferred=preferred, preference_reason=preference_reason,
            target_xy=target_xy, target_kind=target_kind,
            los_available=los_ok, los_blocker=los_blocker or "",
            path_m=self.path_m, state_elapsed=t - self.machine.state_since,
            formation_ok=formation_after)
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

    @property
    def guardian_route(self):
        return ROUTES[GUARDIAN.name]
