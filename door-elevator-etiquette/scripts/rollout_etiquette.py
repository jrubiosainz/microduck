#!/usr/bin/env python3
"""Integration: doors + actors + sense + machine + controller + camera + physics.

The only module that owns all seven concerns at once, and it owns them in a
strict order every control tick:

1. the doors and the scripted people are posed analytically at ``t``;
2. the duck's own measurements are built from that world - BEFORE this tick's
   physics;
3. the machine is advanced on those measurements alone;
4. the tracker is clamped to the leg the new state authorises;
5. the controller emits a command from the state, the clamped target and the
   INDEPENDENT aperture interlock;
6. the walking policy consumes that command and physics is stepped;
7. the world is re-posed at the display time and the camera measures what it
   actually sees, from the same camera a PiP would render from.

Steps 5 and 6 are the only ones that touch locomotion.  The camera work in step 7
happens in an isolated ``MjData`` inside :class:`EtiquetteCamera` and is never
written back, so gaze cannot prop the robot up.

ORDERING NOTE, AND IT IS THE SUBTLE ONE: the machine decides on measurements
taken BEFORE the physics step, never after.  Measuring somebody's position and
then acting on that measurement within the same tick would let a decision be
authorised by a world state that only exists after the decision was made.  One
control tick at 50 Hz is 20 ms, which is honest and is also what a real
perception pipeline incurs.

THE ZONE BOOKKEEPING IS DONE HERE BECAUSE IT NEEDS THE POST-STEP POSE
-----------------------------------------------------------------------
Whether the duck encroached on a threshold is a question about where its body
ACTUALLY ENDED UP, not about where it intended to go.  So the zone depths, the
aperture occupancy and the crossing records are all taken after ``mj_step``,
against the real trunk position, with the duck's conservative planar radius.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np

from contact_geometry import (
    ContactProbe,
    WallProbe,
    duck_planar_radius,
    exact_lateral_half_width,
    exact_planar_radius,
)
from etiquette_actors import ROUTES, people_at, pose_people
from etiquette_aim import bearing_to, select
from etiquette_camera import EtiquetteCamera
from etiquette_cast import ALL_NAMES, GUARDIAN
from etiquette_control import EtiquetteController, Interlock
from etiquette_machine import EtiquetteMachine
from etiquette_markers import pose_leaves, pose_markers, route_polyline
from etiquette_path import (
    LEG_NAMES,
    STATE_LEG,
    build_route,
    careful_bands,
    door_hold_xy,
    in_careful_band,
    leg_bounds,
)
from etiquette_record import build_record
from etiquette_sense import (
    APERTURE_BOX,
    bodies_in_aperture,
    build_interlock,
    build_sense,
    guardian_arc_on_duck_route,
    los_blocked_by,
)
from etiquette_states import (
    DUCK_START_XY,
    DUCK_START_YAW_DEG,
    MONITOR_STATES,
    ZERO_COMMAND_STATES,
)
from etiquette_tally import RolloutTally
from etiquette_tracker import RouteTracker
from etiquette_zones import (
    CABIN_HOLD_XY,
    DOOR_APERTURE,
    DOOR_THRESHOLD,
    LIFT_APERTURE,
    LIFT_PASSAGE,
    LIFT_THRESHOLD,
    REAR_APERTURE,
    WAIT_SIDE_XY,
    cabin_margin_m,
)
from lobby_doors import APERTURE_NAMES, doors_at
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene

# Every scenery geom the clearance gate is measured against, collected from the
# scene's own naming rather than hand-listed, so a scene change cannot silently
# drop a surface from the gate.  ``leaf_`` is included because the sliding door
# panels are real bodies the duck must not walk into.
SCENERY_PREFIXES = ("wall_", "cabin_wall_", "obs_", "leaf_")

# The zones whose encroachment is tracked every tick, and the state from which
# each one is RELEASED.  Before that state the duck has no business being in the
# zone; from it onwards, being there is the behavior working.
ZONE_RELEASE: dict[str, str] = {
    "concourse_door_threshold": "FOLLOW_THROUGH",
    "lift_front_threshold": "FOLLOW_GUARDIAN_IN",
    "lift_front_passage": "FOLLOW_GUARDIAN_IN",
}
TRACKED_ZONES = {
    "concourse_door_threshold": DOOR_THRESHOLD,
    "lift_front_threshold": LIFT_THRESHOLD,
    "lift_front_passage": LIFT_PASSAGE,
}
# The order the states run in, used only to decide whether a zone has been
# released yet.  Reading it from the declared STATES tuple keeps it in step.
from etiquette_states import STATES  # noqa: E402

_STATE_INDEX = {name: index for index, name in enumerate(STATES)}


def scenery_geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = [
        name for geom in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
        and name.startswith(SCENERY_PREFIXES)
    ]
    if not names:
        raise RuntimeError("no scenery geoms found; the clearance gate is vacuous")
    return tuple(names)


class EtiquetteRollout:
    """One deterministic door-and-lift rollout, with or without rendering."""

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

        # -- the route, built once from geometry the duck could measure ----
        self.route = build_route()
        self.leg_bounds = leg_bounds(self.route)
        self.careful_bands = careful_bands(self.route)
        self.tracker = RouteTracker(self.route)
        self.tracker.set_leg_end(self.leg_bounds[0])
        self._route_points = route_polyline(self.route)
        self._waypoints = [np.asarray(c, dtype=np.float64)
                           for c in self.route.corners]

        pose_people(self.model, self.data, people_at(0.0), 0.0)
        pose_leaves(self.model, self.data, doors_at(0.0))
        mujoco.mj_forward(self.model, self.data)
        self.duck_radius = duck_planar_radius(self.model, self.data, self.trunk)
        self.duck_exact_radius = exact_planar_radius(
            self.model, self.data, self.trunk)
        self.duck_lateral_half = exact_lateral_half_width(
            self.model, self.data, self.trunk)
        # POSE-ZERO SAMPLE, not a gait maximum: measured once, here, with the
        # people posed at t=0 and their arms down.  Reported for context only.
        # No gate consumes it; clearance is measured every tick by ContactProbe
        # against the real geoms at the real pose.
        self.adult_lateral_half = max(
            exact_lateral_half_width(
                self.model, self.data, self.model.body(f"person_{name}").id)
            for name in ALL_NAMES)

        self.contacts = ContactProbe(self.model, self.trunk, ALL_NAMES)
        self.scenery_geoms = scenery_geom_names(self.model)
        self.scenery = WallProbe(self.model, self.trunk, self.scenery_geoms)

        self.camera = EtiquetteCamera(self.model, self.data,
                                      self.runner.qpos_idx, self.trunk,
                                      pip_size, CTRL_HZ)
        self.machine = EtiquetteMachine(ctrl_hz=CTRL_HZ)
        self.machine.set_guardian(GUARDIAN.name)
        self.controller = EtiquetteController(ctrl_hz=CTRL_HZ)

        self.records: list[dict] = []
        self.tally = RolloutTally(self.dt, float(self.data.xpos[self.trunk][2]))
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()
        self._previous_people = people_at(0.0)
        self._previous_doors = doors_at(0.0)
        self._camera_state = self.camera.update(
            self.data, duck_yaw=self.runner.yaw(self.data),
            subject=GUARDIAN.name)

    # -- helpers -----------------------------------------------------------
    def _leg_end_for(self, state: str) -> float:
        """The arc length the current state authorises walking to.

        A state with no leg keeps the tracker where it is, which is what makes
        every holding point structural: there is no target beyond it.
        """
        leg = STATE_LEG.get(state)
        if leg is None:
            return self.tracker.arc_s
        return self.leg_bounds[leg]

    def _zone_released(self, zone: str, state: str) -> bool:
        release = ZONE_RELEASE.get(zone)
        if release is None:
            return True
        return _STATE_INDEX.get(state, -1) >= _STATE_INDEX[release]

    # -- one control tick -------------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        pos_before = self.data.xpos[self.trunk].copy()
        duck_xy_before = np.array([float(pos_before[0]), float(pos_before[1])])
        duck_yaw = self.runner.yaw(self.data)

        # -- measurements taken BEFORE this tick's physics -----------------
        people = self._previous_people
        doors = self._previous_doors
        state_before = self.machine.state
        self.tracker.set_leg_end(self._leg_end_for(state_before))
        self.tracker.project(duck_xy_before)

        sense = build_sense(
            t=t, duck_xy=duck_xy_before, route=self.route,
            arc_s=self.tracker.arc_s,
            leg_end_m=self._leg_end_for(state_before),
            people=people, doors=doors)

        state, changed = self.machine.update(t, sense)

        # -- the target, clamped to the leg the NEW state authorises --------
        self.tracker.set_leg_end(self._leg_end_for(state))
        aim = select(state, duck_xy=duck_xy_before, tracker=self.tracker,
                     people=people)
        target_xy, target_kind = aim.target_xy, aim.kind
        subject = aim.subject

        # -- the INDEPENDENT refusal ----------------------------------------
        interlock = build_interlock(
            duck_xy=duck_xy_before, people=people, doors=doors,
            route=self.route, arc_s=self.tracker.arc_s)
        self.tally.note_interlock(interlock.blocked, interlock.reason)

        careful = (in_careful_band(self.careful_bands, self.tracker.arc_s)
                   and state in STATE_LEG)
        command = self.controller.update(
            state, duck_xy_before, duck_yaw, target_xy=target_xy,
            remaining_m=self.tracker.remaining_m, careful=careful,
            interlock=interlock)

        # THE EXACT-ZERO CLAIM, CHECKED AS IT IS MADE.  Recording a violation
        # here rather than only in the metrics means the trace names the tick.
        if state in ZERO_COMMAND_STATES \
                and float(np.max(np.abs(command))) != 0.0:
            self.tally.note_zero_violation(t, state, command)

        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        # -- the world at the display time ----------------------------------
        duck_pos = self.data.xpos[self.trunk].copy()
        duck_xy = np.array([float(duck_pos[0]), float(duck_pos[1])])
        duck_yaw_after = self.runner.yaw(self.data)

        display_t = min(t + self.dt, self.seconds)
        people_now = people_at(display_t)
        doors_now = doors_at(display_t)
        pose_people(self.model, self.data, people_now, display_t)
        pose_leaves(self.model, self.data, doors_now)

        travelled = float(np.linalg.norm(duck_xy - self._previous_xy))
        self.tally.note_pose(float(duck_pos[2]), travelled)

        pose_markers(self.model, self.data,
                     route_points=self._route_points,
                     waypoints=self._waypoints,
                     hold_xy=(door_hold_xy()
                              if state in ("APPROACH_DOOR", "YIELD_EXITERS")
                              else (CABIN_HOLD_XY
                                    if state in ("FOLLOW_GUARDIAN_IN",
                                                 "POSITION_INSIDE", "RIDE")
                                    else None)),
                     wait_xy=(WAIT_SIDE_XY
                              if state in ("APPROACH_LIFT", "WAIT_SIDE",
                                           "DOORS_OPEN",
                                           "LET_OCCUPANTS_EXIT")
                              else None),
                     records=self.records)
        mujoco.mj_forward(self.model, self.data)

        # -- the camera, in its isolated copy --------------------------------
        camera_state = self.camera.update(
            self.data, duck_yaw=duck_yaw_after, subject=subject,
            look_at=aim.look_at)
        self._camera_state = camera_state

        entry = camera_state["people"][subject]
        subject_visible = bool(entry["visible"])
        blocker = ("" if subject_visible
                   else self.camera.blocking_geom(subject))

        # -- safety, measured against the REAL post-step pose ----------------
        clearances = {name: self.contacts.distance(self.data, name)
                      for name in ALL_NAMES}
        nearest = min(clearances, key=clearances.get)
        scenery_gap, scenery_geom = self.scenery.distance(self.data)
        self.tally.note_clearance(clearances, nearest, scenery_gap,
                                  scenery_geom)

        # -- the zones, from where the body ACTUALLY ended up ----------------
        zone_depths = {}
        for name, band in TRACKED_ZONES.items():
            depth = band.depth_into(duck_xy, self.duck_radius)
            zone_depths[name] = depth
            self.tally.note_zone(name, depth, display_t,
                                 self._zone_released(name, state))

        aperture_occupancy = {}
        for name in APERTURE_NAMES:
            box = APERTURE_BOX[name]
            duck_in = box.contains(duck_xy, self.duck_radius)
            others = bodies_in_aperture(people_now, name, 0.0)
            aperture_occupancy[name] = {"duck": duck_in, "others": others}
            self.tally.note_aperture(name, duck_in, others)
            if duck_in:
                door = doors_now[name]
                self.tally.note_crossing(name, display_t, door.fraction,
                                         door.effective_gap_m, duck_xy)

        margin = cabin_margin_m(duck_xy)
        self.tally.note_cabin(sense.inside_cabin, margin,
                              riding=state in ("RIDE", "POSITION_INSIDE"))

        # -- order relative to the guardian ----------------------------------
        guardian_arc = guardian_arc_on_duck_route(
            self.route, people_now[GUARDIAN.name].pos)
        guardian_gap = float(guardian_arc - self.tracker.arc_s)
        # Only counted once she has set off: before that the duck is
        # legitimately ahead of somebody who is not yet walking, and the gap
        # carries no information about overtaking.
        self.tally.note_order(
            guardian_gap, counts=people_now[GUARDIAN.name].speed > 1e-6
            or ROUTES[GUARDIAN.name].arc_at(display_t) > 0.0)

        # -- per-state bookkeeping -------------------------------------------
        peak = float(np.max(np.abs(command)))
        self.tally.note_command(state, peak, travelled)
        if state in STATE_LEG:
            self.tally.note_walk(travelled, self.tracker.cross_track_m)

        # -- visibility, conditioned on line of sight ------------------------
        # LOS accounts for OTHER PEOPLE as well as static geometry.  A body
        # between the camera and the subject makes seeing them impossible in
        # exactly the way a wall does, and holding the duck responsible for it
        # graded the scenario's geometry rather than the robot: see
        # ``etiquette_sense.los_blocked_by``.
        eye_xy = self.camera.render_data.cam_xpos[self.camera.camera_id][:2]
        los_blocker = los_blocked_by(eye_xy, subject, people_now)
        los_ok = not los_blocker
        self.tally.note_visibility(
            subject=subject, visible=subject_visible, los_ok=los_ok,
            monitoring=state in MONITOR_STATES, blocker=blocker, t=display_t)

        record = build_record(
            display_t=display_t, state=state, machine=self.machine,
            command=command, duck_xy=duck_xy, duck_yaw_after=duck_yaw_after,
            duck_pos=duck_pos, min_trunk_z=self.tally.min_trunk_z,
            camera_state=camera_state, clearances=clearances, nearest=nearest,
            scenery_gap=scenery_gap, scenery_geom=scenery_geom,
            people=people_now, doors=doors_now, sense=sense,
            tracker=self.tracker, subject=subject,
            subject_visible=subject_visible, subject_blocker=blocker,
            los_available=los_ok, los_blocker=los_blocker or "",
            path_m=self.tally.path_m,
            state_elapsed=t - self.machine.state_since,
            target_xy=target_xy, target_kind=target_kind,
            interlock=interlock, zone_depths=zone_depths,
            aperture_occupancy=aperture_occupancy, cabin_margin_m=margin,
            guardian_gap_m=guardian_gap, careful=careful)
        self.records.append(record)

        self._previous_people = people_now
        self._previous_doors = doors_now
        self._previous_xy = duck_xy.copy()
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
    def actor_routes(self):
        return ROUTES

    @property
    def leg_names(self):
        return LEG_NAMES
