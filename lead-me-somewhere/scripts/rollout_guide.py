#!/usr/bin/env python3
"""Integration: request + planner + actors + follower + camera + machine +
controller + physics.

The only module that owns all eight concerns at once, and it owns them in a
strict order every control tick:

1. the scripted actors are posed analytically at ``t``;
2. the machine is advanced on measurements taken BEFORE this tick's physics;
3. the controller emits a command from the state and the current target;
4. the walking policy consumes that command and physics is stepped;
5. the duck's new position extends the follower's trail, and she advances along
   it;
6. everybody is re-posed and the camera measures what it actually sees, from the
   same camera a PiP would render from.

Steps 3 and 4 are the only ones that touch locomotion.  The camera work in step
6 happens in an isolated ``MjData`` inside :class:`GuideCamera` and is never
written back, so neither gaze nor the arrival gesture can prop the robot up.

ORDERING NOTE, and it is the subtle one: the machine decides on measurements
taken BEFORE the physics step, never after.  Measuring the follower and then
acting on that measurement within the same tick would let a decision be
authorised by a world state that only exists after the decision was made.  One
control tick at 50 Hz is 20 ms, which is honest and is also what a real
perception pipeline incurs.

THE FOLLOWER WALKS THE DUCK'S OWN TRAIL, AND THAT IS WHAT MAKES THIS
FALSIFIABLE.  See ``guide_follower``: she can only occupy path the duck has
already laid down, clamped to stay a fixed arc length behind, so "the duck led
her" is structural rather than asserted, and "waiting worked" is a measurement
of her closing the gap while the duck's command was exactly zero.
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
    exact_planar_radius,
)
from guide_actors import ROUTES, people_at, pose_people
from guide_camera import GuideCamera
from guide_cast import ALL_NAMES, FOLLOWER
from guide_control import GuideController
from guide_follower import Follower
from guide_layout import (
    DESTINATION_KEYS,
    DESTINATIONS,
    occluder_between,
)
from guide_machine import GuideMachine
from guide_planner import Planner, tubes_from_states
from guide_record import build_record
from guide_aim import (
    bearing_to,
    facing_error_deg,
    reached_standing_point,
    select,
)
from guide_markers import ROUTE_DISCS, pose_markers
from guide_tally import RolloutTally
from guide_states import (
    ARRIVE_RADIUS_M,
    CATCHUP_DISTANCE_M,
    CHECK_ARC_TARGET_DEG,
    DUCK_START_XY,
    DUCK_START_YAW_DEG,
    FACE_TOLERANCE_DEG,
    FOLLOWER_START_XY,
    LAG_DISTANCE_M,
    LOOK_BACK_TOLERANCE_DEG,
    MONITOR_STATES,
    REQUESTED_DESTINATION,
    REQUEST_T_S,
    SAFETY_MAX_DISTANCE_M,
)
from guide_tracker import RouteTracker
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene, wrap_angle

# Every scenery geom the clearance gate is measured against, collected from the
# scene's own naming rather than hand-listed, so a scene change cannot silently
# drop a surface from the gate.  ``dest_`` is included because the destination
# pylons are real bodies the duck must not walk into on its way to standing in
# front of one.
SCENERY_PREFIXES = ("obs_", "wall_", "dest_")


def scenery_geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = [
        name for geom in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
        and name.startswith(SCENERY_PREFIXES)
    ]
    if not names:
        raise RuntimeError("no scenery geoms found; the clearance gate is vacuous")
    return tuple(names)


class GuideRollout:
    """One deterministic lead-me-somewhere rollout, with or without rendering."""

    def __init__(self, policy_path: str | Path, seconds: float, *,
                 scene: str | Path | None = None,
                 robot_dir: str | Path | None = None,
                 pip_size: tuple[int, int] = (300, 216),
                 requested: str = REQUESTED_DESTINATION):
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
        self.requested = requested

        self.follower = Follower(FOLLOWER_START_XY, DUCK_START_XY)
        pose_people(self.model, self.data, people_at(0.0, self.follower), 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.duck_radius = duck_planar_radius(self.model, self.data, self.trunk)
        self.duck_exact_radius = exact_planar_radius(
            self.model, self.data, self.trunk)
        # POSE-ZERO SAMPLE, not a gait maximum: measured once, here, with the
        # people posed at t=0 and their arms down.  Reported for context only.
        # No gate consumes it; clearance is measured every tick by ContactProbe
        # against the real geoms at the real pose.
        self.adult_half_extent = exact_planar_radius(
            self.model, self.data, self.model.body(f"person_{FOLLOWER.name}").id)

        self.contacts = ContactProbe(self.model, self.trunk, ALL_NAMES)
        self.scenery_geoms = scenery_geom_names(self.model)
        self.scenery = WallProbe(self.model, self.trunk, self.scenery_geoms)

        self.camera = GuideCamera(self.model, self.data, self.runner.qpos_idx,
                                  self.trunk, pip_size, CTRL_HZ)
        self.machine = GuideMachine(ctrl_hz=CTRL_HZ)
        self.machine.set_follower(FOLLOWER.name)
        self.controller = GuideController(ctrl_hz=CTRL_HZ)
        self.planner = Planner()
        self.plan = None
        self.tracker: RouteTracker | None = None

        self.records: list[dict] = []
        # Everything the run accumulates lives in one object with no physics in
        # it, so this file stays about ORDER and that one stays about TALLYING.
        self.tally = RolloutTally(self.dt, float(self.data.xpos[self.trunk][2]))
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()

        self._previous_people = people_at(0.0, self.follower)
        self._camera_state = self.camera.update(
            self.data, duck_yaw=self.runner.yaw(self.data),
            subject=FOLLOWER.name)
        self._waiting_spot: np.ndarray | None = None
        self._route_points: list[np.ndarray] = []
        # Path walked in the CURRENT CHECK_FOLLOWER.  Expected to be the
        # exact-zero drift and nothing more; reported so that claim is measured.
        self._check_path_m = 0.0
        self.max_check_path_m = 0.0

    # -- markers ---------------------------------------------------------
    # Posing the world-space discs is pure PRESENTATION and lives in
    # ``guide_markers``; it is called here only so the markers move on the same
    # tick as everything else they annotate.

    # -- planning --------------------------------------------------------
    def _search_route(self, t: float) -> None:
        """Search the route from the duck's MEASURED pose and MEASURED crowd.

        Called once, when the machine enters PLAN.  Everything it consumes is a
        measurement the duck could have taken: its own trunk position, and each
        other person's current position and velocity.  Nothing reads a route
        object or a schedule, which is what keeps the plan a consequence of
        measurement rather than a lookup.
        """
        duck_xy = self.data.xpos[self.trunk][:2].copy()
        tubes = tubes_from_states(self._previous_people, FOLLOWER.name)
        self.plan = self.planner.plan(duck_xy, self.machine.destination, tubes)
        self.tracker = RouteTracker(self.plan.route)
        self._route_points = self.tracker.polyline(ROUTE_DISCS)
        self.machine.note_plan(t, self.plan)

    # -- one control tick -------------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        pos_before = self.data.xpos[self.trunk].copy()
        duck_xy_before = np.array([float(pos_before[0]), float(pos_before[1])])
        duck_yaw = self.runner.yaw(self.data)

        # -- the semantic request ------------------------------------------
        # A simulator event, delivered once, naming one of three keys.  The
        # machine resolves it by exact lookup; there is no speech and no fuzzy
        # matching, and that limitation is stated everywhere it surfaces.
        if t >= REQUEST_T_S and self.machine.destination is None:
            self.machine.receive(t, self.requested, DESTINATION_KEYS)

        # -- measurements taken BEFORE this tick's physics -------------------
        previous_follower = self._previous_people[FOLLOWER.name]
        follower_range = float(np.linalg.norm(
            previous_follower.pos - duck_xy_before))
        follower_entry = self._camera_state["people"][FOLLOWER.name]
        follower_visible = bool(follower_entry["visible"])
        eye_xy = self.camera.render_data.cam_xpos[self.camera.camera_id][:2]
        los_blocker = occluder_between(eye_xy, previous_follower.pos)
        los_ok = los_blocker is None

        remaining = (self.tracker.remaining_m if self.tracker is not None
                     else 1e9)
        arrived_at_stand = (self.tracker is not None and reached_standing_point(
            duck_xy_before, self.machine.destination, ARRIVE_RADIUS_M))
        facing_error = facing_error_deg(duck_xy_before, duck_yaw,
                                        self.machine.destination)
        follower_yaw = bearing_to(duck_xy_before, previous_follower.pos)
        look_at_yaw = None

        state_before = self.machine.state
        state, changed = self.machine.update(
            t,
            distance_m=follower_range,
            visible=follower_visible,
            los_available=los_ok,
            route_remaining_m=(0.0 if arrived_at_stand else remaining),
            facing_ok=(facing_error is not None
                       and facing_error <= FACE_TOLERANCE_DEG))

        if state == "PLAN" and self.plan is None:
            self._search_route(t)
            state = self.machine.state
            changed = True
        # CHECK_FOLLOWER ends when the duck's HEAD has actually come round to
        # her.  It is a claim about where the camera is pointing — measured
        # through the same rendering camera the PiP uses and the visibility gate
        # grades — so the rollout makes it rather than the machine.  It is NOT a
        # claim about the body: this model's turn in place is measured at
        # 1.6 deg/s, so the trunk barely moves and the command stays exactly
        # zero throughout.
        if state == "CHECK_FOLLOWER":
            looking = abs(math.degrees(wrap_angle(
                follower_yaw - self.camera.view_yaw))) \
                <= LOOK_BACK_TOLERANCE_DEG
            # Her bearing relative to the TRUNK, which is what decides whether
            # the head will still reach her a few seconds from now as she walks
            # the duck's trail towards it.
            bearing_deg = abs(math.degrees(wrap_angle(follower_yaw - duck_yaw)))
            if self._waiting_spot is None:
                self._waiting_spot = duck_xy_before.copy()
            self.machine.confirm_check(
                t, looking_back=looking, distance_m=follower_range,
                visible=follower_visible,
                bearing_ok=bearing_deg <= CHECK_ARC_TARGET_DEG)
            state = self.machine.state
            look_at_yaw = follower_yaw
            if state == "WAIT_FOR_PERSON":
                # The spot is where the duck actually STOPPED, not where the lag
                # was first noticed: the arc moved it, and the clearance gate
                # must grade the place it waits.
                self._waiting_spot = duck_xy_before.copy()
        elif state not in MONITOR_STATES:
            self._waiting_spot = None
            self.tally.reset_check()

        # -- target selection ---------------------------------------------
        aim = select(state, duck_xy=duck_xy_before, tracker=self.tracker,
                     destination=self.machine.destination,
                     follower_yaw=follower_yaw,
                     arrive_radius_m=ARRIVE_RADIUS_M)
        target_xy, target_kind = aim.target_xy, aim.kind
        cross_track = aim.cross_track_m
        if aim.look_at_yaw is not None:
            look_at_yaw = aim.look_at_yaw
        if aim.remaining_m < 1e8:
            remaining = aim.remaining_m

        command = self.controller.update(
            state, duck_xy_before, duck_yaw, target_xy=target_xy,
            look_at_yaw=look_at_yaw, cross_track_m=cross_track,
            route_remaining_m=remaining)

        # THE EXACT-ZERO CLAIM, CHECKED AS IT IS MADE.  Recording a violation
        # here rather than only in the metrics means the trace names the tick.
        from guide_states import ZERO_COMMAND_STATES
        if state in ZERO_COMMAND_STATES and float(np.max(np.abs(command))) != 0.0:
            self.tally.note_zero_violation(t, state, command)

        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        # -- the follower walks the duck's trail ---------------------------
        duck_pos = self.data.xpos[self.trunk].copy()
        duck_xy = np.array([float(duck_pos[0]), float(duck_pos[1])])
        duck_yaw_after = self.runner.yaw(self.data)
        self.follower.push_duck(duck_xy)
        # She sets off only once the duck has acknowledged and started, so the
        # opening seconds are a person waiting to be led rather than a person
        # already walking.
        self.follower.update(
            t, self.dt,
            moving=self.machine.state not in ("RECEIVE_DESTINATION", "PLAN"))

        display_t = min(t + self.dt, self.seconds)
        people = people_at(display_t, self.follower)
        pose_people(self.model, self.data, people, display_t)

        travelled = float(np.linalg.norm(duck_xy - self._previous_xy))
        self.tally.note_pose(float(duck_pos[2]), travelled)

        pose_markers(self.model, self.data, state=state,
                     route_points=self._route_points,
                     plan=self.plan,
                     destination=self.machine.destination,
                     waiting_spot=self._waiting_spot,
                     records=self.records)
        mujoco.mj_forward(self.model, self.data)

        gesture_elapsed = (t - self.machine.state_since
                           if state == "INDICATE" else None)
        # During INDICATE the head faces the destination it led her to; her
        # visibility is still measured, and still graded, through the same
        # camera.
        look_target = (np.array([self.machine.destination.xy[0],
                                 self.machine.destination.xy[1], 0.55])
                       if state in ("INDICATE", "DONE")
                       and self.machine.destination is not None else None)
        camera_state = self.camera.update(
            self.data, duck_yaw=duck_yaw_after, subject=FOLLOWER.name,
            look_at=look_target, gesture_elapsed=gesture_elapsed)
        self._camera_state = camera_state

        entry = camera_state["people"][FOLLOWER.name]
        now_visible = bool(entry["visible"])
        blocker = "" if now_visible else self.camera.blocking_geom(FOLLOWER.name)

        # -- safety --------------------------------------------------------
        clearances = {name: self.contacts.distance(self.data, name)
                      for name in ALL_NAMES}
        nearest = min(clearances, key=clearances.get)
        scenery_gap, scenery_geom = self.scenery.distance(self.data)
        self.tally.note_clearance(clearances, nearest, FOLLOWER.name,
                                  scenery_gap, scenery_geom)

        # -- bookkeeping, measured AFTER the step ---------------------------
        follower_now = people[FOLLOWER.name]
        range_after = float(np.linalg.norm(follower_now.pos - duck_xy))

        peak = float(np.max(np.abs(command)))
        self.tally.note_command(state, peak)
        if state in ("LEAD", "RESUME"):
            self.tally.note_lead(travelled, cross_track)
        if state == "CHECK_FOLLOWER":
            self.tally.note_check(travelled, len(self.machine.episodes))

        # THE GUIDE-LEADS INVARIANT.  Measured along the duck's own trail rather
        # than in a body frame: she is ahead if her arc length exceeds the
        # duck's, which on a shared path is the only meaning "ahead" can have.
        # A body-frame test would call her ahead every time the duck turned a
        # corner, which is the opposite of what it is for.
        self.tally.note_lead_gap(self.follower.trail_gap_m)

        # THE SAFETY INTERVAL.  A distance beyond the maximum is not itself a
        # failure — the duck has to notice and stop, which takes time.  What
        # must not happen is a PROLONGED interval, so the running length of the
        # current breach is what is tracked and its maximum is what is graded.
        self.tally.note_safety(range_after)

        # -- visibility, conditioned on line of sight ----------------------
        eye_xy_after = self.camera.render_data.cam_xpos[
            self.camera.camera_id][:2]
        los_blocker_after = occluder_between(eye_xy_after, follower_now.pos)
        los_ok_after = los_blocker_after is None
        self.tally.note_visibility(
            visible=now_visible, los_ok=los_ok_after,
            monitoring=state in MONITOR_STATES, blocker=blocker)

        # -- per-episode bookkeeping ---------------------------------------
        if state in MONITOR_STATES:
            self.tally.note_monitor_tick(
                len(self.machine.episodes), travelled=travelled,
                range_m=range_after, waiting_spot=self._waiting_spot,
                scenery_gap=scenery_gap, los_ok=los_ok_after,
                visible=now_visible)
        # THE WAITING CLAIM IS GRADED ON WAIT_FOR_PERSON ALONE, because that is
        # the only state that asserts the duck stopped.  CHECK_FOLLOWER walks a
        # bounded arc to bring an astern follower inside the head's MEASURED
        # 170 deg reach, and is accounted separately.
        if state == "WAIT_FOR_PERSON":
            self.tally.note_wait_tick(len(self.machine.episodes), peak=peak,
                                      travelled=travelled)

        self._previous_people = people
        self._previous_xy = duck_xy.copy()

        destination_distance = None
        if self.machine.destination is not None:
            destination_distance = float(np.linalg.norm(
                self.machine.destination.position - duck_xy))

        record = build_record(
            display_t=display_t, state=state, machine=self.machine,
            command=command, duck_xy=duck_xy, duck_yaw_after=duck_yaw_after,
            duck_pos=duck_pos, min_trunk_z=self.tally.min_trunk_z,
            camera_state=camera_state, clearances=clearances, nearest=nearest,
            scenery_gap=scenery_gap, scenery_geom=scenery_geom, people=people,
            follower=self.follower, tracker=self.tracker,
            follower_range=range_after, follower_visible=now_visible,
            follower_blocker=blocker, los_available=los_ok_after,
            los_blocker=los_blocker_after or "", path_m=self.tally.path_m,
            state_elapsed=t - self.machine.state_since,
            target_xy=target_xy, target_kind=target_kind,
            look_at_yaw=look_at_yaw, destination=self.machine.destination,
            destination_distance=destination_distance,
            facing_error_deg=facing_error,
            lagging=range_after > LAG_DISTANCE_M,
            unseen=not now_visible,
            waiting_spot=self._waiting_spot,
            safety_breach_s=self.tally.safety_breach_s)
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
    def destinations(self):
        return DESTINATIONS

    @property
    def actor_routes(self):
        return ROUTES
