#!/usr/bin/env python3
"""Integration: actors + tracking + prediction + planner + machine + controller
+ camera + physics.

The only module that owns all of it at once, and it owns them in a strict order
every control tick:

1. the scripted traffic is posed analytically at ``t``;
2. the duck's own measurements are built from that world - BEFORE this tick's
   physics - and its tracker differentiates them into velocities;
3. the planner predicts occupancy and scores BOTH corridors;
4. the machine is advanced on those measurements and that decision alone;
5. the controller emits a command from the state, the chosen corridor and the
   INDEPENDENT proximity interlock;
6. the walking policy consumes that command and physics is stepped;
7. the world is re-posed at the display time and the camera measures what it
   actually sees, from the same camera a PiP would render from.

Steps 5 and 6 are the only ones that touch locomotion.  The camera work in step 7
happens in an isolated ``MjData`` inside :class:`SlalomCamera` and is never
written back, so gaze cannot prop the robot up.

ORDERING NOTE, AND IT IS THE SUBTLE ONE: the machine decides on measurements
taken BEFORE the physics step, never after.  Measuring somebody's position and
then acting on that measurement within the same tick would let a decision be
authorised by a world state that only exists after the decision was made.  One
control tick at 50 Hz is 20 ms, which is honest and is also what a real
perception pipeline incurs.

THE PLANNER RUNS AT A LOWER RATE THAN THE CONTROLLER, ON PURPOSE
------------------------------------------------------------------
Scoring six corridors against seven predicted bodies over sixteen horizon
samples is 672 distance evaluations, and doing it at 50 Hz costs more than the
physics.  It runs at :data:`PLAN_HZ` and the decision is HELD between updates,
which is also what a real planner does.  The controller still closes its heading
loop every tick, so the robot is not stepping open-loop between plans.
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
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene
from slalom_actors import ROUTES, actors_at, pose_actors
from slalom_aim import select
from slalom_camera import SlalomCamera
from slalom_cast import ALL_NAMES
from slalom_control import ON_CORRIDOR_M, SlalomController
from slalom_course import (
    DUCK_START_XY,
    DUCK_START_YAW_DEG,
    GOAL_BEACON_XY,
    GOAL_XY,
    goal_contains,
)
from slalom_machine import SlalomMachine
from slalom_markers import (
    TRAIL_STRIDE,
    corridor_points,
    plan_polyline,
    pose_markers,
)
from slalom_plan import (
    Decision,
    choose_corridor,
    duck_at,
    horizon_times,
    nearest_threat,
    predict_occupancy,
)
from slalom_record import build_record
from slalom_sense import (
    Tracker,
    bodies_in_lane,
    build_interlock,
    build_sense,
    los_blocked_by,
    measured_positions,
)
from slalom_states import (
    LATERAL_OFFSETS,
    MONITOR_STATES,
    PURSUIT_LOOKAHEAD_M,
    STATES,
    ZERO_COMMAND_STATES,
)
from slalom_tally import RolloutTally

# How often the planner re-scores.  See the module docstring: 10 Hz is five
# control ticks per plan, which is far faster than the 3.2 s horizon it reasons
# over and fast enough that a decision is never more than 0.1 s stale.
PLAN_HZ = 10.0

# Every scenery geom the clearance gate is measured against, collected from the
# scene's own naming rather than hand-listed, so a scene change cannot silently
# drop a surface from the gate.
SCENERY_PREFIXES = ("wall_", "obs_", "goal_pylon", "goal_beacon")

# States in which the duck walks its CAREFUL command rather than its cruise:
# while committing to a corridor and while executing a pass, so a heading error
# beside a moving body costs less lateral travel before it is corrected.
CAREFUL_STATES = ("CHOOSE_LEFT", "CHOOSE_RIGHT", "PASS")

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


class SlalomRollout:
    """One deterministic dynamic-slalom rollout, with or without rendering."""

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
        self.plan_every = max(1, int(round(CTRL_HZ / PLAN_HZ)))

        pose_actors(self.model, self.data, actors_at(0.0), 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.duck_radius = duck_planar_radius(self.model, self.data, self.trunk)
        self.duck_exact_radius = exact_planar_radius(
            self.model, self.data, self.trunk)
        self.duck_lateral_half = exact_lateral_half_width(
            self.model, self.data, self.trunk)
        # POSE-ZERO SAMPLE, not a gait maximum: measured once, here, with the
        # actors posed at t=0.  Reported for context only.  No gate consumes it;
        # clearance is measured every tick by ContactProbe against the real
        # geoms at the real pose.
        self.actor_lateral_half = max(
            exact_lateral_half_width(
                self.model, self.data, self.model.body(f"actor_{name}").id)
            for name in ALL_NAMES)

        self.contacts = ContactProbe(self.model, self.trunk, ALL_NAMES,
                                     prefix="actor_")
        self.scenery_geoms = scenery_geom_names(self.model)
        self.scenery = WallProbe(self.model, self.trunk, self.scenery_geoms)

        self.camera = SlalomCamera(self.model, self.data, self.runner.qpos_idx,
                                   self.trunk, pip_size, CTRL_HZ)
        self.machine = SlalomMachine(ctrl_hz=CTRL_HZ)
        self.controller = SlalomController(ctrl_hz=CTRL_HZ)
        self.tracker = Tracker(self.dt)

        self.records: list[dict] = []
        self.tally = RolloutTally(self.dt, float(self.data.xpos[self.trunk][2]))
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()
        self._previous_actors = actors_at(0.0)
        self._decision = Decision(side="")
        # The corridor object the machine has COMMITTED to, or None.  Holding
        # the object rather than just its offset is what lets the duck pursue a
        # fixed world line; see ``slalom_plan.Corridor.origin``.
        self._committed = None
        self._predictions: list[dict] = []
        self._tracks: list = []
        self._previous_threat_range = float("inf")
        # The duck's own measured ground speed, from its two most recent
        # measured positions.  Used only to decide when it has genuinely
        # stopped inside the goal band.
        self._measured_speed = 0.0
        self._committed_offset = 0.0
        self._camera_state = self.camera.update(
            self.data, duck_yaw=self.runner.yaw(self.data), subject="",
            look_at=np.array([GOAL_BEACON_XY[0], GOAL_BEACON_XY[1], 0.34]))

    # -- helpers -----------------------------------------------------------
    def _pursuit_target(self, duck_xy) -> np.ndarray:
        """A point ahead on the COMMITTED corridor's fixed world line.

        Built from the corridor object the planner scored and the machine
        committed to, so the duck walks the exact line that was graded.  The
        along-track coordinate is the duck's own projection onto that line plus
        a lookahead, which is what makes the target advance as the duck does
        rather than receding from it.
        """
        corridor = self._committed
        if corridor is None or corridor.origin is None:
            return np.asarray(GOAL_XY, dtype=np.float64)
        duck = np.asarray(duck_xy, dtype=np.float64)[:2]
        along = float((duck - corridor.origin) @ corridor.direction)
        return corridor.line_point(along + PURSUIT_LOOKAHEAD_M)

    def _lateral_error(self, duck_xy) -> float:
        """Signed distance from the duck to its committed corridor line.

        Measured perpendicular to that FIXED line.  Measuring it against a line
        rebuilt from the duck's current position would make the error identically
        equal to the offset forever, which is exactly the bug that ran
        CHOOSE_RIGHT into its ceiling.
        """
        corridor = self._committed
        if corridor is None or corridor.origin is None:
            return float(np.asarray(duck_xy, dtype=np.float64)[1])
        duck = np.asarray(duck_xy, dtype=np.float64)[:2]
        normal = np.array([-corridor.direction[1], corridor.direction[0]])
        return float((duck - corridor.origin) @ normal) - corridor.offset_m

    # -- one control tick -------------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        pos_before = self.data.xpos[self.trunk].copy()
        duck_xy_before = np.array([float(pos_before[0]), float(pos_before[1])])
        duck_yaw = self.runner.yaw(self.data)

        # -- measurements taken BEFORE this tick's physics -----------------
        actors = self._previous_actors
        state_before = self.machine.state

        # The tracker differentiates the duck's OWN measurements of the world.
        tracks = self.tracker.update(measured_positions(actors))
        self._tracks = tracks

        # -- the planner, at its own rate ----------------------------------
        if index % self.plan_every == 0 or not self._predictions:
            threat, ttc, threat_range = nearest_threat(
                duck_xy_before, duck_yaw, tracks)
            self._decision = choose_corridor(
                duck_xy_before, tracks, ttc_s=ttc, threat=threat,
                threat_range_m=threat_range)
            self._predictions = predict_occupancy(
                [tr for tr in tracks if tr.name == threat] if threat
                else tracks[:1])
            self._threat = threat
            self._ttc = ttc
            self._threat_range = threat_range
        threat = getattr(self, "_threat", "")
        ttc = getattr(self, "_ttc", float("inf"))
        threat_range = getattr(self, "_threat_range", float("inf"))

        clearances_before = {name: self.contacts.distance(self.data, name)
                             for name in ALL_NAMES}
        nearest_before = min(clearances_before, key=clearances_before.get)

        offset_before = 0.0
        sense = build_sense(
            duck_xy=duck_xy_before, duck_yaw=duck_yaw, tracks=tracks,
            decision=self._decision, threat=threat, threat_ttc_s=ttc,
            threat_range_m=threat_range,
            previous_range_m=self._previous_threat_range,
            lateral_error_m=self._lateral_error(duck_xy_before),
            measured_min_clearance_m=clearances_before[nearest_before],
            goal_visible=bool(self._camera_state["goal"]["visible"]),
            measured_speed_mps=self._measured_speed,
            actors=actors,
            encounter_body=self.machine.encounter_body)

        state, changed = self.machine.update(t, sense)

        # A newly committed corridor fixes the WORLD LINE the duck walks until
        # the pass is over.  Re-scoring it every tick would let the duck drift
        # between corridors mid-pass, which is not a decision - it is a wobble.
        if changed and state in ("CHOOSE_LEFT", "CHOOSE_RIGHT"):
            if self._decision.corridor is not None:
                self._committed = self._decision.corridor
                self.tally.note_prediction(
                    self.machine.completed_passes,
                    threat=threat, side=self._decision.side,
                    predicted_m=float(
                        self._decision.corridor.worst_clearance_m))
        if changed and state == "REPLAN":
            self._committed = None

        # -- the target ------------------------------------------------------
        corridor_target = (self._pursuit_target(duck_xy_before)
                           if state in ("CHOOSE_LEFT", "CHOOSE_RIGHT", "PASS")
                           and self._committed is not None else None)
        goal_target = (np.asarray(GOAL_XY, dtype=np.float64)
                       if state in ("ADVANCE", "REPLAN", "THREAT") else None)
        aim = select(state, duck_xy=duck_xy_before,
                     threat=self.machine.encounter_body or threat,
                     actors=actors, corridor_target=corridor_target,
                     goal_target=goal_target)
        target_xy, target_kind, subject = aim.target_xy, aim.kind, aim.subject

        # -- the INDEPENDENT refusal ----------------------------------------
        interlock = build_interlock(
            duck_xy=duck_xy_before, duck_yaw=duck_yaw, actors=actors,
            clearances=clearances_before)
        self.tally.note_interlock(interlock.blocked, interlock.reason)

        goal_distance = float(np.linalg.norm(
            duck_xy_before - np.asarray(GOAL_XY, dtype=np.float64)))
        careful = state in CAREFUL_STATES
        # The remaining distance handed to the controller is the distance to the
        # GOAL only when the duck is actually heading there.  During a pass it
        # is chasing a corridor point a lookahead away, and feeding it the goal
        # distance would make it ease to its settle command in the middle of an
        # encounter.
        remaining = (goal_distance if state in ("ADVANCE", "REPLAN", "THREAT")
                     else 1e9)
        command = self.controller.update(
            state, duck_xy_before, duck_yaw, target_xy=target_xy,
            remaining_m=remaining, careful=careful, interlock=interlock)

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
        actors_now = actors_at(display_t)
        pose_actors(self.model, self.data, actors_now, display_t)

        travelled = float(np.linalg.norm(duck_xy - self._previous_xy))
        lateral_travel = float(duck_xy[1] - self._previous_xy[1])
        self._measured_speed = travelled / self.dt
        self.tally.note_pose(float(duck_pos[2]), travelled)
        self.tally.note_lateral(float(duck_xy[1]), lateral_travel)

        # -- the markers, drawn from the SAME objects the planner scored -----
        left_points = right_points = None
        pred_points: list = []
        gaze_body = self.machine.encounter_body or threat
        if state in ("THREAT", "CHOOSE_LEFT", "CHOOSE_RIGHT", "WAIT", "PASS"):
            best_left = max(
                (c for c in self._decision.all_corridors if c.side == "left"),
                key=lambda c: c.worst_clearance_m, default=None)
            best_right = max(
                (c for c in self._decision.all_corridors if c.side == "right"),
                key=lambda c: c.worst_clearance_m, default=None)
            if best_left is not None:
                left_points = corridor_points(duck_xy, best_left.offset_m)
            if best_right is not None:
                right_points = corridor_points(duck_xy, best_right.offset_m)
            if gaze_body:
                pred_points = [
                    np.asarray(sample["bodies"][gaze_body])
                    for sample in self._predictions
                    if gaze_body in sample["bodies"]]

        trail = [np.asarray(r["duck_xy"]) for r in self.records[::-TRAIL_STRIDE]]
        pose_markers(
            self.model, self.data,
            plan_points=plan_polyline(duck_xy, GOAL_XY),
            trail_points=trail,
            left_points=left_points, right_points=right_points,
            pred_points=pred_points, goal_xy=GOAL_XY)
        mujoco.mj_forward(self.model, self.data)

        # -- the camera, in its isolated copy --------------------------------
        camera_state = self.camera.update(
            self.data, duck_yaw=duck_yaw_after, subject=subject,
            look_at=aim.look_at)
        self._camera_state = camera_state

        if subject:
            entry = camera_state["bodies"][subject]
            subject_visible = bool(entry["visible"])
            blocker = ("" if subject_visible
                       else self.camera.blocking_geom(subject))
        else:
            subject_visible = bool(camera_state["goal"]["visible"])
            blocker = "" if subject_visible else "out_of_frustum"

        # -- safety, measured against the REAL post-step pose ----------------
        clearances = {name: self.contacts.distance(self.data, name)
                      for name in ALL_NAMES}
        nearest = min(clearances, key=clearances.get)
        scenery_gap, scenery_geom = self.scenery.distance(self.data)
        self.tally.note_clearance(clearances, nearest, scenery_gap,
                                  scenery_geom)
        if state in ("CHOOSE_LEFT", "CHOOSE_RIGHT", "PASS"):
            body = self.machine.encounter_body or threat
            if body:
                self.tally.note_measured_for(
                    self.machine.completed_passes, clearances.get(body, 1e9))

        # -- per-state bookkeeping -------------------------------------------
        peak = float(np.max(np.abs(command)))
        self.tally.note_command(state, peak, travelled, command)
        self.tally.note_zero_plateau(
            display_t, state, peak, state in ZERO_COMMAND_STATES)
        self.tally.note_zero_episode(
            state, state in ZERO_COMMAND_STATES, travelled, duck_xy,
            display_t)
        if state in ("ADVANCE", "CHOOSE_LEFT", "CHOOSE_RIGHT", "PASS",
                     "REPLAN", "THREAT"):
            self.tally.note_walk(travelled)

        in_lane = bodies_in_lane(actors_now, duck_xy)
        self.tally.note_lane(in_lane)

        # -- visibility, conditioned on line of sight ------------------------
        # LOS accounts for OTHER BODIES as well as static geometry.  A body
        # between the camera and the subject makes seeing them impossible in
        # exactly the way a crate does, and holding the duck responsible for it
        # would grade the scenario's geometry rather than the robot.
        eye_xy = self.camera.render_data.cam_xpos[self.camera.camera_id][:2]
        target_point = (actors_now[subject].pos if subject
                        else np.asarray(GOAL_BEACON_XY, dtype=np.float64))
        los_blocker = los_blocked_by(eye_xy, target_point, actors_now,
                                     exclude=subject)
        los_ok = not los_blocker
        self.tally.note_visibility(
            subject=subject, visible=subject_visible, los_ok=los_ok,
            monitoring=state in MONITOR_STATES, blocker=blocker, t=display_t)

        goal_los = not los_blocked_by(
            eye_xy, np.asarray(GOAL_BEACON_XY, dtype=np.float64), actors_now)
        goal_distance_after = float(np.linalg.norm(
            duck_xy - np.asarray(GOAL_XY, dtype=np.float64)))
        self.tally.note_goal(
            distance_m=goal_distance_after,
            inside=goal_contains(duck_xy, 0.0),
            visible=bool(camera_state["goal"]["visible"]),
            los_ok=goal_los, t=display_t)

        record = build_record(
            display_t=display_t, state=state, machine=self.machine,
            command=command, duck_xy=duck_xy, duck_yaw_after=duck_yaw_after,
            duck_pos=duck_pos, min_trunk_z=self.tally.min_trunk_z,
            camera_state=camera_state, clearances=clearances, nearest=nearest,
            scenery_gap=scenery_gap, scenery_geom=scenery_geom,
            actors=actors_now, sense=sense, decision=self._decision,
            tracks=tracks, subject=subject, subject_visible=subject_visible,
            subject_blocker=blocker, los_available=los_ok,
            los_blocker=los_blocker or "", path_m=self.tally.path_m,
            state_elapsed=t - self.machine.state_since,
            target_xy=target_xy, target_kind=target_kind, interlock=interlock,
            lane_offset_m=float(duck_xy[1]), bodies_in_lane=in_lane,
            goal_distance_m=goal_distance_after,
            predictions=self._predictions, careful=careful,
            encounter_index=self.machine.completed_passes)
        self.records.append(record)

        self._previous_actors = actors_now
        self._previous_xy = duck_xy.copy()
        self._previous_threat_range = (threat_range if threat
                                       else float("inf"))
        return record

    def run(self, on_frame=None, progress=None) -> list[dict]:
        for index in range(self.total_steps):
            record = self.step(index)
            if progress is not None:
                progress(index, record)
            if on_frame is not None:
                on_frame(index, record)
        self.tally.close(self.seconds,
                         self.data.xpos[self.trunk][:2].copy())
        return self.records

    @property
    def actor_routes(self):
        return ROUTES
