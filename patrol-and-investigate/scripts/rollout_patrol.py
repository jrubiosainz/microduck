#!/usr/bin/env python3
"""Integration: facility + population + camera + detector + plan + machine +
controller + physics.

The only module that owns all of it at once, and it owns them in a strict order
every control tick:

1. the scripted population is posed analytically at ``t``;
2. the duck's own measurements are built from that world - BEFORE this tick's
   physics;
3. the camera measures what it can actually see, and the DETECTOR is fed only
   that;
4. the machine is advanced on those measurements and that verdict alone;
5. the controller emits a command from the state, the chosen target and the
   INDEPENDENT proximity/zone interlock;
6. the walking policy consumes that command and physics is stepped;
7. the world is re-posed at the display time and the camera measures again,
   from the same camera a PiP would render from.

Steps 5 and 6 are the only ones that touch locomotion.  The camera work happens
in an isolated ``MjData`` inside :class:`PatrolCamera` and is never written
back, so gaze cannot prop the robot up.

ORDERING NOTE, AND IT IS THE SUBTLE ONE: the machine decides on measurements
taken BEFORE the physics step, never after.  Measuring somebody's position and
then acting on that measurement within the same tick would let a decision be
authorised by a world state that only exists after the decision was made.  One
control tick at 50 Hz is 20 ms, which is honest and is also what a real
perception pipeline incurs.

THE CAMERA IS MEASURED TWICE PER TICK, AND THAT IS DELIBERATE.  The pre-physics
pass is what the DETECTOR sees, so a detection is caused by a world state that
existed before the duck acted on it.  The post-physics pass is what the PiP
renders and what the visibility metrics grade, so the picture and the
percentages agree.  Using one pass for both would either let the duck react to
its own future or grade it on a frame it never had.
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
from patrol_actors import bodies_at, pose_bodies
from patrol_aim import select
from patrol_camera import PatrolCamera
from patrol_cast import ALL_NAMES
from patrol_control import Interlock, PatrolController
from patrol_detect import Detector
from patrol_facility import CIRCUIT, DUCK_START_YAW_DEG, HOME, home_contains
from patrol_investigate import plan_standoff
from patrol_machine import PatrolMachine
from patrol_markers import (
    TRAIL_STRIDE,
    memory_points,
    pose_markers,
    standoff_points,
)
from patrol_plan import PatrolPlan, circuit_polyline, pursuit_point
from patrol_record import build_record
from patrol_sense import (
    build_interlock,
    build_sense,
    in_standoff_band,
    los_blocked_by,
    measured_positions,
    zone_gap_m,
)
from patrol_states import (
    MONITOR_STATES,
    SCAN_ARC_COMPLETE_DEG,
    SCAN_STATES,
    STATES,
    WALKING_STATES,
    ZERO_COMMAND_STATES,
)
from patrol_tally import PatrolTally

# Every scenery geom the clearance gate is measured against, collected from the
# scene's own naming rather than hand-listed, so a scene change cannot silently
# drop a surface from the gate.
SCENERY_PREFIXES = ("wall_", "obs_")

# States in which the duck walks its APPROACH command rather than its patrol
# cruise: closing on an anomaly, where a heading error costs the most.
APPROACH_STATES = ("APPROACH",)


def scenery_geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = [
        name for geom in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
        and name.startswith(SCENERY_PREFIXES)
    ]
    if not names:
        raise RuntimeError("no scenery geoms found; the clearance gate is vacuous")
    return tuple(names)


class PatrolRollout:
    """One deterministic patrol-and-investigate rollout, with or without rendering."""

    def __init__(self, policy_path: str | Path, seconds: float, *,
                 scene: str | Path | None = None,
                 robot_dir: str | Path | None = None,
                 pip_size: tuple[int, int] = (300, 216)):
        self.model = load_scene(scene, robot_dir)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qpos[1] = HOME.xy
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

        pose_bodies(self.model, self.data, bodies_at(0.0), 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.duck_radius = duck_planar_radius(self.model, self.data, self.trunk)
        self.duck_exact_radius = exact_planar_radius(
            self.model, self.data, self.trunk)
        self.duck_lateral_half = exact_lateral_half_width(
            self.model, self.data, self.trunk)

        self.contacts = ContactProbe(self.model, self.trunk, ALL_NAMES,
                                     prefix="actor_")
        self.scenery_geoms = scenery_geom_names(self.model)
        self.scenery = WallProbe(self.model, self.trunk, self.scenery_geoms)

        self.camera = PatrolCamera(self.model, self.data, self.runner.qpos_idx,
                                   self.trunk, pip_size, CTRL_HZ)
        self.machine = PatrolMachine(ctrl_hz=CTRL_HZ)
        self.controller = PatrolController(ctrl_hz=CTRL_HZ)
        self.plan = PatrolPlan()
        self.detector = Detector(self.dt)

        self.records: list[dict] = []
        self.tally = PatrolTally(self.dt, float(self.data.xpos[self.trunk][2]))
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()
        self._previous_bodies = bodies_at(0.0)
        self._measured_speed = 0.0
        self._standoff_plan = None
        self._verdict = None
        # The body a detection is about, latched from the tick the detection is
        # made until the branch that handles it closes.  See :meth:`step`.
        self._latched = ""
        self._latched_verdict = None
        self._route_points = circuit_polyline()
        self._camera_state = self.camera.update(
            self.data, duck_yaw=self.runner.yaw(self.data), subject="",
            look_at=np.array([HOME.xy[0], HOME.xy[1], 0.30]),
            present={n: s.present for n, s in self._previous_bodies.items()})

    # -- one control tick -------------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        pos_before = self.data.xpos[self.trunk].copy()
        duck_xy_before = np.array([float(pos_before[0]), float(pos_before[1])])
        duck_yaw = self.runner.yaw(self.data)
        state_before = self.machine.state
        bodies = self._previous_bodies

        # -- what the duck measured BEFORE this tick's physics --------------
        positions = measured_positions(bodies)
        clearances_before = {name: self.contacts.distance(self.data, name)
                             for name in ALL_NAMES}
        nearest_before = min(clearances_before, key=clearances_before.get)

        # -- the DETECTOR is fed only what the camera saw --------------------
        ready = self.detector.feed(
            t, visibility=self._camera_state["bodies"], positions=positions,
            duck_xy=duck_xy_before)

        # THE CANDIDATE IS LATCHED, AND THAT IS A SCAR.
        #
        # A first version recomputed the candidate from ``ready`` every tick
        # whenever the machine had no subject yet, and read the machine's
        # subject otherwise.  Both halves fail on the same tick: DETECT is
        # entered on tick N, but ``_subject`` is only assigned when
        # ``_detect_state`` runs on tick N+1 - and if the body happened to leave
        # the frustum on exactly that tick, ``ready`` was empty, the subject was
        # never assigned, and the branch ran to CLASSIFY with no target at all.
        # MEASURED consequence: the trolley was detected, dismissed with an
        # EMPTY verdict, never added to ``settled``, and re-detected three times
        # in the first 22 s, while the checkpoint it was found at was visited
        # four times.
        #
        # Latching here - at the moment the detection is made, in the module
        # that owns the world - means every downstream state acts on the body
        # the detection was ABOUT, which is what the machine's docstring claims
        # and what a robot with a working tracker would do.
        candidate, verdict = "", None
        subject = self.machine.subject or self._latched
        if subject:
            candidate = subject
            # THE VERDICT IS RE-DERIVED FROM THE ACCUMULATED EVIDENCE UNTIL IT
            # IS COMMITTED, WHICH IS WHAT MAKES THE OBSERVATION WORTH DOING.
            # A verdict frozen at the instant of detection would mean the
            # approach and the multi-angle observation changed nothing - the
            # robot would be performing an inspection and then reporting a
            # conclusion it had already reached.  Re-deriving means the dwell
            # accumulated while observing raises the rule margin, and the
            # confidence proxy reported at CLASSIFY reflects what the duck knew
            # THEN rather than what it suspected on first sight.
            #
            # Once ``record`` has committed a verdict it is frozen, so the
            # figure that appears in the log is the one the decision was made
            # on and cannot be rewritten by later evidence.
            verdict = (self.detector.verdicts.get(subject)
                       or self.detector.classify(subject)
                       or self._latched_verdict)
        else:
            for name in ready:
                found = self.detector.classify(name)
                if found is not None:
                    candidate, verdict = name, found
                    break

        candidate_visible = bool(
            candidate
            and self._camera_state["bodies"].get(candidate, {}).get("visible"))
        candidate_range = (
            float(np.linalg.norm(positions[candidate] - duck_xy_before))
            if candidate in positions else float("inf"))

        # -- the standoff plan, made once per investigation -------------------
        standoff_ready = False
        standoff_remaining = float("inf")
        in_band = False
        if subject and subject in positions:
            if self._standoff_plan is None and state_before in (
                    "INVESTIGATE_PLAN", "APPROACH"):
                self._standoff_plan = plan_standoff(
                    subject, positions[subject], duck_xy_before)
            if self._standoff_plan is not None and self._standoff_plan.ok:
                standoff_ready = True
                standoff_remaining = float(np.linalg.norm(
                    self._standoff_plan.standoff_xy - duck_xy_before))
            in_band = in_standoff_band(candidate_range, subject)

        sense = build_sense(
            plan=self.plan, duck_xy=duck_xy_before,
            measured_speed_mps=self._measured_speed,
            scan_arc_deg=self.camera.scan_arc_deg,
            scan_complete=self.camera.scan_arc_deg >= SCAN_ARC_COMPLETE_DEG,
            bodies_seen=tuple(self._camera_state["visible_bodies"]),
            candidate=candidate,
            candidate_verdict="" if verdict is None else verdict.verdict,
            candidate_rule="" if verdict is None else verdict.rule,
            candidate_confidence=0.0 if verdict is None else verdict.confidence,
            candidate_investigate=bool(verdict is not None
                                       and verdict.investigate),
            candidate_visible=candidate_visible,
            candidate_range_m=candidate_range,
            target_range_m=candidate_range,
            standoff_ready=standoff_ready,
            standoff_remaining_m=standoff_remaining,
            in_band=in_band,
            observe_elapsed_s=t - self.machine.state_since,
            observations_done=self.machine._angle_index,
            measured_min_clearance_m=clearances_before[nearest_before])

        state, changed = self.machine.update(t, sense)
        self._on_transition(t, state, changed, state_before, sense, verdict,
                            duck_xy_before, candidate_range)

        # -- the target -------------------------------------------------------
        aim = select(state, duck_xy=duck_xy_before,
                     subject=self.machine.subject, positions=positions,
                     plan=self.plan,
                     standoff_xy=(self._standoff_plan.standoff_xy
                                  if self._standoff_plan is not None
                                  and self._standoff_plan.ok else None),
                     observe_angle_deg=self.machine.observe_angle_deg,
                     watch_deg=self.plan.target_watch_deg)
        raw_target = aim.target_xy
        target_xy = (pursuit_point(duck_xy_before, raw_target)
                     if raw_target is not None else None)
        remaining = (float(np.linalg.norm(raw_target - duck_xy_before))
                     if raw_target is not None else 1e9)

        # -- the INDEPENDENT refusal ------------------------------------------
        interlock = build_interlock(
            duck_xy=duck_xy_before, duck_yaw=duck_yaw, bodies=bodies,
            clearances=clearances_before, target_xy=raw_target)
        self.tally.note_interlock(interlock.blocked, interlock.reason)

        command = self.controller.update(
            state, duck_xy_before, duck_yaw, target_xy=target_xy,
            remaining_m=remaining, approach=state in APPROACH_STATES,
            interlock=interlock)

        # THE EXACT-ZERO CLAIM, CHECKED AS IT IS MADE.
        if state in ZERO_COMMAND_STATES \
                and float(np.max(np.abs(command))) != 0.0:
            self.tally.note_zero_violation(t, state, command)

        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        return self._after_physics(index, t, state, sense, command, aim,
                                   interlock, verdict, target_xy)

    def _on_transition(self, t, state, changed, state_before, sense, verdict,
                       duck_xy, candidate_range) -> None:
        """Everything that happens exactly once, at a state change.

        Kept out of the tick body so the order above stays readable, and so the
        two things that MUST happen once - interrupting the patrol and
        completing a checkpoint - are visibly once rather than idempotent by
        accident.
        """
        from patrol_episode import Investigation

        if not changed:
            return

        # LATCH the detection the moment it is made, so no later tick can lose
        # the body it was about.  See the scar comment in :meth:`step`.
        if state == "DETECT":
            self._latched = sense.candidate
            self._latched_verdict = verdict

        # BREAKING OFF: snapshot the route before a single step is taken.
        if state == "INVESTIGATE_PLAN" and self.plan.open_interruption is None:
            entry = self.plan.interrupt(
                t, duck_xy,
                f"{self.machine.subject} needs investigating",
                target=self.machine.subject)
            self.machine.open_investigation(Investigation(
                index=len(self.machine.investigations),
                target=self.machine.subject,
                detected_at_s=t, detect_range_m=candidate_range,
                interrupted_checkpoint=entry.target_name,
                interrupted_index=entry.target_index))

        if state == "APPROACH" and self.machine.investigation is not None:
            investigation = self.machine.investigation
            investigation.approach_began_s = t
            investigation.approach_start_range_m = candidate_range
            if self._standoff_plan is not None and self._standoff_plan.ok:
                chosen = self._standoff_plan.chosen
                investigation.standoff_xy = chosen.xy
                # The SURFACE standoff the planner aimed for.  Recorded from
                # the chosen candidate rather than left at its default, so the
                # metrics can show what was planned beside what was measured -
                # an earlier version left it at 0.0, which read as a robot that
                # had planned to stand on top of the thing.
                investigation.standoff_m = chosen.standoff_m
                investigation.rejected_standoffs = sum(
                    1 for c in self._standoff_plan.candidates if not c.ok)

        if state == "OBSERVE" and self.machine.investigation is not None:
            self.machine.investigation.approach_end_range_m = candidate_range

        # THE VERDICT is recorded once, when CLASSIFY is entered - for a
        # DISMISSAL exactly as for an escalation.  Recording the dismissal is
        # what settles the body so it is not re-detected at every subsequent
        # checkpoint, and it is also what makes "the distractor was explicitly
        # dismissed" a logged decision rather than an absence of one.
        if state == "CLASSIFY":
            recorded = verdict or self._latched_verdict
            if recorded is not None:
                if recorded.name not in self.detector.verdicts:
                    self.detector.record(recorded)
                self.machine.record_verdict(t, recorded.as_record())
                self._verdict = recorded.as_record()
                investigation = self.machine.investigation
                if investigation is not None:
                    investigation.verdict = recorded.verdict
                    investigation.rule = recorded.rule
                    investigation.confidence = recorded.confidence

        # RESUMING: close the interruption and measure how well it got back.
        if state == "RESUME" and self.plan.open_interruption is not None:
            entry = self.plan.resume(t, duck_xy)
            investigation = self.machine.investigation
            if investigation is not None:
                investigation.resumed_at_s = t
                investigation.return_error_m = entry.return_error_m or 0.0
                investigation.resumed_checkpoint = entry.resumed_target_name
                investigation.approach_path_m = self.tally.approach_path_m.get(
                    investigation.index, 0.0)
                investigation.min_clearance_m = \
                    self.tally.investigation_min_clearance.get(
                        investigation.index, float("inf"))
                investigation.min_zone_gap_m = \
                    self.tally.investigation_min_zone_gap.get(
                        investigation.index, float("inf"))
                self.machine.close_investigation(t)
            self._standoff_plan = None

        # A dismissal also closes the branch's plan, without an interruption.
        if state == "PATROL" and state_before in ("CLASSIFY", "RESUME"):
            self._standoff_plan = None
            self._verdict = None
            self._latched = ""
            self._latched_verdict = None

        # COMPLETING A CHECKPOINT: exactly once, on leaving it for the next leg.
        #
        # CLASSIFY IS IN THIS LIST BECAUSE A DISMISSAL STILL COMPLETES THE
        # CHECKPOINT.  The duck stopped, scanned, found something, explained it
        # and moved on - that is a completed checkpoint, and an earlier version
        # that only completed on CLEAR or RESUME left the duck re-arriving at
        # ``dock-gate`` four times because the leg's target never advanced.
        if state == "PATROL" \
                and state_before in ("CLEAR", "CLASSIFY", "RESUME") \
                and not self.plan.finished_circuit:
            visited = self.machine.visited_names
            if visited and visited[-1] == self.plan.target_name:
                self.tally.note_checkpoint(
                    self.plan.target_name, sense.target_remaining_m)
                self.plan.complete_checkpoint(self.plan.target_name)

        if state == "SCAN":
            self.camera.begin_scan()

    def _after_physics(self, index, t, state, sense, command, aim, interlock,
                       verdict, target_xy) -> dict:
        """Re-pose the world at the display time, measure, and record."""
        duck_pos = self.data.xpos[self.trunk].copy()
        duck_xy = np.array([float(duck_pos[0]), float(duck_pos[1])])
        duck_yaw_after = self.runner.yaw(self.data)

        display_t = min(t + self.dt, self.seconds)
        bodies_now = bodies_at(display_t)
        pose_bodies(self.model, self.data, bodies_now, display_t)

        travelled = float(np.linalg.norm(duck_xy - self._previous_xy))
        self._measured_speed = travelled / self.dt
        self.tally.note_pose(float(duck_pos[2]), travelled)

        # -- the markers, from the SAME objects the behavior acts on ----------
        trail = [np.asarray(r["duck_xy"]) for r in self.records[::-TRAIL_STRIDE]]
        open_interruption = self.plan.open_interruption
        memory = (memory_points(open_interruption.resume_xy,
                                self.plan.target_xy)
                  if open_interruption is not None else None)
        standoff = (standoff_points(duck_xy, self._standoff_plan.standoff_xy)
                    if state == "APPROACH" and self._standoff_plan is not None
                    and self._standoff_plan.ok else None)
        positions_now = measured_positions(bodies_now)
        subject = self.machine.subject
        pose_markers(
            self.model, self.data, route_points=self._route_points,
            trail_points=trail, memory=memory, standoff=standoff,
            target_xy=(positions_now.get(subject) if subject else None),
            checkpoint_xy=self.plan.target_xy)
        mujoco.mj_forward(self.model, self.data)

        # -- the camera, in its isolated copy ---------------------------------
        present = {n: s.present for n, s in bodies_now.items()}
        if aim.scanning:
            look_at = self.camera.scan_target(duck_xy,
                                              self.plan.target_watch_deg)
        elif aim.look_at is not None:
            look_at = aim.look_at
        else:
            goal = self.plan.target_xy
            look_at = np.array([float(goal[0]), float(goal[1]), 0.30])
        camera_state = self.camera.update(
            self.data, duck_yaw=duck_yaw_after, subject=aim.subject,
            look_at=look_at, present=present)
        self._camera_state = camera_state
        if state in SCAN_STATES:
            self.camera.note_scan_arc()

        watched = aim.subject
        if watched:
            entry = camera_state["bodies"][watched]
            subject_visible = bool(entry["visible"])
            blocker = ("" if subject_visible
                       else self.camera.blocking_geom(watched))
        else:
            subject_visible = True
            blocker = ""

        # -- safety, measured against the REAL post-step pose ----------------
        clearances = {name: self.contacts.distance(self.data, name)
                      for name in ALL_NAMES}
        nearest = min(clearances, key=clearances.get)
        scenery_gap, scenery_geom = self.scenery.distance(self.data)
        gap = zone_gap_m(duck_xy)
        self.tally.note_clearance(clearances, nearest, scenery_gap,
                                  scenery_geom)
        self.tally.note_zone(gap)

        investigation = self.machine.investigation
        if investigation is not None and subject:
            self.tally.note_standoff(investigation.index,
                                     clearances.get(subject, 1e9), gap)
            if state == "APPROACH":
                self.tally.note_approach(investigation.index, travelled)
            if state == "RETURN_TO_PATROL":
                self.tally.note_return(investigation.index, travelled)

        # -- per-state bookkeeping -------------------------------------------
        peak = float(np.max(np.abs(command)))
        self.tally.note_command(state, peak, travelled, command)
        self.tally.note_zero_plateau(display_t, state, peak,
                                     state in ZERO_COMMAND_STATES)
        self.tally.note_zero_episode(state, state in ZERO_COMMAND_STATES,
                                     travelled, duck_xy, display_t)
        if state in WALKING_STATES:
            self.tally.note_walk(travelled)
        self.tally.note_seen(camera_state["visible_bodies"])

        # -- visibility, conditioned on line of sight ------------------------
        eye_xy = self.camera.render_data.cam_xpos[self.camera.camera_id][:2]
        if watched and watched in positions_now:
            los_blocker = los_blocked_by(eye_xy, positions_now[watched],
                                         bodies_now, exclude=watched)
        else:
            los_blocker = ""
        self.tally.note_visibility(
            subject=watched, visible=subject_visible, los_ok=not los_blocker,
            monitoring=state in MONITOR_STATES, blocker=blocker, t=display_t,
            active=bool(camera_state["aim_in_frustum"]))

        # THE HOME MEASUREMENT ONLY COUNTS ONCE THE CIRCUIT IS ACTUALLY DONE.
        # The duck STARTS on the guard-post pad, so a naive "is it inside the
        # pad" test reported the patrol complete at t = 0.02 s - true of the
        # geometry and false of the behavior.  Gating on the plan's own
        # completion makes "it returned home" mean it came BACK, and leaving the
        # distance at infinity until then stops the closest-approach figure
        # being the zero it starts at.
        if self.plan.finished_circuit:
            self.tally.note_home(
                distance_m=float(np.linalg.norm(duck_xy - HOME.position)),
                inside=home_contains(duck_xy), t=display_t)

        record = build_record(
            display_t=display_t, state=state, machine=self.machine,
            plan=self.plan, command=command, duck_xy=duck_xy,
            duck_yaw_after=duck_yaw_after, duck_pos=duck_pos,
            min_trunk_z=self.tally.min_trunk_z, camera_state=camera_state,
            clearances=clearances, nearest=nearest, scenery_gap=scenery_gap,
            scenery_geom=scenery_geom, bodies=bodies_now, sense=sense,
            subject=watched, subject_visible=subject_visible,
            subject_blocker=blocker, los_available=not los_blocker,
            los_blocker=los_blocker, path_m=self.tally.path_m,
            state_elapsed=t - self.machine.state_since, target_xy=target_xy,
            target_kind=aim.kind, interlock=interlock,
            standoff_plan=self._standoff_plan, verdict=self._verdict,
            investigation=investigation, zone_gap_m=gap,
            scan_arc_deg=self.camera.scan_arc_deg,
            camera_active=bool(camera_state["aim_in_frustum"]))
        self.records.append(record)

        self._previous_bodies = bodies_now
        self._previous_xy = duck_xy.copy()
        return record

    def run(self, on_frame=None, progress=None) -> list[dict]:
        for index in range(self.total_steps):
            record = self.step(index)
            if progress is not None:
                progress(index, record)
            if on_frame is not None:
                on_frame(index, record)
            if self.machine.finished and index > 10:
                # The patrol is over.  Keep stepping so the video has a settled
                # tail, but stop if the caller only wants the behavior.
                pass
        self.tally.close(self.seconds,
                         self.data.xpos[self.trunk][:2].copy())
        return self.records
