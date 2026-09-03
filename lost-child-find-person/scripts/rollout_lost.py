#!/usr/bin/env python3
"""Integration: people + camera + identity + machine + controller + physics.

The only module that owns all six concerns at once, and it owns them in a strict
order every control tick:

1. the people are posed analytically at ``t`` (scripted, non-colliding);
2. the machine advances on the PREVIOUS tick's camera verdict;
3. the controller emits a command from the state and the current target;
4. the walking policy consumes that command and physics is stepped;
5. the people are re-posed at ``t + dt`` and the camera measures what it
   actually sees, from the same camera the PiP renders from;
6. identity is scored on that fresh measurement, ready for the next tick.

Steps 3 and 4 are the only ones that touch locomotion.  The camera work in
step 5 happens in an isolated ``MjData`` inside :class:`LostCamera` and is never
written back, so gaze cannot prop the robot up.

ORDERING NOTE, and it is the subtle one: the machine consumes the camera verdict
from the PREVIOUS tick.  Measuring and deciding within one tick would let a
reacquisition be authorised by a sighting that only exists after the decision.
One control tick at 50 Hz is 20 ms, which is honest and is also what a real
perception pipeline would incur.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from contact_geometry import ContactProbe, WallProbe, duck_planar_radius, exact_planar_radius
from lost_camera import LostCamera
from lost_cast import ALL_NAMES, GUARDIAN
from lost_constants import ARRIVE_TOLERANCE_M, VX_CLOSE
from lost_control import LostController
from lost_geometry import (
    DUCK_START_XY,
    DUCK_START_YAW_DEG,
    FOLLOW_DISTANCE_M,
    STANDOFF_TARGET_M,
    safe_standoff_point,
)
from lost_identity import IdentityTracker
from lost_machine import LostMachine
from lost_memory import GuardianTrail, line_of_sight_available, plan_route, route_progress
from lost_people import people_at, pose_people
from lost_record import build_record
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene

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


class LostRollout:
    """One deterministic lost-child rollout, with or without rendering."""

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
        # people posed at t=0 and their arms down.  Reported as
        # ``adult_half_extent_m`` for context only.  Over a full gait cycle the
        # guardian's exact half-extent runs 0.1375 -> 0.2629 m, so this figure
        # must not be read as a bound.  No gate consumes it; clearance is
        # measured every tick by ``ContactProbe`` against the real geoms.
        self.adult_half_extent = exact_planar_radius(
            self.model, self.data, self.model.body(f"person_{GUARDIAN.name}").id)

        self.contacts = ContactProbe(self.model, self.trunk, ALL_NAMES)
        self.scenery_geoms = scenery_geom_names(self.model)
        self.scenery = WallProbe(self.model, self.trunk, self.scenery_geoms)

        self.camera = LostCamera(self.model, self.data, self.runner.qpos_idx,
                                 self.trunk, pip_size, CTRL_HZ)
        self.machine = LostMachine(ctrl_hz=CTRL_HZ)
        self.machine.set_guardian(GUARDIAN.name)
        self.controller = LostController(ctrl_hz=CTRL_HZ)
        self.identity = IdentityTracker(
            reference=GUARDIAN.descriptor(), guardian=GUARDIAN.name, dt=self.dt)
        self.trail = GuardianTrail()

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

        # Per-phase accumulators.
        self.state_command_max: dict[str, float] = {}
        self.state_steps: dict[str, int] = {}
        self.follow_visible: list[bool] = []
        self.follow_range: list[float] = []
        # Rejoin bookkeeping, per cycle.
        self.rejoin_path: dict[int, float] = {}
        self.rejoin_start_range: dict[int, float] = {}
        self.rejoin_end_range: dict[int, float] = {}
        self.rejoin_start_xy: dict[int, np.ndarray] = {}
        self.rejoin_end_xy: dict[int, np.ndarray] = {}
        self.rejoin_visible: dict[int, list[bool]] = {}
        self.rejoin_visible_with_los: dict[int, list[bool]] = {}
        self.rejoin_min_clearance: dict[int, float] = {}
        self.rejoin_routes: dict[int, dict] = {}
        # Occlusion bookkeeping: runs of consecutive invisible ticks while the
        # duck was actively looking for her, attributed to the blocking geom.
        self.occlusion_runs: list[dict] = []
        self._occlusion_open: dict | None = None
        # Which look-alikes were ever camera-visible, and when.
        self.lookalike_seen: dict[str, float] = {}

        self._route = None
        self._route_goal: np.ndarray | None = None
        # Monotonic waypoint cursor. A stateless "first point farther than
        # tolerance" selector starts chasing an already-passed corner again as
        # soon as its distance grows, producing an endless loop around it.
        self._route_index = 1
        self._reached_goal = False
        self._previous_people = people_at(0.0)
        self._camera_state = self.camera.update(
            self.data, duck_yaw=self.runner.yaw(self.data),
            subject=GUARDIAN.name, scanning=False)
        self._sighting = None
        self._confirmed_s = 0.0

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

    def _pose_markers(self, route_target) -> None:
        """Show the last-known footprint, the world trail, and the rejoin route."""
        self._place("last_seen_marker", self.trail.last_seen_xy, 0.012)
        self._place("target_marker", route_target, 0.011)
        for index in range(14):
            point = (self.trail.points[index]
                     if index < len(self.trail.points) else None)
            self._place(f"trail_{index}", point, 0.007)
        waypoints = (list(self._route.waypoints[1:])
                     if self._route is not None else [])
        for index in range(8):
            self._place(f"route_{index}",
                        waypoints[index] if index < len(waypoints) else None,
                        0.008)
        refused = self.identity.distinct_rejected()[:2]
        for index in range(2):
            name = refused[index] if index < len(refused) else None
            xy = (self._previous_people[name].pos if name else None)
            self._place(f"reject_marker_{index}", xy, 0.009)

    # -- planning --------------------------------------------------------
    def _plan_rejoin(self, t: float, duck_xy) -> None:
        """Plan a collision-safe route to a standoff behind the guardian.

        Planned ONCE at the moment of reacquisition, from the guardian's
        confirmed position, and then replayed.  Re-planning every tick would let
        the duck drift toward her as she moves and would make "the route avoided
        the kiosk" a statement about the last tick rather than about the path
        that was walked.
        """
        guardian_xy = self._previous_people[GUARDIAN.name].pos
        goal = safe_standoff_point(guardian_xy, duck_xy, STANDOFF_TARGET_M)
        self._route_goal = np.asarray(goal, dtype=np.float64)
        self._route = plan_route(
            duck_xy, self._route_goal,
            people_at_time=lambda when: {
                n: s.pos for n, s in people_at(min(when, self.seconds)).items()
                if n != GUARDIAN.name},
            speed=0.209, t0=t)
        self._route_index = 1 if len(self._route.waypoints) > 1 else 0
        self.rejoin_routes[self.machine.cycle_index] = self._route.as_record()

    # -- one control tick -------------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        pos_before = self.data.xpos[self.trunk].copy()
        duck_xy_before = np.array([float(pos_before[0]), float(pos_before[1])])

        # The machine consumes the PREVIOUS tick's camera verdict; see the
        # module docstring for why the measurement may not come first.
        guardian_entry = self._camera_state["people"][GUARDIAN.name]
        guardian_visible = bool(guardian_entry["visible"])
        guardian_range = float(np.linalg.norm(
            self._previous_people[GUARDIAN.name].pos - duck_xy_before))

        state_before = self.machine.state
        state, changed = self.machine.update(
            t, guardian_visible=guardian_visible,
            guardian_confirmed_s=self._confirmed_s,
            best_candidate=self._sighting,
            reached_goal=self._reached_goal, tracker=self.identity)

        # A refusal is recorded at the moment the machine enters REJECT, using
        # the sighting that was on the table when it decided.
        if changed and state == "REJECT":
            pending = getattr(self.machine, "_pending_reject", None)
            sighting = pending if pending is not None else self._sighting
            if sighting is not None and sighting.name == self.machine.candidate:
                self.machine.note_rejection(self.identity.reject(sighting, t))
            elif self.machine.candidate is not None:
                # The candidate left the frame before it could be re-scored.
                # Refuse it on the last score it had, rather than silently
                # dropping it: a candidate the duck stopped considering is still
                # a candidate the duck did not lock onto.
                last = next((s for s in reversed(self.records)
                             if s.get("sighting")
                             and s["sighting"]["name"] == self.machine.candidate),
                            None)
                if last is not None:
                    from lost_identity import Sighting
                    record = last["sighting"]
                    revived = Sighting(
                        record["name"], t, record["score"], record["penalties"],
                        tuple(record["readable"]), record["complete_descriptor"],
                        record["range_m"], record["off_axis_deg"],
                        "candidate", record["reason"])
                    self.machine.note_rejection(self.identity.reject(revived, t))

        if changed and state == "REJOIN":
            self._plan_rejoin(t, duck_xy_before)
            self.controller.reset()
            self.rejoin_start_range[self.machine.cycle_index] = guardian_range
            self.rejoin_start_xy[self.machine.cycle_index] = duck_xy_before.copy()
        if changed and state == "FOLLOW" and state_before == "REJOIN":
            self._route = None
            self._route_goal = None
            self._reached_goal = False

        # -- target selection --------------------------------------------
        route_target = None
        settle = False
        if state == "FOLLOW":
            route_target = self._previous_people[GUARDIAN.name].pos
        elif state == "REJOIN" and self._route is not None:
            last = len(self._route.waypoints) - 1
            while (self._route_index < last
                   and float(np.linalg.norm(
                       self._route.waypoints[self._route_index]
                       - duck_xy_before)) <= 0.18):
                self._route_index += 1
            route_target = self._route.waypoints[self._route_index]
            remaining = float(np.linalg.norm(self._route_goal - duck_xy_before))
            settle = self._route_index == last and remaining <= 0.30
            self._reached_goal = bool(remaining <= ARRIVE_TOLERANCE_M)

        duck_yaw = self.runner.yaw(self.data)
        command = self.controller.update(
            state, duck_xy_before, duck_yaw, target_xy=route_target,
            range_m=guardian_range if state == "FOLLOW" else None,
            settle=settle)
        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        # -- re-pose the world and measure -------------------------------
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

        self._pose_markers(route_target)
        mujoco.mj_forward(self.model, self.data)

        camera_state = self.camera.update(
            self.data, duck_yaw=duck_yaw_after, subject=self.machine.subject,
            scanning=(state in ("STOP", "SEARCH_SWEEP")))
        self._camera_state = camera_state

        entry = camera_state["people"][GUARDIAN.name]
        now_visible = bool(entry["visible"])
        blocker = "" if now_visible else self.camera.blocking_geom(GUARDIAN.name)
        if now_visible:
            self.trail.observe(display_t, people[GUARDIAN.name].pos)

        # Identity scoring on the FRESH measurement, for the next tick.
        self._sighting = self.identity.best_candidate(
            display_t, camera_state["people"])
        guardian_sighting = None
        if self._sighting is not None and self._sighting.name == GUARDIAN.name:
            guardian_sighting = self._sighting
        else:
            from lost_identity import evaluate
            self.identity.note_visible(GUARDIAN.name, now_visible)
            guardian_sighting = evaluate(
                GUARDIAN.name, display_t, self.identity.reference, entry)
        self._confirmed_s = self.identity.confirm(guardian_sighting)
        if self._sighting is not None and self._sighting.verdict == "accept" \
                and self._sighting.name == GUARDIAN.name:
            self.identity.accept(self._sighting, display_t)

        # -- safety ------------------------------------------------------
        clearances = {name: self.person_distance(name) for name in ALL_NAMES}
        nearest = min(clearances, key=clearances.get)
        if clearances[nearest] < self.min_person_clearance:
            self.min_person_clearance = clearances[nearest]
            self.min_person_name = nearest
        if clearances[nearest] <= 0.0:
            self.contact_steps += 1
        scenery_gap, scenery_geom = self.scenery_distance()
        if scenery_gap < self.min_scenery_clearance:
            self.min_scenery_clearance = scenery_gap
            self.min_scenery_geom = scenery_geom

        # -- bookkeeping -------------------------------------------------
        peak = float(np.max(np.abs(command)))
        self.state_command_max[state] = max(
            self.state_command_max.get(state, 0.0), peak)
        self.state_steps[state] = self.state_steps.get(state, 0) + 1

        los_ok, los_blocker = line_of_sight_available(
            self.camera.render_data.cam_xpos[self.camera.camera_id][:2],
            people[GUARDIAN.name].pos)

        if state == "FOLLOW":
            self.follow_visible.append(now_visible)
            self.follow_range.append(float(np.linalg.norm(
                people[GUARDIAN.name].pos - duck_xy)))

        cycle = self.machine.cycle_index
        if state == "REJOIN":
            self.rejoin_path[cycle] = self.rejoin_path.get(cycle, 0.0) + travelled
            self.rejoin_visible.setdefault(cycle, []).append(now_visible)
            if los_ok:
                self.rejoin_visible_with_los.setdefault(cycle, []).append(now_visible)
            self.rejoin_end_range[cycle] = float(np.linalg.norm(
                people[GUARDIAN.name].pos - duck_xy))
            self.rejoin_end_xy[cycle] = duck_xy.copy()
            self.rejoin_min_clearance[cycle] = min(
                self.rejoin_min_clearance.get(cycle, float("inf")),
                clearances[nearest])

        # Occlusion runs, only while the duck is actively looking for her.
        searching = state in ("LOST", "STOP", "SEARCH_SWEEP", "CANDIDATE",
                              "REJECT")
        if searching and not now_visible:
            if self._occlusion_open is None:
                self._occlusion_open = {
                    "start_s": display_t, "end_s": display_t,
                    "blockers": {}, "cycle": cycle}
            self._occlusion_open["end_s"] = display_t
            key = blocker or "out_of_frustum"
            self._occlusion_open["blockers"][key] = (
                self._occlusion_open["blockers"].get(key, 0) + 1)
        elif self._occlusion_open is not None:
            run = self._occlusion_open
            run["duration_s"] = round(run["end_s"] - run["start_s"] + self.dt, 3)
            self.occlusion_runs.append(run)
            self._occlusion_open = None

        # Record camera evidence for every non-guardian candidate that can be
        # refused.  Restricting this bookkeeping to the two authored
        # look-alikes made a genuinely visible crowd distractor fail the
        # evidence gate merely because its role label was "crowd".
        for name in ALL_NAMES:
            if name == GUARDIAN.name:
                continue
            if camera_state["people"][name]["visible"] and name not in self.lookalike_seen:
                self.lookalike_seen[name] = display_t

        self._previous_people = people
        self._previous_xy = duck_xy.copy()

        record = build_record(
            display_t=display_t, state=state, machine=self.machine,
            cycle_index=cycle, command=command, duck_xy=duck_xy,
            duck_yaw_after=duck_yaw_after, duck_pos=duck_pos,
            min_trunk_z=self.min_trunk_z, subject=self.machine.subject,
            camera_state=camera_state, clearances=clearances, nearest=nearest,
            scenery_gap=scenery_gap, scenery_geom=scenery_geom, people=people,
            trail=self.trail, sighting=self._sighting,
            guardian_range=float(np.linalg.norm(
                people[GUARDIAN.name].pos - duck_xy)),
            guardian_visible=now_visible, guardian_blocker=blocker,
            confirmed_s=self._confirmed_s,
            invisible_for=self.machine._invisible_for,
            route=self._route, route_target=route_target,
            los_available=los_ok, los_blocker=los_blocker,
            path_m=self.path_m, state_elapsed=t - self.machine.state_since,
            rejections=self.identity.rejections)
        self.records.append(record)
        return record

    def run(self, on_frame=None, progress=None) -> list[dict]:
        for index in range(self.total_steps):
            record = self.step(index)
            if progress is not None:
                progress(index, record)
            if on_frame is not None:
                on_frame(index, record)
        # Close any open occlusion run so the metrics see it.
        if self._occlusion_open is not None:
            run = self._occlusion_open
            run["duration_s"] = round(run["end_s"] - run["start_s"] + self.dt, 3)
            self.occlusion_runs.append(run)
            self._occlusion_open = None
        self.machine.finish(self.seconds)
        return self.records
