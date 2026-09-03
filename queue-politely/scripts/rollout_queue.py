#!/usr/bin/env python3
"""Integration: people + queue reading + machine + camera + walking policy.

The only module that owns all five concerns at once, and it owns them in a
strict order every control tick:

1. the people are posed analytically at ``t`` (scripted, non-colliding);
2. the machine advances on the duck's pose and the PREVIOUS tick's queue
   reading and camera verdict;
3. the controller emits a command from the state, the trunk pose and the target
   arc;
4. the walking policy consumes that command and physics is stepped;
5. the people are re-posed at ``t + dt`` and the camera measures what it
   actually sees, from the same camera the PiP renders from.

Steps 3 and 4 are the only ones that touch locomotion.  The camera work in
step 5 happens in an isolated ``MjData`` inside :class:`QueueCamera` and is
never written back, so gaze cannot prop the robot up.

ORDERING NOTE, and it is the subtle one: the machine consumes the queue reading
from the PREVIOUS tick.  Reading and deciding within one tick would let a join
be authorised, or an advance released, by a state that only exists after the
decision.  One control tick at 50 Hz is 20 ms, which is honest and is also what
a real perception pipeline would incur.
"""

from __future__ import annotations

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
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene
from queue_camera import QueueCamera
from queue_geometry import (
    DUCK_PLANAR_RADIUS,
    DUCK_START_XY,
    in_join_band,
)
from queue_control import QueueController
from queue_machine import QueueMachine
from queue_model import judge_gaps, read_queue, rejected_available_gaps
from queue_record import build_record
from queue_tracker import RolloutTracker
from queue_path import PATH
from queue_people import (
    ADULT_HALF_EXTENT_M,
    ALL_NAMES,
    CLERK,
    QUEUE_NAMES,
    people_at,
    pose_people,
)

# Every scenery geom the clearance gate is measured against, built from the
# scene's own naming rather than hand-listed, so a scene change cannot silently
# drop a surface from the gate.
SCENERY_PREFIXES = ("post_", "rope_", "counter_", "back_wall", "side_wall",
                    "shelf_")
# Where the duck aims while walking in from outside the lane: the open end of
# the queue path.
APPROACH_TARGET = PATH.point_at(PATH.length - 0.35)


def scenery_geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = [
        name for geom in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
        and name.startswith(SCENERY_PREFIXES)
    ]
    if not names:
        raise RuntimeError("no scenery geoms found; the clearance gate is vacuous")
    return tuple(names)


class QueueRollout:
    """One deterministic queue-politely rollout, with or without rendering."""

    def __init__(self, policy_path: str | Path, seconds: float, *,
                 scene: str | Path | None = None,
                 robot_dir: str | Path | None = None,
                 pip_size: tuple[int, int] = (300, 220)):
        self.model = load_scene(scene, robot_dir)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        from queue_geometry import DUCK_START_YAW_DEG
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
        self.adult_half_extent = exact_planar_radius(
            self.model, self.data, self.model.body("person_alvarez").id)

        self.contacts = ContactProbe(self.model, self.trunk, ALL_NAMES)
        self.scenery_geoms = scenery_geom_names(self.model)
        self.scenery = WallProbe(self.model, self.trunk, self.scenery_geoms)

        self.camera = QueueCamera(
            self.model, self.data, self.runner.qpos_idx, self.trunk, pip_size)
        self.machine = QueueMachine(ctrl_hz=CTRL_HZ)
        self.controller = QueueController(ctrl_hz=CTRL_HZ)

        self.records: list[dict] = []
        self.min_trunk_z = float(self.data.xpos[self.trunk][2])
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()
        self.path_m = 0.0
        self.min_person_clearance = float("inf")
        self.min_person_name = ""
        self.min_scenery_clearance = float("inf")
        self.min_scenery_geom = ""
        # All per-cycle and per-state accumulators live in the tracker.
        self.tracker = RolloutTracker()
        self.join_evidence: dict | None = None
        self.first_reading: dict | None = None

        self._previous_people = people_at(0.0)
        self._previous_reading = self._read(0.0)
        self._previous_gaps = judge_gaps(
            self._previous_reading, ADULT_HALF_EXTENT_M)
        self._camera_state = self.camera.update(
            self.data, state="APPROACH",
            duck_yaw=self.runner.yaw(self.data), subject=None)

    # -- perception -------------------------------------------------------
    def _read(self, t: float):
        """Project everybody onto the path.  The duck is never told the order."""
        people = people_at(t)
        positions = {
            name: (float(state.pos[0]), float(state.pos[1]))
            for name, state in people.items()
            if name != CLERK.name and (state.in_queue or name not in QUEUE_NAMES)
        }
        return read_queue(positions, exclude=(CLERK.name,))

    def person_distance(self, name: str, cutoff: float = 1.5) -> float:
        return self.contacts.distance(self.data, name, cutoff)

    def scenery_distance(self, cutoff: float = 1.0) -> tuple[float, str]:
        return self.scenery.distance(self.data, cutoff)

    # -- markers ----------------------------------------------------------
    def _pose_markers(self, reading, gaps) -> None:
        """Show the tail, the duck's target footprint and the refused gaps."""
        def place(body_name: str, xy, z: float) -> None:
            body = self.model.body(body_name)
            mocap = int(self.model.body_mocapid[body.id])
            if xy is None:
                self.data.mocap_pos[mocap] = (0.0, 0.0, -3.0)
                return
            self.data.mocap_pos[mocap] = (float(xy[0]), float(xy[1]), z)

        place("tail_marker",
              reading.members[reading.tail].xy if reading.tail else None, 0.012)
        target = (PATH.point_at(self.machine.target_arc)
                  if self.machine.target_arc is not None else None)
        place("target_marker", target, 0.011)
        refused = [g for g in gaps if not g.accepted and g.fits]
        for index in range(3):
            place(f"reject_marker_{index}",
                  refused[index].xy if index < len(refused) else None, 0.010)

    # -- one control tick -------------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        pos_before = self.data.xpos[self.trunk].copy()
        duck_xy_before = (float(pos_before[0]), float(pos_before[1]))
        previous_state = self.machine.state

        # The machine consumes the PREVIOUS tick's reading; see the module
        # docstring for why the measurement may not come first.
        reading = self._previous_reading
        gaps = self._previous_gaps
        arc_before, cross_before, off_path_before = PATH.project(duck_xy_before)

        predecessor = self.machine.predecessor
        predecessor_arc = (
            reading.members[predecessor].arc_m
            if predecessor and predecessor in reading.members else None)
        # Everybody still in line who is ahead of the duck.  Counted from the
        # live reading rather than from the clock, so reaching the counter is a
        # consequence of the queue emptying and not of elapsed time.
        remaining = sum(
            1 for name in reading.order
            if reading.members[name].arc_m < arc_before)

        state, changed = self.machine.update(
            t, duck_arc=float(arc_before), duck_off_path_m=float(off_path_before),
            reading=reading, gaps=gaps, predecessor_arc=predecessor_arc,
            predecessors_remaining=remaining)
        if changed and state in ("JOIN", "ADVANCE"):
            self.controller.reset()

        duck_yaw = self.runner.yaw(self.data)
        command = self.controller.update(
            state, duck_xy_before, duck_yaw, duck_arc=float(arc_before),
            target_arc=self.machine.target_arc,
            approach_target=APPROACH_TARGET)
        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        display_t = min(t + self.dt, self.seconds)
        display_people = people_at(display_t)
        pose_people(self.model, self.data, display_people, display_t)

        duck_pos = self.data.xpos[self.trunk].copy()
        duck_xy = (float(duck_pos[0]), float(duck_pos[1]))
        duck_yaw_after = self.runner.yaw(self.data)
        self.min_trunk_z = min(self.min_trunk_z, float(duck_pos[2]))
        travelled = float(np.linalg.norm(duck_pos[:2] - self._previous_xy))
        self.path_m += travelled

        display_reading = self._read(display_t)
        display_gaps = judge_gaps(display_reading, ADULT_HALF_EXTENT_M)
        self._pose_markers(display_reading, display_gaps)
        mujoco.mj_forward(self.model, self.data)

        arc, cross, off_path = PATH.project(duck_xy)
        subject = self.machine.predecessor
        camera_state = self.camera.update(
            self.data, state=state, duck_yaw=duck_yaw_after, subject=subject)
        self._camera_state = camera_state

        clearances = {name: self.person_distance(name) for name in ALL_NAMES}
        nearest = min(clearances, key=clearances.get)
        if clearances[nearest] < self.min_person_clearance:
            self.min_person_clearance = clearances[nearest]
            self.min_person_name = nearest
        scenery_gap, scenery_geom = self.scenery_distance()
        if scenery_gap < self.min_scenery_clearance:
            self.min_scenery_clearance = scenery_gap
            self.min_scenery_geom = scenery_geom

        # -- per-state bookkeeping ---------------------------------------
        cycle_index = len(self.machine.cycles)
        self.tracker.note_step(
            state=state, command=command, cycle_index=cycle_index,
            arc_before=arc_before, cross=cross, travelled=travelled,
            subject=subject, camera_state=camera_state,
            display_reading=display_reading)
        truth = self.tracker.note_order(
            display_t=display_t, state=state,
            display_reading=display_reading, display_people=display_people)

        if self.first_reading is None and state == "IDENTIFY_TAIL":
            self.first_reading = {
                **display_reading.as_record(),
                "truth": truth,
                "gaps": [g.as_record() for g in display_gaps],
                "rejected_available": [
                    g.as_record() for g in rejected_available_gaps(display_gaps)],
            }

        if changed and state == "WAIT" and self.join_evidence is None:
            ok, longitudinal, lateral = in_join_band(
                float(arc), float(cross),
                display_reading.members[self.machine.joined_behind].arc_m
                if self.machine.joined_behind in display_reading.members
                else float("nan"))
            self.join_evidence = {
                "t": display_t, "duck_arc_m": float(arc),
                "behind": self.machine.joined_behind,
                "longitudinal_m": longitudinal, "lateral_m": lateral,
                "in_band": bool(ok),
                "duck_xy": list(duck_xy),
            }

        self._previous_people = display_people
        self._previous_reading = display_reading
        self._previous_gaps = display_gaps
        self._previous_xy = duck_pos[:2].copy()

        record = build_record(
            display_t=display_t, state=state, machine=self.machine,
            cycle_index=cycle_index, command=command, duck_xy=duck_xy,
            duck_yaw_after=duck_yaw_after, duck_pos=duck_pos,
            min_trunk_z=self.min_trunk_z, arc=arc, cross=cross,
            off_path=off_path, subject=subject,
            display_reading=display_reading, truth=truth,
            display_gaps=display_gaps,
            rejected_available=rejected_available_gaps(display_gaps),
            camera_state=camera_state, clearances=clearances, nearest=nearest,
            scenery_gap=scenery_gap, scenery_geom=scenery_geom,
            display_people=display_people, path_m=self.path_m,
            state_elapsed=t - self.machine.state_since)
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

    # The gate layer and the tests read these directly; the tracker owns them.
    @property
    def cycle_path(self): return self.tracker.cycle_path

    @property
    def cycle_start_arc(self): return self.tracker.cycle_start_arc

    @property
    def cycle_tracking(self): return self.tracker.cycle_tracking

    @property
    def cycle_tracking_after_service(self):
        return self.tracker.cycle_tracking_after_service

    @property
    def cycle_command_max(self): return self.tracker.cycle_command_max

    @property
    def cycle_cross_track(self): return self.tracker.cycle_cross_track

    @property
    def stationary_command_max(self): return self.tracker.stationary_command_max

    @property
    def order_samples(self): return self.tracker.order_samples
