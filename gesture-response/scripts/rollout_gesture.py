#!/usr/bin/env python3
"""Integration: arena + cast + camera + detector + machine + controller + physics.

The only module that owns all of it at once, in a strict order every tick:

1. the scripted people are posed analytically at ``t``;
2. the duck's own measurements are built from that world - BEFORE this tick's
   physics;
3. the camera measures what it can actually see, and the DETECTOR is fed only
   that;
4. the machine is advanced on those measurements and that verdict alone;
5. the controller emits a command from the state and the INDEPENDENT interlock;
6. the walking policy consumes that command and physics is stepped;
7. the world is re-posed at the display time and the camera measures again,
   from the same camera a PiP would render from.

Steps 5 and 6 are the only ones that touch locomotion.  The camera work happens
in an isolated ``MjData`` inside :class:`GestureCamera` and is never written
back, so gaze cannot prop the robot up.

ORDERING NOTE, THE SUBTLE ONE: the machine decides on measurements taken BEFORE
the physics step, never after.  Measuring an arm and then acting on that
measurement within the same tick would let a decision be authorised by a world
state that only exists after the decision was made.  One control tick at 50 Hz
is 20 ms, which is honest and is what a real perception pipeline incurs.

THE CAMERA IS MEASURED TWICE PER TICK, DELIBERATELY.  The pre-physics pass is
what the DETECTOR reads a gesture from, so an acceptance is caused by a world
state that existed before the duck acted on it.  The post-physics pass is what
the PiP renders and what the visibility metrics grade, so the picture and the
percentages agree.
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
from gest_actors import bodies_at, pose_bodies
from gest_arena import DUCK_START, DUCK_START_YAW_DEG
from gest_camera import GestureCamera
from gest_cast import ALL_NAMES, INSTRUCTOR
from gest_control import GestureController
from gest_detect import GestureDetector
from gest_machine import GestureMachine
from gest_phases import RolloutPhases
from gest_sense import (
    build_interlock,
    build_sense,
    measured_positions,
    measured_yaws,
    ranges_from,
)
from gest_states import (
    SETTLED_MPS,
    STANDOFF_STOP_M,
    STANDOFF_TARGET_M,
    TURN_TARGET_DEG,
)
from gest_tally import GestureTally

# Every scenery geom the clearance gate is measured against, collected from the
# scene's own naming rather than hand-listed, so a scene change cannot silently
# drop a surface from the gate.
SCENERY_PREFIXES = ("wall_", "obs_")


def scenery_geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = [
        name for geom in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
        and name.startswith(SCENERY_PREFIXES)
    ]
    if not names:
        raise RuntimeError("no scenery geoms found; the clearance gate is vacuous")
    return tuple(names)


class GestureRollout(RolloutPhases):
    """One deterministic gesture-response rollout, with or without rendering.

    The per-tick ORDER lives here; the two bookkeeping phases - what happens
    once at a state change, and what is measured after physics - live in
    :class:`gest_phases.RolloutPhases`.
    """

    def __init__(self, policy_path: str | Path, seconds: float, *,
                 scene: str | Path | None = None,
                 robot_dir: str | Path | None = None,
                 pip_size: tuple[int, int] = (300, 216)):
        self.model = load_scene(scene, robot_dir)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qpos[1] = DUCK_START
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

        self.camera = GestureCamera(self.model, self.data, self.runner.qpos_idx,
                                    self.trunk, pip_size, CTRL_HZ)
        self.machine = GestureMachine(ctrl_hz=CTRL_HZ)
        self.controller = GestureController(ctrl_hz=CTRL_HZ)
        self.detector = GestureDetector(self.dt, INSTRUCTOR)

        self.records: list[dict] = []
        self.tally = GestureTally(self.dt, float(self.data.xpos[self.trunk][2]))
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()
        self._previous_bodies = bodies_at(0.0)
        self._measured_speed = 0.0

        # The pose the duck held when the CURRENT command was accepted.  Every
        # turn and reverse claim is measured against these, never against a
        # command register or a state name.
        self._reference_xy = self._previous_xy.copy()
        self._reference_yaw = self.runner.yaw(self.data)
        self._stop_hold_s = 0.0
        self._stop_reference_xy = self._previous_xy.copy()
        # The command magnitude on the tick BEFORE a STOP was confirmed, and
        # how many ticks the register took to reach exactly zero afterwards.
        self._last_peak = 0.0
        self._stop_zero_ticks: int | None = None
        # Confirm evidence accumulating for the NEXT episode: how many OBSERVE
        # and CONFIRM ticks preceded it, and on how many of those the
        # instructor was visible with a fully readable arm.
        self._pending_confirm_ticks = 0
        self._pending_visible_ticks = 0
        self._pending_readable_ticks = 0

        self._camera_state = self.camera.update(
            self.data, duck_yaw=self.runner.yaw(self.data), subject="",
            look_at=np.array([DUCK_START[0], DUCK_START[1] + 2.0, 0.42]),
            present={n: s.present for n, s in self._previous_bodies.items()})

    # -- one control tick ---------------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        pos_before = self.data.xpos[self.trunk].copy()
        duck_xy_before = np.array([float(pos_before[0]), float(pos_before[1])])
        duck_yaw = self.runner.yaw(self.data)
        state_before = self.machine.state
        bodies = self._previous_bodies

        # -- what the duck measured BEFORE this tick's physics --------------
        positions = measured_positions(bodies)
        yaws = measured_yaws(bodies)
        ranges = ranges_from(duck_xy_before, positions)
        clearances_before = {name: self.contacts.distance(self.data, name)
                             for name in ALL_NAMES}
        nearest_before = min(clearances_before, key=clearances_before.get)

        # -- the DETECTOR is fed only what the camera saw --------------------
        view = self.detector.feed(
            t, visibility=self._camera_state["bodies"],
            keypoints=self._camera_state["keypoints"],
            yaws=yaws, ranges=ranges)
        confirmed = self.detector.confirmed(t)

        instructor_entry = self._camera_state["bodies"].get(INSTRUCTOR, {})
        instructor_visible = bool(instructor_entry.get("visible"))
        arm_readable = bool(
            instructor_entry.get("arm_readable", {}).get("l")
            or instructor_entry.get("arm_readable", {}).get("r"))
        instructor_range = float(ranges.get(INSTRUCTOR, 1e9))
        instructor_clearance = float(clearances_before.get(INSTRUCTOR, 1e9))

        # THE STOP HOLD IS MEASURED, NOT TIMED FROM THE STATE.  It accumulates
        # only while the duck is genuinely below the settled speed, so a robot
        # that logged EXECUTE_STOP while still coasting would not accrue it.
        #
        # It is measured on the state the machine is ALREADY in, one tick before
        # the machine can act on it, which costs exactly one control tick of
        # credit: a STOP that has physically held for the full 2.00 s reports
        # 1.98 s.  The machine's own exit test carries that tick, so the gate is
        # graded against the same quantity the machine decided on rather than
        # against a bar the measurement can never reach.
        if state_before == "EXECUTE_STOP" \
                and self._measured_speed <= SETTLED_MPS:
            self._stop_hold_s += self.dt
        elif state_before != "EXECUTE_STOP":
            self._stop_hold_s = 0.0
        elif state_before != "EXECUTE_STOP":
            self._stop_hold_s = 0.0

        sense = build_sense(
            detector_view=view, confirmed=confirmed,
            instructor_visible=instructor_visible, arm_readable=arm_readable,
            instructor_range_m=instructor_range,
            measured_speed_mps=self._measured_speed,
            duck_yaw=duck_yaw, reference_yaw=self._reference_yaw,
            reference_xy=self._reference_xy, duck_xy=duck_xy_before,
            instructor_clearance_m=instructor_clearance,
            stop_hold_s=self._stop_hold_s,
            measured_min_clearance_m=clearances_before[nearest_before])

        state, changed = self.machine.update(t, sense)
        self._on_transition(t, state, changed, state_before, sense, confirmed,
                            duck_xy_before, duck_yaw, instructor_range,
                            instructor_clearance)

        # -- the target ---------------------------------------------------
        # THE APPROACH TARGET IS THE STANDOFF POINT, not the person.  Driving at
        # the person and stopping on a clearance test would leave the duck's
        # heading pointed into her at the moment it stopped; aiming at a point
        # on the line between them at the standoff distance means the manoeuvre
        # itself ends safely.
        target_xy = None
        remaining = 1e9
        if state == "EXECUTE_APPROACH" and INSTRUCTOR in positions:
            person = positions[INSTRUCTOR]
            span = person - duck_xy_before
            distance = float(np.linalg.norm(span))
            if distance > 1e-6:
                unit = span / distance
                target_xy = person - unit * (
                    STANDOFF_TARGET_M + self.duck_radius)
                remaining = max(0.0, distance - STANDOFF_STOP_M
                                - self.duck_radius)

        # -- the INDEPENDENT refusal ---------------------------------------
        interlock = build_interlock(
            duck_xy=duck_xy_before, duck_yaw=duck_yaw, bodies=bodies,
            clearances=clearances_before, state=state, target_xy=target_xy)
        self.tally.note_interlock(interlock.blocked, interlock.reason)

        command = self.controller.update(
            state, duck_xy_before, duck_yaw, target_xy=target_xy,
            remaining_m=remaining,
            turned_deg=sense.yaw_delta_deg, turn_target_deg=TURN_TARGET_DEG,
            reference_yaw=self._reference_yaw, interlock=interlock)

        peak = float(np.max(np.abs(command)))
        # HOW PROMPTLY A STOP ZEROED THE REGISTER, measured by tick index.
        if state == "EXECUTE_STOP" and self._stop_zero_ticks is None:
            self._stop_zero_ticks = 0 if peak == 0.0 else None
        elif state == "EXECUTE_STOP" and self._stop_zero_ticks is not None:
            pass

        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        self._last_peak = peak
        return self._after_physics(index, t, state, sense, command, interlock,
                                   target_xy, view)

    # -- transitions ---------------------------------------------------------
    def run(self, on_frame=None, progress=None) -> list[dict]:
        for index in range(self.total_steps):
            record = self.step(index)
            if progress is not None:
                progress(index, record)
            if on_frame is not None:
                on_frame(index, record)
        self.machine.close_episode(self.seconds)
        self.tally.close()
        return self.records
