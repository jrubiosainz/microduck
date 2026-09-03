#!/usr/bin/env python3
"""Integration layer: people routes + recall machine + camera + walking policy.

This is the only module that owns all four concerns at once, and it owns them
in a strict order every control tick:

1. the adults are posed analytically at ``t`` (scripted, non-colliding mocap);
2. the recall machine advances on the scripted calls and the CAMERA's verdict;
3. the approach controller emits a command from range and bearing error;
4. the walking policy consumes that command and physics is stepped;
5. the adults are re-posed at ``t + dt`` and the attention camera measures what
   it actually sees from the same camera the PiP renders from.

Steps 3 and 4 are the only ones that touch locomotion.  The camera work in
step 5 happens in an isolated ``MjData`` inside :class:`AttentionCamera` and is
never written back, so gaze cannot prop the robot up.

ORDERING NOTE, and it is the subtle one: the machine consumes the camera
verdict from the PREVIOUS tick.  Measuring first and then deciding within one
tick would let a lock be justified by a camera pose that only exists after the
decision.  One control tick of latency at 50 Hz is 20 ms, which is honest and
also what a real perception pipeline would incur.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from attention_camera import AttentionCamera
from contact_geometry import ContactProbe, duck_planar_radius
from people_routes import ADULT_NAMES, crowd_at, pose_crowd
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene, wrap_angle
from recall_model import (
    STANDOFF_TARGET,
    ApproachController,
    Call,
    RecallMachine,
    calls_active_at,
)

# Markers are cosmetic mocap discs; parking them below the floor hides them.
HIDDEN_Z = -0.30


def bearing_sector(bearing_rad: float) -> str:
    """Compass sector of a call bearing, used to prove varied directions."""
    names = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
    index = int(round(math.degrees(bearing_rad) % 360.0 / 45.0)) % 8
    return names[index]


class RecallRollout:
    """One deterministic come-here-recall rollout, with or without rendering."""

    def __init__(self, policy_path: str | Path, seconds: float, *,
                 calls: tuple[Call, ...],
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
            1, int(round((1.0 / CTRL_HZ) / self.model.opt.timestep)))
        self.seconds = float(seconds)
        self.total_steps = int(self.seconds * CTRL_HZ)
        self.dt = 1.0 / CTRL_HZ
        self.calls = calls

        pose_crowd(self.model, self.data, crowd_at(0.0), 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.duck_radius = duck_planar_radius(self.model, self.data, self.trunk)
        # Exact contact geometry, including the analytic box fallback that the
        # MuJoCo mesh-vs-box narrowphase artifact makes necessary.
        self.contacts = ContactProbe(self.model, self.trunk, ADULT_NAMES)

        self.camera = AttentionCamera(
            self.model, self.data, self.runner.qpos_idx, self.trunk, pip_size)
        self.machine = RecallMachine(ctrl_hz=CTRL_HZ)
        self.controller = ApproachController(ctrl_hz=CTRL_HZ)
        self.markers = {
            name: int(self.model.body_mocapid[self.model.body(name).id])
            for name in ("call_ring", "goal_marker", "call_beacon")
        }
        self.records: list[dict] = []
        self.transitions: list[dict] = []
        self.min_trunk_z = float(self.data.xpos[self.trunk][2])
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()
        # The camera verdict the machine will consume on the NEXT tick.
        self._camera_state = {
            "gate_open": False, "subject_visible": False,
            "subject_off_axis_deg": 180.0, "visible": [],
            "off_axis_deg": {name: 180.0 for name in ADULT_NAMES},
            "subject": None, "subject_range_m": float("nan"),
            "view_yaw": 0.0, "view_pitch": 0.0,
            "gaze_yaw": 0.0, "gaze_pitch": 0.0,
        }

    # -- exact contact ---------------------------------------------------
    def min_surface_distance(self, name: str, cutoff: float = 1.2) -> float:
        """Smallest surface separation between the duck and one adult.

        Delegates to :class:`contact_geometry.ContactProbe`, which documents the
        two MuJoCo narrowphase traps this measurement has to survive.
        """
        return self.contacts.distance(self.data, name, cutoff)

    # -- markers ---------------------------------------------------------
    def _place_markers(self, crowd, caller: str | None, locked: str | None,
                       duck_xy: np.ndarray) -> None:
        if caller is None:
            for mocap in self.markers.values():
                self.data.mocap_pos[mocap] = np.array([0.0, 0.0, HIDDEN_Z])
            return
        caller_xy = crowd[caller].pos
        # Ring on the floor under whoever is calling.
        self.data.mocap_pos[self.markers["call_ring"]] = np.array(
            [caller_xy[0], caller_xy[1], 0.008])
        # Beacon floating above them, so the caller is obvious from any angle.
        self.data.mocap_pos[self.markers["call_beacon"]] = np.array(
            [caller_xy[0], caller_xy[1], 0.86])
        if locked is not None:
            delta = duck_xy - caller_xy
            distance = float(np.linalg.norm(delta))
            goal = caller_xy + (STANDOFF_TARGET * delta / max(distance, 1e-9))
            self.data.mocap_pos[self.markers["goal_marker"]] = np.array(
                [goal[0], goal[1], 0.011])
        else:
            self.data.mocap_pos[self.markers["goal_marker"]] = np.array(
                [0.0, 0.0, HIDDEN_Z])

    # -- per-cycle evidence ---------------------------------------------
    def _accumulate(self, state: str, crowd, duck_xy: np.ndarray,
                    duck_yaw: float, camera_state: dict) -> None:
        """Fold this tick's evidence into the open cycle."""
        current = self.machine.current
        locked = self.machine.locked
        if not current or locked is None:
            return
        clearance = self.min_surface_distance(locked)
        current["min_caller_clearance_m"] = min(
            current.get("min_caller_clearance_m", float("inf")), clearance)
        caller_xy = crowd[locked].pos
        caller_range = float(np.linalg.norm(caller_xy - duck_xy))
        if state == "APPROACH":
            current["approach_steps"] = current.get("approach_steps", 0) + 1
            current["approach_visible_steps"] = current.get(
                "approach_visible_steps", 0) + int(camera_state["subject_visible"])
            current["approach_path_m"] = current.get("approach_path_m", 0.0) + float(
                np.linalg.norm(duck_xy - self._previous_xy))
            start = np.asarray(current["approach_start_xy"], dtype=np.float64)
            current["approach_net_m"] = float(np.linalg.norm(duck_xy - start))
            current["approach_min_range_m"] = min(
                current.get("approach_min_range_m", float("inf")), caller_range)
        elif state == "ARRIVED":
            current["arrived_steps"] = current.get("arrived_steps", 0) + 1
            current["arrived_visible_steps"] = current.get(
                "arrived_visible_steps", 0) + int(camera_state["subject_visible"])
            # Facing: how far the duck's TRUNK yaw is from pointing at the
            # caller.  Head gaze is excluded on purpose - the gate is about the
            # physical body's orientation, and gaze lives in an isolated MjData.
            delta = caller_xy - duck_xy
            desired = math.atan2(float(delta[1]), float(delta[0]))
            current["final_facing_error_deg"] = math.degrees(
                wrap_angle(desired - duck_yaw))

    # -- one control tick -----------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        crowd = crowd_at(t)
        sounding = calls_active_at(self.calls, t)
        # Who is calling from the machine's point of view: the call it has
        # committed to, or the earliest one currently sounding.
        active_call = self.machine.active_call
        caller = (
            active_call.caller if active_call is not None
            else (min(sounding, key=lambda c: c.start_s).caller if sounding else None)
        )

        duck_xy_before = self.data.xpos[self.trunk][:2].copy()
        caller_range = (
            float(np.linalg.norm(crowd[caller].pos - duck_xy_before))
            if caller is not None else None
        )

        previous_state = self.machine.state
        previous_locked = self.machine.locked
        # The machine consumes the PREVIOUS tick's camera verdict; see the
        # module docstring for why the measurement cannot come first.
        camera_before = self._camera_state
        state, changed = self.machine.update(
            t,
            calls=self.calls,
            caller_range=caller_range,
            gate_open=bool(camera_before["gate_open"]),
            caller_visible=bool(camera_before["subject_visible"]),
        )
        locked = self.machine.locked

        if changed and state == "CALLER_LOCK":
            # Record WHY this lock was allowed, from the camera verdict that
            # actually authorised it.
            self.machine.current["lock_gate_open"] = bool(camera_before["gate_open"])
            self.machine.current["lock_caller_visible"] = bool(
                camera_before["subject_visible"])
            self.machine.current["lock_off_axis_deg"] = float(
                camera_before["subject_off_axis_deg"])
            self.machine.current["lock_is_active_caller"] = bool(
                self.machine.active_call is not None
                and locked == self.machine.active_call.caller)
            delta = crowd[locked].pos - duck_xy_before
            bearing = math.atan2(float(delta[1]), float(delta[0]))
            self.machine.current["call_bearing_deg"] = math.degrees(bearing)
            self.machine.current["call_sector"] = bearing_sector(bearing)
        if changed and state == "APPROACH":
            self.controller.reset()
            self.machine.current["approach_start_xy"] = [
                float(v) for v in duck_xy_before]
        # A caller change inside an open cycle would invalidate the whole gate.
        if (self.machine.current and locked is not None
                and previous_locked is not None and locked != previous_locked):
            self.machine.current["caller_changed"] = True

        duck_yaw = self.runner.yaw(self.data)
        heading_error = 0.0
        if locked is not None:
            delta = crowd[locked].pos - duck_xy_before
            desired = math.atan2(float(delta[1]), float(delta[0]))
            heading_error = wrap_angle(desired - duck_yaw)
        self._place_markers(crowd, caller, locked, duck_xy_before)

        command = self.controller.update(state, heading_error, caller_range)
        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        display_t = min(t + self.dt, self.seconds)
        display_crowd = crowd_at(display_t)
        duck_pos = self.data.xpos[self.trunk].copy()
        duck_yaw_after = self.runner.yaw(self.data)
        pose_crowd(
            self.model, self.data, display_crowd, display_t,
            caller=caller, duck_xy=duck_pos[:2],
            wave=caller is not None,
        )
        mujoco.mj_forward(self.model, self.data)
        self._place_markers(display_crowd, caller, locked, duck_pos[:2])
        mujoco.mj_forward(self.model, self.data)

        self.min_trunk_z = min(self.min_trunk_z, float(duck_pos[2]))
        camera_state = self.camera.update(
            self.data, state=state, state_elapsed=t - self.machine.state_since,
            duck_yaw=duck_yaw_after, caller=caller, locked=locked)
        self._camera_state = camera_state

        clearances = {
            name: self.min_surface_distance(name) for name in ADULT_NAMES
        }
        nearest = min(clearances, key=clearances.get)
        self._accumulate(state, display_crowd, duck_pos[:2], duck_yaw_after,
                         camera_state)
        self._previous_xy = duck_pos[:2].copy()

        display_range = (
            float(np.linalg.norm(display_crowd[caller].pos - duck_pos[:2]))
            if caller is not None else float("nan")
        )

        if changed:
            self.transitions.append({
                "t": t, "from": previous_state, "to": state,
                "caller": caller, "locked": locked,
                "cycle": self.machine.current.get(
                    "cycle", len(self.machine.cycles) + 1),
            })

        record = {
            "t": display_t,
            "state": state,
            "state_elapsed_s": t - self.machine.state_since,
            "caller": caller,
            "locked": locked,
            "cycle": len(self.machine.cycles) + (1 if self.machine.current else 0),
            "call_active": bool(sounding),
            "caller_range_m": display_range,
            "caller_visible": bool(camera_state["subject_visible"]),
            "gate_open": bool(camera_state["gate_open"]),
            "subject_off_axis_deg": float(camera_state["subject_off_axis_deg"]),
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
            "refused_count": len(self.machine.refused_calls),
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
