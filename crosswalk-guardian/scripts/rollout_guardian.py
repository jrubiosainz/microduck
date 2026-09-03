#!/usr/bin/env python3
"""Integration layer: traffic + guardian machine + camera + walking policy.

This is the only module that owns all four concerns at once, and it owns them
in a strict order every control tick:

1. the road users are posed analytically at ``t`` (scripted, non-colliding);
2. the guardian machine advances on the duck's position and the CAMERA's
   sector verdict from the previous tick;
3. the controller emits a command from the state and the trunk pose;
4. the walking policy consumes that command and physics is stepped;
5. the road users are re-posed at ``t + dt`` and the camera measures what it
   actually sees from the same camera the PiP renders from.

Steps 3 and 4 are the only ones that touch locomotion.  The camera work in
step 5 happens in an isolated ``MjData`` inside :class:`GuardianCamera` and is
never written back, so gaze cannot prop the robot up.

ORDERING NOTE, and it is the subtle one: the machine consumes the camera
verdict from the PREVIOUS tick.  Measuring first and then deciding within one
tick would let a scan phase be satisfied by a camera pose that only exists
after the decision.  One control tick at 50 Hz is 20 ms, which is honest and is
also what a real perception pipeline would incur.

The gap decision is likewise taken on the PREVIOUS tick's traffic state, for
the same reason: a decision justified by where the cars will be after the
decision is not a decision.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from conflict import evaluate_gap
from contact_geometry import ContactProbe, duck_planar_radius
from guardian_camera import GuardianCamera
from guardian_model import LOOK_SECTOR, GuardianController, GuardianMachine
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene
from street import (
    CURB_STOP_X,
    DUCK_PLANAR_RADIUS,
    START_X,
    START_Y,
    WAIT_LINE_X,
    encroaches_wait_line,
    in_lane,
    in_road,
    in_safe_zone,
)
from traffic import VEHICLE_NAMES, pose_traffic, traffic_at


class GuardianRollout:
    """One deterministic crosswalk-guardian rollout, with or without rendering."""

    def __init__(self, policy_path: str | Path, seconds: float, *,
                 scene: str | Path | None = None,
                 robot_dir: str | Path | None = None,
                 pip_size: tuple[int, int] = (300, 220)):
        self.model = load_scene(scene, robot_dir)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        # The scene's STAND keyframe already starts the duck on the near
        # pavement, but mj_resetData restores qpos0, so the start pose is set
        # here explicitly rather than depending on which reset was used.
        self.data.qpos[0], self.data.qpos[1] = START_X, START_Y
        self.policy = PolicyRunner(policy_path)
        self.runner = self.policy.reset(self.model, self.data)
        self.trunk = self.model.body("trunk_base").id
        self.decimation = max(
            1, int(round((1.0 / CTRL_HZ) / self.model.opt.timestep)))
        self.seconds = float(seconds)
        self.total_steps = int(self.seconds * CTRL_HZ)
        self.dt = 1.0 / CTRL_HZ

        pose_traffic(self.model, self.data, traffic_at(0.0), 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.duck_radius = duck_planar_radius(self.model, self.data, self.trunk)
        self.contacts = ContactProbe(self.model, self.trunk, VEHICLE_NAMES)

        self.camera = GuardianCamera(
            self.model, self.data, self.runner.qpos_idx, self.trunk, pip_size)
        self.machine = GuardianMachine(ctrl_hz=CTRL_HZ)
        self.controller = GuardianController(ctrl_hz=CTRL_HZ)

        self.records: list[dict] = []
        self.transitions: list[dict] = []
        self.min_trunk_z = float(self.data.xpos[self.trunk][2])
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()
        self.path_m = 0.0
        self.crossing_path_m = 0.0
        self.crossing_start_xy: np.ndarray | None = None
        # Worst (smallest) encroachment margin before CROSSING: how close the
        # duck's leading surface got to the wait line while still forbidden to
        # pass it.  Positive means it never encroached.
        self.min_wait_line_margin = float("inf")
        self.max_x_before_crossing = float(self.data.xpos[self.trunk][0])
        # The camera verdict the machine will consume on the NEXT tick.
        self._camera_state = {
            "sectors": {"left": {"fraction": 0.0, "visible": False},
                        "right": {"fraction": 0.0, "visible": False}},
            "left_visible": False, "right_visible": False,
            "left_fraction": 0.0, "right_fraction": 0.0,
            "visible_vehicles": [], "view_yaw": 0.0, "view_pitch": 0.0,
            "gaze_yaw": 0.0, "gaze_pitch": 0.0, "target_yaw": 0.0,
        }
        # The traffic the machine will judge on the NEXT tick.
        self._previous_traffic = traffic_at(0.0)

    # -- exact contact ---------------------------------------------------
    def min_surface_distance(self, name: str, cutoff: float = 1.5) -> float:
        """Smallest surface separation between the duck and one road user."""
        return self.contacts.distance(self.data, name, cutoff)

    # -- one control tick -----------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        traffic = traffic_at(t)
        pos_before = self.data.xpos[self.trunk].copy()
        x_before = float(pos_before[0])
        previous_state = self.machine.state

        # The machine consumes the PREVIOUS tick's camera verdict and the
        # PREVIOUS tick's traffic; see the module docstring for why neither
        # measurement may come first.
        camera_before = self._camera_state
        sector_needed = LOOK_SECTOR.get(previous_state)
        sector_visible = (
            bool(camera_before["sectors"][sector_needed]["visible"])
            if sector_needed is not None else False
        )
        decision = None
        if previous_state == "WAIT_FOR_GAP":
            decision = evaluate_gap(
                self._previous_traffic, pos_before[:2], start_x=x_before)

        state, changed = self.machine.update(
            t, trunk_x=x_before, sector_visible=sector_visible,
            decision=decision)

        if changed and state == "CROSSING":
            self.controller.reset()
            self.crossing_start_xy = pos_before[:2].copy()

        duck_yaw = self.runner.yaw(self.data)
        command = self.controller.update(
            state, x_before, duck_yaw, float(pos_before[1]))
        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        display_t = min(t + self.dt, self.seconds)
        display_traffic = traffic_at(display_t)
        pose_traffic(self.model, self.data, display_traffic, display_t)
        mujoco.mj_forward(self.model, self.data)

        duck_pos = self.data.xpos[self.trunk].copy()
        duck_yaw_after = self.runner.yaw(self.data)
        duck_x = float(duck_pos[0])
        self.min_trunk_z = min(self.min_trunk_z, float(duck_pos[2]))
        travelled = float(np.linalg.norm(duck_pos[:2] - self._previous_xy))
        self.path_m += travelled
        if state == "CROSSING":
            self.crossing_path_m += travelled

        # Encroachment: BEFORE committing, the duck's leading surface must stay
        # behind the near wait line.  Graded on the inflated footprint, not the
        # trunk centre.
        if not self.machine.committed:
            leading = duck_x + DUCK_PLANAR_RADIUS
            self.min_wait_line_margin = min(
                self.min_wait_line_margin, -WAIT_LINE_X - leading)
            self.max_x_before_crossing = max(self.max_x_before_crossing, duck_x)

        camera_state = self.camera.update(
            self.data, state=state, duck_yaw=duck_yaw_after, t=display_t)
        self._camera_state = camera_state
        self._previous_traffic = display_traffic

        clearances = {
            name: self.min_surface_distance(name) for name in VEHICLE_NAMES}
        nearest = min(clearances, key=clearances.get)
        self._previous_xy = duck_pos[:2].copy()

        if changed:
            self.transitions.append({
                "t": t, "from": previous_state, "to": state,
                "trunk_x": duck_x,
            })

        # The gap arithmetic is recomputed for DISPLAY on every tick, so the
        # HUD can show a live time-to-conflict even outside WAIT_FOR_GAP.  It
        # is never fed back into the machine.
        display_decision = evaluate_gap(
            display_traffic, duck_pos[:2], start_x=duck_x)

        record = {
            "t": display_t,
            "state": state,
            "state_elapsed_s": t - self.machine.state_since,
            "command": [float(v) for v in command],
            "duck_xy": [float(duck_pos[0]), float(duck_pos[1])],
            "duck_yaw_deg": math.degrees(duck_yaw_after),
            "trunk_z_m": float(duck_pos[2]),
            "min_trunk_z_m": self.min_trunk_z,
            "in_road": in_road(duck_x),
            "in_near_lane": in_lane(duck_x, "near"),
            "in_far_lane": in_lane(duck_x, "far"),
            "in_safe_zone": in_safe_zone(duck_x),
            "encroaches": encroaches_wait_line(duck_x),
            "wait_line_margin_m": -WAIT_LINE_X - (duck_x + DUCK_PLANAR_RADIUS),
            "left_visible": bool(camera_state["left_visible"]),
            "right_visible": bool(camera_state["right_visible"]),
            "left_fraction": float(camera_state["left_fraction"]),
            "right_fraction": float(camera_state["right_fraction"]),
            "visible_vehicles": list(camera_state["visible_vehicles"]),
            "view_yaw_deg": math.degrees(camera_state["view_yaw"]),
            "gaze_yaw_deg": math.degrees(camera_state["gaze_yaw"]),
            "gap_safe": bool(display_decision.safe),
            "gap_margin_s": float(display_decision.worst_margin_s),
            "gap_limiting": display_decision.limiting_vehicle,
            "crossing_estimate_s": float(display_decision.crossing_duration_s),
            "vehicle_windows": {
                c.name: [c.vehicle_window.start, c.vehicle_window.end]
                for c in display_decision.conflicts if not c.vehicle_window.empty
            },
            "duck_windows": {
                lane: [
                    display_decision.conflicts[0].duck_window.start
                    if display_decision.conflicts else 0.0,
                    display_decision.conflicts[0].duck_window.end
                    if display_decision.conflicts else 0.0,
                ]
                for lane in ("near",)
            },
            "nearest_vehicle": nearest,
            "nearest_clearance_m": float(clearances[nearest]),
            "path_m": self.path_m,
            "crossing_path_m": self.crossing_path_m,
            "rejected_gaps": len(self.machine.rejected_gaps),
            "vehicle_xy": {
                name: [float(v.pos[0]), float(v.pos[1])]
                for name, v in display_traffic.items()
            },
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

    # -- derived evidence -------------------------------------------------
    @property
    def crossing_net_m(self) -> float:
        """Net displacement over the CROSSING phase."""
        if self.crossing_start_xy is None:
            return 0.0
        crossing = [r for r in self.records if r["state"] == "CROSSING"]
        if not crossing:
            return 0.0
        end = np.array(crossing[-1]["duck_xy"], dtype=np.float64)
        return float(np.linalg.norm(end - self.crossing_start_xy))
