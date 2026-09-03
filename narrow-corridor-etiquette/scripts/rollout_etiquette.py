#!/usr/bin/env python3
"""Integration layer: people + etiquette machine + camera + walking policy.

This is the only module that owns all four concerns at once, and it owns them
in a strict order every control tick:

1. the adults are posed analytically at ``t`` (scripted, non-colliding);
2. the etiquette machine advances on the duck's pose and the PREVIOUS tick's
   encounter prediction and camera verdict;
3. the controller emits a command from the state, the trunk pose and the target;
4. the walking policy consumes that command and physics is stepped;
5. the adults are re-posed at ``t + dt`` and the camera measures what it
   actually sees from the same camera the PiP renders from.

Steps 3 and 4 are the only ones that touch locomotion.  The camera work in
step 5 happens in an isolated ``MjData`` inside :class:`EtiquetteCamera` and is
never written back, so gaze cannot prop the robot up.

ORDERING NOTE, and it is the subtle one: the machine consumes the encounter
prediction and the camera verdict from the PREVIOUS tick.  Measuring first and
then deciding within one tick would let a pull-over be authorised, or a yield
released, by a state that only exists after the decision.  One control tick at
50 Hz is 20 ms, which is honest and is also what a real perception pipeline
would incur.
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
    exact_lateral_half_width,
    exact_planar_radius,
)
from corridor import (
    ALCOVE_BY_NAME,
    ALCOVE_NAMES,
    CENTER_PASSAGE_HALF,
    DESTINATION_X,
    DUCK_PLANAR_RADIUS,
    START_X,
    START_Y,
    at_destination,
    center_passage_intrusion,
    clears_center_passage,
    wall_clearance,
)
from encounter import (
    CRUISE_SPEED_MPS,
    UNSAFE_PROXIMITY_M,
    choose_alcove,
    most_urgent,
    predict_encounters,
)
from etiquette_camera import EtiquetteCamera
from etiquette_model import EtiquetteController, EtiquetteMachine
from people import PERSON_NAMES, people_at, pose_people
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene

# Every wall, cheek, crate and door geom the clearance gate is measured
# against.  Built from the scene's own naming rather than hand-listed, so a
# scene change cannot silently drop a surface from the gate.
WALL_GEOM_PREFIXES = ("wall_", "bay_", "lobby_back")
WALL_GEOM_EXCLUDE = ("_floor",)


def wall_geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Names of every scenery geom the duck must not touch.

    The floor patches inside each recess are excluded: they are paint, and the
    duck is standing on them.  Everything else that bounds the corridor — the
    wall segments, the alcove backs and cheeks, the crates and the lobby's back
    wall — is a surface a real robot would hit.
    """
    names: list[str] = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if not name:
            continue
        if any(part in name for part in WALL_GEOM_EXCLUDE):
            continue
        if name.startswith(WALL_GEOM_PREFIXES):
            names.append(name)
    if not names:
        raise RuntimeError("no wall geoms found; the clearance gate is vacuous")
    return tuple(names)


class EtiquetteRollout:
    """One deterministic narrow-corridor-etiquette rollout, with or without rendering."""

    def __init__(self, policy_path: str | Path, seconds: float, *,
                 scene: str | Path | None = None,
                 robot_dir: str | Path | None = None,
                 pip_size: tuple[int, int] = (300, 220)):
        self.model = load_scene(scene, robot_dir)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        # The scene's STAND keyframe already starts the duck at the corridor
        # mouth, but mj_resetData restores qpos0, so the start pose is set here
        # explicitly rather than depending on which reset was used.
        self.data.qpos[0], self.data.qpos[1] = START_X, START_Y
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
        self.duck_radius = duck_planar_radius(
            self.model, self.data, self.trunk)
        self.duck_exact_radius = exact_planar_radius(
            self.model, self.data, self.trunk)
        self.duck_lateral_half = exact_lateral_half_width(
            self.model, self.data, self.trunk)
        person_body = self.model.body(f"person_{PERSON_NAMES[0]}").id
        self.adult_lateral_half = exact_lateral_half_width(
            self.model, self.data, person_body)
        self.adult_exact_radius = exact_planar_radius(
            self.model, self.data, person_body)

        self.contacts = ContactProbe(self.model, self.trunk, PERSON_NAMES)
        self.wall_geoms = wall_geom_names(self.model)
        self.walls = WallProbe(self.model, self.trunk, self.wall_geoms)

        self.camera = EtiquetteCamera(
            self.model, self.data, self.runner.qpos_idx, self.trunk, pip_size)
        self.machine = EtiquetteMachine(ctrl_hz=CTRL_HZ)
        self.controller = EtiquetteController(ctrl_hz=CTRL_HZ)

        self.records: list[dict] = []
        self.transitions: list[dict] = []
        self.min_trunk_z = float(self.data.xpos[self.trunk][2])
        self._previous_xy = self.data.xpos[self.trunk][:2].copy()
        self.path_m = 0.0
        self.forward_progress_m = 0.0
        self.max_x = float(self.data.xpos[self.trunk][0])
        self.min_wall_clearance = float("inf")
        self.min_wall_geom = ""
        self.min_person_clearance = float("inf")
        # Per-phase accumulators, keyed by cycle index.
        self.pull_over_path: dict[int, float] = {}
        self.rejoin_path: dict[int, float] = {}
        self.yield_tracking: dict[int, list[bool]] = {}
        # Tracking restricted to the window the duck can physically see out of
        # its recess; see the sightline note in ``corridor.Alcove``.
        self.yield_tracking_in_view: dict[int, list[bool]] = {}
        self.yield_command_max: dict[int, float] = {}
        self.yield_person_side: dict[int, list[float]] = {}
        self.detect_proximity: dict[int, float] = {}

        # The camera verdict and the encounter prediction the machine will
        # consume on the NEXT tick.
        self._camera_state = self.camera.update(
            self.data, state="CRUISE",
            duck_yaw=self.runner.yaw(self.data),
            duck_pos=self.data.xpos[self.trunk], tracked=None, t=0.0)
        self._previous_people = people_at(0.0)
        self._previous_range: dict[str, float] = {}

    # -- exact clearance -------------------------------------------------
    def person_distance(self, name: str, cutoff: float = 1.5) -> float:
        """Smallest surface separation between the duck and one adult."""
        return self.contacts.distance(self.data, name, cutoff)

    def wall_distance(self, cutoff: float = 1.0) -> tuple[float, str]:
        """Smallest surface separation between the duck and any wall geom."""
        return self.walls.distance(self.data, cutoff)

    # -- one control tick -----------------------------------------------
    def step(self, index: int) -> dict:
        t = index * self.dt
        pos_before = self.data.xpos[self.trunk].copy()
        x_before, y_before = float(pos_before[0]), float(pos_before[1])
        previous_state = self.machine.state
        speed_before = float(np.linalg.norm(
            pos_before[:2] - self._previous_xy)) / self.dt

        # The machine consumes the PREVIOUS tick's people state; see the module
        # docstring for why the measurement may not come first.
        people = self._previous_people
        encounter = None
        if previous_state in ("CRUISE", "RESUME"):
            encounter = most_urgent(
                people, (x_before, y_before), CRUISE_SPEED_MPS)

        # Range, side and closing sense for the adult being yielded to.
        tracked_name = self.machine.yielding_to or self.machine._cycle.get(
            "person") if self.machine._cycle else self.machine.yielding_to
        person_range = person_receding = person_behind = None
        if tracked_name is not None and tracked_name in people:
            person = people[tracked_name]
            person_range = self.person_distance(tracked_name, cutoff=6.0)
            previous_range = self._previous_range.get(tracked_name)
            person_receding = (
                previous_range is not None and person_range > previous_range)
            # "Behind" means the adult has passed the duck's station in its own
            # direction of travel, which is the only definition that works for
            # both a head-on and an overtaking encounter.
            person_behind = (
                float(person.pos[0]) - x_before) * float(person.direction) > 0.0
            self._previous_range[tracked_name] = person_range

        state, changed = self.machine.update(
            t, duck_xy=(x_before, y_before), duck_speed_mps=speed_before,
            encounter=encounter, person_range_m=person_range,
            person_receding=person_receding, person_behind=person_behind)

        if changed and state == "PULL_OVER":
            self.controller.reset()

        duck_yaw = self.runner.yaw(self.data)
        command = self.controller.update(
            state, x_before, y_before, duck_yaw,
            park_y=self.machine.park_y, target_x=self.machine.target_x,
            alcove_name=self.machine.target)
        self.runner.step(self.data, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        display_t = min(t + self.dt, self.seconds)
        display_people = people_at(display_t)
        pose_people(self.model, self.data, display_people, display_t)
        mujoco.mj_forward(self.model, self.data)

        duck_pos = self.data.xpos[self.trunk].copy()
        duck_yaw_after = self.runner.yaw(self.data)
        duck_x, duck_y = float(duck_pos[0]), float(duck_pos[1])
        self.min_trunk_z = min(self.min_trunk_z, float(duck_pos[2]))
        travelled = float(np.linalg.norm(duck_pos[:2] - self._previous_xy))
        self.path_m += travelled
        self.forward_progress_m += max(0.0, duck_x - float(pos_before[0]))
        self.max_x = max(self.max_x, duck_x)

        cycle_index = len(self.machine.cycles) + 1
        if state == "PULL_OVER":
            self.pull_over_path[cycle_index] = (
                self.pull_over_path.get(cycle_index, 0.0) + travelled)
        elif state == "REJOIN":
            self.rejoin_path[cycle_index] = (
                self.rejoin_path.get(cycle_index, 0.0) + travelled)

        # The adult the duck should be watching, for the camera.
        tracked_xy = None
        if tracked_name is not None and tracked_name in display_people:
            tracked_xy = display_people[tracked_name].pos

        camera_state = self.camera.update(
            self.data, state=state, duck_yaw=duck_yaw_after,
            duck_pos=duck_pos, tracked=tracked_xy, t=display_t)
        self._camera_state = camera_state
        self._previous_people = display_people

        clearances = {
            name: self.person_distance(name) for name in PERSON_NAMES}
        nearest = min(clearances, key=clearances.get)
        self.min_person_clearance = min(
            self.min_person_clearance, clearances[nearest])
        wall_gap, wall_geom = self.wall_distance()
        if wall_gap < self.min_wall_clearance:
            self.min_wall_clearance = wall_gap
            self.min_wall_geom = wall_geom

        if state == "YIELD":
            seen = bool(camera_state["people"].get(
                tracked_name, {}).get("visible", False)) if tracked_name else False
            self.yield_tracking.setdefault(cycle_index, []).append(seen)
            self.yield_command_max[cycle_index] = max(
                self.yield_command_max.get(cycle_index, 0.0),
                float(np.max(np.abs(command))))
            if tracked_name in display_people:
                offset = float(display_people[tracked_name].pos[0]) - duck_x
                self.yield_person_side.setdefault(cycle_index, []).append(offset)
                # Tracking is ALSO recorded over the window the duck can
                # physically see out of its recess.  An adult beyond the mouth's
                # sightline is behind an opaque wall, and a gate that demanded
                # the duck see through it would be unsatisfiable by any robot.
                alcove = ALCOVE_BY_NAME.get(self.machine.target or "")
                span = (alcove.sightline_half_span_m if alcove
                        else float("inf"))
                if abs(offset) <= span:
                    self.yield_tracking_in_view.setdefault(
                        cycle_index, []).append(seen)

        if changed:
            self.transitions.append({
                "t": t, "from": previous_state, "to": state,
                "trunk_x": duck_x, "trunk_y": duck_y,
            })
            if state == "DETECT":
                self.detect_proximity[cycle_index] = min(
                    clearances.values())

        self._previous_xy = duck_pos[:2].copy()

        # Live prediction for the HUD.  Never fed back into the machine.
        display_encounters = predict_encounters(
            display_people, (duck_x, duck_y), CRUISE_SPEED_MPS)
        soonest = display_encounters[0] if display_encounters else None
        # The live scorecard the overlay draws.  Recomputed for DISPLAY on
        # every tick from the SAME scorer the machine used, so the panel can
        # never show a different verdict from the one the decision was taken
        # on; it is never fed back.
        alcove_scores = []
        if soonest is not None and soonest.approaching:
            alcove_scores = [
                score.as_record()
                for score in choose_alcove(
                    soonest, (duck_x, duck_y)).candidates
            ]

        intrusion = center_passage_intrusion(duck_y)
        record = {
            "t": display_t,
            "state": state,
            "state_elapsed_s": t - self.machine.state_since,
            "cycle": cycle_index,
            "command": [float(v) for v in command],
            "duck_xy": [duck_x, duck_y],
            "duck_yaw_deg": math.degrees(duck_yaw_after),
            "trunk_z_m": float(duck_pos[2]),
            "min_trunk_z_m": self.min_trunk_z,
            "clears_passage": clears_center_passage(duck_y),
            "passage_intrusion_m": intrusion,
            "wall_clearance_geometric_m": wall_clearance(duck_x, duck_y),
            "wall_clearance_m": wall_gap,
            "wall_nearest_geom": wall_geom,
            "at_destination": at_destination(duck_x),
            "destination_remaining_m": max(0.0, DESTINATION_X - duck_x),
            "target_alcove": self.machine.target,
            "target_park_y": self.machine.park_y if self.machine.target else None,
            "tracked_person": tracked_name,
            "tracked_visible": bool(
                camera_state["people"].get(tracked_name, {}).get(
                    "visible", False)) if tracked_name else False,
            "tracked_fraction": float(
                camera_state["people"].get(tracked_name, {}).get(
                    "fraction", 0.0)) if tracked_name else 0.0,
            "visible_people": list(camera_state["visible_people"]),
            "view_yaw_deg": math.degrees(camera_state["view_yaw"]),
            "gaze_yaw_deg": math.degrees(camera_state["gaze_yaw"]),
            "nearest_person": nearest,
            "nearest_clearance_m": float(clearances[nearest]),
            "person_clearances": {k: float(v) for k, v in clearances.items()},
            "person_xy": {
                name: [float(p.pos[0]), float(p.pos[1])]
                for name, p in display_people.items()
            },
            "person_moving": {
                name: bool(p.moving) for name, p in display_people.items()},
            "soonest_person": soonest.name if soonest else None,
            "soonest_time_to_meet_s": (
                soonest.time_to_meet_s if soonest else float("inf")),
            "soonest_range_m": soonest.range_m if soonest else float("inf"),
            "soonest_counterfactual_m": (
                soonest.counterfactual_clearance_m if soonest else 0.0),
            "alcove_scores": alcove_scores,
            "path_m": self.path_m,
            "completed_cycles": len(self.machine.cycles),
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
