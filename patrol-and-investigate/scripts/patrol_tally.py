#!/usr/bin/env python3
"""Everything the rollout ACCUMULATES, in one object with no physics in it.

Split out of ``rollout_patrol`` so the tick loop stays about ORDER - what is
measured before what - and this stays about TALLYING.  The rollout owns one of
these and forwards each tick's measurements into it; ``patrol_summary`` reads it
afterwards.

FOUR ACCUMULATORS HERE ENCODE A DECISION THAT IS EASY TO GET WRONG
--------------------------------------------------------------------
* :meth:`note_zero_episode` records each CONTIGUOUS visit to a zero-command
  state with the path and net displacement it accumulated.  Per EPISODE rather
  than per state, because a state entered five times from a walk legitimately
  settles five times, and summing those into a per-state total then comparing it
  against a single-transient bound compares the wrong quantities.

* :meth:`note_zone` keeps the duck's own smallest distance to the restricted
  rectangle over the WHOLE run, not only during the intrusion investigation.  A
  robot that clipped the zone on an unrelated patrol leg would otherwise pass.

* :meth:`note_standoff` records, per investigation, the closest the duck ever
  came to the body it was observing - measured with the contact probe against
  the real geoms, not derived from the planned standoff.  The band gate compares
  the two, and it can only do that because they are accumulated separately.

* :meth:`note_visibility` conditions on line of sight, so a body genuinely
  behind the central rack does not count against the camera.  The rack is
  0.72 m tall against a 0.20 m eye, so on this scene that exclusion is real
  rather than theoretical.
"""

from __future__ import annotations

import numpy as np


class PatrolTally:
    """Per-run and per-phase accumulators.  No MuJoCo, no policy, no time."""

    def __init__(self, dt: float, initial_trunk_z: float):
        self.dt = float(dt)

        # -- locomotion health ------------------------------------------
        self.min_trunk_z = float(initial_trunk_z)
        self.path_m = 0.0
        self.fallen_steps = 0
        self.contact_steps = 0

        # -- clearance ---------------------------------------------------
        self.min_body_clearance = float("inf")
        self.min_body_name = ""
        self.min_scenery_clearance = float("inf")
        self.min_scenery_geom = ""
        self.min_clearance_by_body: dict[str, float] = {}

        # -- the restricted zone -------------------------------------------
        # The duck's own smallest signed distance to the marked rectangle's
        # edge, over the WHOLE run.  Positive means it stayed outside.
        self.min_zone_gap_m = float("inf")
        self.zone_breach_steps = 0

        # -- per-state ----------------------------------------------------
        self.state_steps: dict[str, int] = {}
        self.state_command_max: dict[str, float] = {}
        self.state_path_m: dict[str, float] = {}
        self.zero_command_violations: list[dict] = []
        # Largest |vy| the controller ever emitted.  There is no strafe on this
        # policy, so this must stay exactly zero for the whole run.
        self.max_abs_vy_command = 0.0

        # -- walking ------------------------------------------------------
        self.walk_path_m = 0.0
        self.walk_steps = 0
        # Path walked during each investigation's APPROACH, per investigation.
        self.approach_path_m: dict[int, float] = {}
        self.return_path_m: dict[int, float] = {}

        # -- zero-command episodes ------------------------------------------
        self.zero_episodes: list[dict] = []
        self._zero_episode: dict | None = None

        # -- zero-command plateaus outside the legitimate states -----------
        self.longest_illegal_zero_run = 0
        self._current_zero_run = 0
        self.illegal_zero_windows: list[dict] = []
        self._zero_window_start: float | None = None

        # -- the investigations ----------------------------------------------
        # index -> the closest MEASURED surface clearance to the observed body.
        self.investigation_min_clearance: dict[int, float] = {}
        self.investigation_min_zone_gap: dict[int, float] = {}

        # -- the circuit ------------------------------------------------------
        self.checkpoint_arrival_error_m: dict[str, float] = {}
        self.reached_home_at_s: float | None = None
        self.home_steps = 0
        self.min_home_distance_m = float("inf")

        # -- visibility, conditioned on line of sight ----------------------
        self.visible_steps = 0
        self.los_steps = 0
        self.visible_with_los = 0
        self.monitor_steps = 0
        self.monitor_los_steps = 0
        self.monitor_visible_with_los = 0
        self.blocked_by: dict[str, int] = {}
        self.subject_steps: dict[str, int] = {}
        self.subject_sequence: list[dict] = []
        self.subject_visible_los: dict[str, int] = {}
        self.subject_los: dict[str, int] = {}
        # Ticks where the camera was ACTIVE on a target: pointing at something
        # it was meant to be watching.  Reported as a fraction of the ticks
        # where line of sight existed.
        self.camera_active_steps = 0

        # -- the interlock ---------------------------------------------------
        self.interlock_holds = 0
        self.interlock_reasons: dict[str, int] = {}

        # -- how populated the facility was ------------------------------------
        self.max_bodies_visible = 0
        self.bodies_ever_seen: set[str] = set()

    # -- locomotion --------------------------------------------------------
    def note_pose(self, trunk_z: float, travelled: float) -> None:
        self.min_trunk_z = min(self.min_trunk_z, float(trunk_z))
        if float(trunk_z) < 0.09:
            self.fallen_steps += 1
        self.path_m += float(travelled)

    def note_command(self, state: str, peak: float, travelled: float,
                     command=None) -> None:
        self.state_command_max[state] = max(
            self.state_command_max.get(state, 0.0), float(peak))
        self.state_steps[state] = self.state_steps.get(state, 0) + 1
        self.state_path_m[state] = \
            self.state_path_m.get(state, 0.0) + float(travelled)
        # MEASURED per tick rather than asserted from the controller's source,
        # because a gate that reads the implementation it is grading cannot fail.
        if command is not None:
            self.max_abs_vy_command = max(
                self.max_abs_vy_command, abs(float(command[1])))

    def note_zero_violation(self, t: float, state: str, command) -> None:
        self.zero_command_violations.append(
            {"t": round(float(t), 3), "state": state,
             "command": [float(v) for v in command]})

    def note_zero_plateau(self, t: float, state: str, peak: float,
                          legitimate: bool) -> None:
        """Count consecutive exact zeros OUTSIDE the states allowed to hold one.

        A stall on a patrol leg would be invisible in a per-state command
        summary - the maximum would still be 0.34 - and shows up only as a run
        of zeros.
        """
        if legitimate or peak != 0.0:
            if self._current_zero_run and self._zero_window_start is not None:
                self.illegal_zero_windows.append({
                    "from_s": round(self._zero_window_start, 3),
                    "to_s": round(t, 3),
                    "steps": self._current_zero_run,
                })
            self._current_zero_run = 0
            self._zero_window_start = None
            return
        if self._current_zero_run == 0:
            self._zero_window_start = float(t)
        self._current_zero_run += 1
        self.longest_illegal_zero_run = max(
            self.longest_illegal_zero_run, self._current_zero_run)

    def note_zero_episode(self, state: str, legitimate: bool, travelled: float,
                          position, t: float) -> None:
        """Accumulate one contiguous visit to a zero-command state."""
        if not legitimate:
            self._close_episode(position, t)
            return
        if self._zero_episode is None or self._zero_episode["state"] != state:
            self._close_episode(position, t)
            self._zero_episode = {
                "state": state, "from_s": round(float(t), 3),
                "to_s": round(float(t), 3), "path_m": 0.0,
                "_start": [float(position[0]), float(position[1])]}
        self._zero_episode["path_m"] += float(travelled)

    def _close_episode(self, position, t: float) -> None:
        if self._zero_episode is None:
            return
        entry = self._zero_episode
        entry["to_s"] = round(float(t), 3)
        entry["net_m"] = round(float(np.linalg.norm(
            np.asarray(position, dtype=float)
            - np.asarray(entry.pop("_start"), dtype=float))), 5)
        entry["path_m"] = round(entry["path_m"], 5)
        self.zero_episodes.append(entry)
        self._zero_episode = None

    def note_walk(self, travelled: float) -> None:
        self.walk_steps += 1
        self.walk_path_m += float(travelled)

    def note_approach(self, index: int, travelled: float) -> None:
        self.approach_path_m[index] = \
            self.approach_path_m.get(index, 0.0) + float(travelled)

    def note_return(self, index: int, travelled: float) -> None:
        self.return_path_m[index] = \
            self.return_path_m.get(index, 0.0) + float(travelled)

    # -- clearance ---------------------------------------------------------
    def note_clearance(self, clearances: dict, nearest: str,
                       scenery_gap: float, scenery_geom: str) -> None:
        if clearances[nearest] < self.min_body_clearance:
            self.min_body_clearance = clearances[nearest]
            self.min_body_name = nearest
        if clearances[nearest] <= 0.0:
            self.contact_steps += 1
        for name, gap in clearances.items():
            current = self.min_clearance_by_body.get(name, float("inf"))
            if gap < current:
                self.min_clearance_by_body[name] = float(gap)
        if scenery_gap < self.min_scenery_clearance:
            self.min_scenery_clearance = scenery_gap
            self.min_scenery_geom = scenery_geom

    def note_zone(self, gap_m: float) -> None:
        """The duck's own distance to the restricted rectangle, every tick."""
        self.min_zone_gap_m = min(self.min_zone_gap_m, float(gap_m))
        if float(gap_m) < 0.0:
            self.zone_breach_steps += 1

    def note_standoff(self, index: int, clearance_m: float,
                      zone_gap_m: float) -> None:
        """The closest the duck came to the body it was observing."""
        current = self.investigation_min_clearance.get(index, float("inf"))
        self.investigation_min_clearance[index] = min(
            current, float(clearance_m))
        current_zone = self.investigation_min_zone_gap.get(index, float("inf"))
        self.investigation_min_zone_gap[index] = min(
            current_zone, float(zone_gap_m))

    # -- the circuit --------------------------------------------------------
    def note_checkpoint(self, name: str, error_m: float) -> None:
        self.checkpoint_arrival_error_m[name] = float(error_m)

    def note_home(self, *, distance_m: float, inside: bool, t: float) -> None:
        self.min_home_distance_m = min(self.min_home_distance_m,
                                       float(distance_m))
        if inside:
            self.home_steps += 1
            if self.reached_home_at_s is None:
                self.reached_home_at_s = float(t)

    # -- visibility -----------------------------------------------------------
    def note_visibility(self, *, subject: str, visible: bool, los_ok: bool,
                        monitoring: bool, blocker: str, t: float,
                        active: bool) -> None:
        key = subject or "route"
        self.subject_steps[key] = self.subject_steps.get(key, 0) + 1
        if not self.subject_sequence or \
                self.subject_sequence[-1]["subject"] != key:
            self.subject_sequence.append(
                {"subject": key, "from_s": round(float(t), 3),
                 "to_s": round(float(t), 3)})
        else:
            self.subject_sequence[-1]["to_s"] = round(float(t), 3)

        if active:
            self.camera_active_steps += 1
        if visible:
            self.visible_steps += 1
        if los_ok:
            self.los_steps += 1
            self.subject_los[key] = self.subject_los.get(key, 0) + 1
            if visible:
                self.visible_with_los += 1
                self.subject_visible_los[key] = \
                    self.subject_visible_los.get(key, 0) + 1
        if monitoring:
            self.monitor_steps += 1
            if los_ok:
                self.monitor_los_steps += 1
                if visible:
                    self.monitor_visible_with_los += 1
        if not visible and subject:
            reason = blocker or "out_of_frustum"
            self.blocked_by[reason] = self.blocked_by.get(reason, 0) + 1

    def note_seen(self, visible_bodies: list[str]) -> None:
        self.max_bodies_visible = max(self.max_bodies_visible,
                                      len(visible_bodies))
        self.bodies_ever_seen.update(visible_bodies)

    # -- the interlock --------------------------------------------------------
    def note_interlock(self, blocked: bool, reason: str) -> None:
        if not blocked:
            return
        self.interlock_holds += 1
        self.interlock_reasons[reason] = \
            self.interlock_reasons.get(reason, 0) + 1

    def close(self, t: float, position=None) -> None:
        """Flush any open zero-plateau window and zero-episode at the run's end."""
        if self._current_zero_run and self._zero_window_start is not None:
            self.illegal_zero_windows.append({
                "from_s": round(self._zero_window_start, 3),
                "to_s": round(float(t), 3),
                "steps": self._current_zero_run,
            })
            self._current_zero_run = 0
            self._zero_window_start = None
        if self._zero_episode is not None and position is not None:
            self._close_episode(position, t)
