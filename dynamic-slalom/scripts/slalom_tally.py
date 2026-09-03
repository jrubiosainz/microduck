#!/usr/bin/env python3
"""Everything the rollout ACCUMULATES, in one object with no physics in it.

Split out of ``rollout_slalom`` so that the tick loop stays about ORDER — what
is measured before what — and this stays about TALLYING.  The rollout owns one
of these and forwards each tick's measurements into it; ``slalom_summary`` reads
it afterwards.

FOUR ACCUMULATORS HERE ENCODE A DECISION THAT IS EASY TO GET WRONG
--------------------------------------------------------------------
* :meth:`note_prediction` keeps, per encounter, BOTH the worst clearance the
  planner PREDICTED for the corridor it chose and the worst clearance that was
  actually MEASURED afterwards.  The bracketing gate compares the two, and it
  can only do that because they are accumulated separately rather than one being
  derived from the other.

* :meth:`note_command` records per-state path length, which is what turns "it
  held still while waiting" into a distance on the floor rather than a claim
  about the command.

* :meth:`note_lateral` tracks the duck's signed offset from the nominal lane,
  keeping the extremes on BOTH hands.  A duck that dodged only left would show a
  positive maximum and a negative minimum near zero, which is exactly the
  failure the alternation gate exists to catch.

* :meth:`note_zero_plateau` counts consecutive exactly-zero-command ticks
  OUTSIDE the states where a zero is legitimate.  A behavior that stalled
  mid-course would look fine in a per-state summary and shows up here.
"""

from __future__ import annotations

import numpy as np


class RolloutTally:
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

        # -- the lane, on both hands ---------------------------------------
        self.max_left_offset_m = -float("inf")
        self.max_right_offset_m = float("inf")
        self.max_abs_offset_m = 0.0
        self.lateral_path_m = 0.0

        # -- zero-command episodes ------------------------------------------
        # Each CONTIGUOUS visit to a zero-command state, with the path and net
        # displacement it accumulated.  Per EPISODE rather than per state,
        # because a state entered three times from a walk legitimately settles
        # three times; see ``slalom_thresholds`` for the measurement.
        self.zero_episodes: list[dict] = []
        self._zero_episode: dict | None = None

        # -- zero-command plateaus outside the legitimate states -----------
        self.longest_illegal_zero_run = 0
        self._current_zero_run = 0
        self.illegal_zero_windows: list[dict] = []
        self._zero_window_start: float | None = None

        # -- predictions versus what happened -------------------------------
        # encounter index -> {"predicted_m", "measured_m", "side", "threat"}
        self.predictions: dict[int, dict] = {}

        # -- the goal --------------------------------------------------------
        self.goal_steps = 0
        self.goal_visible_steps = 0
        self.goal_los_steps = 0
        self.goal_visible_with_los = 0
        self.reached_goal_at_s: float | None = None
        self.min_goal_distance_m = float("inf")

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

        # -- the interlock ---------------------------------------------------
        self.interlock_holds = 0
        self.interlock_reasons: dict[str, int] = {}

        # -- how much traffic was actually in the way ------------------------
        self.max_bodies_in_lane = 0
        self.lane_occupied_steps = 0

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
        # The largest lateral command ever emitted.  MEASURED per tick rather
        # than asserted from the controller's source, because a gate that reads
        # the implementation it is grading cannot fail.
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

        A stall in ADVANCE would be invisible in a per-state command summary —
        the maximum would still be 0.34 — and shows up only as a run of zeros.
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
        """Accumulate one contiguous visit to a zero-command state.

        Opened when such a state is entered and closed when it is left, so the
        stillness gate can grade each settling transient separately instead of
        summing several into a total that no single-transient bound can match.
        """
        import numpy as _np

        if not legitimate:
            if self._zero_episode is not None:
                entry = self._zero_episode
                entry["to_s"] = round(float(t), 3)
                entry["net_m"] = round(float(_np.linalg.norm(
                    _np.asarray(position, dtype=float)
                    - _np.asarray(entry.pop("_start"), dtype=float))), 5)
                entry["path_m"] = round(entry["path_m"], 5)
                self.zero_episodes.append(entry)
                self._zero_episode = None
            return
        if self._zero_episode is None or self._zero_episode["state"] != state:
            if self._zero_episode is not None:
                entry = self._zero_episode
                entry["to_s"] = round(float(t), 3)
                entry["net_m"] = round(float(_np.linalg.norm(
                    _np.asarray(position, dtype=float)
                    - _np.asarray(entry.pop("_start"), dtype=float))), 5)
                entry["path_m"] = round(entry["path_m"], 5)
                self.zero_episodes.append(entry)
            self._zero_episode = {
                "state": state, "from_s": round(float(t), 3),
                "to_s": round(float(t), 3), "path_m": 0.0,
                "_start": [float(position[0]), float(position[1])]}
        self._zero_episode["path_m"] += float(travelled)

    def note_walk(self, travelled: float) -> None:
        self.walk_steps += 1
        self.walk_path_m += float(travelled)

    # -- the lane ------------------------------------------------------------
    def note_lateral(self, offset_m: float, lateral_travel: float) -> None:
        """The duck's signed offset from the lane, and its lateral path length.

        Both hands are kept because the alternation claim is about the duck
        having genuinely gone BOTH ways: a maximum alone cannot distinguish a
        slalom from a single wide dodge.
        """
        value = float(offset_m)
        self.max_left_offset_m = max(self.max_left_offset_m, value)
        self.max_right_offset_m = min(self.max_right_offset_m, value)
        self.max_abs_offset_m = max(self.max_abs_offset_m, abs(value))
        self.lateral_path_m += abs(float(lateral_travel))

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

    # -- predictions ---------------------------------------------------------
    def note_prediction(self, index: int, *, threat: str, side: str,
                        predicted_m: float) -> None:
        """The clearance the planner PREDICTED for the corridor it committed to.

        Recorded once per encounter, at the moment of commitment, so a later
        re-score cannot overwrite the number the decision was actually made on.
        """
        if index in self.predictions:
            return
        self.predictions[index] = {
            "threat": threat, "side": side,
            "predicted_m": float(predicted_m),
            "measured_m": float("inf"),
        }

    def note_measured_for(self, index: int, clearance_m: float) -> None:
        """The worst clearance actually MEASURED during that encounter."""
        entry = self.predictions.get(index)
        if entry is None:
            return
        entry["measured_m"] = min(entry["measured_m"], float(clearance_m))

    # -- the goal -------------------------------------------------------------
    def note_goal(self, *, distance_m: float, inside: bool, visible: bool,
                  los_ok: bool, t: float) -> None:
        self.min_goal_distance_m = min(self.min_goal_distance_m,
                                       float(distance_m))
        if inside:
            self.goal_steps += 1
            if self.reached_goal_at_s is None:
                self.reached_goal_at_s = float(t)
        if visible:
            self.goal_visible_steps += 1
        if los_ok:
            self.goal_los_steps += 1
            if visible:
                self.goal_visible_with_los += 1

    # -- traffic --------------------------------------------------------------
    def note_lane(self, bodies_in_lane: list[str]) -> None:
        self.max_bodies_in_lane = max(self.max_bodies_in_lane,
                                      len(bodies_in_lane))
        if bodies_in_lane:
            self.lane_occupied_steps += 1

    # -- visibility -----------------------------------------------------------
    def note_visibility(self, *, subject: str, visible: bool, los_ok: bool,
                        monitoring: bool, blocker: str, t: float) -> None:
        key = subject or "goal"
        self.subject_steps[key] = self.subject_steps.get(key, 0) + 1
        if not self.subject_sequence or \
                self.subject_sequence[-1]["subject"] != key:
            self.subject_sequence.append(
                {"subject": key, "from_s": round(float(t), 3),
                 "to_s": round(float(t), 3)})
        else:
            self.subject_sequence[-1]["to_s"] = round(float(t), 3)

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
        if not visible:
            reason = blocker or "out_of_frustum"
            self.blocked_by[reason] = self.blocked_by.get(reason, 0) + 1

    # -- the interlock --------------------------------------------------------
    def note_interlock(self, blocked: bool, reason: str) -> None:
        if not blocked:
            return
        self.interlock_holds += 1
        self.interlock_reasons[reason] = \
            self.interlock_reasons.get(reason, 0) + 1

    def close(self, t: float, position=None) -> None:
        """Flush any open zero-plateau window and zero-episode at the run's end."""
        import numpy as _np

        if self._current_zero_run and self._zero_window_start is not None:
            self.illegal_zero_windows.append({
                "from_s": round(self._zero_window_start, 3),
                "to_s": round(float(t), 3),
                "steps": self._current_zero_run,
            })
            self._current_zero_run = 0
            self._zero_window_start = None
        if self._zero_episode is not None:
            entry = self._zero_episode
            entry["to_s"] = round(float(t), 3)
            start = entry.pop("_start")
            entry["net_m"] = (
                round(float(_np.linalg.norm(
                    _np.asarray(position, dtype=float)
                    - _np.asarray(start, dtype=float))), 5)
                if position is not None else 0.0)
            entry["path_m"] = round(entry["path_m"], 5)
            self.zero_episodes.append(entry)
            self._zero_episode = None
