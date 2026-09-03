#!/usr/bin/env python3
"""Everything the rollout ACCUMULATES, in one object with no physics in it.

Split out of ``rollout_guide`` so that the tick loop stays about ORDER — what is
measured before what — and this stays about TALLYING.  The rollout owns one of
these and forwards each tick's measurements into it; ``guide_summary`` reads it
afterwards.

Two accumulators here encode a decision that is easy to get wrong, so they are
methods rather than bare counters:

* :meth:`note_lead_gap` measures "the duck was in front" along the SHARED TRAIL
  rather than in a body frame.  A body-frame test would call the follower ahead
  every time the duck turned a corner, which is the opposite of what the
  invariant is for.

* :meth:`note_safety` tracks the RUNNING LENGTH of the current breach, not a
  count of breached ticks.  A distance beyond the safety maximum is not itself a
  failure — the duck has to notice and stop, which takes time.  What must not
  happen is a PROLONGED interval, so the maximum continuous run is what the gate
  grades.
"""

from __future__ import annotations

import numpy as np

from guide_states import SAFETY_MAX_DISTANCE_M


class RolloutTally:
    """Per-run and per-episode accumulators.  No MuJoCo, no policy, no time."""

    def __init__(self, dt: float, initial_trunk_z: float):
        self.dt = float(dt)

        # -- locomotion health ------------------------------------------
        self.min_trunk_z = float(initial_trunk_z)
        self.path_m = 0.0
        self.fallen_steps = 0
        self.contact_steps = 0

        # -- clearance ---------------------------------------------------
        self.min_person_clearance = float("inf")
        self.min_person_name = ""
        self.min_follower_clearance = float("inf")
        self.min_scenery_clearance = float("inf")
        self.min_scenery_geom = ""

        # -- per-state ----------------------------------------------------
        self.state_steps: dict[str, int] = {}
        self.state_command_max: dict[str, float] = {}
        self.zero_command_violations: list[dict] = []

        # -- leading ------------------------------------------------------
        self.lead_path_m = 0.0
        self.lead_steps = 0
        self.max_cross_track_m = 0.0
        self.max_follower_range_m = 0.0

        # -- visibility, conditioned on line of sight ---------------------
        self.visible_steps = 0
        self.los_steps = 0
        self.visible_with_los = 0
        self.monitor_steps = 0
        self.monitor_los_steps = 0
        self.monitor_visible_with_los = 0
        self.blocked_by: dict[str, int] = {}

        # -- the guide-leads invariant ------------------------------------
        self.follower_ahead_steps = 0
        self.min_lead_gap_m = float("inf")

        # -- safety --------------------------------------------------------
        self._safety_run_s = 0.0
        self.max_safety_breach_s = 0.0

        # -- squaring up ---------------------------------------------------
        self.check_path_m = 0.0
        self.max_check_path_m = 0.0

        # -- per-episode, keyed by episode index --------------------------
        self.episode_wait_command_peak: dict[int, float] = {}
        self.episode_wait_steps: dict[int, int] = {}
        self.episode_wait_only_steps: dict[int, int] = {}
        self.episode_wait_moved_m: dict[int, float] = {}
        self.episode_check_path_m: dict[int, float] = {}
        self.episode_duck_moved_m: dict[int, float] = {}
        self.episode_follower_closed_m: dict[int, float] = {}
        self.episode_start_distance: dict[int, float] = {}
        self.episode_wait_xy: dict[int, np.ndarray] = {}
        self.episode_wait_scenery: dict[int, float] = {}
        self.episode_visible_steps: dict[int, int] = {}
        self.episode_los_steps: dict[int, int] = {}

    # -- locomotion --------------------------------------------------------
    def note_pose(self, trunk_z: float, travelled: float) -> None:
        self.min_trunk_z = min(self.min_trunk_z, float(trunk_z))
        if float(trunk_z) < 0.09:
            self.fallen_steps += 1
        self.path_m += float(travelled)

    def note_command(self, state: str, peak: float) -> None:
        self.state_command_max[state] = max(
            self.state_command_max.get(state, 0.0), float(peak))
        self.state_steps[state] = self.state_steps.get(state, 0) + 1

    def note_zero_violation(self, t: float, state: str, command) -> None:
        self.zero_command_violations.append(
            {"t": round(float(t), 3), "state": state,
             "command": [float(v) for v in command]})

    # -- clearance ---------------------------------------------------------
    def note_clearance(self, clearances: dict, nearest: str,
                       follower_name: str, scenery_gap: float,
                       scenery_geom: str) -> None:
        if clearances[nearest] < self.min_person_clearance:
            self.min_person_clearance = clearances[nearest]
            self.min_person_name = nearest
        if clearances[nearest] <= 0.0:
            self.contact_steps += 1
        self.min_follower_clearance = min(
            self.min_follower_clearance, clearances[follower_name])
        if scenery_gap < self.min_scenery_clearance:
            self.min_scenery_clearance = scenery_gap
            self.min_scenery_geom = scenery_geom

    # -- leading ------------------------------------------------------------
    def note_lead(self, travelled: float, cross_track: float) -> None:
        self.lead_steps += 1
        self.lead_path_m += float(travelled)
        self.max_cross_track_m = max(self.max_cross_track_m, float(cross_track))

    def note_check(self, travelled: float, episode_index: int) -> None:
        self.check_path_m += float(travelled)
        self.max_check_path_m = max(self.max_check_path_m, self.check_path_m)
        self.episode_check_path_m[episode_index] = self.check_path_m

    def reset_check(self) -> None:
        self.check_path_m = 0.0

    def note_lead_gap(self, trail_gap_m: float) -> None:
        """Measured along the duck's own trail, which is the only place 'ahead'
        has a meaning on a shared path."""
        self.min_lead_gap_m = min(self.min_lead_gap_m, float(trail_gap_m))
        if trail_gap_m < 0.0:
            self.follower_ahead_steps += 1

    def note_safety(self, range_m: float) -> None:
        """Track the CONTINUOUS length of the current breach, not a count."""
        self.max_follower_range_m = max(self.max_follower_range_m,
                                        float(range_m))
        if range_m > SAFETY_MAX_DISTANCE_M:
            self._safety_run_s += self.dt
            self.max_safety_breach_s = max(self.max_safety_breach_s,
                                           self._safety_run_s)
        else:
            self._safety_run_s = 0.0

    @property
    def safety_breach_s(self) -> float:
        return self._safety_run_s

    # -- visibility ---------------------------------------------------------
    def note_visibility(self, *, visible: bool, los_ok: bool, monitoring: bool,
                        blocker: str) -> None:
        if visible:
            self.visible_steps += 1
        if los_ok:
            self.los_steps += 1
            if visible:
                self.visible_with_los += 1
        if monitoring:
            self.monitor_steps += 1
            if los_ok:
                self.monitor_los_steps += 1
                if visible:
                    self.monitor_visible_with_los += 1
        if not visible:
            key = blocker or "out_of_frustum"
            self.blocked_by[key] = self.blocked_by.get(key, 0) + 1

    # -- per-episode --------------------------------------------------------
    def note_monitor_tick(self, index: int, *, travelled: float,
                          range_m: float, waiting_spot, scenery_gap: float,
                          los_ok: bool, visible: bool) -> None:
        self.episode_wait_steps[index] = self.episode_wait_steps.get(index, 0) + 1
        self.episode_duck_moved_m[index] = \
            self.episode_duck_moved_m.get(index, 0.0) + float(travelled)
        self.episode_start_distance.setdefault(index, float(range_m))
        self.episode_follower_closed_m[index] = \
            self.episode_start_distance[index] - float(range_m)
        if waiting_spot is not None:
            self.episode_wait_xy[index] = np.asarray(waiting_spot).copy()
        self.episode_wait_scenery[index] = min(
            self.episode_wait_scenery.get(index, float("inf")),
            float(scenery_gap))
        if los_ok:
            self.episode_los_steps[index] = \
                self.episode_los_steps.get(index, 0) + 1
            if visible:
                self.episode_visible_steps[index] = \
                    self.episode_visible_steps.get(index, 0) + 1

    def note_wait_tick(self, index: int, *, peak: float,
                       travelled: float) -> None:
        """WAIT_FOR_PERSON only.

        The waiting claim is graded on this state ALONE, because it is the only
        one that asserts the duck stopped.  Folding CHECK_FOLLOWER in here would
        let a state with a different contract vouch for it.
        """
        self.episode_wait_command_peak[index] = max(
            self.episode_wait_command_peak.get(index, 0.0), float(peak))
        self.episode_wait_moved_m[index] = \
            self.episode_wait_moved_m.get(index, 0.0) + float(travelled)
        self.episode_wait_only_steps[index] = \
            self.episode_wait_only_steps.get(index, 0) + 1
