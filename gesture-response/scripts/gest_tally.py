#!/usr/bin/env python3
"""Per-tick accumulation of everything a gate is graded on.

Kept apart from the rollout so the tick body stays about ORDER and this stays
about EVIDENCE.  Nothing here decides anything; it only counts what happened, so
a gate can be graded on a number that was accumulated as the run went rather
than reconstructed from a trace afterwards.

WHY EACH COUNTER EXISTS
------------------------
Every field below is the honest form of a claim this behavior makes:

* ``zero_violations`` - the strongest claim here is that a zero-command state
  emits a LITERAL ``(0, 0, 0)``.  It is checked on the tick the command is
  produced, not inferred from a trace, and any breach is recorded with the state
  and the offending vector so a failure names itself.
* ``sub_gait_ticks`` - a command strictly between zero and a MEASURED gait onset
  logs an intention and moves the robot millimetres.  Counted per tick against
  :func:`gest_control.is_sub_gait` rather than against this module's own idea of
  the onset.
* ``max_abs_vy`` - lateral commands are a yaw disturbance wearing a strafe's
  clothes on this policy, so the gate requires an exact zero over every tick.
* ``min_clearance`` / ``min_scenery_gap`` - surface clearances measured
  analytically every control tick, because the people are non-colliding proxies
  and MuJoCo's own contact count would be vacuous.
* ``fallen_steps`` / ``min_trunk_z`` - the only two numbers that say the robot
  stayed on its feet.
"""

from __future__ import annotations

import numpy as np


class GestureTally:
    """Counts, minima and per-state totals, accumulated as the run happens."""

    def __init__(self, dt: float, start_trunk_z: float):
        self.dt = float(dt)
        self.ticks = 0

        # -- locomotion health ------------------------------------------
        self.min_trunk_z = float(start_trunk_z)
        self.final_trunk_z = float(start_trunk_z)
        self.fallen_steps = 0
        self.path_m = 0.0
        self.walk_path_m = 0.0

        # -- the command contract ----------------------------------------
        self.zero_violations: list[dict] = []
        self.sub_gait_ticks = 0
        self.sub_gait_examples: list[dict] = []
        self.max_abs_vy = 0.0
        self.max_abs_vx = 0.0
        self.min_vx = 0.0
        self.command_ticks_by_state: dict[str, int] = {}
        self.path_by_state: dict[str, float] = {}
        self.nonzero_ticks_by_state: dict[str, int] = {}

        # -- safety -------------------------------------------------------
        self.min_clearance = float("inf")
        self.min_clearance_body = ""
        self.min_clearance_t = 0.0
        self.min_scenery_gap = float("inf")
        self.min_scenery_geom = ""
        self.contacts = 0
        self.outside_area_ticks = 0
        self.interlock_ticks = 0
        self.interlock_reasons: dict[str, int] = {}

        # -- the camera ----------------------------------------------------
        self.aim_in_frustum_ticks = 0
        self.instructor_visible_ticks = 0
        self.instructor_arm_readable_ticks = 0
        self.monitor_ticks = 0
        self.monitor_visible_ticks = 0
        self.monitor_los_ticks = 0
        self.occluder_hits: dict[str, int] = {}
        self.people_seen: set[str] = set()

        # -- stillness -----------------------------------------------------
        # Path accumulated during every tick that was supposed to be an exact
        # zero.  This is what turns "it held still" into a floor measurement.
        self.zero_state_path_m = 0.0
        self.zero_state_ticks = 0
        self.worst_zero_episode_m = 0.0
        self._zero_episode_m = 0.0
        self._in_zero_episode = False

    # -- locomotion ---------------------------------------------------------
    def note_pose(self, trunk_z: float, travelled: float,
                  fallen_bar: float) -> None:
        self.ticks += 1
        self.min_trunk_z = min(self.min_trunk_z, float(trunk_z))
        self.final_trunk_z = float(trunk_z)
        if float(trunk_z) < fallen_bar:
            self.fallen_steps += 1
        self.path_m += float(travelled)

    def note_walk(self, travelled: float) -> None:
        self.walk_path_m += float(travelled)

    # -- the command contract ------------------------------------------------
    def note_command(self, t: float, state: str, command, travelled: float,
                     is_zero_state: bool, sub_gait: bool) -> None:
        vector = np.asarray(command, dtype=np.float64)
        peak = float(np.max(np.abs(vector)))
        self.command_ticks_by_state[state] = \
            self.command_ticks_by_state.get(state, 0) + 1
        self.path_by_state[state] = \
            self.path_by_state.get(state, 0.0) + float(travelled)
        if peak != 0.0:
            self.nonzero_ticks_by_state[state] = \
                self.nonzero_ticks_by_state.get(state, 0) + 1

        self.max_abs_vy = max(self.max_abs_vy, abs(float(vector[1])))
        self.max_abs_vx = max(self.max_abs_vx, float(vector[0]))
        self.min_vx = min(self.min_vx, float(vector[0]))

        if is_zero_state and peak != 0.0:
            self.zero_violations.append({
                "t": round(float(t), 3), "state": state,
                "command": [round(float(v), 6) for v in vector]})
        if sub_gait:
            self.sub_gait_ticks += 1
            if len(self.sub_gait_examples) < 8:
                self.sub_gait_examples.append({
                    "t": round(float(t), 3), "state": state,
                    "vx": round(float(vector[0]), 6)})

        # Stillness, per contiguous zero-command episode.
        if is_zero_state:
            self.zero_state_ticks += 1
            self.zero_state_path_m += float(travelled)
            self._zero_episode_m += float(travelled)
            self._in_zero_episode = True
        elif self._in_zero_episode:
            self.worst_zero_episode_m = max(self.worst_zero_episode_m,
                                            self._zero_episode_m)
            self._zero_episode_m = 0.0
            self._in_zero_episode = False

    def note_interlock(self, blocked: bool, reason: str) -> None:
        if not blocked:
            return
        self.interlock_ticks += 1
        self.interlock_reasons[reason] = \
            self.interlock_reasons.get(reason, 0) + 1

    # -- safety ---------------------------------------------------------------
    def note_clearance(self, t: float, clearances: dict[str, float],
                       nearest: str, scenery_gap: float,
                       scenery_geom: str) -> None:
        gap = float(clearances[nearest])
        if gap < self.min_clearance:
            self.min_clearance = gap
            self.min_clearance_body = nearest
            self.min_clearance_t = round(float(t), 3)
        if gap <= 0.0:
            self.contacts += 1
        if float(scenery_gap) < self.min_scenery_gap:
            self.min_scenery_gap = float(scenery_gap)
            self.min_scenery_geom = scenery_geom

    def note_area(self, inside: bool) -> None:
        if not inside:
            self.outside_area_ticks += 1

    # -- the camera ------------------------------------------------------------
    def note_camera(self, *, aim_in_frustum: bool, monitoring: bool,
                    instructor_visible: bool, arm_readable: bool,
                    los_ok: bool, blocker: str, seen: list[str]) -> None:
        if aim_in_frustum:
            self.aim_in_frustum_ticks += 1
        if instructor_visible:
            self.instructor_visible_ticks += 1
        if arm_readable:
            self.instructor_arm_readable_ticks += 1
        if monitoring:
            self.monitor_ticks += 1
            if instructor_visible:
                self.monitor_visible_ticks += 1
            if los_ok:
                self.monitor_los_ticks += 1
        if blocker:
            self.occluder_hits[blocker] = self.occluder_hits.get(blocker, 0) + 1
        self.people_seen.update(seen)

    # -- closing ----------------------------------------------------------------
    def close(self) -> None:
        if self._in_zero_episode:
            self.worst_zero_episode_m = max(self.worst_zero_episode_m,
                                            self._zero_episode_m)

    def as_record(self) -> dict:
        return {
            "ticks": self.ticks,
            "min_trunk_z": round(self.min_trunk_z, 5),
            "final_trunk_z": round(self.final_trunk_z, 5),
            "fallen_steps": self.fallen_steps,
            "path_m": round(self.path_m, 4),
            "walk_path_m": round(self.walk_path_m, 4),
            "zero_violations": self.zero_violations,
            "zero_violation_count": len(self.zero_violations),
            "sub_gait_ticks": self.sub_gait_ticks,
            "sub_gait_examples": self.sub_gait_examples,
            "max_abs_vy": round(self.max_abs_vy, 8),
            "max_abs_vx": round(self.max_abs_vx, 6),
            "min_vx": round(self.min_vx, 6),
            "command_ticks_by_state": dict(self.command_ticks_by_state),
            "path_by_state": {k: round(v, 5)
                              for k, v in self.path_by_state.items()},
            "nonzero_ticks_by_state": dict(self.nonzero_ticks_by_state),
            "min_clearance_m": (None if self.min_clearance > 1e8
                                else round(self.min_clearance, 4)),
            "min_clearance_body": self.min_clearance_body,
            "min_clearance_t_s": self.min_clearance_t,
            "min_scenery_gap_m": (None if self.min_scenery_gap > 1e8
                                  else round(self.min_scenery_gap, 4)),
            "min_scenery_geom": self.min_scenery_geom,
            "contacts": self.contacts,
            "outside_area_ticks": self.outside_area_ticks,
            "interlock_ticks": self.interlock_ticks,
            "interlock_reasons": dict(self.interlock_reasons),
            "aim_in_frustum_ticks": self.aim_in_frustum_ticks,
            "aim_in_frustum_fraction": round(
                self.aim_in_frustum_ticks / max(self.ticks, 1), 4),
            "instructor_visible_ticks": self.instructor_visible_ticks,
            "instructor_arm_readable_ticks": self.instructor_arm_readable_ticks,
            "monitor_ticks": self.monitor_ticks,
            "monitor_visible_ticks": self.monitor_visible_ticks,
            "monitor_visible_fraction": round(
                self.monitor_visible_ticks / max(self.monitor_ticks, 1), 4),
            "monitor_los_ticks": self.monitor_los_ticks,
            "occluder_hits": dict(self.occluder_hits),
            "people_seen": sorted(self.people_seen),
            "zero_state_ticks": self.zero_state_ticks,
            "zero_state_path_m": round(self.zero_state_path_m, 5),
            "worst_zero_episode_m": round(self.worst_zero_episode_m, 5),
        }
