#!/usr/bin/env python3
"""The two tick phases that are about BOOKKEEPING rather than ordering.

:class:`GestureRollout` owns the strict per-tick order - pose the world, measure,
decide, command, step physics.  Steps 4 and 7 of that order are large and are
about something different from ordering: what happens once at a state change,
and what is measured after the physics step.  They live here as a mixin so the
rollout body stays readable as a sequence, and so each phase can be read on its
own.

This is a split by CONCERN, not a split by line count: both methods below need
the rollout's own state, and neither makes sense detached from it.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from policy_runtime import FALLEN_TRUNK_Z
from gest_actors import bodies_at, pose_bodies
from gest_aim import select
from gest_arena import inside_area
from gest_cast import ALL_NAMES, INSTRUCTOR
from gest_control import is_sub_gait
from gest_markers import TRAIL_STRIDE, heading_points, pose_markers
from gest_record import build_record
from gest_sense import los_blocked_by, measured_positions
from gest_states import (
    MONITOR_STATES,
    WALKING_STATES,
    ZERO_COMMAND_STATES,
)


class RolloutPhases:
    """Transition bookkeeping and post-physics measurement for one rollout."""

    # -- transitions ---------------------------------------------------------
    def _on_transition(self, t, state, changed, state_before, sense, confirmed,
                       duck_xy, duck_yaw, instructor_range,
                       instructor_clearance) -> None:
        """Everything that happens exactly once, at a state change."""
        if not changed:
            return

        # THE ACTION'S OWN RESULT IS FROZEN THE MOMENT ITS EXECUTE STATE ENDS.
        # Everything after this is ACK, during which the duck holds an exact
        # zero and its gait unwinds - which moves the very quantities the turn
        # and reverse gates are graded on.  See ``Episode.execute_*``.
        #
        # AN INTERRUPTED EPISODE MUST BE FROZEN HERE TOO, and missing that was a
        # real gate failure.  ``_interrupt`` closes the COME episode and opens
        # the STOP one in the same transition, so by the time this ran the
        # machine's open episode was already the STOP and the COME kept its
        # default zeros - the approach gate then read 0.000 m of path for a walk
        # that had visibly covered 1.75 m.  Freezing the episode that was open
        # BEFORE the transition is what makes an interrupted action still
        # carry its own measurements.
        if (state_before.startswith("EXECUTE_") or state_before == "GOODBYE") \
                and not (state.startswith("EXECUTE_") or state == "GOODBYE"):
            self._freeze_execute(t, sense, instructor_range)
        elif state == "EXECUTE_STOP" and state_before in WALKING_STATES:
            # The interrupt path: the episode that was open a moment ago has
            # just been closed by the machine, so freeze the last one it closed.
            self._freeze_execute(t, sense, instructor_range, interrupted=True)

        # A CONFIRMED COMMAND IS COMMITTED ONCE, on the tick it is acted on.
        if confirmed is not None and (state.startswith("EXECUTE_")
                                      or state == "GOODBYE"):
            self.detector.accept(confirmed)

        # THE REFERENCE POSE IS LATCHED THE MOMENT AN ACTION BEGINS.  Every
        # turn and reverse claim is a delta from here, so an action that began
        # from a different pose than the one it is graded against is impossible.
        if state.startswith("EXECUTE_") or state == "GOODBYE":
            self._reference_xy = np.asarray(duck_xy, dtype=np.float64).copy()
            self._reference_yaw = float(duck_yaw)
            episode = self.machine.episode
            if episode is not None:
                episode.start_xy = (float(duck_xy[0]), float(duck_xy[1]))
                episode.start_yaw_deg = math.degrees(float(duck_yaw))
                episode.start_range_m = float(instructor_range)
                # The command magnitude on the tick BEFORE this state began.
                # A STOP that interrupted nothing would show a zero here, which
                # is why it is recorded rather than assumed.
                episode.command_before_stop = float(self._last_peak)
                episode.confirm_ticks = self._pending_confirm_ticks
                episode.confirm_visible_ticks = self._pending_visible_ticks
                episode.confirm_readable_ticks = self._pending_readable_ticks
                if self.machine.interrupts:
                    last = self.machine.interrupts[-1]
                    if abs(last["t"] - t) < 1e-6:
                        episode.interrupts_command = last["interrupted"]
            self._pending_confirm_ticks = 0
            self._pending_visible_ticks = 0
            self._pending_readable_ticks = 0

        if state == "EXECUTE_STOP":
            self._stop_hold_s = 0.0
            self._stop_reference_xy = np.asarray(
                duck_xy, dtype=np.float64).copy()
            self._stop_zero_ticks = None

        # GESTURE READING IS NARROWED, NOT SHUT DOWN, WHILE THE DUCK WALKS.
        # A STOP exists to interrupt motion already under way, so it must be
        # readable exactly when the duck is moving; every other command is
        # refused mid-manoeuvre.  In a state where the duck is already still,
        # nothing may be read at all until it returns to READY.
        if state in WALKING_STATES:
            self.detector.suspend(interrupt_only=True)
        elif state.startswith("EXECUTE_") or state == "GOODBYE":
            self.detector.suspend()
        if state in ("READY", "OBSERVE") \
                and (self.detector.suspended or self.detector.interrupt_only):
            self.detector.resume()

    def _freeze_execute(self, t, sense, instructor_range,
                        interrupted: bool = False) -> None:
        """Record what an action achieved, at the instant it stopped acting.

        ``interrupted`` selects the episode the machine has just CLOSED rather
        than the one it has open, because an interrupt closes one episode and
        opens another within a single transition.
        """
        if interrupted:
            episode = self.machine.episodes[-1] if self.machine.episodes else None
        else:
            episode = self.machine.episode
        if episode is None:
            return
        episode.execute_ended_s = float(t)
        episode.execute_yaw_delta_deg = float(sense.yaw_delta_deg)
        episode.execute_back_m = float(sense.back_along_heading_m)
        episode.execute_path_m = float(episode.path_m)
        episode.execute_end_range_m = float(instructor_range)
        episode.execute_min_clearance_m = float(episode.min_clearance_m)
        episode.execute_in_standoff_band = bool(sense.in_standoff_band)

    # -- after physics -------------------------------------------------------

    # -- after physics -------------------------------------------------------
    def _after_physics(self, index, t, state, sense, command, interlock,
                       target_xy, view) -> dict:
        """Re-pose the world at the display time, measure, and record."""
        duck_pos = self.data.xpos[self.trunk].copy()
        duck_xy = np.array([float(duck_pos[0]), float(duck_pos[1])])
        duck_yaw_after = self.runner.yaw(self.data)

        display_t = min(t + self.dt, self.seconds)
        bodies_now = bodies_at(display_t)
        pose_bodies(self.model, self.data, bodies_now, display_t)

        travelled = float(np.linalg.norm(duck_xy - self._previous_xy))
        self._measured_speed = travelled / self.dt
        self.tally.note_pose(float(duck_pos[2]), travelled, FALLEN_TRUNK_Z)

        # -- the markers, from the SAME objects the behavior acts on --------
        trail = [np.asarray(r["duck_xy"]) for r in self.records[::-TRAIL_STRIDE]]
        positions_now = measured_positions(bodies_now)
        # THE HEADING RAY is drawn along the heading the CURRENT command is
        # closing on, so a LEFT turn and a RIGHT turn are visibly opposite in
        # the wide shot rather than only in the HUD.
        heading = heading_points(duck_xy, duck_yaw_after)
        pose_markers(
            self.model, self.data, trail_points=trail, heading=heading,
            target_xy=target_xy,
            focus_xy=positions_now.get(self.detector.acquisition.locked))
        mujoco.mj_forward(self.model, self.data)

        # -- the camera, in its isolated copy -------------------------------
        present = {n: s.present for n, s in bodies_now.items()}
        aim = select(state, locked=self.detector.acquisition.locked,
                     positions=positions_now)
        if aim.searching:
            look_at = self.camera.search_target(duck_xy, duck_yaw_after)
        else:
            look_at = aim.look_at
        camera_state = self.camera.update(
            self.data, duck_yaw=duck_yaw_after, subject=aim.subject,
            look_at=look_at, present=present)
        self._camera_state = camera_state
        if aim.searching:
            pass
        else:
            self.camera.begin_search()

        # -- safety, measured against the REAL post-step pose ---------------
        clearances = {name: self.contacts.distance(self.data, name)
                      for name in ALL_NAMES}
        nearest = min(clearances, key=clearances.get)
        scenery_gap, scenery_geom = self.scenery.distance(self.data)
        self.tally.note_clearance(display_t, clearances, nearest, scenery_gap,
                                  scenery_geom)
        self.tally.note_area(bool(inside_area(duck_xy, self.duck_radius)))

        # -- the command contract, checked as it is made --------------------
        is_zero_state = state in ZERO_COMMAND_STATES
        self.tally.note_command(display_t, state, command, travelled,
                                is_zero_state, is_sub_gait(float(command[0])))
        if state in WALKING_STATES:
            self.tally.note_walk(travelled)

        # -- visibility, conditioned on line of sight -----------------------
        instructor_entry = camera_state["bodies"].get(INSTRUCTOR, {})
        instructor_visible = bool(instructor_entry.get("visible"))
        readable = instructor_entry.get("arm_readable", {"l": False, "r": False})
        arm_readable = bool(readable.get("l") or readable.get("r"))
        eye_xy = self.camera.render_data.cam_xpos[self.camera.camera_id][:2]
        if INSTRUCTOR in positions_now:
            los_blocker = los_blocked_by(eye_xy, positions_now[INSTRUCTOR],
                                         bodies_now, exclude=INSTRUCTOR)
        else:
            los_blocker = ""
        blocker = ("" if instructor_visible
                   else self.camera.blocking_geom(INSTRUCTOR))
        self.tally.note_camera(
            aim_in_frustum=bool(camera_state["aim_in_frustum"]),
            monitoring=state in MONITOR_STATES,
            instructor_visible=instructor_visible, arm_readable=arm_readable,
            los_ok=not los_blocker, blocker=blocker,
            seen=camera_state["visible_bodies"])

        # -- the open episode accumulates its own evidence -------------------
        episode = self.machine.episode
        if episode is not None:
            episode.path_m += travelled
            episode.yaw_delta_deg = sense.yaw_delta_deg
            episode.back_along_heading_m = sense.back_along_heading_m
            episode.forward_along_heading_m = -sense.back_along_heading_m
            episode.end_xy = (float(duck_xy[0]), float(duck_xy[1]))
            episode.end_yaw_deg = math.degrees(float(duck_yaw_after))
            episode.end_range_m = float(np.linalg.norm(
                positions_now[INSTRUCTOR] - duck_xy)
                if INSTRUCTOR in positions_now else 0.0)
            episode.command_peak = max(episode.command_peak,
                                       float(np.max(np.abs(command))))
            episode.command_vx_peak = max(episode.command_vx_peak,
                                          float(command[0]))
            episode.command_vx_min = min(episode.command_vx_min,
                                         float(command[0]))
            episode.min_clearance_m = min(episode.min_clearance_m,
                                          float(clearances[nearest]))
            if state == "EXECUTE_STOP":
                episode.stop_hold_s = self._stop_hold_s
                episode.stop_drift_m = float(np.linalg.norm(
                    duck_xy - self._stop_reference_xy))
                if episode.ticks_to_zero is None \
                        and float(np.max(np.abs(command))) == 0.0:
                    episode.ticks_to_zero = 0

        # -- the confirm evidence, per episode --------------------------------
        # Counted on OBSERVE and CONFIRM ticks, and on the WALKING ticks where
        # an interrupt could be read, so "every acceptance required visibility"
        # is a per-episode measurement rather than a run average.
        if state in ("OBSERVE", "CONFIRM") or self.detector.interrupt_only:
            self._pending_confirm_ticks += 1
            self._pending_visible_ticks += 1 if instructor_visible else 0
            self._pending_readable_ticks += 1 if arm_readable else 0

        record = build_record(
            display_t=display_t, state=state, machine=self.machine,
            detector=self.detector, view=view, command=command,
            duck_xy=duck_xy, duck_yaw_after=duck_yaw_after, duck_pos=duck_pos,
            min_trunk_z=self.tally.min_trunk_z, camera_state=camera_state,
            clearances=clearances, nearest=nearest, scenery_gap=scenery_gap,
            scenery_geom=scenery_geom, bodies=bodies_now, sense=sense,
            instructor_visible=instructor_visible, arm_readable=arm_readable,
            los_available=not los_blocker, los_blocker=los_blocker,
            path_m=self.tally.path_m, state_elapsed=t - self.machine.state_since,
            target_xy=target_xy, interlock=interlock,
            camera_active=bool(camera_state["aim_in_frustum"]))
        self.records.append(record)

        self._previous_bodies = bodies_now
        self._previous_xy = duck_xy.copy()
        return record
