#!/usr/bin/env python3
"""Frame rendering: the plaza shot, the head-camera PiP, the composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``PpsCamera.render_data`` - the ISOLATED ``MjData`` copy in which the head has
been posed for this tick - so the gaze the viewer sees is the one that was
measured, while the authoritative walking state stays untouched.

Importing this module is what pulls in ``imageio`` and ``PIL``; the headless
gate never imports it, so ``scripts/validate_pps.py`` has no rendering
dependency at all.

FRAMES ARE STREAMED TO ffmpeg, NOT STAGED ON DISK
---------------------------------------------------
190 s at 50 fps is 9500 frames, which as PNGs is several gigabytes of scratch.
:class:`PpsFrameWriter` therefore pipes raw RGB straight into one ffmpeg process
and never writes an intermediate image, which also removes the encode/decode
round trip through PNG.  Individual stills can still be requested with
``--frames`` when a contact sheet needs them.

THE VIEW MODEL IS BUILT HERE, NOT IN THE PANELS
-------------------------------------------------
A rollout record is the trace the gates read; the HUD needs a few derived things
on top of it (who is which colour right now, where each prediction's closest
approach lies, which episodes have closed).  Those are assembled once per frame
in :meth:`PpsFrameWriter.view_of` so every panel reads the SAME numbers, and so
a panel can never quietly recompute one differently.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from pps_cast import ALL_NAMES, WARD
from pps_geometry import escort_point
from pps_hud_style import person_color
from pps_hud_views import TRAIL_STRIDE
from pps_overlay import compose
from pps_render_camera import CAM_ELEVATION, PlazaCamera
from pps_script import EXPECTED_EPISODES

# How much of the duck's recent path the plan view draws behind it.  60 s of
# control ticks: long enough that a whole encounter - leave the slot, walk the
# arc, hold, come back - reads as one curve.
TRAIL_TICKS = 3000


class PpsFrameWriter:
    """Renders frames and streams them into ffmpeg."""

    def __init__(self, rollout, args, pip_w: int, pip_h: int):
        self.rollout = rollout
        self.args = args
        self.frames = 0
        self.every = max(1, int(round((1.0 / rollout.dt) / args.fps)))
        self.frames_dir = Path(args.frames) if getattr(args, "frames", "") \
            else None
        if self.frames_dir is not None:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        model = rollout.model
        self.renderer = mujoco.Renderer(model, height=args.height,
                                        width=args.width)
        self.pip_renderer = mujoco.Renderer(model, height=pip_h, width=pip_w)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.elevation = CAM_ELEVATION
        start = rollout.records[0]["duck_xy"] if rollout.records else (0.62, -2.42)
        self.rig = PlazaCamera(start)
        self.pip_cam = rollout.camera.camera_id

        # Timeline content accumulated as the rollout runs, so a frame drawn at
        # t only ever shows what had already happened by t.  A viewer must never
        # see a state marked on the timeline before it happens.
        self.history: list[dict] = []
        self._open_state: str | None = None
        self._trail: list[list[float]] = []
        # frame number -> the tick time and state it was drawn from.  The
        # contact sheet selects by TIME through this manifest rather than by
        # assuming frame == t * fps, which is false whenever the control rate is
        # not an exact multiple of the frame rate.
        self.manifest: list[dict] = []
        self._ffmpeg = self._start_ffmpeg()

    # -- the encoder -------------------------------------------------------
    def _start_ffmpeg(self):
        """One ffmpeg process, fed raw RGB frames on stdin."""
        out = Path(self.args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.args.width}x{self.args.height}",
            "-r", str(self.args.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "medium",
            "-crf", str(self.args.crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out),
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE)

    # -- the view model ----------------------------------------------------
    def view_of(self, record: dict) -> dict:
        """Everything the HUD draws for this tick, assembled once."""
        machine = self.rollout.machine
        states = self.rollout.previous_states
        selected, secondary = machine.selected, machine.secondary
        camera_state = self.rollout.camera_state

        people = {}
        for name in ALL_NAMES:
            state = states[name]
            people[name] = {
                "xy": [float(state.pos[0]), float(state.pos[1])],
                "yaw_deg": float(np.degrees(state.yaw)),
                "present": bool(state.present),
                "visible": bool(camera_state["people"][name]["visible"]),
            }

        # Each prediction drawn where the duck predicted the person's CLOSEST
        # APPROACH, not where they currently are: that point is what the
        # intrusion test is actually applied to.
        prediction_points = []
        for entry in record["predictions"]:
            state = states.get(entry["name"])
            if state is None or not state.present:
                continue
            point = state.pos + state.velocity * min(entry["ttc_s"], 4.0)
            prediction_points.append({
                "name": entry["name"],
                "point": [float(point[0]), float(point[1])],
                "intrusion": bool(entry["intrusion"])})

        ward_state = states[WARD]
        slot = escort_point(ward_state.pos, ward_state.yaw)
        subject = camera_state["subject"]

        target = record["target"]
        state_name = record["state"]
        target_kind = {
            "ESCORT": "escort slot: beside and behind Aina",
            "MONITOR": "escort slot: beside and behind Aina",
            "RETURN_ESCORT": "escort slot: returning",
            "RECOVER": "escort slot: recovering",
            "INTERPOSE": "interpose station: between Aina and the intruder",
            "HOLD_BUFFER": "interpose station: holding it",
            "ESCAPE_GAP": "escape gap: measured clear of both",
        }.get(state_name, "no station: holding or yielding")

        return {
            "t": record["t"],
            "state": state_name,
            "state_held_s": max(0.0, record["t"] - machine.state_since),
            "command": [float(v) for v in record["command"]],
            "command_peak": float(record["command_peak"]),
            "trunk_z": float(record["trunk_z"]),
            "duck_xy": record["duck_xy"],
            "duck_yaw_deg": float(record["duck_yaw_deg"]),
            "ward": WARD,
            "ward_xy": record["ward_xy"],
            "ward_range_m": float(record["ward_range_m"]),
            "people": people,
            "predictions": record["predictions"],
            "prediction_points": prediction_points,
            "selected": selected,
            "secondary": secondary,
            "threat_range_m": record["threat_range_m"],
            "between": bool(record["between"]),
            "target": target,
            "target_kind": target_kind,
            "target_distance_m": record["target_distance_m"],
            "escort_slot": [float(slot[0]), float(slot[1])],
            "escort_distance_m": float(record["escort_distance_m"]),
            "min_person_clearance_m": float(record["min_person_clearance_m"]),
            "nearest_person": record["nearest_person"],
            "scenery_clearance_m": float(record["scenery_clearance_m"]),
            "nearest_scenery": record["nearest_scenery"],
            "camera_subject": subject,
            "subject_visible": bool(
                camera_state["people"][subject]["visible"]),
            "expected_episodes": list(EXPECTED_EPISODES),
            "closed_episodes": list(machine.episodes),
            "person_ink": lambda name: person_color(
                name, WARD, selected, secondary),
        }

    def _note_events(self, record) -> None:
        """Accumulate timeline content up to this tick only."""
        state = record["state"]
        if not self.history or state != self._open_state:
            self.history.append(
                {"state": state, "from_s": record["t"], "to_s": record["t"]})
            self._open_state = state
        else:
            self.history[-1]["to_s"] = record["t"]

    # -- one frame ---------------------------------------------------------
    def write(self, index: int, record: dict) -> None:
        # Camera easing and the trail are updated on EVERY control tick, before
        # the frame rate is applied, so the preview and the final render fly the
        # same camera path and draw the same trail.
        selected = self.rollout.machine.selected
        threat_xy = None
        if selected and selected in self.rollout.previous_states:
            state = self.rollout.previous_states[selected]
            if state.present:
                threat_xy = state.pos
        self.rig.advance(record["duck_xy"], record["ward_xy"], threat_xy)
        self._trail.append(record["duck_xy"])
        if len(self._trail) > TRAIL_TICKS:
            self._trail.pop(0)
        if index % self.every:
            return
        self._note_events(record)

        gaze = self.rollout.camera.render_data
        self.camera.lookat[:] = self.rig.lookat
        self.camera.distance = self.rig.distance
        self.camera.azimuth = self.rig.azimuth_at(record["t"])
        self.renderer.update_scene(gaze, camera=self.camera)
        main = self.renderer.render()
        self.pip_renderer.update_scene(gaze, camera=self.pip_cam)
        pip = self.pip_renderer.render()

        image = compose(main, pip, view=self.view_of(record),
                        total_seconds=self.rollout.seconds,
                        history=self.history,
                        episodes=list(self.rollout.machine.episodes),
                        trail=self._trail[::TRAIL_STRIDE])
        frame = np.asarray(image, dtype=np.uint8)
        self._ffmpeg.stdin.write(frame.tobytes())
        if self.frames_dir is not None:
            import imageio.v2 as imageio
            imageio.imwrite(self.frames_dir / f"f{self.frames:05d}.png", frame)
        self.manifest.append({"frame": self.frames, "t": record["t"],
                              "state": record["state"]})
        self.frames += 1

    def close(self) -> int:
        """Finish the encode and return ffmpeg's exit status."""
        self._ffmpeg.stdin.close()
        return self._ffmpeg.wait()

    def write_manifest(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest))
        return path
