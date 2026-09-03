#!/usr/bin/env python3
"""Frame rendering: the wide shot, the head-camera PiP, the composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``SlalomCamera.render_data`` - the ISOLATED copy in which the head has been
posed for this tick - so the gaze the viewer sees is the one that was measured,
while the authoritative walking state stays untouched.

Importing this module is what pulls in ``imageio`` and ``PIL``; the headless
gate never imports it, so ``validate_slalom.py`` has no rendering dependency at
all.  ``test_the_headless_gate_imports_no_rendering_stack`` pins that claim by
blocking those modules in ``sys.meta_path``, and this module is deliberately
outside the list it checks.

FRAMES ARE STREAMED TO ffmpeg, NOT STAGED ON DISK
---------------------------------------------------
A 90 s run at 50 fps is 4500 frames, which as PNGs is well over a gigabyte of
scratch.  :class:`FrameWriter` therefore pipes raw RGB straight into an ffmpeg
process and never writes an intermediate image, which also removes the
encode/decode round trip through PNG.  A frame directory can still be requested
with ``--frames`` when individual stills are wanted for a contact sheet.

THE CAMERA FOLLOWS THE DECISION, NOT JUST THE DUCK
----------------------------------------------------
This behavior is about a RELATIONSHIP between the duck and a body crossing its
path, and the interesting moments are exactly the ones where both are in shot:
the duck committing to a corridor while somebody walks through the other one.  A
look-at locked to the duck alone pushes the crossing body out of frame precisely
when the video needs to show both.  The look-at is therefore biased toward the
midpoint of the duck and whatever it is currently negotiating with, and the
camera distance opens with their separation.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from slalom_course import FLOOR_HALF, GOAL_XY
from slalom_markers import TRAIL_STRIDE
from video_overlay import compose

# The wide camera.  MEASURED by ``tools/probe_framing.py``, which replays the
# REAL recorded trace through this module's own easing and distance ramp and
# scores each candidate on whether the duck is on screen, unoccluded, clear of
# the HUD panels, and large enough to see, with its negotiated body in frame
# too.  Replaced wholesale by the probe's answer, never adjusted by eye.
#
#   azimuth 38, elevation -52, distance 4.60-6.20
#     -> duck on screen 1.000, duck unoccluded 1.000,
#        subject on screen 0.989, duck clear of the HUD panels 0.943,
#        duck 23.9 px across at the median, never below 20.7 px
#
# THE ELEVATION IS THE MEASUREMENT THAT MATTERS.  A slalom is a plan-like
# behavior, so the instinct is to fly high - but the probe shows that past
# -58 deg the crate stacks start eclipsing the duck from directly above:
# unoccluded visibility collapses from 1.000 at -52 deg to 0.778 at -64 deg and
# 0.280 at -70 deg.  The shallow angle also keeps the bodies reading as bodies.
#
# THE AZIMUTH IS CHOSEN ON HUD CLEARANCE, BECAUSE VISIBILITY DOES NOT DECIDE IT.
# At -52 deg every azimuth reaches 1.000 duck visibility.  What separates them is
# where the duck falls in frame: at 38 deg it sits clear of the HUD panels in
# 0.943 of samples against 0.889 at 55 deg and 0.887 at 90 deg.
CAM_AZIMUTH = 38.0
CAM_ELEVATION = -52.0
CAM_DISTANCE_NEAR = 4.60
CAM_DISTANCE_FAR = 6.20
# The separation at which the far distance is reached, matched to the widest
# duck-to-subject gap the run actually produces.
SEPARATION_FOR_FAR_M = 3.40

# The look-at is biased toward the duck but pulled toward whatever it is
# negotiating with, so both stay in frame during every decision.
LOOKAT_SUBJECT_BIAS = 0.62
# Applied once per CONTROL TICK, so this is a 0.44 s time constant at 50 Hz.
LOOKAT_EASE = 0.045
LOOKAT_Z = 0.42
EYE_WALL_MARGIN_M = 0.30
EYE_CLEARS_SCENE_Z = 2.20
AZIMUTH_SWING_DEG = 4.0
AZIMUTH_SWING_PERIOD_S = 30.0

# How much of the duck's recent path the plan view draws behind it.
TRAIL_TICKS = 1200


class FrameWriter:
    """Renders frames and streams them into ffmpeg.

    ``frames_dir`` is optional: when given, individual PNGs are also written so
    a contact sheet can select stills by time.  The default writes none, which
    is what keeps a 4500-frame render off the disk entirely.
    """

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
        self.camera.distance = CAM_DISTANCE_NEAR
        self.camera.elevation = CAM_ELEVATION
        self.camera.azimuth = CAM_AZIMUTH
        start = rollout.records[0]["duck_xy"] if rollout.records else (-4.0, 0.0)
        self.lookat = np.array([start[0], start[1], LOOKAT_Z], dtype=np.float64)
        self.distance = CAM_DISTANCE_NEAR
        self.pip_cam = rollout.camera.camera_id

        # Timeline content accumulated as the rollout runs, so a frame drawn at
        # t only ever shows what had already happened by t.  A viewer must never
        # see a state marked on the timeline before it happens.
        self.summary: dict = {"state_windows": []}
        self._open_state: str | None = None
        # The duck's own recent path, kept at full control rate so the trail is
        # identical at any output frame rate.
        self._trail: list[list[float]] = []
        # frame number -> the tick time and state it was drawn from.  The
        # contact sheet selects by TIME through this manifest rather than by
        # assuming frame == t * fps, which is false whenever the control rate is
        # not an exact multiple of the frame rate.
        self.manifest: list[dict] = []
        self._ffmpeg = self._start_ffmpeg()

    # -- the encoder -------------------------------------------------------
    def _start_ffmpeg(self):
        """One ffmpeg process, fed raw RGB frames on stdin.

        H.264 with ``yuv420p`` and ``+faststart`` so the result plays anywhere,
        at the exact frame rate the frames were composed for.
        """
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

    # -- the camera --------------------------------------------------------
    def _advance_camera(self, record) -> None:
        """Ease the look-at toward the pair's midpoint.  ONE CONTROL TICK.

        THIS RUNS ON EVERY CONTROL TICK, NOT ON EVERY WRITTEN FRAME, AND THAT IS
        THE WHOLE POINT.  An ease applied per written frame advances 4 times a
        second in a 4 fps preview and 50 times a second in the final render, so
        the two would fly different camera paths and the preview would stop
        being evidence about the video.
        """
        duck = np.array(record["duck_xy"], dtype=np.float64)
        subject_name = record["subject"]
        if subject_name == "goal":
            subject = np.asarray(GOAL_XY, dtype=np.float64)
        else:
            subject = np.array(
                record["actor_xy"].get(subject_name, record["duck_xy"]),
                dtype=np.float64)
        target = np.array([
            LOOKAT_SUBJECT_BIAS * duck[0]
            + (1.0 - LOOKAT_SUBJECT_BIAS) * subject[0],
            LOOKAT_SUBJECT_BIAS * duck[1]
            + (1.0 - LOOKAT_SUBJECT_BIAS) * subject[1],
            LOOKAT_Z])
        self.lookat += LOOKAT_EASE * (target - self.lookat)

        separation = float(np.linalg.norm(duck - subject))
        wanted = CAM_DISTANCE_NEAR + (CAM_DISTANCE_FAR - CAM_DISTANCE_NEAR) * \
            min(max(separation / SEPARATION_FOR_FAR_M, 0.0), 1.0)
        self.distance += LOOKAT_EASE * (wanted - self.distance)

    def _aim_camera(self, record) -> None:
        """Point the free camera at the eased look-at, kept out of the scenery.

        The bound is DERIVED from the camera's own geometry rather than
        declared.  ``eye = lookat - forward * distance``, so the eye's HEIGHT is
        ``lookat_z + distance * sin(|elevation|)``.  When that clears
        :data:`EYE_CLEARS_SCENE_Z` the camera is flying over the whole course and
        no horizontal clamp is needed.
        """
        azimuth = CAM_AZIMUTH + AZIMUTH_SWING_DEG * math.sin(
            record["t"] / AZIMUTH_SWING_PERIOD_S)
        elevation = math.radians(CAM_ELEVATION)
        heading = math.radians(azimuth)
        eye_z = self.lookat[2] + self.distance * abs(math.sin(elevation))

        if eye_z >= EYE_CLEARS_SCENE_Z:
            self.camera.lookat[0] = float(self.lookat[0])
            self.camera.lookat[1] = float(self.lookat[1])
        else:
            forward_x = math.cos(elevation) * math.cos(heading)
            forward_y = math.cos(elevation) * math.sin(heading)
            half_x = FLOOR_HALF[0] - EYE_WALL_MARGIN_M
            half_y = FLOOR_HALF[1] - EYE_WALL_MARGIN_M
            shift_x = forward_x * self.distance
            shift_y = forward_y * self.distance
            self.camera.lookat[0] = float(np.clip(
                self.lookat[0], -half_x + shift_x, half_x + shift_x))
            self.camera.lookat[1] = float(np.clip(
                self.lookat[1], -half_y + shift_y, half_y + shift_y))
        self.camera.lookat[2] = float(self.lookat[2])
        self.camera.azimuth = azimuth
        self.camera.distance = self.distance

    def _note_events(self, record) -> None:
        """Accumulate timeline content up to this tick only."""
        state = record["state"]
        if state != self._open_state:
            self.summary["state_windows"].append(
                {"state": state, "start": record["t"], "end": record["t"]})
            self._open_state = state
        else:
            self.summary["state_windows"][-1]["end"] = record["t"]

    # -- one frame ---------------------------------------------------------
    def write(self, index: int, record: dict) -> None:
        # Camera easing and the trail are updated on EVERY control tick, before
        # the frame rate is applied, so the preview and the final render fly the
        # same camera path and draw the same trail.
        self._advance_camera(record)
        self._trail.append(record["duck_xy"])
        if len(self._trail) > TRAIL_TICKS:
            self._trail.pop(0)
        if index % self.every:
            return
        self._note_events(record)

        # Both views are rendered from the camera's isolated render_data, in
        # which the head has been posed for THIS tick.
        gaze = self.rollout.camera.render_data
        self._aim_camera(record)
        self.renderer.update_scene(gaze, camera=self.camera)
        main = self.renderer.render()
        self.pip_renderer.update_scene(gaze, camera=self.pip_cam)
        pip = self.pip_renderer.render()

        image = compose(main, pip, record=record,
                        total_seconds=self.rollout.seconds,
                        summary=self.summary,
                        trail=self._trail[::TRAIL_STRIDE])
        frame = np.asarray(image, dtype=np.uint8)
        self._ffmpeg.stdin.write(frame.tobytes())
        if self.frames_dir is not None:
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
