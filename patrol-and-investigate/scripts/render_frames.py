#!/usr/bin/env python3
"""Frame rendering: the wide shot, the head-camera PiP, the composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``PatrolCamera.render_data`` - the ISOLATED copy in which the head has been
posed for this tick - so the gaze the viewer sees is the one that was measured,
while the authoritative walking state stays untouched.

Importing this module is what pulls in ``imageio`` and ``PIL``; the headless
gate never imports it, so ``validate_patrol.py`` has no rendering dependency at
all.  ``test_the_headless_gate_imports_no_rendering_stack`` pins that claim by
blocking those modules in ``sys.meta_path``, and this module is deliberately
outside the list it checks.

FRAMES ARE STREAMED TO ffmpeg, NOT STAGED ON DISK
---------------------------------------------------
A 145 s run at 50 fps is 7250 frames, which as PNGs is well over a gigabyte of
scratch.  :class:`FrameWriter` therefore pipes raw RGB straight into an ffmpeg
process and never writes an intermediate image, which also removes the
encode/decode round trip through PNG.  A frame directory can still be requested
with ``--frames`` when individual stills are wanted for a contact sheet.

THE CAMERA FOLLOWS THE DUCK, AND OPENS UP FOR AN INVESTIGATION
----------------------------------------------------------------
For most of this behavior the interesting thing is the ROBOT ON ITS ROUTE, so
the look-at stays on the duck.  During an investigation the interesting thing is
the RELATIONSHIP between the duck and what it is looking at, so the look-at is
biased toward their midpoint and the distance opens with their separation -
which is exactly when both need to be in shot together.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from patrol_facility import FLOOR_HALF
from patrol_markers import TRAIL_STRIDE
from video_overlay import compose

# The wide camera.  MEASURED by ``tools/probe_framing.py``, which replays the
# REAL recorded trace through this module's own easing and distance ramp and
# scores each candidate on whether the duck is on screen, unoccluded, clear of
# the HUD panels, and large enough to see, with whatever it is investigating in
# frame too.  Replaced wholesale by the probe's answer, never adjusted by eye.
#
#   azimuth 38, elevation -52, distance 3.60-5.00
#     -> duck on screen 1.000, duck unoccluded 0.999,
#        subject on screen 0.995, duck clear of the HUD panels 1.000,
#        duck 33.1 px across at the median, never below 24.7 px
#
# THE ELEVATION IS THE MEASUREMENT THAT MATTERS, AND IT BEHAVES THE OPPOSITE
# WAY ROUND TO THE SIBLING BEHAVIOR.  On the depot floor of ``dynamic-slalom``
# flying higher made things worse, because crate stacks eclipsed the duck from
# directly above.  Here the fixtures are FEWER and TALLER and the circuit runs
# round a central island, so a shallow camera looks THROUGH the racking: at
# -38 deg unoccluded visibility falls to 0.973 at this azimuth and to 0.706 at
# 90 deg, while by -52 deg it is 0.999 from every azimuth tested.  Going higher
# still buys nothing measurable - -58 and -64 both report 1.000 - and costs the
# sense of a robot walking a floor, so -52 is where the gain stops.
#
# THE AZIMUTH IS CHOSEN ON THE INVESTIGATION SHOTS.  Every azimuth keeps the
# duck on screen and clear of the HUD, so what separates them is whether the
# thing being investigated is in frame too: 38 deg holds the subject in 0.995
# of investigation ticks against 0.983 at 125 deg.
CAM_AZIMUTH = 38.0
CAM_ELEVATION = -52.0
CAM_DISTANCE_NEAR = 3.60
CAM_DISTANCE_FAR = 5.00
# The separation at which the far distance is reached, matched to the widest
# duck-to-target gap the run actually produces.
SEPARATION_FOR_FAR_M = 2.40

# The look-at is biased toward the duck but pulled toward whatever it is
# investigating, so both stay in frame during every approach and observation.
LOOKAT_SUBJECT_BIAS = 0.66
# Applied once per CONTROL TICK, so this is a 0.44 s time constant at 50 Hz.
LOOKAT_EASE = 0.045
LOOKAT_Z = 0.36
EYE_WALL_MARGIN_M = 0.30
EYE_CLEARS_SCENE_Z = 2.20
AZIMUTH_SWING_DEG = 5.0
AZIMUTH_SWING_PERIOD_S = 34.0

# How much of the duck's recent path the plan view draws behind it.  Long
# enough that a whole diversion and return is visible as one curve.
TRAIL_TICKS = 1800


class FrameWriter:
    """Renders frames and streams them into ffmpeg.

    ``frames_dir`` is optional: when given, individual PNGs are also written so
    a contact sheet can select stills by time.  The default writes none, which
    is what keeps a 7000-frame render off the disk entirely.
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
        start = rollout.records[0]["duck_xy"] if rollout.records else (0.0, -0.9)
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
        """Ease the look-at toward the duck, or toward the pair.  ONE TICK.

        THIS RUNS ON EVERY CONTROL TICK, NOT ON EVERY WRITTEN FRAME, AND THAT IS
        THE WHOLE POINT.  An ease applied per written frame advances 4 times a
        second in a 4 fps preview and 50 times a second in the final render, so
        the two would fly different camera paths and the preview would stop
        being evidence about the video.
        """
        duck = np.array(record["duck_xy"], dtype=np.float64)
        subject_name = record["subject"]
        subject = record["actor_xy"].get(subject_name)
        if subject is None:
            target = np.array([duck[0], duck[1], LOOKAT_Z])
            separation = 0.0
        else:
            subject = np.array(subject, dtype=np.float64)
            target = np.array([
                LOOKAT_SUBJECT_BIAS * duck[0]
                + (1.0 - LOOKAT_SUBJECT_BIAS) * subject[0],
                LOOKAT_SUBJECT_BIAS * duck[1]
                + (1.0 - LOOKAT_SUBJECT_BIAS) * subject[1],
                LOOKAT_Z])
            separation = float(np.linalg.norm(duck - subject))
        self.lookat += LOOKAT_EASE * (target - self.lookat)

        wanted = CAM_DISTANCE_NEAR + (CAM_DISTANCE_FAR - CAM_DISTANCE_NEAR) * \
            min(max(separation / SEPARATION_FOR_FAR_M, 0.0), 1.0)
        self.distance += LOOKAT_EASE * (wanted - self.distance)

    def _aim_camera(self, record) -> None:
        """Point the free camera at the eased look-at, kept out of the scenery.

        The bound is DERIVED from the camera's own geometry rather than
        declared.  ``eye = lookat - forward * distance``, so the eye's HEIGHT is
        ``lookat_z + distance * sin(|elevation|)``.  When that clears
        :data:`EYE_CLEARS_SCENE_Z` the camera is flying over the whole facility
        and no horizontal clamp is needed.
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
