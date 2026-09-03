#!/usr/bin/env python3
"""Frame rendering: the wide shot, the head-camera PiP, the composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``GestureCamera.render_data`` - the ISOLATED copy in which the head has been
posed for this tick - so the gaze the viewer sees is the one that was measured,
while the authoritative walking state stays untouched.

Importing this module is what pulls in ``imageio`` and ``PIL``; the headless
gate never imports it, so ``validate_gesture.py`` has no rendering dependency at
all.  ``test_the_headless_gate_imports_no_rendering_stack`` pins that claim by
blocking those modules in ``sys.meta_path``, and this module is deliberately
outside the list it checks.

FRAMES ARE STREAMED TO ffmpeg, NOT STAGED ON DISK
---------------------------------------------------
An 87 s run at 50 fps is 4360 frames, which as PNGs is well over a gigabyte of
scratch.  :class:`FrameWriter` therefore pipes raw RGB straight into an ffmpeg
process and never writes an intermediate image, which also removes the
encode/decode round trip through PNG.  A frame directory can still be requested
with ``--frames`` when individual stills are wanted for a contact sheet.

THE CAMERA HOLDS BOTH THE DUCK AND THE INSTRUCTOR
---------------------------------------------------
In this behavior the subject is not the robot alone - it is the robot AND the
person giving it commands, because a video that showed only the duck could not
show what it was responding to.  The look-at therefore sits between them for the
whole session, and the distance opens with their separation so both stay in
shot when the duck is at its farthest.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from gest_arena import FLOOR_HALF
from gest_markers import TRAIL_STRIDE
from video_overlay import compose

# The wide camera.  MEASURED by ``tools/probe_framing.py``, which replays the
# REAL recorded trace through this module's own easing and distance ramp and
# scores every candidate on five things, rejecting outright any that hides
# either subject.  Replaced wholesale by the probe's answer, never adjusted by
# eye.
#
#   azimuth 38, elevation -36, distance 4.20-5.00, look-at bias 0.50
#     -> duck on screen 1.000, duck unoccluded 0.961, instructor on screen AND
#        clear of the HUD columns 0.952, duck 29.3 px across at the median, and
#        the gesturing arm 34.6 px long projected - the longest arm of any
#        candidate that keeps both subjects visible
#
# THE CONSTRAINT THAT DECIDES THIS IS THE LOOK-AT CLAMP, AND FINDING IT COST
# TWO WRONG ANSWERS.  ``_aim_camera`` below clamps the look-at whenever the eye
# would sit below :data:`EYE_CLEARS_SCENE_Z`, so the camera is never placed
# inside a wall.  The sibling patrol flies at -52 deg, where the eye is 3.2 m
# up and the clamp never fires - so it was invisible there.  At the shallow
# elevations this behavior first tried for arm legibility (-18 deg, eye 1.5 m)
# the clamp fires on nearly every tick and shifts the look-at by a whole camera
# distance, which pushed the instructor out to x = 32 px, behind the opaque
# left HUD column, for the entire COME and STOP.
#
# That was VERIFIED rather than guessed: rendering a MuJoCo segmentation frame
# with the same camera put her centroid at (32, 320) against the probe's
# analytic (21, 280) - agreement to about 10 px, so the probe was right and the
# preview frames had simply not been looked at closely enough.
#
# The probe now only offers candidates whose eye clears the walls at the near
# distance, so the clamp is provably inert.  Within that set the choice is a
# real trade: a shallower camera makes the arm longer, a higher one keeps both
# subjects inside the clear band between the HUD columns.  -36 deg at 4.20 m is
# where the arm is longest subject to both staying visible.
CAM_AZIMUTH = 38.0
CAM_ELEVATION = -36.0
CAM_DISTANCE_NEAR = 4.20
CAM_DISTANCE_FAR = 5.00
# The separation at which the far distance is reached, matched to the widest
# duck-to-instructor gap the run actually produces (2.90 m at the start).
SEPARATION_FOR_FAR_M = 2.90

# The look-at sits exactly BETWEEN the duck and the instructor.  Unlike a
# patrol, where the robot alone is the subject, here the pair is: a frame
# without her cannot show what the duck was responding to.  0.50 is the probe's
# own answer rather than a preference.
LOOKAT_SUBJECT_BIAS = 0.50
# Applied once per CONTROL TICK, so this is a 0.44 s time constant at 50 Hz.
LOOKAT_EASE = 0.045
LOOKAT_Z = 0.42
EYE_WALL_MARGIN_M = 0.30
EYE_CLEARS_SCENE_Z = 2.20
AZIMUTH_SWING_DEG = 4.0
AZIMUTH_SWING_PERIOD_S = 40.0

# How much of the duck's recent path the plan view draws behind it.  Long
# enough that the approach, both arcs and the reverse read as one curve.
TRAIL_TICKS = 2600


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
        start = rollout.records[0]["duck_xy"] if rollout.records else (0.0, -1.6)
        self.lookat = np.array([start[0], start[1], LOOKAT_Z], dtype=np.float64)
        self.distance = CAM_DISTANCE_NEAR
        self.pip_cam = rollout.camera.camera_id

        # Timeline content accumulated as the rollout runs, so a frame drawn at
        # t only ever shows what had already happened by t.  A viewer must never
        # see a state marked on the timeline before it happens.
        self.history: list[dict] = []
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
        # THE INSTRUCTOR IS ALWAYS THE SECOND SUBJECT, whether or not she is
        # locked yet: the video has to show what the duck is being told from the
        # first frame, including the search that precedes the lock.
        instructor = record["bodies"].get(record["instructor"], {}).get("xy")
        if instructor is None:
            target = np.array([duck[0], duck[1], LOOKAT_Z])
            separation = 0.0
        else:
            instructor = np.array(instructor, dtype=np.float64)
            target = np.array([
                LOOKAT_SUBJECT_BIAS * duck[0]
                + (1.0 - LOOKAT_SUBJECT_BIAS) * instructor[0],
                LOOKAT_SUBJECT_BIAS * duck[1]
                + (1.0 - LOOKAT_SUBJECT_BIAS) * instructor[1],
                LOOKAT_Z])
            separation = float(np.linalg.norm(duck - instructor))
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
                        expected=self.expected,
                        refusals=self.refusals,
                        interrupts=self.rollout.machine.interrupts,
                        history=self.history,
                        trail=self._trail[::TRAIL_STRIDE])
        frame = np.asarray(image, dtype=np.uint8)
        self._ffmpeg.stdin.write(frame.tobytes())
        if self.frames_dir is not None:
            imageio.imwrite(self.frames_dir / f"f{self.frames:05d}.png", frame)
        self.manifest.append({"frame": self.frames, "t": record["t"],
                              "state": record["state"]})
        self.frames += 1

    @property
    def refusals(self) -> list[dict]:
        """Every logged rejection so far, for the REFUSED panel."""
        return self.rollout.detector.rejections

    @property
    def expected(self) -> list[str]:
        from gest_script import EXPECTED_COMMANDS
        return list(EXPECTED_COMMANDS)

    def close(self) -> int:
        """Finish the encode and return ffmpeg's exit status."""
        self._ffmpeg.stdin.close()
        return self._ffmpeg.wait()

    def write_manifest(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest))
        return path
