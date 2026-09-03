#!/usr/bin/env python3
"""Frame rendering: wide shot, head-camera PiP, composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``QueueCamera.render_data`` - the ISOLATED copy in which the head has been posed
- so the gaze the viewer sees is the gaze that was measured, while the
authoritative walking state stays untouched.

Importing this module is what pulls in ``imageio`` and ``PIL``; the no-render
gate never imports it, so a headless validation run has no rendering
dependencies at all.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from queue_people import departures
from video_overlay import compose


class FrameWriter:
    """Renders and writes one PNG per output frame."""

    def __init__(self, rollout, args, pip_w: int, pip_h: int):
        self.rollout = rollout
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.every = max(1, int(round(rollout.dt ** -1 / args.fps)))

        model = rollout.model
        self.renderer = mujoco.Renderer(model, height=args.height,
                                        width=args.width)
        self.pip_renderer = mujoco.Renderer(model, height=pip_h, width=pip_w)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        # A HAIRPIN QUEUE IS A HARD THING TO FILM.  A low angle hides the fold
        # behind the people standing on it, and a shot from directly above
        # turns a 25 cm robot into a dot.  These parameters are MEASURED rather
        # than eyeballed: ``tools/probe_camera.py`` projects the duck and five
        # queue stations into pixel coordinates for every candidate camera and
        # scores how often the duck is on screen AND clear of the HUD panels.
        # The first draft (azimuth 62, distance 4.05) left the duck underneath
        # the left-hand panels for 89 of 290 sampled frames; this one scores
        # clear_fraction = 1.000 with queue_visible_mean = 1.000.
        self.camera.distance = 4.50
        self.camera.elevation = -34
        self.camera.azimuth = 38
        self.lookat = np.array([0.30, -0.62, 0.22], dtype=np.float64)
        self.pip_cam = rollout.camera.camera_id
        # The timeline's fixed content: the service schedule is a property of
        # the scenario, not of the rollout, so it is computed once.
        self.summary = {
            "services": departures(rollout.seconds),
            "state_windows": [],
        }
        self._open_state: str | None = None

    def _frame_camera(self, record) -> None:
        """Hold the whole queue in shot, easing toward the duck.

        A pure follow-cam on a 25 cm robot inside a 4.6 m queue loses the
        queue, which is the subject; a locked-off wide shot loses the duck.
        The look-at therefore sits between the lane's centre and the duck,
        biased toward the duck, and eases rather than cutting.
        """
        duck = np.array(record["duck_xy"], dtype=np.float64)
        target = np.array([0.55 * 0.30 + 0.45 * duck[0],
                           0.55 * -0.62 + 0.45 * duck[1], 0.22])
        self.lookat += 0.05 * (target - self.lookat)
        self.camera.lookat[:] = self.lookat
        # A slow lean so the hall reads as three-dimensional rather than flat,
        # kept small because the probe scored the framing at a fixed azimuth.
        self.camera.azimuth = 38.0 + 4.0 * np.sin(record["t"] / 13.0)

    def write(self, index: int, record: dict) -> None:
        if index % self.every:
            return
        state = record["state"]
        if state != self._open_state:
            self.summary["state_windows"].append(
                {"state": state, "start": record["t"], "end": record["t"]})
            self._open_state = state
        else:
            self.summary["state_windows"][-1]["end"] = record["t"]

        gaze = self.rollout.camera.render_data
        self._frame_camera(record)
        self.renderer.update_scene(gaze, camera=self.camera)
        main = self.renderer.render()
        self.pip_renderer.update_scene(gaze, camera=self.pip_cam)
        pip = self.pip_renderer.render()

        image = compose(main, pip, record=record,
                        total_seconds=self.rollout.seconds,
                        machine_summary=self.summary)
        imageio.imwrite(self.out / f"f{self.frames:05d}.png", np.asarray(image))
        self.frames += 1
