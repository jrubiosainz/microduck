#!/usr/bin/env python3
"""Frame rendering: wide shot, attention-camera PiP, composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``AttentionCamera.render_data`` — the ISOLATED copy in which the head has been
posed — so the gaze the viewer sees is the gaze that was measured, while the
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

from video_overlay import compose


class FrameWriter:
    """Renders and writes one PNG per output frame."""

    def __init__(self, rollout, args, pip_w: int, pip_h: int, *,
                 calls, expected_order):
        self.rollout = rollout
        self.args = args
        self.calls = calls
        self.expected_order = expected_order
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.every = max(1, int(round(rollout.dt ** -1 / args.fps)))

        model = rollout.model
        self.renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        self.pip_renderer = mujoco.Renderer(model, height=pip_h, width=pip_w)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 3.8
        self.camera.elevation = -22
        self.camera.azimuth = 128
        self.lookat = np.array([0.0, 0.0, 0.30], dtype=np.float64)
        self.attention_cam = rollout.camera.camera_id

    def _frame_camera(self, record) -> None:
        """Keep the duck and the current caller both inside the wide shot.

        The camera follows a smoothed midpoint between the duck and whoever is
        calling, and pulls back far enough to hold both plus the surrounding
        people.  Without this the duck drifts to a corner during long
        approaches and the caller leaves frame entirely.
        """
        duck = np.array(record["duck_xy"], dtype=np.float64)
        focus = duck
        distance = 3.8
        subject = record["locked"] or record["caller"]
        if subject is not None:
            adult = self.rollout.camera.render_data.mocap_pos[
                self.rollout.camera.people[subject]
            ][:2]
            focus = 0.5 * (duck + adult)
            separation = float(np.linalg.norm(adult - duck))
            distance = float(np.clip(2.9 + 1.15 * separation, 3.3, 5.0))
        target = np.array([focus[0], focus[1], 0.30])
        self.lookat += 0.05 * (target - self.lookat)
        self.camera.lookat[:] = self.lookat
        self.camera.distance += 0.05 * (distance - self.camera.distance)
        # Slow orbit so the scene reads as three-dimensional.
        self.camera.azimuth = 128.0 + 9.0 * np.sin(record["t"] / 12.0)

    def write(self, index: int, record: dict) -> None:
        if index % self.every:
            return
        gaze = self.rollout.camera.render_data
        self._frame_camera(record)
        self.renderer.update_scene(gaze, camera=self.camera)
        main = self.renderer.render()
        self.pip_renderer.update_scene(gaze, camera=self.attention_cam)
        pip = self.pip_renderer.render()

        image = compose(
            main, pip,
            record=record,
            total_seconds=self.rollout.seconds,
            cycles=self.rollout.machine.cycles,
            cycles_target=len(self.expected_order),
            calls=self.calls,
            expected_order=self.expected_order,
        )
        imageio.imwrite(self.out / f"f{self.frames:05d}.png", np.asarray(image))
        self.frames += 1
