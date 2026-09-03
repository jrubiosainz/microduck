#!/usr/bin/env python3
"""Frame rendering: corridor shot, head-camera PiP, composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``EtiquetteCamera.render_data`` — the ISOLATED copy in which the head has been
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

from corridor import (
    CORRIDOR_X_MAX,
    CORRIDOR_X_MIN,
    corridor_passing_geometry,
)
from people import corridor_passes
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
        # A NARROW CORRIDOR IS A HARD THING TO FILM.  A side-on shot is blocked
        # by the near wall; a shot from directly above loses the walls entirely
        # and with them the whole point of the scene.  The camera therefore sits
        # high and behind, looking down the corridor at a steep angle, which
        # keeps both walls, both alcove mouths and the floor paint in frame.
        self.camera.distance = 2.15
        self.camera.elevation = -38
        self.camera.azimuth = 158
        self.lookat = np.array([rollout.records[0]["duck_xy"][0]
                                if rollout.records else -1.95,
                                0.0, 0.20], dtype=np.float64)
        self.pip_cam = rollout.camera.camera_id
        # Precomputed once: the pedestrian timeline is a property of the
        # schedule, not of the rollout, so it does not need recomputing per
        # frame.
        self.summary = {
            "passes": corridor_passes(
                rollout.seconds, x_low=CORRIDOR_X_MIN, x_high=CORRIDOR_X_MAX),
            "yield_windows": [],
            "passing_geometry": corridor_passing_geometry(),
        }
        self._yield_open: float | None = None

    def _frame_camera(self, record) -> None:
        """Follow the duck down the corridor, leaning toward the approaching adult.

        The shot has to show two things at once: where the duck's footprint is
        relative to the paint, and the person bearing down on it.  Following the
        duck alone loses the adult; a fixed wide shot makes a 25 cm robot a
        speck in a 6 m corridor.

        The look-at therefore sits between the duck and whichever adult is
        closest, biased toward the duck, and the camera pulls back a little
        while that adult is still far away so the viewer can see them coming.
        """
        duck = np.array(record["duck_xy"], dtype=np.float64)
        nearest = 9.0
        nearest_x = duck[0]
        for name, (ax, ay) in record["person_xy"].items():
            gap = abs(ax - duck[0])
            if gap < nearest:
                nearest = gap
                nearest_x = ax
        lead = float(np.clip(0.35 * (nearest_x - duck[0]), -0.55, 0.55))
        target = np.array([duck[0] + lead, 0.30 * duck[1], 0.20])
        self.lookat += 0.06 * (target - self.lookat)
        self.camera.lookat[:] = self.lookat
        distance = float(np.clip(1.95 + 0.16 * nearest, 2.0, 2.75))
        self.camera.distance += 0.05 * (distance - self.camera.distance)
        # A slow lean so the corridor reads as three-dimensional rather than as
        # a flat diagram, kept small because a big orbit in a tight corridor
        # swings the near wall across the whole frame.
        self.camera.azimuth = 158.0 + 7.0 * np.sin(record["t"] / 11.0)

    def write(self, index: int, record: dict) -> None:
        if index % self.every:
            return
        # Keep the timeline's yield bars in step with the machine.
        if record["state"] == "YIELD" and self._yield_open is None:
            self._yield_open = record["t"]
            self.summary["yield_windows"].append([record["t"], record["t"]])
        elif record["state"] != "YIELD" and self._yield_open is not None:
            self._yield_open = None
        if self._yield_open is not None and self.summary["yield_windows"]:
            self.summary["yield_windows"][-1][1] = record["t"]

        gaze = self.rollout.camera.render_data
        self._frame_camera(record)
        self.renderer.update_scene(gaze, camera=self.camera)
        main = self.renderer.render()
        self.pip_renderer.update_scene(gaze, camera=self.pip_cam)
        pip = self.pip_renderer.render()

        image = compose(
            main, pip,
            record=record,
            total_seconds=self.rollout.seconds,
            machine_summary=self.summary,
        )
        imageio.imwrite(self.out / f"f{self.frames:05d}.png", np.asarray(image))
        self.frames += 1
