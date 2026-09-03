#!/usr/bin/env python3
"""Frame rendering: wide shot, head-camera PiP, composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``GuardianCamera.render_data`` — the ISOLATED copy in which the head has been
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

from traffic import crossing_arrivals
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
        self.camera.distance = 4.2
        self.camera.elevation = -24
        self.camera.azimuth = 152
        self.lookat = np.array([0.0, 0.0, 0.25], dtype=np.float64)
        self.guardian_cam = rollout.camera.camera_id
        # Precomputed once: the traffic timeline is a property of the schedule,
        # not of the rollout, so it does not need recomputing per frame.
        self.summary = {
            "arrivals": crossing_arrivals(rollout.seconds),
            "crossing_window": None,
        }

    def _frame_camera(self, record) -> None:
        """Frame the crossing, keeping the duck and the live traffic in shot.

        The shot has to show two things at once: the duck's position relative
        to the paint, and the vehicles bearing down on it.  Following the duck
        alone loses the traffic; a fixed wide shot makes the duck a speck.

        The camera therefore looks at a point between the duck and the crossing
        centre, biased a little up the road toward whichever approaching
        vehicle is nearest, and pulls back when that vehicle is still far away
        so the viewer can see it coming.
        """
        duck = np.array(record["duck_xy"], dtype=np.float64)
        # The nearest road user that has not yet passed the crossing, and which
        # side of the crossing it is approaching from.
        nearest = 9.0
        nearest_y = 0.0
        for vx, vy in record["vehicle_xy"].values():
            # far lane (x>0) travels +y, so an approaching one has y<0
            incoming = vy < 0.0 if vx > 0.0 else vy > 0.0
            if incoming and abs(vy) < nearest:
                nearest = abs(vy)
                nearest_y = vy
        # Lead the shot toward the approaching vehicle — but the lead is
        # anchored on the DUCK, not on the world origin.
        #
        # MEASURED FROM THE PREVIEW: leading from the origin put the look-at at
        # y = -1.10 while the duck sat at y = +0.32, an offset of 1.51 m at a
        # 4.2 m camera distance, and the duck left frame entirely for most of
        # the SAFE tail.  The lead is now relative to the duck and capped at
        # 0.75 m, which keeps the approaching vehicle in shot without ever
        # letting the subject slide out of it.
        lead_y = float(np.clip(0.30 * (nearest_y - duck[1]), -0.75, 0.75))
        target = np.array([0.60 * duck[0], duck[1] + lead_y, 0.25])
        self.lookat += 0.05 * (target - self.lookat)
        self.camera.lookat[:] = self.lookat
        distance = float(np.clip(3.5 + 0.15 * nearest, 3.6, 4.8))
        self.camera.distance += 0.04 * (distance - self.camera.distance)
        # Slow orbit so the street reads as three-dimensional.
        self.camera.azimuth = 152.0 + 10.0 * np.sin(record["t"] / 13.0)

    def write(self, index: int, record: dict) -> None:
        if index % self.every:
            return
        # Keep the timeline's crossing bar in step with the machine.
        commit = self.rollout.machine.commit
        if commit.get("committed_at_s") is not None:
            self.summary["crossing_window"] = [
                commit["committed_at_s"],
                commit.get("arrived_at_s", record["t"]),
            ]
        gaze = self.rollout.camera.render_data
        self._frame_camera(record)
        self.renderer.update_scene(gaze, camera=self.camera)
        main = self.renderer.render()
        self.pip_renderer.update_scene(gaze, camera=self.guardian_cam)
        pip = self.pip_renderer.render()

        image = compose(
            main, pip,
            record=record,
            total_seconds=self.rollout.seconds,
            machine_summary=self.summary,
        )
        imageio.imwrite(self.out / f"f{self.frames:05d}.png", np.asarray(image))
        self.frames += 1
