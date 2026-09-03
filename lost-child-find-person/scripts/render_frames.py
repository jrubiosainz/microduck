#!/usr/bin/env python3
"""Frame rendering: the wide shot, the head-camera PiP, the composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``LostCamera.render_data`` — the ISOLATED copy in which the head has been posed
— so the gaze the viewer sees is the gaze that was measured, while the
authoritative walking state stays untouched.

Importing this module is what pulls in ``imageio`` and ``PIL``; the headless
gate never imports it, so ``validate_lost.py`` has no rendering dependency at
all.  ``test_no_render_imports`` pins that claim.

THE MAIN CAMERA IS MEASURED, NOT EYEBALLED
-------------------------------------------
A 25 cm robot in a 6.6 x 4.6 m hall is easy to lose, and the one thing this
video may never do is hide the duck behind the very kiosk whose occlusion it is
demonstrating.  The framing below was chosen by replaying the REAL recorded
trace through candidate cameras — with THIS module's easing, swing and clamp —
and scoring, per sampled frame, whether the duck was on screen, clear of the HUD
panels, and unoccluded in 3D against the real solid volumes, plus whether the
guardian and the kiosk were in shot and how large the duck actually appeared.

    azimuth 210, elevation -40, distance 3.8, bias 0.78, clamp (2.0, 1.5)
      -> duck unoccluded 1.000, duck clear of the panels 0.992,
         guardian unoccluded 1.000, kiosk in shot 1.000,
         duck apparent height 50.9 px

THE WALLS ARE WHY THE FIRST CHOICE WAS WRONG.  An earlier probe scored only the
obstacles and picked azimuth 290, which scored a perfect 1.000 on paper and then
put the hall's own 1.24 m north wall between the camera and the duck for the
whole second cycle.  The scoring above includes the four walls as solid volumes,
which is what moved the answer to azimuth 210.  Elevation -34 with distance 3.6
makes the duck larger still (54 px) but drops guardian visibility to 0.500, so
it is rejected: the guardian must be on screen for the follow and rejoin phases
to be gradeable at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from video_overlay import compose

# MEASURED by replaying the recorded trace; see the module docstring.
CAM_AZIMUTH = 210.0
CAM_ELEVATION = -40.0
CAM_DISTANCE = 3.8
# The look-at eases between the duck and the guardian, biased toward the duck,
# so both are held in frame while the subject of the shot stays the robot.
LOOKAT_BIAS = 0.78
LOOKAT_EASE = 0.045
LOOKAT_Z = 0.28
# The look-at is clamped inside the hall.  Without it the guardian's final
# station near the west wall drags the camera far enough back that the kiosk
# leaves the shot on 3.3% of frames; with it, the kiosk is in shot on all of
# them and nothing else changes.
LOOKAT_CLAMP_X = 2.0
LOOKAT_CLAMP_Y = 1.5
# A slow lean, kept small because the framing was scored at a fixed azimuth.
AZIMUTH_SWING_DEG = 3.0
AZIMUTH_SWING_PERIOD_S = 17.0


class FrameWriter:
    """Renders and writes one PNG per output frame."""

    def __init__(self, rollout, args, pip_w: int, pip_h: int):
        self.rollout = rollout
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        # Control ticks per output frame.
        self.every = max(1, int(round((1.0 / rollout.dt) / args.fps)))

        model = rollout.model
        self.renderer = mujoco.Renderer(model, height=args.height,
                                        width=args.width)
        self.pip_renderer = mujoco.Renderer(model, height=pip_h, width=pip_w)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = CAM_DISTANCE
        self.camera.elevation = CAM_ELEVATION
        self.camera.azimuth = CAM_AZIMUTH
        self.lookat = np.array([0.3, 0.0, LOOKAT_Z], dtype=np.float64)
        self.pip_cam = rollout.camera.camera_id

        # Timeline content accumulated as the rollout runs, so a frame drawn at
        # t only ever shows events that had already happened by t.
        self.summary: dict = {"state_windows": [], "losses": [],
                              "reacquisitions": []}
        self._open_state: str | None = None
        # PRESENTATION MEMORY, not behavior state.  The identity tracker clears
        # its sighting the moment a candidate is refused and put on cooldown,
        # so a frame drawn during REJECT has no sighting at all and the panel
        # would read "no candidate in view" underneath the word REFUSED.  The
        # writer therefore remembers the last sighting it saw OF EACH BODY and
        # hands the overlay the one belonging to the candidate currently being
        # refused.  Keyed by name rather than kept as a single slot because the
        # sweep frequently picks up the NEXT candidate while the previous one is
        # still inside its refusal hold — at t=21.76 the tracker is already
        # scoring faruq while mira is the subject — and a single slot would then
        # draw faruq's evidence under mira's name.  Nothing here is fed back
        # into the rollout.
        self._sighting_by_name: dict[str, dict] = {}
        # frame number -> the tick time and state it was drawn from.  The
        # contact sheet selects by TIME through this manifest rather than by
        # assuming frame == t * fps, which is false whenever the control rate is
        # not an exact multiple of the frame rate: at 4 fps the writer emits
        # every 12th tick, so frame 200 is t = 48.02 s and not t = 50.00 s.
        self.manifest: list[dict] = []

    def _frame_camera(self, record) -> None:
        """Hold the duck and the guardian in shot, easing rather than cutting.

        The easing state advances on every WRITTEN frame, which is what the
        framing probe replayed, so the measured scores describe this path.
        """
        duck = np.array(record["duck_xy"], dtype=np.float64)
        guardian = np.array(
            record["person_xy"][record["guardian"]], dtype=np.float64)
        target = np.array([
            LOOKAT_BIAS * duck[0] + (1.0 - LOOKAT_BIAS) * guardian[0],
            LOOKAT_BIAS * duck[1] + (1.0 - LOOKAT_BIAS) * guardian[1],
            LOOKAT_Z])
        self.lookat += LOOKAT_EASE * (target - self.lookat)
        self.camera.lookat[0] = float(np.clip(
            self.lookat[0], -LOOKAT_CLAMP_X, LOOKAT_CLAMP_X))
        self.camera.lookat[1] = float(np.clip(
            self.lookat[1], -LOOKAT_CLAMP_Y, LOOKAT_CLAMP_Y))
        self.camera.lookat[2] = float(self.lookat[2])
        self.camera.azimuth = CAM_AZIMUTH + AZIMUTH_SWING_DEG * np.sin(
            record["t"] / AZIMUTH_SWING_PERIOD_S)

    def _note_events(self, record) -> None:
        state = record["state"]
        if state != self._open_state:
            self.summary["state_windows"].append(
                {"state": state, "start": record["t"], "end": record["t"]})
            self._open_state = state
            if state == "LOST":
                self.summary["losses"].append(record["t"])
            if state == "REACQUIRED":
                self.summary["reacquisitions"].append(record["t"])
        else:
            self.summary["state_windows"][-1]["end"] = record["t"]

    def write(self, index: int, record: dict) -> None:
        # Presentation memory is updated on EVERY control tick, before the frame
        # rate is applied.  A candidate can be scored for as few as three ticks
        # — mira is sighted only over t=21.46..21.50 — which a 4 fps preview
        # sampling every twelfth tick would step straight over, leaving her
        # refusal panel blank in the preview and populated in the final render.
        # Observing the stream at full rate makes the overlay identical at any
        # frame rate.
        if record.get("sighting"):
            self._sighting_by_name[record["sighting"]["name"]] = record["sighting"]
        if index % self.every:
            return
        self._note_events(record)

        # Both views are rendered from the camera's isolated render_data, in
        # which the head has been posed for THIS tick.
        gaze = self.rollout.camera.render_data
        self._frame_camera(record)
        self.renderer.update_scene(gaze, camera=self.camera)
        main = self.renderer.render()
        self.pip_renderer.update_scene(gaze, camera=self.pip_cam)
        pip = self.pip_renderer.render()

        image = compose(main, pip, record=record,
                        total_seconds=self.rollout.seconds,
                        summary=self.summary,
                        last_sighting=self._sighting_by_name.get(
                            record.get("subject")))
        imageio.imwrite(self.out / f"f{self.frames:05d}.png", np.asarray(image))
        self.manifest.append({"frame": self.frames, "t": record["t"],
                              "state": record["state"]})
        self.frames += 1

    def write_manifest(self) -> Path:
        path = self.out / "frames.json"
        path.write_text(json.dumps(self.manifest))
        return path
