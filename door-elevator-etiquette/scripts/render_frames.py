#!/usr/bin/env python3
"""Frame rendering: the wide shot, the head-camera PiP, the composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``EtiquetteCamera.render_data`` - the ISOLATED copy in which the head has been
posed for this tick - so the gaze the viewer sees is the one that was measured,
while the authoritative walking state stays untouched.

Importing this module is what pulls in ``imageio`` and ``PIL``; the headless
gate never imports it, so ``validate_etiquette.py`` has no rendering dependency
at all.  ``test_the_headless_gate_imports_no_rendering_stack`` pins that claim
by blocking those modules in ``sys.meta_path``, and this module is deliberately
outside the list it checks.

THE CAMERA FOLLOWS THE PAIR, NOT THE DUCK
------------------------------------------
This behavior is about a RELATIONSHIP between the duck and the person ahead of
it, and the interesting moments are exactly the ones where the gap opens: the
duck stopped outside a doorway while she walks through it, the duck beside a
lift while she stands in it.  A look-at locked to the duck pushes her out of
frame precisely when the video needs to show both.  The look-at is therefore the
MIDPOINT of the duck and its current subject, and the camera distance opens with
their separation.

THE ELEVATION IS A MEASUREMENT, NOT A TASTE
--------------------------------------------
The building is divided by 1.35 m partitions and the lift car has 1.15 m walls,
so a shallow camera films the outside of a box for the entire second half of the
run.  ``tools/probe_framing.py`` replays the REAL recorded trace through this
module's own easing and scores candidates on whether the duck and its subject
are on screen and unoccluded in 3D against every wall, leaf and person.  The
constants below are that probe's answer, replaced wholesale rather than nudged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from etiquette_markers import TRAIL_STRIDE
from lobby_layout import FLOOR_HALF
from video_overlay import compose

# MEASURED by ``tools/probe_framing.py`` on the real trace, replayed through
# this module's own easing and distance ramp.  See the module docstring:
# replaced wholesale by the probe's answer, never adjusted by eye.
#
#   azimuth 30, elevation -66, distance 3.90-5.20
#     -> duck on screen 1.000, duck unoccluded 1.000,
#        subject on screen 0.990, subject unoccluded 0.980,
#        camera eye clear of the scenery 1.000,
#        duck clear of the HUD panels 1.000,
#        pair spread 0.159 frame widths
#
# THE ELEVATION IS THE MEASUREMENT THAT MATTERS, AND IT IS NOT A TASTE.
# The building is divided by 1.35 m partitions and the car has 1.15 m walls, so
# a shallow camera films the outside of a box for the whole lift sequence.  The
# sweep was extended TWICE because its optimum kept landing on the edge of the
# grid - a grid whose best result sits at a boundary has found the edge of the
# grid, not an optimum - and every candidate at or above 0.98 duck-visibility
# sits at -66 deg or steeper.
#
# THE AZIMUTH IS CHOSEN ON COMPOSITION, BECAUSE VISIBILITY DOES NOT DECIDE IT.
# Once the camera clears the partitions, many azimuths reach 1.000 duck
# visibility together.  What separates them is where the OTHER body falls in
# frame: at 30 deg the pair sits 0.159 frame widths apart, against 0.083 at
# 330 deg where the route stacks them front-to-back and they overlap.
CAM_AZIMUTH = 30.0
CAM_ELEVATION = -66.0
CAM_DISTANCE_NEAR = 3.90
CAM_DISTANCE_FAR = 5.20
# The separation at which the far distance is reached.  Matched to the widest
# duck-to-subject gap the run actually produces, so the shot is widest exactly
# when the behavior is most stretched.
SEPARATION_FOR_FAR_M = 3.20

# The look-at is the MIDPOINT of the duck and whoever it is watching.  1.0 would
# track the duck alone and push the other body out of frame during every yield,
# which is the moment the video exists to show.
LOOKAT_SUBJECT_BIAS = 0.5
# Applied once per CONTROL TICK, so this is a 0.44 s time constant at 50 Hz.
LOOKAT_EASE = 0.045
# The look-at height.  Raised above the duck's own 0.12 m trunk because the shot
# is steep: at -66 deg the frame's lower third is floor immediately in front of
# the camera, and a look-at at duck height puts the pair at the very bottom of
# frame where the timeline panel sits.
LOOKAT_Z = 0.58
# Margin held between the camera eye and the perimeter, while the eye is below
# the wall tops.  Above ``EYE_CLEARS_SCENE_Z`` the eye is flying over everything
# that could be drawn across it and no horizontal clamp is needed.  At -66 deg
# and 3.90-5.20 m the eye rides 3.6-4.8 m above the look-at, so it is ALWAYS
# above this and the clamp never fires - which the probe confirms with an
# eye-clear score of 1.000.
EYE_WALL_MARGIN_M = 0.30
EYE_CLEARS_SCENE_Z = 2.20
# A slow lean, kept small because the framing was chosen at a fixed azimuth.
AZIMUTH_SWING_DEG = 5.0
AZIMUTH_SWING_PERIOD_S = 26.0

# How much of the duck's recent path the plan view draws behind it.
TRAIL_TICKS = 900


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
        self.camera.distance = CAM_DISTANCE_NEAR
        self.camera.elevation = CAM_ELEVATION
        self.camera.azimuth = CAM_AZIMUTH
        start = rollout.records[0]["duck_xy"] if rollout.records else (-3.3, -0.4)
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

    def _advance_camera(self, record) -> None:
        """Ease the look-at toward the pair's midpoint.  ONE CONTROL TICK.

        THIS RUNS ON EVERY CONTROL TICK, NOT ON EVERY WRITTEN FRAME, AND THAT IS
        THE WHOLE POINT.  An ease applied per written frame advances 4 times a
        second in a 4 fps preview and 50 times a second in the final render, so
        the two would fly different camera paths and the preview would stop
        being evidence about the video.
        """
        duck = np.array(record["duck_xy"], dtype=np.float64)
        subject = np.array(
            record["person_xy"].get(record["subject"], record["duck_xy"]),
            dtype=np.float64)
        target = np.array([
            LOOKAT_SUBJECT_BIAS * duck[0] + (1.0 - LOOKAT_SUBJECT_BIAS) * subject[0],
            LOOKAT_SUBJECT_BIAS * duck[1] + (1.0 - LOOKAT_SUBJECT_BIAS) * subject[1],
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
        :data:`EYE_CLEARS_SCENE_Z` the camera is flying over the whole building
        and no horizontal clamp is needed - nothing can be drawn between it and
        the floor.  Below that height the eye must stay inside the footprint.
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

        route_points = [p.tolist() for p in self.rollout._route_points]
        image = compose(main, pip, record=record,
                        total_seconds=self.rollout.seconds,
                        summary=self.summary,
                        trail=self._trail[::TRAIL_STRIDE],
                        route_points=route_points)
        imageio.imwrite(self.out / f"f{self.frames:05d}.png", np.asarray(image))
        self.manifest.append({"frame": self.frames, "t": record["t"],
                              "state": record["state"]})
        self.frames += 1

    def write_manifest(self) -> Path:
        path = self.out / "frames.json"
        path.write_text(json.dumps(self.manifest))
        return path
