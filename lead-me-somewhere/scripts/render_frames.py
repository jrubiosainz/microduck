#!/usr/bin/env python3
"""Frame rendering: the wide shot, the head-camera PiP, the composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``GuideCamera.render_data`` — the ISOLATED copy in which the head has been posed
for this tick — so the gaze and the arrival gesture the viewer sees are the ones
that were measured, while the authoritative walking state stays untouched.

Importing this module is what pulls in ``imageio`` and ``PIL``; the headless
gate never imports it, so ``validate_guide.py`` has no rendering dependency at
all.  ``test_the_headless_gate_imports_no_rendering_stack`` pins that claim, and
this module is deliberately outside the list it checks.

THE CAMERA FOLLOWS THE PAIR, NOT THE DUCK
------------------------------------------
This behavior is about a RELATIONSHIP between two bodies that are 0.6-1.9 m
apart, and the interesting moments are exactly the ones where that gap opens.  A
look-at locked to the duck pushes her out of frame precisely when the video
needs to show her falling behind, which is the one thing it exists to show.  The
look-at is therefore the MIDPOINT of the duck and the follower, and the camera
distance opens with their separation so both stay in shot as the gap grows.

WHY THE LOOK-AT IS CLAMPED
---------------------------
A free camera orbiting a look-at near the perimeter puts its eye beyond the
wall, and MuJoCo then draws the near wall as a slab across the shot.  The bounds
are DERIVED from the camera's own geometry rather than declared: ``eye = lookat
- forward * distance``, so requiring the eye to stay inside the concourse is an
asymmetric box on the look-at.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from guide_cast import FOLLOWER
from guide_layout import FLOOR_HALF
from video_overlay import compose

# MEASURED by replaying the REAL recorded trace through this module's own
# easing, distance ramp and look-at clamp; see ``tools/probe_framing.py``.
# Replaced wholesale by the probe's answer, never nudged by eye.
#
#   azimuth 180, elevation -56, distance 3.15-4.35
#     -> duck on screen 1.000, duck unoccluded 0.937,
#        follower on screen 1.000, follower unoccluded 0.926,
#        camera eye inside the concourse 1.000,
#        duck clear of the HUD panels 0.958,
#        duck apparent height 15.2 px, pair separation 238 px
#
# THE ELEVATION IS THE MEASUREMENT THAT MATTERS, AND IT IS NOT A TASTE.
# This concourse is divided by two 2.05 m full-height slabs.  The first framing
# attempt inherited -27 deg from the sibling promenade behavior, which puts the
# camera eye at about 1.6 m — BELOW the partitions — so for every leg of the
# route that runs behind one, the shot is of the slab.  The preview showed
# exactly that.  The probe swept 360 candidates and EVERY result above 0.90
# duck-visibility sits at -56 deg, which lifts the eye to roughly 2.9 m and
# looks OVER both barriers.  There is no shallow angle that works in this hall,
# and the cost is a smaller duck (15 px against the promenade's 33), which is
# the right trade: a small duck you can see beats a large one behind a wall.
#
# ``follower unoccluded 0.926`` is accepted rather than chased: the remaining
# ticks are the ones where a crowd actor genuinely walks between the camera and
# her, which is what a populated hall looks like.
CAM_AZIMUTH = 180.0
CAM_ELEVATION = -56.0
# Distance opens with the duck-follower separation, so both stay in frame while
# the gap grows.  These are the ends of that ramp, and the probe scored the
# ramp rather than a fixed distance.
CAM_DISTANCE_NEAR = 3.15
CAM_DISTANCE_FAR = 4.35
# The separation at which the far distance is reached.  Matched to the MEASURED
# safety maximum, so the shot is widest exactly when the behavior is most
# stretched.
SEPARATION_FOR_FAR_M = 2.60

# The look-at is the MIDPOINT of the pair.  1.0 would track the duck alone and
# push her out of frame during every wait, which is the moment the video exists
# to show.
LOOKAT_DUCK_BIAS = 0.5
# Applied once per CONTROL TICK, so this is a 0.44 s time constant at 50 Hz.
LOOKAT_EASE = 0.045
# The look-at height.  Raised well above the duck's own 0.14 m trunk because
# the shot is steep: at -56 deg the frame's lower third is floor immediately in
# front of the camera, and a look-at at duck height puts the pair at the very
# bottom of frame where the timeline panel sits.  MEASURED on the preview: at
# 0.32 m the arrival — the payoff shot — had the duck behind the timeline; at
# 0.62 m the pair sits in the middle third for the whole run.
LOOKAT_Z = 0.62
# Margin held between the camera eye and the concourse wall, in metres — but
# only while the eye is BELOW the wall tops.
#
# THE CLAMP IS CONDITIONAL, AND THAT IS A DERIVATION RATHER THAN A LOOSENING.
# The sibling promenade behavior clamped the look-at unconditionally because its
# camera sat at -26 deg, which put the eye about 1.2 m up — inside the wall
# band, where stepping outside the hall makes MuJoCo draw the near wall as a
# slab across the lens.  This behavior's camera is at -56 deg, so the eye rides
# ``distance * sin(56 deg)`` = 2.6-3.6 m above the look-at, and with
# ``LOOKAT_Z = 0.62`` that is 3.2-4.2 m: FAR above the 1.24 m wall tops.
#
# Applying the inherited clamp there cost the arrival shot.  Near the LIFTS
# corner the unclamped look-at wants x = +2.7 m; the clamp forced it to +1.34 m,
# 1.4 m away, which slid the pair to the bottom of frame and put the payoff —
# the duck indicating at the destination — behind the timeline panel.  MEASURED
# on the preview at f00255 and f00270.
#
# So the eye is required to clear the SCENE rather than to stay inside the
# floor plan: inside the hall horizontally, OR above everything that could be
# drawn across it.
EYE_WALL_MARGIN_M = 0.30
# Wall top plus the tallest scenery, with headroom.  Above this the eye cannot
# be occluded by the perimeter however far outside the footprint it sits.
EYE_CLEARS_SCENE_Z = 2.60
# A slow lean, kept small because the framing was chosen at a fixed azimuth.
AZIMUTH_SWING_DEG = 4.0
AZIMUTH_SWING_PERIOD_S = 23.0

# How much of the duck's recent path the plan view draws behind it.
TRAIL_TICKS = 900
TRAIL_STRIDE = 25


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
        self.lookat = np.array([-2.5, -1.4, LOOKAT_Z], dtype=np.float64)
        self.distance = CAM_DISTANCE_NEAR
        self.pip_cam = rollout.camera.camera_id

        # Timeline content accumulated as the rollout runs, so a frame drawn at
        # t only ever shows events that had already happened by t.  A viewer
        # must never see a wait marked on the timeline before it happens.
        self.summary: dict = {"state_windows": [], "episodes": []}
        self._open_state: str | None = None
        self._episodes_seen = 0
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
        follower = np.array(record["follower_xy"], dtype=np.float64)
        target = np.array([
            LOOKAT_DUCK_BIAS * duck[0] + (1.0 - LOOKAT_DUCK_BIAS) * follower[0],
            LOOKAT_DUCK_BIAS * duck[1] + (1.0 - LOOKAT_DUCK_BIAS) * follower[1],
            LOOKAT_Z])
        self.lookat += LOOKAT_EASE * (target - self.lookat)

        # Open the shot as the pair separates, so a wait does not push her out
        # of frame at the exact moment the video is about her being behind.
        separation = float(record["follower_range_m"])
        wanted = CAM_DISTANCE_NEAR + (CAM_DISTANCE_FAR - CAM_DISTANCE_NEAR) * \
            min(max(separation / SEPARATION_FOR_FAR_M, 0.0), 1.0)
        self.distance += LOOKAT_EASE * (wanted - self.distance)

    def _aim_camera(self, record) -> None:
        """Point the free camera at the eased look-at, kept out of the scenery.

        The bound is DERIVED from the camera's own geometry rather than
        declared.  ``eye = lookat - forward * distance``, so the eye's HEIGHT is
        ``lookat_z + distance * sin(|elevation|)``.  When that clears
        :data:`EYE_CLEARS_SCENE_Z` the camera is flying over the whole hall and
        no horizontal clamp is needed — nothing can be drawn between it and the
        floor.  Below that height the eye must stay inside the footprint, which
        is the asymmetric box the sibling behavior derived.
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
        """Accumulate timeline content from the machine, up to this tick only.

        Read from the live machine rather than from a pre-computed summary, so a
        frame at t carries exactly the events the duck had decided by t.
        """
        state = record["state"]
        if state != self._open_state:
            self.summary["state_windows"].append(
                {"state": state, "start": record["t"], "end": record["t"]})
            self._open_state = state
        else:
            self.summary["state_windows"][-1]["end"] = record["t"]

        machine = self.rollout.machine
        while self._episodes_seen < len(machine.episodes):
            self.summary["episodes"].append(machine.episodes[self._episodes_seen])
            self._episodes_seen += 1
        # An episode in progress is shown as soon as it is DETECTED, because the
        # detection is the event the viewer needs to see, but its resume time is
        # only added once it exists.
        if machine._episode and machine._episode not in self.summary["episodes"]:
            live = dict(machine._episode)
            if not any(e.get("detected_at_s") == live.get("detected_at_s")
                       for e in self.summary["episodes"]):
                self.summary["episodes"].append(live)

    def write(self, index: int, record: dict) -> None:
        # Camera easing and the trail are updated on EVERY control tick, before
        # the frame rate is applied, so the 4 fps preview and the 50 fps final
        # fly the same camera path and draw the same trail.
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
