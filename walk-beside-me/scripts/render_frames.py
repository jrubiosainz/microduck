#!/usr/bin/env python3
"""Frame rendering: the wide shot, the head-camera PiP, the composed overlay.

Rendering is deliberately isolated from the rollout.  Every frame is drawn from
``BesideCamera.render_data`` — the ISOLATED copy in which the head has been
posed for this tick — so the gaze the viewer sees is the gaze that was measured,
while the authoritative walking state stays untouched.

Importing this module is what pulls in ``imageio`` and ``PIL``; the headless
gate never imports it, so ``validate_beside.py`` has no rendering dependency at
all.  ``test_the_headless_gate_imports_no_rendering_stack`` pins that claim, and
this module is deliberately outside the list it checks.

THE MAIN CAMERA IS MEASURED, NOT EYEBALLED
-------------------------------------------
``tools/probe_framing.py`` replays the REAL recorded trace through candidate
cameras — with THIS module's easing, swing and clamp — and scores, per sampled
frame, whether the duck was on screen, clear of the HUD panels, and unoccluded
in 3D against real solid volumes including the four perimeter walls and every
person as a standing cylinder, plus whether the guardian was in shot, whether
the kiosk was in shot during the switch window, and how large the duck appeared.
The chosen values are recorded in :data:`FRAMING_EVIDENCE` and quoted in the
README.

WHY THE LOOK-AT IS CLAMPED
---------------------------
Her route runs from x = -4.10 to x = +4.45, which is 8.55 m of travel.  An
unclamped look-at follows her the whole way and the camera ends up looking at
the far wall; the clamp keeps the shot inside the promenade so the duck stays
large enough to read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from promenade_layout import FLOOR_HALF
from video_overlay import compose

# MEASURED by replaying the recorded trace through ``tools/probe_framing.py``;
# see the module docstring and the README.  Replaced wholesale by the probe's
# answer, never nudged by eye.
#
#   azimuth 10, elevation -26, distance 2.8, bias 1.00, ease 0.045,
#   look-at z 0.30, look-at bounds DERIVED from eye containment
#     -> duck on screen 0.998, duck unoccluded 0.998,
#        camera eye inside the promenade 1.000,
#        duck clear of the HUD panels 0.984, guardian in shot 1.000,
#        kiosk in shot through the switch window 0.892,
#        duck apparent height 32.7 px,
#        formation separation 180 px mean / 136 px minimum
#
# THE FORMATION-SEPARATION TERM IS THE ONE THIS BEHAVIOR NEEDED.  "Beside" is a
# LATERAL offset, so a camera looking along her direction of travel projects the
# whole formation onto a few pixels and the video stops showing the thing it is
# about.  The probe scores the on-screen duck-to-guardian separation over the
# BESIDE states specifically, and that term is what selected this family.
#
# THE LOOK-AT BOUNDS ARE DERIVED, NOT CHOSEN, AND THAT IS WORTH 0.27 OF PANEL
# CLEARANCE.  ``eye = lookat - forward * distance``, so keeping the camera's own
# eye inside the promenade is a box constraint on the look-at, SHIFTED by the
# camera's own offset and therefore asymmetric.  Every symmetric clamp tried
# earlier had to be tightened to whichever side was worst, which stopped the
# camera following the duck down a 12.4 m promenade and left it drifting behind
# a HUD panel: the best symmetric setting scored 0.712 panel clearance against
# the 0.984 below, at identical containment.
#
# THE CONTAINMENT TERM ITSELF CAME FROM A RENDERED FRAME, NOT FROM THEORY.  A
# free camera orbiting a look-at near the perimeter puts its eye beyond the
# wall, and MuJoCo then draws the near wall as a slab across the shot.  The
# first rendered frame showed exactly that.
#
# ``kiosk in shot 0.892`` is accepted: the kiosk leaves frame towards the end of
# the 6-32 s window, by which time the refusal it caused has been shown and the
# duck is closing on the far side.  Buying it back costs duck size and formation
# legibility, which matter for the whole 86 s rather than for a few seconds.
CAM_AZIMUTH = 10.0
CAM_ELEVATION = -26.0
CAM_DISTANCE = 2.8

# The look-at follows the DUCK.  ``bias`` was swept and 1.00 won: the duck is
# the subject, the guardian is 0.5-0.7 m away and therefore in frame anyway, and
# biasing toward her only pushes the duck outward into a HUD panel.
LOOKAT_BIAS = 1.00
# Applied once per CONTROL TICK, so this is a 0.44 s time constant at 50 Hz.
LOOKAT_EASE = 0.045
LOOKAT_Z = 0.30
# Margin held between the camera eye and the promenade wall, in metres.
EYE_WALL_MARGIN_M = 0.25
# A slow lean, kept small because the framing was scored at a fixed azimuth.
AZIMUTH_SWING_DEG = 3.0
AZIMUTH_SWING_PERIOD_S = 19.0

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
        self.camera.distance = CAM_DISTANCE
        self.camera.elevation = CAM_ELEVATION
        self.camera.azimuth = CAM_AZIMUTH
        self.lookat = np.array([-3.0, -2.0, LOOKAT_Z], dtype=np.float64)
        self.pip_cam = rollout.camera.camera_id

        # Timeline content accumulated as the rollout runs, so a frame drawn at
        # t only ever shows events that had already happened by t.  A viewer
        # must never see a switch marked on the timeline before it happens.
        self.summary: dict = {"state_windows": [], "decisions": [],
                              "switches": []}
        self._open_state: str | None = None
        self._decisions_seen = 0
        self._switches_seen = 0
        # The duck's own recent path, kept at full control rate so the trail is
        # identical at any output frame rate.
        self._trail: list[list[float]] = []
        # frame number -> the tick time and state it was drawn from.  The
        # contact sheet selects by TIME through this manifest rather than by
        # assuming frame == t * fps, which is false whenever the control rate is
        # not an exact multiple of the frame rate.
        self.manifest: list[dict] = []

    def _advance_camera(self, record) -> None:
        """Ease the look-at toward the duck/guardian blend.  ONE CONTROL TICK.

        THIS RUNS ON EVERY CONTROL TICK, NOT ON EVERY WRITTEN FRAME, AND THAT IS
        THE WHOLE POINT.  An ease applied per written frame advances 4 times a
        second in a 4 fps preview and 50 times a second in the final render, so
        the two would fly different camera paths and the preview would stop being
        evidence about the video.  It would also mean the framing probe — which
        samples the trace at its own stride — measured a third path that neither
        render produces.  Driving it from the control tick makes all three
        identical.
        """
        duck = np.array(record["duck_xy"], dtype=np.float64)
        guardian = np.array(
            record["person_xy"][record["guardian"]], dtype=np.float64)
        target = np.array([
            LOOKAT_BIAS * duck[0] + (1.0 - LOOKAT_BIAS) * guardian[0],
            LOOKAT_BIAS * duck[1] + (1.0 - LOOKAT_BIAS) * guardian[1],
            LOOKAT_Z])
        self.lookat += LOOKAT_EASE * (target - self.lookat)

    def _aim_camera(self, record) -> None:
        """Point the free camera at the eased look-at, kept inside the hall.

        The bounds are DERIVED from the camera's own geometry rather than
        declared: ``eye = lookat - forward * distance``, so requiring the eye to
        stay ``EYE_WALL_MARGIN_M`` inside the promenade is an asymmetric box on
        the look-at.  ``tools.probe_framing.eye_safe_lookat_bounds`` is the same
        derivation, which is what makes the probe's scores describe this path.
        """
        azimuth = CAM_AZIMUTH + AZIMUTH_SWING_DEG * math.sin(
            record["t"] / AZIMUTH_SWING_PERIOD_S)
        elevation = math.radians(CAM_ELEVATION)
        heading = math.radians(azimuth)
        forward_x = math.cos(elevation) * math.cos(heading)
        forward_y = math.cos(elevation) * math.sin(heading)
        half_x = FLOOR_HALF[0] - EYE_WALL_MARGIN_M
        half_y = FLOOR_HALF[1] - EYE_WALL_MARGIN_M
        shift_x = forward_x * CAM_DISTANCE
        shift_y = forward_y * CAM_DISTANCE
        self.camera.lookat[0] = float(np.clip(
            self.lookat[0], -half_x + shift_x, half_x + shift_x))
        self.camera.lookat[1] = float(np.clip(
            self.lookat[1], -half_y + shift_y, half_y + shift_y))
        self.camera.lookat[2] = float(self.lookat[2])
        self.camera.azimuth = azimuth

    def _note_events(self, record) -> None:
        """Accumulate timeline content from the machine, up to this tick only.

        Read from the live machine rather than from a pre-computed summary, so
        a frame at t carries exactly the events the duck had decided by t.
        """
        state = record["state"]
        if state != self._open_state:
            self.summary["state_windows"].append(
                {"state": state, "start": record["t"], "end": record["t"]})
            self._open_state = state
        else:
            self.summary["state_windows"][-1]["end"] = record["t"]

        machine = self.rollout.machine
        while self._decisions_seen < len(machine.decisions):
            self.summary["decisions"].append(
                machine.decisions[self._decisions_seen])
            self._decisions_seen += 1
        while self._switches_seen < len(machine.switches):
            self.summary["switches"].append(machine.switches[self._switches_seen])
            self._switches_seen += 1

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

        image = compose(main, pip, record=record,
                        total_seconds=self.rollout.seconds,
                        summary=self.summary,
                        trail=self._trail[::TRAIL_STRIDE],
                        cross_waypoints=[w.tolist() for w
                                         in self.rollout._cross_waypoints])
        imageio.imwrite(self.out / f"f{self.frames:05d}.png", np.asarray(image))
        self.manifest.append({"frame": self.frames, "t": record["t"],
                              "state": record["state"]})
        self.frames += 1

    def write_manifest(self) -> Path:
        path = self.out / "frames.json"
        path.write_text(json.dumps(self.manifest))
        return path
