#!/usr/bin/env python3
"""The wide plaza camera: one eased rig, shared by the probe and the renderer.

WHY THIS IS A MODULE AND NOT THREE CONSTANTS IN THE WRITER
------------------------------------------------------------
The framing is MEASURED rather than chosen by eye: ``tools/probe_pps_framing``
replays the real recorded trace through this exact easing and scores every
candidate on whether the duck, the protected person and the active intruder
stay inside the band of screen the HUD leaves clear.  A probe that scored a
different camera path from the one the renderer flies would be measuring
nothing, so both import this class.

THE HUD LEAVES A NARROW CLEAR BAND, AND THAT IS THE BINDING CONSTRAINT
-----------------------------------------------------------------------
At 960x640 the left column occupies x < 278 and the right column x > 650, so
only about 370 px of width is unobstructed.  A camera that keeps the duck
centred but pushes the intruder behind an opaque panel has hidden the thing the
behavior is about.  The probe therefore scores CLEAR-BAND containment, not mere
on-screen containment.

THE LOOK-AT FOLLOWS THE ENCOUNTER, NOT THE ROBOT
--------------------------------------------------
The subject of this behavior is never the duck alone: it is the duck, the person
it protects, and whoever is currently walking at her.  A frame with only the
robot in it cannot show what it was responding to.  So the look-at is the
centroid of those subjects, weighted toward the pair the encounter is actually
about, and the distance opens with their spread so nobody leaves the frame at
the widest moment of a squeeze.

EASING RUNS PER CONTROL TICK, NOT PER WRITTEN FRAME
-----------------------------------------------------
An ease applied per written frame advances 4 times a second in a 4 fps preview
and 50 times a second in the final render, so the two would fly different camera
paths and the preview would stop being evidence about the video.
:meth:`PlazaCamera.advance` is called on every control tick regardless of the
output frame rate.
"""

from __future__ import annotations

import math

import numpy as np

# MEASURED by tools/probe_pps_framing.py against the real 9500-tick trace.
# The probe replays this module's own easing and scores each candidate on five
# things per tick: the duck, the protected person and the active intruder each
# inside the band of screen the HUD leaves clear, the duck's projected size,
# and the smallest on-screen gap between any two subjects.
#
#   azimuth 210, elevation -36, distance 4.20-8.20, spread_for_far 3.60
#     -> duck in the clear band 1.000, ward 1.000, active intruder 1.000,
#        duck 26.7 px across at the median, and the three subjects a median
#        64.6 px apart - the widest separation of any candidate that keeps all
#        three inside the band for the entire session.
#
# THE TERM THAT ACTUALLY DECIDES THIS IS THE SEPARATION ONE, AND FINDING THAT
# OUT COST TWO WRONG SWEEPS.  Containment SATURATES: across the whole upper
# half of the azimuth circle the duck and the ward are inside the clear band on
# 1.000 of ticks, so the first two sweeps were choosing between candidates that
# differed only in the third decimal of a number that was already perfect, and
# both of them returned an answer sitting on the edge of their own grid - the
# tell that the grid, not the scene, was deciding.
#
# What containment cannot see is SUPERPOSITION.  A camera placed near the
# Aina-to-intruder axis keeps all three subjects comfortably on screen and
# stacks them on top of each other, which destroys exactly the geometry this
# behavior claims: the robot standing BETWEEN two people.  Adding the median
# pairwise on-screen gap as a scored term made the choice real, moved the
# optimum to a value interior to the grid, and is what 64.6 px buys.
#
# The elevation is the other half of that trade and it goes the opposite way to
# intuition: -36 deg is STEEPER than the -28 that maximises the duck's pixel
# size, because looking further down the z axis spreads bodies apart in the
# image plane instead of projecting them onto one line.  4 px of duck is worth
# 6 px of separation here.
CAM_AZIMUTH = 210.0
CAM_ELEVATION = -36.0
CAM_DISTANCE_NEAR = 4.20
CAM_DISTANCE_FAR = 8.20
# The subject spread at which the far distance is reached.
SPREAD_FOR_FAR_M = 3.60
# Weight on the duck when the look-at centroid is formed.  The rest is split
# between the protected person and the active intruder.
DUCK_BIAS = 0.42
WARD_BIAS = 0.36
# Applied once per CONTROL TICK, so this is a 0.44 s time constant at 50 Hz.
LOOKAT_EASE = 0.045
LOOKAT_Z = 0.46
# A slow drift so the plaza reads as three-dimensional over 190 s without ever
# swinging far enough to put a subject behind a HUD column.
AZIMUTH_SWING_DEG = 5.0
AZIMUTH_SWING_PERIOD_S = 62.0


class PlazaCamera:
    """The eased free-camera rig.  Advance every control tick; aim to render."""

    def __init__(self, start_xy=(0.62, -2.42)):
        self.lookat = np.array([float(start_xy[0]), float(start_xy[1]),
                                LOOKAT_Z], dtype=np.float64)
        self.distance = float(CAM_DISTANCE_NEAR)

    # -- the subjects ------------------------------------------------------
    @staticmethod
    def subjects(duck_xy, ward_xy, threat_xy=None):
        """The points the camera is responsible for keeping in shot."""
        points = [np.asarray(duck_xy, dtype=np.float64)[:2],
                  np.asarray(ward_xy, dtype=np.float64)[:2]]
        if threat_xy is not None:
            points.append(np.asarray(threat_xy, dtype=np.float64)[:2])
        return points

    @staticmethod
    def _centroid(duck_xy, ward_xy, threat_xy):
        duck = np.asarray(duck_xy, dtype=np.float64)[:2]
        ward = np.asarray(ward_xy, dtype=np.float64)[:2]
        if threat_xy is None:
            # No selected intruder: the pair is the subject, and the duck keeps
            # slightly more weight because it is the thing being graded.
            total = DUCK_BIAS + WARD_BIAS
            return (DUCK_BIAS * duck + WARD_BIAS * ward) / total
        threat = np.asarray(threat_xy, dtype=np.float64)[:2]
        rest = 1.0 - DUCK_BIAS - WARD_BIAS
        return DUCK_BIAS * duck + WARD_BIAS * ward + rest * threat

    @staticmethod
    def _spread(points) -> float:
        """The widest separation between any two subjects, in metres."""
        widest = 0.0
        for index, first in enumerate(points):
            for second in points[index + 1:]:
                widest = max(widest, float(np.linalg.norm(first - second)))
        return widest

    # -- one control tick --------------------------------------------------
    def advance(self, duck_xy, ward_xy, threat_xy=None) -> None:
        """Ease the look-at and the distance by ONE control tick."""
        target = self._centroid(duck_xy, ward_xy, threat_xy)
        goal = np.array([target[0], target[1], LOOKAT_Z])
        self.lookat += LOOKAT_EASE * (goal - self.lookat)

        spread = self._spread(self.subjects(duck_xy, ward_xy, threat_xy))
        wanted = CAM_DISTANCE_NEAR + (CAM_DISTANCE_FAR - CAM_DISTANCE_NEAR) * \
            min(max(spread / SPREAD_FOR_FAR_M, 0.0), 1.0)
        self.distance += LOOKAT_EASE * (wanted - self.distance)

    def azimuth_at(self, t: float) -> float:
        return CAM_AZIMUTH + AZIMUTH_SWING_DEG * math.sin(
            t / AZIMUTH_SWING_PERIOD_S)

    # -- projection, for the probe and for any framing assertion -----------
    def eye(self, t: float) -> np.ndarray:
        """Where the camera actually sits, in world metres.

        DERIVED from the same azimuth/elevation/distance MuJoCo uses, so the
        probe's geometry is the renderer's geometry rather than a model of it.
        """
        azimuth = math.radians(self.azimuth_at(t))
        elevation = math.radians(CAM_ELEVATION)
        forward = np.array([
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation)])
        return self.lookat - forward * self.distance

    def project(self, point_xyz, t: float, width: int, height: int,
                fovy_deg: float) -> tuple[float, float, bool]:
        """Project a world point to pixels.  Returns ``(x, y, in_front)``.

        MuJoCo's free camera uses a vertical FOV and a right-handed frame with
        ``-z`` forward, which is what this reproduces: the probe has to agree
        with the renderer to about a pixel or its scores are fiction.
        """
        azimuth = math.radians(self.azimuth_at(t))
        elevation = math.radians(CAM_ELEVATION)
        forward = np.array([
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation)])
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= max(float(np.linalg.norm(right)), 1e-9)
        up = np.cross(right, forward)

        delta = np.asarray(point_xyz, dtype=np.float64) - self.eye(t)
        depth = float(delta @ forward)
        if depth <= 1e-6:
            return 0.0, 0.0, False
        tan_v = math.tan(math.radians(fovy_deg) * 0.5)
        tan_h = tan_v * (width / height)
        ndc_x = float(delta @ right) / (depth * tan_h)
        ndc_y = float(delta @ up) / (depth * tan_v)
        return ((0.5 * (1.0 + ndc_x)) * width,
                (0.5 * (1.0 - ndc_y)) * height, True)
