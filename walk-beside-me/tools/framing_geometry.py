#!/usr/bin/env python3
"""Camera projection and scene-occlusion primitives for the framing probe.

Split out of ``probe_framing`` so that module stays about SCORING and this stays
about GEOMETRY: a pinhole camera, the scene's solid volumes, and the two
predicates a score is built from.

Everything here is pure geometry against the scene's own declared layout.  It
imports nothing from the rollout and never steps physics, so a candidate camera
can be graded from a recorded trace alone.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from beside_cast import BY_NAME  # noqa: E402
from promenade_layout import FLOOR_HALF, OBSTACLES, WALL_HALF_Z  # noqa: E402

FOVY_DEG = 45.0
DUCK_TOP_M = 0.25
PERSON_RADIUS_M = 0.24

# The HUD rectangles the duck must stay out of, at 960x640.  These MIRROR the
# boxes in ``video_overlay.compose``; a layout change is re-scored here rather
# than re-guessed.  The free band the duck has to live in is therefore narrow —
# roughly x 318..646 over the full height — and that is deliberate: the probe
# should be told the truth about the layout, not a flattering version of it.
PANELS = (
    (0, 0, 960, 30),        # title strip
    (12, 38, 316, 158),     # status
    (12, 164, 316, 330),    # side risk
    (12, 336, 316, 476),    # formation error
    (12, 482, 316, 522),    # cast legend
    (648, 38, 948, 254),    # head-camera PiP
    (640, 262, 948, 548),   # plan view
    (324, 554, 948, 628),   # timeline
)


def _volumes(record) -> list[tuple]:
    """Every solid volume in the scene at this instant, for occlusion tests."""
    volumes: list[tuple] = []
    for obstacle in OBSTACLES:
        if obstacle.kind == "circle":
            volumes.append(("cyl", np.array(obstacle.center), obstacle.radius,
                            obstacle.height_m))
        else:
            volumes.append(("box", np.array(obstacle.center),
                            np.array(obstacle.half), obstacle.height_m))
    half_x, half_y = FLOOR_HALF
    wall_h = 2.0 * WALL_HALF_Z
    for center, half in (((0.0, half_y), (half_x, 0.03)),
                         ((0.0, -half_y), (half_x, 0.03)),
                         ((half_x, 0.0), (0.03, half_y)),
                         ((-half_x, 0.0), (0.03, half_y))):
        volumes.append(("box", np.array(center), np.array(half), wall_h))
    for name, xy in record["person_xy"].items():
        volumes.append(("cyl", np.array(xy, dtype=np.float64), PERSON_RADIUS_M,
                        BY_NAME[name].height_m))
    return volumes


def _blocked(eye, target, volumes, ignore=None, samples: int = 44) -> bool:
    """Does the segment eye->target pass through any solid volume?

    ``ignore`` is the planar centre of the body being LOOKED AT.  Without it a
    ray aimed at somebody's own chest is blocked by their own cylinder and every
    candidate scores zero visibility for them, which is exactly how the first
    run of this probe reported ``guardian 0.000`` for all 384 candidates.

    The ignore test is applied ONCE, here, to build the volume list — not inside
    the sample loop.  An ``np.allclose`` per volume per sample is roughly 800
    million calls over a full sweep and made the probe take longer than the
    render it was choosing parameters for.
    """
    if ignore is not None:
        ignore = np.asarray(ignore, dtype=np.float64)
        volumes = [v for v in volumes
                   if not (v[0] == "cyl"
                           and abs(float(v[1][0]) - float(ignore[0])) < 1e-9
                           and abs(float(v[1][1]) - float(ignore[1])) < 1e-9)]
    eye = np.asarray(eye, dtype=np.float64)
    delta = np.asarray(target, dtype=np.float64) - eye
    for index in range(1, samples):
        point = eye + delta * (index / samples)
        if point[2] < 0.0:
            return True
        for volume in volumes:
            if volume[0] == "cyl":
                _, center, radius, height = volume
                if point[2] <= height and math.hypot(
                        point[0] - center[0], point[1] - center[1]) <= radius:
                    return True
            else:
                _, center, half, height = volume
                if point[2] <= height and abs(point[0] - center[0]) <= half[0] \
                        and abs(point[1] - center[1]) <= half[1]:
                    return True
    return False


class Camera:
    """A MuJoCo free camera: azimuth, elevation, distance about a look-at."""

    def __init__(self, azimuth, elevation, distance, width=960, height=640):
        self.azimuth = math.radians(azimuth)
        self.elevation = math.radians(elevation)
        self.distance = distance
        self.width, self.height = width, height
        self.focal = (height / 2.0) / math.tan(math.radians(FOVY_DEG) / 2.0)

    def basis(self, azimuth_deg_offset: float = 0.0):
        azimuth = self.azimuth + math.radians(azimuth_deg_offset)
        forward = np.array([math.cos(self.elevation) * math.cos(azimuth),
                            math.cos(self.elevation) * math.sin(azimuth),
                            math.sin(self.elevation)])
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return forward, right, up

    def eye(self, lookat, azimuth_deg_offset: float = 0.0):
        forward, _, _ = self.basis(azimuth_deg_offset)
        return np.asarray(lookat, dtype=np.float64) - forward * self.distance

    def project(self, point, lookat, azimuth_deg_offset: float = 0.0):
        """World point to pixels, or ``None`` when behind the camera."""
        forward, right, up = self.basis(azimuth_deg_offset)
        delta = np.asarray(point, dtype=np.float64) - self.eye(
            lookat, azimuth_deg_offset)
        depth = float(delta @ forward)
        if depth <= 1e-6:
            return None
        return (self.width / 2.0 + self.focal * float(delta @ right) / depth,
                self.height / 2.0 - self.focal * float(delta @ up) / depth)


def eye_safe_lookat_bounds(camera, azimuth_deg_offset=0.0, margin=0.25):
    """The look-at box that keeps the camera's OWN EYE inside the promenade.

    ``eye = lookat - forward * distance``, so requiring the eye to stay inside
    the hall is a box constraint on the look-at — SHIFTED by the camera's own
    offset, and therefore asymmetric.  Deriving it beats guessing a symmetric
    clamp: a symmetric box has to be tightened to whichever side is worst, which
    stops the camera following the duck to the far end of a promenade whose
    long axis is 12.4 m and leaves the duck drifting behind a HUD panel.
    """
    forward, _, _ = camera.basis(azimuth_deg_offset)
    half_x, half_y = FLOOR_HALF[0] - margin, FLOOR_HALF[1] - margin
    shift_x = float(forward[0]) * camera.distance
    shift_y = float(forward[1]) * camera.distance
    return ((-half_x + shift_x, half_x + shift_x),
            (-half_y + shift_y, half_y + shift_y))


def _on_screen(px, margin: int = 8) -> bool:
    return (px is not None and margin <= px[0] <= 960 - margin
            and margin <= px[1] <= 640 - margin)


def _clear_of_panels(px) -> bool:
    if px is None:
        return False
    return not any(x0 <= px[0] <= x1 and y0 <= px[1] <= y1
                   for x0, y0, x1, y1 in PANELS)

