#!/usr/bin/env python3
"""Geometry emitters: hall, lane, barriers, markers and people.

Each returns a chunk of MJCF.  The important property is that the LANE PAINT IS
THE PATH: every stripe is ``PATH.point_at(s)`` for an ``s`` the decision layer
also uses, so the picture and the arithmetic cannot drift apart.  A test pins
that by projecting every emitted lane geom back onto the path.
"""

from __future__ import annotations

import math

import numpy as np

from queue_geometry import (
    BARRIER_HALF_M,
    BARRIER_MOUTH_M,
    COUNTER_FRONT_X,
    LANE_PAINT_HALF_M,
)
from queue_path import PATH
from queue_people import ALL_NAMES


def _box(name, pos, size, material, quat=None) -> str:
    extra = f' quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"' if quat else ""
    return (
        f'        <geom name="{name}" type="box" pos="{pos[0]:.4f} {pos[1]:.4f} '
        f'{pos[2]:.4f}" size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"'
        f'{extra} material="{material}" contype="0" conaffinity="0" />\n'
    )


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def hall() -> str:
    """Floor slab, back wall, shelving and the service counter."""
    parts = ["\n        <!-- hall floor, back wall, counter -->\n"]
    parts.append(_box("hall_floor", (0.10, -0.62, 0.0015),
                      (2.85, 1.85, 0.0015), "hallmat"))
    parts.append(_box("back_wall", (2.92, -0.62, 0.62),
                      (0.06, 1.85, 0.62), "wallmat"))
    parts.append(_box("side_wall", (0.10, 1.20, 0.62),
                      (2.85, 0.06, 0.62), "wallmat"))
    for index, y in enumerate((0.42, -0.14)):
        parts.append(_box(f"shelf_{index}", (1.24, y, 0.34 + 0.24 * index),
                          (0.10, 0.26, 0.012), "shelfmat"))
    # The counter: a solid plinth with an overhanging top.  It sits at POSITIVE
    # x, because the queue runs OUT along -x from the service station at the
    # origin, so the person being served stands at the origin facing +x.  (A
    # first draft put it at -x, directly on top of the queue's own first leg:
    # the scenery probe measured -0.200 m of overlap at arc 0.55, i.e. the
    # counter was standing where the second person in line was standing.)
    depth = 0.22
    mid_x = COUNTER_FRONT_X + depth
    parts.append(_box("counter_body", (mid_x, 0.0, 0.145),
                      (depth, 0.62, 0.145), "countermat"))
    parts.append(_box("counter_top", (mid_x, 0.0, 0.298),
                      (depth + 0.022, 0.645, 0.013), "countertopmat"))
    # THE SIGN HANGS OFF-CENTRE, at +y.  Centred, it sat exactly on the
    # sightline from the duck's head camera to the clerk once the duck reached
    # the counter, and the last eight seconds of PiP were a blank green board.
    parts.append(_box("counter_sign", (mid_x + 0.02, 0.40, 0.50),
                      (0.012, 0.22, 0.085), "signmat"))
    return "".join(parts)


def lane() -> str:
    """The painted queue lane, drawn FROM the path the duck reasons about."""
    parts = ["\n        <!-- painted lane: every stripe is PATH.point_at(s) -->\n"]
    step = 0.11
    count = int(PATH.length / step)
    for index in range(count):
        s = (index + 0.5) * step
        point = PATH.point_at(s)
        yaw = PATH.away_heading_at(s)
        quat = _yaw_quat(yaw)
        parts.append(_box(f"lane_{index}", (point[0], point[1], 0.0026),
                          (0.5 * step, LANE_PAINT_HALF_M, 0.0026),
                          "lanemat", quat))
        if index % 3 == 0:
            parts.append(_box(f"lanemid_{index}", (point[0], point[1], 0.0034),
                              (0.030, 0.013, 0.0034), "lanecentremat", quat))
    return "".join(parts)


def barriers() -> str:
    """Posts and ropes down both sides of the lane, following the same path.

    THE BARRIER RUN STOPS SHORT OF THE PATH'S END, leaving an open mouth the
    duck walks in through.  Roping the lane all the way round would enclose it,
    and the duck would have to pass through a rope to join at all - which the
    non-colliding scenery would happily allow and the clearance gate would
    correctly fail.  A real queue has an entrance; this is it.
    """
    parts = ["\n        <!-- rope barriers: non-colliding, gate-measured -->\n"]
    spacing = 0.42
    stop_at = PATH.length - BARRIER_MOUTH_M
    previous: dict[int, np.ndarray] = {}
    index = 0
    while index * spacing <= stop_at:
        s = index * spacing
        point = PATH.point_at(s)
        yaw = PATH.away_heading_at(s)
        normal = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        for sign, tag in ((+1, "l"), (-1, "r")):
            here = point + normal * (sign * BARRIER_HALF_M)
            parts.append(
                f'        <geom name="post_{tag}_{index}" type="cylinder"'
                f' pos="{here[0]:.4f} {here[1]:.4f} 0.1450" size="0.021 0.145"'
                f' material="postmat" contype="0" conaffinity="0" />\n'
            )
            if sign in previous:
                before = previous[sign]
                mid = 0.5 * (before + here)
                span = here - before
                length = float(np.linalg.norm(span))
                rope_yaw = math.atan2(float(span[1]), float(span[0]))
                parts.append(_box(
                    f"rope_{tag}_{index}", (mid[0], mid[1], 0.253),
                    (0.5 * length, 0.010, 0.010), "ropemat",
                    _yaw_quat(rope_yaw)))
            previous[sign] = here
        index += 1
    return "".join(parts)


def markers() -> str:
    """Mocap discs: the duck's target footprint, the true tail, rejected gaps."""
    entries = [("target_marker", "targetmat", 0.130, 0.010),
               ("tail_marker", "tailmat", 0.105, 0.011)]
    entries += [(f"reject_marker_{i}", "rejectmat", 0.098, 0.009)
                for i in range(3)]
    parts = ["\n        <!-- decision markers: mocap discs, non-colliding -->\n"]
    for name, material, radius, z in entries:
        parts.append(
            f'        <body name="{name}" mocap="true" pos="0 0 -3.0">\n'
            f'            <geom name="{name}_disc" type="cylinder"'
            f' size="{radius:.3f} {z:.3f}" material="{material}"'
            f' contype="0" conaffinity="0" />\n'
            f"        </body>\n"
        )
    return "".join(parts)


def person_block(name: str) -> str:
    """One adult: torso, head, face flash, two hinged legs, two hinged arms."""
    lines = [
        f'        <body name="person_{name}" mocap="true" pos="0 0 -3.0">',
        f'            <geom name="{name}_torso" type="capsule"'
        f' fromto="0 0 -0.10 0 0 0.16" size="0.078" material="{name}_shirt"'
        f' contype="0" conaffinity="0" />',
        f'            <geom name="{name}_head" type="sphere" pos="0 0 0.255"'
        f' size="0.064" material="skinmat" contype="0" conaffinity="0" />',
        f'            <geom name="{name}_face" type="sphere" pos="0.060 0 0.252"'
        f' size="0.019" material="{name}_shirt" contype="0" conaffinity="0" />',
    ]
    for side, sy in (("l", 0.036), ("r", -0.036)):
        lines.append(
            f'            <body name="{name}_leg_{side}" pos="0 {sy:+.3f} -0.10">'
            f'<joint name="{name}_hip_{side}" type="hinge" axis="0 1 0"'
            f' range="-40 40" damping="1" />'
            f'<geom type="capsule" fromto="0 0 0 0 0 -0.26" size="0.033"'
            f' material="trousermat" contype="0" conaffinity="0" />'
            f'<geom type="capsule" fromto="0 0 -0.26 0.072 0 -0.272" size="0.035"'
            f' material="shoemat" contype="0" conaffinity="0" /></body>'
        )
    for side, sy in (("l", 0.078), ("r", -0.078)):
        lines.append(
            f'            <body name="{name}_arm_{side}" pos="0 {sy:+.3f} 0.115">'
            f'<joint name="{name}_shoulder_{side}" type="hinge" axis="0 1 0"'
            f' range="-40 40" damping="1" />'
            f'<geom type="capsule" fromto="0 0 0 0 0 -0.205" size="0.026"'
            f' material="skinmat" contype="0" conaffinity="0" /></body>'
        )
    lines.append("        </body>")
    return "\n".join(lines) + "\n"
