#!/usr/bin/env python3
"""Probe wide-shot camera parameters against the framing the scene requires.

A hairpin queue is a hard thing to film, and the constraint is measurable
rather than aesthetic: at every moment of the rollout the duck must be inside
the frame AND outside the region the HUD panels occupy, while the queue itself
stays visible.  This script projects the duck and each queue station into pixel
coordinates for a candidate camera and reports the violations, so the framing is
chosen from measurement instead of from re-rendering and squinting.

    python tools/probe_camera.py --records /tmp/qp-records.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from policy_runtime import load_scene  # noqa: E402
from queue_path import PATH  # noqa: E402

# Screen regions the HUD occupies, as (x0, y0, x1, y1) in output pixels.  The
# duck must not spend the rollout underneath one of them.
HUD_BOXES = (
    (0, 0, 960, 30),          # title strip
    (12, 40, 330, 500),       # left column
    (640, 40, 948, 566),      # PiP + plan view
    (12, 574, 948, 628),      # timeline
)


def project(model, camera, width, height, point):
    """World point -> pixel coordinates for a mjvCamera-style free camera."""
    azimuth = math.radians(camera["azimuth"])
    elevation = math.radians(camera["elevation"])
    lookat = np.asarray(camera["lookat"], dtype=np.float64)
    distance = camera["distance"]
    forward = np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation)])
    eye = lookat - forward * distance
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    fovy = math.radians(float(model.vis.global_.fovy))
    tan_v = math.tan(0.5 * fovy)
    tan_h = tan_v * width / height

    delta = np.asarray(point, dtype=np.float64) - eye
    depth = float(delta @ forward)
    if depth <= 1e-6:
        return None
    x = float(delta @ right) / (depth * tan_h)
    y = float(delta @ up) / (depth * tan_v)
    return (0.5 * width * (1.0 + x), 0.5 * height * (1.0 - y))


def in_box(pixel, box) -> bool:
    return box[0] <= pixel[0] <= box[2] and box[1] <= pixel[1] <= box[3]


def evaluate(model, records, camera_fn, width, height, margin=26):
    """Fraction of frames in which the duck is framed clear of the HUD."""
    offscreen = 0
    behind_hud = 0
    queue_visible = []
    for record in records:
        camera = camera_fn(record)
        duck = np.array([record["duck_xy"][0], record["duck_xy"][1], 0.16])
        pixel = project(model, camera, width, height, duck)
        if pixel is None or not (
                margin <= pixel[0] <= width - margin
                and margin <= pixel[1] <= height - margin):
            offscreen += 1
            continue
        if any(in_box(pixel, box) for box in HUD_BOXES):
            behind_hud += 1
        seen = 0
        for station in (0.0, 1.10, 2.20, 3.20, 4.30):
            point = PATH.point_at(station)
            station_pixel = project(
                model, camera, width, height,
                np.array([point[0], point[1], 0.30]))
            if station_pixel and 0 <= station_pixel[0] <= width and (
                    0 <= station_pixel[1] <= height):
                seen += 1
        queue_visible.append(seen / 5.0)
    total = max(len(records), 1)
    return {
        "offscreen_frames": offscreen,
        "behind_hud_frames": behind_hud,
        "clear_fraction": round(1.0 - (offscreen + behind_hud) / total, 4),
        "queue_visible_mean": round(float(np.mean(queue_visible)), 4)
        if queue_visible else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot-dir", default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args()

    model = load_scene(args.scene, args.robot_dir)
    records = json.loads(Path(args.records).read_text())
    records = records[::10]

    print(f"{'azim':>6} {'elev':>6} {'dist':>6} {'bias':>5} | "
          f"{'off':>5} {'hud':>5} {'clear':>7} {'queue':>7}")
    print("-" * 60)
    best = None
    for azimuth in (38.0, 46.0, 54.0, 62.0, 118.0, 126.0, 134.0, 142.0):
        for elevation in (-34.0, -40.0, -46.0):
            for distance in (3.5, 4.0, 4.5):
                for bias in (0.45, 0.66):
                    def camera_fn(record, azimuth=azimuth, elevation=elevation,
                                  distance=distance, bias=bias):
                        duck = record["duck_xy"]
                        return {
                            "azimuth": azimuth, "elevation": elevation,
                            "distance": distance,
                            "lookat": [(1 - bias) * 0.30 + bias * duck[0],
                                       (1 - bias) * -0.62 + bias * duck[1],
                                       0.22],
                        }
                    result = evaluate(model, records, camera_fn,
                                      args.width, args.height)
                    score = result["clear_fraction"] + result["queue_visible_mean"]
                    if best is None or score > best[0]:
                        best = (score, azimuth, elevation, distance, bias, result)
                    print(f"{azimuth:6.0f} {elevation:6.0f} {distance:6.2f} "
                          f"{bias:5.2f} | {result['offscreen_frames']:5d} "
                          f"{result['behind_hud_frames']:5d} "
                          f"{result['clear_fraction']:7.3f} "
                          f"{result['queue_visible_mean']:7.3f}")
    print(f"\nBEST azimuth={best[1]} elevation={best[2]} distance={best[3]} "
          f"bias={best[4]}  {best[5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
