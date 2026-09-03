#!/usr/bin/env python3
"""The two spatial HUD views: the plan view and the timeline.

The plan view draws the REAL queue path, the REAL barrier lines and everybody's
REAL footprint, so a viewer can grade the join and each advance geometrically
rather than taking the numbers on trust.  The timeline places states and service
completions against wall-clock time.
"""

from __future__ import annotations

import math

import numpy as np

from hud_style import (
    ACCENT,
    BAD,
    DIM,
    F10,
    F11,
    GOOD,
    INK,
    LANE,
    STATE_COLORS,
    WARN,
    _panel,
)
from queue_geometry import BARRIER_HALF_M, DUCK_PLANAR_RADIUS
from queue_path import PATH
from queue_people import ADULT_HALF_EXTENT_M, CLERK


class PlanView:
    """Plan view of the queue: the real path, the barriers, real footprints."""

    def __init__(self, box):
        self.x0, self.y0, self.x1, self.y1 = box
        # World bounds covering the whole lane plus the duck's entry.
        self.wx0, self.wx1 = -1.55, 2.75
        self.wy0, self.wy1 = -1.85, 0.55
        span_x = self.wx1 - self.wx0
        span_y = self.wy1 - self.wy0
        self.scale = min((self.x1 - self.x0 - 16) / span_x,
                         (self.y1 - self.y0 - 26) / span_y)
        self.ox = self.x0 + 8 + 0.5 * (
            (self.x1 - self.x0 - 16) - span_x * self.scale)
        self.oy = self.y0 + 20 + 0.5 * (
            (self.y1 - self.y0 - 26) - span_y * self.scale)

    def to_px(self, xy):
        # World +x runs right, world +y runs UP the screen.
        px = self.ox + (float(xy[0]) - self.wx0) * self.scale
        py = self.oy + (self.wy1 - float(xy[1])) * self.scale
        return px, py

    def draw(self, draw, record):
        _panel(draw, (self.x0, self.y0, self.x1, self.y1))
        draw.text((self.x0 + 8, self.y0 + 4), "PLAN VIEW - queue path",
                  font=F11, fill=DIM)

        # The barrier lines, offset from the path by its own normal.
        polyline = PATH.polyline(0.05)
        for sign in (+1.0, -1.0):
            points = []
            for index, point in enumerate(polyline):
                s = index * PATH.length / (len(polyline) - 1)
                yaw = PATH.away_heading_at(s)
                normal = np.array([-math.sin(yaw), math.cos(yaw)])
                points.append(self.to_px(point + normal * sign * BARRIER_HALF_M))
            draw.line(points, fill=LANE, width=2)
        draw.line([self.to_px(p) for p in polyline], fill=(58, 72, 96), width=1)

        # The counter.
        draw.rectangle([self.to_px((0.30, 0.62)), self.to_px((0.74, -0.62))],
                       outline=(120, 96, 66))

        # People: queue members filled, bystanders hollow.
        radius = max(2.5, ADULT_HALF_EXTENT_M * self.scale)
        order = record["inferred_order"]
        for name, xy in record["person_xy"].items():
            if name == CLERK.name:
                continue
            px, py = self.to_px(xy)
            in_queue = name in order
            if in_queue:
                place = order.index(name) + 1
                color = BAD if name == record["inferred_tail"] else ACCENT
                draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                             fill=color)
                draw.text((px + radius + 2, py - 6), str(place), font=F10,
                          fill=INK)
            else:
                draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                             outline=DIM)

        # The duck's real footprint.
        duck_r = max(2.5, DUCK_PLANAR_RADIUS * self.scale)
        dx, dy = self.to_px(record["duck_xy"])
        draw.ellipse([dx - duck_r, dy - duck_r, dx + duck_r, dy + duck_r],
                     outline=GOOD, width=2)
        heading = math.radians(record["duck_yaw_deg"])
        draw.line([(dx, dy),
                   (dx + math.cos(heading) * duck_r * 1.9,
                    dy - math.sin(heading) * duck_r * 1.9)],
                  fill=GOOD, width=2)

        # The target station.
        if record["target_arc_m"] is not None:
            tx, ty = self.to_px(PATH.point_at(record["target_arc_m"]))
            draw.line([(tx - 5, ty), (tx + 5, ty)], fill=WARN, width=2)
            draw.line([(tx, ty - 5), (tx, ty + 5)], fill=WARN, width=2)


class Timeline:
    """States, services and advances against wall-clock time."""

    def __init__(self, box, total_seconds: float):
        self.x0, self.y0, self.x1, self.y1 = box
        self.total = max(total_seconds, 1e-6)

    def to_px(self, t: float) -> float:
        return self.x0 + 8 + (self.x1 - self.x0 - 16) * (t / self.total)

    def draw(self, draw, record, summary):
        _panel(draw, (self.x0, self.y0, self.x1, self.y1))
        base = self.y1 - 14
        draw.line([(self.to_px(0.0), base), (self.to_px(self.total), base)],
                  fill=(74, 82, 98), width=1)

        for window in summary.get("state_windows", []):
            color = STATE_COLORS.get(window["state"], DIM)
            draw.rectangle(
                [self.to_px(window["start"]), base - 9,
                 max(self.to_px(window["end"]), self.to_px(window["start"]) + 1),
                 base - 3],
                fill=color)
        for event in summary.get("services", []):
            x = self.to_px(event["served_at_s"])
            draw.line([(x, base + 1), (x, base + 7)], fill=WARN, width=2)
        cursor = self.to_px(record["t"])
        draw.line([(cursor, base - 13), (cursor, base + 9)], fill=INK, width=2)
        draw.text((self.x0 + 8, self.y0 + 3),
                  f"t = {record['t']:5.2f}s / {self.total:.0f}s"
                  "     bars: state    ticks: service completed",
                  font=F10, fill=DIM)
