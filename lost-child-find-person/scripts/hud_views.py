#!/usr/bin/env python3
"""The two spatial HUD views: the plan view and the timeline.

The plan view draws the REAL hall — every obstacle at its real footprint, every
adult at their real position, the duck at its real planar radius — so a viewer
can grade the occlusion and the rejoin geometrically instead of taking the
numbers on trust.  Two things it draws are the whole argument of the behavior:

* **the sightline**, drawn from the duck to the guardian and coloured by whether
  the camera can actually see her.  When the kiosk is between them the line
  crosses the kiosk's real footprint on screen, so "the loss is geometric" is
  something the viewer SEES rather than something the caption claims.
* **the planned rejoin route**, drawn as the waypoint polyline that was planned
  once at reacquisition, so the path walked can be compared against the path
  intended.

The timeline places every state, every refusal and every loss against
wall-clock time, so the two cycles are legible as a shape.
"""

from __future__ import annotations

import math

import numpy as np

from hud_style import (
    ACCENT,
    BAD,
    DIM,
    F09,
    F10,
    F11,
    GOOD,
    GRID,
    INK,
    OUTLINE,
    STATE_COLORS,
    TRAIL,
    WARN,
    fit,
    panel,
    role_color,
    title,
)
from lost_cast import GUARDIAN
from lost_geometry import DUCK_PLANAR_RADIUS
from plaza_layout import FLOOR_HALF, OBSTACLES


class PlanView:
    """Plan view of the concourse: real footprints, sightline, planned route."""

    def __init__(self, box):
        self.x0, self.y0, self.x1, self.y1 = box
        # The whole hall, with a small margin so the walls are not on the edge.
        self.wx0, self.wx1 = -FLOOR_HALF[0] - 0.10, FLOOR_HALF[0] + 0.10
        self.wy0, self.wy1 = -FLOOR_HALF[1] - 0.10, FLOOR_HALF[1] + 0.10
        span_x = self.wx1 - self.wx0
        span_y = self.wy1 - self.wy0
        self.scale = min((self.x1 - self.x0 - 16) / span_x,
                         (self.y1 - self.y0 - 28) / span_y)
        self.ox = self.x0 + 8 + 0.5 * (
            (self.x1 - self.x0 - 16) - span_x * self.scale)
        self.oy = self.y0 + 22 + 0.5 * (
            (self.y1 - self.y0 - 28) - span_y * self.scale)

    def to_px(self, xy):
        """World to pixels.  World +x runs right, world +y runs UP the screen."""
        px = self.ox + (float(xy[0]) - self.wx0) * self.scale
        py = self.oy + (self.wy1 - float(xy[1])) * self.scale
        return px, py

    def _draw_obstacles(self, draw) -> None:
        """Every obstacle at its true footprint; occluders filled, the bench not.

        The fill is the distinction that matters to this behavior: a filled
        shape can hide the guardian, an outlined one can only be walked around.
        """
        for obstacle in OBSTACLES:
            if obstacle.kind == "circle":
                cx, cy = self.to_px(obstacle.center)
                r = obstacle.radius * self.scale
                box = [cx - r, cy - r, cx + r, cy + r]
                if obstacle.occludes:
                    draw.ellipse(box, fill=(52, 60, 78), outline=OUTLINE)
                else:
                    draw.ellipse(box, outline=DIM)
            else:
                hx, hy = obstacle.half
                a = self.to_px((obstacle.center[0] - hx, obstacle.center[1] + hy))
                b = self.to_px((obstacle.center[0] + hx, obstacle.center[1] - hy))
                box = [a[0], a[1], b[0], b[1]]
                if obstacle.occludes:
                    draw.rectangle(box, fill=(52, 60, 78), outline=OUTLINE)
                else:
                    draw.rectangle(box, outline=DIM)
        # Name the principal occluder, which is the one the story turns on.
        kx, ky = self.to_px((0.45, 0.20))
        draw.text((kx - 14, ky - 5), "KIOSK", font=F09, fill=DIM)

    def _draw_trail_and_route(self, draw, record) -> None:
        trail = record["trail"]
        points = trail.get("points") or []
        if len(points) >= 2:
            draw.line([self.to_px(p) for p in points], fill=TRAIL, width=1)
        last_seen = trail.get("last_seen_xy")
        if last_seen is not None:
            lx, ly = self.to_px(last_seen)
            draw.line([(lx - 5, ly - 5), (lx + 5, ly + 5)], fill=TRAIL, width=2)
            draw.line([(lx - 5, ly + 5), (lx + 5, ly - 5)], fill=TRAIL, width=2)

        route = record.get("route")
        if route:
            waypoints = route.get("waypoints") or []
            if len(waypoints) >= 2:
                draw.line([self.to_px(p) for p in waypoints], fill=ACCENT,
                          width=2)
            for point in waypoints[1:]:
                px, py = self.to_px(point)
                draw.ellipse([px - 2.5, py - 2.5, px + 2.5, py + 2.5],
                             outline=ACCENT)

    def _draw_sightline(self, draw, record) -> None:
        """Duck to guardian, coloured by what the CAMERA actually reports.

        Drawn before the bodies so a person standing on the line is drawn over
        it, which is exactly what a blocked sightline looks like from above.
        """
        guardian_xy = record["person_xy"].get(GUARDIAN.name)
        if guardian_xy is None:
            return
        start = self.to_px(record["duck_xy"])
        end = self.to_px(guardian_xy)
        visible = record["guardian_visible"]
        if visible:
            draw.line([start, end], fill=GOOD, width=1)
            return
        # Dashed red: the duck cannot see her along this line.
        segments = 26
        for index in range(segments):
            if index % 2:
                continue
            a = (start[0] + (end[0] - start[0]) * index / segments,
                 start[1] + (end[1] - start[1]) * index / segments)
            b = (start[0] + (end[0] - start[0]) * (index + 1) / segments,
                 start[1] + (end[1] - start[1]) * (index + 1) / segments)
            draw.line([a, b], fill=BAD, width=1)

    def _draw_people(self, draw, record) -> None:
        """Everybody at their real position.

        A body the camera can currently see gets a bright ring; a body it cannot
        is drawn dim.  The guardian additionally gets a filled disc in her own
        teal, so the eye can find her in one glance at any point in the video.
        """
        radius = max(3.0, 0.16 * self.scale)
        subject = record.get("subject")
        refused = {r["name"] for r in record.get("rejections", [])}
        for name, xy in record["person_xy"].items():
            px, py = self.to_px(xy)
            role = record["person_role"][name]
            seen = record["person_visible"][name]
            box = [px - radius, py - radius, px + radius, py + radius]
            if name == GUARDIAN.name:
                draw.ellipse(box, fill=role_color(role),
                             outline=GOOD if seen else DIM)
                draw.text((px + radius + 2, py - 6), "GUARDIAN", font=F09,
                          fill=GOOD if seen else DIM)
            elif name in refused:
                draw.ellipse(box, outline=BAD, width=2)
                draw.text((px + radius + 2, py - 6), name, font=F09, fill=BAD)
            elif name == subject:
                draw.ellipse(box, outline=WARN, width=2)
                draw.text((px + radius + 2, py - 6), name, font=F09, fill=WARN)
            else:
                draw.ellipse(box, outline=role_color(role) if seen else GRID)

    def _draw_duck(self, draw, record) -> None:
        """The duck at its real planar radius, with its heading and its gaze.

        Two rays leave the duck: the body heading, and the HEAD's world-frame
        view direction.  Drawing both is what makes "the body stood still while
        the head swept the hall" visible in the plan view, which is the single
        most counter-intuitive claim this behavior makes.
        """
        radius = max(3.0, DUCK_PLANAR_RADIUS * self.scale)
        dx, dy = self.to_px(record["duck_xy"])
        draw.ellipse([dx - radius, dy - radius, dx + radius, dy + radius],
                     outline=INK, width=2)
        heading = math.radians(record["duck_yaw_deg"])
        draw.line([(dx, dy), (dx + math.cos(heading) * radius * 1.7,
                              dy - math.sin(heading) * radius * 1.7)],
                  fill=INK, width=2)
        gaze = math.radians(record["view_yaw_deg"])
        draw.line([(dx, dy), (dx + math.cos(gaze) * radius * 3.4,
                              dy - math.sin(gaze) * radius * 3.4)],
                  fill=WARN if record["scanning"] else ACCENT, width=1)

    def draw(self, draw, record) -> None:
        panel(draw, (self.x0, self.y0, self.x1, self.y1))
        title(draw, (self.x0, self.y0, self.x1), "PLAN VIEW - real footprints")
        draw.rectangle([self.to_px((-FLOOR_HALF[0], FLOOR_HALF[1])),
                        self.to_px((FLOOR_HALF[0], -FLOOR_HALF[1]))],
                       outline=GRID)
        self._draw_obstacles(draw)
        self._draw_trail_and_route(draw, record)
        self._draw_sightline(draw, record)
        self._draw_people(draw, record)
        self._draw_duck(draw, record)
        legend = ("sightline: green = the camera sees her"
                  if record["guardian_visible"] else
                  "sightline: red dashes = blocked")
        draw.text((self.x0 + 8, self.y1 - 14),
                  fit(draw, legend, F09, self.x1 - self.x0 - 16), font=F09,
                  fill=GOOD if record["guardian_visible"] else BAD)


class Timeline:
    """States, losses and refusals against wall-clock time."""

    def __init__(self, box, total_seconds: float):
        self.x0, self.y0, self.x1, self.y1 = box
        self.total = max(float(total_seconds), 1e-6)

    def to_px(self, t: float) -> float:
        return self.x0 + 10 + (self.x1 - self.x0 - 20) * (t / self.total)

    def draw(self, draw, record, summary) -> None:
        panel(draw, (self.x0, self.y0, self.x1, self.y1))
        # The spine sits low in the panel: refusal labels are stacked ABOVE it
        # on two staggered rows, and the header line above those, so the three
        # refusals inside six seconds neither overprint each other nor the
        # header.
        base = self.y1 - 12

        for window in summary.get("state_windows", []):
            color = STATE_COLORS.get(window["state"], DIM)
            start = self.to_px(window["start"])
            end = max(self.to_px(window["end"]), start + 1.0)
            draw.rectangle([start, base - 9, end, base - 2], fill=color)

        # Losses below the spine, refusals above it: two different kinds of
        # event, so they never share a row and cannot be confused.
        for loss in summary.get("losses", []):
            x = self.to_px(loss)
            draw.line([(x, base + 1), (x, base + 7)], fill=BAD, width=2)
        for index, rejection in enumerate(record.get("rejections", [])):
            x = self.to_px(rejection["t"])
            draw.line([(x, base - 16), (x, base - 11)], fill=BAD, width=2)
            draw.text((x - 9, base - 28 - (index % 2) * 11), rejection["name"],
                      font=F09, fill=BAD)
        for accepted in summary.get("reacquisitions", []):
            x = self.to_px(accepted)
            draw.line([(x, base - 16), (x, base - 11)], fill=GOOD, width=2)

        cursor = self.to_px(record["t"])
        draw.line([(cursor, base - 12), (cursor, base + 9)], fill=INK, width=2)

        draw.text((self.x0 + 10, self.y0 + 4),
                  fit(draw, f"t = {record['t']:5.2f}s / {self.total:.0f}s"
                      f"     cycle {record['cycle_index'] + 1}"
                      "     bars: state"
                      "     below: lost"
                      "     above: refusal / confirmed",
                      F10, self.x1 - self.x0 - 20),
                  font=F10, fill=DIM)
