#!/usr/bin/env python3
"""The two spatial HUD views: the plan view and the timeline.

The plan view draws the REAL concourse — every obstacle at its true footprint,
every adult at their real position, the duck at its measured planar radius — so
the route can be graded geometrically instead of taken on trust.  Four things it
draws are the whole argument of the behavior:

* **all three destinations**, each in its own colour, with the requested one
  ringed.  A viewer can see that the duck had a choice and which one it took.
* **the searched route**, as the filleted centreline the duck actually pursues,
  so the bends are legible as a shape and "the route avoided the partitions" is
  checkable by eye against the drawn footprints.
* **the follower and the line to her**, coloured by whether the camera can see
  her right now, so a wait is visibly caused by the person rather than by a
  timer.
* **the waiting spot**, marked where the duck stopped, so "it waited somewhere
  safe" is a position on a map rather than a number in a table.

The timeline places every state and every episode against wall-clock time, so
the whole run reads as one picture.
"""

from __future__ import annotations

import math

import numpy as np

from guide_cast import FOLLOWER
from guide_layout import DESTINATIONS, FLOOR_HALF, OBSTACLES
from guide_states import LAG_DISTANCE_M
from hud_style import (
    ACCENT,
    BAD,
    DIM,
    F09,
    F10,
    GOOD,
    GRID,
    INK,
    OUTLINE,
    ROUTE,
    STATE_COLORS,
    TRAIL,
    WARN,
    destination_ink,
    fit,
    panel,
    role_color,
    title,
)

# The duck's MEASURED pose-zero planar half-extent on this scene.  Deliberately
# NOT the planner's inflated ``DUCK_PLANNING_RADIUS_M``: that constant is a
# bounding-sphere figure used to make routes conservative, and drawing it would
# put a body on screen that is wider than the robot.
DUCK_DRAW_RADIUS_M = 0.1162


class PlanView:
    """Plan view of the concourse: footprints, the route, the three goals."""

    def __init__(self, box):
        self.x0, self.y0, self.x1, self.y1 = box
        self.wx0, self.wx1 = -FLOOR_HALF[0] - 0.10, FLOOR_HALF[0] + 0.10
        self.wy0, self.wy1 = -FLOOR_HALF[1] - 0.10, FLOOR_HALF[1] + 0.10
        span_x = self.wx1 - self.wx0
        span_y = self.wy1 - self.wy0
        self.scale = min((self.x1 - self.x0 - 16) / span_x,
                         (self.y1 - self.y0 - 30) / span_y)
        self.ox = self.x0 + 8 + 0.5 * (
            (self.x1 - self.x0 - 16) - span_x * self.scale)
        self.oy = self.y0 + 24 + 0.5 * (
            (self.y1 - self.y0 - 30) - span_y * self.scale)

    def to_px(self, xy):
        """World to pixels.  World +x runs right, world +y runs UP the screen."""
        px = self.ox + (float(xy[0]) - self.wx0) * self.scale
        py = self.oy + (self.wy1 - float(xy[1])) * self.scale
        return px, py

    def _draw_obstacles(self, draw) -> None:
        """True footprints; full-height occluders filled, low furniture not.

        The fill is the distinction that matters: a filled shape is tall enough
        to hide a person from the camera, an outlined one can only constrain the
        robot.
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

    def _draw_destinations(self, draw, record) -> None:
        """All three, in their own colours, with the requested one ringed."""
        requested = record["requested_destination"]
        for destination in DESTINATIONS:
            px, py = self.to_px(destination.xy)
            ink = destination_ink(destination.key)
            r = max(4.0, 0.16 * self.scale)
            chosen = destination.key == requested
            draw.rectangle([px - r, py - r, px + r, py + r],
                           fill=ink if chosen else None, outline=ink)
            if chosen:
                draw.ellipse([px - r - 4, py - r - 4, px + r + 4, py + r + 4],
                             outline=GOOD)
            draw.text((px - 12, py + r + 2), destination.key, font=F09,
                      fill=ink if chosen else DIM)

    def _draw_route(self, draw, route_points, record) -> None:
        """The searched route, and how far along it the duck is."""
        if len(route_points) >= 2:
            draw.line([self.to_px(p) for p in route_points], fill=ROUTE,
                      width=1)
        for point in record.get("waypoints_xy", []) or []:
            px, py = self.to_px(point)
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], outline=ROUTE)

    def _draw_people(self, draw, record) -> None:
        """Everybody at their real position; the follower filled in her pink."""
        radius = max(3.0, 0.17 * self.scale)
        duck = np.array(record["duck_xy"], dtype=np.float64)
        for name, xy in record["person_xy"].items():
            px, py = self.to_px(xy)
            role = record["person_role"][name]
            seen = record["person_visible"][name]
            box = [px - radius, py - radius, px + radius, py + radius]
            if name == FOLLOWER.name:
                # The line to her IS the relationship the behavior is about, so
                # it is drawn and coloured by what the camera can see.
                dx, dy = self.to_px(duck)
                lagging = record["follower_range_m"] > LAG_DISTANCE_M
                draw.line([(dx, dy), (px, py)],
                          fill=BAD if lagging else (GOOD if seen else WARN),
                          width=1)
                draw.ellipse(box, fill=role_color(role),
                             outline=GOOD if seen else BAD)
                draw.text((px + radius + 3, py - 6), name, font=F09,
                          fill=GOOD if seen else BAD)
            else:
                draw.ellipse(box, outline=role_color(role) if seen else GRID)
                draw.text((px + radius + 3, py - 5), name, font=F09,
                          fill=DIM if seen else GRID)

    def _draw_duck(self, draw, record) -> None:
        """The duck at its MEASURED planar radius, with heading and gaze."""
        radius = max(3.0, DUCK_DRAW_RADIUS_M * self.scale)
        dx, dy = self.to_px(record["duck_xy"])
        draw.ellipse([dx - radius, dy - radius, dx + radius, dy + radius],
                     outline=INK, width=2)
        heading = math.radians(record["duck_yaw_deg"])
        draw.line([(dx, dy), (dx + math.cos(heading) * radius * 1.8,
                              dy - math.sin(heading) * radius * 1.8)],
                  fill=INK, width=2)
        gaze = math.radians(record["view_yaw_deg"])
        draw.line([(dx, dy), (dx + math.cos(gaze) * radius * 3.6,
                              dy - math.sin(gaze) * radius * 3.6)],
                  fill=ACCENT, width=1)

    def _draw_wait(self, draw, record) -> None:
        if record["waiting_spot"] is None:
            return
        px, py = self.to_px(record["waiting_spot"])
        r = max(5.0, 0.14 * self.scale)
        draw.ellipse([px - r, py - r, px + r, py + r], outline=WARN, width=2)
        draw.text((px - 12, py - r - 12), "WAIT", font=F09, fill=WARN)

    def _draw_trail(self, draw, trail) -> None:
        if len(trail) >= 2:
            draw.line([self.to_px(p) for p in trail], fill=TRAIL, width=1)

    def draw(self, draw, record, trail=(), route_points=()) -> None:
        panel(draw, (self.x0, self.y0, self.x1, self.y1))
        title(draw, (self.x0, self.y0, self.x1, self.y1),
              "PLAN VIEW - real footprints, searched route")
        draw.rectangle([self.to_px((-FLOOR_HALF[0], FLOOR_HALF[1])),
                        self.to_px((FLOOR_HALF[0], -FLOOR_HALF[1]))],
                       outline=GRID)
        self._draw_obstacles(draw)
        self._draw_destinations(draw, record)
        self._draw_route(draw, route_points, record)
        self._draw_trail(draw, trail)
        self._draw_wait(draw, record)
        self._draw_people(draw, record)
        self._draw_duck(draw, record)
        draw.text((self.x0 + 8, self.y1 - 14),
                  fit(draw, "blue line = the route the planner searched; "
                      "line to her = what the camera sees", F09,
                      self.x1 - self.x0 - 16),
                  font=F09, fill=DIM)


class Timeline:
    """States and lag episodes against wall-clock time."""

    def __init__(self, box, total_seconds: float):
        self.x0, self.y0, self.x1, self.y1 = box
        self.total = max(float(total_seconds), 1e-6)

    def to_px(self, t: float) -> float:
        return self.x0 + 10 + (self.x1 - self.x0 - 20) * (t / self.total)

    def draw(self, draw, record, summary) -> None:
        panel(draw, (self.x0, self.y0, self.x1, self.y1))
        base = self.y1 - 13

        for window in summary.get("state_windows", []):
            color = STATE_COLORS.get(window["state"], DIM)
            start = self.to_px(window["start"])
            end = max(self.to_px(window["end"]), start + 1.0)
            draw.rectangle([start, base - 9, end, base - 2], fill=color)

        # Detections above the spine, the waits below it: two different kinds of
        # event, so they never share a row.  Nothing is drawn before it happens.
        for index, episode in enumerate(summary.get("episodes", [])):
            x = self.to_px(episode["detected_at_s"])
            draw.line([(x, base - 16), (x, base - 11)], fill=BAD, width=2)
            draw.text((x - 8, base - 28 - (index % 2) * 11),
                      episode["cause"], font=F09, fill=BAD)
            if "resumed_at_s" in episode:
                x1 = self.to_px(episode["resumed_at_s"])
                draw.line([(x, base + 4), (x1, base + 4)], fill=WARN, width=2)
                draw.text((x, base + 6),
                          f"waited {episode.get('wait_duration_s', 0):.0f}s",
                          font=F09, fill=WARN)

        cursor = self.to_px(record["t"])
        draw.line([(cursor, base - 12), (cursor, base + 3)], fill=INK, width=2)

        draw.text((self.x0 + 10, self.y0 + 4),
                  fit(draw, f"t = {record['t']:5.2f}s / {self.total:.0f}s"
                      "     bars: state"
                      "     above: lag detected"
                      "     below: the duck waiting",
                      F10, self.x1 - self.x0 - 20), font=F10, fill=DIM)
