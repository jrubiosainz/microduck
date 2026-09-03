#!/usr/bin/env python3
"""The two spatial HUD views: the plan view and the timeline.

The plan view draws the REAL promenade — every obstacle at its true footprint,
every adult at their real position, the duck at its measured planar radius — so
the side decision can be graded geometrically instead of taken on trust.  Four
things it draws are the whole argument of the behavior:

* **both candidate slots**, at the positions the chooser actually evaluates,
  each coloured by that side's live verdict.  When the kiosk enters the left
  slot's swept lane, the left disc turns red ON SCREEN while the duck is still
  standing in it, several seconds before the duck moves.
* **the swept lane**, the slot projected forward along her own predicted motion
  over the 3 s lookahead.  This is why the refusal happens before the duck is
  level with the kiosk, and drawing it is what makes that visible rather than
  surprising.
* **the crossing waypoints**, so the rear-going path is a plan the viewer can
  see before it is walked.
* **her route**, drawn as the filleted centreline, so the three bends are
  legible as a shape and "the formation followed the bend" is checkable.

The timeline places every state and every side decision against wall-clock time,
so the whole 86 s reads as one picture.
"""

from __future__ import annotations

import math

import numpy as np

from beside_actors import ROUTES
from beside_cast import GUARDIAN
from beside_geometry import (
    BESIDE_TARGET_M,
    SIDE_LOOKAHEAD_S,
    slot_point,
)
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
    STATE_COLORS,
    TRAIL,
    WARN,
    fit,
    panel,
    role_color,
    side_ink,
    title,
)
from promenade_layout import FLOOR_HALF, OBSTACLES

# The duck's MEASURED pose-zero planar half-extent on this scene.  Deliberately
# NOT ``beside_geometry.DUCK_PLANAR_RADIUS``: that constant is an inherited
# sizing figure (0.1303 m) which does not describe this robot, and drawing it
# would put a figure on screen that no measurement produced.  The rollout
# measures this number at construction and the metrics report it.
DUCK_DRAW_RADIUS_M = 0.1162

# Her heading arrow, a lighter tint of her own teal.
GUARDIAN_HEADING = (200, 240, 236)


class PlanView:
    """Plan view of the promenade: footprints, both slots, the swept lane."""

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

        The fill is the distinction that matters here: a filled shape is tall
        enough to hide a person from the camera, an outlined one can only take a
        side away from the duck.  ``hedge_s`` is deliberately outlined — it is
        0.45 m, which blocks the robot but hides nobody.
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
        # Name the two bodies the story turns on, and nothing else.
        for label, xy in (("HEDGE", (-4.65, -3.02)), ("KIOSK", (-2.20, -1.25))):
            px, py = self.to_px(xy)
            draw.text((px - 15, py - 5), label, font=F09, fill=DIM)

    def _draw_route(self, draw) -> None:
        """Her filleted centreline, so the three bends are legible as a shape."""
        route = ROUTES[GUARDIAN.name]
        points = [self.to_px(route.pose_at_arc(
            route.length * index / 120.0)[0]) for index in range(121)]
        draw.line(points, fill=(58, 74, 68), width=1)

    def _draw_slots(self, draw, record) -> None:
        """Both candidate slots and the swept lane, coloured by live verdict.

        The lane is the slot projected along her own predicted motion over the
        lookahead the chooser uses, which is what makes a refusal arrive BEFORE
        the duck reaches the obstacle rather than as it hits it.
        """
        guardian_xy = np.asarray(record["person_xy"][GUARDIAN.name],
                                 dtype=np.float64)
        yaw = math.radians(record["guardian_yaw_deg"])
        speed = float(record["guardian_speed_mps"])
        velocity = np.array([math.cos(yaw), math.sin(yaw)]) * speed
        for side, verdict in ((1, record["verdict_left"]),
                              (-1, record["verdict_right"])):
            color = GOOD if verdict["usable"] else BAD
            lane = [self.to_px(slot_point(
                guardian_xy + velocity * (SIDE_LOOKAHEAD_S * i / 6.0), yaw,
                side, lateral=BESIDE_TARGET_M)) for i in range(7)]
            draw.line(lane, fill=color if verdict["usable"] else BAD, width=1)
            px, py = lane[0]
            r = max(3.0, 0.10 * self.scale)
            draw.ellipse([px - r, py - r, px + r, py + r],
                         outline=side_ink(side), width=2)
            if record["side"] == side:
                draw.ellipse([px - r - 3, py - r - 3, px + r + 3, py + r + 3],
                             outline=color)
            if not verdict["usable"]:
                draw.line([(px - 5, py - 5), (px + 5, py + 5)], fill=BAD, width=2)
                draw.line([(px - 5, py + 5), (px + 5, py - 5)], fill=BAD, width=2)

    def _draw_cross(self, draw, waypoints) -> None:
        """The rear crossing waypoints, while a crossing is planned.

        Passed in by the frame writer rather than read from the record: the
        waypoints are presentation, and ``beside_record`` is the frozen dict the
        acceptance gate grades.  Nothing the overlay wants is worth adding a
        field to it.
        """
        if len(waypoints) >= 2:
            draw.line([self.to_px(p) for p in waypoints], fill=ACCENT, width=1)
        for point in waypoints:
            px, py = self.to_px(point)
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], outline=ACCENT)

    def _draw_people(self, draw, record) -> None:
        """Everybody at their real position; the guardian filled in her teal."""
        radius = max(3.0, 0.17 * self.scale)
        for name, xy in record["person_xy"].items():
            px, py = self.to_px(xy)
            role = record["person_role"][name]
            seen = record["person_visible"][name]
            box = [px - radius, py - radius, px + radius, py + radius]
            if name == GUARDIAN.name:
                draw.ellipse(box, fill=role_color(role),
                             outline=GOOD if seen else DIM)
                yaw = math.radians(record["guardian_yaw_deg"])
                draw.line([(px, py), (px + math.cos(yaw) * radius * 2.2,
                                      py - math.sin(yaw) * radius * 2.2)],
                          fill=GUARDIAN_HEADING, width=2)
                draw.text((px + radius + 3, py - 6), "nadia", font=F09,
                          fill=GOOD if seen else DIM)
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

    def _draw_trail(self, draw, trail) -> None:
        if len(trail) >= 2:
            draw.line([self.to_px(p) for p in trail], fill=TRAIL, width=1)

    def draw(self, draw, record, trail=(), cross_waypoints=()) -> None:
        panel(draw, (self.x0, self.y0, self.x1, self.y1))
        title(draw, (self.x0, self.y0, self.x1, self.y1),
              "PLAN VIEW - real footprints, both candidate slots")
        draw.rectangle([self.to_px((-FLOOR_HALF[0], FLOOR_HALF[1])),
                        self.to_px((FLOOR_HALF[0], -FLOOR_HALF[1]))],
                       outline=GRID)
        self._draw_obstacles(draw)
        self._draw_route(draw)
        self._draw_trail(draw, trail)
        self._draw_slots(draw, record)
        self._draw_cross(draw, cross_waypoints)
        self._draw_people(draw, record)
        self._draw_duck(draw, record)
        draw.text((self.x0 + 8, self.y1 - 14),
                  fit(draw, "thin line from each slot = the 3 s lane the "
                      "chooser grades", F09, self.x1 - self.x0 - 16),
                  font=F09, fill=DIM)


class Timeline:
    """States and side decisions against wall-clock time."""

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

        # Decisions above the spine, the completed switch below it: two
        # different kinds of event, so they never share a row.
        for index, decision in enumerate(summary.get("decisions", [])):
            x = self.to_px(decision["t"])
            draw.line([(x, base - 16), (x, base - 11)], fill=WARN, width=2)
            draw.text((x - 10, base - 28 - (index % 2) * 11),
                      decision["side_name"], font=F09, fill=WARN)
        for switch in summary.get("switches", []):
            x0 = self.to_px(switch["blocked_at_s"])
            x1 = self.to_px(switch["joined_at_s"])
            draw.line([(x0, base + 4), (x1, base + 4)], fill=ACCENT, width=2)
            draw.text((x0, base + 6), "switch left \u2192 right", font=F09,
                      fill=ACCENT)

        cursor = self.to_px(record["t"])
        draw.line([(cursor, base - 12), (cursor, base + 3)], fill=INK, width=2)

        draw.text((self.x0 + 10, self.y0 + 4),
                  fit(draw, f"t = {record['t']:5.2f}s / {self.total:.0f}s"
                      "     bars: state"
                      "     above: side decision"
                      "     below: the physical switch",
                      F10, self.x1 - self.x0 - 20), font=F10, fill=DIM)
