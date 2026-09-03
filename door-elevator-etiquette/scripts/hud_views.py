#!/usr/bin/env python3
"""The two spatial HUD views: the plan view and the timeline.

The plan view draws the REAL building - every wall and jamb at its true
footprint, every door leaf at its measured open fraction, every person at their
real position, the duck at its measured planar radius - so the etiquette can be
graded geometrically instead of taken on trust.  Four things it draws are the
whole argument of the behavior:

* **the etiquette zones**, shaded: the threshold band the duck must stay out of
  until the exiters clear, and the lift's exit passage it must never block.  A
  viewer can see the duck's own footprint sitting outside them.
* **the door leaves**, drawn where their MEASURED open fraction puts them, so
  "it did not walk through a closed door" is checkable by eye.
* **the guardian and the line to her**, coloured by whether the camera can see
  her right now, so waiting is visibly caused by a person rather than a timer.
* **the duck's route**, as the filleted centreline it actually pursues, with the
  leg it has been released for picked out.

The timeline places every state against wall-clock time, so the whole run reads
as one picture - and because six of the thirteen states are amber, it shows at a
glance that most of this behavior is the robot waiting its turn.
"""

from __future__ import annotations

import numpy as np

from etiquette_cast import GUARDIAN
from etiquette_zones import (
    DOOR_THRESHOLD,
    LIFT_PASSAGE,
    LIFT_THRESHOLD,
)
from hud_style import (
    ACCENT,
    BAD,
    DIM,
    DOOR_INK,
    F09,
    F10,
    GOOD,
    GRID,
    GUARDIAN_IN_HUD,
    INK,
    OUTLINE,
    ROUTE,
    STATE_COLORS,
    TRAIL,
    WARN,
    door_ink,
    fit,
    panel,
    role_color,
    title,
)
from lobby_layout import CABIN_X, CABIN_Y, FLOOR_HALF, STATIC_OBSTACLES

# The duck's MEASURED pose-zero planar half-extent on this scene.  Deliberately
# NOT the conservative bounding-sphere figure the zone gates use: that one
# over-states the robot on purpose, and drawing it would put a body on screen
# wider than the robot actually is.
DUCK_DRAW_RADIUS_M = 0.0827


class PlanView:
    """A scale plan of the building, redrawn every frame."""

    def __init__(self, box):
        self.box = box
        self.x0, self.y0, self.x1, self.y1 = box
        inner_w = (self.x1 - self.x0) - 16
        inner_h = (self.y1 - self.y0) - 30
        self.scale = min(inner_w / (2.0 * FLOOR_HALF[0]),
                         inner_h / (2.0 * FLOOR_HALF[1]))
        self.cx = self.x0 + (self.x1 - self.x0) / 2
        self.cy = self.y0 + 20 + inner_h / 2

    def to_px(self, xy):
        """World metres to panel pixels.  +y world is up on screen."""
        return (self.cx + float(xy[0]) * self.scale,
                self.cy - float(xy[1]) * self.scale)

    def _rect(self, draw, center, half, fill=None, outline=None, width=1):
        a = self.to_px((center[0] - half[0], center[1] + half[1]))
        b = self.to_px((center[0] + half[0], center[1] - half[1]))
        draw.rectangle([a[0], a[1], b[0], b[1]], fill=fill, outline=outline,
                       width=width)

    def _band(self, draw, band, fill):
        center = band.center()
        half = (0.5 * (band.x_range[1] - band.x_range[0]),
                0.5 * (band.y_range[1] - band.y_range[0]))
        self._rect(draw, center, half, fill=fill)

    def draw(self, draw, record, trail, route_points) -> None:
        panel(draw, self.box)
        title(draw, self.box, "PLAN   the building, to scale")

        # the floor
        self._rect(draw, (0.0, 0.0), FLOOR_HALF, fill=(26, 30, 38),
                   outline=OUTLINE)
        # the cabin, so the car reads as a car
        self._rect(draw,
                   (0.5 * (CABIN_X[0] + CABIN_X[1]),
                    0.5 * (CABIN_Y[0] + CABIN_Y[1])),
                   (0.5 * (CABIN_X[1] - CABIN_X[0]),
                    0.5 * (CABIN_Y[1] - CABIN_Y[0])),
                   fill=(34, 40, 50))

        # THE ZONES, drawn before everything else so bodies sit on top of them.
        self._band(draw, DOOR_THRESHOLD, (58, 34, 34))
        self._band(draw, LIFT_THRESHOLD, (58, 34, 34))
        self._band(draw, LIFT_PASSAGE, (58, 46, 28))

        # every static surface at its true footprint
        for obstacle in STATIC_OBSTACLES:
            self._rect(draw, obstacle.center, obstacle.half, fill=(64, 70, 84))

        # THE DOOR LEAVES, at their measured open fraction
        from lobby_doors import APERTURES
        for name, spec in APERTURES.items():
            fraction = record["door_fraction"][name]
            half_w = 0.5 * float(spec["clear_w"])
            travel = float(spec["travel"]) * fraction
            ink = door_ink(fraction)
            for sign in (-1.0, +1.0):
                center = (float(spec["plane_x"]),
                          float(spec["center_y"])
                          + sign * (half_w + travel))
                self._rect(draw, center, (0.03, half_w), fill=ink)

        # the duck's route
        if route_points:
            points = [self.to_px(p) for p in route_points]
            draw.line(points, fill=ROUTE, width=1)

        # its recent trail
        if len(trail) > 1:
            draw.line([self.to_px(p) for p in trail], fill=TRAIL, width=1)

        # everybody
        subject = record["subject"]
        for name, position in record["person_xy"].items():
            px, py = self.to_px(position)
            role = record["person_role"][name]
            ink = role_color(role)
            radius = 4.0 if name == GUARDIAN.name else 3.0
            draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                         fill=ink)
            if name == subject:
                draw.ellipse([px - radius - 3, py - radius - 3,
                              px + radius + 3, py + radius + 3], outline=INK)

        # the duck, at its exact measured half-extent
        duck = self.to_px(record["duck_xy"])
        r = DUCK_DRAW_RADIUS_M * self.scale
        draw.ellipse([duck[0] - r, duck[1] - r, duck[0] + r, duck[1] + r],
                     fill=GOOD, outline=INK)

        # the sightline to whoever it is watching, coloured by whether the real
        # camera can actually see them
        target = record["person_xy"].get(subject)
        if target is not None:
            tx, ty = self.to_px(target)
            draw.line([duck[0], duck[1], tx, ty],
                      fill=GOOD if record["subject_visible"] else BAD, width=1)


class Timeline:
    """Every state against wall-clock time, accumulated as the run proceeds."""

    def __init__(self, box, total_seconds: float):
        self.box = box
        self.total = float(total_seconds)

    def draw(self, draw, record, summary) -> None:
        panel(draw, self.box)
        title(draw, self.box, "TIMELINE   most of this behavior is waiting")
        x0, y0, x1, y1 = self.box
        left, right = x0 + 8, x1 - 8
        top = y0 + 20
        height = 11

        def to_px(t: float) -> float:
            return left + (right - left) * min(max(t / self.total, 0.0), 1.0)

        draw.rectangle([left, top, right, top + height], fill=(30, 35, 45),
                       outline=(52, 58, 72))
        for window in summary.get("state_windows", []):
            a, b = to_px(window["start"]), to_px(window["end"])
            if b - a < 1.0:
                b = a + 1.0
            draw.rectangle([a, top, b, top + height],
                           fill=STATE_COLORS.get(window["state"], DIM))

        # the now-marker
        now = to_px(record["t"])
        draw.line([(now, top - 3), (now, top + height + 3)], fill=INK, width=2)

        # Second row: the CURRENT door fractions as chips.  Not a strip that
        # grew over time - the summary carries no door history, and drawing one
        # would imply the overlay knows something it does not.
        #
        # THE ROW HAS TO FIT INSIDE THE PANEL.  The first preview put it at
        # top + height + 6 in a 48 px panel, which clipped the chips against the
        # bottom edge and cut the labels in half.
        y = top + height + 4
        draw.text((left, y), f"{record['t']:6.2f} s", font=F09, fill=DIM)
        chip = left + 62
        for name, label in (("concourse_door", "door"),
                            ("lift_front", "lift"),
                            ("lift_rear", "rear")):
            fraction = record["door_fraction"][name]
            draw.rectangle([chip, y + 2, chip + 8, y + 9],
                           fill=door_ink(fraction))
            draw.text((chip + 12, y), f"{label} {fraction:.2f}", font=F09,
                      fill=DIM)
            chip += 78
