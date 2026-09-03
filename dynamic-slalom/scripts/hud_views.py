#!/usr/bin/env python3
"""The two spatial HUD views: the plan view and the timeline.

The plan view draws the REAL course - every crate, pallet and cone at its true
footprint, every moving body at its real position, the duck at its measured
planar radius - so the slalom can be graded geometrically instead of taken on
trust.  Four things it draws are the whole argument of the behavior:

* **the predicted occupancy**, as the same discs the planner scored: where the
  threatening body is expected to be over the next several seconds.  A viewer
  can see the thing the robot was reasoning about, not only its conclusion.
* **both candidate corridors**, in their own colours, so the rejected one is
  visible beside the chosen one for as long as the decision lasts.
* **the lane and the goal band**, so "it went somewhere" is a picture rather
  than a number.
* **the duck's own trail**, which is what makes the turning path - the only way
  this robot can move sideways - visible as a curve.

The timeline places every state against wall-clock time, so the whole run reads
as one picture: five encounter cycles, each a burst of THREAT / CHOOSE / PASS /
REPLAN, with the amber WAIT bands standing out as the moments the duck refused
both options.
"""

from __future__ import annotations

import numpy as np

from hud_style import (
    BAD,
    DIM,
    F09,
    GOAL_INK,
    GOOD,
    INK,
    LEFT_INK,
    OUTLINE,
    PRED_INK,
    RIGHT_INK,
    ROUTE,
    STATE_COLORS,
    TRAIL,
    kind_color,
    panel,
    side_ink,
    title,
)
from slalom_course import (
    FLOOR_HALF,
    GOAL_BAND_HALF,
    GOAL_XY,
    LANE_HALF_W,
    STATIC_OBSTACLES,
)

# The duck's MEASURED pose-zero planar half-extent on this scene.  Deliberately
# NOT the conservative bounding-sphere figure the clearance gates use: that one
# over-states the robot on purpose, and drawing it would put a body on screen
# wider than the robot actually is.
DUCK_DRAW_RADIUS_M = 0.0827


class PlanView:
    """A scale plan of the depot floor, redrawn every frame."""

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

    def _disc(self, draw, xy, radius_px, **kw):
        px, py = self.to_px(xy)
        draw.ellipse([px - radius_px, py - radius_px,
                      px + radius_px, py + radius_px], **kw)

    def draw(self, draw, record, trail) -> None:
        panel(draw, self.box)
        title(draw, self.box, "PLAN   the course, to scale")

        # the floor
        self._rect(draw, (0.0, 0.0), FLOOR_HALF, fill=(26, 30, 38),
                   outline=OUTLINE)
        # the nominal lane band the duck is trying to walk down
        self._rect(draw, (0.0, 0.0), (FLOOR_HALF[0] - 0.4, LANE_HALF_W),
                   fill=(30, 38, 48))

        # the arrival band
        self._rect(draw, GOAL_XY, GOAL_BAND_HALF, fill=(28, 66, 46),
                   outline=GOAL_INK)

        # every static body at its true footprint
        for obstacle in STATIC_OBSTACLES:
            if obstacle.kind == "cylinder":
                self._disc(draw, obstacle.center,
                           max(obstacle.radius * self.scale, 1.5),
                           fill=(96, 74, 48))
            else:
                self._rect(draw, obstacle.center, obstacle.half,
                           fill=(64, 70, 84))

        # THE PREDICTED OCCUPANCY: the same samples the planner scored.
        threat = record["threat"]
        for entry in record.get("predicted_occupancy", []):
            position = entry["bodies"].get(threat)
            if position is None:
                continue
            self._disc(draw, position, 2.0, fill=PRED_INK)

        # BOTH candidate corridors, as the lines they represent.
        decision = record.get("decision") or {}
        for corridor in decision.get("corridors", []):
            origin = corridor.get("origin")
            if origin is None or not corridor.get("safe"):
                continue
            offset = corridor["offset_m"]
            start = np.asarray(origin, dtype=float)
            direction = np.asarray(GOAL_XY, dtype=float) - start
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                continue
            direction /= norm
            normal = np.array([-direction[1], direction[0]])
            a = start + normal * offset
            b = a + direction * min(2.0, norm)
            draw.line([self.to_px(a), self.to_px(b)],
                      fill=side_ink(corridor["side"]), width=1)

        # its recent trail: the turning path made visible
        if len(trail) > 1:
            draw.line([self.to_px(p) for p in trail], fill=TRAIL, width=1)

        # every moving body, coloured by what it is
        for name, position in record["actor_xy"].items():
            kind = record["actor_kind"].get(name, "")
            radius = 4.0 if name == threat else 3.0
            self._disc(draw, position, radius, fill=kind_color(kind))
            if name == threat:
                px, py = self.to_px(position)
                draw.ellipse([px - radius - 3, py - radius - 3,
                              px + radius + 3, py + radius + 3], outline=INK)

        # the duck, at its exact measured half-extent
        duck = self.to_px(record["duck_xy"])
        r = max(DUCK_DRAW_RADIUS_M * self.scale, 2.0)
        draw.ellipse([duck[0] - r, duck[1] - r, duck[0] + r, duck[1] + r],
                     fill=GOOD, outline=INK)

        # the sightline to whatever it is watching, coloured by whether the
        # real camera can see it
        target = (GOAL_XY if record["subject"] == "goal"
                  else record["actor_xy"].get(record["subject"]))
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
        title(draw, self.box,
              "TIMELINE   five encounters, each a decision")
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

        # Second row: the run's decisions so far, as chips.  Accumulated from
        # what has ALREADY happened by this tick, never from the whole run - a
        # viewer must not see a decision marked before it is taken.
        y = top + height + 4
        draw.text((left, y), f"{record['t']:6.2f} s", font=F09, fill=DIM)
        chip = left + 62
        for index, side in enumerate(record["pass_sides"]):
            draw.rectangle([chip, y + 2, chip + 8, y + 9], fill=side_ink(side))
            draw.text((chip + 12, y), f"{index + 1}:{side[0].upper()}",
                      font=F09, fill=DIM)
            chip += 46
        if record["state"] == "WAIT":
            draw.text((chip, y), "WAITING - neither side safe", font=F09,
                      fill=STATE_COLORS["WAIT"])
