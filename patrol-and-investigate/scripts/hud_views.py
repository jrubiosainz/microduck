#!/usr/bin/env python3
"""The two spatial HUD views: the plan view and the timeline.

The plan view draws the REAL facility - every fixture at its true footprint, the
restricted rectangle where it actually is, every body at its measured position,
the duck at its measured planar radius - so the patrol can be graded
geometrically instead of taken on trust.  Four things it draws are the argument
of the behavior:

* **the patrol circuit**, with the checkpoints marked done, current and pending,
  so the ORDER is a picture;
* **the remembered route** - the line from the interruption point to the
  checkpoint the duck was walking to - drawn for the whole diversion, which is
  the claim this behavior turns on;
* **the restricted zone**, so a viewer can see the duck stay out of it;
* **the duck's own trail**, which is what makes the diversion and the return
  visible as one curve.

The timeline places every state against wall-clock time, so the whole run reads
as one picture: five checkpoint stops, each a burst of STOP / SCAN / CLEAR, with
the two investigation branches standing out as long orange excursions.
"""

from __future__ import annotations

import numpy as np

from hud_style import (
    BAD,
    CHECKPOINT,
    DIM,
    F09,
    GOOD,
    INK,
    MEMORY,
    OUTLINE,
    ROUTE,
    STANDOFF,
    STATE_COLORS,
    TRAIL,
    ZONE,
    kind_color,
    panel,
    title,
    verdict_ink,
)
from patrol_facility import (
    CHECKPOINTS,
    FIXTURES,
    FLOOR_HALF,
    HOME,
    HOME_PAD_HALF,
    RESTRICTED_ZONE,
)

# The duck's MEASURED pose-zero planar half-extent on this scene.  Deliberately
# NOT the conservative bounding-sphere figure the clearance gates use: that one
# over-states the robot on purpose, and drawing it would put a body on screen
# wider than the robot actually is.
DUCK_DRAW_RADIUS_M = 0.0827


class PlanView:
    """A scale plan of the facility floor, redrawn every frame."""

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
        title(draw, self.box, "PLAN   the facility, to scale")

        # the floor
        self._rect(draw, (0.0, 0.0), FLOOR_HALF, fill=(26, 30, 38),
                   outline=OUTLINE)

        # THE RESTRICTED ZONE, so a viewer can see the duck stay out of it.
        self._rect(draw, RESTRICTED_ZONE.center, RESTRICTED_ZONE.half,
                   fill=(62, 30, 26), outline=ZONE)

        # the guard post
        self._rect(draw, HOME.xy, HOME_PAD_HALF, fill=(38, 34, 58),
                   outline=TRAIL)

        # every fixture at its true footprint
        for fixture in FIXTURES:
            if fixture.kind == "cylinder":
                self._disc(draw, fixture.center,
                           max(fixture.radius * self.scale, 1.5),
                           fill=(84, 78, 70))
            else:
                self._rect(draw, fixture.center, fixture.half,
                           fill=(64, 70, 84))

        # THE PATROL CIRCUIT, with each checkpoint marked done/current/pending.
        corners = [HOME.position] + [c.position for c in CHECKPOINTS] \
            + [HOME.position]
        draw.line([self.to_px(p) for p in corners], fill=ROUTE, width=1)
        completed = set(record["completed_names"])
        for checkpoint in CHECKPOINTS:
            if checkpoint.name in completed:
                colour = CHECKPOINT
            elif checkpoint.name == record["target_name"]:
                colour = ROUTE
            else:
                colour = DIM
            self._disc(draw, checkpoint.xy, 3.5, fill=colour)

        # THE REMEMBERED ROUTE: drawn for the whole diversion.
        resume = record["resume_xy"]
        if record["interrupted"] and resume is not None:
            target = next((c.position for c in CHECKPOINTS
                           if c.name == record["interrupted_target"]), None)
            if target is not None:
                draw.line([self.to_px(resume), self.to_px(target)],
                          fill=MEMORY, width=2)
                self._disc(draw, resume, 3.0, fill=MEMORY)

        # the approach line to a standoff, while there is one
        standoff = record["standoff_xy"]
        if standoff is not None and record["state"] == "APPROACH":
            draw.line([self.to_px(record["duck_xy"]), self.to_px(standoff)],
                      fill=STANDOFF, width=1)
            self._disc(draw, standoff, 2.5, fill=STANDOFF)

        # its recent trail: the diversion and the return, as one curve
        if len(trail) > 1:
            draw.line([self.to_px(p) for p in trail], fill=TRAIL, width=1)

        # every PRESENT body, coloured by what it is
        candidate = record["candidate"]
        for name, position in record["actor_xy"].items():
            kind = record["actor_kind"].get(name, "")
            radius = 4.0 if name == candidate else 3.0
            self._disc(draw, position, radius, fill=kind_color(kind))
            if name == candidate:
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
        subject = record["subject"]
        target = record["actor_xy"].get(subject)
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
              "TIMELINE   five checkpoints, two investigations")
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

        # Second row: the checkpoints completed and the verdicts reached SO
        # FAR.  Accumulated from what has already happened by this tick - a
        # viewer must not see a decision marked before it is taken.
        y = top + height + 4
        draw.text((left, y), f"{record['t']:6.2f} s", font=F09, fill=DIM)
        chip = left + 62
        draw.text((chip, y), f"checkpoints {record['completed']}/5", font=F09,
                  fill=CHECKPOINT)
        chip += 108
        for name, verdict in zip(record["verdict_targets"],
                                 record["verdicts_so_far"]):
            draw.rectangle([chip, y + 2, chip + 8, y + 9],
                           fill=verdict_ink(verdict))
            draw.text((chip + 12, y), f"{name[:7]}:{verdict[:4]}", font=F09,
                      fill=DIM)
            chip += 94
        if record["interrupted"]:
            draw.text((chip, y), "PATROL INTERRUPTED - route held", font=F09,
                      fill=MEMORY)
