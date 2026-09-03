#!/usr/bin/env python3
"""The two spatial HUD views: the plan view and the timeline.

The plan view draws the REAL training area - every fixture at its true
footprint, every person at their measured position, the duck at its measured
planar radius - so the behavior can be graded geometrically instead of taken on
trust.  Four things it draws are the argument:

* **who is locked**, drawn as a ring around one person only, so a viewer can
  see the other four being seen and not obeyed;
* **the commanded heading ray**, drawn from the duck along the heading its
  current command is closing on, which is what makes a LEFT turn and a RIGHT
  turn visibly opposite in the plan rather than only in a number;
* **the safe standoff ring** around the instructor during an approach;
* **the duck's own trail**, which is what makes the approach, the two arcs and
  the reverse legible as one continuous path.

The timeline places every state against wall-clock time, so the whole session
reads as one picture: six commands, each a READY -> OBSERVE -> CONFIRM burst
followed by a coloured execute block, with the refused gestures marked beneath.
"""

from __future__ import annotations

import math

from hud_style import (
    BAD,
    DIM,
    F09,
    GOOD,
    HEADING,
    INK,
    LOCKED,
    OTHER,
    OUTLINE,
    READING,
    STANDOFF,
    STATE_COLORS,
    TRAIL,
    ZERO,
    panel,
    title,
)
from gest_arena import (
    DUCK_START,
    FIXTURES,
    FLOOR_HALF,
    INSTRUCTOR_MARK,
    MARK_HALF,
)

# The duck's MEASURED pose-zero planar half-extent on this scene.  Deliberately
# NOT the conservative bounding-sphere figure the clearance gates use: that one
# over-states the robot on purpose, and drawing it would put a body on screen
# wider than the robot actually is.
DUCK_DRAW_RADIUS_M = 0.0827

STANDOFF_MIN_M = 0.45
STANDOFF_MAX_M = 0.75


class PlanView:
    """A scale plan of the training area, redrawn every frame."""

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
        title(draw, self.box, "PLAN   the training area, to scale")

        # the floor
        self._rect(draw, (0.0, 0.0), FLOOR_HALF, fill=(26, 30, 38),
                   outline=OUTLINE)

        # the two painted marks
        self._rect(draw, INSTRUCTOR_MARK, MARK_HALF, fill=(30, 44, 60),
                   outline=LOCKED)
        self._rect(draw, DUCK_START, MARK_HALF, fill=(38, 34, 58),
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

        # THE DUCK'S OWN TRAIL
        if len(trail) >= 2:
            points = [self.to_px(p) for p in trail]
            draw.line(points, fill=TRAIL, width=2)

        locked = record.get("locked", "")
        bodies = record.get("bodies", {})

        # THE SAFE STANDOFF BAND around the instructor, drawn as a ring so the
        # approach can be seen to END inside it rather than through it.
        if locked and locked in bodies and bodies[locked]["present"]:
            centre = bodies[locked]["xy"]
            for radius, ink in ((STANDOFF_MIN_M, STANDOFF),
                                (STANDOFF_MAX_M, STANDOFF)):
                self._disc(draw, centre, radius * self.scale, outline=ink)

        # every person, at their measured position and facing
        for name, entry in bodies.items():
            if not entry["present"]:
                continue
            xy = entry["xy"]
            is_locked = name == locked
            ink = LOCKED if is_locked else OTHER
            self._disc(draw, xy, max(0.13 * self.scale, 2.5), fill=ink)
            # facing tick, so a viewer can see the instructor faces the duck
            yaw = math.radians(float(entry["yaw_deg"]))
            nose = (xy[0] + 0.30 * math.cos(yaw), xy[1] + 0.30 * math.sin(yaw))
            draw.line([self.to_px(xy), self.to_px(nose)], fill=ink, width=1)
            # THE LOCK RING: exactly one person ever wears it.
            if is_locked:
                self._disc(draw, xy, 0.26 * self.scale, outline=LOCKED)
            # a person mid-gesture is marked, so an ignored one is visible
            if entry.get("gesture", "REST") not in ("REST", ""):
                self._disc(draw, xy, 0.20 * self.scale,
                           outline=READING if not is_locked else GOOD)
            px, py = self.to_px(xy)
            draw.text((px + 6, py - 12), name, font=F09, fill=ink)

        # THE DUCK, at its measured planar radius
        duck = record.get("duck_xy", [0.0, 0.0])
        peak = float(record.get("command_peak", 0.0))
        duck_ink = ZERO if peak == 0.0 else STATE_COLORS.get(
            record.get("state", ""), INK)
        self._disc(draw, duck, max(DUCK_DRAW_RADIUS_M * self.scale, 2.0),
                   fill=duck_ink)

        # THE COMMANDED HEADING RAY.  This is what makes the two turns visibly
        # opposite in the plan rather than only in the ACTION panel.
        yaw = math.radians(float(record.get("duck_yaw_deg", 0.0)))
        tip = (duck[0] + 0.55 * math.cos(yaw), duck[1] + 0.55 * math.sin(yaw))
        draw.line([self.to_px(duck), self.to_px(tip)], fill=HEADING, width=2)

        # the approach target, when there is one
        target = record.get("target_xy")
        if target is not None:
            self._disc(draw, target, max(0.07 * self.scale, 2.0),
                       outline=GOOD)


class Timeline:
    """Every state against wall-clock time, as one readable strip."""

    def __init__(self, box, seconds: float):
        self.box = box
        self.seconds = max(float(seconds), 1e-6)

    def draw(self, draw, record, history, refusals, interrupts) -> None:
        panel(draw, self.box)
        title(draw, self.box, "SESSION   every state, on one timeline")
        x0, y0, x1, y1 = self.box
        left, right = x0 + 10, x1 - 10
        top = y0 + 24
        height = 14

        def to_px(t: float) -> float:
            return left + (right - left) * min(
                max(float(t) / self.seconds, 0.0), 1.0)

        draw.rectangle([left, top, right, top + height], fill=(28, 32, 42))

        # every state as a coloured span
        for entry in history:
            colour = STATE_COLORS.get(entry["state"], DIM)
            a, b = to_px(entry["from_s"]), to_px(entry["to_s"])
            if b - a < 1.0:
                b = a + 1.0
            draw.rectangle([a, top, b, top + height], fill=colour)

        # the REFUSALS, marked underneath: half the behavior is saying no
        for entry in refusals:
            px = to_px(entry["t"])
            draw.line([(px, top + height + 2), (px, top + height + 8)],
                      fill=BAD, width=1)

        # the INTERRUPT, marked above, because it is the one moment a command
        # cut into another one
        for entry in interrupts:
            px = to_px(entry["t"])
            draw.line([(px, top - 7), (px, top - 1)], fill=ZERO, width=2)

        # now
        px = to_px(record.get("t", 0.0))
        draw.line([(px, top - 4), (px, top + height + 4)], fill=INK, width=1)

        y = top + height + 12
        draw.text((left, y), "0s", font=F09, fill=DIM)
        draw.text((right - 26, y), f"{self.seconds:.0f}s", font=F09, fill=DIM)
        draw.text((left + 44, y), "| refused    | STOP interrupt", font=F09,
                  fill=DIM)
