#!/usr/bin/env python3
"""The two spatial HUD views: the scale plan of the plaza, and the timeline.

THE PLAN VIEW IS WHERE THE GEOMETRY CLAIM IS CHECKABLE
--------------------------------------------------------
Every geometric claim this behavior makes is about WHERE the duck stood relative
to two moving people, and a wide camera shot flattens exactly that.  So the plan
draws, to scale and every frame:

* the **buffer disc** around the protected person - the 1.95 m radius the whole
  behavior is defined against, drawn as a filled region so an intruder can be
  seen crossing INTO it;
* the **between-line** from Aina to the selected intruder, plus the duck's own
  bearing from Aina, so "the duck is between them" is a visible angle rather
  than a boolean in a panel;
* every **prediction** as a ghost marker at the person's predicted closest
  approach, which is what the duck actually reasoned about rather than where
  the person currently is;
* the **station** the duck chose, and the duck's own trail behind it.

Fixtures are drawn at their true footprints, so a viewer can see the escape gap
being chosen away from the planter rather than being told it was.

THE TIMELINE READS AS A SHAPE BEFORE IT READS AS LABELS
---------------------------------------------------------
190 s of thirteen states is too much to label individually at 940 px, so the
states are coloured by PHASE (see :mod:`pps_hud_style`) and the episodes are
marked underneath with the name of the person each one was about.  The result
is legible at a glance: four red-violet protective arcs at alternating times, an
orange yield, and one yellow squeeze.
"""

from __future__ import annotations

import math

from hud_style import F09, F10, panel, title
from pps_hud_style import (BAD, BUFFER, DIM, GOOD, HEADING, INK, OUTLINE,
                           PHASE_LEGEND, SECOND, STATION, THREAT, TRAIL, WARD,
                           WATCHED, ZERO, state_color)
from pps_plaza import DUCK_START, FIXTURES, FLOOR_HALF, MARK_HALF
from pps_states import BUFFER_M, DUCK_PLANAR_RADIUS

# The duck's MEASURED planar half-extent is the conservative bounding-sphere
# figure the clearance gates use; it deliberately OVER-states the robot, so the
# plan draws the smaller true footprint and the gates keep the larger one.
DUCK_DRAW_RADIUS_M = 0.0827
# One trail point per this many control ticks.  190 s at 50 Hz is 9500 ticks;
# drawing every one is both slow and illegible.
TRAIL_STRIDE = 10


class PlanView:
    """A scale plan of the plaza, redrawn every written frame."""

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

    def draw(self, draw, view: dict, trail) -> None:
        panel(draw, self.box)
        title(draw, self.box, "PLAN   the plaza to scale, and the buffer")

        self._rect(draw, (0.0, 0.0), FLOOR_HALF, fill=(26, 30, 38),
                   outline=OUTLINE)
        self._rect(draw, DUCK_START, MARK_HALF, fill=(38, 34, 58),
                   outline=TRAIL)

        for fixture in FIXTURES:
            ink = (96, 104, 122) if fixture.occludes else (64, 70, 84)
            if fixture.kind == "cylinder":
                self._disc(draw, fixture.center,
                           max(fixture.radius * self.scale, 1.5), fill=ink)
            else:
                self._rect(draw, fixture.center, fixture.half, fill=ink)

        ward_xy = view["ward_xy"]
        # THE BUFFER, drawn first so everything else sits on top of it.  This
        # is the 1.95 m disc the entire behavior is defined against.
        self._disc(draw, ward_xy, BUFFER_M * self.scale,
                   fill=(22, 46, 60), outline=BUFFER)

        if len(trail) >= 2:
            draw.line([self.to_px(p) for p in trail], fill=TRAIL, width=2)

        duck = view["duck_xy"]
        selected, secondary = view["selected"], view["secondary"]
        people = view["people"]

        # THE BETWEEN-LINE.  Aina -> intruder is the bearing the duck has to
        # get onto; Aina -> duck is where it actually is.  Drawing both makes
        # between-ness an angle a viewer can see closing.
        if selected and selected in people and people[selected]["present"]:
            threat_xy = people[selected]["xy"]
            draw.line([self.to_px(ward_xy), self.to_px(threat_xy)],
                      fill=THREAT, width=1)
            draw.line([self.to_px(ward_xy), self.to_px(duck)],
                      fill=GOOD if view["between"] else DIM, width=1)
        if secondary and secondary in people and people[secondary]["present"]:
            draw.line([self.to_px(ward_xy), self.to_px(people[secondary]["xy"])],
                      fill=SECOND, width=1)

        # EVERY PREDICTION as a ghost at its predicted closest approach: what
        # the duck reasoned about, not where the person currently stands.
        for entry in view["prediction_points"]:
            self._disc(draw, entry["point"], max(0.05 * self.scale, 1.5),
                       outline=THREAT if entry["intrusion"] else WATCHED)

        for name, entry in people.items():
            if not entry["present"]:
                continue
            xy = entry["xy"]
            ink = view["person_ink"](name)
            self._disc(draw, xy, max(0.13 * self.scale, 2.5), fill=ink)
            yaw = math.radians(float(entry["yaw_deg"]))
            nose = (xy[0] + 0.30 * math.cos(yaw), xy[1] + 0.30 * math.sin(yaw))
            draw.line([self.to_px(xy), self.to_px(nose)], fill=ink, width=1)
            if name == view["ward"]:
                self._disc(draw, xy, 0.26 * self.scale, outline=WARD)
            if name == selected:
                self._disc(draw, xy, 0.22 * self.scale, outline=THREAT)
            px, py = self.to_px(xy)
            draw.text((px + 6, py - 12), name, font=F09, fill=ink)

        # THE STATION the duck chose, and the escort slot it returns to.
        slot = view["escort_slot"]
        self._disc(draw, slot, max(0.09 * self.scale, 2.0), outline=STATION)
        target = view["target"]
        if target is not None:
            self._disc(draw, target, max(0.10 * self.scale, 2.5),
                       outline=STATION, width=2)
            draw.line([self.to_px(duck), self.to_px(target)], fill=STATION,
                      width=1)

        # THE DUCK, coloured violet at an exact zero so a hold is visible in
        # the plan as well as in the COMMAND panel.
        peak = view["command_peak"]
        duck_ink = ZERO if peak == 0.0 else state_color(view["state"])
        self._disc(draw, duck, max(DUCK_DRAW_RADIUS_M * self.scale, 2.0),
                   fill=duck_ink)
        yaw = math.radians(view["duck_yaw_deg"])
        tip = (duck[0] + 0.50 * math.cos(yaw), duck[1] + 0.50 * math.sin(yaw))
        draw.line([self.to_px(duck), self.to_px(tip)], fill=HEADING, width=2)

        draw.text((self.x0 + 10, self.y1 - 14), (
            f"buffer {BUFFER_M:.2f} m   duck radius {DUCK_PLANAR_RADIUS:.3f} m"
        ), font=F09, fill=DIM)


class Timeline:
    """Every state and every episode against wall-clock time, as one strip."""

    def __init__(self, box, seconds: float):
        self.box = box
        self.seconds = max(float(seconds), 1e-6)

    def _to_px(self, t: float) -> float:
        left, right = self.box[0] + 10, self.box[2] - 10
        return left + (right - left) * min(
            max(float(t) / self.seconds, 0.0), 1.0)

    def draw(self, draw, view: dict, history, episodes) -> None:
        panel(draw, self.box)
        title(draw, self.box, "SESSION   every state and episode, 190 s")
        x0, y0, x1, y1 = self.box
        left, right = x0 + 10, x1 - 10
        top = y0 + 22
        height = 13

        draw.rectangle([left, top, right, top + height], fill=(28, 32, 42))
        for entry in history:
            a, b = self._to_px(entry["from_s"]), self._to_px(entry["to_s"])
            if b - a < 1.0:
                b = a + 1.0
            draw.rectangle([a, top, b, top + height],
                           fill=state_color(entry["state"]))

        # EVERY CLOSED EPISODE, marked with the person it was about.  This is
        # what makes the four alternating cycles, the yield and the squeeze
        # legible as six separate things rather than as colour changes.
        kind_ink = {"intrusion": THREAT, "ward_approach": (232, 108, 40),
                    "squeeze": (250, 224, 92)}
        for entry in episodes:
            a = self._to_px(entry["started_at_s"])
            b = self._to_px(entry.get("ended_at_s", entry["started_at_s"]))
            ink = kind_ink.get(entry["kind"], DIM)
            draw.rectangle([a, top + height + 2, max(b, a + 1.0),
                            top + height + 6], fill=ink)
            draw.text((a, top + height + 8), str(entry["selected"]), font=F09,
                      fill=ink)

        # NOW
        px = self._to_px(view["t"])
        draw.line([(px, top - 4), (px, top + height + 4)], fill=INK, width=1)

        y = y1 - 14
        draw.text((left, y), "0s", font=F09, fill=DIM)
        draw.text((right - 30, y), f"{self.seconds:.0f}s", font=F09, fill=DIM)
        legend_x = left + 34
        for label, ink in PHASE_LEGEND:
            draw.rectangle([legend_x, y + 3, legend_x + 9, y + 10], fill=ink)
            draw.text((legend_x + 13, y), label, font=F09, fill=DIM)
            legend_x += 22 + int(draw.textbbox((0, 0), label, font=F09)[2])
