#!/usr/bin/env python3
"""HUD palette, fonts and drawing primitives for the dynamic-slalom overlay.

One visual language for every panel, and one place where a text width is
measured.  This module imports nothing from the rollout, so a test that only
wants to check the palette can import it without building a scene.

THE COLOUR CODE IS THE ARGUMENT
--------------------------------
This behavior turns on one question asked ten times a second: **which way round,
or neither?**  So the strongest colours belong to the answer:

* ``GOOD`` green  - the duck is walking, on a corridor it has committed to.
* ``WARN`` amber  - the duck has stopped because NEITHER side was safe.
* ``BAD`` red     - a predicted clearance below the planner's own bar.
* ``LEFT``/``RIGHT`` - the two corridors, and they keep their hue everywhere:
  in the panel, in the plan view, on the floor of the scene itself.

A viewer should be able to look at the corridor colours on the floor, the
corridor colours in the decision panel, and the timeline, and see the same
decision three times.
"""

from __future__ import annotations

from PIL import ImageFont

# -- palette ----------------------------------------------------------------
PANEL = (16, 18, 24)
INK = (232, 236, 244)
DIM = (150, 158, 172)
GOOD = (78, 210, 130)
BAD = (238, 96, 78)
WARN = (246, 190, 86)
ACCENT = (96, 176, 246)
GRID = (58, 66, 82)
OUTLINE = (70, 78, 94)
TRAIL = (128, 108, 196)
ROUTE = (92, 200, 250)

# The two corridors.  These MATCH the scene materials ``leftmat`` and
# ``rightmat`` in ``tools/build_scene.py``, so the lane drawn on the depot floor
# and the lane named in the HUD are the same colour.
LEFT_INK = (107, 219, 250)
RIGHT_INK = (250, 184, 92)
# The predicted-occupancy discs, matching ``predmat``.
PRED_INK = (250, 92, 107)
# The goal band, matching ``goalmat``.
GOAL_INK = (66, 230, 133)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "PLAN": ACCENT,
    "ADVANCE": GOOD,
    "THREAT": (250, 140, 90),
    "CHOOSE_LEFT": LEFT_INK,
    "CHOOSE_RIGHT": RIGHT_INK,
    "WAIT": WARN,
    "PASS": (120, 240, 180),
    "REPLAN": ACCENT,
    "GOAL": GOAL_INK,
    "DONE": DIM,
}

# One line of plain English per state.  A viewer who has never read the README
# should still be able to say what the robot is doing.
STATE_CAPTION: dict[str, str] = {
    "PLAN": "planning a line to the arrival band",
    "ADVANCE": "walking the planned line to the goal",
    "THREAT": "somebody is predicted to cross: scoring both corridors",
    "CHOOSE_LEFT": "turning out onto the LEFT corridor",
    "CHOOSE_RIGHT": "turning out onto the RIGHT corridor",
    "WAIT": "stopped: NEITHER corridor is predicted safe",
    "PASS": "passing, on the corridor it committed to",
    "REPLAN": "the crossing is behind: replanning to the goal",
    "GOAL": "arrived, standing in the band",
    "DONE": "at the destination",
}


def side_ink(side: str) -> tuple[int, int, int]:
    """The colour of a corridor, wherever it is drawn."""
    return {"left": LEFT_INK, "right": RIGHT_INK, "wait": WARN}.get(side, DIM)


def _font(size: int):
    """A monospaced face if the host has one, so numeric columns line up."""
    for path in ("/System/Library/Fonts/SFNSMono.ttf",
                 "/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Supplemental/Courier New.ttf",
                 "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F09 = _font(10)
F10 = _font(11)
F11 = _font(12)
F13 = _font(14)


def clearance_ink(value: float, bar: float) -> tuple[int, int, int]:
    """Green above the planner's own safety bar, red below it.

    The threshold is ``slalom_states.SAFE_CLEARANCE_M`` itself, so the HUD and
    the planner agree on what "safe" means by construction rather than by two
    numbers that happen to match.
    """
    return GOOD if value >= bar else BAD


def text_w(draw, text: str, font) -> int:
    return int(draw.textbbox((0, 0), text, font=font)[2])


def panel(draw, box, alpha: int = 205, outline=OUTLINE) -> None:
    draw.rectangle(box, fill=PANEL + (alpha,), outline=outline)


def title(draw, box, text: str) -> None:
    """A panel heading, ellipsised so it cannot spill outside its panel."""
    max_width = box[2] - box[0] - 16
    draw.text((box[0] + 8, box[1] + 4), fit(draw, text, F11, max_width),
              font=F11, fill=DIM)


def bar(draw, box, fraction: float, color, back=(38, 44, 56)) -> None:
    """A horizontal fill bar, clamped."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=back)
    fraction = min(max(float(fraction), 0.0), 1.0)
    if fraction > 0.0:
        draw.rectangle([x0, y0, x0 + (x1 - x0) * fraction, y1], fill=color)


def span_bar(draw, box, lo: float, hi: float, value: float, color,
             bands=(), marks=()) -> float:
    """A value on a fixed scale, with optional acceptance bands and marks.

    Returns the pixel x of the drawn value, so a caller can label it.  The bands
    are what make a distance legible: the viewer sees the acceptance window as a
    bright region and the live value as a tick inside or outside it, rather than
    having to compare two numbers.

    ``bands`` is a SEQUENCE and every band is drawn in this one call, because
    each call repaints the background: calling twice to add a second band erases
    the first one and silently leaves half the scale unmarked.
    """
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(30, 35, 45), outline=(52, 58, 72))
    span = max(hi - lo, 1e-9)

    def to_px(v: float) -> float:
        return x0 + (x1 - x0) * min(max((v - lo) / span, 0.0), 1.0)

    for band_lo, band_hi in bands:
        draw.rectangle([to_px(band_lo), y0 + 1, to_px(band_hi), y1 - 1],
                       fill=(38, 62, 52))
    for mark_value, mark_color in marks:
        px = to_px(mark_value)
        draw.line([(px, y0), (px, y1)], fill=mark_color, width=1)
    if lo < 0.0 < hi:
        zero = to_px(0.0)
        draw.line([(zero, y0), (zero, y1)], fill=GRID)
    px = to_px(value)
    draw.line([(px, y0 - 2), (px, y1 + 2)], fill=color, width=2)
    return px


def _ellipsise(draw, text: str, font, max_width: int) -> str:
    while text and int(draw.textbbox((0, 0), text, font=font)[2]) > max_width:
        text = (text[:-2] + "\u2026") if len(text) > 2 else text[:-1]
    return text


def fit(draw, text: str, font, max_width: int) -> str:
    """One line, ellipsised to fit.  For headings and single-line fields."""
    return _ellipsise(draw, text, font, max_width)


def kind_color(kind: str) -> tuple[int, int, int]:
    """Colour by what a body IS: a pedestrian, a cart, a carried box.

    This is legend information about the SCENE rather than the duck's belief.
    What the duck measured about a body is shown in the traffic panel.
    """
    return {"pedestrian": (224, 73, 122),
            "cart": (140, 170, 250),
            "box": (222, 170, 96)}.get(kind, DIM)
