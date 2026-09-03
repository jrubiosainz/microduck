#!/usr/bin/env python3
"""HUD palette, fonts and drawing primitives for the patrol overlay.

One visual language for every panel, and one place where a text width is
measured.  This module imports nothing from the rollout, so a test that only
wants to check the palette can import it without building a scene.

THE COLOUR CODE IS THE ARGUMENT
--------------------------------
This behavior asks two questions repeatedly: **where am I on my route**, and
**what is that**?  So the strongest colours belong to the answers:

* ``GOOD`` green   - clear: the checkpoint is fine, the thing is explained.
* ``WARN`` amber   - something is being investigated.
* ``BAD`` red      - an escalating verdict: suspicious, or an intrusion.
* ``ROUTE`` blue   - the patrol circuit and the checkpoint being walked to.
* ``MEMORY`` orange - THE ROUTE HELD IN MEMORY while the duck is somewhere
  else.  It is its own colour because it is the thing this behavior exists to
  demonstrate, and a viewer should be able to see it persist on the floor
  throughout an investigation.
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

# The patrol circuit, matching the scene material ``routemat``.
ROUTE = (92, 200, 250)
# The remembered route, matching ``memorymat``.  See the module docstring.
MEMORY = (250, 184, 92)
# The approach line to a standoff, matching ``standoffmat``.
STANDOFF = (250, 92, 107)
# The restricted zone, matching ``zonetapemat``.
ZONE = (250, 107, 56)
# A completed checkpoint, matching ``cpdonemat``.
CHECKPOINT = (76, 240, 153)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "PATROL": ROUTE,
    "CHECKPOINT_STOP": ACCENT,
    "SCAN": (140, 210, 250),
    "CLEAR": GOOD,
    "DETECT": WARN,
    "INVESTIGATE_PLAN": (250, 150, 90),
    "APPROACH": STANDOFF,
    "OBSERVE": (250, 120, 140),
    "CLASSIFY": (240, 90, 110),
    "RETURN_TO_PATROL": MEMORY,
    "RESUME": (200, 190, 120),
    "HOME": CHECKPOINT,
    "DONE": DIM,
}

# One line of plain English per state.  A viewer who has never read the README
# should still be able to say what the robot is doing.
STATE_CAPTION: dict[str, str] = {
    "PATROL": "walking the circuit to the next checkpoint",
    "CHECKPOINT_STOP": "stopped ON the checkpoint, command exactly zero",
    "SCAN": "sweeping the head across the facility",
    "CLEAR": "checkpoint clear: nothing to investigate",
    "DETECT": "something new is in the camera: classifying it",
    "INVESTIGATE_PLAN": "planning a SAFE standoff to observe from",
    "APPROACH": "closing to a safe observation distance",
    "OBSERVE": "holding several viewing angles, standing still",
    "CLASSIFY": "recording what it decided, and why",
    "RETURN_TO_PATROL": "walking back to the interrupted point",
    "RESUME": "patrol resumed toward the SAME checkpoint",
    "HOME": "patrol complete: standing down at the guard post",
    "DONE": "at the guard post",
}

# Colour by VERDICT, which is the duck's own conclusion.
VERDICT_COLORS: dict[str, tuple[int, int, int]] = {
    "suspicious": BAD,
    "intrusion": (250, 76, 140),
    "benign": GOOD,
}


def verdict_ink(verdict: str) -> tuple[int, int, int]:
    return VERDICT_COLORS.get(verdict, DIM)


def kind_color(kind: str) -> tuple[int, int, int]:
    """Colour by what a body IS: staff, an intruder, an object, a trolley.

    This is legend information about the SCENE rather than the duck's belief.
    What the duck concluded is shown by the verdict colours.
    """
    return {"staff": (120, 190, 250),
            "intruder": (250, 200, 90),
            "object": (200, 150, 100),
            "trolley": (170, 180, 190)}.get(kind, DIM)


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
    the first and silently leaves half the scale unmarked.
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


def wrap(draw, text: str, font, max_width: int, max_lines: int = 3
         ) -> list[str]:
    """Word-wrap into at most ``max_lines``, ellipsising the last one.

    The classification RULE is a sentence rather than a number, and it is the
    most important thing the HUD says - so it gets real wrapping instead of
    being cut off at the panel edge.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if int(draw.textbbox((0, 0), candidate, font=font)[2]) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if lines and len(lines) == max_lines:
        lines[-1] = _ellipsise(draw, lines[-1], font, max_width)
    return lines
