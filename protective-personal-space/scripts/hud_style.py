#!/usr/bin/env python3
"""HUD palette, fonts and drawing primitives for the gesture overlay.

One visual language for every panel, and one place where a text width is
measured.  This module imports nothing from the rollout, so a test that only
wants to check the palette can import it without building a scene.

THE COLOUR CODE IS THE ARGUMENT
--------------------------------
This behavior asks two questions repeatedly: **who is talking to me**, and
**what did they say**?  So the strongest colours belong to the answers:

* ``LOCKED`` blue  - the instructor: the one person whose gestures count.
* ``OTHER`` grey   - everybody else.  Deliberately dull, because the whole
  point is that a stranger's perfectly good gesture changes nothing.
* ``READING`` amber - a gesture is being read but is NOT yet confirmed.
* ``GOOD`` green   - confirmed and being carried out.
* ``BAD`` red      - refused: the ambiguous pose, or a stranger's command.
* ``ZERO`` violet  - the command register is at an EXACT zero.  It has its own
  colour because "it stopped" is the claim this behavior most needs a viewer to
  be able to check at a glance.
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

# The locked instructor, matching the scene material ``focusmat``.
LOCKED = (96, 176, 246)
# Everybody else.  Dull on purpose - see the module docstring.
OTHER = (128, 136, 150)
# A gesture being read but not yet confirmed, matching ``targetmat``.
READING = (246, 190, 86)
# The command register at an exact zero.
ZERO = (176, 148, 246)
# The commanded heading ray, matching ``headingmat``.
HEADING = (92, 200, 250)
# The safe standoff band the approach must end inside.
STANDOFF = (250, 92, 107)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "READY": DIM,
    "OBSERVE": LOCKED,
    "CONFIRM": READING,
    "EXECUTE_APPROACH": GOOD,
    "EXECUTE_STOP": ZERO,
    "EXECUTE_TURN_LEFT": (140, 210, 250),
    "EXECUTE_TURN_RIGHT": (250, 150, 90),
    "EXECUTE_BACK_UP": (200, 120, 240),
    "ACK": (76, 240, 153),
    "GOODBYE": (250, 120, 140),
    "DONE": DIM,
}

# One line of plain English per state.  A viewer who has never read the README
# should still be able to say what the robot is doing.
STATE_CAPTION: dict[str, str] = {
    "READY": "sweeping the head, looking for its instructor",
    "OBSERVE": "watching her, command exactly zero, nothing read yet",
    "CONFIRM": "reading a gesture: timing it before acting",
    "EXECUTE_APPROACH": "COME: closing to a safe standoff distance",
    "EXECUTE_STOP": "STOP: command cut to an exact zero, holding",
    "EXECUTE_TURN_LEFT": "TURN LEFT: walking a real left arc",
    "EXECUTE_TURN_RIGHT": "TURN RIGHT: walking a real right arc",
    "EXECUTE_BACK_UP": "BACK UP: reversing along its own heading",
    "ACK": "acknowledging, standing still",
    "GOODBYE": "waved off: acknowledging the end of the session",
    "DONE": "session complete, standing down",
}

# Colour by what the duck DECIDED about a reading.
READING_COLORS: dict[str, tuple[int, int, int]] = {
    "confirmed": GOOD,
    "reading": READING,
    "refused": BAD,
    "ignored": OTHER,
}


def reading_ink(kind: str) -> tuple[int, int, int]:
    return READING_COLORS.get(kind, DIM)


def kind_color(role: str) -> tuple[int, int, int]:
    """Colour by a person's ROLE in the scene: the instructor, or an adult.

    This is legend information about the SCENE rather than the duck's belief.
    Who the duck actually locked onto is shown by :data:`LOCKED`.
    """
    return {"instructor": (120, 190, 250),
            "adult": (170, 180, 190)}.get(role, DIM)


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
