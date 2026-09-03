#!/usr/bin/env python3
"""HUD palette, fonts and drawing primitives for the lead-me-somewhere overlay.

One visual language for every panel, and one place where a text width is
measured.  This module imports nothing from the rollout, so a test that only
wants to check the palette can import it without building a scene.

THE COLOUR CODE IS THE ARGUMENT
--------------------------------
This behavior turns on one question asked every tick: **is the person still
with me?**  So FOLLOWER STATE, not robot state, owns the strongest colours:

* ``GOOD`` green  — she is close enough and the camera can see her.
* ``BAD`` red     — she is beyond the measured lag threshold, or unseen.
* ``WARN`` amber  — the duck has stopped and is waiting for her.
* ``DIM``         — information the duck has no opinion about.

State colours are drawn from that same small set, so a viewer never has to hold
two unrelated legends in their head.  The two states in which the duck is
actually leading are the only ones in green, which makes "the duck is walking
her somewhere right now" readable at a glance from the timeline alone.

THE THREE DESTINATIONS KEEP THEIR OWN COLOURS EVERYWHERE
---------------------------------------------------------
The single most important thing this video has to show is that the duck went to
the place that was REQUESTED, out of three that existed.  Each destination's hue
is taken from ``guide_layout.DESTINATIONS`` itself, so the pylon in the picture,
the disc in the plan view and the chip in the request panel are the same colour
by construction rather than by two lists agreeing.
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

# Her shirt colour in the scene, so the legend swatch and the person in the
# picture are the same pink rather than two different reds.
FOLLOWER_IN_HUD = (224, 73, 122)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "RECEIVE_DESTINATION": ACCENT,
    "PLAN": ACCENT,
    "LEAD": GOOD,
    "CHECK_FOLLOWER": BAD,
    "WAIT_FOR_PERSON": WARN,
    "RESUME": GOOD,
    "ARRIVE": ACCENT,
    "INDICATE": (120, 240, 180),
    "DONE": DIM,
}

# One line of plain English per state.  A viewer who has never read the README
# should still be able to say what the robot is doing.
STATE_CAPTION: dict[str, str] = {
    "RECEIVE_DESTINATION": "a person asks to be taken somewhere",
    "PLAN": "searching a route round the partitions and the crowd",
    "LEAD": "leading her along the planned route",
    "CHECK_FOLLOWER": "stopped: she has fallen behind, looking back",
    "WAIT_FOR_PERSON": "waiting here until she catches up",
    "RESUME": "she is back with me; leading on",
    "ARRIVE": "reached the standing point",
    "INDICATE": "this is the place you asked for",
    "DONE": "delivered",
}


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


def destination_ink(key: str) -> tuple[int, int, int]:
    """The one hue for a destination, taken from the layout itself.

    Derived rather than restated, so the pylon in the picture and the chip in
    the HUD cannot drift apart.
    """
    from guide_layout import DESTINATION_BY_KEY
    destination = DESTINATION_BY_KEY.get(key)
    if destination is None:
        return DIM
    r, g, b = destination.color
    return (int(r * 255), int(g * 255), int(b * 255))


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
    """A horizontal fill bar, clamped.  Used for gaps against their margins."""
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


def role_color(role: str) -> tuple[int, int, int]:
    """Colour by CAST ROLE: the person being led, the crowd, the background.

    This is legend information about the scene rather than the duck's belief.
    What the duck measured about her is shown in the follower panel.
    """
    return {"follower": FOLLOWER_IN_HUD,
            "crowd": (222, 128, 96)}.get(role, DIM)

