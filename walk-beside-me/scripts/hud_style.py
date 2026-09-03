#!/usr/bin/env python3
"""HUD palette, fonts and drawing primitives for the walk-beside-me overlay.

One visual language for every panel, and one place where a text width is
measured.  This module imports nothing from the rollout, so a test that only
wants to check the palette can import it without building a scene.

THE COLOUR CODE IS THE ARGUMENT
--------------------------------
This behavior turns on one question asked twice a tick: **is the side the duck
is standing in still usable?**  So SIDE RISK, not state, owns the strongest
colours:

* ``GOOD`` green  — this side is usable; the measured gaps clear both margins.
* ``BAD`` red     — this side is refused, with a named measured cause.
* ``WARN`` amber  — a manoeuvre is in progress: falling back, crossing, joining.
* ``DIM``         — information the duck has no opinion about.

State colours are drawn from that same small set, so a viewer never has to hold
two unrelated legends in their head.  The two formation states are the only ones
in green, which makes "the duck is beside her right now" readable at a glance
from the timeline alone.

LEFT IS PURPLE AND RIGHT IS BLUE, EVERYWHERE
---------------------------------------------
The single most confusable thing in this video is which side is which, because
the duck changes sides half way through and the wide camera looks at the scene
from behind her. ``LEFT_INK``/``RIGHT_INK`` are therefore used for the slot
discs in the plan view, the two side-risk rows and the side field in the status
panel — the same hue for the same side in all three places.
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

# Her shirt colour in the scene, so the legend swatch and the person in the
# picture are the same teal rather than two different greens.
GUARDIAN_IN_HUD = (37, 157, 152)

# The two sides, one hue each, used identically in every view.
LEFT_INK = (168, 130, 246)
RIGHT_INK = (86, 190, 226)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "ACQUIRE": DIM,
    "JOIN_SIDE": ACCENT,
    "BESIDE_LEFT": GOOD,
    "BESIDE_RIGHT": GOOD,
    "SIDE_BLOCKED": BAD,
    "FALL_BACK": WARN,
    "CROSS_BEHIND": WARN,
    "JOIN_OTHER_SIDE": ACCENT,
}

# One line of plain English per state.  A viewer who has never read the README
# should still be able to say what the robot is doing.
STATE_CAPTION: dict[str, str] = {
    "ACQUIRE": "walking up to her; grading both sides",
    "JOIN_SIDE": "closing into the chosen side slot",
    "BESIDE_LEFT": "walking beside her, on her left",
    "BESIDE_RIGHT": "walking beside her, on her right",
    "SIDE_BLOCKED": "this side is measured unusable",
    "FALL_BACK": "dropping astern to cross behind her",
    "CROSS_BEHIND": "crossing behind her, never in front",
    "JOIN_OTHER_SIDE": "coming up the far side into formation",
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
             bands=()) -> float:
    """A signed value on a fixed scale, with optional acceptance bands drawn.

    Returns the pixel x of the drawn value, so a caller can label it.  The bands
    are what make the lateral offset legible: the viewer sees the 0.45-0.75 m
    acceptance window as a bright region and the live value as a tick inside or
    outside it, rather than having to compare two numbers.

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


def side_ink(side) -> tuple[int, int, int]:
    """The one hue for a side, used identically in every view."""
    if side == 1 or side == "left":
        return LEFT_INK
    if side == -1 or side == "right":
        return RIGHT_INK
    return DIM


def role_color(role: str) -> tuple[int, int, int]:
    """Colour by CAST ROLE: the guardian, the two oncoming, the background.

    This is legend information about the scene rather than the duck's belief.
    What the duck measured about a side is shown in the side-risk panel.
    """
    return {"guardian": GUARDIAN_IN_HUD,
            "oncoming": (222, 128, 96)}.get(role, DIM)
