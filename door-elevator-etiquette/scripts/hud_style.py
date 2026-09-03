#!/usr/bin/env python3
"""HUD palette, fonts and drawing primitives for the door-and-lift overlay.

One visual language for every panel, and one place where a text width is
measured.  This module imports nothing from the rollout, so a test that only
wants to check the palette can import it without building a scene.

THE COLOUR CODE IS THE ARGUMENT
--------------------------------
This behavior turns on one question asked every tick: **is it my turn to move
yet?**  So the strongest colours belong to the answer:

* ``GOOD`` green  - the duck is walking a leg it has been released for.
* ``WARN`` amber  - the duck has stopped and is giving way to somebody.
* ``BAD`` red     - somebody is in an aperture the duck wants, or a door is shut.
* ``DIM``         - information the duck has no opinion about.

Six of the thirteen states are amber, and that is the point of the timeline: a
viewer should see at a glance that most of this behavior is the robot waiting.

THE DOORS KEEP THEIR OWN COLOUR EVERYWHERE
-------------------------------------------
A door's open fraction is the quantity the whole second half of the run turns
on, so it gets its own hue and appears identically in the door panel, the plan
view and the timeline - one number, drawn the same way wherever it is shown.
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

# The guardian's shirt colour in the scene, so the legend swatch and the person
# in the picture are the same pink rather than two different reds.
GUARDIAN_IN_HUD = (224, 73, 122)
# The door open fraction's own hue; see the module docstring.
DOOR_INK = (120, 196, 232)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "APPROACH_DOOR": ACCENT,
    "YIELD_EXITERS": WARN,
    "FOLLOW_THROUGH": GOOD,
    "APPROACH_LIFT": ACCENT,
    "WAIT_SIDE": WARN,
    "DOORS_OPEN": WARN,
    "LET_OCCUPANTS_EXIT": WARN,
    "FOLLOW_GUARDIAN_IN": GOOD,
    "POSITION_INSIDE": GOOD,
    "RIDE": (120, 240, 180),
    "DOORS_OPEN_TARGET": WARN,
    "FOLLOW_OUT": GOOD,
    "DONE": DIM,
}

# One line of plain English per state.  A viewer who has never read the README
# should still be able to say what the robot is doing.
STATE_CAPTION: dict[str, str] = {
    "APPROACH_DOOR": "walking up to the automatic door",
    "YIELD_EXITERS": "stopped outside the threshold: two people are coming out",
    "FOLLOW_THROUGH": "through the doorway, behind her - never alongside",
    "APPROACH_LIFT": "crossing the lobby to the lift",
    "WAIT_SIDE": "waiting BESIDE the doors, not in front of them",
    "DOORS_OPEN": "the doors are opening; holding still",
    "LET_OCCUPANTS_EXIT": "letting everybody out before going in",
    "FOLLOW_GUARDIAN_IN": "boarding after her",
    "POSITION_INSIDE": "moving to the side of the car, out of the way",
    "RIDE": "riding: exactly still in a sealed car",
    "DOORS_OPEN_TARGET": "arrived; she steps out first",
    "FOLLOW_OUT": "following her off the lift",
    "DONE": "off the lift, on the target floor",
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


def door_ink(fraction: float) -> tuple[int, int, int]:
    """The colour of a door at a given open fraction.

    Red while it is shut enough to be impassable, the door hue once it is open
    enough to walk through.  The threshold is ``lobby_doors``' own, so the HUD
    and the gate agree on what "open" means by construction.
    """
    from lobby_doors import DOOR_PASSABLE_FRACTION
    return DOOR_INK if fraction >= DOOR_PASSABLE_FRACTION else BAD


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
    """Colour by CAST ROLE: the guardian, the traffic, the background.

    This is legend information about the SCENE rather than the duck's belief.
    What the duck measured about a person is shown in the traffic panel.
    """
    return {"guardian": GUARDIAN_IN_HUD,
            "door_exiter": (222, 128, 96),
            "occupant": (246, 190, 86)}.get(role, DIM)

