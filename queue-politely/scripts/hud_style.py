#!/usr/bin/env python3
"""HUD palette, fonts and drawing primitives.

Shared by every panel so the overlay has one visual language, and so a text
measurement helper exists in exactly one place - the gap scorecard's columns are
laid out from measured text width rather than fixed offsets, after an earlier
version overlapped "fits" into "REJECT" on every row.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from queue_geometry import (
    BARRIER_HALF_M,
    DUCK_PLANAR_RADIUS,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
)
from queue_path import PATH
from queue_people import ADULT_HALF_EXTENT_M, CLERK, QUEUE_NAMES

PANEL = (16, 18, 24)
INK = (232, 236, 244)
DIM = (150, 158, 172)
GOOD = (78, 210, 130)
BAD = (238, 96, 78)
WARN = (246, 190, 86)
ACCENT = (96, 176, 246)
LANE = (70, 92, 122)

STATE_COLORS = {
    "APPROACH": ACCENT,
    "OBSERVE_QUEUE": WARN,
    "IDENTIFY_TAIL": WARN,
    "EVALUATE_GAPS": WARN,
    "JOIN": GOOD,
    "WAIT": DIM,
    "ADVANCE": GOOD,
    "AT_COUNTER": GOOD,
    "DONE": GOOD,
}


def _font(size: int):
    for path in ("/System/Library/Fonts/SFNSMono.ttf",
                 "/System/Library/Fonts/Menlo.ttc",
                 "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F10 = _font(11)
F11 = _font(12)
F13 = _font(14)
F16 = _font(17)


def _text_w(draw, text, font) -> int:
    return int(draw.textbbox((0, 0), text, font=font)[2])


def _panel(draw, box, alpha=205, outline=(70, 78, 94)):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=PANEL + (alpha,), outline=outline)
