#!/usr/bin/env python3
"""HUD palette, fonts and drawing primitives for the lost-child overlay.

One visual language for every panel, and one place where a text width is
measured.  The identity panel lays its columns out from MEASURED text width
rather than fixed offsets, because the refusal reasons are full sentences of
very different lengths ("wears a cap; guardian does not" against "shirt colour
differs from the guardian's teal") and fixed columns overlapped them.

THE COLOUR CODE IS THE ARGUMENT
--------------------------------
The whole behavior turns on one distinction: the duck is either looking at its
GUARDIAN or looking at somebody who merely resembles her.  So identity, not
state, owns the strongest colours:

* ``GOOD`` green  — the guardian, accepted, confirmed.
* ``BAD`` red     — a refused candidate.
* ``WARN`` amber  — a candidate still being evaluated, verdict not yet reached.
* ``DIM``         — everybody the duck has no opinion about.

State colours are deliberately drawn from the same small set so a viewer never
has to hold two unrelated legends in their head.

This module imports nothing from the rollout, so it can be imported by a test
that only wants to check the palette.
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
TRAIL = (128, 108, 196)
GRID = (58, 66, 82)
OUTLINE = (70, 78, 94)

# The guardian's actual shirt colour, so the legend swatch and the person in the
# picture are the same teal rather than two different greens.
GUARDIAN_IN_HUD = (38, 158, 153)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "FOLLOW": GOOD,
    "LOST": BAD,
    "STOP": BAD,
    "SEARCH_SWEEP": WARN,
    "CANDIDATE": WARN,
    "REJECT": BAD,
    "REACQUIRED": ACCENT,
    "REJOIN": ACCENT,
    "SAFE": GOOD,
    "DONE": GOOD,
}

# One line of plain English per state, shown in the status panel.  A viewer who
# has never read the README should still be able to say what the robot is doing.
STATE_CAPTION: dict[str, str] = {
    "FOLLOW": "following the guardian",
    "LOST": "guardian lost - halting",
    "STOP": "stopped; position unknown",
    "SEARCH_SWEEP": "sweeping the hall with the head",
    "CANDIDATE": "evaluating a candidate",
    "REJECT": "candidate refused",
    "REACQUIRED": "guardian identity confirmed",
    "REJOIN": "walking the planned rejoin",
    "SAFE": "rejoined, at safe standoff",
    "DONE": "done",
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
F16 = _font(17)


def text_w(draw, text: str, font) -> int:
    return int(draw.textbbox((0, 0), text, font=font)[2])


def panel(draw, box, alpha: int = 205, outline=OUTLINE) -> None:
    draw.rectangle(box, fill=PANEL + (alpha,), outline=outline)


def title(draw, box, text: str) -> None:
    """A panel's heading, ellipsised to the panel so it cannot spill outside."""
    max_width = box[2] - box[0] - 16 if len(box) > 2 else 10_000
    draw.text((box[0] + 8, box[1] + 4),
              _ellipsise(draw, text, F11, max_width), font=F11, fill=DIM)


def bar(draw, box, fraction: float, color, back=(38, 44, 56)) -> None:
    """A horizontal fill bar, clamped, used for scores and confirmation timers."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=back)
    fraction = min(max(float(fraction), 0.0), 1.0)
    if fraction > 0.0:
        draw.rectangle([x0, y0, x0 + (x1 - x0) * fraction, y1], fill=color)


def verdict_color(verdict: str) -> tuple[int, int, int]:
    return {"accept": GOOD, "candidate": WARN, "reject": BAD}.get(verdict, DIM)


def wrap(draw, text: str, font, max_width: int, max_lines: int = 2) -> list[str]:
    """Greedy word wrap to a MEASURED pixel width.

    The refusal reasons are full sentences of very different lengths, so the
    panels wrap on measured width rather than on a character count.  Overflow is
    ellipsised on the LAST PERMITTED line rather than allowed to run past the
    panel, because an overflowing reason lands on top of the panel below it.
    """
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    line = ""
    for word in words:
        probe = f"{line} {word}".strip()
        if line and int(draw.textbbox((0, 0), probe, font=font)[2]) > max_width:
            lines.append(line)
            if len(lines) == max_lines:
                # No room left: mark the tail as elided and stop.
                lines[-1] = _ellipsise(draw, f"{line} \u2026", font, max_width)
                return lines
            line = word
        else:
            line = probe
    if line:
        lines.append(_ellipsise(draw, line, font, max_width))
    return lines[:max_lines]


def _ellipsise(draw, text: str, font, max_width: int) -> str:
    while text and int(draw.textbbox((0, 0), text, font=font)[2]) > max_width:
        text = (text[:-2] + "\u2026") if len(text) > 2 else text[:-1]
    return text


def fit(draw, text: str, font, max_width: int) -> str:
    """One line, ellipsised to fit.  For headings and single-line fields."""
    return _ellipsise(draw, text, font, max_width)


def role_color(role: str) -> tuple[int, int, int]:
    """Colour by CAST ROLE, so a look-alike is visually distinct from the crowd.

    This is legend information about the scene, not the duck's belief: the duck
    does not know who is a look-alike.  The identity panel shows the belief.
    """
    return {"guardian": GUARDIAN_IN_HUD, "lookalike": WARN}.get(role, DIM)
