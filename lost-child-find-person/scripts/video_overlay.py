#!/usr/bin/env python3
"""Compose one output frame: wide shot, head-camera PiP, and the HUD panels.

Layout only.  The panels live in ``hud_panels``, the spatial views in
``hud_views`` and the palette in ``hud_style``; this module decides where each
goes and draws the PiP chrome.

THE PiP CHROME CARRIES THE TWO DISCLOSURES
-------------------------------------------
The picture-in-picture is rendered from the EXACT camera every visibility
measurement is taken through, so what the viewer sees and what the gate graded
are the same frustum.  Two things about it must never be implied away, so they
are drawn into the PiP itself rather than left to the README:

* it is **stabilized** — the rig sits at the physical head camera's position but
  holds a level horizon, and it is labelled as such;
* the identity read off it is a **semantic proxy**, not an RGB classifier.

The footer under the PiP names the body the head is aimed at and whether the
camera can see it, which is the single most useful caption in the video.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from hud_panels import (
    draw_candidate,
    draw_guardian,
    draw_legend,
    draw_refusals,
    draw_status,
)
from hud_style import (
    BAD,
    DIM,
    F09,
    F10,
    F13,
    GOOD,
    INK,
    WARN,
    fit,
    panel,
    text_w,
)
from hud_views import PlanView, Timeline
from lost_cast import GUARDIAN

TITLE = "microduck - LOST CHILD / FIND MY PERSON"
SUBTITLE = ("the duck stands still whenever it does not know where its "
            "guardian is")


def _draw_pip(image, draw, pip, record, origin) -> tuple[int, int]:
    """Paste the PiP and draw its chrome.  Returns its bottom-right corner."""
    pip_image = Image.fromarray(np.asarray(pip)).convert("RGB")
    pip_w, pip_h = pip_image.size
    pip_x, pip_y = origin
    image.paste(pip_image, (pip_x, pip_y))
    draw.rectangle([pip_x - 1, pip_y - 1, pip_x + pip_w, pip_y + pip_h],
                   outline=(96, 104, 124))

    draw.rectangle([pip_x, pip_y, pip_x + pip_w, pip_y + 17],
                   fill=(12, 14, 20, 210))
    draw.text((pip_x + 6, pip_y + 3), "HEAD CAMERA (stabilized)", font=F09,
              fill=DIM)
    mode = "SWEEPING" if record["scanning"] else "tracking"
    draw.text((pip_x + pip_w - 6 - text_w(draw, mode, F09), pip_y + 3), mode,
              font=F09, fill=WARN if record["scanning"] else DIM)

    # Footer: who the head is aimed at, and whether the camera has them.
    subject = record.get("subject")
    visible = record["guardian_visible"]
    if subject == GUARDIAN.name or subject is None:
        text = (f"guardian priya  {'VISIBLE' if visible else 'not visible'}")
        color = GOOD if visible else BAD
    else:
        seen = record["person_visible"].get(subject, False)
        text = f"candidate {subject}  {'in frame' if seen else 'lost from frame'}"
        color = WARN if seen else BAD
    draw.rectangle([pip_x, pip_y + pip_h - 17, pip_x + pip_w, pip_y + pip_h],
                   fill=(12, 14, 20, 210))
    draw.text((pip_x + 6, pip_y + pip_h - 15),
              fit(draw, text, F09, pip_w - 12), font=F09, fill=color)
    return pip_x, pip_y + pip_h


def compose(main: np.ndarray, pip: np.ndarray, *, record: dict,
            total_seconds: float, summary: dict,
            last_sighting: dict | None = None) -> Image.Image:
    """Compose one output frame from the wide shot, the PiP and the record."""
    image = Image.fromarray(np.asarray(main)).convert("RGB")
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Title strip.  The subtitle is drawn only if it fits beside the title
    # without colliding, because the title is fixed and the frame width is not.
    panel(draw, (0, 0, width, 30), alpha=195, outline=None)
    draw.text((12, 7), TITLE, font=F13, fill=INK)
    subtitle_x = width - 12 - text_w(draw, SUBTITLE, F09)
    if subtitle_x > 24 + text_w(draw, TITLE, F13):
        draw.text((subtitle_x, 11), SUBTITLE, font=F09, fill=DIM)

    # PiP, top right.
    pip_h = np.asarray(pip).shape[0]
    pip_w = np.asarray(pip).shape[1]
    pip_x, pip_bottom = _draw_pip(image, draw, pip, record,
                                  (width - pip_w - 12, 38))

    # Left column: state, guardian, identity check, refusal log, cast legend.
    # The boxes are sized to the tallest content each can hold — the identity
    # panel must fit four term bars plus a two-line reason, and the refusal log
    # three entries of two lines — so nothing overflows at 640 px high.
    draw_status(draw, (12, 38, 316, 176), record)
    draw_guardian(draw, (12, 182, 316, 298), record)
    draw_candidate(draw, (12, 304, 316, 494), record,
                   last_sighting=last_sighting)
    draw_refusals(draw, (12, 500, 316, 600), record)
    draw_legend(draw, (12, 606, 316, 632))

    # Right column under the PiP: the plan view.
    PlanView((width - 320, pip_bottom + 8, width - 12, height - 92)).draw(
        draw, record)
    # Timeline across the bottom, clear of the left column.  84 px tall so the
    # staggered refusal labels sit between the header and the state spine.
    Timeline((324, height - 86, width - 12, height - 12), total_seconds).draw(
        draw, record, summary)

    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
