#!/usr/bin/env python3
"""Compose one output frame: wide shot, head-camera PiP, and the HUD panels.

Layout only.  The panels live in ``hud_panels``, the spatial views in
``hud_views`` and the palette in ``hud_style``; this module decides where each
goes and draws the PiP chrome.

THE PiP CHROME CARRIES THE TWO DISCLOSURES
-------------------------------------------
The picture-in-picture is rendered from the EXACT camera every visibility
measurement is taken through — ``beside_camera.BesideCamera.camera_id``, at the
same 300x216 pixel geometry that sets its horizontal FOV — so what the viewer
sees and what the gate graded are the same frustum.  Two things about it must
never be implied away, so they are drawn into the PiP itself rather than left to
the README:

* it is **stabilized** — the rig sits exactly where the physical head camera
  sits, but holds a level horizon so a human can read it while the duck's trunk
  pitches through its gait;
* the identity read off it is a **semantic proxy** — MuJoCo body identity inside
  a real frustum with a real occlusion ray cast, not an RGB classifier.

The footer names the body the head is aimed at and whether that exact camera can
see her, which is the most useful caption in the video.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from beside_cast import GUARDIAN
from hud_panels import (
    draw_formation,
    draw_legend,
    draw_side_risk,
    draw_status,
)
from hud_style import (
    BAD,
    DIM,
    F09,
    F13,
    GOOD,
    INK,
    fit,
    panel,
    text_w,
)
from hud_views import PlanView, Timeline

TITLE = "microduck - WALK BESIDE ME"
SUBTITLE = ("the duck holds a side, and changes side only when it MEASURES "
            "that side gone")


def _draw_pip(image, draw, pip, record, origin) -> tuple[int, int]:
    """Paste the PiP and draw its chrome.  Returns its bottom-left and bottom."""
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
    label = "semantic proxy"
    draw.text((pip_x + pip_w - 6 - text_w(draw, label, F09), pip_y + 3), label,
              font=F09, fill=DIM)

    visible = record["guardian_visible"]
    samples = record["guardian_sample_count"]
    text = (f"guardian nadia  {'VISIBLE' if visible else 'not visible'}  "
            f"{samples}/5 samples  {record['guardian_range_m']:.2f} m")
    draw.rectangle([pip_x, pip_y + pip_h - 17, pip_x + pip_w, pip_y + pip_h],
                   fill=(12, 14, 20, 210))
    draw.text((pip_x + 6, pip_y + pip_h - 15),
              fit(draw, text, F09, pip_w - 12), font=F09,
              fill=GOOD if visible else BAD)
    return pip_x, pip_y + pip_h


def compose(main: np.ndarray, pip: np.ndarray, *, record: dict,
            total_seconds: float, summary: dict, trail=(),
            cross_waypoints=()) -> Image.Image:
    """Compose one output frame from the wide shot, the PiP and the record."""
    image = Image.fromarray(np.asarray(main)).convert("RGB")
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Title strip.  The subtitle is drawn only if it fits beside the title
    # without colliding, because the title is fixed and the width is not.
    panel(draw, (0, 0, width, 30), alpha=195, outline=None)
    draw.text((12, 7), TITLE, font=F13, fill=INK)
    subtitle_x = width - 12 - text_w(draw, SUBTITLE, F09)
    if subtitle_x > 24 + text_w(draw, TITLE, F13):
        draw.text((subtitle_x, 11), SUBTITLE, font=F09, fill=DIM)

    # PiP, top right.
    pip_h, pip_w = np.asarray(pip).shape[0], np.asarray(pip).shape[1]
    _, pip_bottom = _draw_pip(image, draw, pip, record,
                              (width - pip_w - 12, 38))

    # Left column: what it is doing, what it measured, how well it is holding.
    draw_status(draw, (12, 38, 316, 158), record)
    draw_side_risk(draw, (12, 164, 316, 330), record)
    draw_formation(draw, (12, 336, 316, 476), record)
    draw_legend(draw, (12, 482, 316, 522), record)

    # Right column under the PiP: the plan view.
    PlanView((width - 320, pip_bottom + 8, width - 12, height - 92)).draw(
        draw, record, trail=trail, cross_waypoints=cross_waypoints)
    # Timeline across the bottom, clear of the left column.
    Timeline((324, height - 86, width - 12, height - 12), total_seconds).draw(
        draw, record, summary)

    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
