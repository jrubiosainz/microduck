#!/usr/bin/env python3
"""Compose one output frame: wide shot, PiP, and the HUD panels.

Layout only.  The panels themselves live in ``hud_panels`` and ``hud_views``,
and the palette in ``hud_style``; this module decides where each goes and draws
the PiP chrome.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from hud_panels import _draw_gap_panel, _draw_order_panel, _draw_status
from hud_style import BAD, DIM, F10, F13, GOOD, INK, _panel, _text_w
from hud_views import PlanView, Timeline


def compose(main: np.ndarray, pip: np.ndarray, *, record: dict,
            total_seconds: float, machine_summary: dict) -> Image.Image:
    """Compose one output frame: wide shot, PiP, and the HUD panels."""
    image = Image.fromarray(np.asarray(main)).convert("RGB")
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Title strip.
    _panel(draw, (0, 0, width, 30), alpha=190, outline=None)
    draw.text((12, 7), "microduck - QUEUE POLITELY", font=F13, fill=INK)
    subtitle = ("order inferred by projection onto the queue path, "
                "not by distance or coordinate")
    draw.text((width - 12 - _text_w(draw, subtitle, F10), 10), subtitle,
              font=F10, fill=DIM)

    # PiP, top right, beneath the title strip.
    pip_image = Image.fromarray(np.asarray(pip)).convert("RGB")
    pip_w, pip_h = pip_image.size
    pip_x, pip_y = width - pip_w - 12, 40
    image.paste(pip_image, (pip_x, pip_y))
    draw.rectangle([pip_x - 1, pip_y - 1, pip_x + pip_w, pip_y + pip_h],
                   outline=(96, 104, 124))
    label = "HEAD CAMERA (stabilized)"
    draw.rectangle([pip_x, pip_y, pip_x + pip_w, pip_y + 16],
                   fill=(12, 14, 20, 200))
    draw.text((pip_x + 6, pip_y + 2), label, font=F10, fill=DIM)
    subject = record["predecessor"]
    if subject:
        seen = record["subject_visible"]
        text = (f"watching {subject}  "
                f"{'VISIBLE' if seen else 'occluded'} "
                f"{record['subject_fraction'] * 100:3.0f}%")
        draw.rectangle([pip_x, pip_y + pip_h - 16, pip_x + pip_w, pip_y + pip_h],
                       fill=(12, 14, 20, 200))
        draw.text((pip_x + 6, pip_y + pip_h - 14), text, font=F10,
                  fill=GOOD if seen else BAD)

    # Left column: order, gaps, status.
    _draw_order_panel(draw, (12, 40, 300, 220), record)
    _draw_gap_panel(draw, (12, 226, 300, 348), record)
    _draw_status(draw, (12, 354, 300, 514), record)

    # Plan view under the PiP, timeline across the bottom.
    PlanView((width - 320, pip_y + pip_h + 8, width - 12, height - 74)).draw(
        draw, record)
    Timeline((12, height - 66, width - 12, height - 12), total_seconds).draw(
        draw, record, machine_summary)

    image = Image.alpha_composite(image.convert("RGBA"), layer)
    return image.convert("RGB")
