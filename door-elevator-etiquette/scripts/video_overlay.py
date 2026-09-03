#!/usr/bin/env python3
"""Compose one output frame: wide shot, head-camera PiP, and the HUD panels.

Layout only.  The panels live in ``hud_panels``, the spatial views in
``hud_views`` and the palette in ``hud_style``; this module decides where each
goes and draws the PiP chrome.

THE PiP CHROME CARRIES THE DISCLOSURES
---------------------------------------
The picture-in-picture is rendered from the EXACT camera every visibility
measurement is taken through - ``etiquette_camera.EtiquetteCamera.camera_id``,
at the same 300x216 pixel geometry that sets its horizontal FOV - so what the
viewer sees and what the gate graded are the same frustum.  Three things about
it must never be implied away, so they are drawn into the PiP itself rather than
left to the README:

* it is **stabilized** - the rig sits exactly where the physical head camera
  sits, but holds a level horizon so a human can read it while the duck's trunk
  pitches through its gait;
* the identity read off it is a **semantic proxy** - MuJoCo body identity inside
  a real frustum with a real occlusion ray cast, not an RGB classifier;
* the doors are **scripted kinematic proxies**, not something the robot operated.

The footer names the person the head is tracking and whether that exact camera
can see them, which is the most useful caption in the video.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from hud_panels import (
    draw_doors,
    draw_legend,
    draw_order,
    draw_safety,
    draw_state,
    draw_traffic,
)
from hud_style import BAD, DIM, F09, F13, GOOD, INK, fit, panel, text_w
from hud_views import PlanView, Timeline

TITLE = "microduck - DOOR & ELEVATOR ETIQUETTE"
SUBTITLE = ("stops outside a doorway while people come out, waits beside a "
            "lift, lets everybody off, and follows its guardian in and out")


def _draw_pip(image, draw, pip, record, origin) -> int:
    """Paste the PiP and draw its chrome.  Returns its bottom edge."""
    x, y = origin
    height, width = pip.shape[0], pip.shape[1]
    image.paste(Image.fromarray(pip), (x, y))
    draw.rectangle([x - 1, y - 1, x + width, y + height], outline=(70, 78, 94))

    label = "HEAD CAMERA - stabilized rig at the head-camera position"
    draw.rectangle([x, y, x + width, y + 14], fill=(16, 18, 24))
    draw.text((x + 4, y + 2), fit(draw, label, F09, width - 8), font=F09,
              fill=DIM)

    footer = y + height - 28
    draw.rectangle([x, footer, x + width, y + height], fill=(16, 18, 24))
    subject = record["subject"]
    visible = record["subject_visible"]
    draw.text((x + 4, footer + 2), f"watching {subject}", font=F09, fill=INK)
    mark = "IN FRAME" if visible else "not in frame"
    draw.text((x + width - 4 - text_w(draw, mark, F09), footer + 2), mark,
              font=F09, fill=GOOD if visible else BAD)
    draw.text((x + 4, footer + 14),
              fit(draw, "identity: MuJoCo body id, not an RGB classifier", F09,
                  width - 8), font=F09, fill=DIM)
    return y + height


def compose(main, pip, *, record, total_seconds, summary, trail,
            route_points) -> Image.Image:
    """One finished frame.

    ``main`` and ``pip`` are both rendered from the camera's ISOLATED render
    data, in which the head has been posed for this tick - so the gaze the
    viewer sees is the gaze that was measured, and the walking physics never saw
    either.
    """
    image = Image.fromarray(np.asarray(main)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    # -- title -------------------------------------------------------------
    panel(draw, (0, 0, width, 40), alpha=190, outline=None)
    draw.text((12, 6), TITLE, font=F13, fill=INK)
    draw.text((12, 24), fit(draw, SUBTITLE, F09, width - 24), font=F09,
              fill=DIM)

    # -- the PiP, top right --------------------------------------------------
    pip_w = pip.shape[1]
    pip_x = width - pip_w - 10
    pip_bottom = _draw_pip(image, draw, pip, record, (pip_x, 44))

    # -- the right column: doors, traffic, order, safety ---------------------
    # THE HEIGHTS ARE BUDGETED AGAINST THE TIMELINE, NOT CHOSEN.
    # The first preview ran the column to y=616 while the timeline panel starts
    # at y=584, so the SAFETY panel's scenery-clearance line - the one number
    # proving the duck never touched a wall - was drawn underneath it and could
    # not be read.  The column now ends at 580, four pixels clear.
    column_x0, column_x1 = pip_x, width - 10
    y = pip_bottom + 8
    for panel_draw, panel_height in ((draw_doors, 84), (draw_traffic, 84),
                                     (draw_order, 66), (draw_safety, 60)):
        panel_draw(draw, (column_x0, y, column_x1, y + panel_height), record)
        y += panel_height + 6

    # -- the left column: state, plan view, legend ---------------------------
    left_x0, left_x1 = 10, 10 + 268
    draw_state(draw, (left_x0, 44, left_x1, 44 + 96), record)
    plan = PlanView((left_x0, 146, left_x1, 146 + 196))
    plan.draw(draw, record, trail, route_points)
    draw_legend(draw, (left_x0, 348, left_x1, 348 + 112), record)

    # -- the timeline, along the bottom --------------------------------------
    timeline = Timeline((10, height - 56, width - 10, height - 8),
                        total_seconds)
    timeline.draw(draw, record, summary)
    return image
