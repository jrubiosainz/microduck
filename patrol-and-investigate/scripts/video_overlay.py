#!/usr/bin/env python3
"""Compose one output frame: wide shot, head-camera PiP, and the HUD panels.

Layout only.  The panels live in ``hud_panels``, the spatial views in
``hud_views`` and the palette in ``hud_style``; this module decides where each
goes and draws the PiP chrome.

THE PiP CHROME CARRIES THE DISCLOSURES
---------------------------------------
The picture-in-picture is rendered from the EXACT camera every visibility
measurement is taken through - ``patrol_camera.PatrolCamera.camera_id``, at the
same 300x216 pixel geometry that sets its horizontal FOV - so what the viewer
sees and what the gate graded are the same frustum.  Three things about it must
never be implied away, so they are drawn into the PiP itself rather than left to
the README:

* it is **stabilized** - the rig sits exactly where the physical head camera
  sits, but holds a level horizon so a human can read it while the duck's trunk
  pitches through its gait;
* the identity read off it is a **semantic proxy** - MuJoCo body identity inside
  a real frustum with a real occlusion ray cast, not an RGB classifier;
* the staff are **scripted and never yield**, so nothing the duck avoided was
  avoiding it back.

The footer names what the head is tracking and whether that exact camera can see
it, which is the most useful caption in the video.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from hud_panels import (
    draw_legend,
    draw_memory,
    draw_patrol,
    draw_safety,
    draw_state,
    draw_target,
)
from hud_style import BAD, DIM, F09, F13, GOOD, INK, fit, panel, text_w
from hud_views import PlanView, Timeline

TITLE = "microduck - PATROL & INVESTIGATE"
SUBTITLE = ("walks a five-checkpoint circuit, stops and scans at each, breaks "
            "off to investigate two anomalies from a safe standoff, dismisses "
            "a benign one, and resumes exactly where it left off")


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
    scanning = record["state"] == "SCAN"
    caption = ("sweeping the facility" if scanning
               else f"watching {subject}")
    draw.text((x + 4, footer + 2), fit(draw, caption, F09, width - 70),
              font=F09, fill=INK)
    mark = "IN FRAME" if visible else "not in frame"
    draw.text((x + width - 4 - text_w(draw, mark, F09), footer + 2), mark,
              font=F09, fill=GOOD if visible else BAD)
    draw.text((x + 4, footer + 14),
              fit(draw, "identity: MuJoCo body id, not an RGB classifier", F09,
                  width - 8), font=F09, fill=DIM)
    return y + height


def compose(main, pip, *, record, total_seconds, summary, trail) -> Image.Image:
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

    # -- the right column: memory, target, safety ----------------------------
    # THE HEIGHTS ARE BUDGETED AGAINST THE TIMELINE, NOT CHOSEN.  The column
    # must end clear of the timeline panel, which starts at height - 56, or the
    # SAFETY panel's zone line - the number proving the rule was obeyed - would
    # be drawn underneath it and could not be read.
    column_x0, column_x1 = pip_x, width - 10
    y = pip_bottom + 8
    for panel_draw, panel_height in ((draw_memory, 84), (draw_target, 116),
                                     (draw_safety, 76)):
        panel_draw(draw, (column_x0, y, column_x1, y + panel_height), record)
        y += panel_height + 6

    # -- the left column: state, patrol, plan view, legend -------------------
    left_x0, left_x1 = 10, 10 + 268
    draw_state(draw, (left_x0, 44, left_x1, 44 + 110), record)
    draw_patrol(draw, (left_x0, 160, left_x1, 160 + 78), record)
    plan = PlanView((left_x0, 244, left_x1, 244 + 176))
    plan.draw(draw, record, trail)
    draw_legend(draw, (left_x0, 426, left_x1, 426 + 84), record)

    # -- the timeline, along the bottom --------------------------------------
    timeline = Timeline((10, height - 56, width - 10, height - 8),
                        total_seconds)
    timeline.draw(draw, record, summary)
    return image
