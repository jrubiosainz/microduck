#!/usr/bin/env python3
"""Compose one output frame: wide shot, head-camera PiP, and the HUD panels.

Layout only.  The panels live in ``hud_panels``, the spatial views in
``hud_views`` and the palette in ``hud_style``; this module decides where each
goes and draws the PiP chrome.

THE PiP CHROME CARRIES THE DISCLOSURES
---------------------------------------
The picture-in-picture is rendered from the EXACT camera every visibility
measurement is taken through - ``gest_camera.GestureCamera.camera_id``, at the
same 300x216 pixel geometry that sets its horizontal FOV - so what the viewer
sees and what the gate graded are the same frustum.  Three things about it must
never be implied away, so they are drawn into the PiP itself rather than left to
the README:

* it is **stabilized** - the rig sits exactly where the physical head camera
  sits, but holds a level horizon so a human can read it while the duck's trunk
  pitches through its gait;
* the identity read off it is a **semantic proxy** - MuJoCo body identity inside
  a real frustum with a real occlusion ray cast, not an RGB classifier, and the
  gesture is a rule set over measured arm geometry rather than learned
  perception;
* the people are **scripted and never react**, so nothing the duck did was
  answered by anybody.

The footer names who the head is tracking and whether that exact camera can read
their ARM, which is the harder of the two gates and the one the behavior turns
on.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from hud_panels import (
    action_panel,
    command_panel,
    reading_panel,
    refusals_panel,
    sequence_panel,
    who_panel,
)
from hud_style import BAD, DIM, F09, F13, GOOD, INK, fit, panel, text_w
from hud_views import PlanView, Timeline

TITLE = "microduck - GESTURE RESPONSE"
SUBTITLE = ("takes six hand commands from ONE instructor - come, stop, point "
            "left, point right, back up, goodbye - refuses an ambiguous "
            "gesture, and ignores four strangers making the same signs")


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
    locked = record.get("locked", "")
    readable = record.get("instructor_arm_readable", False)
    caption = (f"watching {locked}" if locked
               else "sweeping for the instructor")
    draw.text((x + 4, footer + 2), fit(draw, caption, F09, width - 84),
              font=F09, fill=INK)
    # THE ARM GATE is what the footer reports, not the body gate: a person can
    # be comfortably in frame with their raised hand outside it.
    mark = "ARM READABLE" if readable else "arm not readable"
    draw.text((x + width - 4 - text_w(draw, mark, F09), footer + 2), mark,
              font=F09, fill=GOOD if readable else BAD)
    draw.text((x + 4, footer + 14),
              fit(draw, "identity + gesture: semantic proxy, not RGB", F09,
                  width - 8), font=F09, fill=DIM)
    return y + height


def compose(main, pip, *, record, total_seconds, expected, refusals,
            interrupts, history, trail) -> Image.Image:
    """One finished frame.

    ``main`` and ``pip`` are both rendered from the camera's ISOLATED render
    data, in which the head has been posed for this tick - so the gaze the
    viewer sees is the gaze that was measured, and the walking physics never saw
    either.
    """
    image = Image.fromarray(np.asarray(main)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    thresholds = record.get("thresholds", {})

    # -- title -------------------------------------------------------------
    panel(draw, (0, 0, width, 40), alpha=190, outline=None)
    draw.text((12, 6), TITLE, font=F13, fill=INK)
    draw.text((12, 24), fit(draw, SUBTITLE, F09, width - 24), font=F09,
              fill=DIM)

    # -- the PiP, top right --------------------------------------------------
    pip_w = pip.shape[1]
    pip_x = width - pip_w - 10
    pip_bottom = _draw_pip(image, draw, pip, record, (pip_x, 44))

    # -- the right column: reading, action, refusals -------------------------
    # THE HEIGHTS ARE BUDGETED AGAINST THE TIMELINE, NOT CHOSEN, and the budget
    # is arithmetic rather than taste.  The PiP ends at 260; the timeline panel
    # starts at ``height - 70`` = 570.  Each panel below is tall enough for its
    # own worst-case content - the READING panel's two wrapped rule lines plus
    # its proxy label, the REFUSED panel's two entries at 26 px each - so
    # nothing is drawn under the timeline or clipped at a panel edge.  An
    # earlier budget lost the third refusal and the timeline's own legend.
    column_x0, column_x1 = pip_x, width - 10
    y = pip_bottom + 8
    for panel_draw, panel_height in (
            (lambda d, b: reading_panel(d, b, record, thresholds), 112),
            (lambda d, b: action_panel(d, b, record, thresholds), 82),
            (lambda d, b: refusals_panel(d, b, record, refusals), 80)):
        panel_draw(draw, (column_x0, y, column_x1, y + panel_height))
        y += panel_height + 6

    # -- the left column: who, command, sequence, plan ------------------------
    left_x0, left_x1 = 10, 10 + 268
    who_panel(draw, (left_x0, 44, left_x1, 44 + 104), record, thresholds)
    command_panel(draw, (left_x0, 154, left_x1, 154 + 78), record)
    sequence_panel(draw, (left_x0, 238, left_x1, 238 + 116), record, expected)
    plan = PlanView((left_x0, 360, left_x1, height - 84))
    plan.draw(draw, record, trail)

    # -- the timeline, along the bottom --------------------------------------
    timeline = Timeline((10, height - 70, width - 10, height - 8),
                        total_seconds)
    timeline.draw(draw, record, history, refusals, interrupts)
    return image
