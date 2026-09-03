#!/usr/bin/env python3
"""Compose one output frame: the plaza shot, the head-camera PiP, the HUD.

Layout only.  Panels live in :mod:`pps_hud_panels`, the spatial views in
:mod:`pps_hud_views`, the palette in :mod:`pps_hud_style`; this module decides
where each goes and draws the PiP chrome.

THE PiP CHROME CARRIES THE DISCLOSURES
---------------------------------------
The picture-in-picture is rendered from ``PpsCamera.camera_id`` - the model
camera literally named ``pps_camera`` - at the same 300x216 geometry that sets
its horizontal FOV, which is the EXACT frustum every visibility number in the
metrics was measured through.  What the viewer sees and what the gates graded
are therefore the same optics.  Three things about that must never be implied
away, so they are drawn into the PiP itself rather than left to the README:

* the rig is **rendering-only and stabilized** - it sits exactly where the
  physical head camera sits, in an isolated ``MjData`` copy, and holds a level
  horizon so a human can read it while the trunk pitches through its gait.  The
  walking physics never sees it;
* identity is a **semantic proxy** - MuJoCo body id, not an RGB classifier -
  while the frustum containment and the occlusion ray cast behind every
  visibility number are real;
* the people are **scripted and never react**, so no clearance on screen is
  somebody being polite.

The footer names who the head is tracking and whether that exact camera can see
them, which is the gate the visibility numbers are built from.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from hud_style import F09, F13, fit, panel, text_w
from pps_hud_panels import (command_panel, progress_panel, safety_panel,
                            state_panel, station_panel, threats_panel)
from pps_hud_style import BAD, DIM, GOOD, INK, WARD, state_caption, state_color
from pps_hud_views import PlanView, Timeline

TITLE = "microduck - PROTECTIVE PERSONAL SPACE"
SUBTITLE = ("escorts ONE person across a plaza of seven strangers, predicts "
            "four intrusions from alternating bearings and interposes, "
            "dismisses a false near-pass, escapes a two-person squeeze, and "
            "RETREATS when the person it protects walks at it")


def _draw_pip(image, draw, pip, view, origin) -> int:
    """Paste the PiP and draw its chrome.  Returns its bottom edge."""
    x, y = origin
    height, width = pip.shape[0], pip.shape[1]
    image.paste(Image.fromarray(pip), (x, y))
    draw.rectangle([x - 1, y - 1, x + width, y + height], outline=(70, 78, 94))

    label = "HEAD CAMERA 'pps_camera' - rendering-only stabilized rig"
    draw.rectangle([x, y, x + width, y + 14], fill=(16, 18, 24))
    draw.text((x + 4, y + 2), fit(draw, label, F09, width - 8), font=F09,
              fill=DIM)

    footer = y + height - 28
    draw.rectangle([x, footer, x + width, y + height], fill=(16, 18, 24))
    subject = view["camera_subject"]
    visible = view["subject_visible"]
    draw.text((x + 4, footer + 2),
              fit(draw, f"tracking {subject}", F09, width - 66), font=F09,
              fill=WARD if subject == view["ward"] else INK)
    mark = "IN FRAME" if visible else "not in frame"
    draw.text((x + width - 4 - text_w(draw, mark, F09), footer + 2), mark,
              font=F09, fill=GOOD if visible else BAD)
    draw.text((x + 4, footer + 14), fit(
        draw, "identity: body-id proxy; frustum + occlusion are real", F09,
        width - 8), font=F09, fill=DIM)
    return y + height


def _caption_strip(draw, box, view) -> None:
    """One sentence of plain English about what is happening right now."""
    panel(draw, box, alpha=210)
    state = view["state"]
    draw.text((box[0] + 10, box[1] + 5), f"{view['t']:6.2f}s", font=F09,
              fill=DIM)
    draw.text((box[0] + 66, box[1] + 4), state, font=F09,
              fill=state_color(state))
    draw.text((box[0] + 190, box[1] + 4), fit(
        draw, state_caption(state), F09, box[2] - box[0] - 200), font=F09,
        fill=INK)


def compose(main, pip, *, view, total_seconds, history, episodes, trail
            ) -> Image.Image:
    """One finished frame.

    ``main`` and ``pip`` are both rendered from the camera's ISOLATED render
    data, in which the head has been posed for this tick - so the gaze the
    viewer sees is the gaze that was measured, and the walking physics saw
    neither.
    """
    image = Image.fromarray(np.asarray(main)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    # -- title --------------------------------------------------------------
    panel(draw, (0, 0, width, 42), alpha=190, outline=None)
    draw.text((12, 5), TITLE, font=F13, fill=INK)
    draw.text((12, 23), fit(draw, SUBTITLE, F09, width - 24), font=F09,
              fill=DIM)

    # -- the PiP, top right --------------------------------------------------
    pip_w = pip.shape[1]
    pip_x = width - pip_w - 10
    pip_bottom = _draw_pip(image, draw, pip, view, (pip_x, 46))

    # -- the right column: threats, station, episodes ------------------------
    # THE HEIGHTS ARE BUDGETED AGAINST THE TIMELINE, NOT CHOSEN.  The PiP ends
    # at 262; the timeline panel starts at height - 78 = 562.  Each panel below
    # is tall enough for its own worst-case content - THREATS at five predicted
    # rows plus its three summary lines, STATION at its two tolerance labels,
    # EPISODES at six ordered entries - so nothing is drawn under the timeline
    # or clipped at a panel edge.
    column_x0, column_x1 = pip_x, width - 10
    y = pip_bottom + 8
    for panel_draw, panel_height in (
            (threats_panel, 152),
            (station_panel, 120),
            (progress_panel, 108)):
        panel_draw(draw, (column_x0, y, column_x1, y + panel_height), view)
        y += panel_height + 6

    # -- the left column: state, command, clearance, plan --------------------
    left_x0, left_x1 = 10, 10 + 268
    state_panel(draw, (left_x0, 46, left_x1, 46 + 92), view)
    command_panel(draw, (left_x0, 144, left_x1, 144 + 82), view)
    safety_panel(draw, (left_x0, 232, left_x1, 232 + 94), view)
    PlanView((left_x0, 332, left_x1, height - 92)).draw(draw, view, trail)

    # -- the caption strip and the timeline, along the bottom ----------------
    _caption_strip(draw, (10, height - 88, width - 10, height - 70), view)
    Timeline((10, height - 66, width - 10, height - 8), total_seconds).draw(
        draw, view, history, episodes)
    return image
