#!/usr/bin/env python3
"""The three text HUD panels: queue order, candidate places, live status.

Each answers a question the metrics also answer numerically, so the video and
the JSON can be checked against each other:

* **the order panel** - who the duck thinks is in line, in what order, which one
  it has identified as the tail, and what the two naive readings would have
  said.  The naive answers are drawn STRUCK THROUGH, so "projection beats
  max-coordinate" is visible rather than asserted.
* **the gap scorecard** - every candidate standing place with its measured
  separation, whether the duck would physically FIT, and the verdict.  The two
  refusals the behavior turns on are the rows that say "fits" and "REJECT" at
  the same time, and they are boxed.
* **the status panel** - state, arc, cross-track, standoff, and whether the
  command is exactly zero.
"""

from __future__ import annotations

from hud_style import (
    ACCENT,
    BAD,
    DIM,
    F10,
    F11,
    F13,
    F16,
    GOOD,
    INK,
    STATE_COLORS,
    WARN,
    _panel,
    _text_w,
)
from queue_geometry import STANDOFF_MAX_M, STANDOFF_MIN_M


def _draw_order_panel(draw, box, record):
    x0, y0, x1, y1 = box
    _panel(draw, box)
    draw.text((x0 + 8, y0 + 5), "QUEUE ORDER", font=F11, fill=DIM)
    draw.text((x0 + 8, y0 + 18), "by arc length along the path", font=F10,
              fill=DIM)
    y = y0 + 34
    order = record["inferred_order"]
    for index, name in enumerate(order):
        tail = name == record["inferred_tail"]
        label = f"{index + 1}. {name}"
        draw.text((x0 + 10, y), label, font=F13, fill=BAD if tail else INK)
        cursor = x0 + 10 + _text_w(draw, label, F13) + 8
        if tail:
            draw.text((cursor, y + 2), "TRUE TAIL", font=F10, fill=BAD)
            cursor += _text_w(draw, "TRUE TAIL", F10) + 8
        if name == record["predecessor"]:
            draw.text((cursor, y + 2), "< AHEAD", font=F10, fill=GOOD)
        y += 17

    y += 4
    draw.text((x0 + 10, y), "naive readings would say:", font=F10, fill=DIM)
    y += 14
    for label, key in (("max range", "naive_range_tail"),
                       ("max -x", "naive_x_tail")):
        value = record.get(key)
        wrong = value != record["inferred_tail"]
        text = f"  {label}: tail = {value}"
        draw.text((x0 + 10, y), text, font=F10, fill=BAD if wrong else DIM)
        if wrong:
            width = _text_w(draw, text, F10)
            draw.line([(x0 + 10, y + 6), (x0 + 10 + width, y + 6)],
                      fill=BAD, width=1)
            draw.text((x0 + 14 + width, y), "WRONG", font=F10, fill=BAD)
        y += 13


def _draw_gap_panel(draw, box, record):
    x0, y0, x1, y1 = box
    _panel(draw, box)
    draw.text((x0 + 8, y0 + 5), "CANDIDATE PLACES", font=F11, fill=DIM)
    draw.text((x0 + 8, y0 + 18), "fits?  /  allowed?", font=F10, fill=DIM)
    y = y0 + 34
    # COLUMNS ARE LAID OUT FROM MEASURED TEXT WIDTH, right to left.  The first
    # draft drew the fits/verdict columns at fixed offsets and they overlapped
    # on every row - "fits" ran straight into "REJECT" - because the panel is
    # narrower than the sum of the widest cells.
    verdict_x = x1 - 10 - _text_w(draw, "REJECT", F10)
    fits_x = verdict_x - 8 - _text_w(draw, "tight", F10)
    sep_x = fits_x - 8 - _text_w(draw, "0.00m", F10)
    name_limit = sep_x - x0 - 16
    for gap in record["gaps"][:6]:
        accepted = gap["verdict"] == "join"
        fits = gap["physically_fits"]
        name = gap["gap"].replace("between_", "").replace("_", " ")
        while _text_w(draw, name, F10) > name_limit and len(name) > 4:
            name = name[:-2] + "\u2026" if not name.endswith("\u2026") \
                else name[:-2] + "\u2026"
        draw.text((x0 + 10, y), name, font=F10, fill=INK if accepted else DIM)
        draw.text((sep_x, y), f"{gap['separation_m']:.2f}m", font=F10, fill=DIM)
        draw.text((fits_x, y), "fits" if fits else "tight", font=F10,
                  fill=GOOD if fits else DIM)
        verdict = "JOIN" if accepted else "REJECT"
        draw.text((verdict_x, y), verdict, font=F10,
                  fill=GOOD if accepted else BAD)
        # The rows that matter: physically available AND refused.
        if fits and not accepted:
            draw.rectangle([x0 + 6, y - 2, x1 - 6, y + 12], outline=(112, 52, 44))
        y += 15


def _draw_status(draw, box, record):
    x0, y0, x1, y1 = box
    _panel(draw, box)
    state = record["state"]
    draw.text((x0 + 10, y0 + 6), state, font=F16,
              fill=STATE_COLORS.get(state, INK))
    standoff = record["standoff_m"]
    command = record["command"]
    zero = all(abs(v) == 0.0 for v in command)
    rows = [
        ("arc s", f"{record['duck_arc_m']:.3f} m"),
        ("cross-track", f"{record['duck_cross_track_m']:+.3f} m"),
        ("standoff",
         f"{standoff:.3f} m" if standoff is not None else "-"),
        ("ahead", record["predecessor"] or "-"),
        ("still ahead", str(record["predecessors_remaining"])),
        ("command",
         "EXACTLY ZERO" if zero
         else f"{command[0]:+.2f} {command[1]:+.2f} {command[2]:+.2f}"),
        ("trunk z", f"{record['trunk_z_m']:.4f} m"),
        ("nearest person", f"{record['nearest_clearance_m']:.3f} m"),
        ("nearest scenery", f"{record['scenery_clearance_m']:.3f} m"),
    ]
    y = y0 + 30
    for label, value in rows:
        draw.text((x0 + 10, y), label, font=F10, fill=DIM)
        color = INK
        if label == "command":
            color = GOOD if zero else ACCENT
        elif label == "standoff" and standoff is not None:
            color = GOOD if STANDOFF_MIN_M <= standoff <= STANDOFF_MAX_M else WARN
        draw.text((x1 - 10 - _text_w(draw, value, F10), y), value,
                  font=F10, fill=color)
        y += 14
