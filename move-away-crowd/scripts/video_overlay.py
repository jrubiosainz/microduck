#!/usr/bin/env python3
"""HUD, state pipeline, threat panel, cycle timeline and attention-camera PiP.

Presentation only: nothing here is allowed to influence the rollout.  The
composed frame answers four questions a viewer should not have to infer —
which adult is the threat, what state the behavior is in, whether the duck is
still upright, and what the duck itself can see.

The PiP geometry is imported from ``attention_camera`` rather than redefined,
because that same rectangle sets the frustum every visibility percentage is
measured against.  Drawing one shape while measuring another would let the HUD
disagree with the picture.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from attention_camera import PIP_H, PIP_W

# Shirt colours, matching the materials in the generated scene.
ADULT_RGB = {
    "blue": (26, 82, 235),
    "green": (26, 179, 71),
    "red": (224, 31, 36),
    "yellow": (242, 184, 26),
    "purple": (158, 51, 224),
    "orange": (245, 115, 20),
    "teal": (15, 173, 173),
    "pink": (242, 107, 168),
}
STATE_RGB = {
    "SCANNING": (255, 210, 75),
    "THREAT_LOCK": (255, 105, 110),
    "EVADING": (90, 190, 255),
    "SETTLING": (180, 200, 255),
    "CLEAR": (120, 255, 170),
}
PIPELINE = ("SCANNING", "THREAT_LOCK", "EVADING", "SETTLING", "CLEAR")
HUD_H = 150
INK = (232, 240, 250)
DIM = (150, 166, 186)
PANEL = (4, 8, 14)


def _font(size=14, mono=True):
    for path in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _adult_color(name):
    return ADULT_RGB.get(name, (200, 200, 200))


def compose(main_rgb, pip_rgb, *, record, total_seconds, cycles,
            cycles_target, carries_box, contact_free):
    """Draw one finished frame from one rollout record.

    ``cycles`` is the list of encounters CLOSED so far; it drives both the
    completed counter and the timeline, so the two can never disagree.
    """
    image = Image.fromarray(main_rgb)
    draw = ImageDraw.Draw(image)
    font, small, tiny = _font(15), _font(12), _font(11)
    width, height = image.size
    cycles_done = len(cycles)

    state = record["state"]
    locked = record["locked"]
    state_color = STATE_RGB[state]
    accent = _adult_color(locked) if locked else state_color

    # ---- header ------------------------------------------------------
    draw.rectangle([0, 0, width, HUD_H], fill=PANEL)
    draw.text((12, 8), f"MOVE-AWAY · CROWD    t={record['t']:05.2f}s",
              fill=accent, font=font)
    draw.text((12, 30),
              f"STATE {state:<11s} ({record['state_elapsed_s']:4.1f}s)   "
              f"CYCLE {record['cycle']}   COMPLETED {cycles_done}/{cycles_target}",
              fill=state_color, font=small)

    if locked:
        box = "carrying a box" if carries_box else "hands free"
        draw.text((12, 50),
                  f"THREAT: {locked.upper():<7s} ({box})   "
                  f"range={record['locked_range_m']:.2f} m   "
                  f"{'IN VIEW' if record['locked_visible'] else 'occluded'}",
                  fill=_adult_color(locked), font=small)
    else:
        seen = ", ".join(record["visible"]) or "nobody"
        draw.text((12, 50), f"NO THREAT LOCKED   camera sees: {seen}",
                  fill=DIM, font=small)

    if record["threat"] is not None:
        draw.text((12, 70),
                  f"predicted closest approach {record['threat_clearance_m']:.2f} m "
                  f"in {record['threat_ttc_s']:.1f} s   "
                  f"bearing {record['threat_bearing_deg']:+6.1f}°",
                  fill=INK, font=small)
    else:
        draw.text((12, 70), "predicted closest approach: everyone will miss",
                  fill=DIM, font=small)

    command = record["command"]
    moving = any(abs(v) > 1e-9 for v in command)
    draw.text((12, 90),
              f"command vx={command[0]:+.3f} vy={command[1]:+.3f} "
              f"wz={command[2]:+.3f}  {'WALKING' if moving else 'STOPPED (exactly zero)'}",
              fill=(120, 220, 255) if moving else (150, 230, 170), font=small)
    draw.text((12, 110),
              f"escape heading error {record['heading_error_deg']:+6.1f}°   "
              f"nearest adult {record['nearest_adult']:<7s} "
              f"clearance {record['nearest_clearance_m']:+.3f} m "
              f"{'' if contact_free else '  ** CONTACT **'}",
              fill=INK if contact_free else (255, 90, 90), font=small)
    upright = record["trunk_z_m"] >= 0.09
    draw.text((12, 130),
              f"trunk z={record['trunk_z_m']:.3f} m   "
              f"min={record['min_trunk_z_m']:.3f} m   "
              f"{'UPRIGHT' if upright else 'FALLEN'}",
              fill=(145, 255, 165) if upright else (255, 90, 90), font=small)

    # ---- attention camera PiP ---------------------------------------
    px0, py0 = width - PIP_W - 12, HUD_H + 12
    border = accent if (locked and record["locked_visible"]) else (255, 190, 60)
    draw.rectangle([px0 - 4, py0 - 24, px0 + PIP_W + 4, py0 + PIP_H + 4],
                   fill=(2, 5, 8))
    label = f"TRACKING {locked.upper()}" if locked else "SCANNING PLAZA"
    draw.text((px0, py0 - 21), f"DUCK CAMERA · {label}", fill=border, font=small)
    image.paste(Image.fromarray(pip_rgb), (px0, py0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([px0 - 1, py0 - 1, px0 + PIP_W, py0 + PIP_H],
                   outline=border, width=3)
    cx, cy = px0 + PIP_W // 2, py0 + PIP_H // 2
    draw.line([cx - 10, cy, cx + 10, cy], fill=(255, 245, 95), width=1)
    draw.line([cx, cy - 10, cx, cy + 10], fill=(255, 245, 95), width=1)
    draw.text((px0 + 6, py0 + PIP_H - 18),
              "stabilized · world-up", fill=(190, 205, 225), font=tiny)
    if locked and record["locked_visible"]:
        draw.text((px0 + 8, py0 + 8), f"{locked.upper()} LOCKED",
                  fill=_adult_color(locked), font=small)

    # ---- who the camera can see -------------------------------------
    seen = set(record["visible"])
    legend_y = HUD_H + 14
    draw.text((14, legend_y - 14), "CAMERA SEES", fill=DIM, font=tiny)
    for index, name in enumerate(ADULT_RGB):
        y = legend_y + index * 19
        color = _adult_color(name)
        visible = name in seen
        draw.rectangle([14, y, 27, y + 13],
                       fill=color if visible else (30, 38, 50),
                       outline=color, width=1)
        text = name
        if name == locked:
            text = f"{name} ◄ threat"
        draw.text((33, y), text, fill=color if visible else (86, 98, 116),
                  font=tiny)

    # ---- state pipeline ---------------------------------------------
    pipe_y = height - 84
    slot = (width - 32) // len(PIPELINE)
    for index, name in enumerate(PIPELINE):
        x = 16 + index * slot
        active = name == state
        fill = STATE_RGB[name] if active else (30, 38, 50)
        draw.rounded_rectangle([x, pipe_y, x + slot - 14, pipe_y + 28],
                               radius=7, fill=fill, outline=STATE_RGB[name],
                               width=2)
        draw.text((x + 8, pipe_y + 7), name,
                  fill=(5, 10, 16) if active else STATE_RGB[name], font=tiny)
        if index < len(PIPELINE) - 1:
            draw.text((x + slot - 11, pipe_y + 7), "→", fill=DIM, font=small)

    # ---- cycle timeline ---------------------------------------------
    # One coloured block per COMPLETED encounter, positioned at the moment the
    # evasion actually happened, so the viewer can see the behavior repeat.
    bar_y = height - 42
    bar_x0, bar_x1 = 16, width - 16
    draw.rounded_rectangle([bar_x0, bar_y, bar_x1, bar_y + 20], radius=6,
                           fill=(18, 24, 34), outline=(52, 64, 82), width=1)
    span = max(bar_x1 - bar_x0, 1)
    for cycle in cycles:
        start = cycle.get("evade_start_s", cycle["lock_s"])
        end = cycle.get("evade_end_s", start)
        x0 = bar_x0 + int(span * min(start / total_seconds, 1.0))
        x1 = bar_x0 + int(span * min(end / total_seconds, 1.0))
        color = _adult_color(cycle["threat"])
        draw.rounded_rectangle([x0, bar_y + 3, max(x1, x0 + 9), bar_y + 17],
                               radius=4, fill=color)
        draw.text((x0 + 3, bar_y + 4), cycle["threat"][:2].upper(),
                  fill=(6, 10, 16), font=tiny)
    head = bar_x0 + int(span * min(record["t"] / total_seconds, 1.0))
    draw.line([head, bar_y - 3, head, bar_y + 23], fill=(255, 255, 255), width=2)
    return image
