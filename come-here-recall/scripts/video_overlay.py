#!/usr/bin/env python3
"""HUD, state pipeline, caller panel, recall timeline and attention-camera PiP.

Presentation only: nothing here is allowed to influence the rollout.  The
composed frame answers four questions a viewer should not have to infer —
who is calling, whether the duck has found them, how far it still has to walk,
and what the duck itself can see.

The PiP geometry is imported from ``attention_camera`` rather than redefined,
because that same rectangle sets the frustum every visibility percentage is
measured against.  Drawing one shape while measuring another would let the HUD
disagree with the picture.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from attention_camera import PIP_H, PIP_W
from recall_model import ACQUIRE_CONE_DEG, STANDOFF_MAX, STANDOFF_MIN

# Shirt colours, matching the materials in the generated scene.
ADULT_RGB = {
    "blue": (26, 82, 240),
    "green": (26, 184, 71),
    "red": (230, 31, 36),
    "yellow": (245, 189, 26),
    "purple": (163, 51, 230),
}
STATE_RGB = {
    "LISTEN": (255, 214, 80),
    "SEARCH": (255, 150, 60),
    "CALLER_LOCK": (255, 105, 110),
    "APPROACH": (90, 190, 255),
    "ARRIVED": (120, 255, 170),
}
PIPELINE = ("LISTEN", "SEARCH", "CALLER_LOCK", "APPROACH", "ARRIVED")
HUD_H = 150
INK = (232, 240, 250)
DIM = (150, 166, 186)
PANEL = (4, 8, 14)


def _font(size=14):
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


def compose(main_rgb, pip_rgb, *, record, total_seconds, cycles, cycles_target,
            calls, expected_order):
    """Draw one finished frame from one rollout record.

    ``cycles`` is the list of recalls CLOSED so far; it drives both the
    completed counter and the timeline, so the two can never disagree.
    """
    image = Image.fromarray(main_rgb)
    draw = ImageDraw.Draw(image)
    font, small, tiny = _font(15), _font(12), _font(11)
    width, height = image.size
    done = len(cycles)

    state = record["state"]
    caller = record["caller"]
    locked = record["locked"]
    state_color = STATE_RGB[state]
    accent = _adult_color(locked or caller) if (locked or caller) else state_color

    # ---- header ------------------------------------------------------
    draw.rectangle([0, 0, width, HUD_H], fill=PANEL)
    draw.text((12, 8), f"COME HERE · RECALL    t={record['t']:05.2f}s",
              fill=accent, font=font)
    draw.text((12, 30),
              f"STATE {state:<12s} ({record['state_elapsed_s']:4.1f}s)   "
              f"RECALL {record['cycle']}   COMPLETED {done}/{cycles_target}",
              fill=state_color, font=small)

    if caller:
        who = "LOCKED" if locked else "calling — not found yet"
        draw.text((12, 50),
                  f"CALLER: {caller.upper():<7s} ({who})   "
                  f"range={record['caller_range_m']:.2f} m   "
                  f"{'IN VIEW' if record['caller_visible'] else 'NOT VISIBLE'}",
                  fill=_adult_color(caller), font=small)
    else:
        seen = ", ".join(record["visible"]) or "nobody"
        draw.text((12, 50), f"NOBODY IS CALLING   camera sees: {seen}",
                  fill=DIM, font=small)

    if caller:
        gate = record["gate_open"]
        draw.text((12, 70),
                  f"acquisition gate {'OPEN' if gate else 'closed'}   "
                  f"off-axis {record['subject_off_axis_deg']:5.1f}° "
                  f"(needs ≤{ACQUIRE_CONE_DEG:.0f}°)   "
                  f"heading error {record['heading_error_deg']:+6.1f}°",
                  fill=(145, 255, 165) if gate else INK, font=small)
    else:
        draw.text((12, 70), "acquisition gate closed — listening",
                  fill=DIM, font=small)

    command = record["command"]
    moving = any(abs(v) > 1e-9 for v in command)
    draw.text((12, 90),
              f"command vx={command[0]:+.3f} vy={command[1]:+.3f} "
              f"wz={command[2]:+.3f}  "
              f"{'WALKING' if moving else 'STOPPED (exactly zero)'}",
              fill=(120, 220, 255) if moving else (150, 230, 170), font=small)
    draw.text((12, 110),
              f"standoff band {STANDOFF_MIN:.2f}–{STANDOFF_MAX:.2f} m   "
              f"nearest adult {record['nearest_adult']:<7s} "
              f"clearance {record['nearest_clearance_m']:+.3f} m   "
              f"refused calls {record['refused_count']}",
              fill=INK, font=small)
    upright = record["trunk_z_m"] >= 0.09
    draw.text((12, 130),
              f"trunk z={record['trunk_z_m']:.3f} m   "
              f"min={record['min_trunk_z_m']:.3f} m   "
              f"{'UPRIGHT' if upright else 'FALLEN'}",
              fill=(145, 255, 165) if upright else (255, 90, 90), font=small)

    # ---- attention camera PiP ---------------------------------------
    px0, py0 = width - PIP_W - 12, HUD_H + 12
    border = accent if (caller and record["caller_visible"]) else (255, 190, 60)
    draw.rectangle([px0 - 4, py0 - 24, px0 + PIP_W + 4, py0 + PIP_H + 4],
                   fill=(2, 5, 8))
    if locked:
        label = f"TRACKING {locked.upper()}"
    elif caller:
        label = f"SEARCHING FOR {caller.upper()}"
    else:
        label = "IDLE SCAN"
    draw.text((px0, py0 - 21), f"DUCK CAMERA · {label}", fill=border, font=small)
    image.paste(Image.fromarray(pip_rgb), (px0, py0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([px0 - 1, py0 - 1, px0 + PIP_W, py0 + PIP_H],
                   outline=border, width=3)
    cx, cy = px0 + PIP_W // 2, py0 + PIP_H // 2
    # Acquisition cone drawn to scale against the camera's real vertical FOV,
    # so the circle a viewer sees IS the gate the lock is tested against.
    cone_px = int(PIP_H * (ACQUIRE_CONE_DEG / 29.0) * 0.5)
    draw.ellipse([cx - cone_px, cy - cone_px, cx + cone_px, cy + cone_px],
                 outline=(255, 245, 95) if record["gate_open"] else (110, 125, 145),
                 width=2)
    draw.line([cx - 10, cy, cx + 10, cy], fill=(255, 245, 95), width=1)
    draw.line([cx, cy - 10, cx, cy + 10], fill=(255, 245, 95), width=1)
    draw.text((px0 + 6, py0 + PIP_H - 18),
              f"stabilized · world-up · gate {ACQUIRE_CONE_DEG:.0f}°",
              fill=(190, 205, 225), font=tiny)
    if locked and record["caller_visible"]:
        draw.text((px0 + 8, py0 + 8), f"{locked.upper()} LOCKED",
                  fill=_adult_color(locked), font=small)

    # ---- who the camera can see -------------------------------------
    # Drawn on its own dark panel.  The first preview drew this text straight
    # onto the plaza floor, where the bright checker texture made every label
    # unreadable at 960x640.
    seen = set(record["visible"])
    legend_y = HUD_H + 14
    panel_h = len(ADULT_RGB) * 19 + len(expected_order) * 18 + 56
    draw.rectangle([8, legend_y - 20, 152, legend_y + panel_h], fill=PANEL)
    draw.text((14, legend_y - 16), "CAMERA SEES", fill=DIM, font=tiny)
    for index, name in enumerate(ADULT_RGB):
        y = legend_y + index * 19
        color = _adult_color(name)
        visible = name in seen
        draw.rectangle([14, y, 27, y + 13],
                       fill=color if visible else (30, 38, 50),
                       outline=color, width=1)
        text = name
        if name == locked:
            text = f"{name} ◄lock"
        elif name == caller:
            text = f"{name} ◄call"
        draw.text((33, y), text, fill=color if visible else (86, 98, 116),
                  font=tiny)

    # ---- requested call order ---------------------------------------
    order_y = legend_y + len(ADULT_RGB) * 19 + 12
    draw.text((14, order_y), "CALL ORDER", fill=DIM, font=tiny)
    for index, name in enumerate(expected_order):
        y = order_y + 16 + index * 18
        complete = index < done
        color = _adult_color(name)
        mark = "✓" if complete else ("»" if index == done else "·")
        draw.text((14, y), f"{mark} {index + 1}. {name}",
                  fill=color if complete or index == done else (86, 98, 116),
                  font=tiny)

    # ---- state pipeline ---------------------------------------------
    # The whole lower strip sits on its own dark band.  Without it the caption
    # and the pipeline row overlapped both the floor and each other.
    draw.rectangle([0, height - 104, width, height], fill=PANEL)
    pipe_y = height - 96
    slot = (width - 32) // len(PIPELINE)
    for index, name in enumerate(PIPELINE):
        x = 16 + index * slot
        active = name == state
        fill = STATE_RGB[name] if active else (30, 38, 50)
        draw.rounded_rectangle([x, pipe_y, x + slot - 14, pipe_y + 28],
                               radius=7, fill=fill, outline=STATE_RGB[name],
                               width=2)
        draw.text((x + 6, pipe_y + 8), name,
                  fill=(5, 10, 16) if active else STATE_RGB[name], font=tiny)
        if index < len(PIPELINE) - 1:
            draw.text((x + slot - 11, pipe_y + 7), "→", fill=DIM, font=small)

    # ---- call / recall timeline -------------------------------------
    # Calls are drawn as hollow bars where they SOUND; completed approaches are
    # drawn solid, so a viewer can see the refused blue call never turned into
    # an approach.
    draw.text((16, pipe_y + 31), "calls (hollow) · completed recalls (solid)",
              fill=DIM, font=tiny)
    bar_y = height - 28
    bar_x0, bar_x1 = 16, width - 16
    draw.rounded_rectangle([bar_x0, bar_y, bar_x1, bar_y + 20], radius=6,
                           fill=(18, 24, 34), outline=(52, 64, 82), width=1)
    span = max(bar_x1 - bar_x0, 1)

    def _x(seconds):
        return bar_x0 + int(span * min(max(seconds, 0.0) / total_seconds, 1.0))

    for call in calls:
        x0, x1 = _x(call.start_s), _x(call.end_s)
        color = _adult_color(call.caller)
        draw.rounded_rectangle([x0, bar_y + 2, max(x1, x0 + 6), bar_y + 8],
                               radius=3,
                               outline=color if call.expected else (255, 90, 90),
                               width=1)
    for cycle in cycles:
        start = cycle.get("approach_start_s", cycle.get("lock_s", 0.0))
        end = cycle.get("arrived_end_s", cycle.get("approach_end_s", start))
        x0, x1 = _x(start), _x(end)
        color = _adult_color(cycle["caller"])
        draw.rounded_rectangle([x0, bar_y + 10, max(x1, x0 + 9), bar_y + 18],
                               radius=3, fill=color)
        draw.text((x0 + 3, bar_y + 9), cycle["caller"][:2].upper(),
                  fill=(6, 10, 16), font=tiny)
    head = _x(record["t"])
    draw.line([head, bar_y - 3, head, bar_y + 23], fill=(255, 255, 255), width=2)
    return image
