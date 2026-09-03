#!/usr/bin/env python3
"""HUD, state pipeline, conflict timeline, road map and head-camera PiP.

Presentation only: nothing here is allowed to influence the rollout.  The
composed frame answers five questions a viewer should not have to infer —
where the duck is on the street, which road users matter right now, how long
until each of them reaches the crossing, why the duck is or is not moving, and
what the duck itself can see.

The PiP geometry is imported from ``guardian_camera`` rather than redefined,
because that same rectangle sets the frustum every sector-visibility percentage
is measured against.  Drawing one shape while measuring another would let the
HUD disagree with the picture.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_scene import VEHICLES as BUILD_VEHICLES  # noqa: E402
from conflict import SAFETY_MARGIN_S  # noqa: E402
from guardian_camera import PIP_H, PIP_W  # noqa: E402
from street import (  # noqa: E402
    CROSSWALK_HALF_SPAN,
    LANE_SIDE,
    ROAD_HALF_WIDTH,
    SAFE_ZONE_SPAN,
    WAIT_LINE_X,
)
from traffic import VEHICLE_NAMES  # noqa: E402

# Vehicle colours, DERIVED from the scene generator rather than restated.
#
# The first draft hard-coded this dict, and adding a seventh vehicle to the
# schedule crashed the render with ``KeyError: 'courier'`` after the whole
# rollout had already run.  A presentation table that has to be kept in sync by
# hand will eventually disagree with the scene it is drawing.
def _vehicle_palette() -> dict[str, tuple[int, int, int]]:
    palette: dict[str, tuple[int, int, int]] = {}
    for name, _kind, body_rgba, _trim in BUILD_VEHICLES:
        red, green, blue = (float(v) for v in body_rgba.split()[:3])
        palette[name] = (int(red * 255), int(green * 255), int(blue * 255))
    return palette


VEHICLE_RGB = _vehicle_palette()
STATE_RGB = {
    "APPROACH_CURB": (120, 200, 255),
    "STOP": (255, 120, 110),
    "LOOK_LEFT": (255, 176, 60),
    "LOOK_RIGHT": (110, 190, 255),
    "LOOK_LEFT_AGAIN": (255, 176, 60),
    "WAIT_FOR_GAP": (255, 224, 90),
    "CROSSING": (120, 255, 170),
    "SAFE": (110, 255, 130),
}
PIPELINE = (
    "APPROACH_CURB", "STOP", "LOOK_LEFT", "LOOK_RIGHT", "LOOK_LEFT_AGAIN",
    "WAIT_FOR_GAP", "CROSSING", "SAFE",
)
SHORT = {
    "APPROACH_CURB": "APPROACH", "STOP": "STOP", "LOOK_LEFT": "LOOK ◀",
    "LOOK_RIGHT": "LOOK ▶", "LOOK_LEFT_AGAIN": "LOOK ◀◀",
    "WAIT_FOR_GAP": "WAIT GAP", "CROSSING": "CROSSING", "SAFE": "SAFE",
}
HUD_H = 150
INK = (232, 240, 250)
DIM = (150, 166, 186)
PANEL = (4, 8, 14)
GOOD = (145, 255, 165)
BAD = (255, 96, 96)
WARN = (255, 205, 90)


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


def _road_map(draw, record, x0, y0, width, height, small, tiny):
    """Plan view of the street: lanes, crossing, duck and every road user.

    This is the panel that makes the decision legible.  The duck's position is
    drawn against the same lane edges the occupancy gates use, so a viewer can
    see for themselves that it stopped short of the paint and that it never
    stood still between them.
    """
    # y in the world maps to the panel's HORIZONTAL axis, x to the vertical:
    # the camera looks along the road, so this orientation matches the wide
    # shot rather than transposing it.
    span_y = 9.0
    draw.rectangle([x0, y0, x0 + width, y0 + height], fill=(10, 14, 20),
                   outline=(52, 64, 82))

    def px(world_y):
        return x0 + width * (0.5 + world_y / (2.0 * span_y))

    def py(world_x):
        # x from -2.6 (near pavement) to +2.6 (far pavement)
        return y0 + height * (0.5 - world_x / 5.2)

    # road surface
    draw.rectangle([x0, py(ROAD_HALF_WIDTH), x0 + width, py(-ROAD_HALF_WIDTH)],
                   fill=(26, 28, 33))
    draw.line([x0, py(0.0), x0 + width, py(0.0)], fill=(120, 118, 70), width=1)
    # zebra
    for index in range(7):
        bar_y = -0.51 + index * 0.170
        draw.rectangle(
            [px(bar_y - 0.045), py(ROAD_HALF_WIDTH),
             px(bar_y + 0.045), py(-ROAD_HALF_WIDTH)],
            fill=(150, 150, 148))
    # wait line and safe zone
    draw.line([px(-CROSSWALK_HALF_SPAN), py(-WAIT_LINE_X),
               px(CROSSWALK_HALF_SPAN), py(-WAIT_LINE_X)],
              fill=(250, 209, 31), width=2)
    draw.rectangle([px(-0.5), py(SAFE_ZONE_SPAN[1]), px(0.5),
                    py(SAFE_ZONE_SPAN[0])], outline=(51, 242, 115), width=1)

    # every road user, with a tail showing which way it is going
    for name in VEHICLE_NAMES:
        vx, vy = record["vehicle_xy"][name]
        if abs(vy) > span_y:
            continue
        color = VEHICLE_RGB[name]
        cx, cy = px(vy), py(vx)
        draw.rectangle([cx - 5, cy - 4, cx + 5, cy + 4], fill=color)
        direction = 1.0 if vx > 0 else -1.0     # far lane goes +y
        tail = px(vy - direction * 0.55)
        draw.line([cx, cy, tail, cy], fill=color, width=1)

    # the duck
    dx, dy = record["duck_xy"]
    cx, cy = px(dy), py(dx)
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 246, 120),
                 outline=(20, 20, 10))
    draw.text((x0 + 5, y0 + 3), "PLAN VIEW", fill=DIM, font=tiny)
    draw.text((x0 + width - 42, y0 + 3), "◀ left", fill=(255, 176, 60),
              font=tiny)
    draw.text((x0 + 5, y0 + height - 14), "right ▶", fill=(110, 190, 255),
              font=tiny)


def _conflict_panel(draw, record, x0, y0, width, small, tiny):
    """Per-vehicle predicted crossing-corridor window, as a bar chart.

    Each bar spans the interval during which that vehicle's body is predicted
    to occupy the pedestrian corridor, on a 0-14 s axis anchored at now.  The
    duck's own predicted lane occupancy is drawn behind them, so overlap — the
    thing the gap decision refuses — is visible directly rather than inferred
    from a number.

    The whole panel sits on its own dark backing.  The first preview drew this
    chart straight onto the street, where the bright pavement and the white
    zebra made every label and every pale bar unreadable — the same defect
    ``come-here-recall`` hit with its legend.
    """
    horizon = 14.0
    row_h = 15
    label_w = 52
    height = 26 + len(VEHICLE_NAMES) * row_h + 16
    draw.rectangle([x0 - 6, y0 - 4, x0 + width + 6, y0 + height],
                   fill=(6, 10, 16), outline=(46, 56, 72))
    draw.text((x0, y0), "PREDICTED CORRIDOR OCCUPANCY (s from now)",
              fill=DIM, font=tiny)
    base = y0 + 14
    track_x0 = x0 + label_w
    track_w = width - label_w - 4

    def bar_x(seconds):
        """Panel x for a time in seconds. Anchored at the track's own origin."""
        return track_x0 + int(
            track_w * min(max(seconds, 0.0), horizon) / horizon)

    # the duck's own predicted window, if it stepped off now
    duck_window = record.get("duck_windows", {}).get("near")
    if duck_window:
        draw.rectangle([bar_x(duck_window[0]), base + 8,
                        bar_x(duck_window[1]), base + 10 + len(VEHICLE_NAMES) * row_h],
                       fill=(56, 50, 20))

    for index, name in enumerate(VEHICLE_NAMES):
        y = base + 10 + index * row_h
        color = VEHICLE_RGB[name]
        draw.text((x0, y), f"{name[:7]}", fill=color, font=tiny)
        draw.line([track_x0, y + 6, track_x0 + track_w, y + 6],
                  fill=(38, 46, 58), width=1)
        window = record["vehicle_windows"].get(name)
        if window is None:
            draw.text((track_x0 + 2, y), "clear", fill=(84, 96, 112),
                      font=tiny)
            continue
        a, b = bar_x(window[0]), bar_x(window[1])
        draw.rectangle([a, y + 2, max(b, a + 3), y + 10], fill=color)
    draw.text((x0, base + 12 + len(VEHICLE_NAMES) * row_h),
              f"duck window ▖   needs {SAFETY_MARGIN_S:.1f} s clear each side",
              fill=(190, 180, 110), font=tiny)


def compose(main_rgb, pip_rgb, *, record, total_seconds, machine_summary):
    """Draw one finished frame from one rollout record."""
    image = Image.fromarray(main_rgb)
    draw = ImageDraw.Draw(image)
    font, small, tiny = _font(15), _font(12), _font(11)
    width, height = image.size

    state = record["state"]
    state_color = STATE_RGB[state]
    command = record["command"]
    moving = any(abs(v) > 1e-9 for v in command)

    # ---- header ------------------------------------------------------
    draw.rectangle([0, 0, width, HUD_H], fill=PANEL)
    draw.text((12, 8), f"CROSSWALK GUARDIAN    t={record['t']:05.2f}s",
              fill=state_color, font=font)
    draw.text((12, 30),
              f"STATE {SHORT[state]:<10s} ({record['state_elapsed_s']:4.1f}s)   "
              f"x={record['duck_xy'][0]:+.3f} m  y={record['duck_xy'][1]:+.3f} m  "
              f"yaw={record['duck_yaw_deg']:+6.1f}°",
              fill=state_color, font=small)

    # Where the duck is relative to the paint — the claim the gates grade.
    if record["in_safe_zone"]:
        place, place_color = "IN THE FAR-SIDE SAFE ZONE", GOOD
    elif record["in_far_lane"]:
        place, place_color = "IN THE FAR LANE (traffic from the RIGHT)", WARN
    elif record["in_near_lane"]:
        place, place_color = "IN THE NEAR LANE (traffic from the LEFT)", WARN
    elif record["encroaches"]:
        place, place_color = "PAST THE WAIT LINE", BAD
    else:
        place, place_color = "BEHIND THE WAIT LINE", GOOD
    draw.text((12, 50),
              f"{place}   wait-line margin {record['wait_line_margin_m']:+.3f} m",
              fill=place_color, font=small)

    # The gap decision, in the same terms the machine used.
    margin = record["gap_margin_s"]
    if margin >= SAFETY_MARGIN_S:
        gap_text = f"GAP SAFE   worst margin {margin:+.2f} s"
        gap_color = GOOD
    else:
        gap_text = (f"GAP UNSAFE  worst margin {margin:+.2f} s  "
                    f"(limiting: {record['gap_limiting']})")
        gap_color = BAD
    draw.text((12, 70),
              f"{gap_text}   needs ≥{SAFETY_MARGIN_S:.1f} s   "
              f"est. crossing {record['crossing_estimate_s']:.1f} s",
              fill=gap_color, font=small)

    draw.text((12, 90),
              f"command vx={command[0]:+.3f} vy={command[1]:+.3f} "
              f"wz={command[2]:+.3f}  "
              f"{'WALKING' if moving else 'STOPPED (exactly zero)'}",
              fill=(120, 220, 255) if moving else GOOD, font=small)
    draw.text((12, 110),
              f"look ◀ left {record['left_fraction']:.2f}   "
              f"look ▶ right {record['right_fraction']:.2f}   "
              f"nearest {record['nearest_vehicle']:<8s} "
              f"clearance {record['nearest_clearance_m']:+.3f} m   "
              f"rejected gaps {record['rejected_gaps']}",
              fill=INK, font=small)
    upright = record["trunk_z_m"] >= 0.09
    draw.text((12, 130),
              f"trunk z={record['trunk_z_m']:.3f} m   "
              f"min={record['min_trunk_z_m']:.3f} m   "
              f"path={record['path_m']:.2f} m   "
              f"{'UPRIGHT' if upright else 'FALLEN'}",
              fill=GOOD if upright else BAD, font=small)

    # ---- head-camera PiP ---------------------------------------------
    px0, py0 = width - PIP_W - 12, HUD_H + 12
    if state in ("LOOK_LEFT", "LOOK_LEFT_AGAIN"):
        border, label = (255, 176, 60), "LOOKING LEFT"
    elif state == "LOOK_RIGHT":
        border, label = (110, 190, 255), "LOOKING RIGHT"
    elif state == "WAIT_FOR_GAP":
        border, label = (255, 224, 90), "WATCHING BOTH WAYS"
    elif state == "CROSSING":
        border, label = (120, 255, 170), "CROSSING"
    else:
        border, label = (150, 166, 186), "AHEAD"
    draw.rectangle([px0 - 4, py0 - 24, px0 + PIP_W + 4, py0 + PIP_H + 4],
                   fill=(2, 5, 8))
    draw.text((px0, py0 - 21), f"DUCK HEAD CAMERA · {label}", fill=border,
              font=small)
    image.paste(Image.fromarray(pip_rgb), (px0, py0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([px0 - 1, py0 - 1, px0 + PIP_W, py0 + PIP_H],
                   outline=border, width=3)
    cx, cy = px0 + PIP_W // 2, py0 + PIP_H // 2
    draw.line([cx - 12, cy, cx + 12, cy], fill=(255, 245, 95), width=1)
    draw.line([cx, cy - 12, cx, cy + 12], fill=(255, 245, 95), width=1)
    # Which sector this phase is graded against, and whether it is satisfied.
    if state in ("LOOK_LEFT", "LOOK_LEFT_AGAIN", "LOOK_RIGHT"):
        sector = "right" if state == "LOOK_RIGHT" else "left"
        fraction = record[f"{sector}_fraction"]
        seen = record[f"{sector}_visible"]
        draw.rectangle([px0 + 4, py0 + 4, px0 + 170, py0 + 22],
                       fill=(4, 10, 16))
        draw.text((px0 + 8, py0 + 6),
                  f"{sector} road {fraction * 100:3.0f}% "
                  f"{'✓ SEEN' if seen else '…'}",
                  fill=GOOD if seen else WARN, font=small)
    draw.text((px0 + 6, py0 + PIP_H - 18),
              "stabilized · world-up · head position",
              fill=(190, 205, 225), font=tiny)
    if record["visible_vehicles"]:
        draw.text((px0 + 6, py0 + PIP_H - 33),
                  "sees: " + ", ".join(record["visible_vehicles"][:4]),
                  fill=(210, 220, 235), font=tiny)

    # ---- plan view + conflict panel ----------------------------------
    map_x, map_y = 10, HUD_H + 12
    _road_map(draw, record, map_x, map_y, 268, 150, small, tiny)
    _conflict_panel(draw, record, map_x, map_y + 160, 268, small, tiny)

    # ---- state pipeline ----------------------------------------------
    draw.rectangle([0, height - 104, width, height], fill=PANEL)
    pipe_y = height - 96
    slot = (width - 24) // len(PIPELINE)
    for index, name in enumerate(PIPELINE):
        x = 12 + index * slot
        active = name == state
        done = PIPELINE.index(state) > index
        if active:
            fill, ink = STATE_RGB[name], (5, 10, 16)
        elif done:
            fill, ink = (24, 52, 40), (120, 200, 150)
        else:
            fill, ink = (26, 32, 44), STATE_RGB[name]
        draw.rounded_rectangle([x, pipe_y, x + slot - 8, pipe_y + 26],
                               radius=6, fill=fill, outline=STATE_RGB[name],
                               width=2)
        draw.text((x + 5, pipe_y + 7), SHORT[name], fill=ink, font=tiny)

    # ---- traffic timeline --------------------------------------------
    draw.text((12, pipe_y + 30),
              "road-user passes over the crossing (hollow) · "
              "duck crossing (solid) · ▮ now",
              fill=DIM, font=tiny)
    bar_y = height - 28
    bar_x0, bar_x1 = 12, width - 12
    draw.rounded_rectangle([bar_x0, bar_y, bar_x1, bar_y + 20], radius=6,
                           fill=(18, 24, 34), outline=(52, 64, 82), width=1)
    span = max(bar_x1 - bar_x0, 1)

    def _x(seconds):
        return bar_x0 + int(span * min(max(seconds, 0.0) / total_seconds, 1.0))

    for entry in machine_summary.get("arrivals", []):
        x0, x1 = _x(entry["enter_s"]), _x(entry["exit_s"])
        color = VEHICLE_RGB[entry["vehicle"]]
        draw.rounded_rectangle([x0, bar_y + 2, max(x1, x0 + 4), bar_y + 8],
                               radius=2, outline=color, width=1)
    crossing = machine_summary.get("crossing_window")
    if crossing:
        x0, x1 = _x(crossing[0]), _x(crossing[1])
        draw.rounded_rectangle([x0, bar_y + 11, max(x1, x0 + 6), bar_y + 18],
                               radius=2, fill=(120, 255, 170))
        draw.text((x0 + 3, bar_y + 10), "CROSS", fill=(6, 10, 16), font=tiny)
    head = _x(record["t"])
    draw.line([head, bar_y - 3, head, bar_y + 23], fill=(255, 255, 255), width=2)
    return image
