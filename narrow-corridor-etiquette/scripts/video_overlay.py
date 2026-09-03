#!/usr/bin/env python3
"""HUD, state pipeline, corridor plan view, alcove scorecard and head-camera PiP.

Presentation only: nothing here is allowed to influence the rollout.  The
composed frame answers five questions a viewer should not have to infer — where
the duck is in the corridor, which bays it is choosing between and why it
refused the ones it refused, whether its footprint is out of the passage the
adult needs, why it is or is not moving, and what the duck itself can see.

The PiP geometry is imported from ``etiquette_camera`` rather than redefined,
because that same rectangle sets the frustum every tracking percentage is
measured against.  Drawing one shape while measuring another would let the HUD
disagree with the picture.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corridor import (  # noqa: E402
    ALCOVES,
    ALCOVE_BY_NAME,
    CENTER_PASSAGE_HALF,
    CORRIDOR_HALF_WIDTH,
    CORRIDOR_X_MAX,
    CORRIDOR_X_MIN,
    DESTINATION_HALF,
    DESTINATION_X,
    corridor_passing_geometry,
)
from etiquette_camera import PIP_H, PIP_W  # noqa: E402
from people import PEDESTRIANS, PERSON_NAMES  # noqa: E402

# Person colours, DERIVED from the schedule rather than restated.  A
# presentation table kept in sync by hand will eventually disagree with the
# scene it is drawing — the crosswalk behavior crashed a full render with a
# KeyError for exactly that reason.
def _person_palette() -> dict[str, tuple[int, int, int]]:
    palette: dict[str, tuple[int, int, int]] = {}
    for person in PEDESTRIANS:
        red, green, blue = (float(v) for v in person.rgba.split()[:3])
        palette[person.name] = (int(red * 255), int(green * 255),
                                int(blue * 255))
    return palette


PERSON_RGB = _person_palette()
STATE_RGB = {
    "CRUISE": (120, 200, 255),
    "DETECT": (255, 205, 90),
    "SELECT_ALCOVE": (255, 176, 60),
    "PULL_OVER": (255, 140, 90),
    "YIELD": (255, 110, 110),
    "CLEAR": (170, 220, 140),
    "REJOIN": (140, 210, 255),
    "RESUME": (120, 220, 255),
    "DONE": (110, 255, 130),
}
PIPELINE = ("CRUISE", "DETECT", "SELECT_ALCOVE", "PULL_OVER", "YIELD",
            "CLEAR", "REJOIN", "RESUME", "DONE")
SHORT = {
    "CRUISE": "CRUISE", "DETECT": "DETECT", "SELECT_ALCOVE": "SELECT BAY",
    "PULL_OVER": "PULL OVER", "YIELD": "YIELD", "CLEAR": "CLEAR",
    "REJOIN": "REJOIN", "RESUME": "RESUME", "DONE": "DONE",
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


def _corridor_map(draw, record, x0, y0, width, height, small, tiny):
    """Plan view of the corridor: walls, bays, the centre passage, everybody.

    This is the panel that makes the decision legible.  The duck is drawn
    against the same wall lines and the same passage edges the gates are graded
    on, so a viewer can see for themselves that its footprint left the passage
    and came back to the centreline.
    """
    # x maps to the panel's HORIZONTAL axis, y to the vertical, matching the
    # wide shot rather than transposing it.
    span_x = (CORRIDOR_X_MIN - 0.15, CORRIDOR_X_MAX + 0.15)
    span_y = 0.62
    draw.rectangle([x0, y0, x0 + width, y0 + height], fill=(10, 14, 20),
                   outline=(52, 64, 82))

    def px(world_x):
        return x0 + width * (world_x - span_x[0]) / (span_x[1] - span_x[0])

    def py(world_y):
        return y0 + height * (0.5 - world_y / (2.0 * span_y))

    # the corridor floor
    draw.rectangle([px(CORRIDOR_X_MIN), py(CORRIDOR_HALF_WIDTH),
                    px(CORRIDOR_X_MAX), py(-CORRIDOR_HALF_WIDTH)],
                   fill=(30, 32, 36))
    # every alcove, tinted by whether it can actually be used
    for alcove in ALCOVES:
        low, high = alcove.x_span
        outer = alcove.side * alcove.outer_y
        inner = alcove.side * CORRIDOR_HALF_WIDTH
        fill = (26, 62, 38) if alcove.clears_passage else (62, 34, 26)
        draw.rectangle(
            [px(low), py(max(outer, inner)), px(high), py(min(outer, inner))],
            fill=fill)
        if alcove.blocked_from is not None:
            block_inner = alcove.side * alcove.blocked_from
            draw.rectangle(
                [px(low), py(max(outer, block_inner)),
                 px(high), py(min(outer, block_inner))],
                fill=(150, 106, 52))
    # walls
    for sign in (+1.0, -1.0):
        draw.line([px(CORRIDOR_X_MIN), py(sign * CORRIDOR_HALF_WIDTH),
                   px(CORRIDOR_X_MAX), py(sign * CORRIDOR_HALF_WIDTH)],
                  fill=(120, 124, 130), width=1)
    # the centre passage the adult needs, and the centreline
    draw.rectangle([px(CORRIDOR_X_MIN), py(CENTER_PASSAGE_HALF),
                    px(CORRIDOR_X_MAX), py(-CENTER_PASSAGE_HALF)],
                   outline=(226, 110, 60), width=1)
    draw.line([px(CORRIDOR_X_MIN), py(0.0), px(CORRIDOR_X_MAX), py(0.0)],
              fill=(150, 140, 70), width=1)
    # destination
    draw.rectangle([px(DESTINATION_X - DESTINATION_HALF),
                    py(CORRIDOR_HALF_WIDTH),
                    px(DESTINATION_X + DESTINATION_HALF),
                    py(-CORRIDOR_HALF_WIDTH)],
                   outline=(51, 242, 115), width=2)

    # the selected bay, highlighted
    target = record.get("target_alcove")
    if target:
        alcove = ALCOVE_BY_NAME[target]
        low, high = alcove.x_span
        outer = alcove.side * alcove.outer_y
        inner = alcove.side * CORRIDOR_HALF_WIDTH
        draw.rectangle(
            [px(low), py(max(outer, inner)), px(high), py(min(outer, inner))],
            outline=(120, 255, 170), width=2)

    # every adult
    for name in PERSON_NAMES:
        ax, ay = record["person_xy"][name]
        if not (span_x[0] <= ax <= span_x[1]):
            continue
        colour = PERSON_RGB[name]
        cx, cy = px(ax), py(ay)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=colour)

    # the duck, drawn as its actual footprint rather than a dot
    dx, dy = record["duck_xy"]
    radius_px = max(3, int(width * 0.1303 / (span_x[1] - span_x[0])))
    cx, cy = px(dx), py(dy)
    draw.ellipse([cx - radius_px, cy - radius_px,
                  cx + radius_px, cy + radius_px],
                 fill=(255, 246, 120), outline=(20, 20, 10))
    draw.text((x0 + 5, y0 + 3), "CORRIDOR PLAN", fill=DIM, font=tiny)
    draw.text((x0 + width - 46, y0 + 3), "+Y left", fill=DIM, font=tiny)


def _alcove_panel(draw, record, x0, y0, width, small, tiny):
    """One row per bay: usable depth, whether it clears, and its verdict.

    The whole panel sits on its own dark backing.  A previous behavior in this
    lab drew a chart straight onto a bright scene and made every label
    unreadable; the same mistake is not repeated here.
    """
    row_h = 26
    height = 24 + len(ALCOVES) * row_h + 12
    draw.rectangle([x0 - 6, y0 - 4, x0 + width + 6, y0 + height],
                   fill=(6, 10, 16), outline=(46, 56, 72))
    draw.text((x0, y0), "PULL-OVER CANDIDATES", fill=DIM, font=tiny)
    scores = {s["alcove"]: s for s in record.get("alcove_scores", [])}
    selected = record.get("target_alcove")
    base = y0 + 16
    # The verdict is right-aligned into its own column.  MEASURED FROM THE
    # PREVIEW: drawing it at a fixed offset from the right edge overlapped the
    # bay name on every row, because "too shallow" is far wider than "viable".
    verdict_right = x0 + width - 6
    for index, alcove in enumerate(ALCOVES):
        y = base + index * row_h
        score = scores.get(alcove.name)
        if alcove.name == selected:
            colour, verdict = GOOD, "SELECTED"
        elif score is None:
            colour, verdict = DIM, ""
        elif score["viable"]:
            colour, verdict = (170, 220, 140), "viable"
        elif not score["clears_passage"]:
            colour = BAD
            verdict = ("blocked" if alcove.blocked_from is not None
                       else "too shallow")
        elif score["behind"]:
            colour, verdict = (120, 130, 145), "behind"
        else:
            colour, verdict = WARN, "too far"
        draw.rectangle([x0, y + 1, x0 + 4, y + 18], fill=colour)
        draw.text((x0 + 9, y), f"{alcove.name}", fill=colour, font=tiny)
        detail = (f"depth {alcove.usable_outer_y - CORRIDOR_HALF_WIDTH:.2f} m  "
                  f"reach {alcove.max_trunk_abs_y:.3f}")
        if score is not None and score["reachable"]:
            detail += f"  margin {score['time_margin_s']:+.1f}s"
        draw.text((x0 + 9, y + 11), detail, fill=DIM, font=tiny)
        if verdict:
            text_width = draw.textlength(verdict, font=tiny)
            draw.text((verdict_right - text_width, y + 5), verdict,
                      fill=colour, font=tiny)


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
    geometry = machine_summary.get("passing_geometry",
                                   corridor_passing_geometry())

    # ---- header ------------------------------------------------------
    draw.rectangle([0, 0, width, HUD_H], fill=PANEL)
    draw.text((12, 8),
              f"NARROW CORRIDOR ETIQUETTE    t={record['t']:05.2f}s    "
              f"cycle {record['completed_cycles']}/2",
              fill=state_color, font=font)
    draw.text((12, 30),
              f"STATE {SHORT[state]:<11s} ({record['state_elapsed_s']:4.1f}s)   "
              f"x={record['duck_xy'][0]:+.3f} m  y={record['duck_xy'][1]:+.3f} m  "
              f"yaw={record['duck_yaw_deg']:+6.1f}°   "
              f"{record['destination_remaining_m']:.2f} m to go",
              fill=state_color, font=small)

    # Where the duck is relative to the passage the adult needs.  Standing on
    # the centreline is only worth flagging while there is somebody to make
    # room for: at the destination, and while cruising an empty corridor, the
    # middle is exactly where the duck belongs.
    if record["at_destination"]:
        place, place_color = "ARRIVED AT THE DESTINATION", GOOD
    elif record["clears_passage"]:
        place, place_color = "FOOTPRINT CLEAR OF THE PASSAGE", GOOD
    elif state in ("DETECT", "SELECT_ALCOVE", "PULL_OVER"):
        place, place_color = "STILL IN THE CENTRE PASSAGE", WARN
    else:
        place, place_color = "ON THE CORRIDOR CENTRELINE", INK
    draw.text((12, 50),
              f"{place}   intrusion {record['passage_intrusion_m']:+.3f} m   "
              f"passage half-width {CENTER_PASSAGE_HALF:.3f} m",
              fill=place_color, font=small)

    # The premise, restated from the same arithmetic the tests check.
    draw.text((12, 70),
              f"corridor {geometry['corridor_width_m']:.2f} m wide: best "
              f"side-by-side gap {geometry['best_possible_surface_gap_m']:+.3f} m "
              f"vs {geometry['safe_gap_m']:.2f} m needed → "
              f"CANNOT PASS ABREAST (short by {geometry['shortfall_m']:.3f} m)",
              fill=WARN, font=small)

    draw.text((12, 90),
              f"command vx={command[0]:+.3f} vy={command[1]:+.3f} "
              f"wz={command[2]:+.3f}  "
              f"{'WALKING' if moving else 'STOPPED (exactly zero)'}",
              fill=(120, 220, 255) if moving else GOOD, font=small)

    if record["soonest_person"] and not record["at_destination"]:
        ttm = record["soonest_time_to_meet_s"]
        ttm_text = "—" if ttm == float("inf") else f"{ttm:5.2f}s"
        draw.text((12, 110),
                  f"nearest encounter {record['soonest_person']:<6s} "
                  f"range {record['soonest_range_m']:5.2f} m  "
                  f"meet in {ttm_text}  "
                  f"if it just walked past: "
                  f"{record['soonest_counterfactual_m']:+.3f} m clearance",
                  fill=INK, font=small)
    elif record["at_destination"]:
        draw.text((12, 110),
                  f"{record['completed_cycles']} etiquette cycles completed · "
                  "pulled over, waited, and carried on",
                  fill=GOOD, font=small)
    upright = record["trunk_z_m"] >= 0.09
    draw.text((12, 130),
              f"trunk z={record['trunk_z_m']:.3f} m   "
              f"min={record['min_trunk_z_m']:.3f} m   "
              f"path={record['path_m']:.2f} m   "
              f"person {record['nearest_clearance_m']:+.3f} m   "
              f"wall {record['wall_clearance_m']:+.3f} m   "
              f"{'UPRIGHT' if upright else 'FALLEN'}",
              fill=GOOD if upright else BAD, font=small)

    # ---- head-camera PiP ---------------------------------------------
    px0, py0 = width - PIP_W - 12, HUD_H + 12
    tracked = record.get("tracked_person")
    if state == "YIELD":
        border, label = (255, 110, 110), f"WATCHING {tracked or ''}".strip()
    elif state in ("DETECT", "SELECT_ALCOVE", "PULL_OVER"):
        border, label = (255, 176, 60), f"TRACKING {tracked or ''}".strip()
    elif state in ("REJOIN", "RESUME"):
        border, label = (140, 210, 255), "BACK TO THE CORRIDOR"
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
    if tracked:
        seen = record["tracked_visible"]
        draw.rectangle([px0 + 4, py0 + 4, px0 + 176, py0 + 22],
                       fill=(4, 10, 16))
        draw.text((px0 + 8, py0 + 6),
                  f"{tracked} {record['tracked_fraction'] * 100:3.0f}% "
                  f"{'✓ IN VIEW' if seen else '…'}",
                  fill=GOOD if seen else WARN, font=small)
    draw.text((px0 + 6, py0 + PIP_H - 18),
              "stabilized · world-up · head position",
              fill=(190, 205, 225), font=tiny)

    # ---- plan view + alcove scorecard --------------------------------
    map_x, map_y = 10, HUD_H + 12
    _corridor_map(draw, record, map_x, map_y, 300, 150, small, tiny)
    _alcove_panel(draw, record, map_x, map_y + 162, 316, small, tiny)

    # ---- state pipeline ----------------------------------------------
    draw.rectangle([0, height - 104, width, height], fill=PANEL)
    pipe_y = height - 96
    slot = (width - 24) // len(PIPELINE)
    active_index = PIPELINE.index(state)
    for index, name in enumerate(PIPELINE):
        x = 12 + index * slot
        active = name == state
        done = active_index > index
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

    # ---- timeline ----------------------------------------------------
    draw.text((12, pipe_y + 30),
              "adult in the corridor (hollow) · duck yielding in a bay (solid) "
              "· ▮ now",
              fill=DIM, font=tiny)
    bar_y = height - 28
    bar_x0, bar_x1 = 12, width - 12
    draw.rounded_rectangle([bar_x0, bar_y, bar_x1, bar_y + 20], radius=6,
                           fill=(18, 24, 34), outline=(52, 64, 82), width=1)
    span = max(bar_x1 - bar_x0, 1)

    def _x(seconds):
        return bar_x0 + int(span * min(max(seconds, 0.0) / total_seconds, 1.0))

    for entry in machine_summary.get("passes", []):
        x0, x1 = _x(entry["enter_s"]), _x(entry["exit_s"])
        colour = PERSON_RGB[entry["person"]]
        draw.rounded_rectangle([x0, bar_y + 2, max(x1, x0 + 4), bar_y + 8],
                               radius=2, outline=colour, width=1)
    for window in machine_summary.get("yield_windows", []):
        x0, x1 = _x(window[0]), _x(window[1])
        draw.rounded_rectangle([x0, bar_y + 11, max(x1, x0 + 6), bar_y + 18],
                               radius=2, fill=(255, 140, 120))
        draw.text((x0 + 3, bar_y + 10), "YIELD", fill=(6, 10, 16), font=tiny)
    head = _x(record["t"])
    draw.line([head, bar_y - 3, head, bar_y + 23], fill=(255, 255, 255), width=2)
    return image
