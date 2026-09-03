#!/usr/bin/env python3
"""The HUD panels: state, the two side verdicts, and the formation error.

Every number drawn here comes straight out of the per-tick record built by
``beside_record``, which is the SAME dict the acceptance gate reads.  The
overlay therefore cannot show a figure the gate did not grade, and a viewer
reading the HUD is reading the measurement rather than a caption about it.

THE SIDE-RISK PANEL IS THE POINT OF THE VIDEO
----------------------------------------------
Both sides are graded every control tick against two different margins — 0.22 m
to a static surface, 0.55 m to a predicted pedestrian — and the panel draws both
gaps as bars against their own margin, for BOTH sides, at once.  That is what
makes the switch legible: the viewer watches the left row's static bar collapse
as the kiosk enters the swept lane, sees the row turn red with the cause
``static:kiosk`` written next to it, and only then does the duck move.  Cause
before effect, on screen, in that order.

A gap is drawn against its own margin rather than on a shared scale, because the
two thresholds differ by a factor of 2.5 and a shared axis would make the person
margin look permanently safer than it is.
"""

from __future__ import annotations

from hud_style import (
    ACCENT,
    BAD,
    DIM,
    F09,
    F10,
    F11,
    F13,
    GOOD,
    GUARDIAN_IN_HUD,
    INK,
    STATE_CAPTION,
    STATE_COLORS,
    WARN,
    bar,
    fit,
    panel,
    role_color,
    side_ink,
    span_bar,
    text_w,
    title,
)

# The two refusal margins, imported rather than restated so the bars are drawn
# against the thresholds the chooser actually applies.
from beside_geometry import (
    BESIDE_MAX_M,
    BESIDE_MIN_M,
    BESIDE_LONG_TOLERANCE_M,
    FORWARD_HALF_PLANE_M,
    SIDE_PERSON_MARGIN_M,
    SIDE_STATIC_MARGIN_M,
)


def draw_status(draw, box, record) -> None:
    """What the duck is doing, which side it is on, and its command."""
    x0, y0, x1, y1 = box
    panel(draw, box)
    title(draw, box, "STATE")
    state = record["state"]
    color = STATE_COLORS.get(state, DIM)
    draw.text((x0 + 8, y0 + 18), fit(draw, state, F13, x1 - x0 - 16),
              font=F13, fill=color)
    draw.text((x0 + 8, y0 + 36),
              fit(draw, STATE_CAPTION.get(state, ""), F09, x1 - x0 - 16),
              font=F09, fill=DIM)

    side = record["side_name"]
    draw.text((x0 + 8, y0 + 54), "side", font=F09, fill=DIM)
    draw.text((x0 + 54, y0 + 54), (side or "none").upper(), font=F10,
              fill=side_ink(side))
    draw.text((x0 + 118, y0 + 54), "formation", font=F09, fill=DIM)
    ok = record["formation_ok"]
    draw.text((x0 + 196, y0 + 54), "HELD" if ok else "closing", font=F10,
              fill=GOOD if ok else WARN)

    # Command and trunk height: the two numbers that say the physics is real.
    vx, vy, wz = record["command"][0], record["command"][1], record["command"][2]
    draw.text((x0 + 8, y0 + 72),
              f"cmd vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}", font=F09,
              fill=INK if record["command_peak"] > 0 else DIM)
    draw.text((x0 + 8, y0 + 86),
              f"trunk z {record['trunk_z']:.4f} m   min {record['min_trunk_z']:.4f} m",
              font=F09, fill=DIM)
    draw.text((x0 + 8, y0 + 100),
              f"path {record['path_m']:.2f} m   she walks "
              f"{record['guardian_speed_mps']:.3f} m/s", font=F09, fill=DIM)


def _risk_row(draw, box, verdict, label: str, is_own: bool) -> None:
    """One side's two measured gaps, each against its own margin."""
    x0, y0, x1, y1 = box
    usable = verdict["usable"]
    ink = side_ink(1 if label == "LEFT" else -1)
    draw.text((x0, y0), label, font=F10, fill=ink)
    verdict_text = "usable" if usable else "REFUSED"
    draw.text((x0 + 46, y0), verdict_text, font=F10,
              fill=GOOD if usable else BAD)
    if is_own:
        draw.text((x0 + 108, y0), "\u25c0 duck here", font=F09, fill=INK)

    # Static gap against the 0.22 m margin.
    static = float(verdict["static_gap_m"])
    draw.text((x0, y0 + 15), "static", font=F09, fill=DIM)
    bar(draw, (x0 + 42, y0 + 16, x0 + 132, y0 + 23),
        static / (2.0 * SIDE_STATIC_MARGIN_M),
        GOOD if static >= SIDE_STATIC_MARGIN_M else BAD)
    draw.text((x0 + 138, y0 + 15),
              f"{static:+.3f} m  {verdict['static_name'][:9]}", font=F09,
              fill=GOOD if static >= SIDE_STATIC_MARGIN_M else BAD)

    # Predicted-person gap against the 0.55 m margin.
    person = float(verdict["person_gap_m"])
    draw.text((x0, y0 + 29), "people", font=F09, fill=DIM)
    bar(draw, (x0 + 42, y0 + 30, x0 + 132, y0 + 37),
        person / (2.0 * SIDE_PERSON_MARGIN_M),
        GOOD if person >= SIDE_PERSON_MARGIN_M else BAD)
    draw.text((x0 + 138, y0 + 29),
              f"{person:.3f} m  {verdict['person_name'][:9]}", font=F09,
              fill=GOOD if person >= SIDE_PERSON_MARGIN_M else BAD)

    if not usable:
        draw.text((x0, y0 + 43),
                  fit(draw, f"cause  {verdict['cause']}:{verdict['detail']}",
                      F09, x1 - x0), font=F09, fill=BAD)


def draw_side_risk(draw, box, record) -> None:
    """Both sides, both margins, every tick.  The cause of every switch."""
    x0, y0, x1, y1 = box
    panel(draw, box)
    title(draw, box, "SIDE RISK - graded every tick, both sides")
    own = record["side"]
    _risk_row(draw, (x0 + 8, y0 + 22, x1 - 8, y0 + 80), record["verdict_left"],
              "LEFT", own == 1)
    _risk_row(draw, (x0 + 8, y0 + 84, x1 - 8, y0 + 142), record["verdict_right"],
              "RIGHT", own == -1)
    draw.text((x0 + 8, y1 - 15),
              fit(draw, f"margins  static {SIDE_STATIC_MARGIN_M:.2f} m   "
                  f"predicted people {SIDE_PERSON_MARGIN_M:.2f} m over 3.0 s",
                  F09, x1 - x0 - 16), font=F09, fill=DIM)


def draw_formation(draw, box, record) -> None:
    """Lateral offset against the band, longitudinal against the half-plane."""
    x0, y0, x1, y1 = box
    panel(draw, box)
    title(draw, box, "FORMATION ERROR - in her frame")

    # Lateral: signed, so the side switch is visible as the value crossing zero.
    lateral = float(record["lateral_m"])
    inside = BESIDE_MIN_M <= abs(lateral) <= BESIDE_MAX_M
    draw.text((x0 + 8, y0 + 20), "lateral", font=F09, fill=DIM)
    draw.text((x0 + 70, y0 + 19),
              f"{lateral:+.3f} m  ({'left' if lateral > 0 else 'right'})",
              font=F10, fill=GOOD if inside else WARN)
    # BOTH mirrored bands in ONE call: her left in +, her right in -, so the
    # duck is seen travelling out of one acceptance window and into the other.
    span_bar(draw, (x0 + 8, y0 + 38, x1 - 8, y0 + 48), -0.95, 0.95, lateral,
             GOOD if inside else WARN,
             bands=((BESIDE_MIN_M, BESIDE_MAX_M),
                    (-BESIDE_MAX_M, -BESIDE_MIN_M)))
    draw.text((x0 + 8, y0 + 51), "her right", font=F09, fill=side_ink(-1))
    right_label = "her left"
    draw.text((x1 - 8 - text_w(draw, right_label, F09), y0 + 51), right_label,
              font=F09, fill=side_ink(1))

    # Longitudinal: the half-plane limit is the claim, so it is drawn as a band
    # whose upper edge IS the limit.
    longitudinal = float(record["longitudinal_m"])
    ahead = longitudinal > FORWARD_HALF_PLANE_M
    draw.text((x0 + 8, y0 + 68), "longitud.", font=F09, fill=DIM)
    draw.text((x0 + 70, y0 + 67),
              f"{longitudinal:+.3f} m  ({'AHEAD' if longitudinal > 0 else 'astern'})",
              font=F10, fill=BAD if ahead else GOOD)
    span_bar(draw, (x0 + 8, y0 + 86, x1 - 8, y0 + 96), -1.35, 0.45,
             longitudinal, BAD if ahead else GOOD,
             bands=((-BESIDE_LONG_TOLERANCE_M, FORWARD_HALF_PLANE_M),))
    draw.text((x0 + 8, y0 + 100),
              fit(draw, f"band ends at +{FORWARD_HALF_PLANE_M:.2f} m: past it "
                  "the duck is in front of her", F09, x1 - x0 - 16),
              font=F09, fill=DIM)

    # The rear crossing waypoint, only while one exists.
    kind = record["target_kind"]
    if kind:
        target = record["target_xy"]
        text = f"target  {kind}"
        if target is not None:
            text += f"  ({target[0]:+.2f}, {target[1]:+.2f})"
        draw.text((x0 + 8, y0 + 116), fit(draw, text, F09, x1 - x0 - 16),
                  font=F09, fill=ACCENT if "cross" in kind else DIM)


def draw_legend(draw, box, record) -> None:
    """Who is who in the picture, coloured as they are in the plan view.

    Laid out from MEASURED text width and stopped at the panel edge, because
    five names at fixed spacing overflow a 304 px panel and print on top of one
    another.  Names that do not fit are dropped rather than overprinted.
    """
    x0, y0, x1, y1 = box
    panel(draw, box)
    x = x0 + 8
    for name, role in record["person_role"].items():
        width = 10 + 4 + text_w(draw, name, F09) + 8
        if x + width > x1 - 8:
            break
        draw.rectangle([x, y0 + 10, x + 8, y0 + 17], fill=role_color(role))
        seen = record["person_visible"].get(name, False)
        draw.text((x + 12, y0 + 8), name, font=F09, fill=INK if seen else DIM)
        x += width
