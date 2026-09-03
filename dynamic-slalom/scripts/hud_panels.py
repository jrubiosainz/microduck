#!/usr/bin/env python3
"""The HUD panels: what the duck predicted and decided this tick, as numbers.

Every panel reads the SAME per-tick record the acceptance gate reads, so the
overlay cannot show a figure the gate did not grade.  Nothing here computes
anything about the behavior; it only lays out quantities that already exist.

FIVE PANELS, EACH ANSWERING ONE QUESTION A VIEWER WILL ASK
------------------------------------------------------------
* **STATE**    - what is it doing, in plain English, and is it moving?
* **DECISION** - which corridor did it take, which did it refuse, and by how
  much?  This is the panel the whole behavior exists to justify, so both
  corridors are always shown side by side, never just the winner.
* **THREAT**   - who is crossing, how far away, and how long until the conflict?
* **PROGRESS** - how far to the goal, how far off the lane, and can it see the
  destination?
* **SAFETY**   - the smallest MEASURED clearance to any body and any surface,
  which is the number that would expose a decision that looked good and went
  badly.

THE PREDICTED AND THE MEASURED ARE NEVER MIXED
------------------------------------------------
The DECISION panel shows PREDICTED clearances - what the planner promised.  The
SAFETY panel shows MEASURED ones - what actually happened.  Keeping them in
separate panels with different labels is deliberate: the acceptance gate exists
to compare them, and an overlay that blurred the two would make the comparison
impossible to check by eye.
"""

from __future__ import annotations

from hud_style import (
    ACCENT,
    BAD,
    DIM,
    F09,
    F10,
    F13,
    GOAL_INK,
    GOOD,
    INK,
    LEFT_INK,
    PRED_INK,
    RIGHT_INK,
    STATE_CAPTION,
    STATE_COLORS,
    WARN,
    bar,
    clearance_ink,
    fit,
    kind_color,
    panel,
    side_ink,
    span_bar,
    text_w,
    title,
)
from slalom_states import SAFE_CLEARANCE_M


def draw_state(draw, box, record) -> None:
    """What it is doing, and whether it is moving at all."""
    panel(draw, box)
    title(draw, box, "STATE")
    x, y = box[0] + 8, box[1] + 20
    state = record["state"]
    draw.text((x, y), state, font=F13, fill=STATE_COLORS.get(state, INK))
    y += 19
    draw.text((x, y), fit(draw, STATE_CAPTION.get(state, ""), F09,
                          box[2] - box[0] - 16), font=F09, fill=DIM)
    y += 15

    command = record["command_peak"]
    moving = command > 0.0
    draw.text((x, y), "command", font=F09, fill=DIM)
    draw.text((x + 62, y), f"{command:.2f}", font=F10,
              fill=GOOD if moving else WARN)
    # THE EXACT ZERO IS THE CLAIM, so it is spelled out rather than implied by
    # a bar that would look the same at 0.00 and 0.004.
    draw.text((x + 100, y), "walking" if moving else "EXACTLY ZERO",
              font=F09, fill=GOOD if moving else WARN)
    y += 14
    draw.text((x, y), f"t {record['t']:6.2f} s", font=F09, fill=DIM)
    draw.text((x + 84, y), f"z {record['trunk_z']:.4f} m", font=F09, fill=DIM)
    y += 13
    vx, _, wz = record["command"]
    draw.text((x, y), f"vx {vx:+.2f}   wz {wz:+.2f}   vy 0.00 (no strafe)",
              font=F09, fill=DIM)
    if record.get("interlock_blocked"):
        y += 13
        draw.text((x, y), fit(draw, "HELD: " + record["interlock_reason"], F09,
                              box[2] - box[0] - 16), font=F09, fill=BAD)


def draw_decision(draw, box, record) -> None:
    """BOTH corridors, always: the one taken and the one refused.

    A panel that showed only the winner could not distinguish a decision from a
    habit, which is exactly what this behavior has to demonstrate.
    """
    panel(draw, box)
    title(draw, box, "DECISION   predicted, both hands")
    x, y = box[0] + 8, box[1] + 19
    width = box[2] - box[0] - 16

    chosen_side = record["decision_side"]
    chosen = record["chosen_clearance_m"]
    rejected_side = record["rejected_side"]
    rejected = record["rejected_clearance_m"]

    if chosen_side == "wait":
        draw.text((x, y), "NEITHER SIDE SAFE", font=F10, fill=WARN)
        draw.text((x + 132, y), "-> WAIT", font=F10, fill=WARN)
    elif chosen_side:
        draw.text((x, y), "chose", font=F09, fill=DIM)
        draw.text((x + 44, y), chosen_side.upper(), font=F13,
                  fill=side_ink(chosen_side))
        draw.text((x + 108, y), f"{chosen:+.3f} m predicted", font=F09,
                  fill=clearance_ink(chosen, SAFE_CLEARANCE_M))
    else:
        draw.text((x, y), "no threat: straight to the goal", font=F09,
                  fill=DIM)
    y += 18

    # The two corridors on one scale, so the comparison is visual rather than
    # arithmetic.  The safety bar is drawn as the acceptance band's left edge.
    for side, value in (("left", chosen if chosen_side == "left" else
                         (rejected if rejected_side == "left" else None)),
                        ("right", chosen if chosen_side == "right" else
                         (rejected if rejected_side == "right" else None))):
        draw.text((x, y), side, font=F09, fill=side_ink(side))
        if value is None:
            draw.text((x + 44, y), "-", font=F09, fill=DIM)
        else:
            span_bar(draw, (x + 44, y + 2, box[2] - 62, y + 10),
                     -0.4, 1.2, value, side_ink(side),
                     bands=((SAFE_CLEARANCE_M, 1.2),), marks=((0.0, BAD),))
            draw.text((box[2] - 58, y), f"{value:+.3f}", font=F09,
                      fill=clearance_ink(value, SAFE_CLEARANCE_M))
        y += 15

    draw.text((x, y), f"planner bar {SAFE_CLEARANCE_M:.2f} m", font=F09,
              fill=DIM)
    passes = record["passes_completed"]
    sides = "".join(s[0].upper() for s in record["pass_sides"])
    draw.text((x + 128, y), f"passes {passes}  {sides}", font=F09, fill=INK)


def draw_threat(draw, box, record) -> None:
    """Who is crossing, how far, and how long until the predicted conflict."""
    panel(draw, box)
    title(draw, box, "THREAT   predicted from measured velocity")
    x, y = box[0] + 8, box[1] + 19
    threat = record["threat"]

    if not threat:
        draw.text((x, y), "nobody predicted to cross the line", font=F09,
                  fill=GOOD)
        y += 14
    else:
        kind = record["actor_kind"].get(threat, "")
        draw.text((x, y), threat, font=F10, fill=kind_color(kind))
        draw.text((x + 62, y), kind, font=F09, fill=DIM)
        receding = record["threat_receding"]
        draw.text((x + 128, y), "receding" if receding else "CLOSING",
                  font=F09, fill=GOOD if receding else WARN)
        y += 14
        range_m = record["threat_range_m"]
        ttc = record["threat_ttc_s"]
        if range_m is not None:
            draw.text((x, y), f"range {range_m:5.2f} m", font=F09, fill=DIM)
        if ttc is not None:
            draw.text((x + 108, y), f"worst at +{ttc:.1f} s", font=F09,
                      fill=PRED_INK)
        y += 14

    watching = record["subject"]
    visible = record["subject_visible"]
    draw.text((x, y), "watching", font=F09, fill=DIM)
    draw.text((x + 62, y), watching, font=F09,
              fill=GOAL_INK if watching == "goal" else INK)
    draw.text((x + 132, y), "SEEN" if visible else "not seen", font=F09,
              fill=GOOD if visible else BAD)
    y += 13
    in_lane = record["bodies_in_lane"]
    draw.text((x, y), "in the lane ahead", font=F09, fill=DIM)
    draw.text((x + 124, y), ", ".join(in_lane) if in_lane else "clear",
              font=F09, fill=WARN if in_lane else GOOD)


def draw_progress(draw, box, record) -> None:
    """How far to the goal, how far off the lane, and can it see the band."""
    panel(draw, box)
    title(draw, box, "PROGRESS   toward the arrival band")
    x, y = box[0] + 8, box[1] + 19
    width = box[2] - box[0] - 16

    remaining = record["goal_remaining_m"]
    draw.text((x, y), "to goal", font=F09, fill=DIM)
    bar(draw, (x + 62, y + 3, x + 62 + int(width * 0.42), y + 9),
        1.0 - min(remaining / 7.8, 1.0), GOAL_INK)
    draw.text((x + 62 + int(width * 0.42) + 6, y), f"{remaining:5.2f} m",
              font=F09, fill=GOAL_INK)
    y += 14

    offset = record["lane_offset_m"]
    draw.text((x, y), "off lane", font=F09, fill=DIM)
    span_bar(draw, (x + 62, y + 2, box[2] - 62, y + 10), -0.7, 0.7, offset,
             LEFT_INK if offset > 0 else RIGHT_INK, marks=((0.0, INK),))
    draw.text((box[2] - 58, y), f"{offset:+.2f} m", font=F09,
              fill=LEFT_INK if offset > 0 else RIGHT_INK)
    y += 15

    goal_seen = record["goal_visible"]
    draw.text((x, y), "goal in the head camera", font=F09, fill=DIM)
    draw.text((x + 152, y), "SEEN" if goal_seen else "no", font=F09,
              fill=GOOD if goal_seen else DIM)


def draw_safety(draw, box, record) -> None:
    """The smallest MEASURED clearance to any body and any surface.

    Measured, not predicted: this is the panel that would expose a decision that
    looked good on the prediction and went badly on the floor.
    """
    panel(draw, box)
    title(draw, box, "SAFETY   MEASURED surface clearance")
    x, y = box[0] + 8, box[1] + 19
    body = record["min_body_clearance_m"]
    scenery = record["scenery_clearance_m"]
    draw.text((x, y), "bodies", font=F09, fill=DIM)
    draw.text((x + 60, y), f"{body:+.3f} m", font=F09,
              fill=GOOD if body > 0.0 else BAD)
    draw.text((x + 132, y), record["nearest_body"], font=F09, fill=DIM)
    y += 13
    draw.text((x, y), "scenery", font=F09, fill=DIM)
    draw.text((x + 60, y), f"{scenery:+.3f} m", font=F09,
              fill=GOOD if scenery > 0.0 else BAD)
    draw.text((x + 132, y), fit(draw, record["nearest_scenery"], F09,
                                box[2] - box[0] - 146), font=F09, fill=DIM)


def draw_legend(draw, box, record) -> None:
    """What the picture's colours mean, and the disclosures.

    THE PANEL HEIGHT IS BUDGETED AGAINST THIS CONTENT, NOT CHOSEN.  The first
    preview gave it 116 px for eight rows at 13 px plus a heading, so the last
    two lines - the ones carrying the RGB-classifier disclosure - were drawn
    outside the panel and clipped against the frame.  A disclosure that cannot
    be read is not a disclosure, so the corridor swatches share one row and the
    body kinds share another, bringing the content to five rows.
    """
    panel(draw, box)
    title(draw, box, "LEGEND")
    x, y = box[0] + 8, box[1] + 20
    draw.rectangle([x, y + 2, x + 8, y + 9], fill=LEFT_INK)
    draw.text((x + 13, y), "left", font=F09, fill=DIM)
    draw.rectangle([x + 48, y + 2, x + 56, y + 9], fill=RIGHT_INK)
    draw.text((x + 61, y), "right corridor", font=F09, fill=DIM)
    draw.rectangle([x + 160, y + 2, x + 168, y + 9], fill=PRED_INK)
    draw.text((x + 173, y), "predicted", font=F09, fill=DIM)
    y += 13
    chip = x
    for kind, label in (("pedestrian", "person"), ("cart", "cart"),
                        ("box", "box")):
        draw.rectangle([chip, y + 2, chip + 8, y + 9], fill=kind_color(kind))
        draw.text((chip + 13, y), label, font=F09, fill=DIM)
        chip += 62
    y += 15
    draw.text((x, y), "traffic is SCRIPTED and never yields", font=F09,
              fill=DIM)
    y += 12
    draw.text((x, y), "identity is a MuJoCo body id, not an", font=F09,
              fill=DIM)
    y += 11
    draw.text((x, y), "RGB classifier", font=F09, fill=DIM)
