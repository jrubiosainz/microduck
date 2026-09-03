#!/usr/bin/env python3
"""The HUD panels: what the duck measured this tick, drawn as numbers and bars.

Every panel reads the SAME per-tick record the acceptance gate reads, so the
overlay cannot show a figure the gate did not grade.  Nothing here computes
anything about the behavior; it only lays out quantities that already exist.

FIVE PANELS, EACH ANSWERING ONE QUESTION A VIEWER WILL ASK
------------------------------------------------------------
* **STATE** - what is it doing, in plain English, and is it moving?
* **DOORS** - which apertures are open, how wide, and is the duck in one?
* **TRAFFIC** - who is still in the way, and who is the head watching?
* **ORDER** - where is the duck relative to the guardian on the shared path?
* **SAFETY** - the smallest measured clearance to any person and any surface.

The ZONE readouts live in the door panel rather than in their own, because a
threshold only means something next to the door it belongs to.
"""

from __future__ import annotations

from hud_style import (
    ACCENT,
    BAD,
    DIM,
    DOOR_INK,
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
    door_ink,
    fit,
    panel,
    role_color,
    span_bar,
    text_w,
    title,
)


def draw_state(draw, box, record) -> None:
    """What it is doing, and whether it is moving at all."""
    panel(draw, box)
    title(draw, box, "STATE")
    x, y = box[0] + 8, box[1] + 20
    state = record["state"]
    ink = STATE_COLORS.get(state, INK)
    draw.text((x, y), state, font=F13, fill=ink)
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
    draw.text((x + 100, y), "EXACTLY ZERO" if not moving else "walking",
              font=F09, fill=WARN if not moving else GOOD)
    y += 14
    draw.text((x, y), f"t {record['t']:6.2f} s", font=F09, fill=DIM)
    draw.text((x + 84, y), f"z {record['trunk_z']:.4f} m", font=F09, fill=DIM)
    if record.get("careful"):
        y += 13
        draw.text((x, y), "careful pace: inside an aperture band",
                  font=F09, fill=ACCENT)
    if record.get("interlock_blocked"):
        y += 13
        draw.text((x, y), fit(draw, "HELD: " + record["interlock_reason"], F09,
                              box[2] - box[0] - 16), font=F09, fill=BAD)


def draw_doors(draw, box, record) -> None:
    """Every aperture's measured open fraction, and where the duck is."""
    panel(draw, box)
    title(draw, box, "DOORS   open fraction and clear gap")
    x, y = box[0] + 8, box[1] + 19
    width = box[2] - box[0] - 16
    labels = {"concourse_door": "concourse", "lift_front": "lift front",
              "lift_rear": "lift rear"}
    for name, label in labels.items():
        fraction = record["door_fraction"][name]
        gap = record["door_gap_m"][name]
        occupancy = record["aperture_occupancy"][name]
        draw.text((x, y), label, font=F09, fill=DIM)
        bar(draw, (x + 62, y + 3, x + 62 + int(width * 0.30), y + 9),
            fraction, door_ink(fraction))
        draw.text((x + 62 + int(width * 0.30) + 6, y),
                  f"{fraction:4.2f} {gap:.2f}m", font=F09,
                  fill=door_ink(fraction))
        if occupancy["duck"]:
            draw.text((x + width - 30, y), "DUCK", font=F09, fill=GOOD)
        elif occupancy["others"]:
            draw.text((x + width - 30, y), occupancy["others"][0][:4],
                      font=F09, fill=BAD)
        y += 13

    y += 3
    depths = record["zone_depth_m"]
    door_zone = depths.get("concourse_door_threshold", 0.0)
    passage = depths.get("lift_front_passage", 0.0)
    draw.text((x, y), "threshold", font=F09, fill=DIM)
    draw.text((x + 62, y), "clear" if door_zone == 0.0
              else f"in {door_zone:.2f} m", font=F09,
              fill=GOOD if door_zone == 0.0 else WARN)
    draw.text((x + 128, y), "passage", font=F09, fill=DIM)
    draw.text((x + 186, y), "clear" if passage == 0.0
              else f"in {passage:.2f} m", font=F09,
              fill=GOOD if passage == 0.0 else WARN)


def draw_traffic(draw, box, record) -> None:
    """Who is still in the way, and who the head is watching."""
    panel(draw, box)
    title(draw, box, "TRAFFIC   measured, not scripted")
    x, y = box[0] + 8, box[1] + 19

    subject = record["subject"]
    visible = record["subject_visible"]
    draw.text((x, y), "watching", font=F09, fill=DIM)
    draw.text((x + 58, y), subject, font=F10, fill=role_color(
        record["subject_role"]))
    draw.text((x + 118, y), "SEEN" if visible else "not seen", font=F09,
              fill=GOOD if visible else BAD)
    y += 13
    draw.text((x, y), f"range {record['subject_range_m']:.2f} m", font=F09,
              fill=DIM)
    if not visible and record["subject_blocked_by"]:
        draw.text((x + 92, y), fit(draw, "behind "
                                   + record["subject_blocked_by"], F09,
                                   box[2] - box[0] - 108), font=F09, fill=DIM)
    elif not record["los_available"]:
        draw.text((x + 92, y), fit(draw, "no LOS: "
                                   + record["los_blocked_by"], F09,
                                   box[2] - box[0] - 108), font=F09, fill=DIM)
    y += 14

    pending = record["exiters_pending"]
    draw.text((x, y), "exiters to clear", font=F09, fill=DIM)
    draw.text((x + 128, y), str(pending), font=F10,
              fill=GOOD if pending == 0 else WARN)
    y += 13
    out = record["occupants_exited"]
    inside = record["occupants_in_cabin"]
    draw.text((x, y), "occupants out / in", font=F09, fill=DIM)
    draw.text((x + 128, y), f"{out} / {inside}", font=F10,
              fill=GOOD if inside == 0 else WARN)


def draw_order(draw, box, record) -> None:
    """Where the duck is relative to the guardian, along the shared path."""
    panel(draw, box)
    title(draw, box, "ORDER   gap along the duck's own route")
    x, y = box[0] + 8, box[1] + 19
    gap = record["guardian_gap_m"]
    ahead = gap > 0.0
    draw.text((x, y), "nadia", font=F09, fill=GUARDIAN_IN_HUD)
    draw.text((x + 48, y), "AHEAD" if ahead else "BEHIND - OVERTAKEN",
              font=F10, fill=GOOD if ahead else BAD)
    y += 15
    # Negative is the failure, so the scale shows it: a viewer can see the tick
    # sitting safely to the right of zero for the whole run.
    span_bar(draw, (x, y, box[2] - 10, y + 9), -1.0, 4.0, gap,
             GOOD if ahead else BAD, bands=((0.05, 4.0),),
             marks=((0.0, BAD),))
    y += 14
    draw.text((x, y), f"{gap:+.2f} m", font=F09, fill=GOOD if ahead else BAD)
    flags = []
    if record["guardian_through_door"]:
        flags.append("through door")
    if record["guardian_inside_cabin"]:
        flags.append("in cabin")
    if record["guardian_through_rear"]:
        flags.append("out at target")
    if flags:
        draw.text((x + 62, y), fit(draw, " / ".join(flags), F09,
                                   box[2] - box[0] - 76), font=F09, fill=DIM)


def draw_safety(draw, box, record) -> None:
    """The smallest measured clearance to any person and any surface."""
    panel(draw, box)
    title(draw, box, "SAFETY   measured surface clearance")
    x, y = box[0] + 8, box[1] + 19
    person = record["min_person_clearance_m"]
    scenery = record["scenery_clearance_m"]
    draw.text((x, y), "people", font=F09, fill=DIM)
    draw.text((x + 54, y), f"{person:+.3f} m", font=F09,
              fill=GOOD if person > 0.0 else BAD)
    draw.text((x + 122, y), record["nearest_person"], font=F09, fill=DIM)
    y += 13
    draw.text((x, y), "scenery", font=F09, fill=DIM)
    draw.text((x + 54, y), f"{scenery:+.3f} m", font=F09,
              fill=GOOD if scenery > 0.0 else BAD)
    draw.text((x + 122, y), fit(draw, record["nearest_scenery"], F09,
                                box[2] - box[0] - 136), font=F09, fill=DIM)
    y += 13
    if record["inside_cabin"]:
        margin = record["cabin_margin_m"]
        draw.text((x, y), "in the cabin", font=F09, fill=GOOD)
        draw.text((x + 90, y), f"{margin:+.3f} m to a face", font=F09,
                  fill=GOOD if margin > 0.0 else BAD)


def draw_legend(draw, box, record) -> None:
    """What the picture's colours mean, and the two disclosures."""
    panel(draw, box)
    title(draw, box, "LEGEND")
    x, y = box[0] + 8, box[1] + 20
    for role, label in (("guardian", "nadia - the guardian"),
                        ("door_exiter", "coming out of the door"),
                        ("occupant", "leaving the lift"),
                        ("background", "others in the building")):
        draw.rectangle([x, y + 2, x + 8, y + 9], fill=role_color(role))
        draw.text((x + 14, y), label, font=F09, fill=DIM)
        y += 13
    y += 3
    draw.text((x, y), "doors are SCRIPTED proxies", font=F09, fill=DIM)
    y += 12
    draw.text((x, y), "identity is a MuJoCo body id,", font=F09, fill=DIM)
    y += 11
    draw.text((x, y), "not an RGB classifier", font=F09, fill=DIM)
