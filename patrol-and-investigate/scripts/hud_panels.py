#!/usr/bin/env python3
"""The HUD panels: where the duck is on its route, and what it decided.

Every panel reads the SAME per-tick record the acceptance gate reads, so the
overlay cannot show a figure the gate did not grade.  Nothing here computes
anything about the behavior; it only lays out quantities that already exist.

FIVE PANELS, EACH ANSWERING ONE QUESTION A VIEWER WILL ASK
------------------------------------------------------------
* **STATE**    - what is it doing, in plain English, and is it moving?
* **PATROL**   - which checkpoint is it on, how many are done, and in what
  order?  This is the panel that makes the route a sequence rather than a walk.
* **MEMORY**   - IS THE PATROL INTERRUPTED, and if so what is it holding?  It is
  its own panel because remembering the route is the thing this behavior exists
  to demonstrate, and it must be visible for the whole diversion rather than
  inferable afterwards.
* **TARGET**   - what is it looking at, what did it decide, on what RULE, and at
  what confidence?
* **SAFETY**   - the MEASURED standoff, the MEASURED clearance, and the MEASURED
  distance to the restricted zone.

THE STANDOFF BAND IS DRAWN AS A BAND
--------------------------------------
The safety claim is that the duck stopped INSIDE a window, so the window is
drawn as a bright region with the live value as a tick inside it.  A pair of
numbers would be true and unreadable.
"""

from __future__ import annotations

from hud_style import (
    ACCENT,
    BAD,
    CHECKPOINT,
    DIM,
    F09,
    F10,
    F11,
    F13,
    GOOD,
    INK,
    MEMORY,
    ROUTE,
    STANDOFF,
    STATE_CAPTION,
    STATE_COLORS,
    WARN,
    ZONE,
    bar,
    fit,
    kind_color,
    panel,
    span_bar,
    text_w,
    title,
    verdict_ink,
    wrap,
)
from patrol_facility import CHECKPOINT_NAMES
from patrol_states import STANDOFF_MAX_M, STANDOFF_MIN_M


def draw_state(draw, box, record) -> None:
    """What it is doing, and whether it is moving at all."""
    panel(draw, box)
    title(draw, box, "STATE")
    x, y = box[0] + 8, box[1] + 20
    state = record["state"]
    draw.text((x, y), state, font=F13, fill=STATE_COLORS.get(state, INK))
    y += 19
    # WRAPPED, NOT ELLIPSISED.  The caption is the one line that tells a viewer
    # who has never read the README what the robot is doing, and the longest of
    # them - "holding several viewing angles, standing still" - was being cut to
    # "holding several viewing angles, standing ...", which loses the word that
    # matters most.
    for line in wrap(draw, STATE_CAPTION.get(state, ""), F09,
                     box[2] - box[0] - 16, max_lines=2):
        draw.text((x, y), line, font=F09, fill=DIM)
        y += 11
    y += 4

    command = record["command_peak"]
    moving = command > 0.0
    draw.text((x, y), "command", font=F09, fill=DIM)
    draw.text((x + 62, y), f"{command:.2f}", font=F10,
              fill=GOOD if moving else WARN)
    # THE EXACT ZERO IS THE CLAIM, so it is spelled out rather than implied by
    # a bar that would look the same at 0.00 and 0.004.
    draw.text((x + 104, y), "walking" if moving else "EXACT ZERO",
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


def draw_patrol(draw, box, record) -> None:
    """The checkpoint sequence, and how far through it the duck is."""
    panel(draw, box)
    title(draw, box, "PATROL   five checkpoints, in order")
    x, y = box[0] + 8, box[1] + 19
    width = box[2] - box[0] - 16

    done = record["completed"]
    draw.text((x, y), "progress", font=F09, fill=DIM)
    bar(draw, (x + 62, y + 3, x + 62 + int(width * 0.44), y + 9),
        done / 5.0, CHECKPOINT)
    draw.text((x + 62 + int(width * 0.44) + 6, y), f"{done}/5", font=F09,
              fill=CHECKPOINT)
    y += 16

    # Every checkpoint as a chip, in the declared order: done, current, pending.
    completed = set(record["completed_names"])
    chip = x
    for name in CHECKPOINT_NAMES:
        short = name.split("-")[0][:5]
        if name in completed:
            colour, mark = CHECKPOINT, "done"
        elif name == record["target_name"]:
            colour, mark = ROUTE, "now"
        else:
            colour, mark = DIM, ""
        draw.rectangle([chip, y + 2, chip + 7, y + 9], fill=colour)
        draw.text((chip + 10, y), short, font=F09, fill=colour)
        chip += 50
    y += 15

    draw.text((x, y), "walking to", font=F09, fill=DIM)
    draw.text((x + 72, y), record["target_name"], font=F10, fill=ROUTE)
    draw.text((box[2] - 68, y), f"{record['target_remaining_m']:5.2f} m",
              font=F09, fill=DIM)


def draw_memory(draw, box, record) -> None:
    """IS THE PATROL INTERRUPTED, and what is being held?

    The panel this behavior exists for.  While an investigation runs it names
    the checkpoint the duck was walking to, the point it must come back to, and
    how far away that point currently is - so the route memory is visible on
    screen for the whole diversion rather than inferable from the outcome.
    """
    panel(draw, box)
    title(draw, box, "ROUTE MEMORY   what it is holding")
    x, y = box[0] + 8, box[1] + 19

    if not record["interrupted"]:
        draw.text((x, y), "patrol running, nothing held", font=F09, fill=DIM)
        y += 14
        count = record["interruptions"]
        draw.text((x, y), f"interruptions so far  {count}", font=F09,
                  fill=DIM)
        return

    draw.text((x, y), "INTERRUPTED", font=F10, fill=MEMORY)
    draw.text((x + 96, y), "holding its place", font=F09, fill=DIM)
    y += 15
    draw.text((x, y), "was walking to", font=F09, fill=DIM)
    draw.text((x + 92, y), record["interrupted_target"], font=F10,
              fill=MEMORY)
    y += 14
    resume = record["resume_xy"]
    if resume is not None:
        draw.text((x, y), "resume point", font=F09, fill=DIM)
        draw.text((x + 92, y), f"({resume[0]:+.2f}, {resume[1]:+.2f})",
                  font=F09, fill=MEMORY)
        y += 13
    remaining = record["resume_remaining_m"]
    if remaining is not None and remaining < 1e6:
        draw.text((x, y), "distance back", font=F09, fill=DIM)
        draw.text((x + 92, y), f"{remaining:5.2f} m", font=F09, fill=MEMORY)


def draw_target(draw, box, record) -> None:
    """What it is looking at, and what it concluded about it."""
    panel(draw, box)
    title(draw, box, "TARGET   semantic proxy, rule-margin confidence")
    x, y = box[0] + 8, box[1] + 19
    width = box[2] - box[0] - 16

    candidate = record["candidate"]
    verdict = record["candidate_verdict"]
    if not candidate:
        draw.text((x, y), "nothing under consideration", font=F09, fill=GOOD)
        y += 14
    else:
        kind = record["actor_kind"].get(candidate, "")
        draw.text((x, y), candidate, font=F10, fill=kind_color(kind))
        draw.text((x + 72, y), kind, font=F09, fill=DIM)
        if verdict:
            draw.text((x + 144, y), verdict.upper(), font=F10,
                      fill=verdict_ink(verdict))
        y += 15
        confidence = record["candidate_confidence"]
        if confidence:
            draw.text((x, y), "confidence", font=F09, fill=DIM)
            bar(draw, (x + 72, y + 3, x + 72 + int(width * 0.34), y + 9),
                confidence, verdict_ink(verdict))
            draw.text((x + 72 + int(width * 0.34) + 6, y),
                      f"{confidence:.2f}", font=F09, fill=verdict_ink(verdict))
            y += 15
        for line in wrap(draw, record["candidate_rule"], F09, width,
                         max_lines=3):
            draw.text((x, y), line, font=F09, fill=DIM)
            y += 11

    watching = record["subject"]
    visible = record["subject_visible"]
    y = box[3] - 26
    draw.text((x, y), "watching", font=F09, fill=DIM)
    draw.text((x + 62, y), watching, font=F09,
              fill=ROUTE if watching == "route" else INK)
    draw.text((x + 148, y), "SEEN" if visible else "not seen", font=F09,
              fill=GOOD if visible else BAD)
    y += 13
    verdicts = record["verdict_targets"]
    if verdicts:
        chip = x
        draw.text((chip, y), "decided", font=F09, fill=DIM)
        chip += 58
        for name, decided in zip(verdicts, record["verdicts_so_far"]):
            draw.rectangle([chip, y + 2, chip + 7, y + 9],
                           fill=verdict_ink(decided))
            draw.text((chip + 10, y), name[:7], font=F09, fill=DIM)
            chip += 58


def draw_safety(draw, box, record) -> None:
    """The MEASURED standoff, clearance, and distance to the restricted zone.

    Measured, not planned: this is the panel that would expose an approach that
    looked right and went too close, or a duck that clipped the marked area.
    Every figure here is a SURFACE separation, so the standoff bar and the
    bodies line are in the same units as each other and as the acceptance gate.
    """
    panel(draw, box)
    title(draw, box, "SAFETY   MEASURED, every tick")
    x, y = box[0] + 8, box[1] + 19

    # The standoff band, drawn as a band - in SURFACE clearance, which is the
    # quantity the band is defined in and the gate grades.  Drawing the
    # centre-to-centre range here instead put the tick outside a window the duck
    # was correctly inside, because the two differ by both bodies' radii.
    clearance = record["target_clearance_m"]
    draw.text((x, y), "standoff", font=F09, fill=DIM)
    if clearance is not None and clearance < 1.4:
        span_bar(draw, (x + 62, y + 2, box[2] - 66, y + 10), 0.0, 1.2,
                 clearance, STANDOFF,
                 bands=((STANDOFF_MIN_M, STANDOFF_MAX_M),))
        inside = STANDOFF_MIN_M <= clearance <= STANDOFF_MAX_M
        draw.text((box[2] - 62, y), f"{clearance:5.2f}", font=F09,
                  fill=GOOD if inside else DIM)
    else:
        draw.text((x + 62, y), "no target", font=F09, fill=DIM)
    y += 15

    body = record["min_body_clearance_m"]
    draw.text((x, y), "bodies", font=F09, fill=DIM)
    draw.text((x + 62, y), f"{body:+.3f} m", font=F09,
              fill=GOOD if body > 0.0 else BAD)
    draw.text((x + 138, y), record["nearest_body"], font=F09, fill=DIM)
    y += 13

    zone = record["zone_gap_m"]
    draw.text((x, y), "zone", font=F09, fill=DIM)
    draw.text((x + 62, y), f"{zone:+.3f} m", font=F09,
              fill=GOOD if zone > 0.0 else BAD)
    draw.text((x + 138, y), "outside" if zone > 0.0 else "INSIDE", font=F09,
              fill=ZONE if zone > 0.0 else BAD)
    y += 13
    scenery = record["scenery_clearance_m"]
    draw.text((x, y), "scenery", font=F09, fill=DIM)
    draw.text((x + 62, y), f"{scenery:+.3f} m", font=F09,
              fill=GOOD if scenery > 0.0 else BAD)
    draw.text((x + 138, y), fit(draw, record["nearest_scenery"], F09,
                                box[2] - box[0] - 152), font=F09, fill=DIM)


def draw_legend(draw, box, record) -> None:
    """What the picture's colours mean, and the disclosures.

    THE PANEL HEIGHT IS BUDGETED AGAINST THIS CONTENT, NOT CHOSEN.  A
    disclosure that is drawn outside its panel and clipped against the frame is
    not a disclosure, so the swatches share rows and the content comes to five.
    """
    panel(draw, box)
    title(draw, box, "LEGEND")
    x, y = box[0] + 8, box[1] + 20
    draw.rectangle([x, y + 2, x + 8, y + 9], fill=ROUTE)
    draw.text((x + 13, y), "circuit", font=F09, fill=DIM)
    draw.rectangle([x + 66, y + 2, x + 74, y + 9], fill=MEMORY)
    draw.text((x + 79, y), "remembered route", font=F09, fill=DIM)
    y += 13
    draw.rectangle([x, y + 2, x + 8, y + 9], fill=STANDOFF)
    draw.text((x + 13, y), "approach", font=F09, fill=DIM)
    draw.rectangle([x + 66, y + 2, x + 74, y + 9], fill=ZONE)
    draw.text((x + 79, y), "restricted zone", font=F09, fill=DIM)
    y += 15
    draw.text((x, y), "staff walk scripted routes and never", font=F09,
              fill=DIM)
    y += 11
    draw.text((x, y), "yield; identity is a MuJoCo body id,", font=F09,
              fill=DIM)
    y += 11
    draw.text((x, y), "not an RGB classifier", font=F09, fill=DIM)
