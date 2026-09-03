#!/usr/bin/env python3
"""The HUD panels: what the duck is being told, and what it is doing about it.

Four questions, one panel each, chosen because they are the four a sceptical
viewer should be able to answer from the picture alone:

* **WHO**       - which person is locked, how many others are in frame, and
  whether the camera can actually read the locked one's arm right now.
* **READING**   - what gesture is being read, how far through the confirm
  window it is, and the rule-margin proxy behind it.  A refusal shows here too.
* **COMMAND**   - the literal ``(vx, vy, wz)`` register, with an exact zero
  called out, because "it stopped" is this behavior's strongest claim.
* **ACTION**    - the progress of whatever physical thing is under way, in the
  units its gate is graded on: trunk-yaw degrees for a turn, metres along the
  pre-action heading for a reverse, surface clearance for an approach.

EVERY NUMBER SHOWN IS THE NUMBER THE GATE USES
------------------------------------------------
The safety bar plots SURFACE clearance against the standoff band, not a
centre-to-centre range - those differ by both bodies' radii, about 0.3 m here,
which is enough to draw a duck as outside a window it is correctly inside.
"""

from __future__ import annotations

from hud_style import (
    ACCENT,
    BAD,
    DIM,
    F09,
    F10,
    F11,
    GOOD,
    GRID,
    HEADING,
    INK,
    LOCKED,
    OTHER,
    READING,
    STANDOFF,
    WARN,
    ZERO,
    bar,
    fit,
    panel,
    span_bar,
    title,
    wrap,
)

STANDOFF_MIN_M = 0.45
STANDOFF_MAX_M = 0.75


def who_panel(draw, box, record: dict, thresholds: dict) -> None:
    """WHO the duck is listening to, and whether it can read them.

    The distractor count is shown deliberately: a viewer should see that other
    people were in frame the whole time and that it made no difference.
    """
    panel(draw, box)
    title(draw, box, "WHO   only one person's gestures count")
    x, y = box[0] + 10, box[1] + 22
    width = box[2] - box[0] - 20

    locked = record.get("locked", "")
    state = record.get("acquisition_state", "search")
    if locked:
        draw.text((x, y), fit(draw, f"LOCKED  {locked}", F11, width),
                  font=F11, fill=LOCKED)
    else:
        draw.text((x, y), f"SEARCHING  ({state})", font=F11, fill=DIM)
    y += 17

    # THE ARM GATE, which is strictly harder than the body gate.
    visible = record.get("instructor_visible", False)
    readable = record.get("instructor_arm_readable", False)
    draw.text((x, y), "in camera", font=F09, fill=DIM)
    draw.text((x + 78, y), "YES" if visible else "NO", font=F09,
              fill=GOOD if visible else BAD)
    draw.text((x + 118, y), "arm readable", font=F09, fill=DIM)
    draw.text((x + 216, y), "YES" if readable else "NO", font=F09,
              fill=GOOD if readable else BAD)
    y += 15

    others = [n for n in record.get("visible_bodies", []) if n != locked]
    draw.text((x, y), fit(
        draw, f"others in frame  {len(others)}"
        + (f"   {', '.join(others)}" if others else ""), F09, width),
        font=F09, fill=OTHER)
    y += 14

    range_m = float(record.get("instructor_range_m", 0.0))
    max_range = float(thresholds.get("gesture_max_range_m", 3.2))
    draw.text((x, y), f"range {range_m:5.2f} m", font=F09, fill=INK)
    span_bar(draw, (x + 96, y + 3, box[2] - 12, y + 9), 0.0, max_range + 0.4,
             range_m, LOCKED, marks=((max_range, WARN),))


def reading_panel(draw, box, record: dict, thresholds: dict) -> None:
    """WHAT is being read, and how far through the confirm window it is.

    The bar is the honest part: a gesture is not acted on when it is recognised,
    it is acted on when it has been recognised CONTINUOUSLY for the window - so
    the viewer watches a bar fill rather than a label appear.
    """
    panel(draw, box)
    title(draw, box, "READING   sustained, or it does not count")
    x, y = box[0] + 10, box[1] + 22
    width = box[2] - box[0] - 20

    candidate = record.get("candidate_command", "")
    confirm_s = float(thresholds.get("confirm_s", 0.9))
    progress = float(record.get("confirm_progress", 0.0))
    held = float(record.get("candidate_held_s", 0.0))

    if record.get("detector_suspended"):
        draw.text((x, y), "not reading: carrying out a command", font=F10,
                  fill=DIM)
        y += 17
    elif candidate:
        draw.text((x, y), fit(draw, candidate, F11, width), font=F11,
                  fill=READING)
        y += 17
        bar(draw, (x, y, box[2] - 12, y + 7), progress,
            GOOD if progress >= 1.0 else READING)
        y += 11
        draw.text((x, y), f"held {held:4.2f} / {confirm_s:.2f} s", font=F09,
                  fill=DIM)
        conf = float(record.get("candidate_confidence", 0.0))
        draw.text((x + 132, y), f"margin {conf:4.2f}", font=F09, fill=DIM)
        y += 14
    else:
        draw.text((x, y), "nothing recognised", font=F10, fill=DIM)
        y += 17

    reading = record.get("reading") or {}
    rule = str(reading.get("rule", ""))
    if rule:
        accepted = bool(reading.get("template"))
        for line in wrap(draw, rule, F09, width, max_lines=2):
            draw.text((x, y), line, font=F09,
                      fill=DIM if accepted else BAD)
            y += 12

    # The proxy label, stated in the picture and not only in the README.  Drawn
    # at a fixed offset from the panel BOTTOM so it cannot collide with the
    # wrapped rule text above it however long that runs.
    draw.text((x, box[3] - 14), "rule-margin proxy, not a probability",
              font=F09, fill=DIM)


def command_panel(draw, box, record: dict) -> None:
    """The literal command register, with an exact zero called out.

    This behavior's strongest claim is that certain states emit an EXACT zero,
    so the HUD shows the register itself rather than a speed derived from it.
    """
    panel(draw, box)
    title(draw, box, "COMMAND   the register, verbatim")
    x, y = box[0] + 10, box[1] + 22

    command = record.get("command", [0.0, 0.0, 0.0])
    vx, vy, wz = (float(command[0]), float(command[1]), float(command[2]))
    peak = float(record.get("command_peak", 0.0))

    if peak == 0.0:
        draw.text((x, y), "EXACT ZERO", font=F11, fill=ZERO)
        draw.text((x + 108, y), "(0.000, 0.000, 0.000)", font=F09, fill=DIM)
    else:
        draw.text((x, y), f"vx {vx:+.3f}", font=F11, fill=INK)
        draw.text((x + 92, y), f"wz {wz:+.3f}", font=F11, fill=HEADING)
    y += 19

    # vy is shown ALWAYS, because "no lateral command, ever" is a gate.
    draw.text((x, y), f"vy {vy:+.3f}", font=F09,
              fill=DIM if vy == 0.0 else BAD)
    speed = float(record.get("speed_mps", 0.0))
    draw.text((x + 92, y), f"measured {speed:5.3f} m/s", font=F09, fill=DIM)
    y += 15

    settled = record.get("settled", False)
    draw.text((x, y), "SETTLED" if settled else "moving", font=F09,
              fill=ZERO if settled else DIM)
    if record.get("interlock_blocked"):
        draw.text((x + 92, y), "INTERLOCK HOLD", font=F09, fill=BAD)


def action_panel(draw, box, record: dict, thresholds: dict) -> None:
    """The physical action under way, in the units its gate is graded on."""
    panel(draw, box)
    state = record.get("state", "")
    title(draw, box, "ACTION   measured, not commanded")
    x, y = box[0] + 10, box[1] + 22
    width = box[2] - box[0] - 20

    if state in ("EXECUTE_TURN_LEFT", "EXECUTE_TURN_RIGHT"):
        target = float(thresholds.get("turn_target_deg", 72.0))
        wanted = target if state.endswith("LEFT") else -target
        turned = float(record.get("yaw_delta_deg", 0.0))
        draw.text((x, y), f"trunk yaw {turned:+6.1f} deg", font=F11, fill=INK)
        draw.text((x + 168, y), f"target {wanted:+.0f}", font=F09, fill=DIM)
        y += 18
        span_bar(draw, (x, y, box[2] - 12, y + 8), -target - 20.0,
                 target + 20.0, turned, ACCENT,
                 marks=((wanted, GOOD), (0.0, GRID)))
        y += 14
        draw.text((x, y), "the yaw the trunk ACTUALLY turned through",
                  font=F09, fill=DIM)

    elif state == "EXECUTE_BACK_UP":
        target = float(thresholds.get("back_up_target_m", 0.40))
        back = float(record.get("back_along_heading_m", 0.0))
        draw.text((x, y), f"back {back:+.3f} m", font=F11, fill=INK)
        draw.text((x + 140, y), f"target {target:.2f}", font=F09, fill=DIM)
        y += 18
        span_bar(draw, (x, y, box[2] - 12, y + 8), -0.15, target + 0.20,
                 back, ACCENT, marks=((target, GOOD),))
        y += 14
        draw.text((x, y), fit(
            draw, "displacement along the PRE-ACTION heading", F09, width),
            font=F09, fill=DIM)

    elif state == "EXECUTE_STOP":
        hold = float(record.get("stop_hold_s", 0.0))
        need = float(thresholds.get("stop_hold_s", 2.0))
        draw.text((x, y), f"held still {hold:4.2f} / {need:.2f} s", font=F11,
                  fill=ZERO)
        y += 18
        bar(draw, (x, y, box[2] - 12, y + 8), hold / max(need, 1e-9), ZERO)
        y += 14
        draw.text((x, y), fit(
            draw, "accrues only below the measured settled speed", F09, width),
            font=F09, fill=DIM)

    else:
        # The default view is the SAFETY one: surface clearance against the
        # standoff band, in the units the gate grades.
        clearance = float(record.get("min_clearance_m", 0.0))
        draw.text((x, y), f"nearest person {clearance:5.3f} m", font=F11,
                  fill=INK)
        y += 18
        span_bar(draw, (x, y, box[2] - 12, y + 8), 0.0, 2.0, clearance,
                 GOOD if clearance >= STANDOFF_MIN_M else BAD,
                 bands=((STANDOFF_MIN_M, STANDOFF_MAX_M),))
        y += 14
        draw.text((x, y), fit(
            draw, "SURFACE clearance; green is the safe standoff band",
            F09, width), font=F09, fill=DIM)


def sequence_panel(draw, box, record: dict, expected: list[str]) -> None:
    """The required command sequence, with what has been accepted so far.

    Shown as the ORDER, because "in the right order" is a gate and a viewer
    should be able to watch it being satisfied rather than take it on trust.
    """
    panel(draw, box)
    title(draw, box, "SEQUENCE   six commands, in order")
    accepted = list(record.get("accepted_commands", []))
    x, y = box[0] + 10, box[1] + 22
    for index, command in enumerate(expected):
        done = index < len(accepted)
        current = index == len(accepted)
        ink = GOOD if done else (READING if current else DIM)
        mark = "*" if done else (">" if current else " ")
        draw.text((x, y), f"{mark} {index + 1}. {command}", font=F10, fill=ink)
        if done and accepted[index] != command:
            draw.text((x + 150, y), f"got {accepted[index]}", font=F09,
                      fill=BAD)
        y += 14


def refusals_panel(draw, box, record: dict, refusals: list[dict]) -> None:
    """What the duck was shown and REFUSED, which is half the behavior.

    A robot that only ever says yes has not demonstrated judgment, so the two
    refusals - a stranger's perfectly good gesture and an ambiguous one from the
    right person - get their own panel rather than being absences.
    """
    panel(draw, box)
    title(draw, box, "REFUSED   what it did NOT obey")
    x, y = box[0] + 10, box[1] + 22
    width = box[2] - box[0] - 20

    now = float(record.get("t", 0.0))
    # TWO entries, not three: the panel is 80 px tall and each entry costs 26,
    # so a third would be clipped mid-line.  The most recent are the useful
    # ones, so the list is taken from the end and drawn newest first.
    recent = [r for r in refusals if r["t"] <= now][-2:]
    if not recent:
        draw.text((x, y), "nothing refused yet", font=F09, fill=DIM)
        return
    for entry in reversed(recent):
        who = entry.get("person", "")
        ink = BAD if who == record.get("instructor") else OTHER
        draw.text((x, y), fit(
            draw, f"{entry['t']:6.2f}s  {who}", F09, width), font=F09, fill=ink)
        y += 12
        for line in wrap(draw, str(entry.get("reason", "")), F09, width,
                         max_lines=1):
            draw.text((x + 8, y), line, font=F09, fill=DIM)
            y += 12
        y += 2
