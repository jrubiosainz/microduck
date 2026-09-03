#!/usr/bin/env python3
"""The text HUD panels: identity, status, and the refusal log.

THE IDENTITY PANEL IS THE POINT OF THE WHOLE OVERLAY
-----------------------------------------------------
The behavior's claim is not "the duck walked back to a person".  It is "the duck
declined three people who were in front of it and accepted only its guardian".
A viewer can only grade that if, at the moment of each refusal, they can see:

* WHO is being looked at,
* the score, against both thresholds drawn in place,
* WHICH feature failed, per term, and
* the verdict in words.

So the candidate panel draws the four descriptor terms as individual bars with
the offending term picked out in red, and prints the reason sentence the
identity layer itself produced.  Nothing here is re-derived for the picture: it
is the same ``sighting`` record the acceptance gate grades.

The confirmation timer is drawn as a bar against its own threshold because
reacquisition is a DURATION, not an instant, and a still frame otherwise cannot
distinguish "seen for 0.1 s" from "confirmed for 0.9 s".
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
    INK,
    STATE_CAPTION,
    STATE_COLORS,
    WARN,
    bar,
    fit,
    panel,
    text_w,
    title,
    verdict_color,
    wrap,
)
from lost_cast import BY_NAME, GUARDIAN
from lost_constants import (
    ACCEPT_SCORE,
    CANDIDATE_SCORE,
    REACQUIRE_CONFIRM_S,
)
from hud_style import GUARDIAN_IN_HUD

# The four descriptor terms, in the order the identity layer weights them.
TERMS = ("shirt", "stature", "cap", "satchel")


def draw_status(draw, box, record) -> None:
    """State, plain-English caption, command, and the stationary claim."""
    x0, y0, x1, y1 = box
    panel(draw, box)
    title(draw, box, "STATE")

    state = record["state"]
    color = STATE_COLORS.get(state, DIM)
    draw.text((x0 + 8, y0 + 20), state, font=F13, fill=color)
    draw.text((x0 + 8, y0 + 40), STATE_CAPTION.get(state, ""), font=F10,
              fill=DIM)

    vx, vy, wz = record["command"]
    draw.text((x0 + 8, y0 + 60),
              f"cmd  vx {vx:+.3f}  vy {vy:+.3f}  wz {wz:+.3f}", font=F10,
              fill=INK)
    # The behavior's central safety claim, asserted per frame against the
    # command actually sent to the policy this tick.
    moving = record["command_peak"] > 0.0
    if state in ("LOST", "STOP", "SEARCH_SWEEP", "CANDIDATE", "REJECT",
                 "REACQUIRED", "SAFE"):
        draw.text((x0 + 8, y0 + 76),
                  "STATIONARY - command is exactly zero" if not moving
                  else "MOVING WHILE LOST", font=F10,
                  fill=GOOD if not moving else BAD)
    else:
        draw.text((x0 + 8, y0 + 76), f"walking   path {record['path_m']:.2f} m",
                  font=F10, fill=DIM)

    draw.text((x0 + 8, y0 + 96),
              f"trunk z  {record['trunk_z']:.4f} m   (min {record['min_trunk_z']:.4f})",
              font=F10, fill=INK)
    draw.text((x0 + 8, y0 + 112),
              fit(draw, f"clearance  person {record['min_person_clearance_m']:.3f} m"
                  f"  scenery {record['scenery_clearance_m']:.3f} m",
                  F09, x1 - x0 - 16), font=F09, fill=DIM)
    # The head line is split in two: the joint angle and the sweep flag would
    # not fit on one line at this panel width, and the flag is the half that
    # must never be truncated.
    draw.text((x0 + 8, y0 + 126),
              f"head {record['view_yaw_deg']:+.1f} /"
              f" joint {record['gaze_yaw_deg']:+.1f} deg",
              font=F09, fill=DIM)
    flag = "SWEEPING" if record["scanning"] else "tracking"
    draw.text((x1 - 8 - text_w(draw, flag, F09), y0 + 126), flag, font=F09,
              fill=WARN if record["scanning"] else ACCENT)


def draw_guardian(draw, box, record) -> None:
    """Where the guardian is, whether she is visible, and what blocks her."""
    x0, y0, x1, y1 = box
    panel(draw, box)
    title(draw, box, "GUARDIAN - priya")

    visible = record["guardian_visible"]
    draw.text((x0 + 8, y0 + 20),
              "IN CAMERA" if visible else "NOT VISIBLE", font=F13,
              fill=GOOD if visible else BAD)

    range_m = record["guardian_range_m"]
    draw.text((x0 + 8, y0 + 40),
              f"range {range_m:.3f} m" if range_m is not None else "range -",
              font=F10, fill=INK)

    if visible:
        count = record["guardian_sample_count"]
        draw.text((x0 + 8, y0 + 56), f"body samples seen {count}/5", font=F09,
                  fill=DIM)
        draw.text((x0 + 8, y0 + 70),
                  fit(draw, "readable: "
                      f"{', '.join(record['guardian_readable']) or '-'}",
                      F09, x1 - x0 - 16), font=F09, fill=DIM)
    else:
        blocker = record["guardian_blocked_by"] or "outside the head camera"
        draw.text((x0 + 8, y0 + 56),
                  fit(draw, f"blocked by  {blocker}", F09, x1 - x0 - 16),
                  font=F09, fill=BAD)
        draw.text((x0 + 8, y0 + 70),
                  f"unseen for {record['invisible_for_s']:.2f} s", font=F09,
                  fill=BAD)

    # Confirmation timer: a duration, drawn against its threshold.
    confirmed = record["confirmed_s"]
    draw.text((x0 + 8, y0 + 88),
              f"identity confirmed for {confirmed:.2f} s"
              f" / {REACQUIRE_CONFIRM_S:.2f} s", font=F09,
              fill=GOOD if confirmed >= REACQUIRE_CONFIRM_S else DIM)
    bar(draw, (x0 + 8, y0 + 102, x1 - 8, y0 + 108),
        confirmed / REACQUIRE_CONFIRM_S,
        GOOD if confirmed >= REACQUIRE_CONFIRM_S else ACCENT)


def _draw_terms(draw, box, sighting) -> int:
    """The four descriptor terms as bars; the failing term picked out in red.

    Returns the y coordinate after the block so the caller can continue.
    """
    x0, y, x1 = box
    penalties = sighting["penalties"]
    readable = set(sighting["readable"])
    for term in TERMS:
        label_w = text_w(draw, "stature", F09) + 4
        penalty = penalties.get(term)
        if term not in readable or penalty is None:
            draw.text((x0, y), term, font=F09, fill=DIM)
            draw.text((x0 + label_w + 4, y), "not readable", font=F09, fill=DIM)
            y += 14
            continue
        # A penalty is a MISMATCH, so the bar shows agreement: 1 - penalty.
        agreement = 1.0 - float(penalty)
        color = BAD if penalty > 0.25 else GOOD
        draw.text((x0, y), term, font=F09, fill=color)
        bar(draw, (x0 + label_w, y + 3, x1 - 46, y + 9), agreement, color)
        draw.text((x1 - 42, y), f"{agreement * 100:3.0f}%", font=F09, fill=color)
        y += 14
    return y


def draw_candidate(draw, box, record, last_sighting: dict | None = None) -> None:
    """Who the duck is looking at, its score, and why it decided what it did.

    ``last_sighting`` is PRESENTATION MEMORY supplied by the renderer.  The
    identity tracker clears its sighting the instant a candidate is refused and
    put on cooldown, so during the REJECT hold the record carries no sighting at
    all and this panel would read "no candidate in view" directly underneath the
    state word REFUSED.  Redrawing the sighting the machine actually decided on
    is the honest picture of that moment; it is never fed back into the rollout,
    and it is used only while the refused candidate is still the subject.
    """
    x0, y0, x1, y1 = box
    panel(draw, box)
    title(draw, box, "IDENTITY CHECK - semantic proxy")

    sighting = record.get("sighting")
    refused_now = record["state"] == "REJECT"
    replayed = False
    if not sighting and refused_now and last_sighting is not None \
            and last_sighting.get("name") == record.get("subject"):
        sighting, replayed = last_sighting, True

    if not sighting:
        draw.text((x0 + 8, y0 + 24), "no candidate in view", font=F11, fill=DIM)
        for index, line in enumerate(wrap(
                draw, "the head is sweeping; nobody is close enough to the "
                "optical axis for their appearance to be read", F09,
                x1 - x0 - 20, max_lines=2)):
            draw.text((x0 + 8, y0 + 44 + index * 12), line, font=F09, fill=DIM)
        return

    name = sighting["name"]
    is_guardian = name == GUARDIAN.name
    verdict = sighting["verdict"]
    # The state, not merely the instantaneous verdict, decides the headline: a
    # candidate the machine has REJECTED must read as refused even though the
    # sighting that produced it still scores as "candidate".
    color = BAD if refused_now else verdict_color(verdict)

    draw.text((x0 + 8, y0 + 22), f"looking at  {name}", font=F13, fill=color)
    role = BY_NAME[name].role if name in BY_NAME else "crowd"
    tag = f"{role} - held" if replayed else role
    draw.text((x1 - 8 - text_w(draw, tag, F09), y0 + 26), tag, font=F09,
              fill=DIM)

    # Score against BOTH thresholds, drawn in place on the bar.
    score = float(sighting["score"])
    bar_box = (x0 + 8, y0 + 44, x1 - 46, y0 + 54)
    bar(draw, bar_box, score, color)
    for threshold, tint in ((CANDIDATE_SCORE, WARN), (ACCEPT_SCORE, GOOD)):
        tx = bar_box[0] + (bar_box[2] - bar_box[0]) * threshold
        draw.line([(tx, bar_box[1] - 3), (tx, bar_box[3] + 3)], fill=tint,
                  width=1)
    draw.text((x1 - 42, y0 + 43), f"{score:.3f}", font=F10, fill=color)
    draw.text((x0 + 8, y0 + 58),
              fit(draw, f"candidate {CANDIDATE_SCORE:.2f}  accept {ACCEPT_SCORE:.2f}"
                  f"  {'all 4 features read' if sighting['complete_descriptor'] else 'INCOMPLETE'}",
                  F09, x1 - x0 - 16), font=F09,
              fill=DIM if sighting["complete_descriptor"] else WARN)

    y = _draw_terms(draw, (x0 + 8, y0 + 76, x1 - 8), sighting)

    # The verdict, in the identity layer's own words.
    headline = ("ACCEPTED - this is the guardian"
                if verdict == "accept" and is_guardian
                else "REFUSED" if refused_now else "evaluating")
    draw.text((x0 + 8, y + 4), headline, font=F11, fill=color)
    for index, line in enumerate(
            wrap(draw, sighting["reason"], F09, x1 - x0 - 20, max_lines=2)):
        draw.text((x0 + 8, y + 20 + index * 12), line, font=F09, fill=color)


def draw_refusals(draw, box, record) -> None:
    """The running log of refused candidates: who, when, and why.

    This panel is what makes a still frame from late in the video sufficient
    evidence that the earlier refusals happened.
    """
    x0, y0, x1, y1 = box
    panel(draw, box)
    count = len(record.get("rejections", []))
    title(draw, box, f"REFUSED CANDIDATES ({count})")

    rejections = record.get("rejections", [])
    if not rejections:
        draw.text((x0 + 8, y0 + 22), "none yet", font=F10, fill=DIM)
        return
    # Three entries at 30 px each plus the heading is exactly what the panel
    # holds; a fourth would be drawn over the legend beneath it.  Three is also
    # all this rollout ever produces, and the count is stated so a viewer knows
    # the list is complete rather than truncated.
    y = y0 + 20
    for rejection in rejections[:3]:
        draw.text((x0 + 8, y), f"{rejection['t']:5.2f}s", font=F09, fill=DIM)
        draw.text((x0 + 54, y), rejection["name"], font=F10, fill=BAD)
        score = f"{rejection['score']:.3f}"
        draw.text((x1 - 8 - text_w(draw, score, F09), y), score, font=F09,
                  fill=DIM)
        for line in wrap(draw, rejection["reason"], F09, x1 - x0 - 20,
                         max_lines=1):
            draw.text((x0 + 8, y + 13), line, font=F09, fill=DIM)
        y += 30


def draw_legend(draw, box) -> None:
    """Who is who in the hall, on one row, so the plan view needs no captions.

    This is SCENE information, not the duck's belief: the duck does not know
    which adults were authored as look-alikes.  Saying so keeps the distinction
    between what the scenario knows and what the robot knows.
    """
    x0, y0, x1, y1 = box
    panel(draw, box)
    x = x0 + 8
    for color, label in ((GUARDIAN_IN_HUD, "guardian"),
                         (WARN, "look-alikes"),
                         (DIM, "crowd")):
        draw.ellipse([x, y0 + 8, x + 7, y0 + 15], fill=color)
        draw.text((x + 11, y0 + 6), label, font=F09, fill=DIM)
        x += 11 + text_w(draw, label, F09) + 14
