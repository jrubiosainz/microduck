#!/usr/bin/env python3
"""The HUD panels: the request, the follower, and progress along the route.

Every number drawn here comes straight out of the per-tick record built by
``guide_record``, which is the SAME dict the acceptance gate reads.  The overlay
therefore cannot show a figure the gate did not grade, and a viewer reading the
HUD is reading the measurement rather than a caption about it.

THE FOLLOWER PANEL IS THE POINT OF THE VIDEO
---------------------------------------------
A guide walks with its back to the person it is leading, so the whole question
is whether it still knows where she is.  The panel draws her measured distance
against the MEASURED lag threshold and the catch-up threshold on one scale, with
both thresholds marked, plus whether the camera can currently see her.  That is
what makes the wait legible: the viewer watches the distance bar cross the lag
mark, sees the state turn amber several seconds later once the confirm window
has elapsed, and only then does the duck stop.  Cause before effect, on screen,
in that order.

THE REQUEST PANEL EXISTS TO MAKE THE CHOICE FALSIFIABLE
--------------------------------------------------------
Three destination chips are drawn, in their own colours, with the requested one
marked and the other two visibly not.  Without it a viewer has no way to tell a
guide that obeyed a request from one that walked to the only place it knew.
"""

from __future__ import annotations

import math

from guide_layout import DESTINATIONS
from guide_states import (
    CATCHUP_DISTANCE_M,
    FACE_TOLERANCE_DEG,
    LAG_CONFIRM_S,
    LAG_DISTANCE_M,
    RESUME_CONFIRM_S,
    SAFETY_MAX_DISTANCE_M,
)
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
    ROUTE,
    STATE_CAPTION,
    STATE_COLORS,
    WARN,
    bar,
    destination_ink,
    fit,
    panel,
    role_color,
    span_bar,
    text_w,
    title,
)


def draw_status(draw, box, record) -> None:
    """What the duck is doing, and the command that proves it."""
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

    # The command.  When it is exactly zero that is the CLAIM, not an absence,
    # so it is spelled out rather than left as three zeros to be squinted at.
    vx, vy, wz = record["command"]
    stopped = record["command_peak"] == 0.0
    draw.text((x0 + 8, y0 + 56),
              f"cmd vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}", font=F10,
              fill=DIM if stopped else INK)
    if stopped:
        draw.text((x0 + 8, y0 + 72), "EXACTLY ZERO - standing still",
                  font=F09, fill=WARN)
    else:
        draw.text((x0 + 8, y0 + 72),
                  f"walking; gait onset MEASURED at vx 0.24", font=F09,
                  fill=DIM)
    draw.text((x0 + 8, y0 + 88),
              f"trunk z {record['trunk_z']:.4f} m   min "
              f"{record['min_trunk_z']:.4f} m", font=F09, fill=DIM)
    draw.text((x0 + 8, y0 + 102),
              f"path {record['path_m']:.2f} m", font=F09, fill=DIM)


def draw_request(draw, box, record) -> None:
    """The three candidates, and which one was asked for.

    Without this a viewer cannot tell a guide that obeyed a request from one
    that walked to the only place it knew about.
    """
    x0, y0, x1, y1 = box
    panel(draw, box)
    title(draw, box, "REQUEST - 3 candidates, one asked for")
    requested = record["requested_destination"]
    y = y0 + 22
    for destination in DESTINATIONS:
        chosen = destination.key == requested
        ink = destination_ink(destination.key)
        draw.rectangle([x0 + 10, y + 2, x0 + 20, y + 12],
                       fill=ink if chosen else None, outline=ink)
        draw.text((x0 + 28, y), destination.key, font=F10,
                  fill=INK if chosen else DIM)
        if chosen:
            draw.text((x0 + 110, y), "\u25c0 requested", font=F09, fill=GOOD)
        y += 18
    if record["destination_distance_m"] is not None:
        draw.text((x0 + 8, y1 - 16),
                  fit(draw, f"distance to {requested}: "
                      f"{record['destination_distance_m']:.2f} m", F09,
                      x1 - x0 - 16), font=F09, fill=DIM)


def draw_follower(draw, box, record) -> None:
    """Her measured distance against the thresholds, and whether she is seen.

    Both thresholds are drawn ON the scale rather than quoted beside it, so the
    viewer sees the value cross the mark rather than comparing two numbers.
    """
    x0, y0, x1, y1 = box
    panel(draw, box)
    title(draw, box, "THE PERSON I AM LEADING")

    visible = record["follower_visible"]
    lagging = record["lagging"]
    draw.text((x0 + 8, y0 + 20), record["follower"] or "mara", font=F11,
              fill=role_color("follower"))
    draw.text((x0 + 78, y0 + 20),
              "SEEN" if visible else "NOT SEEN", font=F10,
              fill=GOOD if visible else BAD)
    draw.text((x0 + 150, y0 + 20),
              f"{record['follower_sample_count']}/5 samples", font=F09,
              fill=DIM)

    distance = float(record["follower_range_m"])
    draw.text((x0 + 8, y0 + 40), "distance", font=F09, fill=DIM)
    draw.text((x0 + 74, y0 + 39), f"{distance:.2f} m", font=F10,
              fill=BAD if lagging else GOOD)
    span_bar(draw, (x0 + 8, y0 + 56, x1 - 8, y0 + 68), 0.0,
             SAFETY_MAX_DISTANCE_M, distance,
             BAD if lagging else GOOD,
             bands=((0.0, CATCHUP_DISTANCE_M),),
             marks=((LAG_DISTANCE_M, BAD), (CATCHUP_DISTANCE_M, GOOD)))
    draw.text((x0 + 8, y0 + 71),
              fit(draw, f"green band = caught up (<{CATCHUP_DISTANCE_M:.2f} m)"
                  f"   red mark = lagging (>{LAG_DISTANCE_M:.2f} m)", F09,
                  x1 - x0 - 16), font=F09, fill=DIM)

    # Why the duck has not acted yet, or has.
    if record["state"] in ("LEAD", "RESUME") and lagging:
        draw.text((x0 + 8, y0 + 88),
                  fit(draw, f"lagging - must hold {LAG_CONFIRM_S:.1f}s before "
                      "I stop", F09, x1 - x0 - 16), font=F09, fill=WARN)
    elif record["state"] == "WAIT_FOR_PERSON":
        draw.text((x0 + 8, y0 + 88),
                  fit(draw, f"waiting - resume needs <{CATCHUP_DISTANCE_M:.2f} m"
                      f" AND seen, for {RESUME_CONFIRM_S:.1f}s", F09,
                      x1 - x0 - 16), font=F09, fill=WARN)
    elif record["follower_blocked_by"]:
        draw.text((x0 + 8, y0 + 88),
                  fit(draw, f"blocked by {record['follower_blocked_by']}", F09,
                      x1 - x0 - 16), font=F09, fill=BAD)

    draw.text((x0 + 8, y0 + 104),
              f"she has walked {record['follower_walked_m']:.2f} m  "
              f"behind me by {record['follower_trail_gap_m']:.2f} m",
              font=F09, fill=DIM)
    draw.text((x0 + 8, y0 + 118),
              f"episodes handled: {record['episodes_completed']}", font=F09,
              fill=DIM)


def draw_progress(draw, box, record) -> None:
    """How far along the planned route the duck is, and the arrival facing."""
    x0, y0, x1, y1 = box
    panel(draw, box)
    title(draw, box, "ROUTE PROGRESS")

    arc = record.get("route_arc_s_m")
    total = record.get("route_route_length_m")
    if arc is not None and total:
        fraction = float(record.get("route_progress", 0.0))
        draw.text((x0 + 8, y0 + 20),
                  f"{arc:.2f} / {total:.2f} m", font=F10, fill=INK)
        draw.text((x0 + 130, y0 + 20),
                  f"{fraction * 100:.0f}%", font=F10, fill=ROUTE)
        bar(draw, (x0 + 8, y0 + 38, x1 - 8, y0 + 48), fraction, ROUTE)
        draw.text((x0 + 8, y0 + 54),
                  f"cross-track {record.get('route_cross_track_m', 0.0):.3f} m",
                  font=F09, fill=DIM)
    else:
        draw.text((x0 + 8, y0 + 20), "no route yet", font=F10, fill=DIM)

    facing = record.get("facing_error_deg")
    if facing is not None:
        ok = facing <= FACE_TOLERANCE_DEG
        draw.text((x0 + 8, y0 + 72), "facing the destination", font=F09,
                  fill=DIM)
        draw.text((x0 + 150, y0 + 71), f"{facing:.0f} deg", font=F10,
                  fill=GOOD if ok else DIM)
    kind = record["target_kind"]
    if kind:
        draw.text((x0 + 8, y0 + 88), fit(draw, f"target  {kind}", F09,
                                         x1 - x0 - 16), font=F09,
                  fill=ACCENT if "route" in kind else DIM)


def draw_safety(draw, box, record) -> None:
    """Measured clearance to the nearest person and the nearest surface.

    Both are per-tick analytic measurements against the real geoms at the real
    pose, which is the only honest gate in a scene where nothing collides.
    """
    x0, y0, x1, y1 = box
    panel(draw, box)
    person = float(record["min_person_clearance_m"])
    scenery = float(record["scenery_clearance_m"])
    draw.text((x0 + 8, y0 + 8), "clearance", font=F09, fill=DIM)
    draw.text((x0 + 74, y0 + 7),
              f"{person:.3f} m {record['nearest_person'][:6]}", font=F09,
              fill=GOOD if person > 0.0 else BAD)
    draw.text((x0 + 190, y0 + 7),
              f"{scenery:.3f} m {record['nearest_scenery'][:10]}", font=F09,
              fill=GOOD if scenery > 0.0 else BAD)


def draw_legend(draw, box, record) -> None:
    """Who is who in the picture, coloured as they are in the plan view.

    Laid out from MEASURED text width and stopped at the panel edge, because
    seven names at fixed spacing overflow the panel and print on top of one
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
