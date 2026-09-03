#!/usr/bin/env python3
"""The four HUD panels on the left column: state, command, safety.

Each panel answers one question a sceptical viewer should be able to settle
from the picture alone, and every number drawn is the number the acceptance
gate reads - never a prettier derivative of it.

* **STATE**    - which of the thirteen states is active, what that means in
  plain English, and how long it has been held.
* **COMMAND**  - the literal ``(vx, vy, wz)`` register plus the trunk height
  ``z``.  An exact zero is called out in its own colour because "it stood
  perfectly still on the line between two people" is this behavior's hardest
  claim, and ``vy`` is drawn on every frame because "no lateral command, ever"
  is a gate rather than an observation.
* **SAFETY**   - surface clearance to the nearest person and to the nearest
  static fixture.  SURFACE, not centre-to-centre: those differ by both bodies'
  radii, about 0.42 m here, which is more than enough to draw a duck as safely
  clear of somebody it is standing on.
"""

from __future__ import annotations

from hud_style import F09, F10, F11, bar, fit, panel, span_bar, title
from pps_hud_style import (ACCENT, BAD, DIM, GOOD, GRID, HEADING, INK, STATION,
                           WARD, WARN, ZERO, state_caption, state_color)
from pps_states import (BUFFER_M, DUCK_PLANAR_RADIUS, ESCORT_JOIN_M,
                        INTERPOSE_ON_STATION_M, VX_ONSET, VX_REPOSITION,
                        ZERO_COMMAND_STATES)

# The nominal standing trunk height the final-height gate is graded against,
# and the floor below which a tick counts as a fall.
NOMINAL_Z = 0.116
FALLEN_Z = 0.09


def state_panel(draw, box, view: dict) -> None:
    """WHICH state, what it means, and how long it has been held."""
    panel(draw, box)
    title(draw, box, "STATE   thirteen states, one at a time")
    x, y = box[0] + 10, box[1] + 22
    width = box[2] - box[0] - 20

    state = view["state"]
    draw.text((x, y), fit(draw, state, F11, width - 96), font=F11,
              fill=state_color(state))
    held = view["state_held_s"]
    draw.text((box[2] - 12 - 74, y + 1), f"held {held:5.1f}s", font=F09,
              fill=DIM)
    y += 18

    for line in _two_lines(draw, state_caption(state), width):
        draw.text((x, y), line, font=F09, fill=DIM)
        y += 12

    # The named failures this behavior exists to avoid are declared, not
    # implied: a run that produced one would fail loudly.  Saying so on screen
    # is what stops "it never charged the intruder" being an absence.
    y = box[3] - 15
    draw.text((x, y), fit(
        draw, "never: charge intruder / block both / contact", F09, width),
        font=F09, fill=DIM)


def _two_lines(draw, text: str, width: int) -> list[str]:
    from hud_style import wrap
    return wrap(draw, text, F09, width, max_lines=2)


def command_panel(draw, box, view: dict) -> None:
    """The command register verbatim, plus the trunk height."""
    panel(draw, box)
    title(draw, box, "COMMAND   the register, and z")
    x, y = box[0] + 10, box[1] + 22

    vx, vy, wz = view["command"]
    peak = view["command_peak"]
    must_be_zero = view["state"] in ZERO_COMMAND_STATES

    if peak == 0.0:
        draw.text((x, y), "EXACT ZERO", font=F11, fill=ZERO)
        mark = "declared hold" if must_be_zero else "settled"
        draw.text((x + 104, y + 1), mark, font=F09, fill=DIM)
    else:
        draw.text((x, y), f"vx {vx:+.3f}", font=F11, fill=INK)
        draw.text((x + 96, y), f"wz {wz:+.3f}", font=F11, fill=HEADING)
    y += 19

    # vy on EVERY frame: "no lateral policy command" is a gate, and a gate that
    # is only shown when it is violated cannot be watched being satisfied.
    draw.text((x, y), f"vy {vy:+.3f}", font=F09, fill=DIM if vy == 0.0 else BAD)
    # Gait onset is a cliff, not a ramp: below VX_ONSET the robot does not walk
    # at all, so the mark is what separates a real walk from a decorative one.
    span_bar(draw, (x + 84, y + 2, box[2] - 12, y + 9), -0.40, 0.60, vx,
             ZERO if vx == 0.0 else ACCENT,
             marks=((VX_ONSET, GOOD), (VX_REPOSITION, WARN), (0.0, GRID)))
    y += 15

    z = view["trunk_z"]
    draw.text((x, y), f"trunk z {z:5.3f} m", font=F09,
              fill=INK if z >= FALLEN_Z else BAD)
    draw.text((x + 132, y), f"nominal {NOMINAL_Z:.3f}", font=F09, fill=DIM)


def safety_panel(draw, box, view: dict) -> None:
    """Surface clearance to the nearest person and the nearest fixture."""
    panel(draw, box)
    title(draw, box, "CLEARANCE   surface, not centre distance")
    x, y = box[0] + 10, box[1] + 22
    width = box[2] - box[0] - 20

    person = view["min_person_clearance_m"]
    draw.text((x, y), fit(
        draw, f"person {view['nearest_person']:<7} {person:5.3f} m", F10,
        width), font=F10, fill=INK if person > 0.0 else BAD)
    y += 15
    span_bar(draw, (x, y, box[2] - 12, y + 7), 0.0, 1.60, person,
             GOOD if person > 0.0 else BAD, marks=((0.0, BAD),))
    y += 13

    scenery = view["scenery_clearance_m"]
    draw.text((x, y), fit(
        draw, f"fixture {view['nearest_scenery']:<11} {scenery:5.3f} m", F10,
        width), font=F10, fill=INK if scenery > 0.0 else BAD)
    y += 15
    span_bar(draw, (x, y, box[2] - 12, y + 7), 0.0, 1.60, scenery,
             GOOD if scenery > 0.0 else BAD, marks=((0.0, BAD),))


def threats_panel(draw, box, view: dict) -> None:
    """Every person the duck is PREDICTING, with the numbers it predicted.

    This is the panel that shows judgment rather than reaction.  Each row is one
    constant-velocity prediction the duck made this tick: range, closest
    predicted approach, and time to that approach.  A row is red when the duck
    selected that person, amber when they are the second half of a squeeze, and
    slate when the duck is watching them and doing nothing - which is what the
    false near-pass looks like for its entire crossing, and what the gate calls
    a dismissal.

    The CPA column carries the whole decision: an intrusion is predicted when
    the closest approach falls at least ``PREDICT_MARGIN_M`` INSIDE the buffer,
    so a viewer can read the buffer radius off the header and check each row.
    """
    panel(draw, box)
    title(draw, box, f"THREATS   buffer {BUFFER_M:.2f} m around Aina")
    x, y = box[0] + 10, box[1] + 22
    width = box[2] - box[0] - 20

    draw.text((x, y), "person    range   cpa   ttc  in", font=F09, fill=DIM)
    y += 13
    rows = view["predictions"][:5]
    if not rows:
        draw.text((x, y), "nobody within the alert range", font=F09, fill=DIM)
        y += 13
    for entry in rows:
        ink = view["person_ink"](entry["name"])
        draw.text((x, y), fit(draw, f"{entry['name']:<9}", F09, 62), font=F09,
                  fill=ink)
        draw.text((x + 62, y), f"{entry['range_m']:5.2f}", font=F09, fill=ink)
        draw.text((x + 106, y), f"{entry['cpa_m']:5.2f}", font=F09,
                  fill=BAD if entry["cpa_m"] <= BUFFER_M else DIM)
        draw.text((x + 150, y), f"{entry['ttc_s']:5.2f}", font=F09, fill=DIM)
        draw.text((x + 196, y), "YES" if entry["intrusion"] else "no",
                  font=F09, fill=BAD if entry["intrusion"] else DIM)
        y += 13

    y = box[3] - 42
    selected = view["selected"] or "-"
    secondary = view["secondary"]
    draw.text((x, y), "selected", font=F09, fill=DIM)
    draw.text((x + 66, y), fit(draw, selected, F10, 96), font=F10,
              fill=view["person_ink"](selected))
    if view["threat_range_m"] is not None:
        draw.text((x + 168, y), f"{view['threat_range_m']:5.2f} m to Aina",
                  font=F09, fill=DIM)
    y += 14
    draw.text((x, y), "secondary", font=F09, fill=DIM)
    draw.text((x + 66, y), fit(draw, secondary or "-", F10, 96), font=F10,
              fill=view["person_ink"](secondary) if secondary else DIM)
    y += 14
    draw.text((x, y), fit(
        draw, "constant-velocity prediction from measured poses", F09, width),
        font=F09, fill=DIM)


def station_panel(draw, box, view: dict) -> None:
    """WHERE the duck decided to stand, and how close it is to being there.

    Between-ness is the claim the interpose gate turns on, and it is a bearing
    test rather than a distance one: the duck is between when its bearing from
    Aina is within the tolerance of the intruder's.  It is drawn as a live
    yes/no so a viewer can watch it become true DURING the walk rather than
    being told afterwards that it did.
    """
    panel(draw, box)
    title(draw, box, "STATION   the place it chose to stand")
    x, y = box[0] + 10, box[1] + 22
    width = box[2] - box[0] - 20

    kind = view["target_kind"]
    draw.text((x, y), fit(draw, kind, F10, width), font=F10,
              fill=STATION if view["target"] is not None else DIM)
    y += 16

    distance = view["target_distance_m"]
    if distance is None:
        draw.text((x, y), "no station: escorting", font=F09, fill=DIM)
        y += 14
    else:
        draw.text((x, y), f"to station {distance:5.3f} m", font=F09, fill=INK)
        y += 12
        # The two tolerances that actually decide the machine's transitions,
        # marked on the same scale as the live distance.
        span_bar(draw, (x, y, box[2] - 12, y + 7), 0.0, 2.00, distance,
                 STATION, bands=((0.0, INTERPOSE_ON_STATION_M),),
                 marks=((ESCORT_JOIN_M, WARD),))
        y += 12
        draw.text((x, y), fit(draw, (
            f"on station <= {INTERPOSE_ON_STATION_M:.2f} m; "
            f"escort joined <= {ESCORT_JOIN_M:.2f} m"), F09, width),
            font=F09, fill=DIM)
        y += 14

    between = view["between"]
    draw.text((x, y), "between Aina and the intruder", font=F09, fill=DIM)
    draw.text((box[2] - 12 - 30, y), "YES" if between else "no", font=F09,
              fill=GOOD if between else DIM)
    y += 14

    draw.text((x, y), f"escort slot {view['escort_distance_m']:5.3f} m",
              font=F09, fill=DIM)
    y += 12
    draw.text((x, y), f"Aina at {view['ward_range_m']:5.2f} m", font=F09,
              fill=WARD)
    y += 14
    draw.text((x, y), fit(draw, (
        f"planned with a {DUCK_PLANAR_RADIUS:.3f} m planar radius"), F09,
        width), font=F09, fill=DIM)


def progress_panel(draw, box, view: dict) -> None:
    """The six required episodes, ticked off as the duck actually closes them.

    The order is a gate: a run that produced the same six episodes in a
    different order fails rather than passing on a count.  Drawing it as an
    ordered list lets a viewer watch that gate being satisfied.
    """
    panel(draw, box)
    title(draw, box, "EPISODES   six, in this order")
    x, y = box[0] + 10, box[1] + 22
    expected = view["expected_episodes"]
    closed = view["closed_episodes"]
    for index, kind in enumerate(expected):
        done = index < len(closed)
        current = index == len(closed)
        ink = GOOD if done else (WARN if current else DIM)
        mark = "*" if done else (">" if current else " ")
        label = f"{mark} {index + 1}. {kind}"
        if done:
            label += f"  {closed[index]['selected']}"
        draw.text((x, y), fit(draw, label, F09, box[2] - box[0] - 20),
                  font=F09, fill=ink)
        if done and closed[index]["kind"] != kind:
            draw.text((box[2] - 12 - 46, y), "ORDER", font=F09, fill=BAD)
        y += 13
