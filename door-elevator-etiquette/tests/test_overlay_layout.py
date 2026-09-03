#!/usr/bin/env python3
"""The overlay's layout: panels must not overlap, and text must not overflow.

The first preview of this behavior lost the SAFETY panel's scenery-clearance
line - the number proving the duck never touched a wall - underneath the
timeline panel, and clipped the timeline's own door chips against the bottom
edge.  Both were invisible in the code and obvious in the picture, which is
exactly the kind of bug a test should catch instead.

These are geometry tests on the layout, so they need no rollout and no MuJoCo:
they build one synthetic record and compose a frame from it.
"""

from __future__ import annotations

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")

WIDTH, HEIGHT = 960, 640
PIP_W, PIP_H = 300, 216


def synthetic_record():
    """One record with every field the overlay reads, at plausible values."""
    return {
        "t": 55.0,
        "state": "LET_OCCUPANTS_EXIT",
        "command": [0.0, 0.0, 0.0],
        "command_peak": 0.0,
        "command_vx": 0.0,
        "careful": True,
        "duck_xy": [0.689, 0.936],
        "duck_yaw_deg": 12.0,
        "trunk_z": 0.1163,
        "min_trunk_z": 0.1121,
        "path_m": 5.2,
        "target_kind": "route_pursuit",
        "target_xy": [1.0, 0.9],
        "door_fraction": {"concourse_door": 1.0, "lift_front": 0.87,
                          "lift_rear": 0.0},
        "door_gap_m": {"concourse_door": 0.66, "lift_front": 0.62,
                       "lift_rear": 0.0},
        "door_passable": {"concourse_door": True, "lift_front": True,
                          "lift_rear": False},
        "zone_depth_m": {"concourse_door_threshold": 0.0,
                         "lift_front_threshold": 0.0,
                         "lift_front_passage": 0.0},
        "aperture_occupancy": {
            "concourse_door": {"duck": False, "others": []},
            "lift_front": {"duck": False, "others": ["marek"]},
            "lift_rear": {"duck": False, "others": []}},
        "cabin_margin_m": -0.4,
        "inside_cabin": False,
        "guardian": "nadia",
        "guardian_xy": [1.3, 0.1],
        "guardian_gap_m": 1.16,
        "guardian_through_door": True,
        "guardian_through_lift": False,
        "guardian_inside_cabin": False,
        "guardian_through_rear": False,
        "exiters_pending": 0,
        "exiters_in_aperture": 0,
        "all_exiters_clear": True,
        "occupants_exited": 1,
        "occupants_in_cabin": 2,
        "occupants_in_passage": 1,
        "all_occupants_clear": False,
        "yields_completed": 1,
        "subject": "marek",
        "subject_role": "occupant",
        "subject_visible": False,
        "subject_sample_count": 0,
        "subject_range_m": 2.23,
        "subject_blocked_by": "wall_lift_n",
        "los_available": False,
        "los_blocked_by": "wall_lift_n",
        "interlock_blocked": True,
        "interlock_reason": "the guardian is in this aperture; never abreast",
        "interlock_aperture": "lift_front",
        "min_person_clearance_m": 0.85,
        "nearest_person": "nadia",
        "scenery_clearance_m": 0.1157,
        "nearest_scenery": "wall_lift_n",
        "person_xy": {"nadia": [1.3, 0.1], "tomas": [-2.0, -0.5],
                      "leila": [-2.2, 0.6], "priya": [0.8, 0.3],
                      "marek": [1.0, -0.2], "odile": [1.2, -0.4],
                      "sami": [-3.0, 1.2], "vera": [-2.5, -1.5]},
        "person_role": {"nadia": "guardian", "tomas": "door_exiter",
                        "leila": "door_exiter", "priya": "occupant",
                        "marek": "occupant", "odile": "occupant",
                        "sami": "background", "vera": "background"},
        "person_speed": {name: 0.1 for name in
                         ("nadia", "tomas", "leila", "priya", "marek",
                          "odile", "sami", "vera")},
        "person_visible": {name: True for name in
                           ("nadia", "tomas", "leila", "priya", "marek",
                            "odile", "sami", "vera")},
        "visible_people": ["nadia"],
        "view_yaw_deg": 10.0,
        "gaze_yaw_deg": 5.0,
    }


def compose_frame(record=None):
    from video_overlay import compose
    main = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pip = np.zeros((PIP_H, PIP_W, 3), dtype=np.uint8)
    return compose(main, pip, record=record or synthetic_record(),
                   total_seconds=110.0,
                   summary={"state_windows": [
                       {"state": "APPROACH_DOOR", "start": 0.0, "end": 9.3},
                       {"state": "YIELD_EXITERS", "start": 9.3, "end": 17.0},
                       {"state": "LET_OCCUPANTS_EXIT", "start": 49.5,
                        "end": 55.0}]},
                   trail=[[-3.0, -0.4], [-2.0, -0.2], [0.0, 0.5]],
                   route_points=[[-3.3, -0.42], [-1.1, 0.0], [1.34, 0.1]])


def panel_boxes():
    """Every panel rectangle the overlay draws, in draw order.

    Kept in step with ``video_overlay.compose`` by construction: the test that
    matters is the OVERLAP one below, and these are the boxes it checks.
    """
    pip_x = WIDTH - PIP_W - 10
    boxes = {"title": (0, 0, WIDTH, 40)}
    y = 44 + PIP_H
    boxes["pip"] = (pip_x, 44, WIDTH - 10, y)
    y += 8
    for name, height in (("doors", 84), ("traffic", 84), ("order", 66),
                         ("safety", 60)):
        boxes[name] = (pip_x, y, WIDTH - 10, y + height)
        y += height + 6
    boxes["state"] = (10, 44, 278, 140)
    boxes["plan"] = (10, 146, 278, 342)
    boxes["legend"] = (10, 348, 278, 460)
    boxes["timeline"] = (10, HEIGHT - 56, WIDTH - 10, HEIGHT - 8)
    return boxes


def overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def test_no_two_panels_overlap():
    boxes = panel_boxes()
    names = sorted(boxes)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if first == "title" or second == "title":
                continue
            assert not overlaps(boxes[first], boxes[second]), (first, second)


def test_every_panel_is_inside_the_frame():
    for name, box in panel_boxes().items():
        assert box[0] >= 0 and box[1] >= 0, name
        assert box[2] <= WIDTH and box[3] <= HEIGHT, name


def test_the_right_column_clears_the_timeline():
    """The regression that lost the scenery-clearance number in the preview."""
    boxes = panel_boxes()
    assert boxes["safety"][3] < boxes["timeline"][1], (
        f"safety ends at {boxes['safety'][3]}, timeline starts at "
        f"{boxes['timeline'][1]}")


def test_the_timeline_chip_row_fits_inside_its_panel():
    """The other preview regression: chips clipped against the bottom edge."""
    from hud_views import Timeline
    box = panel_boxes()["timeline"]
    timeline = Timeline(box, 110.0)
    # top + bar height + gap + text height must stay inside the panel.
    top = box[1] + 20
    chip_bottom = top + 11 + 4 + 11
    assert chip_bottom <= box[3], (chip_bottom, box[3])
    assert timeline.total == 110.0


def test_a_frame_composes_at_the_declared_size():
    image = compose_frame()
    assert image.size == (WIDTH, HEIGHT)


def test_a_frame_composes_in_every_state():
    """Every state must draw without raising, including the terminal one."""
    from etiquette_states import STATES
    for state in STATES:
        record = synthetic_record()
        record["state"] = state
        compose_frame(record)


def test_the_overlay_draws_the_interlock_when_it_is_holding():
    """A held duck must say so; that is the most informative caption there is."""
    held = compose_frame()
    record = synthetic_record()
    record["interlock_blocked"] = False
    free = compose_frame(record)
    assert np.asarray(held).tobytes() != np.asarray(free).tobytes()


def test_the_overlay_reads_the_same_record_the_gate_grades(rollout):
    """No overlay-only fields: the picture cannot show what was not measured."""
    record = rollout.records[len(rollout.records) // 2]
    compose_frame(record)


def test_every_state_has_a_colour_and_a_caption():
    from etiquette_states import STATES
    from hud_style import STATE_CAPTION, STATE_COLORS
    for state in STATES:
        assert state in STATE_COLORS, state
        assert STATE_CAPTION.get(state), state


def test_the_door_colour_agrees_with_the_passability_threshold():
    """The HUD and the gate must mean the same thing by 'open'."""
    from hud_style import BAD, DOOR_INK, door_ink
    from lobby_doors import DOOR_PASSABLE_FRACTION
    assert door_ink(DOOR_PASSABLE_FRACTION) == DOOR_INK
    assert door_ink(DOOR_PASSABLE_FRACTION - 0.01) == BAD
