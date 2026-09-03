"""The headless gate must have no rendering dependency, and the HUD must not
invent numbers.

Two separate claims:

* ``scripts/validate_guide.py`` is the loop development happens in.  If it grew
  an ``imageio`` or ``PIL`` import, a validation run would need a rendering
  stack it does not use, and the claim that the gate is renderer-independent
  would be false.  The test proves it by BLOCKING those modules in
  ``sys.meta_path`` and importing the entry point anyway.

* every figure the overlay draws comes out of the per-tick record the gate
  reads.  An overlay that computed its own numbers could show a viewer something
  the gate never graded.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

# Everything the headless path is allowed to reach.  ``render_frames``,
# ``video_overlay``, ``hud_*`` and ``render_lead_me`` are deliberately outside
# this set: they ARE the renderer.
HEADLESS_MODULES = (
    "validate_guide", "rollout_guide", "guide_metrics", "guide_machine",
    "guide_control", "guide_planner", "guide_follower", "guide_camera",
    "guide_actors", "guide_cast", "guide_layout", "guide_record",
    "guide_route", "guide_states", "guide_tracker", "contact_geometry",
    "policy_runtime",
)

RENDER_ONLY = ("imageio", "imageio.v2", "PIL", "PIL.Image", "PIL.ImageDraw",
               "PIL.ImageFont", "matplotlib")


class _Blocker:
    """Refuse to import anything in ``names``, from anywhere."""

    def __init__(self, names):
        self.names = set(names)

    def find_module(self, fullname, path=None):
        return self if fullname.split(".")[0] in self.names else None

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise ImportError(
                f"{fullname} is blocked: the headless gate must not need a "
                "rendering stack")
        return None


def test_no_headless_module_imports_a_rendering_stack():
    """Static check first: it names the offending file if it fails."""
    banned = {name.split(".")[0] for name in RENDER_ONLY}
    for module in HEADLESS_MODULES:
        path = SCRIPTS / f"{module}.py"
        assert path.is_file(), f"{module} is missing"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned, (
                        f"{module}.py imports {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, (
                    f"{module}.py imports from {node.module}")


def test_the_headless_gate_imports_with_the_rendering_stack_blocked():
    """The claim, proved rather than asserted."""
    blocker = _Blocker({name.split(".")[0] for name in RENDER_ONLY})
    saved = {name: sys.modules.pop(name, None)
             for name in list(sys.modules)
             if name.split(".")[0] in blocker.names}
    for module in HEADLESS_MODULES:
        sys.modules.pop(module, None)
    sys.meta_path.insert(0, blocker)
    try:
        for module in HEADLESS_MODULES:
            importlib.import_module(module)
    finally:
        sys.meta_path.remove(blocker)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


def test_the_renderer_is_the_only_place_the_stack_enters():
    """If nothing imported it, the block above would prove nothing."""
    renderer = (SCRIPTS / "render_frames.py").read_text()
    assert "imageio" in renderer
    overlay = (SCRIPTS / "video_overlay.py").read_text()
    assert "PIL" in overlay


def test_render_lead_me_imports_the_renderer_lazily():
    """``--no-render`` must really be dependency-free, which means the import
    lives inside the branch rather than at module scope."""
    tree = ast.parse((SCRIPTS / "render_lead_me.py").read_text())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module)
    assert "render_frames" not in top_level
    assert not any(name.split(".")[0] in {"imageio", "PIL"}
                   for name in top_level)


# -- the overlay draws only what the gate grades ---------------------------

RECORD_KEYS = {
    "t", "state", "state_elapsed_s", "command", "command_peak", "command_vx",
    "duck_xy", "duck_yaw_deg", "trunk_z", "min_trunk_z", "path_m",
    "requested_destination", "destination", "destination_label",
    "destination_xy", "destination_distance_m", "facing_error_deg",
    "candidates", "target_xy", "target_kind", "look_at_yaw_deg", "follower",
    "follower_xy", "follower_range_m", "follower_visible",
    "follower_sample_count", "follower_blocked_by", "follower_speed_mps",
    "follower_trail_gap_m", "follower_walked_m", "follower_stall_label",
    "los_available", "los_blocked_by", "lagging", "unseen", "safety_breach_s",
    "waiting_spot", "episodes_completed", "min_person_clearance_m",
    "nearest_person", "scenery_clearance_m", "nearest_scenery", "person_xy",
    "person_role", "person_visible", "visible_people", "view_yaw_deg",
    "gaze_yaw_deg", "gesture_yaw_deg",
}


@pytest.mark.slow
def test_the_record_carries_every_field_the_hud_reads(rollout):
    """The overlay indexes the record by name.  A missing key is a crash at
    frame 1 of a 40-minute render, so it is checked here instead."""
    record = rollout.records[10]
    missing = RECORD_KEYS - set(record)
    assert not missing, f"the record is missing {sorted(missing)}"
    # The route fields are prefixed and only exist once a route is planned.
    late = rollout.records[-1]
    for key in ("route_arc_s_m", "route_route_length_m", "route_remaining_m",
                "route_cross_track_m", "route_progress"):
        assert key in late, f"{key} missing from a post-plan record"


@pytest.mark.slow
def test_the_hud_cannot_show_a_figure_the_gate_did_not_grade(rollout):
    """Every number the panels draw is looked up from the record, so the two
    cannot disagree.  Checked by rendering one frame's overlay against a record
    and requiring no KeyError."""
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    import hud_panels
    import hud_views

    record = rollout.records[-1]
    image = Image.new("RGBA", (960, 640))
    draw = ImageDraw.Draw(image)
    hud_panels.draw_status(draw, (12, 38, 316, 158), record)
    hud_panels.draw_request(draw, (12, 164, 316, 264), record)
    hud_panels.draw_follower(draw, (12, 270, 316, 410), record)
    hud_panels.draw_progress(draw, (12, 416, 316, 526), record)
    hud_panels.draw_safety(draw, (12, 532, 316, 558), record)
    hud_panels.draw_legend(draw, (12, 564, 316, 594), record)
    hud_views.PlanView((640, 262, 948, 548)).draw(draw, record)
    hud_views.Timeline((324, 554, 948, 628), rollout.seconds).draw(
        draw, record, {"state_windows": [], "episodes": rollout.machine.episodes})


@pytest.mark.slow
def test_the_timeline_never_shows_an_event_before_it_happens(rollout):
    """A viewer must not see a wait marked on the timeline before the duck
    decides it.  The frame writer accumulates events from the LIVE machine, so
    this checks the ordering the accumulation relies on."""
    for episode in rollout.machine.episodes:
        assert episode["detected_at_s"] <= episode["checked_at_s"]
        assert episode["checked_at_s"] <= episode["resumed_at_s"]
