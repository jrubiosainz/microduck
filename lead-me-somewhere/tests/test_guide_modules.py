"""The modules the rollout was split into: aim, tally, markers, thresholds.

The split was made to keep every module under 300 lines of code, and a split is
only safe if the extracted pieces are testable on their own.  These are those
tests — none of them builds a scene.
"""

from __future__ import annotations

import ast
import math
import pathlib

import numpy as np
import pytest

from guide_aim import (
    Aim,
    bearing_to,
    facing_error_deg,
    reached_standing_point,
    select,
)
from guide_layout import DESTINATION_BY_KEY
from guide_route import Route
from guide_states import ARRIVE_RADIUS_M, STATES, ZERO_COMMAND_STATES
from guide_tally import RolloutTally
from guide_thresholds import (
    CHECK_STILL_PATH_M,
    FINAL_DISTANCE_BAND_M,
    MIN_BENDS,
    MIN_EPISODES,
    UPSTREAM_POLICY_SHA,
)
from guide_tracker import RouteTracker

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


# -- module size ------------------------------------------------------------

def code_lines(path: pathlib.Path) -> int:
    """Lines that are neither blank, comment, nor docstring."""
    source = path.read_text()
    tree = ast.parse(source)
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            text = ast.get_docstring(node, clean=False)
            if text and node.body and isinstance(node.body[0], ast.Expr):
                span = node.body[0]
                doc.update(range(span.lineno,
                                 (span.end_lineno or span.lineno) + 1))
    return sum(1 for index, line in enumerate(source.splitlines(), 1)
               if line.strip() and not line.strip().startswith("#")
               and index not in doc)


def test_every_module_stays_under_300_code_lines():
    """A module that outgrows this gets split, because a 600-line file is one
    nobody re-reads before changing."""
    oversized = {path.name: code_lines(path)
                 for path in sorted(SCRIPTS.glob("*.py"))
                 if code_lines(path) >= 300}
    assert not oversized, f"split these: {oversized}"


# -- guide_aim --------------------------------------------------------------

def straight_tracker() -> RouteTracker:
    return RouteTracker(Route("t", ((0.0, 0.0), (4.0, 0.0)), 1.0))


def test_every_zero_command_state_aims_at_nothing_to_walk_to():
    """A walking target in a state whose command is a literal zero would be a
    contradiction the HUD would then draw."""
    for state in ZERO_COMMAND_STATES:
        aim = select(state, duck_xy=(0.0, 0.0), tracker=straight_tracker(),
                     destination=DESTINATION_BY_KEY["LIFTS"],
                     follower_yaw=1.0, arrive_radius_m=ARRIVE_RADIUS_M)
        assert aim.target_xy is None, f"{state} produced a walking target"


def test_leading_aims_along_the_route_not_at_a_waypoint():
    tracker = straight_tracker()
    aim = select("LEAD", duck_xy=(1.0, 0.0), tracker=tracker,
                 destination=None, follower_yaw=0.0,
                 arrive_radius_m=ARRIVE_RADIUS_M)
    assert aim.kind == "route_pursuit"
    assert aim.target_xy is not None
    assert aim.target_xy[0] > 1.0
    assert abs(float(aim.target_xy[1])) < 1e-9
    assert aim.remaining_m < tracker.route.length


def test_monitoring_aims_the_head_and_nothing_else():
    for state in ("CHECK_FOLLOWER", "WAIT_FOR_PERSON"):
        aim = select(state, duck_xy=(0.0, 0.0), tracker=straight_tracker(),
                     destination=None, follower_yaw=2.4,
                     arrive_radius_m=ARRIVE_RADIUS_M)
        assert aim.target_xy is None
        assert aim.look_at_yaw == pytest.approx(2.4)


def test_arriving_closes_on_the_standing_point_then_stops():
    destination = DESTINATION_BY_KEY["LIFTS"]
    far = select("ARRIVE", duck_xy=(0.0, 0.0), tracker=None,
                 destination=destination, follower_yaw=0.0,
                 arrive_radius_m=ARRIVE_RADIUS_M)
    assert far.kind == "standing_point"
    assert far.target_xy is not None
    close = select("ARRIVE", duck_xy=destination.stand, tracker=None,
                   destination=destination, follower_yaw=0.0,
                   arrive_radius_m=ARRIVE_RADIUS_M)
    assert close.target_xy is None
    assert close.remaining_m == 0.0


def test_no_route_and_no_destination_are_normal_not_errors():
    """Both are true early in every run."""
    aim = select("LEAD", duck_xy=(0.0, 0.0), tracker=None, destination=None,
                 follower_yaw=0.0, arrive_radius_m=ARRIVE_RADIUS_M)
    assert aim == Aim()


def test_facing_is_measured_against_the_fixture_not_the_standing_point():
    """A guide that arrived and faced the floor it stands on has indicated
    nothing."""
    destination = DESTINATION_BY_KEY["LIFTS"]
    at_stand = destination.stand
    toward_fixture = bearing_to(at_stand, destination.position)
    assert facing_error_deg(at_stand, toward_fixture, destination) \
        == pytest.approx(0.0, abs=1e-6)
    assert facing_error_deg(at_stand, toward_fixture + math.pi, destination) \
        == pytest.approx(180.0, abs=1e-6)
    # The standing point and the fixture are genuinely different places, or the
    # distinction this test protects would be empty.
    assert float(np.linalg.norm(
        destination.position - destination.stand)) > 0.2


def test_facing_is_undefined_before_a_destination_is_resolved():
    assert facing_error_deg((0.0, 0.0), 0.0, None) is None


def test_reaching_the_standing_point_is_a_radius_not_a_guess():
    destination = DESTINATION_BY_KEY["LIFTS"]
    assert reached_standing_point(destination.stand, destination,
                                  ARRIVE_RADIUS_M)
    away = destination.stand + np.array([ARRIVE_RADIUS_M + 0.1, 0.0])
    assert not reached_standing_point(away, destination, ARRIVE_RADIUS_M)
    assert not reached_standing_point((0.0, 0.0), None, ARRIVE_RADIUS_M)


# -- guide_tally ------------------------------------------------------------

def tally() -> RolloutTally:
    return RolloutTally(dt=0.02, initial_trunk_z=0.116)


def test_a_fall_is_counted_only_below_the_threshold():
    t = tally()
    t.note_pose(0.095, 0.01)
    assert t.fallen_steps == 0
    t.note_pose(0.089, 0.01)
    assert t.fallen_steps == 1
    assert t.min_trunk_z == pytest.approx(0.089)
    assert t.path_m == pytest.approx(0.02)


def test_the_lead_gap_is_measured_along_the_shared_trail():
    """A body-frame test would call her ahead every time the duck turned a
    corner, which is the opposite of what the invariant is for."""
    t = tally()
    t.note_lead_gap(0.9)
    t.note_lead_gap(0.62)
    assert t.follower_ahead_steps == 0
    assert t.min_lead_gap_m == pytest.approx(0.62)
    t.note_lead_gap(-0.05)
    assert t.follower_ahead_steps == 1


def test_the_safety_breach_is_a_continuous_interval_not_a_count():
    """A distance beyond the maximum is not itself a failure; the duck has to
    notice and stop, which takes time.  A PROLONGED interval is the failure."""
    from guide_states import SAFETY_MAX_DISTANCE_M
    t = tally()
    far = SAFETY_MAX_DISTANCE_M + 0.5
    for _ in range(50):                       # 1.0 s beyond the maximum
        t.note_safety(far)
    assert t.max_safety_breach_s == pytest.approx(1.0, abs=1e-9)
    t.note_safety(0.5)                        # recovered: the run resets
    assert t.safety_breach_s == 0.0
    for _ in range(25):                       # a shorter second breach
        t.note_safety(far)
    assert t.max_safety_breach_s == pytest.approx(1.0, abs=1e-9), (
        "the maximum was overwritten by a shorter later breach")


def test_the_waiting_claim_is_tallied_on_wait_for_person_alone():
    """Folding CHECK_FOLLOWER in would let a state with a different contract
    vouch for the exact-zero claim."""
    t = tally()
    t.note_wait_tick(0, peak=0.0, travelled=0.001)
    t.note_check(0.02, 0)
    assert t.episode_wait_command_peak[0] == 0.0
    assert t.episode_wait_moved_m[0] == pytest.approx(0.001)
    assert t.episode_check_path_m[0] == pytest.approx(0.02)


def test_visibility_is_conditioned_and_monitoring_is_counted_separately():
    t = tally()
    t.note_visibility(visible=True, los_ok=True, monitoring=True, blocker="")
    t.note_visibility(visible=False, los_ok=True, monitoring=True,
                      blocker="tessa_torso")
    t.note_visibility(visible=True, los_ok=True, monitoring=False, blocker="")
    assert t.visible_steps == 2
    assert t.monitor_steps == 2
    assert t.monitor_visible_with_los == 1
    assert t.blocked_by == {"tessa_torso": 1}


def test_the_check_path_resets_between_episodes():
    t = tally()
    t.note_check(0.02, 0)
    t.reset_check()
    t.note_check(0.01, 1)
    assert t.episode_check_path_m[1] == pytest.approx(0.01)
    assert t.max_check_path_m == pytest.approx(0.02)


# -- guide_thresholds -------------------------------------------------------

def test_the_thresholds_module_holds_only_data():
    """It exists so the numbers a run is graded against can be read without
    loading the code that computes them."""
    tree = ast.parse((SCRIPTS / "guide_thresholds.py").read_text())
    for node in tree.body:
        assert isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr,
                                 ast.ImportFrom, ast.Import)), (
            f"guide_thresholds contains executable logic: {type(node).__name__}")


def test_the_thresholds_are_the_ones_the_readme_quotes():
    assert MIN_EPISODES == 2
    assert MIN_BENDS == 3
    assert FINAL_DISTANCE_BAND_M == (0.30, 0.95)
    assert CHECK_STILL_PATH_M == 0.030
    assert UPSTREAM_POLICY_SHA.startswith("e36332")


# -- guide_markers ----------------------------------------------------------

def test_the_marker_counts_match_the_scene_builder():
    """A mismatch is a KeyError at frame 1 of a 40-minute render."""
    import guide_markers
    builder = (SCRIPTS.parent / "tools" / "build_scene.py").read_text()
    for name, value in (("ROUTE_DISCS", guide_markers.ROUTE_DISCS),
                        ("WAYPOINT_DISCS", guide_markers.WAYPOINT_DISCS),
                        ("TRAIL_DISCS", guide_markers.TRAIL_DISCS)):
        assert f"{name} = {value}" in builder, (
            f"{name} is {value} in guide_markers but differs in build_scene.py")


def test_markers_are_pure_presentation_and_no_gate_reads_them():
    """If a gate consumed a marker the overlay could change a graded number."""
    metrics = (SCRIPTS / "guide_metrics.py").read_text()
    summary = (SCRIPTS / "guide_summary.py").read_text()
    for source in (metrics, summary):
        assert "guide_markers" not in source
        assert "mocap_pos" not in source
