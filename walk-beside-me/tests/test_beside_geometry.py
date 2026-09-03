#!/usr/bin/env python3
"""The formation frame: the sign convention every claim in this behavior uses.

If ``relative`` returned the wrong sign for "left", every side decision, every
switch record and every gate about the forward half-plane would still be
internally consistent and completely wrong.  These tests pin the convention
against hand-computed geometry rather than against the module's own helpers.

No MuJoCo, no physics, no rollout: pure arithmetic on
:mod:`beside_geometry`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from beside_geometry import (  # noqa: E402
    BESIDE_LONG_TARGET_M,
    BESIDE_LONG_TOLERANCE_M,
    BESIDE_MAX_M,
    BESIDE_MIN_M,
    BESIDE_TARGET_M,
    CROSS_ARRIVE_M,
    CROSS_BEHIND_M,
    CROSS_COMMIT_M,
    CROSS_WAYPOINT_LONG_M,
    DUCK_PLANAR_RADIUS,
    FORWARD_HALF_PLANE_M,
    SIDE_LOOKAHEAD_S,
    SIDE_LOOKAHEAD_SAMPLES,
    SIDE_PERSON_MARGIN_M,
    SIDE_STATIC_MARGIN_M,
    band_verdict,
    crossed_forward_half_plane,
    formation_ok,
    frame,
    in_band,
    relative,
    side_name,
    side_of,
    slot_point,
)


# -- the frame ---------------------------------------------------------------

def test_the_frame_axes_are_orthonormal_and_left_is_forward_rotated_ninety():
    for yaw_deg in (0.0, 37.0, 90.0, 174.0, -122.0):
        forward, left = frame((1.3, -0.4), math.radians(yaw_deg))
        assert float(np.linalg.norm(forward)) == pytest.approx(1.0)
        assert float(np.linalg.norm(left)) == pytest.approx(1.0)
        assert float(forward @ left) == pytest.approx(0.0, abs=1e-12)
        # left is forward rotated +90 deg, i.e. cross(forward, left) is +z.
        cross = float(forward[0] * left[1] - forward[1] * left[0])
        assert cross == pytest.approx(1.0)


def test_left_is_positive_lateral_computed_without_the_helper():
    """A duck standing to the guardian's left has POSITIVE lateral offset."""
    # She faces +x, so her left is +y.
    lateral, longitudinal = relative((0.0, 0.6), (0.0, 0.0), 0.0)
    assert lateral == pytest.approx(0.6)
    assert longitudinal == pytest.approx(0.0)
    # Facing +y, her left is -x.
    lateral, longitudinal = relative((-0.6, 0.0), (0.0, 0.0), math.pi / 2.0)
    assert lateral == pytest.approx(0.6)
    assert longitudinal == pytest.approx(0.0)


def test_ahead_is_positive_longitudinal_in_every_heading():
    for yaw_deg in (0.0, 45.0, 130.0, -95.0):
        yaw = math.radians(yaw_deg)
        ahead = (math.cos(yaw) * 0.9, math.sin(yaw) * 0.9)
        lateral, longitudinal = relative(ahead, (0.0, 0.0), yaw)
        assert longitudinal == pytest.approx(0.9)
        assert lateral == pytest.approx(0.0, abs=1e-12)


def test_relative_is_invariant_to_a_rigid_motion_of_the_whole_pair():
    """Rotating and translating duck and guardian together changes nothing."""
    duck = np.array([1.4, -0.3])
    guardian = np.array([0.5, 0.2])
    yaw = math.radians(18.0)
    reference = relative(duck, guardian, yaw)
    for extra_deg, shift in ((61.0, (3.1, -2.4)), (-140.0, (-0.9, 5.0))):
        extra = math.radians(extra_deg)
        rotation = np.array([[math.cos(extra), -math.sin(extra)],
                             [math.sin(extra), math.cos(extra)]])
        moved_duck = rotation @ duck + np.asarray(shift)
        moved_guardian = rotation @ guardian + np.asarray(shift)
        moved = relative(moved_duck, moved_guardian, yaw + extra)
        assert moved[0] == pytest.approx(reference[0])
        assert moved[1] == pytest.approx(reference[1])


# -- the slot ----------------------------------------------------------------

def test_the_slot_point_round_trips_through_relative():
    """``slot_point`` and ``relative`` are inverses, which is what makes the
    controller's target and the gate's measurement the same statement."""
    for yaw_deg in (0.0, 33.0, 128.0, -74.0):
        yaw = math.radians(yaw_deg)
        guardian = (2.2, -1.1)
        for side in (1, -1):
            point = slot_point(guardian, yaw, side)
            lateral, longitudinal = relative(point, guardian, yaw)
            assert lateral == pytest.approx(side * BESIDE_TARGET_M)
            assert longitudinal == pytest.approx(BESIDE_LONG_TARGET_M)


def test_the_two_slots_are_mirror_images_about_her_centreline():
    yaw = math.radians(23.0)
    guardian = np.array([-1.0, 0.4])
    left = slot_point(guardian, yaw, 1)
    right = slot_point(guardian, yaw, -1)
    midpoint = (left + right) * 0.5
    forward, _ = frame(guardian, yaw)
    # The midpoint lies on her centreline, at the longitudinal target.
    offset = midpoint - guardian
    assert float(np.linalg.norm(offset - forward * BESIDE_LONG_TARGET_M)) \
        == pytest.approx(0.0, abs=1e-12)
    assert float(np.linalg.norm(left - right)) == pytest.approx(
        2.0 * BESIDE_TARGET_M)


def test_the_slot_refuses_a_side_that_is_not_plus_or_minus_one():
    for bad in (0, 2, -3, 0.5, None, "left"):
        with pytest.raises(ValueError):
            slot_point((0.0, 0.0), 0.0, bad)


def test_the_beside_target_sits_inside_the_band_it_is_graded_against():
    assert BESIDE_MIN_M < BESIDE_TARGET_M < BESIDE_MAX_M
    assert in_band(BESIDE_TARGET_M)
    assert band_verdict(BESIDE_TARGET_M) == "in band"
    assert band_verdict(BESIDE_MIN_M - 0.01) == "too close"
    assert band_verdict(BESIDE_MAX_M + 0.01) == "too far"
    assert in_band(BESIDE_MIN_M) and in_band(BESIDE_MAX_M)


def test_the_station_is_behind_her_shoulder_rather_than_abreast():
    """Zero would look like a race; the target is deliberately astern."""
    assert BESIDE_LONG_TARGET_M < 0.0
    assert abs(BESIDE_LONG_TARGET_M) < BESIDE_LONG_TOLERANCE_M


# -- sides -------------------------------------------------------------------

def test_side_of_puts_zero_on_the_right_and_names_both_sides():
    assert side_of(0.4) == 1
    assert side_of(-0.4) == -1
    assert side_of(0.0) == -1, "zero counts as right, as documented"
    assert side_name(1) == "left"
    assert side_name(-1) == "right"


# -- the formation predicate --------------------------------------------------

def test_formation_ok_requires_the_correct_side():
    """A duck perfectly in the LEFT slot is not in the RIGHT formation."""
    assert formation_ok(BESIDE_TARGET_M, BESIDE_LONG_TARGET_M, 1)
    assert not formation_ok(BESIDE_TARGET_M, BESIDE_LONG_TARGET_M, -1)
    assert formation_ok(-BESIDE_TARGET_M, BESIDE_LONG_TARGET_M, -1)
    assert not formation_ok(-BESIDE_TARGET_M, BESIDE_LONG_TARGET_M, 1)


def test_formation_ok_rejects_both_edges_of_the_lateral_band():
    assert not formation_ok(BESIDE_MIN_M - 0.001, BESIDE_LONG_TARGET_M, 1)
    assert not formation_ok(BESIDE_MAX_M + 0.001, BESIDE_LONG_TARGET_M, 1)
    assert formation_ok(BESIDE_MIN_M, BESIDE_LONG_TARGET_M, 1)
    assert formation_ok(BESIDE_MAX_M, BESIDE_LONG_TARGET_M, 1)


def test_formation_ok_rejects_a_duck_trailing_far_behind_her():
    """The whole failure mode this behavior exists to avoid."""
    trailing = BESIDE_LONG_TARGET_M - BESIDE_LONG_TOLERANCE_M - 0.001
    assert not formation_ok(BESIDE_TARGET_M, trailing, 1)
    assert formation_ok(BESIDE_TARGET_M,
                        BESIDE_LONG_TARGET_M - BESIDE_LONG_TOLERANCE_M, 1)
    # ... and equally a duck that has run ahead of its station.
    ahead = BESIDE_LONG_TARGET_M + BESIDE_LONG_TOLERANCE_M + 0.001
    assert not formation_ok(BESIDE_TARGET_M, ahead, 1)


def test_the_longitudinal_tolerance_is_symmetric_about_the_target():
    """Not about zero — a bug here would silently license overtaking."""
    low = BESIDE_LONG_TARGET_M - BESIDE_LONG_TOLERANCE_M
    high = BESIDE_LONG_TARGET_M + BESIDE_LONG_TOLERANCE_M
    assert formation_ok(BESIDE_TARGET_M, low + 1e-9, 1)
    assert formation_ok(BESIDE_TARGET_M, high - 1e-9, 1)
    assert not formation_ok(BESIDE_TARGET_M, low - 1e-3, 1)
    assert not formation_ok(BESIDE_TARGET_M, high + 1e-3, 1)


# -- the forward half-plane ---------------------------------------------------

def test_crossed_forward_half_plane_is_a_strict_test_at_the_limit():
    assert not crossed_forward_half_plane(FORWARD_HALF_PLANE_M)
    assert crossed_forward_half_plane(FORWARD_HALF_PLANE_M + 1e-6)
    assert not crossed_forward_half_plane(-2.0)


def test_the_forward_half_plane_limit_is_small_and_positive():
    """Slightly positive so the gate measures behavior, not stride."""
    assert 0.0 < FORWARD_HALF_PLANE_M < BESIDE_TARGET_M / 2.0


def test_the_half_plane_gate_binds_tighter_than_the_formation_predicate():
    """The overtaking claim is a STRONGER statement than "in formation".

    The formation predicate would still accept a duck 0.43 m ahead of her,
    because its tolerance is sized by the gait-onset cliff rather than by
    courtesy.  The forward half-plane gate sits well inside that, so passing it
    is a real claim about the behavior and not a restatement of the band.
    """
    front_edge = BESIDE_LONG_TARGET_M + BESIDE_LONG_TOLERANCE_M
    assert FORWARD_HALF_PLANE_M < front_edge, (
        "if the half-plane limit were the looser of the two, the gate would be "
        "implied by the formation predicate and would prove nothing")
    # And a duck exactly at the half-plane limit is still inside the formation
    # band, so the gate cannot fail a run the formation predicate calls good.
    assert formation_ok(BESIDE_TARGET_M, FORWARD_HALF_PLANE_M, 1)


# -- the crossover geometry ---------------------------------------------------

def test_the_crossing_waypoint_is_constructed_behind_her_by_construction():
    """Rear-going is a property of the CONSTRUCTION, not a rule applied after."""
    from beside_geometry import cross_point

    for yaw_deg in (0.0, 60.0, -155.0):
        yaw = math.radians(yaw_deg)
        guardian = (0.7, -2.0)
        for side in (1, -1):
            point = cross_point(guardian, yaw, side)
            lateral, longitudinal = relative(point, guardian, yaw)
            assert longitudinal == pytest.approx(CROSS_WAYPOINT_LONG_M)
            assert longitudinal < -CROSS_BEHIND_M, (
                "the waypoint must be deeper astern than the entry gate, so "
                "the duck is still behind her when it arrives")
            assert side_of(lateral) == side
            assert abs(lateral) == pytest.approx(0.5 * BESIDE_TARGET_M)


def test_the_crossing_entry_gate_is_deeper_astern_than_the_station():
    """A duck sitting at its nominal station has NOT yet fallen back.

    The entry gate is measured against the station, not against the outer edge
    of the longitudinal tolerance: that tolerance is sized by the gait-onset
    cliff, and requiring the crossing to clear it as well would make the
    fall-back a test of the speed ladder's quantisation.
    """
    assert CROSS_BEHIND_M > abs(BESIDE_LONG_TARGET_M)
    assert CROSS_BEHIND_M > BESIDE_TARGET_M, (
        "the duck must be further astern than it is to the side, or the "
        "crossing path would still sweep past her hip")
    # The waypoint itself is deeper than the gate that admits the crossing.
    assert CROSS_WAYPOINT_LONG_M < -CROSS_BEHIND_M


def test_the_commit_threshold_is_reached_before_the_far_slot():
    """Committing must happen strictly before arriving, or the state that
    detects the commitment could never fire before the join completes."""
    assert 0.0 < CROSS_COMMIT_M < BESIDE_MIN_M
    assert CROSS_ARRIVE_M > 0.0


# -- the margins --------------------------------------------------------------

def test_a_person_needs_more_room_than_a_wall_because_a_person_moves():
    assert SIDE_PERSON_MARGIN_M > SIDE_STATIC_MARGIN_M


def test_the_static_margin_exceeds_the_duck_conservative_radius():
    """A slot the duck could not physically occupy is refused before it is
    ever walked to."""
    assert SIDE_STATIC_MARGIN_M > DUCK_PLANAR_RADIUS


def test_the_lookahead_is_sampled_at_both_ends_and_in_between():
    assert SIDE_LOOKAHEAD_S > 0.0
    assert SIDE_LOOKAHEAD_SAMPLES >= 3
    horizons = [SIDE_LOOKAHEAD_S * i / (SIDE_LOOKAHEAD_SAMPLES - 1)
                for i in range(SIDE_LOOKAHEAD_SAMPLES)]
    assert horizons[0] == 0.0, "the present must be one of the samples"
    assert horizons[-1] == pytest.approx(SIDE_LOOKAHEAD_S)
    assert len(set(horizons)) == SIDE_LOOKAHEAD_SAMPLES


def test_the_duck_starts_out_of_both_slots_so_the_join_is_a_real_traverse():
    from beside_actors import ROUTES
    from beside_geometry import DUCK_START_XY

    route = ROUTES["nadia"]
    guardian = route.pos_at(0.0)
    yaw = route.yaw_at(0.0)
    lateral, longitudinal = relative(DUCK_START_XY, guardian, yaw)
    assert not formation_ok(lateral, longitudinal, 1)
    assert not formation_ok(lateral, longitudinal, -1)
    assert longitudinal < 0.0, "the duck starts BEHIND her, never ahead"
