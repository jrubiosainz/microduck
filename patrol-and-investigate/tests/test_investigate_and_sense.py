#!/usr/bin/env python3
"""The standoff planner, the sense boundary and the interlock.

Pure geometry on hand-built inputs.  These are the tests that pin the two
independent guards on the restricted zone, and the one property that makes a
refusal safe on a robot that cannot turn in place.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from patrol_actors import CRATE_XY, bodies_at
from patrol_control import Interlock
from patrol_facility import (
    CHECKPOINTS,
    RESTRICTED_ZONE,
    ZONE_STANDOFF_M,
)
from patrol_investigate import (
    CANDIDATE_BEARINGS_DEG,
    observation_look_point,
    plan_standoff,
    range_for_standoff,
    standoff_from_range,
    target_look_point,
)
from patrol_sense import (
    INTERLOCK_CLEARANCE_M,
    INTERLOCK_LOOKAHEAD_M,
    ZONE_INTERLOCK_MARGIN_M,
    build_interlock,
    in_standoff_band,
    los_blocked_by,
    measured_positions,
    zone_gap_m,
)
from patrol_states import (
    DUCK_PLANAR_RADIUS,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
    STANDOFF_TARGET_M,
)


# -- the standoff band ---------------------------------------------------------
def test_the_range_and_standoff_conversions_are_inverses():
    for target in ("crate", "visitor", "trolley"):
        for standoff in (0.45, 0.60, 0.75):
            back = standoff_from_range(
                target, range_for_standoff(target, standoff))
            assert back == pytest.approx(standoff, abs=1e-9)


def test_the_band_test_accepts_only_the_required_window():
    inside = range_for_standoff("crate", STANDOFF_TARGET_M)
    assert in_standoff_band(inside, "crate")
    too_close = range_for_standoff("crate", STANDOFF_MIN_M - 0.05)
    too_far = range_for_standoff("crate", STANDOFF_MAX_M + 0.05)
    assert not in_standoff_band(too_close, "crate")
    assert not in_standoff_band(too_far, "crate")


def test_the_planner_aims_at_the_middle_of_the_band():
    """So the MEASURED stopping error has room on both sides."""
    assert STANDOFF_MIN_M < STANDOFF_TARGET_M < STANDOFF_MAX_M


# -- the standoff planner --------------------------------------------------------
def test_a_standoff_is_found_for_every_anomaly_from_its_checkpoint():
    states = bodies_at(150.0)
    for name in ("crate", "visitor", "trolley"):
        best = min(CHECKPOINTS,
                   key=lambda c: float(np.linalg.norm(
                       c.position - states[name].pos)))
        plan = plan_standoff(name, states[name].pos, best.position)
        assert plan.ok, (name, [c.reason for c in plan.candidates])


def test_the_chosen_standoff_is_the_short_way_round():
    plan = plan_standoff("crate", CRATE_XY, (0.5, 0.5))
    viable = [c for c in plan.candidates if c.ok]
    assert plan.chosen.walk_m == pytest.approx(
        min(c.walk_m for c in viable))


def test_every_candidate_is_kept_so_a_refusal_can_be_explained():
    """A plan that recorded only its answer could not distinguish a decision
    from a default."""
    plan = plan_standoff("crate", CRATE_XY, (0.5, 0.5))
    assert len(plan.candidates) == len(CANDIDATE_BEARINGS_DEG)
    assert all(c.reason or c.ok for c in plan.candidates)


def test_the_planner_never_offers_a_point_inside_the_restricted_zone():
    """THE FIRST OF TWO INDEPENDENT GUARDS.  Approaching an intruder is not a
    licence to enter the area they are standing in."""
    intruder = RESTRICTED_ZONE.center
    for approach_from in ((0.0, 0.0), (-1.0, 0.2), (0.5, 1.5), (-2.6, 1.9)):
        plan = plan_standoff("visitor", intruder, approach_from)
        for candidate in plan.candidates:
            if candidate.ok:
                assert not RESTRICTED_ZONE.contains(candidate.xy), candidate
                assert candidate.zone_gap_m >= ZONE_STANDOFF_M


def test_a_standoff_beyond_the_target_INTO_the_zone_is_rejected_by_name():
    """The case the rule exists for: the far-side observation point of a target
    standing at the zone's edge lies INSIDE the marked rectangle.

    Built deliberately rather than taken from the scenario, because on the real
    run the intruder stands well inside the annex and the approach comes from
    outside it, so the pruning that bites there is the fixture and wall check.
    This is the geometry that isolates the zone rule itself.
    """
    target = (RESTRICTED_ZONE.center[0] + RESTRICTED_ZONE.half[0] + 0.20,
              RESTRICTED_ZONE.center[1])
    plan = plan_standoff("visitor", target, (target[0] + 1.2, target[1]))
    rejected = [c for c in plan.candidates if not c.ok]
    assert any("restricted zone" in c.reason for c in rejected), \
        [(c.bearing_deg, c.reason) for c in rejected]
    for candidate in plan.candidates:
        if candidate.ok:
            assert not RESTRICTED_ZONE.contains(candidate.xy)


def test_a_standoff_behind_a_fixture_is_rejected():
    plan = plan_standoff("crate", CRATE_XY, (0.5, 0.5))
    rejected = [c for c in plan.candidates if not c.ok]
    assert rejected, "the crate sits near a shelf; something must be pruned"


def test_the_observation_angles_sweep_across_the_target():
    """The duck cannot orbit, so the angles are swept by the HEAD from a fixed
    standoff.  Adjacent angles must therefore differ in bearing."""
    duck = (0.0, 0.0)
    target = (1.0, 0.0)
    points = [observation_look_point("crate", target, angle, duck)
              for angle in (-26.0, 0.0, 26.0)]
    bearings = [math.atan2(p[1], p[0]) for p in points]
    assert bearings[0] < bearings[1] < bearings[2]
    for point in points:
        assert float(np.linalg.norm(np.asarray(point[:2]))) == pytest.approx(
            1.0, abs=1e-6)


def test_the_look_point_is_at_the_body_s_own_height():
    person = target_look_point("visitor", (0.0, 0.0))
    obj = target_look_point("crate", (0.0, 0.0))
    assert person[2] > obj[2] > 0.0


# -- the zone, measured every tick -------------------------------------------------
def test_the_zone_gap_is_signed_and_zero_on_the_boundary():
    centre = RESTRICTED_ZONE.center
    assert zone_gap_m(centre) < 0.0
    edge = (centre[0] + RESTRICTED_ZONE.half[0], centre[1])
    assert zone_gap_m(edge) == pytest.approx(0.0, abs=1e-9)
    outside = (centre[0] + RESTRICTED_ZONE.half[0] + 0.5, centre[1])
    assert zone_gap_m(outside) == pytest.approx(0.5, abs=1e-9)


# -- the interlock -----------------------------------------------------------------
def test_the_interlock_refuses_a_step_that_would_enter_the_zone():
    """THE SECOND INDEPENDENT GUARD, computed from a different quantity than the
    planner's, so one being wrong cannot produce a robot in the zone."""
    centre = RESTRICTED_ZONE.center
    just_outside = (centre[0] + RESTRICTED_ZONE.half[0] + 0.10, centre[1])
    interlock = build_interlock(
        duck_xy=just_outside, duck_yaw=math.pi, bodies={}, clearances={})
    assert interlock.blocked
    assert "restricted zone" in interlock.reason


def test_the_interlock_lets_the_duck_walk_AWAY_from_the_zone():
    """A REFUSAL MUST NOT BE A TRAP.  Turning in place is MEASURED to be
    unavailable, so a duck whose forward command is refused cannot turn away
    either - the yaw it needs comes from walking.  A refusal on heading alone
    would be permanent, and MEASURED, it was: the duck stood still for a whole
    40 s ceiling facing the annex."""
    centre = RESTRICTED_ZONE.center
    just_outside = (centre[0] + RESTRICTED_ZONE.half[0] + 0.10, centre[1])
    interlock = build_interlock(
        duck_xy=just_outside, duck_yaw=0.0, bodies={}, clearances={})
    assert not interlock.blocked


def test_the_zone_interlock_margin_is_the_duck_s_own_footprint():
    """It governs whether the duck may MOVE, which is a kinematic question, so
    it forbids exactly what the rule forbids and nothing more.  The planner's
    own margin is deliberately more generous - that is a different question."""
    assert ZONE_INTERLOCK_MARGIN_M == pytest.approx(DUCK_PLANAR_RADIUS)
    assert ZONE_STANDOFF_M + DUCK_PLANAR_RADIUS > ZONE_INTERLOCK_MARGIN_M


class _Body:
    def __init__(self, pos, present=True):
        self.pos = np.asarray(pos, dtype=float)
        self.present = present


def test_the_interlock_refuses_a_body_close_ahead():
    bodies = {"rosa": _Body((0.3, 0.0))}
    interlock = build_interlock(
        duck_xy=(0.0, 0.0), duck_yaw=0.0, bodies=bodies,
        clearances={"rosa": 0.10})
    assert interlock.blocked
    assert interlock.body == "rosa"


def test_the_interlock_ignores_a_body_behind_the_duck():
    bodies = {"rosa": _Body((-0.3, 0.0))}
    interlock = build_interlock(
        duck_xy=(0.0, 0.0), duck_yaw=0.0, bodies=bodies,
        clearances={"rosa": 0.10})
    assert not interlock.blocked


def test_the_interlock_ignores_a_body_the_duck_is_walking_away_from():
    """The same escape rule as the zone, for the same kinematic reason."""
    bodies = {"rosa": _Body((0.3, 0.0))}
    interlock = build_interlock(
        duck_xy=(0.0, 0.0), duck_yaw=0.0, bodies=bodies,
        clearances={"rosa": 0.10}, target_xy=(-2.0, 0.0))
    assert not interlock.blocked


def test_the_interlock_ignores_a_body_that_has_not_appeared_yet():
    bodies = {"crate": _Body((0.3, 0.0), present=False)}
    interlock = build_interlock(
        duck_xy=(0.0, 0.0), duck_yaw=0.0, bodies=bodies,
        clearances={"crate": 0.05})
    assert not interlock.blocked


def test_the_interlock_bar_sits_below_the_standoff_band():
    """It is a backstop for a failure of the approach controller, not the thing
    that normally stops the duck."""
    assert INTERLOCK_CLEARANCE_M < STANDOFF_MIN_M


# -- line of sight ------------------------------------------------------------------
def test_the_central_rack_really_blocks_a_sightline_across_it():
    bodies = {}
    blocker = los_blocked_by((-1.2, 0.0), (1.2, 0.0), bodies)
    assert blocker == "obs_rack_core"


def test_a_body_between_the_eye_and_the_target_blocks_it():
    bodies = {name: _Body((9.0, 9.0)) for name in
              ("rosa", "emil", "nadia", "visitor", "crate", "trolley")}
    bodies["rosa"] = _Body((0.5, -1.9))
    blocker = los_blocked_by((0.0, -1.9), (1.0, -1.9), bodies)
    assert blocker == "rosa"


def test_the_target_itself_never_counts_as_its_own_screen():
    bodies = {name: _Body((9.0, 9.0)) for name in
              ("rosa", "emil", "nadia", "visitor", "crate", "trolley")}
    bodies["crate"] = _Body((1.0, -1.9))
    blocker = los_blocked_by((0.0, -1.9), (1.0, -1.9), bodies,
                             exclude="crate")
    assert blocker == ""


def test_measured_positions_omit_bodies_that_have_not_appeared():
    """A parked body does not exist as far as every layer above the boundary is
    concerned, which is what 'it is not there yet' should mean."""
    early = measured_positions(bodies_at(0.5))
    assert "crate" not in early
    late = measured_positions(bodies_at(140.0))
    assert "crate" in late
