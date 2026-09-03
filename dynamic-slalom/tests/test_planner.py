#!/usr/bin/env python3
"""The planner: prediction, corridor scoring, and the three ways to lose.

Pure geometry, so every property here is asserted on hand-built inputs with no
MuJoCo and no physics.  These are the tests that would have caught the two
worst bugs in this behavior's history — committing on a truncated prediction,
and a corridor line that receded as fast as the duck approached it.
"""

from __future__ import annotations

import numpy as np
import pytest

from slalom_plan import (
    Corridor,
    Track,
    choose_corridor,
    duck_at,
    duck_line,
    horizon_times,
    nearest_threat,
    predict_occupancy,
    score_corridor,
)
from slalom_states import (
    DUCK_PLANAR_RADIUS,
    LATERAL_OFFSETS,
    LATERAL_RATE_MPS,
    PREDICT_HORIZON_S,
    PREDICT_SAMPLES,
    PREDICTION_SLOP_M,
    SAFE_CLEARANCE_M,
    STATIC_MARGIN_M,
    TRUNCATED_SAFE_M,
)


def track(name="body", pos=(0.0, 0.0), vel=(0.0, 0.0), radius=0.26) -> Track:
    return Track(name=name, pos=np.array(pos, dtype=float),
                 velocity=np.array(vel, dtype=float), radius=radius)


# -- prediction ---------------------------------------------------------------
def test_constant_velocity_prediction_is_exactly_that():
    """A track predicts its own straight line, at every horizon sample."""
    t = track(pos=(1.0, -2.0), vel=(0.1, 0.25))
    for dt in horizon_times():
        expected = np.array([1.0 + 0.1 * dt, -2.0 + 0.25 * dt])
        assert np.allclose(t.predict(dt), expected)


def test_the_horizon_covers_the_declared_span():
    times = horizon_times()
    assert len(times) == PREDICT_SAMPLES
    assert times[-1] == pytest.approx(PREDICT_HORIZON_S, abs=1e-9)
    assert times[0] > 0.0, "a zero-second sample would score the present, not a prediction"


def test_a_stationary_body_is_predicted_to_stay_put():
    """The honest prediction about something measured once is that it is still."""
    t = track(vel=(0.0, 0.0))
    assert all(np.allclose(t.predict(dt), t.pos) for dt in horizon_times())


def test_predicted_occupancy_reports_every_body_at_every_sample():
    tracks = [track("a", (0.0, 0.0), (0.2, 0.0)),
              track("b", (1.0, 1.0), (0.0, -0.2))]
    occupancy = predict_occupancy(tracks)
    assert len(occupancy) == PREDICT_SAMPLES
    for sample in occupancy:
        assert set(sample["bodies"]) == {"a", "b"}


# -- the corridor's world line ------------------------------------------------
def test_a_corridor_line_is_fixed_in_the_world_not_relative_to_the_duck():
    """THE REGRESSION TEST FOR THE RECEDING-CORRIDOR BUG.

    A corridor scored from one position must denote the SAME world line when the
    duck has moved along it.  The original bug rebuilt the offset from the
    duck's current pose every tick, so the target stayed 0.26 m to the side of
    wherever the duck was — a line it could never reach.  Here the duck starts
    on the line's origin and walks 1 m toward the goal; the line's own point at
    that along-track distance must be exactly the offset away, laterally.
    """
    corridor = score_corridor(np.array([-4.0, 0.0]), 0.26, "left",
                              [track(pos=(9.0, 9.0))])
    a = corridor.line_point(0.0)
    b = corridor.line_point(1.0)
    # The two points are 1 m apart along the corridor's own direction.
    assert float(np.linalg.norm(b - a)) == pytest.approx(1.0, abs=1e-9)
    # And both sit exactly `offset` to the left of the unoffset line.
    base, direction = duck_line(np.array([-4.0, 0.0]), 0.0)
    normal = np.array([-direction[1], direction[0]])
    for along, point in ((0.0, a), (1.0, b)):
        expected = base + direction * along + normal * 0.26
        assert np.allclose(point, expected, atol=1e-9)


def test_a_corridor_without_a_scored_line_refuses_to_produce_points():
    with pytest.raises(ValueError):
        Corridor(side="left", offset_m=0.26).line_point(1.0)


def test_the_duck_cannot_teleport_sideways_in_the_prediction():
    """``duck_at`` ramps the offset in at the MEASURED lateral rate.

    The widest offered sidestep needs ``0.50 / 0.0475 = 10.5 s`` at that rate,
    which is the whole horizon - so even at the last sample the duck has only
    just arrived.  That is the physical fact the planner has to respect.
    """
    early = duck_at(np.array([0.0, 0.0]), 0.50, 1.0)
    assert abs(early[1]) < 0.50, "reached the full offset in one second"
    assert abs(early[1]) == pytest.approx(LATERAL_RATE_MPS * 1.0, abs=1e-9)
    # Far beyond the horizon the ramp saturates at the requested offset.
    assert abs(duck_at(np.array([0.0, 0.0]), 0.50, 40.0)[1]) == pytest.approx(
        0.50, abs=1e-9)
    # And the smallest offset IS reachable inside the horizon, or no corridor
    # could ever be committed to.
    reached = abs(duck_at(np.array([0.0, 0.0]), min(LATERAL_OFFSETS),
                          PREDICT_HORIZON_S)[1])
    assert reached == pytest.approx(min(LATERAL_OFFSETS), abs=1e-9)


# -- scoring ------------------------------------------------------------------
def test_clearance_is_a_surface_distance_not_a_centre_gap():
    """Both radii and the slop term come off the predicted centre distance."""
    # A body parked far to one side, so the worst sample is a clean geometry
    # problem rather than a closing conflict.
    body = track(pos=(0.0, 3.0), vel=(0.0, 0.0), radius=0.26)
    corridor = score_corridor(np.array([0.0, 0.0]), 0.0, "straight", [body])
    here = duck_at(np.array([0.0, 0.0]), 0.0, corridor.worst_at_s)
    centre_gap = float(np.linalg.norm(body.pos - here))
    assert corridor.worst_clearance_m == pytest.approx(
        centre_gap - 0.26 - DUCK_PLANAR_RADIUS - PREDICTION_SLOP_M, abs=1e-9)


def test_a_corridor_is_rejected_when_its_worst_moment_is_the_horizon_edge():
    """THE REGRESSION TEST FOR THE TRUNCATED-PREDICTION BUG.

    A body still approaching the duck's line at the END of the horizon has not
    been scored, it has been cut off: the conflict is still developing when the
    prediction stops looking, so the number is an artifact of where the window
    ended.  Committing on exactly this shape is what walked the duck into
    ``mara``.

    The body is placed so that it is STILL CLOSING at the last horizon sample
    and is only marginally clear there — below
    :data:`~slalom_states.TRUNCATED_SAFE_M` — which is precisely the situation in
    which the number cannot be trusted.  A body that is comfortably clear even
    at the edge is genuinely clear and is covered by the next test.
    """
    # Closing on the duck's line so that at t = horizon it is still short of it,
    # at a distance that leaves the surface clearance under the truncated bar.
    speed = 0.12
    reach = speed * PREDICT_HORIZON_S
    # Clearance at the edge = |start_y| - reach - radii - slop.  Solve for a
    # value just under TRUNCATED_SAFE_M.
    radii = 0.26 + DUCK_PLANAR_RADIUS + PREDICTION_SLOP_M
    start_y = -(reach + radii + TRUNCATED_SAFE_M - 0.10)
    far_and_closing = track(pos=(duck_at(np.array([0.0, 0.0]), 0.0,
                                         PREDICT_HORIZON_S)[0], start_y),
                            vel=(0.0, speed))
    corridor = score_corridor(np.array([0.0, 0.0]), 0.0, "straight",
                              [far_and_closing])
    assert corridor.worst_at_s == pytest.approx(horizon_times()[-1]), (
        "the fixture must bottom out at the horizon edge for this test to "
        "exercise the truncation rule")
    assert corridor.worst_clearance_m < TRUNCATED_SAFE_M
    assert corridor.worst_clearance_m > SAFE_CLEARANCE_M, (
        "the fixture must be rejected by the TRUNCATION rule, not by the "
        "ordinary safety bar, or it proves nothing")
    assert not corridor.safe
    assert "truncated" in corridor.reject_reason


def test_a_corridor_clear_even_at_the_horizon_edge_survives():
    """The truncation rule must not reject a corridor that is simply empty."""
    harmless = track(pos=(0.0, 40.0), vel=(0.0, 0.0))
    corridor = score_corridor(np.array([0.0, 0.0]), 0.0, "straight",
                              [harmless])
    assert corridor.worst_clearance_m > TRUNCATED_SAFE_M
    assert corridor.safe, corridor.reject_reason


def test_a_corridor_through_a_static_obstacle_is_rejected():
    """``obs_crate_nw`` sits at (-3.05, 1.32) with a 0.34 x 0.26 half-extent."""
    corridor = score_corridor(np.array([-3.9, 1.32]), 0.0, "straight",
                              [track(pos=(0.0, 40.0))])
    assert not corridor.safe
    assert corridor.reject_reason.startswith("static")
    assert corridor.static_gap_m < STATIC_MARGIN_M


def test_an_unsafe_corridor_names_the_body_and_the_moment():
    closing = track("cart", pos=(0.6, 0.0), vel=(0.0, 0.0), radius=0.48)
    corridor = score_corridor(np.array([0.0, 0.0]), 0.0, "straight", [closing])
    assert not corridor.safe
    assert corridor.worst_body == "cart"
    assert corridor.worst_clearance_m < SAFE_CLEARANCE_M


# -- choosing -----------------------------------------------------------------
def test_both_hands_are_always_scored_even_when_the_first_is_fine():
    """The rejected side must be reported, or a choice cannot be justified."""
    decision = choose_corridor(np.array([-4.0, 0.0]),
                               [track(pos=(-2.0, -0.9), vel=(0.0, 0.30))])
    assert decision.rejected is not None
    assert decision.corridor is not None
    assert decision.rejected.side != decision.corridor.side
    assert len(decision.all_corridors) == 2 * len(LATERAL_OFFSETS)


def test_neither_side_safe_produces_a_wait_with_its_reasons():
    """Two bodies converging from both hands leave nowhere to go."""
    decision = choose_corridor(
        np.array([0.0, 0.0]),
        [track("north", pos=(0.7, 0.55), vel=(0.0, -0.05), radius=0.48),
         track("south", pos=(0.7, -0.55), vel=(0.0, 0.05), radius=0.48)])
    assert decision.side == "wait"
    assert decision.corridor is None
    # Even a wait names what it rejected, or the refusal is unexplained.
    assert decision.rejected is not None
    assert decision.rejected.reject_reason


def test_the_winner_has_the_greatest_worst_case_clearance():
    decision = choose_corridor(np.array([-4.0, 0.0]),
                               [track(pos=(-2.0, -0.9), vel=(0.0, 0.30))])
    safe = [c for c in decision.all_corridors if c.safe]
    assert decision.corridor.worst_clearance_m == pytest.approx(
        max(c.worst_clearance_m for c in safe))


def test_ties_break_toward_the_smaller_sidestep():
    """A smaller sidestep is cheaper in both course and time, so it wins ties."""
    empty = [track(pos=(0.0, 60.0))]
    decision = choose_corridor(np.array([0.0, 0.0]), empty)
    assert abs(decision.corridor.offset_m) == pytest.approx(
        min(LATERAL_OFFSETS))


# -- threats ------------------------------------------------------------------
def test_a_body_that_cannot_conflict_is_never_a_threat():
    far = track(pos=(0.0, 30.0), vel=(0.0, 0.0))
    name, ttc, range_m = nearest_threat(np.array([0.0, 0.0]), 0.0, [far])
    assert name == ""
    assert not np.isfinite(ttc)


def test_a_body_beyond_the_engagement_range_is_not_yet_a_threat():
    """Deciding outside the horizon that supports the decision is a guess."""
    from slalom_states import THREAT_RANGE_M
    distant = track(pos=(THREAT_RANGE_M + 1.0, 0.0), vel=(0.0, 0.0))
    name, _, _ = nearest_threat(np.array([0.0, 0.0]), 0.0, [distant])
    assert name == ""


def test_the_nearest_threat_is_the_worst_predicted_one():
    mild = track("mild", pos=(1.2, 0.9), vel=(0.0, 0.0))
    severe = track("severe", pos=(1.0, 0.2), vel=(0.0, 0.0))
    name, _, _ = nearest_threat(np.array([0.0, 0.0]), 0.0, [mild, severe])
    assert name == "severe"
