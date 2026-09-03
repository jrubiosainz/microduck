#!/usr/bin/env python3
"""Threat prediction: closest approach, the false alarm, and the squeeze.

``pps_threat`` is pure: constant-velocity closest-approach prediction from
RELATIVE motion, with no schedule and no simulator.  That is what lets every
boundary below be exercised directly instead of inferred from a rollout.

THE MARGINS ARE TESTED AT THEIR EDGES ON PURPOSE
--------------------------------------------------
A threshold nobody probes either side of is a number, not a decision.  Each
boundary test puts one case just inside and one just outside, so a constant
that moved would flip exactly one of them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pps_states import (ALERT_RANGE_M, BUFFER_M, PREDICT_HORIZON_S,
                        PREDICT_MARGIN_M, PREDICT_TTC_MAX_S,
                        SQUEEZE_SEPARATION_DEG)
from pps_threat import (Prediction, active, angle_separation, predict_all,
                        predict_one, priority, squeeze_pair)

# The closest approach a prediction must reach to count as an intrusion.
CPA_LIMIT = BUFFER_M - PREDICT_MARGIN_M
# The minimum closing speed, below which somebody is not arriving at all.
CLOSING_FLOOR = 0.015


class _Person:
    """The duck-side view of a body: a position, a velocity and presence."""

    def __init__(self, pos, velocity, present=True):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.present = present


class _Ward(_Person):
    pass


def predict(pos, velocity, ward=(0.0, 0.0), ward_velocity=(0.0, 0.0),
            name="x") -> Prediction:
    return predict_one(name, ward, ward_velocity, pos, velocity)


# -- the closest-approach arithmetic -----------------------------------------
def test_a_head_on_walker_has_zero_closest_approach():
    p = predict((1.0, 0.0), (-0.2, 0.0))
    assert p.cpa_m == pytest.approx(0.0, abs=1e-9)
    assert p.ttc_s == pytest.approx(5.0)
    assert p.range_m == pytest.approx(1.0)
    assert p.closing_mps == pytest.approx(0.2)
    assert p.intrusion


@pytest.mark.parametrize("offset", [0.4, 0.9, 1.4])
def test_closest_approach_equals_the_lateral_offset_of_a_parallel_pass(offset):
    """The whole point of CPA: how close they WILL come, not how close they are.

    Started at 1.5 m closing at 0.25 m/s, so the crossing happens at 6 s and
    lands inside the eight-second horizon rather than being clipped by it.
    """
    p = predict((1.5, offset), (-0.25, 0.0))
    assert p.ttc_s == pytest.approx(6.0, abs=1e-6)
    assert p.cpa_m == pytest.approx(offset, abs=1e-6)
    assert p.range_m > p.cpa_m


def test_a_receding_walker_is_never_an_intrusion():
    p = predict((1.0, 0.0), (0.2, 0.0))
    assert p.closing_mps < 0.0
    assert p.ttc_s == 0.0
    assert p.cpa_m == pytest.approx(p.range_m)
    assert not p.intrusion


def test_a_stationary_person_is_never_an_intrusion():
    """Standing inside the buffer is not arriving, however close it is."""
    p = predict((0.5, 0.0), (0.0, 0.0))
    assert p.ttc_s == 0.0
    assert p.closing_mps == pytest.approx(0.0)
    assert not p.intrusion


def test_time_to_closest_approach_is_clipped_to_the_horizon():
    """Predicting further than eight seconds is astrology, not anticipation."""
    p = predict((3.4, 0.0), (-0.01, 0.0))
    assert p.ttc_s == pytest.approx(PREDICT_HORIZON_S)
    assert not p.intrusion


def test_prediction_is_relative_so_a_fleeing_ward_cancels_an_approach():
    """A stranger matching the ward's pace never closes on her."""
    chasing = predict((1.0, 0.0), (-0.2, 0.0))
    matched = predict((1.0, 0.0), (-0.2, 0.0), ward_velocity=(-0.2, 0.0))
    assert chasing.intrusion
    assert not matched.intrusion
    assert matched.closing_mps == pytest.approx(0.0)
    assert matched.cpa_m == pytest.approx(matched.range_m)


@pytest.mark.parametrize("bearing_deg", [0.0, 45.0, 90.0, 179.0, -90.0, -135.0])
def test_bearing_is_measured_from_the_ward_to_the_person(bearing_deg):
    angle = math.radians(bearing_deg)
    pos = (2.0 * math.cos(angle), 2.0 * math.sin(angle))
    p = predict(pos, (0.0, 0.0))
    assert p.bearing_deg == pytest.approx(bearing_deg, abs=1e-6)


def test_bearing_is_unaffected_by_the_wards_own_position():
    """It is a relative bearing, so the same geometry gives the same answer."""
    here = predict((3.0, 0.0), (-0.2, 0.0), ward=(0.0, 0.0))
    there = predict((1.0, 2.0), (-0.2, 0.0), ward=(-2.0, 2.0))
    assert here.bearing_deg == pytest.approx(there.bearing_deg)
    assert here.cpa_m == pytest.approx(there.cpa_m)


def test_a_degenerate_zero_relative_velocity_does_not_divide_by_zero():
    p = predict((1.0, 1.0), (0.0, 0.0))
    assert p.ttc_s == 0.0
    assert np.isfinite(p.cpa_m)
    assert np.isfinite(p.closing_mps)


def test_a_person_standing_exactly_on_the_ward_is_finite():
    p = predict((0.0, 0.0), (0.0, 0.0))
    assert p.range_m == 0.0
    assert p.closing_mps == 0.0
    assert not p.intrusion


# -- the four conditions, each probed at its own boundary --------------------
@pytest.mark.parametrize("cpa,expected", [(CPA_LIMIT - 0.001, True),
                                          (CPA_LIMIT + 0.001, False)])
def test_the_predicted_approach_must_be_inside_the_buffer_by_the_margin(
        cpa, expected):
    """The margin is what makes rejecting a boundary graze robust.

    A person who will graze the buffer edge and pass is not an intrusion; the
    12 cm margin is what stops that being a coin flip.
    """
    p = predict((1.0, cpa), (-0.2, 0.0))
    assert p.cpa_m == pytest.approx(cpa, abs=1e-6)
    assert p.intrusion is expected


@pytest.mark.parametrize("range_m,expected", [(ALERT_RANGE_M - 0.01, True),
                                              (ALERT_RANGE_M + 0.01, False)])
def test_a_person_beyond_alert_range_is_not_even_predicted_on(range_m,
                                                              expected):
    p = predict((range_m, 0.0), (-0.6, 0.0))
    assert p.range_m == pytest.approx(range_m)
    assert p.intrusion is expected


@pytest.mark.parametrize("ttc,expected", [(PREDICT_TTC_MAX_S - 0.5, True),
                                          (PREDICT_TTC_MAX_S + 0.1, False)])
def test_an_arrival_too_far_off_is_not_an_intrusion_yet(ttc, expected):
    """Acting on a person twenty seconds away would spend the escort on nothing."""
    speed = 0.1
    p = predict((ttc * speed, 0.0), (-speed, 0.0))
    assert p.ttc_s == pytest.approx(ttc, abs=1e-6)
    assert p.intrusion is expected


@pytest.mark.parametrize("closing,expected", [(CLOSING_FLOOR - 0.0001, False),
                                              (CLOSING_FLOOR + 0.0001, True)])
def test_a_person_barely_closing_is_not_arriving(closing, expected):
    p = predict((0.1, 0.0), (-closing, 0.0))
    assert p.closing_mps == pytest.approx(closing, abs=1e-9)
    assert p.intrusion is expected


def test_the_false_alarm_geometry_is_rejected_on_closest_approach():
    """Piet's line: a straight near pass that never enters the buffer.

    Taken from the shipped route - he walks from (2.90, 1.55) to (-1.10, 2.10)
    while the ward is near her own line - his predicted closest approach is
    OUTSIDE the buffer entirely, which is what makes the dismissal a
    measurement rather than a timing accident.
    """
    ward = (0.55, -0.30)
    heading = np.array([-1.10 - 2.90, 2.10 - 1.55])
    heading /= float(np.linalg.norm(heading))
    for u in np.linspace(0.0, 1.0, 21):
        pos = np.array([2.90, 1.55]) + u * np.array([-4.00, 0.55])
        p = predict(pos, heading * 0.16, ward=ward, name="piet")
        assert not p.intrusion, (u, p.record())


def test_the_prediction_record_rounds_without_losing_the_decision():
    p = predict((1.0, 0.0), (-0.2, 0.0), name="dario")
    record = p.record()
    assert record["name"] == "dario"
    assert record["intrusion"] is True
    assert set(record) == {"name", "range_m", "cpa_m", "ttc_s", "bearing_deg",
                           "closing_mps", "intrusion"}
    assert record["range_m"] == pytest.approx(p.range_m, abs=5e-4)
    assert record["ttc_s"] == pytest.approx(p.ttc_s, abs=5e-4)


# -- ranking many people ------------------------------------------------------
@pytest.fixture()
def crowd():
    return {
        "mid": _Person((1.2, 0.0), (-0.3, 0.0)),      # ttc 4.0, intruding
        "near": _Person((0.6, 0.0), (-0.1, 0.0)),     # ttc 6.0, intruding
        "away": _Person((1.0, 0.0), (0.3, 0.0)),      # receding
        "far": _Person((3.4, 0.0), (-0.1, 0.0)),      # cpa fine, ttc clipped
    }


def test_predictions_put_the_intruders_first_then_the_soonest(crowd):
    order = [p.name for p in predict_all(_Ward((0, 0), (0, 0)), crowd)]
    assert order[:2] == ["mid", "near"], order
    assert set(order) == set(crowd)


def test_priority_selects_the_soonest_live_intrusion(crowd):
    chosen = priority(predict_all(_Ward((0, 0), (0, 0)), crowd))
    assert chosen is not None and chosen.name == "mid"
    assert [p.name for p in active(predict_all(_Ward((0, 0), (0, 0)), crowd))
            ] == ["mid", "near"]


def test_priority_is_none_when_nobody_is_intruding():
    quiet = {"a": _Person((3.0, 0.0), (0.1, 0.0))}
    predictions = predict_all(_Ward((0, 0), (0, 0)), quiet)
    assert active(predictions) == []
    assert priority(predictions) is None


def test_priority_breaks_a_tie_on_closest_approach():
    """Same arrival time, so the one who will come closer is chosen."""
    tie = {"grazing": _Person((1.0, 1.0), (-0.2, 0.0)),
           "direct": _Person((1.0, 0.0), (-0.2, 0.0))}
    chosen = priority(predict_all(_Ward((0, 0), (0, 0)), tie))
    assert chosen.name == "direct"


def test_an_absent_person_is_not_predicted_on(crowd):
    crowd["mid"].present = False
    names = [p.name for p in predict_all(_Ward((0, 0), (0, 0)), crowd)]
    assert "mid" not in names
    assert priority(predict_all(_Ward((0, 0), (0, 0)), crowd)).name == "near"


def test_an_excluded_person_is_not_predicted_on(crowd):
    """This is how an already-handled intruder stops re-opening an episode."""
    names = [p.name for p in predict_all(_Ward((0, 0), (0, 0)), crowd,
                                         exclude={"mid", "near"})]
    assert "mid" not in names and "near" not in names
    assert set(names) == {"away", "far"}


def test_excluding_everybody_yields_nothing(crowd):
    assert predict_all(_Ward((0, 0), (0, 0)), crowd,
                       exclude=set(crowd)) == []
    assert predict_all(_Ward((0, 0), (0, 0)), {}) == []


# -- the squeeze ---------------------------------------------------------------
@pytest.mark.parametrize("a,b,expected", [(0.0, 180.0, 180.0),
                                          (0.0, -180.0, 180.0),
                                          (170.0, -170.0, 20.0),
                                          (-170.0, 170.0, 20.0),
                                          (10.0, 100.0, 90.0),
                                          (0.0, 0.0, 0.0)])
def test_angle_separation_wraps_and_is_symmetric(a, b, expected):
    assert angle_separation(a, b) == pytest.approx(expected)
    assert angle_separation(b, a) == pytest.approx(expected)
    assert 0.0 <= angle_separation(a, b) <= 180.0


def test_two_people_from_opposite_bearings_are_a_squeeze():
    east = predict((1.0, 0.0), (-0.2, 0.0), name="east")
    west = predict((-1.0, 0.0), (0.2, 0.0), name="west")
    first, second, separation = squeeze_pair([east, west])
    assert {first.name, second.name} == {"east", "west"}
    assert separation == pytest.approx(180.0)
    assert separation >= SQUEEZE_SEPARATION_DEG


@pytest.mark.parametrize("separation,expected", [
    (SQUEEZE_SEPARATION_DEG + 1.0, True),
    (SQUEEZE_SEPARATION_DEG - 1.0, False)])
def test_the_separation_threshold_decides_whether_one_station_covers_both(
        separation, expected):
    """Below the threshold a single interpose station covers both people.

    There is nothing to choose between them, so it is not a squeeze - and
    escaping instead of interposing would be a worse response to one approach.
    """
    angle = math.radians(separation)
    first = predict((1.0, 0.0), (-0.2, 0.0), name="a")
    second = predict((math.cos(angle), math.sin(angle)),
                     (-0.2 * math.cos(angle), -0.2 * math.sin(angle)),
                     name="b")
    assert (squeeze_pair([first, second]) is not None) is expected


def test_a_single_person_is_never_a_squeeze():
    only = predict((1.0, 0.0), (-0.2, 0.0), name="only")
    assert squeeze_pair([only]) is None
    assert squeeze_pair([]) is None


def test_a_person_already_inside_the_buffer_stays_eligible():
    """TTC is zero for somebody who has arrived, and they still pinch.

    Callers pass the already-confirmed live set, so a squeeze must not be
    dropped merely because one half is no longer predicted to be arriving.
    """
    arrived = predict((0.3, 0.0), (0.0, 0.0), name="arrived")
    incoming = predict((-1.0, 0.0), (0.2, 0.0), name="incoming")
    assert arrived.ttc_s == 0.0 and not arrived.intrusion
    pair = squeeze_pair([arrived, incoming])
    assert pair is not None
    assert {pair[0].name, pair[1].name} == {"arrived", "incoming"}


def test_the_soonest_qualifying_pair_wins():
    """Scored on the sooner of the two arrivals, across every eligible pair."""
    soon = predict((1.0, 0.0), (-0.2, 0.0), name="soon")
    sooner = predict((-0.6, 0.0), (0.2, 0.0), name="sooner")
    later = predict((-2.0, 0.0), (0.2, 0.0), name="later")
    assert sooner.ttc_s < soon.ttc_s < later.ttc_s
    first, second, _ = squeeze_pair([soon, later, sooner])
    assert {first.name, second.name} == {"soon", "sooner"}


def test_the_shipped_squeeze_bearings_do_qualify():
    """Kwame at -145.76 deg and Tomas at 65.48 deg, as the run measured them."""
    separation = angle_separation(-145.76, 65.48)
    assert separation == pytest.approx(148.76, abs=0.02)
    assert separation >= SQUEEZE_SEPARATION_DEG
