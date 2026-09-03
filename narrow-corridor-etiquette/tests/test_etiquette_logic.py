#!/usr/bin/env python3
"""Pure decision-logic tests: geometry, predictor, alcove scoring, machine.

No MuJoCo, no ONNX, no rendering.  Everything here runs on the same code the
rollout runs, driven by explicit inputs.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from corridor import (  # noqa: E402
    ADULT_LATERAL_HALF,
    ALCOVES,
    ALCOVE_BY_NAME,
    CENTER_PASSAGE_HALF,
    CLEAR_ABS_Y,
    CORRIDOR_HALF_WIDTH,
    CORRIDOR_X_MAX,
    CORRIDOR_X_MIN,
    DESTINATION_X,
    DUCK_LATERAL_HALF,
    DUCK_PLANAR_RADIUS,
    REJOIN_TOLERANCE_M,
    SAFE_PASSING_GAP_M,
    START_X,
    at_destination,
    center_passage_intrusion,
    clears_center_passage,
    corridor_passing_geometry,
    counterfactual_pass_clearance,
    duck_span_y,
    local_half_width,
    plain_corridor_max_trunk_abs_y,
    wall_clearance,
)
from encounter import (  # noqa: E402
    APPROACH_SPEED_MPS,
    CLEAR_RANGE_M,
    CONCURRENT_LEG_PESSIMISM,
    CRUISE_SPEED_MPS,
    DETECT_HOLD_S,
    DETECT_HORIZON_S,
    LATERAL_DEAD_TIME_S,
    REACH_MARGIN_S,
    SELECT_HOLD_S,
    SETTLE_S,
    STATES,
    STATIONARY_STATES,
    UNSAFE_PROXIMITY_M,
    VX_CRUISE,
    VX_MIN_EFFECTIVE,
    VY_MIN_EFFECTIVE_LEFT,
    VY_MIN_EFFECTIVE_RIGHT,
    VY_PULLOVER_LEFT,
    VY_PULLOVER_RIGHT,
    VY_SPEED_MPS,
    choose_alcove,
    most_urgent,
    predict_encounter,
    predict_encounters,
    score_alcove,
)
from etiquette_model import (  # noqa: E402
    PARK_TOLERANCE_M,
    EtiquetteController,
    EtiquetteMachine,
)
from people import PEDESTRIANS, PERSON_NAMES, people_at  # noqa: E402


@dataclass
class FakePerson:
    """A person stub with only what the predictor reads."""

    name: str = "test"
    pos: np.ndarray = None
    vel: np.ndarray = None
    direction: float = -1.0
    half_length: float = 0.104

    def __post_init__(self):
        if self.pos is None:
            self.pos = np.array([2.0, 0.0])
        if self.vel is None:
            self.vel = np.array([-0.42, 0.0])


# ---------------------------------------------------------------- geometry
class TestCorridorGeometry:
    def test_the_corridor_is_too_narrow_to_pass_abreast(self):
        """The premise of the whole behavior, as arithmetic."""
        geometry = corridor_passing_geometry()
        assert geometry["fits_at_all"], "bodies must physically fit at all"
        assert not geometry["fits_safely"], (
            "if a safe side-by-side pass were possible the duck would have no "
            "reason to pull over and the scenario would be vacuous")
        assert geometry["shortfall_m"] > 0.0

    def test_passing_geometry_uses_lateral_half_widths_not_radii(self):
        """Over-stating either body would fake the premise."""
        geometry = corridor_passing_geometry()
        assert geometry["duck_lateral_half_m"] == DUCK_LATERAL_HALF
        assert geometry["adult_lateral_half_m"] == ADULT_LATERAL_HALF
        assert DUCK_LATERAL_HALF < DUCK_PLANAR_RADIUS, (
            "the exact lateral half-width must be smaller than the "
            "conservative bounding radius, or the premise is being flattered")

    def test_a_wider_corridor_would_permit_a_safe_pass(self):
        """The verdict tracks the geometry rather than being hard-coded."""
        wide = corridor_passing_geometry(half_width=0.45)
        assert wide["fits_safely"]

    def test_the_plain_corridor_cannot_clear_the_passage_anywhere(self):
        """Pulling over is the only option, not merely the better one."""
        assert plain_corridor_max_trunk_abs_y() < CLEAR_ABS_Y

    def test_center_passage_admits_an_adult_with_the_safe_gap(self):
        assert CENTER_PASSAGE_HALF == pytest.approx(
            ADULT_LATERAL_HALF + SAFE_PASSING_GAP_M)

    def test_clears_center_passage_is_graded_on_the_whole_footprint(self):
        edge = CENTER_PASSAGE_HALF + DUCK_PLANAR_RADIUS
        assert not clears_center_passage(edge - 0.001)
        assert clears_center_passage(edge + 0.001)
        assert clears_center_passage(-(edge + 0.001))

    def test_intrusion_is_signed_and_continuous(self):
        deep = center_passage_intrusion(0.0)
        shallow = center_passage_intrusion(0.25)
        clear = center_passage_intrusion(CLEAR_ABS_Y + 0.05)
        assert deep > shallow > 0.0 > clear

    def test_duck_span_is_symmetric_about_the_trunk(self):
        low, high = duck_span_y(0.1)
        assert high - low == pytest.approx(2 * DUCK_PLANAR_RADIUS)

    def test_local_half_width_opens_up_inside_an_alcove(self):
        alcove = ALCOVE_BY_NAME["bay_open"]
        assert local_half_width(alcove.center_x, alcove.side) > CORRIDOR_HALF_WIDTH
        assert local_half_width(alcove.center_x, -alcove.side) == pytest.approx(
            CORRIDOR_HALF_WIDTH)

    def test_local_half_width_treats_an_obstruction_as_wall(self):
        alcove = ALCOVE_BY_NAME["bay_crates"]
        assert local_half_width(alcove.center_x, alcove.side) == pytest.approx(
            alcove.blocked_from)
        assert alcove.usable_outer_y < alcove.outer_y

    def test_wall_clearance_is_positive_on_the_centreline(self):
        assert wall_clearance(0.0, 0.0) > 0.0

    def test_destination_is_inside_the_corridor(self):
        assert CORRIDOR_X_MIN < START_X < DESTINATION_X < CORRIDOR_X_MAX
        assert at_destination(DESTINATION_X)
        assert not at_destination(DESTINATION_X - 0.01)

    def test_counterfactual_clearance_is_negative_down_the_middle(self):
        """Two bodies both near the centreline cannot pass."""
        assert counterfactual_pass_clearance(0.0, 0.02) < 0.0

    def test_counterfactual_clearance_turns_positive_when_far_apart(self):
        assert counterfactual_pass_clearance(0.0, 0.40) > 0.0


class TestAlcoveGeometry:
    def test_exactly_two_alcoves_can_clear_the_passage(self):
        usable = [a for a in ALCOVES if a.clears_passage]
        assert len(usable) == 2
        assert {a.name for a in usable} == {"bay_open", "bay_far"}

    def test_the_shallow_bay_fails_on_depth(self):
        alcove = ALCOVE_BY_NAME["bay_shallow"]
        assert not alcove.blocked
        assert alcove.max_trunk_abs_y < CLEAR_ABS_Y
        assert alcove.clearance_headroom_m < 0.0

    def test_the_crates_bay_fails_on_the_obstruction_not_the_wall(self):
        """Its shell is deep enough; the crates are what disqualify it."""
        alcove = ALCOVE_BY_NAME["bay_crates"]
        assert alcove.blocked
        assert alcove.outer_y - CORRIDOR_HALF_WIDTH >= 0.30, (
            "the recess itself must be deep enough to be usable if empty")
        assert alcove.outer_y - DUCK_PLANAR_RADIUS >= CLEAR_ABS_Y, (
            "an empty recess of this depth would clear the passage")
        assert not alcove.clears_passage, "with crates it must not"

    def test_park_points_clear_the_passage_with_headroom(self):
        for alcove in ALCOVES:
            if not alcove.clears_passage:
                continue
            assert clears_center_passage(alcove.park_y)
            assert abs(alcove.park_y) + DUCK_PLANAR_RADIUS <= (
                alcove.usable_outer_y + 1e-9)

    def test_park_point_sits_between_its_two_limits(self):
        for alcove in ALCOVES:
            if not alcove.clears_passage:
                continue
            assert CLEAR_ABS_Y < abs(alcove.park_y) < alcove.max_trunk_abs_y

    def test_mouths_are_longer_than_the_footprint(self):
        for alcove in ALCOVES:
            assert alcove.x_headroom_m > 0.0

    def test_entry_and_park_stations_are_inside_the_mouth(self):
        """Both stations lie between the cheeks, and the usable bays fit.

        An unusable bay is NOT required to fit the footprint: ``bay_shallow``
        is only 0.10 m deep, so no station inside it can hold the duck without
        its outline crossing the recess's back wall.  That is exactly why the
        scorer refuses it, and asserting the opposite here would contradict the
        behavior.
        """
        for alcove in ALCOVES:
            low, high = alcove.x_span
            assert low <= alcove.entry_x <= high
            assert low <= alcove.park_x <= high
            if alcove.clears_passage:
                assert alcove.footprint_inside(alcove.park_x, alcove.park_y)
                assert alcove.footprint_inside(alcove.entry_x, alcove.park_y)
            else:
                assert not alcove.footprint_inside(
                    alcove.park_x, alcove.side * CLEAR_ABS_Y), (
                    "an unusable bay must not be able to hold a footprint at "
                    "the depth the passage requires")

    def test_same_wall_mouths_never_overlap(self):
        """Overlapping mouths would merge into one opening in the scene."""
        for side in (+1, -1):
            spans = sorted(a.x_span for a in ALCOVES if a.side == side)
            for first, second in zip(spans, spans[1:]):
                assert first[1] <= second[0], (
                    f"alcove mouths {first} and {second} overlap on wall {side}")

    def test_sightline_span_follows_the_mouth_and_the_depth(self):
        """A longer mouth or a shallower park point sees more corridor."""
        alcove = ALCOVE_BY_NAME["bay_open"]
        assert alcove.sightline_half_span_m > alcove.half_length_x
        assert math.isfinite(alcove.sightline_half_span_m)


# --------------------------------------------------------------- predictor
class TestEncounterPrediction:
    def test_a_head_on_adult_is_approaching(self):
        person = FakePerson(pos=np.array([3.0, 0.0]),
                            vel=np.array([-0.42, 0.0]))
        encounter = predict_encounter(person, (0.0, 0.0))
        assert encounter.approaching
        assert encounter.head_on
        assert encounter.closing_speed_mps == pytest.approx(
            CRUISE_SPEED_MPS + 0.42)

    def test_time_to_meet_shrinks_as_the_gap_closes(self):
        near = predict_encounter(
            FakePerson(pos=np.array([1.0, 0.0])), (0.0, 0.0))
        far = predict_encounter(
            FakePerson(pos=np.array([4.0, 0.0])), (0.0, 0.0))
        assert near.time_to_meet_s < far.time_to_meet_s

    def test_an_adult_walking_away_faster_is_not_an_encounter(self):
        person = FakePerson(pos=np.array([1.0, 0.0]),
                            vel=np.array([+0.42, 0.0]), direction=+1.0)
        encounter = predict_encounter(person, (0.0, 0.0))
        assert not encounter.approaching
        assert encounter.time_to_meet_s == float("inf")

    def test_a_slower_adult_ahead_is_still_an_encounter(self):
        """Overtaking falls out of the closing-speed arithmetic."""
        person = FakePerson(pos=np.array([1.0, 0.0]),
                            vel=np.array([+0.05, 0.0]), direction=+1.0)
        encounter = predict_encounter(person, (0.0, 0.0))
        assert encounter.approaching
        assert not encounter.head_on

    def test_an_adult_behind_and_catching_up_is_an_encounter(self):
        person = FakePerson(pos=np.array([-1.0, 0.0]),
                            vel=np.array([+0.42, 0.0]), direction=+1.0)
        encounter = predict_encounter(person, (0.0, 0.0))
        assert encounter.approaching
        assert encounter.head_on is False or encounter.head_on is True

    def test_range_accounts_for_both_bodies(self):
        person = FakePerson(pos=np.array([2.0, 0.0]))
        encounter = predict_encounter(person, (0.0, 0.0))
        assert encounter.range_m == pytest.approx(
            2.0 - person.half_length - DUCK_PLANAR_RADIUS)

    def test_predictor_integrates_measured_speed_not_the_command(self):
        """Using the command would over-state progress threefold."""
        assert CRUISE_SPEED_MPS < 0.5 * VX_CRUISE

    def test_most_urgent_ignores_a_pass_that_would_be_safe_anyway(self):
        person = FakePerson(pos=np.array([1.0, 0.60]))
        assert most_urgent({"test": person}, (0.0, 0.0)) is None

    def test_most_urgent_ignores_an_encounter_beyond_the_horizon(self):
        far = 0.5 + (DETECT_HORIZON_S + 4.0) * (CRUISE_SPEED_MPS + 0.42)
        person = FakePerson(pos=np.array([far, 0.0]))
        assert most_urgent({"test": person}, (0.0, 0.0)) is None

    def test_most_urgent_picks_the_soonest(self):
        people = {
            "near": FakePerson(name="near", pos=np.array([1.5, 0.0])),
            "far": FakePerson(name="far", pos=np.array([3.0, 0.0])),
        }
        assert most_urgent(people, (0.0, 0.0)).name == "near"

    def test_encounters_are_sorted_soonest_first(self):
        people = {
            "far": FakePerson(name="far", pos=np.array([3.0, 0.0])),
            "near": FakePerson(name="near", pos=np.array([1.5, 0.0])),
        }
        order = [e.name for e in predict_encounters(people, (0.0, 0.0))]
        assert order == ["near", "far"]

    def test_detection_horizon_covers_the_whole_manoeuvre(self):
        """Not a taste parameter: it must fit the measured worst case."""
        deepest = max(abs(a.park_y) for a in ALCOVES if a.clears_passage)
        lateral = deepest / VY_SPEED_MPS
        manoeuvre = (LATERAL_DEAD_TIME_S + SETTLE_S
                     + CONCURRENT_LEG_PESSIMISM * lateral)
        assert DETECT_HORIZON_S >= manoeuvre + DETECT_HOLD_S + SELECT_HOLD_S

    def test_detection_happens_well_before_unsafe_proximity(self):
        """Detection range must exceed the range at which it is too late."""
        closing = CRUISE_SPEED_MPS + 0.42
        assert DETECT_HORIZON_S * closing > UNSAFE_PROXIMITY_M * 3.0


# ----------------------------------------------------------- alcove scoring
class TestAlcoveScoring:
    def _encounter(self, adult_x=4.0, duck=(-1.1, 0.0)):
        person = FakePerson(pos=np.array([adult_x, 0.02]),
                            vel=np.array([-0.42, 0.0]))
        return predict_encounter(person, duck)

    def test_a_shallow_bay_is_refused_for_clearance(self):
        encounter = self._encounter()
        score = score_alcove(ALCOVE_BY_NAME["bay_shallow"], encounter, (-1.1, 0.0))
        assert not score.viable
        assert any("shallow" in reason for reason in score.reasons)

    def test_a_blocked_bay_is_refused_for_its_obstruction(self):
        encounter = self._encounter()
        score = score_alcove(ALCOVE_BY_NAME["bay_crates"], encounter, (-1.1, 0.0))
        assert not score.viable
        assert any("obstructed" in reason for reason in score.reasons)

    def test_the_unusable_bays_are_refused_while_still_reachable(self):
        """The refusal must be about geometry, not about distance."""
        encounter = self._encounter()
        for name in ("bay_shallow", "bay_crates"):
            score = score_alcove(ALCOVE_BY_NAME[name], encounter, (-1.1, 0.0))
            assert score.reachable, (
                f"{name} must be reachable, or its rejection proves nothing "
                "about the duck's judgement of clearance")
            assert not score.behind
            assert not score.clears_passage

    def test_an_open_bay_in_reach_is_viable(self):
        encounter = self._encounter()
        score = score_alcove(ALCOVE_BY_NAME["bay_open"], encounter, (-1.1, 0.0))
        assert score.viable
        assert score.time_margin_s >= REACH_MARGIN_S

    def test_a_bay_behind_the_duck_is_refused(self):
        encounter = self._encounter(duck=(2.0, 0.0))
        score = score_alcove(ALCOVE_BY_NAME["bay_open"], encounter, (2.0, 0.0))
        assert score.behind
        assert not score.viable

    def test_an_unreachable_bay_is_refused_for_time(self):
        person = FakePerson(pos=np.array([0.9, 0.02]),
                            vel=np.array([-0.42, 0.0]))
        encounter = predict_encounter(person, (-1.1, 0.0))
        score = score_alcove(ALCOVE_BY_NAME["bay_far"], encounter, (-1.1, 0.0))
        assert not score.viable
        assert any("not reachable" in reason for reason in score.reasons)

    def test_choose_alcove_selects_a_viable_candidate(self):
        decision = choose_alcove(self._encounter(), (-1.1, 0.0))
        assert decision.selected is not None
        assert decision.selected.viable
        assert decision.selected.clears_passage

    def test_choose_alcove_never_selects_an_unusable_bay(self):
        for adult_x in (2.0, 3.0, 4.0, 5.0, 6.0):
            decision = choose_alcove(self._encounter(adult_x), (-1.1, 0.0))
            if decision.selected is not None:
                assert decision.selected.clears_passage

    def test_every_alcove_is_scored(self):
        decision = choose_alcove(self._encounter(), (-1.1, 0.0))
        assert decision.considered == len(ALCOVES)
        assert len(decision.candidates) == len(ALCOVES)

    def test_rejections_carry_reasons(self):
        decision = choose_alcove(self._encounter(), (-1.1, 0.0))
        for score in decision.rejected:
            assert score.reasons

    def test_scoring_uses_the_slower_measured_lateral_speed(self):
        """Using the faster side would flatter a marginal bay."""
        assert VY_SPEED_MPS <= 0.1306

    def test_concurrency_factor_exceeds_the_worst_measured_ratio(self):
        """Measured parked/max-leg ratios peak at 1.28 for the right-hand entry."""
        assert CONCURRENT_LEG_PESSIMISM >= 1.28

    def test_sequential_legs_cost_far_more_than_the_measured_manoeuvre(self):
        """Pins the modelling defect the concurrency factor exists to fix.

        Charging the forward and lateral legs one after the other describes a
        manoeuvre the controller never performs.  MEASURED: the real pull-over
        completes in 0.73-1.28x the LONGER single leg, never in their sum, so a
        sequential model over-charges every candidate and refuses bays the duck
        reaches comfortably.
        """
        encounter = self._encounter()
        alcove = ALCOVE_BY_NAME["bay_open"]
        duck = (-1.1, 0.0)
        concurrent = score_alcove(alcove, encounter, duck)
        forward = (alcove.entry_x - duck[0]) / APPROACH_SPEED_MPS
        lateral = abs(alcove.park_y - duck[1]) / VY_SPEED_MPS
        sequential = 2 * LATERAL_DEAD_TIME_S + SETTLE_S + forward + lateral
        assert concurrent.viable
        assert sequential > concurrent.travel_time_s, (
            "the sequential model must over-charge the manoeuvre")
        assert sequential - concurrent.travel_time_s > 1.0, (
            "and it must do so by enough to change decisions")

    def test_a_sequential_model_would_refuse_a_bay_the_duck_reaches(self):
        """The same defect, shown changing an actual verdict."""
        person = FakePerson(pos=np.array([3.9, 0.02]),
                            vel=np.array([-0.42, 0.0]))
        duck = (-1.1, 0.0)
        encounter = predict_encounter(person, duck)
        alcove = ALCOVE_BY_NAME["bay_open"]
        concurrent = score_alcove(alcove, encounter, duck)
        forward = (alcove.entry_x - duck[0]) / APPROACH_SPEED_MPS
        lateral = abs(alcove.park_y - duck[1]) / VY_SPEED_MPS
        sequential = 2 * LATERAL_DEAD_TIME_S + SETTLE_S + forward + lateral
        assert concurrent.viable, "the duck does reach this bay in time"
        assert sequential + REACH_MARGIN_S > concurrent.time_available_s, (
            "a sequential model would have refused it")


# ----------------------------------------------------------- state machine
def _run_machine(script, ctrl_hz=50.0):
    """Drive a machine through a list of per-tick keyword dicts."""
    machine = EtiquetteMachine(ctrl_hz=ctrl_hz)
    states = []
    dt = 1.0 / ctrl_hz
    for index, kwargs in enumerate(script):
        state, _ = machine.update(index * dt, **kwargs)
        states.append(state)
    return machine, states


def _ticks(count, **kwargs):
    return [dict(kwargs) for _ in range(count)]


class TestStateMachine:
    def _encounter(self, adult_x=4.0, duck=(-1.1, 0.0)):
        person = FakePerson(pos=np.array([adult_x, 0.02]),
                            vel=np.array([-0.42, 0.0]))
        return predict_encounter(person, duck)

    def test_starts_cruising(self):
        machine = EtiquetteMachine()
        assert machine.state == "CRUISE"

    def test_cruise_holds_without_an_encounter(self):
        _, states = _run_machine(
            _ticks(100, duck_xy=(-1.5, 0.0), duck_speed_mps=0.14))
        assert set(states) == {"CRUISE"}

    def test_an_encounter_moves_to_detect(self):
        machine, states = _run_machine(
            _ticks(3, duck_xy=(-1.1, 0.0), duck_speed_mps=0.14,
                   encounter=self._encounter()))
        assert states[0] == "DETECT"

    def test_detect_dwells_before_selecting(self):
        script = _ticks(200, duck_xy=(-1.1, 0.0), duck_speed_mps=0.14,
                        encounter=self._encounter())
        machine, states = _run_machine(script)
        first_select = states.index("SELECT_ALCOVE")
        assert first_select / 50.0 >= DETECT_HOLD_S

    def test_the_full_cycle_runs_in_order(self):
        encounter = self._encounter()
        script = []
        script += _ticks(60, duck_xy=(-1.1, 0.0), duck_speed_mps=0.14,
                         encounter=encounter)      # DETECT + SELECT
        script += _ticks(40, duck_xy=(-1.1, 0.0), duck_speed_mps=0.14,
                         encounter=encounter)
        park = ALCOVE_BY_NAME["bay_open"]
        script += _ticks(60, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.0)        # PULL_OVER -> YIELD
        script += _ticks(60, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.0, person_range_m=0.2,
                         person_receding=False, person_behind=False)
        script += _ticks(60, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.0, person_range_m=1.0,
                         person_receding=True, person_behind=True)
        script += _ticks(40, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.0)        # CLEAR -> REJOIN
        script += _ticks(20, duck_xy=(park.park_x, 0.0), duck_speed_mps=0.1)
        machine, states = _run_machine(script)
        seen = [state for index, state in enumerate(states)
                if index == 0 or state != states[index - 1]]
        assert seen[:7] == ["DETECT", "SELECT_ALCOVE", "PULL_OVER", "YIELD",
                            "CLEAR", "REJOIN", "RESUME"]
        assert machine.completed_cycles == 1

    def test_pull_over_will_not_complete_outside_the_mouth(self):
        """The defect that drove the robot into a cheek."""
        encounter = self._encounter()
        park = ALCOVE_BY_NAME["bay_open"]
        low, _high = park.x_span
        outside = low - 0.30           # correct y, wrong x
        script = _ticks(120, duck_xy=(-1.1, 0.0), duck_speed_mps=0.14,
                        encounter=encounter)
        script += _ticks(200, duck_xy=(outside, park.park_y),
                         duck_speed_mps=0.0)
        machine, states = _run_machine(script)
        assert machine.state == "PULL_OVER" or "pull_over_timeout" in machine.timeouts

    def test_pull_over_will_not_complete_while_still_moving(self):
        encounter = self._encounter()
        park = ALCOVE_BY_NAME["bay_open"]
        script = _ticks(120, duck_xy=(-1.1, 0.0), duck_speed_mps=0.14,
                        encounter=encounter)
        script += _ticks(100, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.30)
        machine, states = _run_machine(script)
        assert "YIELD" not in states

    def test_pull_over_will_not_complete_inside_the_passage(self):
        encounter = self._encounter()
        park = ALCOVE_BY_NAME["bay_open"]
        script = _ticks(120, duck_xy=(-1.1, 0.0), duck_speed_mps=0.14,
                        encounter=encounter)
        script += _ticks(100, duck_xy=(park.park_x, 0.05), duck_speed_mps=0.0)
        machine, states = _run_machine(script)
        assert "YIELD" not in states

    def _reach_yield(self, encounter=None):
        encounter = encounter or self._encounter()
        park = ALCOVE_BY_NAME["bay_open"]
        script = _ticks(120, duck_xy=(-1.1, 0.0), duck_speed_mps=0.14,
                        encounter=encounter)
        script += _ticks(20, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.0)
        return script, park

    def test_yield_does_not_release_while_the_adult_is_alongside(self):
        script, park = self._reach_yield()
        script += _ticks(400, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.0, person_range_m=0.15,
                         person_receding=False, person_behind=False)
        machine, states = _run_machine(script)
        assert machine.state == "YIELD"

    def test_yield_does_not_release_on_range_alone(self):
        """Being far away is not the same as having gone past."""
        script, park = self._reach_yield()
        script += _ticks(400, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.0, person_range_m=3.0,
                         person_receding=True, person_behind=False)
        machine, states = _run_machine(script)
        assert machine.state == "YIELD"

    def test_yield_does_not_release_before_the_clearance_range(self):
        script, park = self._reach_yield()
        script += _ticks(400, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.0,
                         person_range_m=CLEAR_RANGE_M - 0.05,
                         person_receding=True, person_behind=True)
        machine, states = _run_machine(script)
        assert machine.state == "YIELD"

    def test_yield_releases_once_the_adult_is_past_and_clear(self):
        script, park = self._reach_yield()
        script += _ticks(200, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.0,
                         person_range_m=CLEAR_RANGE_M + 0.10,
                         person_receding=True, person_behind=True)
        machine, states = _run_machine(script)
        assert "CLEAR" in states

    def test_rejoin_requires_the_centreline(self):
        script, park = self._reach_yield()
        script += _ticks(200, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.0,
                         person_range_m=CLEAR_RANGE_M + 0.10,
                         person_receding=True, person_behind=True)
        script += _ticks(100, duck_xy=(park.park_x, park.park_y),
                         duck_speed_mps=0.1)
        machine, states = _run_machine(script)
        assert machine.state == "REJOIN"
        assert machine.completed_cycles == 0

    def test_reaching_the_destination_finishes(self):
        machine, states = _run_machine(
            _ticks(5, duck_xy=(DESTINATION_X + 0.05, 0.0),
                   duck_speed_mps=0.14))
        assert machine.state == "DONE"

    def test_done_is_terminal(self):
        script = _ticks(5, duck_xy=(DESTINATION_X + 0.05, 0.0),
                        duck_speed_mps=0.14)
        script += _ticks(50, duck_xy=(DESTINATION_X + 0.05, 0.0),
                         duck_speed_mps=0.14, encounter=self._encounter())
        machine, states = _run_machine(script)
        assert machine.state == "DONE"

    def test_the_counterfactual_is_recorded_at_detection(self):
        encounter = self._encounter()
        machine, _ = _run_machine(
            _ticks(3, duck_xy=(-1.1, 0.0), duck_speed_mps=0.14,
                   encounter=encounter))
        assert machine._cycle["counterfactual_clearance_m"] == pytest.approx(
            encounter.counterfactual_clearance_m)

    def test_a_decision_with_no_viable_alcove_does_not_pull_over(self):
        person = FakePerson(pos=np.array([0.5, 0.02]),
                            vel=np.array([-0.42, 0.0]))
        encounter = predict_encounter(person, (1.2, 0.0))
        machine, states = _run_machine(
            _ticks(120, duck_xy=(1.2, 0.0), duck_speed_mps=0.14,
                   encounter=encounter))
        assert "PULL_OVER" not in states
        assert machine.no_alcove_events

    def test_state_list_matches_the_machine(self):
        assert set(STATES) >= {"CRUISE", "DETECT", "SELECT_ALCOVE",
                               "PULL_OVER", "YIELD", "CLEAR", "REJOIN",
                               "RESUME", "DONE"}


# ------------------------------------------------------------- controller
class TestController:
    def test_stationary_states_command_exactly_zero(self):
        controller = EtiquetteController()
        for state in STATIONARY_STATES:
            command = controller.raw_command(state, 0.0, 0.3, 0.0)
            assert command == (0.0, 0.0, 0.0)

    def test_yield_is_exactly_zero_not_merely_small(self):
        controller = EtiquetteController()
        command = controller.update("YIELD", 0.0, 0.35, 0.2)
        assert float(np.max(np.abs(command))) == 0.0

    def test_cruise_walks_forward_above_the_gait_onset(self):
        controller = EtiquetteController()
        vx, vy, _ = controller.raw_command("CRUISE", -1.5, 0.0, 0.0)
        assert vx >= VX_MIN_EFFECTIVE
        assert vy == 0.0

    def test_cruise_stops_at_the_destination(self):
        controller = EtiquetteController()
        assert controller.raw_command(
            "CRUISE", DESTINATION_X + 0.01, 0.0, 0.0) == (0.0, 0.0, 0.0)

    def test_detect_and_select_keep_walking(self):
        """Stopping dead in a corridor is an obstruction, not etiquette."""
        controller = EtiquetteController()
        for state in ("DETECT", "SELECT_ALCOVE"):
            vx, _, _ = controller.raw_command(state, -1.5, 0.0, 0.0)
            assert vx >= VX_MIN_EFFECTIVE

    def test_pull_over_waits_for_the_mouth_before_stepping(self):
        """The defect that drove the robot into a wall."""
        controller = EtiquetteController()
        alcove = ALCOVE_BY_NAME["bay_open"]
        low, _ = alcove.x_span
        _, vy, _ = controller.raw_command(
            "PULL_OVER", low - 0.30, 0.0, 0.0,
            park_y=alcove.park_y, target_x=alcove.park_x,
            alcove_name=alcove.name)
        assert vy == 0.0, "must not step sideways while short of the mouth"

    def test_pull_over_steps_once_inside_the_mouth(self):
        controller = EtiquetteController()
        alcove = ALCOVE_BY_NAME["bay_open"]
        _, vy, _ = controller.raw_command(
            "PULL_OVER", alcove.park_x, 0.0, 0.0,
            park_y=alcove.park_y, target_x=alcove.park_x,
            alcove_name=alcove.name)
        assert vy != 0.0

    def test_lateral_commands_clear_their_measured_onsets(self):
        assert VY_PULLOVER_LEFT >= VY_MIN_EFFECTIVE_LEFT
        assert abs(VY_PULLOVER_RIGHT) >= VY_MIN_EFFECTIVE_RIGHT

    def test_the_step_direction_matches_the_alcove_side(self):
        controller = EtiquetteController()
        for name in ("bay_open", "bay_far"):
            alcove = ALCOVE_BY_NAME[name]
            _, vy, _ = controller.raw_command(
                "PULL_OVER", alcove.park_x, 0.0, 0.0,
                park_y=alcove.park_y, target_x=alcove.park_x,
                alcove_name=alcove.name)
            assert math.copysign(1.0, vy) == math.copysign(1.0, alcove.park_y)

    def test_the_two_lateral_signs_carry_different_yaw_feedforward(self):
        """Measured: a right-hand step spins +93.6 deg, a left-hand one does not."""
        controller = EtiquetteController()
        left = ALCOVE_BY_NAME["bay_far"]
        right = ALCOVE_BY_NAME["bay_open"]
        _, _, wz_left = controller.raw_command(
            "PULL_OVER", left.park_x, 0.0, 0.0, park_y=left.park_y,
            target_x=left.park_x, alcove_name=left.name)
        _, _, wz_right = controller.raw_command(
            "PULL_OVER", right.park_x, 0.0, 0.0, park_y=right.park_y,
            target_x=right.park_x, alcove_name=right.name)
        assert wz_right < wz_left, (
            "the right-hand step needs a much stronger negative yaw term")
        assert abs(wz_right - wz_left) > 0.20

    def test_rejoin_steps_back_toward_the_centreline(self):
        controller = EtiquetteController()
        _, vy_from_right, _ = controller.raw_command("REJOIN", 0.0, -0.37, 0.0)
        _, vy_from_left, _ = controller.raw_command("REJOIN", 0.0, +0.37, 0.0)
        assert vy_from_right > 0.0
        assert vy_from_left < 0.0

    def test_rejoin_stops_at_the_centreline(self):
        controller = EtiquetteController()
        assert controller.raw_command("REJOIN", 0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)

    def test_rejoin_makes_no_forward_progress(self):
        """A curve out of a mouth ends against the far wall."""
        controller = EtiquetteController()
        vx, _, _ = controller.raw_command("REJOIN", 0.0, -0.37, 0.0)
        assert vx == 0.0

    def test_no_command_ever_falls_between_zero_and_its_onset(self):
        controller = EtiquetteController()
        alcove = ALCOVE_BY_NAME["bay_open"]
        for state in STATES:
            for x in (-1.5, alcove.park_x, DESTINATION_X + 0.1):
                for y in (0.0, -0.2, -0.37, 0.37):
                    vx, vy, _ = controller.raw_command(
                        state, x, y, 0.0, park_y=alcove.park_y,
                        target_x=alcove.park_x, alcove_name=alcove.name)
                    assert vx == 0.0 or abs(vx) >= VX_MIN_EFFECTIVE
                    assert vy == 0.0 or (
                        vy >= VY_MIN_EFFECTIVE_LEFT if vy > 0
                        else abs(vy) >= VY_MIN_EFFECTIVE_RIGHT)

    def test_heading_hold_corrects_toward_the_corridor_axis(self):
        controller = EtiquetteController()
        _, _, wz_left_of_axis = controller.raw_command(
            "CRUISE", -1.5, 0.0, math.radians(-12.0))
        _, _, wz_right_of_axis = controller.raw_command(
            "CRUISE", -1.5, 0.0, math.radians(+12.0))
        assert wz_left_of_axis > 0.0
        assert wz_right_of_axis < 0.0


# ------------------------------------------------------------ the schedule
class TestPedestrianSchedule:
    def test_two_adults_are_scheduled(self):
        assert len(PEDESTRIANS) == 2
        assert len(PERSON_NAMES) == 2

    def test_nobody_teleports(self):
        from people import max_visible_jump
        jump, name, t = max_visible_jump(50.0)
        assert jump < 0.42 * 0.02 * 2.0, (
            f"{name} jumped {jump:.4f} m in one tick at t={t:.2f}")

    def test_nobody_walks_through_anybody(self):
        from people import min_person_separation
        gap, first, second = min_person_separation(50.0)
        assert gap > 0.0, f"{first} and {second} overlap by {-gap:.4f} m"

    def test_adults_hold_a_lane_near_the_centreline(self):
        """If they moved aside the duck would not need to pull over."""
        for person in PEDESTRIANS:
            assert abs(person.lateral_y) < 0.05

    def test_an_adult_never_yields_to_the_duck(self):
        """Constant velocity throughout the pass: no branch reacts to the duck."""
        for person in PEDESTRIANS:
            speeds = {
                round(float(person.vel_at(t)[0]), 6)
                for t in np.arange(person.start_s + 0.1, min(person.end_s, 48.0), 0.5)
            }
            assert len(speeds) == 1

    def test_both_adults_traverse_the_corridor(self):
        from people import corridor_passes
        passes = corridor_passes(50.0, x_low=CORRIDOR_X_MIN, x_high=CORRIDOR_X_MAX)
        assert {entry["person"] for entry in passes} == set(PERSON_NAMES)

    def test_the_two_passes_do_not_overlap_in_time(self):
        """Two separate encounters, not one crowd."""
        from people import corridor_passes
        passes = sorted(
            corridor_passes(50.0, x_low=CORRIDOR_X_MIN, x_high=CORRIDOR_X_MAX),
            key=lambda e: e["enter_s"])
        assert passes[1]["enter_s"] > passes[0]["exit_s"]

    def test_adults_are_faster_than_the_duck(self):
        for person in PEDESTRIANS:
            assert person.speed > 2.0 * CRUISE_SPEED_MPS

    def test_people_at_is_deterministic(self):
        first = people_at(12.34)
        second = people_at(12.34)
        for name in PERSON_NAMES:
            assert np.allclose(first[name].pos, second[name].pos)
