#!/usr/bin/env python3
"""Tests for the pure predictor, state machine and evade controller.

No MuJoCo, no ONNX, no rendering: everything here is geometry and Python.
Run with ``pytest tests/`` from the behavior folder.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from crowd_routes import (  # noqa: E402
    wrap_angle,
    ADULT_NAMES,
    CARRYING_BOX,
    CORRIDORS,
    CORRIDOR_BY_NAME,
    AdultState,
    crowd_at,
)
from threat_model import (  # noqa: E402
    CMD_TAU,
    DEGENERATE_CLEARANCE,
    EVADE_MAX_S,
    EVADE_MIN_S,
    SETTLE_S,
    CLEAR_S,
    LOCK_CONFIRM_S,
    LOCK_MIN_S,
    PREDICT_HORIZON,
    SCAN_MIN_S,
    THREAT_CLEARANCE,
    URGENT_TTC_S,
    VX_BACK,
    VX_EVADE,
    VX_WALK,
    VY_BACK,
    VY_EVADE,
    WZ_MIN_EFFECTIVE,
    AvoidanceMachine,
    EvadeController,
    escape_heading,
    most_urgent,
    predict_approach,
    rank_threats,
)

# The rollout the schedule is designed for, and the MEASURED time the forward
# gait needs to leave standstill (tools/sweep_commands.py: vx=0.28 covers
# 0.279 m in 2.8 s but 0.414 m in 4.0 s, so roughly the first second is onset).
ROLLOUT_SECONDS = 52.0
GAIT_ONSET_S = 1.0
# Adults that exist as visible traffic and are never scheduled to threaten.
BACKGROUND_ADULTS = ("blue", "purple")


# --------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------
def test_head_on_approach_predicts_zero_clearance_and_correct_ttc():
    """An adult walking straight at the duck closes to ~0 in range/speed seconds."""
    approach = predict_approach(
        np.array([2.0, 0.0]), np.array([-0.4, 0.0]), np.array([0.0, 0.0])
    )
    assert approach.min_clearance == pytest.approx(0.0, abs=1e-9)
    assert approach.time_to_closest == pytest.approx(5.0, abs=1e-9)
    assert approach.closing_speed == pytest.approx(0.4, abs=1e-9)
    assert approach.is_threat


def test_offset_pass_predicts_the_offset_as_min_clearance():
    """A perpendicular pass 0.8 m to the side never gets closer than 0.8 m.

    The adult must be moving PERPENDICULAR to the offset for the offset itself
    to be the miss distance; an adult approaching along -x from (3, 0.8) is
    still 3 m away when it is closest in x, so its true closest approach is
    larger than 0.8 m and the closed form must say so.
    """
    perpendicular = predict_approach(
        np.array([0.0, 0.8]), np.array([0.5, 0.0]), np.array([0.0, 0.0])
    )
    assert perpendicular.min_clearance == pytest.approx(0.8, abs=1e-9)
    assert perpendicular.time_to_closest == pytest.approx(0.0, abs=1e-9)
    assert not perpendicular.is_threat  # 0.8 m > THREAT_CLEARANCE

    # Same lateral offset but approaching head-on down the x axis: the horizon
    # is 5 s and the adult needs 6 s to arrive, so it is still short of its
    # true closest approach when the horizon cuts the prediction off.
    incoming = predict_approach(
        np.array([3.0, 0.8]), np.array([-0.5, 0.0]), np.array([0.0, 0.0])
    )
    assert incoming.time_to_closest == pytest.approx(PREDICT_HORIZON)
    assert incoming.min_clearance == pytest.approx(math.hypot(0.5, 0.8), abs=1e-9)


def test_receding_adult_is_not_a_threat_and_is_deprioritised():
    """A close adult walking AWAY must never outrank a real incoming threat.

    Raw urgency alone gets this wrong: the receding adult is four times closer,
    so a pure 1/(clearance x ttc) score ranks it first.  Genuine threats carry
    an explicit priority so the ordering cannot invert.
    """
    near_receding = predict_approach(
        np.array([0.35, 0.0]), np.array([0.5, 0.0]), np.array([0.0, 0.0]),
        name="receding",
    )
    far_approaching = predict_approach(
        np.array([2.0, 0.05]), np.array([-0.45, 0.0]), np.array([0.0, 0.0]),
        name="approaching",
    )
    assert near_receding.closing_speed < 0.0
    assert not near_receding.is_threat
    assert far_approaching.is_threat
    assert far_approaching.score > near_receding.score


def test_a_nearer_threat_outranks_a_further_threat_of_the_same_kind():
    """Within genuine threats the ordering is still by urgency, not arbitrary."""
    near = predict_approach(
        np.array([0.8, 0.05]), np.array([-0.4, 0.0]), np.array([0.0, 0.0]),
        name="near",
    )
    far = predict_approach(
        np.array([1.9, 0.05]), np.array([-0.4, 0.0]), np.array([0.0, 0.0]),
        name="far",
    )
    assert near.is_threat and far.is_threat
    assert near.score > far.score


def test_time_to_closest_is_clamped_to_the_horizon():
    approach = predict_approach(
        np.array([50.0, 0.0]), np.array([-0.4, 0.0]), np.array([0.0, 0.0])
    )
    assert approach.time_to_closest == pytest.approx(PREDICT_HORIZON)


def test_stationary_adult_gives_current_range_as_clearance():
    approach = predict_approach(
        np.array([0.9, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
    )
    assert approach.min_clearance == pytest.approx(0.9)
    assert approach.time_to_closest == pytest.approx(0.0)


def test_duck_velocity_changes_the_predicted_clearance():
    """The counterfactual used by the validation gate must actually respond."""
    adult_pos, adult_vel = np.array([2.0, 0.0]), np.array([-0.4, 0.0])
    standing = predict_approach(adult_pos, adult_vel, np.array([0.0, 0.0]))
    sidestepping = predict_approach(
        adult_pos, adult_vel, np.array([0.0, 0.0]), duck_vel=np.array([0.0, 0.25])
    )
    assert standing.min_clearance == pytest.approx(0.0, abs=1e-9)
    assert sidestepping.min_clearance > 0.5


def test_ranking_prefers_the_more_urgent_of_two_real_threats():
    crowd = {
        "far": AdultState("far", np.array([3.0, 0.0]), np.array([-0.35, 0.0]),
                          math.pi, 0.35, False),
        "near": AdultState("near", np.array([1.0, 0.05]), np.array([-0.40, 0.0]),
                           math.pi, 0.40, False),
    }
    ranked = rank_threats(crowd, np.array([0.0, 0.0]))
    assert [approach.name for approach in ranked] == ["near", "far"]
    assert most_urgent(crowd, np.array([0.0, 0.0])).name == "near"


def test_most_urgent_returns_none_when_everyone_will_miss():
    crowd = {
        name: AdultState(name, np.array([3.0, 2.0 + index]),
                         np.array([0.3, 0.0]), 0.0, 0.3, False)
        for index, name in enumerate(("a", "b"))
    }
    assert most_urgent(crowd, np.array([0.0, 0.0])) is None


def test_escape_heading_points_away_from_predicted_impact():
    """Escape is perpendicular to the threat line, away from where they arrive.

    A threat closing along -x that will pass 0.10 m to the duck's +y side has
    its predicted closest point at +y, so the escape must be almost exactly -y:
    sidestep out of their lane rather than retreat down it.
    """
    approach = predict_approach(
        np.array([1.5, 0.10]), np.array([-0.4, 0.0]), np.array([0.0, 0.0])
    )
    heading = escape_heading(approach, np.array([0.0, 0.0]))
    assert math.sin(heading) < -0.99  # essentially straight -y
    assert abs(math.cos(heading)) < 0.05


def test_escape_heading_mirrors_for_a_threat_passing_on_the_other_side():
    approach = predict_approach(
        np.array([1.5, -0.10]), np.array([-0.4, 0.0]), np.array([0.0, 0.0])
    )
    heading = escape_heading(approach, np.array([0.0, 0.0]))
    assert math.sin(heading) > 0.99  # mirrored: escape to +y


def test_escape_heading_degenerate_case_flees_directly_away():
    approach = predict_approach(
        np.array([1.5, 0.0]), np.array([-0.4, 0.0]), np.array([0.0, 0.0])
    )
    heading = escape_heading(approach, np.array([0.0, 0.0]))
    assert math.cos(heading) < 0.0  # threat is at +x, escape goes -x


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------
def _threat(name="red", clearance=0.10, ttc=1.2, closing=0.4):
    return predict_approach(
        np.array([ttc * closing, clearance]),
        np.array([-closing, 0.0]),
        np.array([0.0, 0.0]),
        name=name,
    )


def _run(machine, threats, seconds, dt=0.02):
    """Drive the machine with a callable ``threats(t)`` and collect states."""
    states = []
    steps = int(seconds / dt)
    for step in range(steps):
        t = step * dt
        state, _ = machine.update(t, threats(t))
        states.append((t, state))
    return states


def test_machine_starts_scanning_and_will_not_lock_before_scan_min():
    """A non-urgent threat waits for the full scan before it can lock."""
    machine = AvoidanceMachine()
    threat = _threat(ttc=4.0, clearance=0.10, closing=0.3)
    assert threat.time_to_closest > URGENT_TTC_S
    for step in range(int(SCAN_MIN_S / 0.02) - 2):
        state, _ = machine.update(step * 0.02, threat)
        assert state == "SCANNING"


def test_machine_requires_sustained_confirmation_before_locking():
    """A threat flickering in and out for one tick must never lock."""
    machine = AvoidanceMachine()
    threat = _threat()
    states = _run(
        machine,
        lambda t: threat if int(t / 0.02) % 2 == 0 else None,
        seconds=6.0,
    )
    assert {state for _, state in states} == {"SCANNING"}


def test_machine_completes_a_full_cycle_and_records_it():
    machine = AvoidanceMachine()
    threat = _threat(name="teal")
    resolved_at = SCAN_MIN_S + LOCK_MIN_S + EVADE_MIN_S + 0.5

    def threats(t):
        # Present until the evade has genuinely been performed, then resolved.
        if t < resolved_at:
            return threat
        return predict_approach(
            np.array([-1.4, 0.0]), np.array([-0.5, 0.0]), np.array([0.0, 0.0]),
            name="teal",
        )

    # Stop before a second cycle can begin, so exactly one is recorded.
    horizon = resolved_at + SETTLE_S + CLEAR_S + 0.1
    states = [state for _, state in _run(machine, threats, seconds=horizon)]
    assert "THREAT_LOCK" in states
    assert "EVADING" in states
    assert "SETTLING" in states
    assert "CLEAR" in states
    assert len(machine.cycles) == 1
    cycle = machine.cycles[0]
    assert cycle["threat"] == "teal"
    assert cycle["evade_start_s"] > cycle["lock_s"]
    assert cycle["evade_end_s"] > cycle["evade_start_s"]
    assert cycle["evade_duration_s"] >= EVADE_MIN_S
    assert cycle["lock_clearance_m"] < THREAT_CLEARANCE
    assert machine.locked is None  # released after the cycle closes


def test_evade_is_never_shorter_than_the_minimum_maneuver():
    """A threat that vanishes the instant EVADING starts must not end it.

    Without a minimum the machine would tick straight into SETTLING, leaving no
    physical displacement for the validation gate to measure — a cycle that
    looks complete in the metrics but shows nothing in the video.
    """
    machine = AvoidanceMachine()
    threat = _threat(name="ghost")
    lock_done = SCAN_MIN_S + LOCK_MIN_S
    _run(machine, lambda t: threat if t < lock_done + 0.02 else None,
         seconds=lock_done + EVADE_MIN_S + SETTLE_S + CLEAR_S + 0.5)
    assert machine.cycles
    assert machine.cycles[0]["evade_duration_s"] >= EVADE_MIN_S


def test_machine_ignores_an_approach_that_is_not_a_genuine_threat():
    """Passing a non-threat Approach must behave exactly like passing None."""
    machine = AvoidanceMachine()
    miss = predict_approach(
        np.array([0.0, 1.2]), np.array([0.4, 0.0]), np.array([0.0, 0.0]),
        name="passerby",
    )
    assert not miss.is_threat
    states = {state for _, state in _run(machine, lambda t: miss, seconds=12.0)}
    assert states == {"SCANNING"}
    assert machine.locked is None


def test_machine_repeats_cycles_for_distinct_threats():
    machine = AvoidanceMachine()
    schedule = (("red", 0.0, 9.0), ("blue", 11.0, 20.0), ("pink", 22.0, 31.0))

    def threats(t):
        for name, start, end in schedule:
            if start <= t < end:
                return _threat(name=name)
        return None

    _run(machine, threats, seconds=40.0)
    assert [cycle["threat"] for cycle in machine.cycles] == ["red", "blue", "pink"]
    assert len(machine.cycles) == 3


def test_evade_times_out_when_the_threat_never_resolves():
    machine = AvoidanceMachine()
    threat = _threat(name="stuck")
    _run(machine, lambda t: threat, seconds=SCAN_MIN_S + LOCK_MIN_S + EVADE_MAX_S + 4.0)
    assert machine.cycles, "expected the timeout to still close the cycle"
    assert machine.cycles[0]["evade_timeout"] is True
    assert machine.cycles[0]["evade_duration_s"] == pytest.approx(EVADE_MAX_S, abs=0.05)


def test_lock_records_the_confirmed_threat_not_a_later_one():
    """Once locked, a newly more urgent adult must not silently steal the lock."""
    machine = AvoidanceMachine()
    first = _threat(name="green")
    second = _threat(name="orange", clearance=0.01, ttc=0.3)

    def threats(t):
        return first if t < SCAN_MIN_S + LOCK_CONFIRM_S + 0.1 else second

    _run(machine, threats, seconds=SCAN_MIN_S + LOCK_MIN_S + 0.5)
    assert machine.locked == "green"


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------
@pytest.mark.parametrize("state", ["SCANNING", "THREAT_LOCK", "SETTLING", "CLEAR"])
def test_stationary_states_command_exactly_zero(state):
    controller = EvadeController()
    controller.command[:] = np.array([0.28, 0.30, 0.6], dtype=np.float32)
    command = controller.update(state, 0.9)
    assert np.array_equal(command, np.zeros(3, dtype=np.float32))


def test_evading_always_crosses_a_measured_gait_onset():
    """Every evade command clears the onset of whichever gait it uses.

    MEASURED onsets differ by direction: forward crosses between vx=0.20
    (10 mm in 6 s) and vx=0.24 (516 mm); backward crosses between vx=-0.30
    (5 mm in 3 s) and vx=-0.34 (0.50 m).  A command that sits below its own
    onset produces no motion at all, not a small one.
    """
    controller = EvadeController()
    for heading_error in np.linspace(-math.pi, math.pi, 73):
        vx, vy, wz = controller.raw_command("EVADING", float(heading_error))
        if vx >= 0.0:
            assert vx >= 0.24, f"vx={vx} below forward onset at {heading_error}"
            assert abs(vy) <= VY_EVADE + 1e-9
        else:
            assert vx <= -0.34, f"vx={vx} below backward onset at {heading_error}"
            assert abs(vy) <= VY_BACK + 1e-9
        # An intermediate wz suppresses the gait (measured: 0.24/+0.35 -> 15 mm
        # in 4 s).  Anything nonzero must be at or above the effective minimum.
        assert wz == 0.0 or abs(wz) >= WZ_MIN_EFFECTIVE


def test_evading_never_stands_still():
    """Whatever the heading error, EVADING must command real locomotion."""
    controller = EvadeController()
    for heading_error in np.linspace(-math.pi, math.pi, 145):
        command = controller.raw_command("EVADING", float(heading_error))
        assert float(np.linalg.norm(command)) > 0.2


def test_large_but_reachable_heading_error_turns_while_walking():
    """Below the backward threshold a big error still turns while walking."""
    controller = EvadeController()
    vx, vy, wz = controller.raw_command("EVADING", math.radians(100.0))
    assert vx == pytest.approx(VX_WALK)
    assert vy == 0.0
    assert wz >= WZ_MIN_EFFECTIVE


def test_heading_error_sign_selects_the_turn_direction():
    controller = EvadeController()
    _, _, left = controller.raw_command("EVADING", math.radians(90.0))
    _, _, right = controller.raw_command("EVADING", math.radians(-90.0))
    assert left > 0.0 and right < 0.0


def test_small_heading_error_uses_a_lateral_walk():
    controller = EvadeController()
    vx, vy, wz = controller.raw_command("EVADING", math.radians(20.0))
    assert vx == pytest.approx(VX_EVADE)
    assert wz == 0.0
    assert 0.0 < vy <= VY_EVADE


def test_command_filter_reaches_the_target_within_a_few_tau():
    controller = EvadeController()
    for _ in range(int(6 * CMD_TAU * 50)):
        command = controller.update("EVADING", 0.0)
    assert command[0] == pytest.approx(VX_EVADE, abs=0.01)


def test_filter_never_leaks_a_command_into_a_stationary_state():
    controller = EvadeController()
    for _ in range(50):
        controller.update("EVADING", math.radians(90.0))
    assert float(np.linalg.norm(controller.update("SETTLING", 0.0))) == 0.0


def test_a_threat_behind_the_duck_is_escaped_by_reversing():
    """The escape must not require walking forward through the threat.

    Run 5's only genuine contact: green locked with a -131.7 deg escape
    heading, and with forward-only primitives the duck spent 1.5 s turning
    while walking toward it, reaching -1.9 mm clearance.
    """
    controller = EvadeController()
    vx, vy, wz = controller.raw_command("EVADING", math.radians(-131.7))
    assert vx < 0.0, "still walking forward into a threat behind the duck"
    assert wz == 0.0, "still turning through the threat"
    assert abs(vx) >= 0.34, "below the MEASURED backward gait onset"


def test_backward_escape_steers_toward_the_escape_side():
    """The local reverse vector must align with the requested world heading.

    Checking only that left/right use opposite lateral signs missed a real
    sign inversion: both signs were opposite and both commands steered away
    from their requested side.  Robot yaw is zero here, so the heading-error
    unit vector is directly comparable with ``(vx, vy)``.
    """
    for degrees in (120.0, 135.0, 150.0, -120.0, -135.0, -150.0):
        controller = EvadeController()
        vx, vy, wz = controller.raw_command("EVADING", math.radians(degrees))
        command = np.array([vx, vy], dtype=np.float64)
        desired = np.array([
            math.cos(math.radians(degrees)),
            math.sin(math.radians(degrees)),
        ])
        alignment = float(command @ desired / np.linalg.norm(command))
        assert vx < 0.0 and wz == 0.0
        assert alignment > 0.90, (degrees, command, alignment)


def test_forward_primitives_still_handle_reachable_escapes():
    """Backing up is for escapes behind only; the rest still walk forward."""
    controller = EvadeController()
    for degrees in (0.0, 20.0, -20.0, 60.0, -60.0, 100.0, -100.0):
        vx, _, _ = controller.raw_command("EVADING", math.radians(degrees))
        assert vx > 0.0, f"{degrees} deg should still be a forward escape"


def test_backward_escape_is_still_silent_outside_evading():
    controller = EvadeController()
    for state in ("SCANNING", "THREAT_LOCK", "SETTLING", "CLEAR"):
        assert controller.raw_command(state, math.radians(-150.0)) == (0.0, 0.0, 0.0)


def test_a_committed_reversal_is_not_abandoned_when_the_escape_flips():
    """An evasion must not change its mind mid-maneuver.

    The escape heading is derived from the predicted impact point, which is
    genuinely unstable when the predicted clearance is near zero: run 6
    measured a +178.6 -> -1.3 deg jump in ONE tick, which abandoned a backward
    escape and walked the duck forward into the adult it was avoiding.
    """
    controller = EvadeController()
    started = controller.raw_command("EVADING", math.radians(-130.0))
    assert started[0] < 0.0
    # The escape direction reverses completely; the maneuver must persist.
    after = controller.raw_command("EVADING", math.radians(50.0))
    assert after[0] < 0.0, "abandoned the reversal after a heading flip"


def test_a_committed_forward_escape_is_also_kept():
    controller = EvadeController()
    assert controller.raw_command("EVADING", math.radians(20.0))[0] > 0.0
    assert controller.raw_command("EVADING", math.radians(-170.0))[0] > 0.0


def test_reset_lets_the_next_evasion_choose_afresh():
    controller = EvadeController()
    assert controller.raw_command("EVADING", math.radians(-130.0))[0] < 0.0
    controller.reset()
    assert controller.raw_command("EVADING", math.radians(20.0))[0] > 0.0


def test_escape_heading_is_stable_through_a_zero_clearance_crossing():
    """The degenerate band must remove the 180 deg flip at zero clearance."""
    duck = np.array([0.0, 0.0])
    adult_vel = np.array([-0.25, -0.02])
    headings = []
    for offset in np.linspace(0.04, -0.04, 41):
        approach = predict_approach(
            np.array([1.2, offset]), adult_vel, duck, name="green"
        )
        headings.append(escape_heading(approach, duck, adult_vel))
    jumps = [
        abs(wrap_angle(b - a)) for a, b in zip(headings, headings[1:])
    ]
    assert max(jumps) < math.radians(30.0), (
        f"escape heading jumps {math.degrees(max(jumps)):.1f} deg near zero "
        "predicted clearance"
    )


def test_degenerate_escape_leaves_the_lane_instead_of_retreating_down_it():
    """A head-on threat must be sidestepped, not outrun.

    Run 7 measured the failure: fleeing directly away from a head-on adult is
    nearly anti-parallel to their travel, so the duck reversed 0.89 m down
    green's lane, was followed from 0.30 m to 0.11 m, and improved the
    predicted clearance by only 0.024 m.
    """
    duck = np.array([0.0, 0.0])
    adult_vel = np.array([-0.25, 0.0])
    approach = predict_approach(
        np.array([1.2, 0.002]), adult_vel, duck, name="green"
    )
    assert approach.min_clearance < DEGENERATE_CLEARANCE
    heading = escape_heading(approach, duck, adult_vel)
    direction = np.array([math.cos(heading), math.sin(heading)])
    # Normal to their travel, not along it.
    assert abs(float(direction @ adult_vel)) < 1e-9


def test_the_duck_cannot_outrun_the_crowd():
    """The measurement that makes sidestepping the ONLY viable escape.

    MEASURED world speeds (tools/sweep_commands.py, 3 s, net displacement over
    time): the duck's fastest escape primitive is (-0.40, -0.34) at 0.211 m/s,
    and every forward primitive is slower still (0.101-0.122 m/s).  The adults
    walk at 0.220-0.229 m/s at their pass point.  Retreating down a threat's
    lane therefore CANNOT work - they close on the duck the whole way - which
    is exactly what run 7 measured when the degenerate escape fled directly
    away: green closed 0.30 m -> 0.11 m over a 0.89 m reversal.
    """
    fastest_escape = 0.211  # m/s, measured
    slowest_adult = min(
        float(np.linalg.norm(corridor.at(corridor.pass_time)[1]))
        for corridor in CORRIDORS
    )
    assert fastest_escape < slowest_adult


def test_degenerate_escape_beats_fleeing_directly_away_over_the_encounter():
    """The sidestep opens the clearance; the retreat only postpones the contact.

    Evaluated at the MEASURED escape speed (0.211 m/s), not the command value,
    and over the WHOLE encounter rather than the 5 s prediction window.

    Over 5 s the retreat scores BETTER (1.005 m vs 0.776 m), because in that
    window a 0.039 m/s closing rate barely dents a 1.2 m gap.  That is exactly
    why the behavior chose it - and exactly why it failed: the duck is slower
    than the adult, so retreating down their lane never resolves, it only
    defers.  Run 7 measured the consequence: a 5.02 s evasion that hit
    EVADE_MAX_S without resolving while green closed 0.30 m -> 0.11 m.
    Judged over the encounter, the sidestep is strictly better.
    """
    duck = np.array([0.0, 0.0])
    adult_pos = np.array([1.2, 0.002])
    adult_vel = np.array([-0.25, 0.0])  # 0.25 m/s, a typical adult
    approach = predict_approach(adult_pos, adult_vel, duck, name="green")
    measured_escape_speed = 0.211  # m/s for the (-0.40, -0.34) primitive
    encounter_horizon = 25.0

    def clearance_after(heading: float, horizon: float) -> float:
        duck_vel = measured_escape_speed * np.array(
            [math.cos(heading), math.sin(heading)]
        )
        return predict_approach(
            adult_pos, adult_vel, duck, duck_vel=duck_vel, horizon=horizon
        ).min_clearance

    sidestep_heading = escape_heading(approach, duck, adult_vel)
    retreat_heading = wrap_angle(approach.bearing + math.pi)

    # Over the prediction window the retreat wins, which is the trap.
    assert clearance_after(retreat_heading, PREDICT_HORIZON) > clearance_after(
        sidestep_heading, PREDICT_HORIZON
    )
    # Over the encounter the sidestep is the one that actually clears.
    assert clearance_after(sidestep_heading, encounter_horizon) > clearance_after(
        retreat_heading, encounter_horizon
    )


def test_a_sidestep_clearance_is_bounded_but_a_retreat_decays():
    """Lengthening the horizon must not erode the sidestep's miss distance.

    The sidestep reaches an interior minimum and stays there; the retreat keeps
    losing ground for as long as the adult keeps walking.
    """
    duck = np.array([0.0, 0.0])
    adult_pos = np.array([1.2, 0.002])
    adult_vel = np.array([-0.25, 0.0])
    approach = predict_approach(adult_pos, adult_vel, duck, name="green")
    speed = 0.211

    def clearance(heading: float, horizon: float) -> float:
        duck_vel = speed * np.array([math.cos(heading), math.sin(heading)])
        return predict_approach(
            adult_pos, adult_vel, duck, duck_vel=duck_vel, horizon=horizon
        ).min_clearance

    sidestep = escape_heading(approach, duck, adult_vel)
    retreat = wrap_angle(approach.bearing + math.pi)
    assert clearance(sidestep, 25.0) == pytest.approx(clearance(sidestep, 10.0))
    assert clearance(retreat, 25.0) < clearance(retreat, 10.0)


def test_degenerate_escape_keeps_the_side_the_duck_is_already_on():
    """The duck must not cross in front of the adult to reach the other side."""
    adult_vel = np.array([-0.25, 0.0])
    for duck_y in (0.004, -0.004):
        duck = np.array([0.0, duck_y])
        approach = predict_approach(
            np.array([1.2, 0.0]), adult_vel, duck, name="green"
        )
        assert approach.min_clearance < DEGENERATE_CLEARANCE
        heading = escape_heading(approach, duck, adult_vel)
        assert math.copysign(1.0, math.sin(heading)) == math.copysign(1.0, duck_y)


# --------------------------------------------------------------------------
# Crowd routes
# --------------------------------------------------------------------------
def test_eight_adults_with_five_carrying_boxes():
    assert len(ADULT_NAMES) == 8
    assert len(CORRIDORS) == 8
    assert len(CARRYING_BOX) == 5
    assert CARRYING_BOX <= set(ADULT_NAMES)


def test_every_adult_keeps_moving_for_the_whole_rollout():
    """Nobody freezes: minimum speed over 60 s must stay clearly above zero."""
    for name in ADULT_NAMES:
        speeds = [crowd_at(t)[name].speed for t in np.arange(0.0, 60.0, 0.25)]
        assert min(speeds) > 0.05, f"{name} stalls (min speed {min(speeds):.3f})"


def test_nobody_teleports():
    """Consecutive samples are continuous: no jump larger than speed * dt."""
    dt = 0.02
    previous = crowd_at(0.0)
    for step in range(1, int(60.0 / dt)):
        current = crowd_at(step * dt)
        for name in ADULT_NAMES:
            jump = float(np.linalg.norm(current[name].pos - previous[name].pos))
            assert jump < 0.02, f"{name} jumped {jump:.4f} m in one tick"
        previous = current


def test_each_adult_passes_its_scheduled_point_at_its_scheduled_time():
    """The solved ellipse must actually realise the encounter it was given."""
    for corridor in CORRIDORS:
        pos, vel = corridor.at(corridor.pass_time)
        assert pos == pytest.approx(np.asarray(corridor.pass_point), abs=1e-9)
        heading = math.atan2(float(vel[1]), float(vel[0]))
        assert abs(wrap_angle(heading - corridor.pass_heading)) < 1e-9


def test_six_adults_are_scheduled_to_pass_within_threat_range():
    close = [
        corridor for corridor in CORRIDORS
        if float(np.linalg.norm(corridor.pass_point)) < 0.30
    ]
    assert len(close) >= 6, f"only {len(close)} adults create a near-pass"


def test_approaches_come_from_several_distinct_bearings():
    bearings = []
    for corridor in CORRIDORS:
        pos, _ = corridor.at(corridor.pass_time - 2.5)
        bearings.append(math.atan2(float(pos[1]), float(pos[0])))
    # At least four different quadrant-scale bearings among the eight adults.
    buckets = {round(bearing / (math.pi / 2)) % 4 for bearing in bearings}
    assert len(buckets) >= 3, f"approach bearings too similar: {sorted(buckets)}"

def test_carried_box_clearance_is_larger_than_torso_only_in_front():
    adult = AdultState("red", np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0,
                       1.0, True)
    plain = AdultState("blue", np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0,
                       1.0, False)
    in_front = np.array([0.25, 0.0])
    behind = np.array([-0.25, 0.0])
    # The box occupies space in front, so clearance there is smaller.
    assert adult.clearance_to(in_front) < plain.clearance_to(in_front)
    # Behind them the box is irrelevant and both are identical.
    assert adult.clearance_to(behind) == pytest.approx(plain.clearance_to(behind))


def test_clearance_is_zero_at_the_torso_surface():
    adult = AdultState("blue", np.array([1.0, 1.0]), np.array([1.0, 0.0]), 0.0,
                       1.0, False)
    from crowd_routes import TORSO_RADIUS

    assert adult.clearance_to(np.array([1.0 + TORSO_RADIUS, 1.0])) == pytest.approx(0.0)


def test_corridor_positions_are_deterministic():
    assert np.array_equal(crowd_at(7.3)["teal"].pos, crowd_at(7.3)["teal"].pos)


# --------------------------------------------------------------------------
# Schedule invariants added after the run-1 and run-2 measurements
# --------------------------------------------------------------------------
def test_every_scheduled_corridor_period_outlasts_the_rollout():
    """Each SCHEDULED adult may cross the plaza at most ONCE in a 52 s rollout.

    Run 1 used 27-47 s periods, so every adult swept the centre twice.  The
    second, unscheduled pass arrived while the duck was committed to a
    different threat and walked through it: 548 of 2600 steps in contact.

    blue and purple are exempt: they never come close enough to be threats, so
    repeating their loop cannot collide with the schedule.
    """
    for corridor in CORRIDORS:
        if corridor.name in BACKGROUND_ADULTS:
            continue
        assert corridor.period > ROLLOUT_SECONDS, (
            f"{corridor.name} period {corridor.period}s repeats within a "
            f"{ROLLOUT_SECONDS}s rollout"
        )


def test_scheduled_encounters_are_separated_by_a_full_cycle():
    """Consecutive near-passes leave room for one complete state-machine cycle."""
    shortest_cycle = (
        SCAN_MIN_S + LOCK_CONFIRM_S + LOCK_MIN_S + EVADE_MIN_S + SETTLE_S + CLEAR_S
    )
    times = sorted(
        corridor.pass_time for corridor in CORRIDORS
        if float(np.linalg.norm(corridor.pass_point)) < 0.30
    )
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert min(gaps) >= shortest_cycle, (
        f"encounters {min(gaps):.2f}s apart cannot fit a {shortest_cycle:.2f}s cycle"
    )


def test_background_adults_never_enter_threat_range_of_the_plaza():
    """blue and purple are traffic, not scheduled threats.

    In run 2 their loops came within ~1.4 m of the centre, inside
    THREAT_CLEARANCE of where the duck ends an evasion, so purple hijacked
    cycle 3 and delayed the reaction to red by 4.5 s.  The requirement is on
    the WHOLE LOOP: a distant pass point is not sufficient, because the ellipse
    can swing back in elsewhere.
    """
    reachable = 1.0  # duck stayed within ~0.9 m of the origin across runs
    for name in BACKGROUND_ADULTS:
        corridor = CORRIDOR_BY_NAME[name]
        closest = min(
            float(np.linalg.norm(corridor.at(float(t))[0]))
            for t in np.arange(0.0, ROLLOUT_SECONDS, 0.05)
        )
        assert closest > reachable + THREAT_CLEARANCE, (
            f"{name} comes within {closest:.2f} m of the plaza centre"
        )


def test_background_adults_stay_inside_the_camera_framing():
    """Traffic must remain visible: pushed out, but not out of the shot."""
    for name in BACKGROUND_ADULTS:
        corridor = CORRIDOR_BY_NAME[name]
        furthest = max(
            float(np.linalg.norm(corridor.at(float(t))[0]))
            for t in np.arange(0.0, ROLLOUT_SECONDS, 0.05)
        )
        assert furthest < 4.5, f"{name} wanders {furthest:.2f} m away"


def test_scheduled_threats_still_reach_the_plaza_centre():
    """Enlarging the loops must not have pushed the real encounters away."""
    for name in ("orange", "green", "red", "teal", "yellow", "pink"):
        corridor = CORRIDOR_BY_NAME[name]
        closest = min(
            float(np.linalg.norm(corridor.at(float(t))[0]))
            for t in np.arange(0.0, ROLLOUT_SECONDS, 0.05)
        )
        assert closest < 0.30, f"{name} never reaches the plaza ({closest:.2f} m)"


def test_walking_speed_is_preserved_by_the_larger_loops():
    """Bigger, slower loops: the speed at each near-pass stays in the same band."""
    for corridor in CORRIDORS:
        _, velocity = corridor.at(corridor.pass_time)
        speed = float(np.linalg.norm(velocity))
        assert 0.18 <= speed <= 0.30, f"{corridor.name} passes at {speed:.3f} m/s"


# --------------------------------------------------------------------------
# Escape heading: why there is deliberately no velocity term
# --------------------------------------------------------------------------
def test_escape_is_already_normal_to_the_threat_path():
    """For an interior closest approach the escape IS the path normal.

    At an interior minimum ``(p + v t*) . v = 0``, so the duck-to-impact vector
    is perpendicular to the adult's velocity and the plain geometric escape is
    already the sidestep.  No velocity term is needed to obtain that.
    """
    adult_vel = np.array([-0.25, 0.0])
    approach = predict_approach(
        np.array([0.9, 0.05]), adult_vel, np.array([0.0, 0.0])
    )
    assert 0.0 < approach.time_to_closest < PREDICT_HORIZON  # interior minimum
    heading = escape_heading(approach, np.array([0.0, 0.0]))
    direction = np.array([math.cos(heading), math.sin(heading)])
    assert abs(float(direction @ adult_vel)) < 1e-9


def test_retreat_beats_sidestep_when_the_minimum_is_beyond_the_horizon():
    """Pins the measurement that REJECTED an explicit velocity-normal escape.

    That variant was implemented and compared on this geometry.  It is
    identical for interior minima, and strictly worse when the closest approach
    lies beyond the horizon: the duck (0.28 m/s) outruns the adults
    (0.22-0.23 m/s), so retreating opens the gap faster than stepping aside
    (3.00 m vs 2.27 m predicted clearance).  If this ever inverts, the simpler
    escape is worth revisiting.
    """
    adult_pos = np.array([3.0, 0.05])
    adult_vel = np.array([-0.25, 0.0])
    duck = np.array([0.0, 0.0])
    approach = predict_approach(adult_pos, adult_vel, duck)
    assert approach.time_to_closest == pytest.approx(PREDICT_HORIZON)

    def clearance_after(heading: float) -> float:
        duck_vel = VX_EVADE * np.array([math.cos(heading), math.sin(heading)])
        return predict_approach(
            adult_pos, adult_vel, duck, duck_vel=duck_vel
        ).min_clearance

    unit = adult_vel / float(np.linalg.norm(adult_vel))
    normal = np.array([-unit[1], unit[0]])
    sidestep = math.atan2(float(normal[1]), float(normal[0]))
    assert clearance_after(escape_heading(approach, duck)) > clearance_after(sidestep)


def test_escape_side_is_mirrored_by_the_side_they_pass_on():
    """Which side the duck steps to follows the side they are predicted to pass."""
    adult_vel = np.array([-0.4, 0.0])
    duck = np.array([0.0, 0.0])
    for offset in (0.12, -0.12):
        approach = predict_approach(np.array([1.5, offset]), adult_vel, duck)
        heading = escape_heading(approach, duck)
        assert math.copysign(1.0, math.sin(heading)) == -math.copysign(1.0, offset)


# --------------------------------------------------------------------------
# Urgent bypass of the scan minimum
# --------------------------------------------------------------------------
def test_an_imminent_threat_locks_without_waiting_for_the_scan_minimum():
    """Looking around politely is not worth being walked into.

    Run-2 cycle 4 locked red only after SCAN_MIN_S, by which point it was 0.16 m
    away with 0.63 s to closest approach - less than the gait needs to leave
    standstill - and the duck was overlapped by 0.073 m.
    """
    machine = AvoidanceMachine()
    urgent = predict_approach(
        np.array([0.30, 0.02]), np.array([-0.30, 0.0]), np.array([0.0, 0.0]),
        name="red",
    )
    assert urgent.time_to_closest <= URGENT_TTC_S
    states = [state for _, state in _run(machine, lambda t: urgent, seconds=1.0)]
    assert "THREAT_LOCK" in states
    assert machine.cycles == [] and machine.locked == "red"
    assert machine.current["lock_s"] < SCAN_MIN_S


def test_the_urgent_bypass_still_requires_confirmation():
    """A single noisy frame must not trigger the bypass."""
    machine = AvoidanceMachine()
    urgent = predict_approach(
        np.array([0.30, 0.02]), np.array([-0.30, 0.0]), np.array([0.0, 0.0]),
        name="red",
    )
    states = _run(
        machine, lambda t: urgent if int(t / 0.02) % 2 == 0 else None, seconds=1.2
    )
    assert {state for _, state in states} == {"SCANNING"}


def test_close_rendered_geometry_starts_evasion_before_the_normal_hold():
    """A carried box near contact must not wait on a centre-point TTC."""
    machine = AvoidanceMachine()
    leisurely = predict_approach(
        np.array([1.30, 0.10]), np.array([-0.30, 0.0]), np.array([0.0, 0.0]),
        name="red",
    )
    assert leisurely.time_to_closest > URGENT_TTC_S
    evade_at = None
    for step in range(200):
        t = step * 0.02
        state, _ = machine.update(
            t,
            leisurely,
            {"surface_clearance_m": 0.20} if machine.locked else None,
        )
        if state == "EVADING":
            evade_at = t
            break
    assert evade_at is not None
    assert evade_at - machine.current["lock_s"] < LOCK_MIN_S


def test_a_leisurely_threat_still_waits_for_the_full_scan():
    """The bypass is for imminent approaches only."""
    machine = AvoidanceMachine()
    distant = predict_approach(
        np.array([1.6, 0.10]), np.array([-0.30, 0.0]), np.array([0.0, 0.0]),
        name="teal",
    )
    assert distant.time_to_closest > URGENT_TTC_S
    for step in range(int(SCAN_MIN_S / 0.02) - 2):
        state, _ = machine.update(step * 0.02, distant)
        assert state == "SCANNING"


def test_minimum_evasion_outlasts_the_measured_gait_onset():
    """EVADE_MIN_S must leave real path length after the gait starts.

    The forward gait spends its first ~1.0 s crossing onset; a 1.8 s evasion
    produced only 0.249 m of path in run 2, under the 0.25 m the metrics gate
    requires as evidence that a maneuver physically happened.
    """
    assert EVADE_MIN_S >= GAIT_ONSET_S + 1.5
