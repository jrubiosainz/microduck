#!/usr/bin/env python3
"""Pure-logic tests for the recall state machine, controller and call script.

No MuJoCo, no ONNX, no rendering.  Everything here runs on the decision layer
alone, which is exactly why the layer was written without simulator imports.

MUTATION DISCIPLINE
-------------------
Every meaningful gate in this file is paired with a test that MUTATES the
implementation (or feeds it a synthetic counterexample) and asserts the gate
FAILS.  A test that only ever sees passing input proves nothing about the gate;
it proves the happy path still works.  The mutation tests are marked
``test_mutation_*`` so they are easy to find and audit.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from people_routes import (  # noqa: E402
    ADULT_NAMES,
    CALLER_NAMES,
    STATION_BY_NAME,
    crowd_at,
    min_person_separation,
)
from recall_model import (  # noqa: E402
    ACQUIRE_CONE_DEG,
    ACQUIRE_CONFIRM_S,
    APPROACH_MAX_S,
    ARRIVED_HOLD_S,
    COAST_M,
    LISTEN_MIN_S,
    LOCK_HOLD_S,
    STANDOFF_MAX,
    STANDOFF_MIN,
    STANDOFF_TARGET,
    STATIONARY_STATES,
    STOP_RANGE,
    VX_CRUISE,
    VX_MIN_EFFECTIVE,
    WZ_MIN_LEFT,
    WZ_MIN_RIGHT,
    ApproachController,
    Call,
    RecallMachine,
    calls_active_at,
    wrap_angle,
)

DT = 1.0 / 50.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def drive(machine: RecallMachine, calls, *, seconds, caller_range,
          gate_open, caller_visible, start=0.0):
    """Advance the machine over a window with callables for the inputs."""
    states = []
    steps = int(round(seconds / DT))
    for index in range(steps):
        t = start + index * DT
        state, changed = machine.update(
            t,
            calls=calls,
            caller_range=caller_range(t, machine) if callable(caller_range)
            else caller_range,
            gate_open=gate_open(t, machine) if callable(gate_open) else gate_open,
            caller_visible=(
                caller_visible(t, machine) if callable(caller_visible)
                else caller_visible),
        )
        states.append((t, state, changed))
    return states


SIMPLE_CALLS = (Call(caller="red", start_s=1.0, duration_s=30.0),)


def run_one_full_cycle(machine=None, calls=SIMPLE_CALLS, ranges=None):
    """Drive one complete LISTEN->...->LISTEN cycle with cooperative inputs."""
    machine = machine or RecallMachine(ctrl_hz=50.0)
    state_ranges = ranges or {}

    def caller_range(t, m):
        if m.state == "APPROACH":
            # Close steadily from 2.0 m to inside the stop range.
            elapsed = t - m.state_since
            return max(0.30, 2.0 - 0.30 * elapsed)
        return state_ranges.get(m.state, 2.0)

    drive(machine, calls, seconds=45.0, caller_range=caller_range,
          gate_open=True, caller_visible=True)
    return machine


# --------------------------------------------------------------------------
# call script semantics
# --------------------------------------------------------------------------
def test_calls_active_at_respects_window():
    call = Call(caller="red", start_s=2.0, duration_s=3.0)
    assert not call.active_at(1.999)
    assert call.active_at(2.0)
    assert call.active_at(4.999)
    assert not call.active_at(5.0)


def test_calls_active_at_returns_every_sounding_call():
    calls = (
        Call(caller="red", start_s=0.0, duration_s=5.0),
        Call(caller="blue", start_s=2.0, duration_s=5.0),
    )
    assert [c.caller for c in calls_active_at(calls, 1.0)] == ["red"]
    assert sorted(c.caller for c in calls_active_at(calls, 3.0)) == ["blue", "red"]
    assert calls_active_at(calls, 20.0) == []


def test_exactly_one_expected_caller_sounds_at_a_time():
    """The scenario's genuine callers must never overlap.

    Reads the SHIPPED call script rather than a fixture, so a later edit that
    makes two real callers overlap fails here instead of quietly changing what
    "one adult calls at a time" means.
    """
    from render_come_here_recall import CALLS

    expected = [call for call in CALLS if call.expected]
    for t in np.arange(0.0, 60.0, 0.05):
        sounding = [call for call in expected if call.active_at(float(t))]
        assert len(sounding) <= 1, f"{len(sounding)} expected callers at t={t}"


def test_call_script_has_three_distinct_expected_callers():
    from render_come_here_recall import CALLS, EXPECTED_ORDER

    expected = [call.caller for call in CALLS if call.expected]
    assert expected == list(EXPECTED_ORDER)
    assert len(set(expected)) >= 3


def test_call_script_contains_an_unexpected_interrupting_call():
    """The no-steal rule needs something to refuse, or it is untested."""
    from render_come_here_recall import CALLS

    interrupts = [call for call in CALLS if not call.expected]
    assert interrupts, "no interrupting call in the script"
    # It must land while an expected call is being served.
    expected = [call for call in CALLS if call.expected]
    for interrupt in interrupts:
        assert any(
            call.start_s < interrupt.start_s < call.end_s for call in expected
        ), "the interrupting call does not overlap any genuine call"


# --------------------------------------------------------------------------
# state machine
# --------------------------------------------------------------------------
def test_machine_starts_listening_and_stays_until_a_call():
    machine = RecallMachine(ctrl_hz=50.0)
    drive(machine, (), seconds=5.0, caller_range=None, gate_open=False,
          caller_visible=False)
    assert machine.state == "LISTEN"
    assert machine.locked is None
    assert machine.cycles == []


def test_machine_requires_listen_minimum_before_searching():
    machine = RecallMachine(ctrl_hz=50.0)
    calls = (Call(caller="red", start_s=0.0, duration_s=30.0),)
    drive(machine, calls, seconds=LISTEN_MIN_S - 0.1, caller_range=2.0,
          gate_open=True, caller_visible=True)
    assert machine.state == "LISTEN"


def test_full_cycle_visits_every_state_in_order():
    machine = RecallMachine(ctrl_hz=50.0)
    visited = []

    def caller_range(t, m):
        if m.state == "APPROACH":
            return max(0.30, 2.0 - 0.30 * (t - m.state_since))
        return 2.0

    for index in range(int(40.0 / DT)):
        state, changed = machine.update(
            index * DT, calls=SIMPLE_CALLS,
            caller_range=caller_range(index * DT, machine),
            gate_open=True, caller_visible=True)
        if not visited or visited[-1] != state:
            visited.append(state)
    assert visited[:6] == [
        "LISTEN", "SEARCH", "CALLER_LOCK", "APPROACH", "ARRIVED", "LISTEN"]
    assert len(machine.cycles) == 1
    assert machine.cycles[0]["caller"] == "red"


def test_lock_requires_the_gate_to_stay_open_for_the_confirm_window():
    machine = RecallMachine(ctrl_hz=50.0)
    # Gate flickers open for one tick in every four: never a continuous window.
    drive(machine, SIMPLE_CALLS, seconds=20.0, caller_range=2.0,
          gate_open=lambda t, m: int(t / DT) % 4 == 0,
          caller_visible=True)
    assert machine.state == "SEARCH"
    assert machine.locked is None


def test_lock_needs_both_gate_and_visibility():
    machine = RecallMachine(ctrl_hz=50.0)
    drive(machine, SIMPLE_CALLS, seconds=20.0, caller_range=2.0,
          gate_open=True, caller_visible=False)
    assert machine.locked is None, "locked without the caller being visible"

    machine = RecallMachine(ctrl_hz=50.0)
    drive(machine, SIMPLE_CALLS, seconds=20.0, caller_range=2.0,
          gate_open=False, caller_visible=True)
    assert machine.locked is None, "locked without the acquisition gate"


def test_confirm_window_is_actually_enforced():
    """Exactly ACQUIRE_CONFIRM_S of continuous gate is needed, not one tick."""
    machine = RecallMachine(ctrl_hz=50.0)
    # LISTEN_MIN_S is measured from the state start, and the call itself only
    # begins at t=1.0, so drive well past both before expecting SEARCH.
    drive(machine, SIMPLE_CALLS, seconds=SIMPLE_CALLS[0].start_s + LISTEN_MIN_S + 0.2,
          caller_range=2.0, gate_open=False, caller_visible=False)
    assert machine.state == "SEARCH"
    # Hold the gate open for slightly LESS than the confirm window.  The
    # machine accumulates confirmation from the SECOND consecutive gated tick
    # (the first only registers the candidate), so drive one tick fewer than
    # the window needs and require that no lock has happened.
    resume = machine.state_since + DT
    ticks = int(round(ACQUIRE_CONFIRM_S / DT))
    drive(machine, SIMPLE_CALLS, seconds=(ticks - 1) * DT,
          caller_range=2.0, gate_open=True, caller_visible=True, start=resume)
    assert machine.state == "SEARCH", "locked before the confirm window elapsed"
    assert machine.locked is None
    # One more gated tick past the window and it does lock, so the test is
    # measuring the boundary rather than a permanently blocked machine.
    drive(machine, SIMPLE_CALLS, seconds=3 * DT, caller_range=2.0,
          gate_open=True, caller_visible=True,
          start=resume + (ticks - 1) * DT)
    assert machine.state == "CALLER_LOCK"


def test_approach_ends_at_the_stop_range():
    machine = run_one_full_cycle()
    cycle = machine.cycles[0]
    assert cycle["arrival_range_m"] <= STOP_RANGE + 1e-9
    assert not cycle["approach_timeout"]


def test_approach_times_out_if_the_caller_is_never_reached():
    machine = RecallMachine(ctrl_hz=50.0)
    drive(machine, SIMPLE_CALLS, seconds=APPROACH_MAX_S + 8.0,
          caller_range=5.0, gate_open=True, caller_visible=True)
    assert machine.cycles, "no cycle closed"
    assert machine.cycles[0]["approach_timeout"] is True


def test_a_served_call_is_not_served_twice():
    """A still-sounding call must not immediately restart the same recall.

    REGRESSION: run 2 of this behavior completed red -> yellow -> YELLOW, and
    the third cycle lasted 0.02 s with zero path because the duck was already
    standing at the standoff distance when the call was re-served.
    """
    # A call long enough that it is STILL SOUNDING after the recall completes,
    # which is the situation that produced the regression.
    calls = (Call(caller="red", start_s=1.0, duration_s=44.0),)
    machine = run_one_full_cycle(calls=calls)
    assert len(machine.cycles) == 1
    assert calls[0].active_at(35.0), "fixture no longer reproduces the bug"
    assert machine.state == "LISTEN"
    assert machine.locked is None


def test_a_genuinely_new_call_from_the_same_person_is_served():
    """The served ledger keys on (caller, start_s), not on the caller alone."""
    calls = (
        Call(caller="red", start_s=1.0, duration_s=12.0),
        Call(caller="red", start_s=20.0, duration_s=12.0),
    )
    machine = RecallMachine(ctrl_hz=50.0)

    def caller_range(t, m):
        if m.state == "APPROACH":
            return max(0.30, 2.0 - 0.40 * (t - m.state_since))
        return 2.0

    drive(machine, calls, seconds=45.0, caller_range=caller_range,
          gate_open=True, caller_visible=True)
    assert len(machine.cycles) == 2
    assert [c["caller"] for c in machine.cycles] == ["red", "red"]


# --------------------------------------------------------------------------
# the no-steal rule
# --------------------------------------------------------------------------
def test_a_call_arriving_mid_cycle_does_not_steal_the_lock():
    calls = (
        Call(caller="red", start_s=1.0, duration_s=25.0),
        Call(caller="blue", start_s=6.0, duration_s=3.0, expected=False),
    )
    machine = RecallMachine(ctrl_hz=50.0)

    def caller_range(t, m):
        if m.state == "APPROACH":
            return max(0.30, 2.0 - 0.12 * (t - m.state_since))
        return 2.0

    drive(machine, calls, seconds=40.0, caller_range=caller_range,
          gate_open=True, caller_visible=True)
    assert machine.cycles, "no cycle closed"
    assert machine.cycles[0]["caller"] == "red"
    assert all(c["caller"] != "blue" for c in machine.cycles)
    assert machine.refused_calls, "the interrupting call was not recorded"
    assert machine.refused_calls[0]["caller"] == "blue"
    assert machine.refused_calls[0]["busy_with"] == "red"


def test_refusal_is_recorded_once_per_interrupting_call():
    calls = (
        Call(caller="red", start_s=1.0, duration_s=25.0),
        Call(caller="blue", start_s=6.0, duration_s=4.0, expected=False),
    )
    machine = RecallMachine(ctrl_hz=50.0)
    drive(machine, calls, seconds=30.0,
          caller_range=lambda t, m: (
              max(0.30, 2.0 - 0.12 * (t - m.state_since))
              if m.state == "APPROACH" else 2.0),
          gate_open=True, caller_visible=True)
    blue = [r for r in machine.refused_calls if r["caller"] == "blue"]
    assert len(blue) == 1, f"blue refusal recorded {len(blue)} times"


def test_mutation_stealing_lock_breaks_the_order():
    """A machine that DOES let a later call steal produces the wrong order.

    This is the counterexample for ``no_caller_change`` and ``caller_order``:
    if the no-steal rule is removed, the served sequence changes.
    """

    class StealingMachine(RecallMachine):
        def update(self, t, *, calls, caller_range, gate_open, caller_visible):
            sounding = calls_active_at(calls, t)
            if self.busy and sounding:
                latest = max(sounding, key=lambda c: c.start_s)
                if self.active_call is not None and latest is not self.active_call:
                    # THE MUTATION: obey the newest caller instead of refusing.
                    self.active_call = latest
                    self.locked = latest.caller
                    if self.current:
                        self.current["caller"] = latest.caller
                        self.current["caller_changed"] = True
            return super().update(
                t, calls=calls, caller_range=caller_range,
                gate_open=gate_open, caller_visible=caller_visible)

    calls = (
        Call(caller="red", start_s=1.0, duration_s=25.0),
        Call(caller="blue", start_s=6.0, duration_s=8.0, expected=False),
    )
    machine = StealingMachine(ctrl_hz=50.0)
    drive(machine, calls, seconds=40.0,
          caller_range=lambda t, m: (
              max(0.30, 2.0 - 0.12 * (t - m.state_since))
              if m.state == "APPROACH" else 2.0),
          gate_open=True, caller_visible=True)
    assert machine.cycles, "mutated machine closed no cycle"
    served = [c["caller"] for c in machine.cycles]
    assert "blue" in served or any(
        c.get("caller_changed") for c in machine.cycles), (
        "the mutation did not change behavior, so the no-steal gate is vacuous")


# --------------------------------------------------------------------------
# approach controller
# --------------------------------------------------------------------------
@pytest.mark.parametrize("state", STATIONARY_STATES)
def test_command_is_exactly_zero_in_every_stationary_state(state):
    controller = ApproachController(ctrl_hz=50.0)
    for _ in range(200):
        command = controller.update(state, math.radians(40.0), 2.0)
        assert command.tolist() == [0.0, 0.0, 0.0]


def test_command_is_zero_the_tick_after_leaving_approach():
    """No decaying tail: the gate tests for EXACT zero, not 'small'."""
    controller = ApproachController(ctrl_hz=50.0)
    for _ in range(50):
        controller.update("APPROACH", 0.0, 2.0)
    assert controller.update("ARRIVED", 0.0, 0.55).tolist() == [0.0, 0.0, 0.0]


def test_approach_emits_a_command_above_the_measured_gait_onset():
    controller = ApproachController(ctrl_hz=50.0)
    command = controller.update("APPROACH", 0.0, 2.0)
    assert command[0] >= VX_MIN_EFFECTIVE
    assert command[0] == pytest.approx(VX_CRUISE)


def test_no_decorative_sub_onset_command_is_ever_emitted():
    """MEASURED: vx=0.20 moves the robot 10 mm in 6 s. vx=0.24 moves 515 mm.

    Any command strictly between zero and the onset would show as motion in the
    HUD and produce none on the floor.
    """
    controller = ApproachController(ctrl_hz=50.0)
    for error_deg in range(-180, 181, 5):
        for distance in (0.30, 0.55, 0.80, 1.20, 2.50):
            vx, _, _ = controller.raw_command(
                "APPROACH", math.radians(error_deg), distance)
            assert vx == 0.0 or vx >= VX_MIN_EFFECTIVE, (
                f"decorative vx={vx} at {error_deg} deg, {distance} m")


def test_controller_stops_inside_the_stop_range():
    controller = ApproachController(ctrl_hz=50.0)
    assert controller.raw_command("APPROACH", 0.0, STOP_RANGE - 0.01) == (
        0.0, 0.0, 0.0)
    assert controller.raw_command("APPROACH", 0.0, STOP_RANGE + 0.30)[0] > 0.0


def test_turn_signs_follow_the_heading_error():
    controller = ApproachController(ctrl_hz=50.0)
    _, _, wz_left = controller.raw_command("APPROACH", math.radians(90.0), 2.0)
    _, _, wz_right = controller.raw_command("APPROACH", math.radians(-90.0), 2.0)
    assert wz_left > 0.0, "a caller to the left must produce a left turn"
    assert wz_right < 0.0, "a caller to the right must produce a right turn"


def test_turn_dead_zones_are_asymmetric_as_measured():
    """The stock policy is NOT mirror-symmetric; the dead zones must not be.

    MEASURED at vx=0.24: wz=+0.25 turns +0.7 deg/s while wz=-0.25 turns
    -8.0 deg/s.  A shared dead zone would either waste the usable right
    authority or emit a useless left command.

    The observable consequence is a BAND of heading errors that produce a right
    turn but no left turn.  With the measured gains that band is roughly
    14-19 deg; the test asserts the band exists and is non-empty rather than
    pinning its exact edges.
    """
    assert WZ_MIN_LEFT > WZ_MIN_RIGHT
    controller = ApproachController(ctrl_hz=50.0)
    asymmetric = [
        deg for deg in range(1, 45)
        if controller.raw_command("APPROACH", math.radians(deg), 2.0)[2] == 0.0
        and controller.raw_command("APPROACH", math.radians(-deg), 2.0)[2] < 0.0
    ]
    assert asymmetric, (
        "no heading error turns right but not left: the dead zones are "
        "effectively symmetric, contradicting the measured yaw rates")
    # Below the smaller dead zone BOTH sides are zero; above the larger one
    # both sides turn.  That brackets the asymmetric band.
    assert controller.raw_command("APPROACH", math.radians(-5.0), 2.0)[2] == 0.0
    assert controller.raw_command("APPROACH", math.radians(35.0), 2.0)[2] > 0.0


def test_mutation_symmetric_dead_zone_emits_an_ineffective_left_turn():
    """Counterexample for the asymmetry: a mirrored controller emits a command
    the MEASURED policy cannot act on (+0.7 deg/s at wz=+0.25)."""

    class MirroredController(ApproachController):
        def raw_command(self, state, heading_error, caller_range):
            if state != "APPROACH":
                return (0.0, 0.0, 0.0)
            wz = math.copysign(
                min(abs(1.05 * heading_error), 0.85), heading_error)
            if abs(wz) < WZ_MIN_RIGHT:      # THE MUTATION: one shared dead zone
                wz = 0.0
            return (VX_CRUISE, 0.0, wz)

    mirrored = MirroredController(ctrl_hz=50.0)
    wz = mirrored.raw_command("APPROACH", math.radians(15.0), 2.0)[2]
    assert 0.0 < wz < WZ_MIN_LEFT, (
        "the mutation must produce a left command below the measured usable "
        "threshold, which is exactly what the real controller refuses to emit")


def test_large_heading_error_slows_down_to_close_the_turn():
    controller = ApproachController(ctrl_hz=50.0)
    straight = controller.raw_command("APPROACH", 0.0, 2.0)[0]
    turning = controller.raw_command("APPROACH", math.radians(-120.0), 2.0)[0]
    assert turning < straight


# --------------------------------------------------------------------------
# standoff band
# --------------------------------------------------------------------------
def test_standoff_band_is_physically_justified():
    """The band must clear the bodies, not merely look tidy.

    Adult torso capsule radius is 0.078 m and the duck's planar half-extent is
    about 0.09 m, so the bodies touch at roughly 0.17 m.
    """
    bodies_touch_at = 0.078 + 0.09
    assert STANDOFF_MIN > bodies_touch_at + 0.25
    assert STANDOFF_MIN < STANDOFF_TARGET < STANDOFF_MAX
    assert STANDOFF_MAX - STANDOFF_MIN >= 0.20


def test_stop_range_accounts_for_the_measured_coast():
    """MEASURED coast after commanding zero: 4.5-8.9 mm, flat to +2.5 s."""
    assert STOP_RANGE == pytest.approx(STANDOFF_TARGET + COAST_M)
    assert COAST_M < 0.02
    # Stopping at STOP_RANGE and coasting must still land inside the band.
    assert STANDOFF_MIN <= STOP_RANGE - COAST_M <= STANDOFF_MAX
    assert STANDOFF_MIN <= STOP_RANGE + COAST_M <= STANDOFF_MAX


# --------------------------------------------------------------------------
# scene layout invariants (pure geometry, no MuJoCo)
# --------------------------------------------------------------------------
def test_at_least_four_adults_are_present():
    assert len(ADULT_NAMES) >= 4


def test_three_distinct_callers_are_defined():
    assert len(set(CALLER_NAMES)) >= 3
    assert set(CALLER_NAMES) <= set(ADULT_NAMES)


def test_every_adult_keeps_moving_for_the_whole_rollout():
    """Nobody freezes: each adult's speed stays strictly positive."""
    for name in ADULT_NAMES:
        speeds = [
            float(np.linalg.norm(STATION_BY_NAME[name].at(float(t))[1]))
            for t in np.arange(0.0, 54.0, 0.5)
        ]
        assert min(speeds) > 0.02, f"{name} nearly stops (min {min(speeds):.4f})"


def test_adults_never_interpenetrate():
    distance, first, second = min_person_separation(54.0, dt=0.25)
    # Two torsos of radius 0.078 m touch at 0.156 m; require far more.
    assert distance > 1.0, f"{first} and {second} come within {distance:.3f} m"


def test_call_bearings_are_widely_separated():
    """The three callers must be reachable from genuinely different directions.

    Replays the recall geometry: from the origin to caller 1, then from that
    standoff point to caller 2, and so on.
    """
    from render_come_here_recall import CALLS

    expected = [call for call in CALLS if call.expected]
    duck = np.zeros(2)
    bearings = []
    for call in expected:
        target = crowd_at(call.start_s + 2.0)[call.caller].pos
        delta = target - duck
        distance = float(np.linalg.norm(delta))
        bearings.append(math.degrees(math.atan2(delta[1], delta[0])))
        duck = target - STANDOFF_TARGET * delta / distance
    for i in range(len(bearings)):
        for j in range(i + 1, len(bearings)):
            gap = abs((bearings[i] - bearings[j] + 180.0) % 360.0 - 180.0)
            assert gap >= 60.0, (
                f"call bearings {bearings[i]:.1f} and {bearings[j]:.1f} "
                f"are only {gap:.1f} deg apart")


def test_acquisition_cone_fits_inside_the_camera_frustum():
    """A gate wider than the frame would accept a caller clinging to the edge.

    The attention camera is 58 deg vertical on a 300x220 PiP: +/-29.0 deg
    vertical and +/-37.1 deg horizontal.
    """
    assert ACQUIRE_CONE_DEG < 29.0
    assert ACQUIRE_CONE_DEG >= 5.0


def test_timings_leave_room_for_three_recalls():
    minimum_cycle = LISTEN_MIN_S + ACQUIRE_CONFIRM_S + LOCK_HOLD_S + ARRIVED_HOLD_S
    assert 3 * minimum_cycle < 54.0


def test_wrap_angle_is_a_proper_wrap():
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap_angle(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)
    for angle in np.arange(-20.0, 20.0, 0.37):
        assert -math.pi <= wrap_angle(float(angle)) < math.pi
