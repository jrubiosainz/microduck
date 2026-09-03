#!/usr/bin/env python3
"""The state machine, the controller, the plan and the detector.

Pure logic on hand-built inputs: no MuJoCo, no policy, no physics anywhere in
this file.  That is the point of keeping those layers free of the simulator -
every transition rule and every classification rule can be exercised directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from patrol_branch import RESUME_TOLERANCE_M
from patrol_control import Interlock, PatrolController
from patrol_detect import INTRUSION_DWELL_S, Detector
from patrol_episode import Sense
from patrol_facility import CHECKPOINT_NAMES, RESTRICTED_ZONE
from patrol_machine import PatrolMachine
from patrol_plan import PatrolPlan, pursuit_point
from patrol_states import (
    ATTENDED_RADIUS_M,
    CHECKPOINT_STOP_S,
    OBSERVE_ANGLES_DEG,
    STANDOFF_MAX_M,
    STANDOFF_MIN_M,
    UNATTENDED_S,
    VX_APPROACH,
    VX_ONSET,
    VX_PATROL,
    VX_SETTLE,
    WZ_MAX_LEFT,
    WZ_MAX_RIGHT,
    ZERO_COMMAND_STATES,
)


def drive(machine, sense_for, seconds=40.0, dt=0.02):
    """Run the machine over hand-built senses, returning the state sequence."""
    seen = []
    for index in range(int(seconds / dt)):
        t = index * dt
        state, changed = machine.update(t, sense_for(t, machine))
        if changed or not seen:
            seen.append(state)
    return seen


# -- the controller -----------------------------------------------------------
def test_every_zero_command_state_returns_an_exact_zero():
    """Not a small number, not a decayed one.  The gate checks this literally."""
    controller = PatrolController()
    for state in ZERO_COMMAND_STATES:
        command = controller.raw_command(
            state, (0.0, 0.0), 0.0, target_xy=(5.0, 5.0), remaining_m=5.0)
        assert command == (0.0, 0.0, 0.0), state


def test_the_controller_never_emits_a_lateral_term():
    """There is no strafe on this policy, so vy must be exactly zero always."""
    controller = PatrolController()
    for state in ("PATROL", "APPROACH", "RETURN_TO_PATROL", "RESUME"):
        for yaw in np.linspace(-np.pi, np.pi, 24):
            for target in ((1.0, 0.0), (-1.0, 0.6), (0.2, -1.4)):
                command = controller.raw_command(
                    state, (0.0, 0.0), float(yaw), target_xy=target,
                    remaining_m=3.0)
                assert command[1] == 0.0


def test_no_walking_command_ever_falls_below_the_measured_gait_onset():
    """MEASURED: vx=0.22 produces 0.009 m in 6 s - no gait at all.  A command
    between zero and the onset is the appearance of care with none of the
    physics."""
    controller = PatrolController()
    for state in ("PATROL", "APPROACH", "RETURN_TO_PATROL", "RESUME"):
        for remaining in (0.05, 0.2, 0.28, 1.0, 5.0):
            vx, _, _ = controller.raw_command(
                state, (0.0, 0.0), 0.0, target_xy=(5.0, 0.0),
                remaining_m=remaining)
            assert vx == 0.0 or vx >= VX_ONSET, (state, remaining, vx)


def test_the_approach_uses_its_own_slower_measured_command():
    controller = PatrolController()
    vx, _, _ = controller.raw_command(
        "APPROACH", (0.0, 0.0), 0.0, target_xy=(5.0, 0.0), remaining_m=5.0,
        approach=True)
    assert vx == VX_APPROACH
    vx, _, _ = controller.raw_command(
        "PATROL", (0.0, 0.0), 0.0, target_xy=(5.0, 0.0), remaining_m=5.0)
    assert vx == VX_PATROL


def test_the_duck_eases_into_a_checkpoint_rather_than_walking_through_it():
    controller = PatrolController()
    vx, _, _ = controller.raw_command(
        "PATROL", (0.0, 0.0), 0.0, target_xy=(5.0, 0.0), remaining_m=0.2)
    assert vx == VX_SETTLE


def test_the_interlock_refuses_before_the_target_is_consulted():
    controller = PatrolController()
    command = controller.raw_command(
        "PATROL", (0.0, 0.0), 0.0, target_xy=(5.0, 0.0), remaining_m=5.0,
        interlock=Interlock(True, "somebody is ahead", "rosa"))
    assert command == (0.0, 0.0, 0.0)


def test_a_state_with_no_target_has_no_command():
    """This is how a zero is structural rather than remembered."""
    controller = PatrolController()
    assert controller.raw_command(
        "PATROL", (0.0, 0.0), 0.0, target_xy=None) == (0.0, 0.0, 0.0)


def test_turning_in_place_is_always_exactly_zero():
    """MEASURED at 1.6 deg/s across the whole command range - not a turn."""
    controller = PatrolController()
    for desired in np.linspace(-np.pi, np.pi, 16):
        assert controller.spin_to(float(desired), 0.0) == 0.0


def test_the_yaw_signs_are_independently_bounded():
    """Each sign saturates at its own measured ceiling.

    The errors are just under a half turn on each side, because ``wrap_angle``
    maps exactly ``+pi`` onto ``-pi`` - so asking for a half turn to the left
    is genuinely ambiguous and the controller is entitled to answer either way.
    """
    controller = PatrolController()
    assert controller.yaw_to(3.0, 0.0) == pytest.approx(WZ_MAX_LEFT)
    assert controller.yaw_to(-3.0, 0.0) == pytest.approx(-WZ_MAX_RIGHT)


def test_each_yaw_sign_has_its_own_measured_dead_band():
    """MEASURED: wz=+0.10 gave +0.7 deg/s while wz=-0.10 gave -8.7 deg/s, so a
    small LEFT command is swallowed by the policy's own right bias."""
    controller = PatrolController()
    from patrol_states import KP_YAW_LEFT, KP_YAW_RIGHT, WZ_MIN_LEFT, WZ_MIN_RIGHT
    tiny_left = (WZ_MIN_LEFT * 0.5) / KP_YAW_LEFT
    tiny_right = (WZ_MIN_RIGHT * 0.5) / KP_YAW_RIGHT
    assert controller.yaw_to(tiny_left, 0.0) == 0.0
    assert controller.yaw_to(-tiny_right, 0.0) == 0.0
    assert WZ_MIN_LEFT > WZ_MIN_RIGHT


# -- the plan and its memory ---------------------------------------------------
def test_the_plan_walks_the_checkpoints_in_order():
    plan = PatrolPlan()
    for name in CHECKPOINT_NAMES:
        assert plan.target_name == name
        plan.complete_checkpoint(name)
    assert plan.finished_circuit
    assert plan.target_name == "guard-post"


def test_completing_a_checkpoint_out_of_order_raises():
    """The ordering is enforced where the record is MADE, and the gate then
    checks the recorded sequence independently."""
    plan = PatrolPlan()
    with pytest.raises(ValueError):
        plan.complete_checkpoint("north-bay")


def test_an_interruption_preserves_the_target_and_the_resume_point():
    plan = PatrolPlan()
    plan.complete_checkpoint("dock-gate")
    entry = plan.interrupt(10.0, (0.5, 0.1), "anomaly", target="crate")
    assert entry.target_name == "east-aisle"
    assert plan.target_name == "east-aisle"
    assert plan.target_index == 1


def test_an_interruption_cannot_advance_the_patrol():
    """This is the structural reason a diversion cannot lose the duck's place."""
    plan = PatrolPlan()
    plan.complete_checkpoint("dock-gate")
    before = plan.target_index
    plan.interrupt(10.0, (0.5, 0.1), "anomaly", target="crate")
    assert plan.target_index == before
    assert plan.target_name == "east-aisle"


def test_resuming_records_the_measured_return_error_and_the_same_target():
    plan = PatrolPlan()
    plan.complete_checkpoint("dock-gate")
    plan.interrupt(10.0, (0.5, 0.1), "anomaly", target="crate")
    entry = plan.resume(30.0, (0.55, 0.13))
    assert entry.resumed_target_name == "east-aisle"
    assert entry.route_preserved if hasattr(entry, "route_preserved") \
        else entry.as_record()["route_preserved"]
    assert entry.return_error_m == pytest.approx(
        float(np.hypot(0.05, 0.03)), abs=1e-6)


def test_a_second_interruption_cannot_overwrite_an_open_one():
    plan = PatrolPlan()
    plan.interrupt(10.0, (0.5, 0.1), "first")
    with pytest.raises(RuntimeError):
        plan.interrupt(11.0, (0.6, 0.2), "second")


def test_resuming_without_an_interruption_raises():
    with pytest.raises(RuntimeError):
        PatrolPlan().resume(1.0, (0.0, 0.0))


def test_the_pursuit_point_stays_a_lookahead_ahead_and_snaps_at_the_end():
    point = pursuit_point((0.0, 0.0), (10.0, 0.0), lookahead_m=0.34)
    assert point[0] == pytest.approx(0.34)
    close = pursuit_point((0.0, 0.0), (0.10, 0.0), lookahead_m=0.34)
    assert close[0] == pytest.approx(0.10)


# -- the detector ---------------------------------------------------------------
def feed_object(detector, name, position, seconds, *, t0=0.0, people=None,
                visible=True):
    """Feed one body to the detector for ``seconds``, as the camera would."""
    steps = int(seconds / detector.dt)
    positions = {name: np.asarray(position, dtype=float)}
    if people:
        positions.update({k: np.asarray(v, dtype=float)
                          for k, v in people.items()})
    visibility = {k: {"visible": visible} for k in positions}
    for index in range(steps):
        detector.feed(t0 + index * detector.dt, visibility=visibility,
                      positions=positions, duck_xy=(0.0, 0.0))
    return detector


def test_an_object_outside_the_camera_gate_is_never_a_candidate():
    """THE CAMERA GATE IS THE WHOLE POINT: nothing can be detected that the head
    camera did not resolve."""
    detector = Detector(0.02)
    ready = feed_object(detector, "crate", (1.0, 0.0), 3.0, visible=False)
    assert detector.observations == {}
    assert detector.classify("crate") is None


def test_an_object_beyond_the_detection_range_is_never_a_candidate():
    detector = Detector(0.02)
    feed_object(detector, "crate", (9.0, 0.0), 3.0)
    assert "crate" not in detector.observations


def test_an_unattended_stationary_object_is_suspicious():
    detector = Detector(0.02)
    feed_object(detector, "crate", (1.0, 0.0), UNATTENDED_S + 1.0)
    verdict = detector.classify("crate")
    assert verdict is not None
    assert verdict.verdict == "suspicious"
    assert verdict.investigate


def test_an_object_that_has_not_stood_long_enough_is_UNRESOLVED_not_innocent():
    """Saying 'not yet' is more honest than defaulting either way."""
    detector = Detector(0.02)
    feed_object(detector, "crate", (1.0, 0.0), 1.0)
    assert detector.classify("crate") is None


def test_an_object_on_a_stow_area_is_benign_however_long_it_stands_there():
    from patrol_actors import TROLLEY_XY
    detector = Detector(0.02)
    feed_object(detector, "trolley", TROLLEY_XY, UNATTENDED_S + 4.0)
    verdict = detector.classify("trolley")
    assert verdict is not None
    assert verdict.verdict == "benign"
    assert not verdict.investigate


def test_an_attended_object_is_benign():
    detector = Detector(0.02)
    feed_object(detector, "crate", (1.0, 0.0), UNATTENDED_S + 2.0,
                people={"emil": (1.3, 0.0)})
    verdict = detector.classify("crate")
    assert verdict is not None
    assert verdict.verdict == "benign"
    assert "emil" in verdict.rule


def test_the_benign_rules_are_checked_before_the_suspicious_one():
    """A guard robot must rule out the ordinary explanations before escalating,
    so the escalation cannot fire on something a cheaper rule already
    explained."""
    from patrol_actors import TROLLEY_XY
    detector = Detector(0.02)
    feed_object(detector, "trolley", TROLLEY_XY, UNATTENDED_S + 6.0,
                people={"emil": (TROLLEY_XY[0] + 0.4, TROLLEY_XY[1])})
    verdict = detector.classify("trolley")
    assert verdict.verdict == "benign"


def test_a_person_merely_walking_past_the_zone_is_not_an_intrusion():
    """The dwell is what separates entering from crossing."""
    detector = Detector(0.02)
    feed_object(detector, "visitor", RESTRICTED_ZONE.center,
                INTRUSION_DWELL_S * 0.5)
    assert detector.classify("visitor") is None


def test_a_person_who_stays_in_the_zone_is_an_intrusion():
    detector = Detector(0.02)
    feed_object(detector, "visitor", RESTRICTED_ZONE.center,
                INTRUSION_DWELL_S + 1.0)
    verdict = detector.classify("visitor")
    assert verdict is not None
    assert verdict.verdict == "intrusion"
    assert verdict.investigate


def test_staff_outside_the_zone_are_never_classified_at_all():
    """A robot that produced a verdict about every person it saw would be
    reporting its own colleagues."""
    detector = Detector(0.02)
    feed_object(detector, "rosa", (1.0, 0.0), 20.0)
    assert detector.classify("rosa") is None


def test_the_confidence_proxy_is_bounded_and_rises_with_the_rule_margin():
    detector_near = Detector(0.02)
    feed_object(detector_near, "crate", (1.0, 0.0), UNATTENDED_S + 0.2)
    detector_far = Detector(0.02)
    feed_object(detector_far, "crate", (1.0, 0.0), UNATTENDED_S + 12.0)
    near = detector_near.classify("crate").confidence
    far = detector_far.classify("crate").confidence
    assert 0.5 <= near <= far <= 0.99


def test_a_recorded_verdict_settles_the_body_so_it_is_not_re_detected():
    detector = Detector(0.02)
    feed_object(detector, "crate", (1.0, 0.0), UNATTENDED_S + 1.0)
    verdict = detector.classify("crate")
    detector.record(verdict)
    assert "crate" in detector.settled
    ready = detector.feed(
        99.0, visibility={"crate": {"visible": True}},
        positions={"crate": np.array([1.0, 0.0])}, duck_xy=(0.0, 0.0))
    assert "crate" not in ready


# -- the machine ----------------------------------------------------------------
def test_the_duck_must_be_stopped_before_it_scans():
    """A robot that scanned while still rolling would never reach SCAN."""
    machine = PatrolMachine()
    def sense_for(t, m):
        return Sense(target_name="dock-gate", at_target=True, settled=False)
    states = drive(machine, sense_for, seconds=5.0)
    assert "CHECKPOINT_STOP" in states
    assert "SCAN" not in states


def test_a_stopped_duck_reaches_scan_after_the_minimum_dwell():
    machine = PatrolMachine()
    def sense_for(t, m):
        return Sense(target_name="dock-gate", at_target=True, settled=True)
    states = drive(machine, sense_for, seconds=CHECKPOINT_STOP_S + 1.0)
    assert states[:2] == ["CHECKPOINT_STOP", "SCAN"]


def test_a_completed_sweep_with_nothing_to_see_reports_clear():
    machine = PatrolMachine()
    def sense_for(t, m):
        return Sense(target_name="dock-gate", at_target=True, settled=True,
                     scan_arc_deg=200.0, scan_complete=m.state == "SCAN")
    states = drive(machine, sense_for, seconds=8.0)
    assert "CLEAR" in states
    assert "DETECT" not in states


def test_a_benign_candidate_never_reaches_the_investigation_states():
    """A dismissal costs no walking: the duck never takes a step toward it."""
    machine = PatrolMachine()
    def sense_for(t, m):
        return Sense(target_name="dock-gate", at_target=True, settled=True,
                     candidate="trolley", candidate_verdict="benign",
                     candidate_visible=True, candidate_investigate=False)
    states = drive(machine, sense_for, seconds=12.0)
    assert "DETECT" in states and "CLASSIFY" in states
    assert "INVESTIGATE_PLAN" not in states
    assert "APPROACH" not in states
    assert "RETURN_TO_PATROL" not in states


def test_an_escalating_candidate_runs_the_whole_investigation_branch():
    machine = PatrolMachine()
    def sense_for(t, m):
        return Sense(
            target_name="east-aisle", at_target=True, settled=True,
            candidate="crate", candidate_verdict="suspicious",
            candidate_visible=True, candidate_investigate=True,
            standoff_ready=True,
            in_standoff_band=m.state in ("APPROACH", "OBSERVE"),
            target_range_m=0.9,
            at_resume_point=m.state == "RETURN_TO_PATROL")
    states = drive(machine, sense_for, seconds=40.0)
    for expected in ("DETECT", "INVESTIGATE_PLAN", "APPROACH", "OBSERVE",
                     "CLASSIFY", "RETURN_TO_PATROL", "RESUME", "PATROL"):
        assert expected in states, (expected, states)


def test_the_approach_ends_on_measured_range_not_on_arrival():
    """A badly-placed standoff point cannot produce a close approach."""
    machine = PatrolMachine()
    machine.state = "APPROACH"
    sense = Sense(candidate="crate", candidate_visible=True,
                  in_standoff_band=False, standoff_remaining_m=0.0,
                  target_range_m=0.2)
    machine.update(0.02, sense)
    assert machine.state == "APPROACH"
    machine.update(0.04, Sense(in_standoff_band=True, target_range_m=0.9))
    assert machine.state == "OBSERVE"


def test_every_declared_observation_angle_is_held():
    machine = PatrolMachine()
    from patrol_episode import Investigation
    machine.open_investigation(Investigation(
        index=0, target="crate", detected_at_s=0.0, detect_range_m=1.0,
        interrupted_checkpoint="east-aisle", interrupted_index=1))
    machine.state = "OBSERVE"
    machine.state_since = 0.0
    for index in range(int(30.0 / 0.02)):
        t = index * 0.02
        machine.update(t, Sense(candidate_visible=True, target_range_m=0.9))
        if machine.state != "OBSERVE":
            break
    held = [o.angle_deg for o in machine.investigation.observations]
    assert held == list(OBSERVE_ANGLES_DEG)


def test_the_machine_reaches_home_only_after_the_circuit_is_finished():
    machine = PatrolMachine()
    machine.update(0.02, Sense(target_name="guard-post", at_target=True,
                               finished_circuit=True, settled=True))
    assert machine.state == "HOME"


def test_home_waits_for_the_body_to_actually_settle():
    """The gait cannot halt instantly; timing from arrival counts the coast."""
    machine = PatrolMachine()
    machine.state = "HOME"
    machine.state_since = 0.0
    for index in range(1, 200):
        machine.update(index * 0.02, Sense(settled=False))
    assert machine.state == "HOME"


def test_every_declared_state_has_a_handler():
    from patrol_states import STATES
    machine = PatrolMachine()
    for state in STATES:
        assert hasattr(machine, f"_{state.lower()}_state"), state
