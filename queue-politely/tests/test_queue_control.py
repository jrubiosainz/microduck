#!/usr/bin/env python3
"""The controller and the state machine, on hand-built inputs.

No MuJoCo.  Two tests here pin defects that measurement caught and that a reader
would otherwise assume could not happen:
``test_arrival_requires_the_arc_to_FALL_to_the_target`` (the inverted comparison
that produced 31 dribbling advances) and
``test_the_advance_target_tracks_a_moving_predecessor`` (the frozen setpoint
that stopped the duck a metre short).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from queue_constants import STATIONARY_STATES, VX_ONSET  # noqa: E402
from queue_control import QueueController, _bendiness  # noqa: E402
from queue_geometry import STANDOFF_TARGET_M  # noqa: E402
from queue_machine import QueueMachine  # noqa: E402
from queue_model import judge_gaps, read_queue  # noqa: E402
from queue_path import PATH  # noqa: E402
from queue_people import (  # noqa: E402
    ADULT_HALF_EXTENT_M,
    QUEUE,
    QUEUE_NAMES,
)

TRUTH = list(QUEUE_NAMES)
STATIONS = {adult.name: adult.initial_arc for adult in QUEUE}


# --------------------------------------------------------------- controller
def test_every_stationary_state_commands_exactly_zero():
    controller = QueueController()
    for state in STATIONARY_STATES:
        command = controller.raw_command(
            state, (1.0, -1.0), 0.4, duck_arc=2.0, target_arc=1.0)
        assert command == (0.0, 0.0, 0.0)
        assert all(value == 0.0 for value in command)


def test_the_controller_never_emits_a_sub_onset_forward_command():
    """Gait onset is a cliff; a command below it is motion in the HUD only."""
    controller = QueueController()
    for arc in np.linspace(0.05, PATH.length, 90):
        for target in (0.0, max(arc - 0.55, 0.0)):
            command = controller.raw_command(
                "ADVANCE", tuple(PATH.point_at(arc)),
                PATH.travel_heading_at(arc), duck_arc=float(arc),
                target_arc=float(target))
            assert command[0] == 0.0 or command[0] >= VX_ONSET


def test_the_controller_stops_at_the_target_and_never_passes_it():
    """Stopping is graded against the controller's own 0.02 m release band.

    The machine's ARRIVE_TOLERANCE_M (0.06 m) is a different quantity: it is how
    close counts as arrived for a state transition, and it is deliberately
    looser than the command release so the duck settles rather than hunting.
    """
    controller = QueueController()
    target = 1.10
    # Clear of the 0.02 m release band on either side, so the assertion does
    # not turn on a floating-point tie at the boundary itself.
    for arc in (1.30, 1.15, 1.125):
        command = controller.raw_command(
            "ADVANCE", tuple(PATH.point_at(arc)),
            PATH.travel_heading_at(arc), duck_arc=arc, target_arc=target)
        assert command[0] >= VX_ONSET
    for arc in (1.115, 1.10, 0.90):
        command = controller.raw_command(
            "ADVANCE", tuple(PATH.point_at(arc)),
            PATH.travel_heading_at(arc), duck_arc=arc, target_arc=target)
        assert command == (0.0, 0.0, 0.0)


def test_the_pursuit_point_stays_on_the_path_around_the_fold():
    """What makes the duck follow the bend instead of cutting across it."""
    from queue_constants import LOOKAHEAD_M
    for arc in np.linspace(0.6, 2.4, 40):
        aim_arc = max(float(arc) - LOOKAHEAD_M, 0.0)
        assert PATH.project(PATH.point_at(aim_arc))[2] == pytest.approx(
            0.0, abs=2e-3)


def test_yaw_command_uses_the_stronger_sign_to_turn_into_the_fold():
    """The fold turns one way, and that way is the measured strong direction."""
    controller = QueueController()
    commands = []
    for arc in np.linspace(1.0, 2.2, 25):
        command = controller.raw_command(
            "ADVANCE", tuple(PATH.point_at(arc)),
            PATH.travel_heading_at(arc), duck_arc=float(arc),
            target_arc=float(arc) - 0.55)
        commands.append(command[2])
    assert min(commands) < -0.15
    assert all(value <= 0.02 for value in commands)


# ------------------------------------------------------------------ machine
def _drive(machine, *, states, duck_arc, reading, gaps, predecessor_arc,
           remaining, dt=0.02, seconds=6.0):
    t = 0.0
    for _ in range(int(seconds / dt)):
        state, _ = machine.update(
            t, duck_arc=duck_arc, duck_off_path_m=0.05, reading=reading,
            gaps=gaps, predecessor_arc=predecessor_arc,
            predecessors_remaining=remaining)
        states.append(state)
        t += dt
    return t


def test_the_machine_reaches_join_only_after_observing_and_identifying():
    positions = {name: tuple(PATH.point_at(arc))
                 for name, arc in STATIONS.items()}
    reading = read_queue(positions)
    gaps = judge_gaps(reading, ADULT_HALF_EXTENT_M)
    machine = QueueMachine()
    seen: list[str] = []
    # Held OFF the path and short of the entry, so APPROACH is a state the
    # machine has to leave rather than one it falls out of immediately.
    t = 0.0
    for _ in range(60):
        state, _ = machine.update(
            t, duck_arc=PATH.length + 0.4, duck_off_path_m=0.95,
            reading=reading, gaps=gaps, predecessor_arc=None,
            predecessors_remaining=5)
        seen.append(state)
        t += 0.02
    assert set(seen) == {"APPROACH"}
    for _ in range(500):
        state, _ = machine.update(
            t, duck_arc=4.30, duck_off_path_m=0.05, reading=reading,
            gaps=gaps, predecessor_arc=None, predecessors_remaining=5)
        seen.append(state)
        t += 0.02
    order = []
    for state in seen:
        if not order or order[-1] != state:
            order.append(state)
    assert order[:5] == ["APPROACH", "OBSERVE_QUEUE", "IDENTIFY_TAIL",
                         "EVALUATE_GAPS", "JOIN"]
    assert machine.joined_behind == "eriksson"
    assert machine.target_arc == pytest.approx(
        STATIONS["eriksson"] + STANDOFF_TARGET_M, abs=1e-6)


def test_arrival_requires_the_arc_to_FALL_to_the_target():
    """The inverted comparison that produced 31 dribbling advances.

    The duck travels toward arc zero, so being at a LARGER arc than the target
    means it has not arrived.  A machine that accepted that would 'arrive'
    before moving at all.
    """
    machine = QueueMachine()
    machine.state = "JOIN"
    machine.target_arc = 1.00
    machine.state_since = 0.0
    for tick in range(120):
        state, _ = machine.update(
            tick * 0.02, duck_arc=3.50, duck_off_path_m=0.02,
            reading=read_queue({}), gaps=[], predecessor_arc=None,
            predecessors_remaining=1)
    assert state == "JOIN"


def test_the_advance_target_tracks_a_moving_predecessor():
    """A queue advance follows a person, not a station frozen at trigger time."""
    machine = QueueMachine()
    machine.state = "ADVANCE"
    machine.state_since = 0.0
    machine.predecessor = "eriksson"
    machine._cycle = {"kind": "advance", "started_s": 0.0}
    machine.update(0.02, duck_arc=2.60, duck_off_path_m=0.02,
                   reading=read_queue({}), gaps=[], predecessor_arc=1.90,
                   predecessors_remaining=1)
    first = machine.target_arc
    machine.update(0.04, duck_arc=2.55, duck_off_path_m=0.02,
                   reading=read_queue({}), gaps=[], predecessor_arc=1.60,
                   predecessors_remaining=1)
    assert machine.target_arc < first
    assert machine.target_arc == pytest.approx(1.60 + STANDOFF_TARGET_M)


def test_at_counter_is_unreachable_while_anybody_is_still_ahead():
    machine = QueueMachine()
    machine.state = "WAIT"
    machine.state_since = 0.0
    machine.predecessor = "eriksson"
    for tick in range(400):
        state, _ = machine.update(
            tick * 0.02, duck_arc=0.60, duck_off_path_m=0.02,
            reading=read_queue({}), gaps=[], predecessor_arc=0.02,
            predecessors_remaining=1)
        assert state != "AT_COUNTER"
