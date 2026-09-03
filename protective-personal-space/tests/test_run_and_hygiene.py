#!/usr/bin/env python3
"""The real run, pinned; and the structural claims about the source itself.

Three kinds of test live here.

* **Pins** - the ordered episodes, the measured bearings and the 27/27 gate
  results of the COMMITTED run, so a behavior change has to update the artifact
  and say so here rather than drifting quietly away from the README.
* **Hygiene** - the import graph parsed with ``ast``, so "the duck never read
  the choreography" is enforced by module structure rather than promised in a
  docstring.
* **Physics** - a genuinely short rollout, for the invariants that only exist
  once MuJoCo and the ONNX policy are in the loop.
"""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from policy_runtime import (ACTION_SCALE, CTRL_HZ, FALLEN_TRUNK_Z,
                            GYRO_SENSOR, NOMINAL_TRUNK_Z, OBS_DIM, SCENE_XML,
                            build_observation, gyro_address)
from pps_cast import ALL_NAMES, WARD
from pps_script import EXPECTED_EPISODES
from pps_states import (FORBIDDEN_STATES, STATES, VX_ONSET,
                        ZERO_COMMAND_STATES)

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

# THE PINS.  Every number below was produced by the committed 190 s run.
EXPECTED_SECONDS = 190.0
EXPECTED_CONTROL_STEPS = 9500
# The episodes the duck actually produced, in the order it produced them.
EXPECTED_EPISODE_KINDS = ("intrusion", "intrusion", "intrusion",
                          "ward_approach", "squeeze", "intrusion")
# Who each intrusion cycle was about, in order.
EXPECTED_INTRUSION_PEOPLE = ("dario", "noor", "yara", "liesl")
# The MEASURED bearing of each protective cycle, including the squeeze.
EXPECTED_BEARINGS_DEG = (-2.37, 178.73, 38.98, -145.76, 51.9)
EXPECTED_SQUEEZE = ("kwame", "tomas")
# The SHA-256 of the stock walking policy this behavior is a controller around.
STOCK_POLICY_SHA = \
    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"

# The modules that make DECISIONS.  None of them may reach the scenario.
DECISION_MODULES = ("pps_machine", "pps_threat", "pps_geometry", "pps_control",
                    "pps_states", "pps_metrics", "pps_camera",
                    "pps_visibility")
# The modules that ARE the scenario: where every body is and when.
SCENARIO_MODULES = ("pps_actors", "pps_script")


def imports_of(module: str) -> set[str]:
    """Every module imported anywhere in ``module``, including function-locally."""
    tree = ast.parse((SCRIPTS / f"{module}.py").read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


# -- hygiene: what the duck is never told ------------------------------------
@pytest.mark.parametrize("module", DECISION_MODULES)
def test_no_decision_module_can_read_the_choreography(module):
    """THE STRUCTURAL VERSION OF 'THE DUCK DID NOT KNOW'.

    The scenario declares where every body will be and when.  If a decision
    module could import it, every claim about prediction and dismissal would be
    unfalsifiable.  Checked on the import graph, so it cannot be evaded by
    importing inside a method.
    """
    forbidden = imports_of(module) & set(SCENARIO_MODULES)
    assert not forbidden, (
        f"{module} imports {forbidden}, so the duck can read the choreography")


def test_the_scenario_modules_really_do_hold_the_answers():
    """The check above is only meaningful if there is something to hide."""
    source = (SCRIPTS / "pps_script.py").read_text()
    assert "ENCOUNTERS" in source
    assert "EXPECTED_EPISODES" in source
    assert len(EXPECTED_EPISODES) == 6


@pytest.mark.parametrize("module", ("pps_machine", "pps_control", "pps_threat",
                                    "pps_states"))
def test_the_decision_core_touches_no_physics(module):
    """The machine decides on measured quantities, never on the simulator."""
    assert "mujoco" not in imports_of(module), module


def test_the_state_machine_is_pure():
    """No numpy either: it decides on booleans and names, which is testable."""
    imported = imports_of("pps_machine")
    for banned in ("mujoco", "numpy", "pps_camera", "pps_actors", "pps_script"):
        assert banned not in imported, f"pps_machine imports {banned}"


def test_the_rollout_is_the_only_place_the_two_halves_meet():
    """It is allowed to know both, because it is what runs the experiment."""
    rollout = imports_of("rollout_pps")
    assert "pps_actors" in rollout
    assert "pps_machine" in rollout
    assert "pps_script" not in rollout, (
        "even the rollout reads bodies, not the schedule")


def test_every_module_is_bounded():
    """A 900-line module is where invariants hide.

    ``contact_geometry`` is exempt and that exemption is the point: it is the
    one module whose length is documentation of two measured simulator traps.
    """
    for path in sorted(SCRIPTS.glob("pps_*.py")):
        lines = len(path.read_text().splitlines())
        assert lines <= 340, f"{path.name} is {lines} lines"


def test_the_forbidden_states_are_declared_and_never_implemented():
    """Naming the failures means a run that produced one would fail loudly."""
    assert FORBIDDEN_STATES
    assert set(FORBIDDEN_STATES).isdisjoint(STATES)
    for state in FORBIDDEN_STATES:
        for path in SCRIPTS.glob("*.py"):
            if path.name == "pps_states.py":
                continue
            assert state not in path.read_text(), (state, path.name)


# -- the policy contract ------------------------------------------------------
def test_the_shipped_policy_is_the_stock_walking_policy():
    """Byte-identical, so the behavior is a controller and not a new policy."""
    policy = REPO / "onnx" / "alpha_walking.onnx"
    if not policy.is_file():
        pytest.skip("policy not present")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    assert digest == STOCK_POLICY_SHA


def test_the_observation_contract_is_the_measured_one():
    assert OBS_DIM == 61
    assert ACTION_SCALE == 0.9
    assert GYRO_SENSOR == "imu_ang_vel"
    assert CTRL_HZ == 50.0
    assert NOMINAL_TRUNK_Z == 0.116
    assert FALLEN_TRUNK_Z == 0.09


def test_the_observation_is_assembled_at_the_declared_width():
    observation = build_observation(
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
        np.zeros(14, dtype=np.float32), np.array([0.24, 0.0, 0.1],
                                                 dtype=np.float32))
    assert observation.shape == (OBS_DIM,)
    assert observation.dtype == np.float32
    assert observation[48:51] == pytest.approx([0.24, 0.0, 0.1])
    assert observation[51:].tolist() == [0.0] * 10, "unused slots stay padded"


def test_a_wrong_observation_width_is_refused():
    with pytest.raises(RuntimeError):
        build_observation(np.zeros(3, dtype=np.float32),
                          np.zeros(3, dtype=np.float32),
                          np.zeros(13, dtype=np.float32),
                          np.zeros(14, dtype=np.float32),
                          np.zeros(14, dtype=np.float32),
                          np.zeros(3, dtype=np.float32))


def test_the_gyro_sensor_is_resolved_exactly_and_never_aliased(model):
    """``mj_name2id`` returns -1 for an unknown sensor and the last address is
    a VALID index, so a wrong name silently feeds a different physical quantity
    into the policy's ``base_ang_vel`` slot.  The robot still walks, which is
    exactly what makes it dangerous.
    """
    address = gyro_address(model)
    assert address >= 0
    with pytest.raises(ValueError, match="refusing sensor"):
        gyro_address(model, "angular-velocity")
    with pytest.raises(ValueError, match="refusing sensor"):
        gyro_address(model, "imu_accel")


# -- the compiled scene -------------------------------------------------------
def test_the_scene_compiles_with_its_meshes(model):
    """Zero meshes means ``meshdir`` did not resolve and the robot is a ghost."""
    assert model.nmesh > 0
    assert model.nu == 14
    assert model.nkey >= 2


def test_the_committed_scene_xml_is_the_one_that_loads():
    assert SCENE_XML.is_file()
    text = SCENE_XML.read_text()
    assert '<include file="robot_walk.xml" />' in text
    assert 'name="pps_camera"' in text
    assert 'name="STAND"' in text


def test_every_actor_body_exists_in_the_compiled_model(model):
    import mujoco
    for name in ALL_NAMES:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 f"actor_{name}")
        assert body >= 0, name
        assert model.body_mocapid[body] >= 0, f"{name} must be mocap"


def test_the_people_and_the_scenery_are_all_non_colliding(model):
    """Deliberate: avoidance is the controller's doing, not the solver's.

    A duck that "blocked" somebody by colliding with them would be
    demonstrating the contact solver rather than a protective policy.
    """
    import mujoco
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if not name:
            continue
        if name.startswith(("obs_", "wall_")) or name == "plaza_floor" \
                or any(name.startswith(f"{person}_") for person in ALL_NAMES):
            assert int(model.geom_contype[index]) == 0, name
            assert int(model.geom_conaffinity[index]) == 0, name


# -- the committed run, pinned ------------------------------------------------
def test_the_run_is_the_declared_length(summary):
    assert summary["seconds"] == EXPECTED_SECONDS
    assert summary["control_steps"] == EXPECTED_CONTROL_STEPS
    assert summary["control_hz"] == CTRL_HZ
    assert summary["control_steps"] == int(EXPECTED_SECONDS * CTRL_HZ)


def test_all_twenty_seven_gates_passed(summary):
    assert summary["gates_total"] == 27
    assert summary["gates_passed"] == 27
    assert summary["all_gates_pass"] is True
    failed = [g["gate"] for g in summary["gate_results"] if not g["pass"]]
    assert not failed, failed


def test_every_gate_result_carries_its_evidence(summary):
    for entry in summary["gate_results"]:
        assert entry["evidence"], entry["gate"]
        assert isinstance(entry["pass"], bool)


def test_the_episode_sequence_is_pinned_in_order(summary):
    """Order, not a count: a run that produced them differently must fail."""
    assert tuple(summary["episode_kinds"]) == EXPECTED_EPISODE_KINDS
    assert sorted(summary["episode_kinds"]) == sorted(EXPECTED_EPISODES)


def test_the_intrusion_cycles_are_the_pinned_four_people(summary):
    assert tuple(summary["intrusion_people"]) == EXPECTED_INTRUSION_PEOPLE
    assert summary["intrusion_count"] == 4
    assert len(set(summary["intrusion_people"])) == 4


def test_the_squeeze_is_the_pinned_pair(summary):
    squeeze = summary["squeeze"]
    assert (squeeze["selected"], squeeze["secondary"]) == EXPECTED_SQUEEZE
    assert squeeze["separation_deg"] >= 92.0
    assert squeeze["kind"] == "squeeze"


def test_the_measured_bearings_are_pinned_and_alternate(summary):
    assert summary["intrusion_bearings_deg"] == pytest.approx(
        list(EXPECTED_BEARINGS_DEG), abs=1e-6)
    assert summary["bearings_alternate"] is True
    sides = [1 if math.cos(math.radians(b)) >= 0 else -1
             for b in summary["intrusion_bearings_deg"]]
    assert all(a != b for a, b in zip(sides, sides[1:])), sides


def test_the_squeeze_counts_as_a_protective_cycle(summary):
    """It has a selected primary, a measured bearing and a harder branch.

    Excluding it would be pretending it did not count because the response was
    ESCAPE_GAP rather than INTERPOSE.
    """
    assert summary["protective_cycle_count"] == 5
    assert len(summary["intrusion_bearings_deg"]) == 5
    assert summary["protective_cycle_count"] == summary["intrusion_count"] + 1


def test_the_false_alarm_was_seen_and_never_acted_on(summary):
    assert summary["false_alarm_seen"] is True
    assert summary["false_alarm_dismissed"] is True
    assert "piet" not in summary["intrusion_people"]
    assert all(e["selected"] != "piet" for e in summary["episodes"])


def test_the_run_visited_every_state_the_behavior_claims(summary):
    visited = set(summary["states_visited"])
    assert visited <= set(STATES)
    for required in ("ESCORT", "MONITOR", "PREDICT_INTRUSION", "INTERPOSE",
                     "HOLD_BUFFER", "THREAT_CLEAR", "RETURN_ESCORT",
                     "PERSON_APPROACH", "RETREAT", "MULTI_THREAT",
                     "ESCAPE_GAP", "RECOVER", "DONE"):
        assert required in visited, required
    assert visited.isdisjoint(FORBIDDEN_STATES)


def test_the_transitions_are_ordered_and_internally_consistent(summary):
    transitions = summary["transitions"]
    assert transitions
    times = [t["t"] for t in transitions]
    assert times == sorted(times)
    for earlier, later in zip(transitions, transitions[1:]):
        assert earlier["to"] == later["from"], (earlier, later)
    assert transitions[0]["from"] == "ESCORT"
    assert transitions[-1]["to"] == "DONE"


def test_the_episodes_are_ordered_and_closed(summary):
    for episode in summary["episodes"]:
        assert episode["ended_at_s"] > episode["started_at_s"]
        assert episode["outcome"] in ("recovered", "yielded_to_ward",
                                      "superseded_by_squeeze")
        assert episode["rows"] > 0
    for earlier, later in zip(summary["episodes"], summary["episodes"][1:]):
        assert earlier["index"] + 1 == later["index"]
        assert earlier["ended_at_s"] <= later["started_at_s"]


def test_the_stillness_claims_are_exact_in_the_committed_run(summary):
    assert summary["zero_state_peak"] == 0.0
    assert summary["sub_gait_ticks"] == 0
    assert summary["max_abs_vy"] == 0.0


def test_the_physical_measurements_are_pinned(summary):
    assert summary["contact_steps"] == 0
    assert summary["fallen_steps"] == 0
    assert summary["min_trunk_z_m"] >= FALLEN_TRUNK_Z
    assert 0.105 <= summary["final_trunk_z_m"] <= 0.125
    assert summary["min_person_clearance_m"] > 0.0
    assert summary["min_scenery_clearance_m"] > 0.0
    assert summary["walk_path_m"] > 4.0
    assert summary["walk_path_m"] <= summary["path_m"]


def test_the_protected_person_and_the_policy_are_the_declared_ones(summary):
    assert summary["protected_person"] == WARD == "aina"
    assert summary["policy_sha256"] == STOCK_POLICY_SHA
    assert summary["observation_dim"] == OBS_DIM
    assert summary["action_scale"] == ACTION_SCALE
    assert summary["gyro_sensor"] == GYRO_SENSOR


# -- the per-tick trace, when one is available --------------------------------
def test_every_tick_reports_a_state_the_behavior_declares(trace):
    for record in trace:
        assert record["state"] in STATES


def test_the_hold_states_really_were_exactly_zero_every_tick(trace):
    """The summary reports a peak; this checks the claim tick by tick."""
    for record in trace:
        if record["state"] in ZERO_COMMAND_STATES:
            assert record["command"] == [0.0, 0.0, 0.0], record["t"]


def test_no_tick_ever_commanded_a_lateral_velocity(trace):
    assert max(abs(r["command"][1]) for r in trace) == 0.0


def test_no_tick_ever_commanded_a_sub_gait_forward_crawl(trace):
    """Below onset the robot does not move, so such a command is decorative."""
    for record in trace:
        forward = record["command"][0]
        assert forward <= 0.0 or forward >= VX_ONSET - 1e-6, record["t"]


def test_the_trunk_never_dropped_to_a_fall(trace):
    assert min(r["trunk_z"] for r in trace) >= FALLEN_TRUNK_Z


def test_clearance_stayed_positive_on_every_tick(trace):
    assert min(r["min_person_clearance_m"] for r in trace) > 0.0
    assert min(r["scenery_clearance_m"] for r in trace) > 0.0


def test_the_path_length_is_monotonic(trace):
    values = [r["path_m"] for r in trace]
    assert values == sorted(values)
    assert values[-1] > 4.0


def test_time_advances_by_one_control_tick_each_record(trace):
    dt = 1.0 / CTRL_HZ
    for earlier, later in zip(trace, trace[1:]):
        assert later["t"] - earlier["t"] == pytest.approx(dt, abs=1e-6)


def test_the_between_flag_is_only_set_while_a_person_is_active(trace):
    for record in trace:
        if record["between"]:
            assert record["active"] is not None, record["t"]


def test_the_duck_stayed_inside_the_plaza(trace):
    from pps_plaza import inside_area
    for record in trace:
        assert inside_area(record["duck_xy"], 0.0), record["t"]


# -- real physics -------------------------------------------------------------
@pytest.mark.slow
def test_a_short_rollout_walks_without_falling(short_rollout):
    assert short_rollout.records
    assert short_rollout.falls == 0
    assert short_rollout.min_z >= FALLEN_TRUNK_Z
    assert short_rollout.path_m > 0.0


@pytest.mark.slow
def test_a_short_rollout_reports_the_stock_policy(short_rollout):
    assert short_rollout.policy_sha == STOCK_POLICY_SHA
    assert short_rollout.decimation == 10
    assert short_rollout.dt == pytest.approx(1.0 / CTRL_HZ)


@pytest.mark.slow
def test_a_short_rollout_keeps_positive_clearance(short_rollout):
    assert short_rollout.contacts_count == 0
    assert short_rollout.min_person > 0.0
    assert short_rollout.min_scenery > 0.0


@pytest.mark.slow
def test_a_short_rollout_starts_in_the_escort_and_measures_everybody(
        short_rollout):
    first = short_rollout.records[0]
    assert first["state"] in ("ESCORT", "MONITOR")
    assert set(short_rollout.previous_states) == set(ALL_NAMES)
    assert first["nearest_person"] in ALL_NAMES
