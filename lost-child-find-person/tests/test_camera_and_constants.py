#!/usr/bin/env python3
"""The head camera and the constants it is measured against, on the real model.

Loads the actual MuJoCo scene and the actual ONNX policy, so these are slower
than the pure-logic suite and they catch a different class of defect: gaze
leaking into the walking state, and per-sample readability drifting away from
the geometry it claims to describe.

THE ISOLATION CLAIM IS ASSERTED BIT-FOR-BIT
--------------------------------------------
The head is a large fraction of this robot's mass and the stock walking policy
was never trained to compensate an imposed head trajectory, so the camera poses
the head only in its own ``MjData`` copy.  ``test_render_only_camera_work_leaves
_the_physical_qpos_bit_identical`` asserts that with ``np.array_equal`` over
qpos, qvel and ctrl across many updates — exact equality, not a tolerance,
because a tolerance would hide exactly the leak it is meant to catch.

The scene, sensor and policy invariants live in ``test_scene_and_policy``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from contact_geometry import ContactProbe, WallProbe, exact_planar_radius  # noqa: E402
from lost_camera import PIP_H, PIP_W, LostCamera  # noqa: E402
from lost_cast import ALL_NAMES, FEATURE_SAMPLES, GUARDIAN  # noqa: E402
from lost_constants import (  # noqa: E402
    SCAN_AMPLITUDE_DEG,
    SCAN_PITCH_DEG,
    SCAN_RATE_DPS,
)
from lost_geometry import (  # noqa: E402
    ADULT_HALF_EXTENT_M,
    DUCK_START_XY,
    DUCK_START_YAW_DEG,
)
from lost_people import people_at, pose_people  # noqa: E402
from policy_runtime import CTRL_HZ, PolicyRunner, load_scene, wrap_angle  # noqa: E402
from rollout_lost import scenery_geom_names  # noqa: E402

POLICY = ROOT / "onnx" / "alpha_walking.onnx"


@pytest.fixture(scope="module")
def model():
    return load_scene()


# --------------------------------------------------------------- the camera
def _camera(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1] = DUCK_START_XY
    half = math.radians(DUCK_START_YAW_DEG) * 0.5
    data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
    pose_people(model, data, people_at(0.0), 0.0)
    mujoco.mj_forward(model, data)
    runner = PolicyRunner(POLICY).reset(model, data)
    camera = LostCamera(model, data, runner.qpos_idx,
                        model.body("trunk_base").id, (PIP_W, PIP_H), CTRL_HZ)
    return camera, data, runner


def test_render_only_camera_work_leaves_the_physical_qpos_bit_identical(model):
    """THE ISOLATION CLAIM.  Exact equality, not a tolerance."""
    camera, data, runner = _camera(model)
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    for step in range(150):
        camera.update(data, duck_yaw=0.2 * math.sin(step * 0.05),
                      subject=GUARDIAN.name if step % 3 else None,
                      scanning=bool(step % 2))
    assert np.array_equal(data.qpos, before[0])
    assert np.array_equal(data.qvel, before[1])
    assert np.array_equal(data.ctrl, before[2])


def test_isolation_is_not_achieved_by_the_head_doing_nothing(model):
    """A camera that never moves would pass the isolation test vacuously."""
    camera, data, _ = _camera(model)
    seen = [camera.update(data, duck_yaw=0.0, subject=None,
                          scanning=True)["gaze_yaw"] for _ in range(200)]
    assert math.degrees(max(seen) - min(seen)) > 90.0


def test_the_sweep_reverses_at_its_measured_amplitude_rather_than_sticking(model):
    """Driving the joint into its stop would stop it being a sweep."""
    camera, data, _ = _camera(model)
    for _ in range(600):
        camera.update(data, duck_yaw=0.0, subject=None, scanning=True)
    assert camera.scan_reversals >= 2
    assert abs(camera._scan_offset) <= math.radians(SCAN_AMPLITUDE_DEG) + 1e-9


def test_the_sweep_uses_only_part_of_the_measured_joint_range(model):
    """The joint spans +/-170 deg; the sweep stays well inside it."""
    assert SCAN_AMPLITUDE_DEG < 170.0
    assert SCAN_RATE_DPS > 0.0
    assert SCAN_PITCH_DEG != 0.0
    camera, data, _ = _camera(model)
    yaw_joint = camera.head_yaw_joint
    low, high = model.jnt_range[yaw_joint]
    assert math.radians(SCAN_AMPLITUDE_DEG) < max(abs(low), abs(high))


def test_the_gaze_joint_never_leaves_its_range_in_the_render_copy(model):
    camera, data, _ = _camera(model)
    low, high = model.jnt_range[camera.head_yaw_joint]
    for step in range(300):
        camera.update(data, duck_yaw=0.3 * math.sin(step * 0.07), subject=None,
                      scanning=True)
        assert low <= camera.gaze_yaw <= high


def test_visibility_is_measured_through_the_camera_the_pip_renders_from(model):
    """The reported percentages and the picture must agree."""
    camera, data, _ = _camera(model)
    assert camera.camera_id == model.camera("lost_camera").id
    state = camera.update(data, duck_yaw=0.0, subject=None, scanning=False)
    assert set(state["people"]) == set(ALL_NAMES)


def test_the_camera_reports_five_samples_per_person(model):
    camera, data, _ = _camera(model)
    state = camera.update(data, duck_yaw=0.0, subject=None, scanning=True)
    for name in ALL_NAMES:
        entry = state["people"][name]
        assert len(entry["samples"]) == 5
        assert entry["sample_count"] == sum(entry["samples"])
        assert 0.0 <= entry["fraction"] <= 1.0


def test_a_person_directly_behind_the_duck_is_not_reported_visible(model):
    """The frustum test is real geometry, not a proximity check."""
    camera, data, _ = _camera(model)
    guardian = camera._person_origin(GUARDIAN.name)
    eye = camera.render_data.cam_xpos[camera.head_cam]
    away = math.atan2(float(guardian[1] - eye[1]),
                      float(guardian[0] - eye[0])) + math.pi
    camera.view_yaw = away
    camera._pose_head(data, 0.0)
    camera._orient_rig()
    seen, _, _ = camera._visible_samples(GUARDIAN.name)
    assert not any(seen)


def test_an_invisible_person_yields_no_readable_features(model):
    camera, data, _ = _camera(model)
    assert camera.readable_features([False] * 5) == set()
    assert camera.observe(GUARDIAN.name, [False] * 5) == {}


def test_a_fully_visible_person_yields_the_complete_descriptor(model):
    camera, data, _ = _camera(model)
    readable = camera.readable_features([True] * 5)
    assert readable == set(FEATURE_SAMPLES)
    observed = camera.observe(GUARDIAN.name, [True] * 5)
    assert set(observed) == set(FEATURE_SAMPLES)
    assert observed["stature"] == pytest.approx(GUARDIAN.height_m)


def test_a_torso_only_view_can_never_read_height_or_headwear(model):
    """The camera decides WHICH features are readable; that is the whole design."""
    camera, data, _ = _camera(model)
    torso = [False, True, True, False, False]
    assert camera.readable_features(torso) == {"shirt", "satchel"}
    observed = camera.observe("mira", torso)
    assert "cap" not in observed and "stature" not in observed


def test_seeing_the_head_but_not_the_knees_still_cannot_read_stature(model):
    camera, data, _ = _camera(model)
    assert "stature" not in camera.readable_features(
        [False, True, True, True, True])


def test_the_off_axis_angle_is_reported_only_over_samples_actually_seen(model):
    """Otherwise a gate could open on somebody standing behind a column."""
    camera, data, _ = _camera(model)
    state = camera.update(data, duck_yaw=0.0, subject=None, scanning=True)
    for name, entry in state["people"].items():
        if not entry["visible"]:
            assert entry["off_axis_deg"] == pytest.approx(180.0), name


def test_the_pip_aspect_sets_the_horizontal_field_of_view(model):
    """The no-render gate and the final render must measure the same frustum."""
    camera, _, _ = _camera(model)
    assert camera.tan_h == pytest.approx((PIP_W / PIP_H) * camera.tan_v)
    assert camera.half_h_deg > camera.half_v_deg


def test_wrap_angle_is_a_half_open_interval():
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi, abs=1e-9)
    for angle in (-7.0, -1.0, 0.5, 4.0, 9.0):
        assert -math.pi <= wrap_angle(angle) < math.pi


# ----------------------------------------------------------- the constants
def test_the_adult_half_extent_constant_is_a_legacy_nominal(model):
    """PINNED AS A LEGACY NOMINAL - it is NOT a measurement of this scene.

    The people swing their arms, so the exact planar half-extent varies over a
    gait cycle: MEASURED over 300 poses on this scene, min 0.1375 (arms down,
    t=0), mean 0.1945, max 0.2629 (mid-stride).  ``ADULT_HALF_EXTENT_M`` is
    0.1647, which is neither the minimum, the mean, nor the maximum, and the
    rollout's own reported ``adult_half_extent_m`` is 0.1375 because it measures
    once at t=0.

    The constant was previously DESCRIBED as the measured full-gait maximum,
    pinned against the scene and produced by ``tools/measure_scene.py``.  All
    three claims were false: the number matches no measurement, and that tool
    does not exist in this project.  The value is retained unchanged so the
    documented standoff prose does not silently move, and this test now pins
    what it actually is rather than restating the old claim.

    It is safe to retain because it is used ONLY by :func:`surface_gap`, which
    is explanatory prose about the standoff band; NO acceptance gate is computed
    from it.  The gates measure real surface clearance every tick through
    ``ContactProbe``, which uses the actual geoms at the actual pose and is
    therefore unaffected by this constant entirely.
    """
    # The constant is a nominal, and is explicitly NOT the gait maximum.
    assert ADULT_HALF_EXTENT_M == pytest.approx(0.1647)

    probe = mujoco.MjData(model)
    pose_people(model, probe, people_at(0.0), 0.0)
    mujoco.mj_forward(model, probe)
    at_rest = exact_planar_radius(
        model, probe, model.body(f"person_{GUARDIAN.name}").id)
    assert at_rest == pytest.approx(0.1375, abs=1e-3)

    widest = 0.0
    for step in range(300):
        t = step * 0.02
        pose_people(model, probe, people_at(t), t)
        mujoco.mj_forward(model, probe)
        widest = max(widest, exact_planar_radius(
            model, probe, model.body(f"person_{GUARDIAN.name}").id))
    assert widest == pytest.approx(0.2629, abs=2e-3)
    assert at_rest < widest

    # The point of the whole note: the constant sits BETWEEN the two, so it
    # describes neither, and the rollout reports the pose-zero value instead.
    assert at_rest < ADULT_HALF_EXTENT_M < widest


def test_no_acceptance_gate_consumes_the_legacy_half_extent():
    """The safety claim must not rest on the legacy nominal.

    ``ADULT_HALF_EXTENT_M`` reaches the gates only if something other than the
    explanatory :func:`surface_gap` reads it or ``CONTACT_SEPARATION_M``.  This
    scans the shipped source so a future edit that wires the nominal into a
    controller, planner or gate fails here instead of silently weakening the
    clearance argument.
    """
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    offenders = []
    for path in sorted(scripts.glob("*.py")):
        for number, line in enumerate(
                path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if "ADULT_HALF_EXTENT_M" not in code and \
                    "CONTACT_SEPARATION_M" not in code:
                continue
            # The definitions themselves, and surface_gap's own body, are the
            # only permitted readers.
            if path.name in ("lost_geometry.py", "lost_cast.py"):
                continue
            offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, (
        "the legacy nominal half-extent leaked into behavior code: "
        + "; ".join(offenders))


def test_the_measured_clearances_stay_positive_against_the_real_geometry(model):
    """The gate's own probe, exercised on the duck's start pose."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0], data.qpos[1] = DUCK_START_XY
    pose_people(model, data, people_at(0.0), 0.0)
    mujoco.mj_forward(model, data)

    walls = WallProbe(model, model.body("trunk_base").id,
                      scenery_geom_names(model))
    gap, geom = walls.distance(data)
    assert gap > 0.0, f"start pose overlaps {geom}"

    contacts = ContactProbe(model, model.body("trunk_base").id, ALL_NAMES)
    for name in ALL_NAMES:
        assert contacts.distance(data, name) > 0.0, name


# ------------------------------------------------------------- integration
@pytest.mark.slow
def test_a_short_real_rollout_stays_upright_and_contact_free():
    """Real physics, real policy, real clearance measurement."""
    from rollout_lost import LostRollout

    rollout = LostRollout(POLICY, 4.0)
    rollout.run()
    assert len(rollout.records) == 200
    assert rollout.min_trunk_z >= 0.09
    assert rollout.min_person_clearance > 0.0
    assert rollout.min_scenery_clearance > 0.0
    assert rollout.fallen_steps == 0
    assert rollout.contact_steps == 0
    for record in rollout.records:
        assert record["guardian"] == GUARDIAN.name
        if record["state"] in ("LOST", "STOP", "SEARCH_SWEEP"):
            assert record["command_peak"] == 0.0
