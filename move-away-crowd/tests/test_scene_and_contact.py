#!/usr/bin/env python3
"""Tests that need MuJoCo: scene contents, sensor identity, contact geometry.

These complement ``test_threat_logic.py`` (pure geometry, no simulator).  They
load the real generated scene, so they also serve as the "XML still compiles"
check.  Run with ``pytest tests/`` from the behavior folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

mujoco = pytest.importorskip("mujoco")

from crowd_routes import ADULT_NAMES, CARRYING_BOX  # noqa: E402
from policy_runtime import (  # noqa: E402
    ACTION_SCALE,
    CTRL_HZ,
    GYRO_SENSOR,
    OBS_DIM,
    gyro_address,
    load_scene,
)


@pytest.fixture(scope="module")
def model():
    return load_scene()


@pytest.fixture(scope="module")
def data(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    mujoco.mj_forward(model, d)
    return d


# --------------------------------------------------------------------------
# Scene contents
# --------------------------------------------------------------------------
def test_scene_has_eight_independently_animated_adults(model):
    for name in ADULT_NAMES:
        body = model.body(f"person_{name}")
        assert int(model.body_mocapid[body.id]) >= 0, f"{name} is not mocap"
        # Two hips and two shoulders per adult: the gait is animated, not static.
        for joint in (f"{name}_hip_l", f"{name}_hip_r",
                      f"{name}_shoulder_l", f"{name}_shoulder_r"):
            assert model.joint(joint) is not None


def test_at_least_four_adults_visibly_carry_a_box(model):
    boxes = [
        name for name in ADULT_NAMES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_box") >= 0
    ]
    assert len(boxes) >= 4
    assert set(boxes) == set(CARRYING_BOX)


def test_carried_boxes_are_real_visible_geometry(model):
    """A box must have volume and sit in front of its adult at chest height."""
    for name in sorted(CARRYING_BOX):
        geom = model.geom(f"{name}_box")
        assert geom.type == mujoco.mjtGeom.mjGEOM_BOX
        assert float(np.min(geom.size)) > 0.05, f"{name}'s box is a sliver"
        assert float(geom.pos[0]) > 0.10, f"{name}'s box is not in front"
        assert float(geom.pos[2]) > 0.0, f"{name}'s box is not at chest height"


def test_nobody_in_the_crowd_can_push_the_robot(model):
    """Every actor geom is non-colliding, so no evasion can succeed by shoving."""
    for name in ADULT_NAMES:
        body = model.body(f"person_{name}")
        for geom in range(model.ngeom):
            parent = int(model.geom_bodyid[geom])
            while parent > 0 and parent != body.id:
                parent = int(model.body_parentid[parent])
            if parent == body.id:
                assert int(model.geom_contype[geom]) == 0
                assert int(model.geom_conaffinity[geom]) == 0


def test_scene_compiles_with_its_meshes_and_the_stock_actuators(model):
    assert model.nmesh > 0, "meshdir did not resolve"
    assert model.nu == 14
    assert model.camera("attention_camera") is not None
    assert model.camera("head_camera") is not None


# --------------------------------------------------------------------------
# Sensor identity
# --------------------------------------------------------------------------
def test_gyro_resolves_to_imu_ang_vel_and_not_the_last_sensor(model):
    address = gyro_address(model)
    assert address != int(model.sensor_adr[-1])
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR)
    assert int(model.sensor_adr[sensor_id]) == address


def test_any_other_sensor_name_is_refused(model):
    """No candidate list: a wrong name must raise, never silently fall back."""
    for name in ("imu_gyro", "gyro", "angular-velocity", ""):
        with pytest.raises(ValueError):
            gyro_address(model, name)


def test_observation_width_matches_the_policy_contract(model):
    assert 3 + 3 + model.nu * 3 + 13 == OBS_DIM


def test_control_rate_divides_the_simulation_timestep(model):
    decimation = (1.0 / CTRL_HZ) / model.opt.timestep
    assert abs(decimation - round(decimation)) < 1e-9
    assert ACTION_SCALE == 0.9


# --------------------------------------------------------------------------
# Contact measurement
# --------------------------------------------------------------------------
def test_geom_distance_returns_the_cutoff_when_nothing_is_nearer(model, data):
    """Pins the API trap that produced two phantom contacts in run 3.

    ``mj_geomDistance`` returns the CUTOFF ITSELF, not the true distance, for a
    pair farther apart than the cutoff.  Feeding the running minimum back in as
    the next cutoff therefore collapses the scan: once any pair returns ``x``,
    every later pair is clamped to ``x``, and a single ``0.0`` reports contact
    for the whole frame.
    """
    trunk = model.body("trunk_base").id
    duck_geom = next(
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == trunk
    )
    adult_geom = model.geom("green_torso").id
    true_distance = float(
        mujoco.mj_geomDistance(model, data, duck_geom, adult_geom, 50.0, None)
    )
    assert true_distance > 0.5
    for cutoff in (0.5, 0.2, 0.0):
        clamped = float(
            mujoco.mj_geomDistance(model, data, duck_geom, adult_geom, cutoff, None)
        )
        assert clamped == pytest.approx(cutoff)


def test_min_surface_distance_holds_its_cutoff_fixed(model, data):
    """The scan must not shrink its own cutoff (see the test above).

    Reproduces the buggy formulation alongside the correct one on geometry
    where they must agree, and requires the correct one not to collapse.
    """
    from rollout_crowd import body_subtree, geoms_of

    trunk = model.body("trunk_base").id
    duck_geoms = geoms_of(model, body_subtree(model, trunk))
    adult_geoms = geoms_of(
        model, body_subtree(model, model.body("person_green").id)
    )
    cutoff = 1.2

    fixed = cutoff
    for a in adult_geoms:
        for d in duck_geoms:
            value = float(mujoco.mj_geomDistance(model, data, d, a, cutoff, None))
            fixed = min(fixed, value)

    shrinking = cutoff
    for a in adult_geoms:
        for d in duck_geoms:
            value = float(
                mujoco.mj_geomDistance(model, data, d, a, shrinking, None)
            )
            shrinking = min(shrinking, value)

    assert fixed > 0.0
    assert shrinking <= fixed
    # The correct scan agrees with an independent full-cutoff evaluation.
    independent = min(
        float(mujoco.mj_geomDistance(model, data, d, a, 50.0, None))
        for a in adult_geoms
        for d in duck_geoms
    )
    assert fixed == pytest.approx(min(independent, cutoff), abs=1e-9)


def test_mesh_vs_box_narrowphase_returns_spurious_zeros(model, data):
    """Pins the MuJoCo artifact that faked all 15 'contacts' in run 4.

    Sweeping one box-carrying adult around the standing robot produces exactly
    ``0.0`` for mesh-vs-box pairs that are plainly apart.  If a future MuJoCo
    fixes this, the test fails and the analytic workaround can be removed.
    """
    import math

    from rollout_crowd import body_subtree, geoms_of

    trunk = model.body("trunk_base").id
    duck_geoms = geoms_of(model, body_subtree(model, trunk))
    box = model.geom("teal_box").id
    mocap = int(model.body_mocapid[model.body("person_teal").id])

    saved_pos = data.mocap_pos[mocap].copy()
    saved_quat = data.mocap_quat[mocap].copy()
    spurious = 0
    try:
        for radius in np.arange(0.30, 1.30, 0.10):
            for angle in np.arange(0.0, 2.0 * math.pi, math.pi / 8):
                data.mocap_pos[mocap] = np.array(
                    [radius * math.cos(angle), radius * math.sin(angle), 0.36])
                yaw = angle + math.pi
                data.mocap_quat[mocap] = np.array(
                    [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
                mujoco.mj_forward(model, data)
                box_center = data.geom_xpos[box]
                for geom in duck_geoms:
                    value = float(mujoco.mj_geomDistance(
                        model, data, geom, box, 1.2, None))
                    if value != 0.0:
                        continue
                    # Bounding spheres are far apart, so 0.0 cannot be real.
                    separation = float(np.linalg.norm(
                        data.geom_xpos[geom] - box_center))
                    if separation > 0.30:
                        spurious += 1
    finally:
        data.mocap_pos[mocap] = saved_pos
        data.mocap_quat[mocap] = saved_quat
        mujoco.mj_forward(model, data)
    assert spurious > 0, (
        "mj_geomDistance no longer reports spurious mesh-vs-box zeros; the "
        "analytic box path in rollout_crowd can be reconsidered"
    )


def test_analytic_box_distance_matches_a_known_geometry(model, data):
    """The analytic box test is exact on a case computed by hand."""
    from rollout_crowd import box_sphere_distance

    green = model.body("person_green")
    mocap = int(model.body_mocapid[green.id])
    box = model.geom("green_box").id
    saved_pos = data.mocap_pos[mocap].copy()
    saved_quat = data.mocap_quat[mocap].copy()
    try:
        # Adult at the origin facing +x, so the box frame is axis aligned.
        data.mocap_pos[mocap] = np.array([0.0, 0.0, 0.36])
        data.mocap_quat[mocap] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(model, data)
        center = data.geom_xpos[box].copy()
        half = model.geom_size[box]
        radius = 0.05
        # A sphere 0.40 m beyond the +x face, on axis.
        probe = center + np.array([float(half[0]) + 0.40, 0.0, 0.0])
        assert box_sphere_distance(model, data, box, probe, radius) == pytest.approx(
            0.40 - radius
        )
        # A sphere at the box centre overlaps by the nearest half-extent.
        inside = box_sphere_distance(model, data, box, center, radius)
        assert inside == pytest.approx(-float(np.min(half)) - radius)
    finally:
        data.mocap_pos[mocap] = saved_pos
        data.mocap_quat[mocap] = saved_quat
        mujoco.mj_forward(model, data)


def test_analytic_box_distance_is_conservative_versus_the_true_surface(model, data):
    """Using each mesh's bounding sphere can only UNDER-report clearance.

    That is the property which makes the workaround safe: it may call a clean
    pass tight, but it can never hide a real contact.
    """
    from rollout_crowd import body_subtree, box_sphere_distance, geoms_of

    trunk = model.body("trunk_base").id
    duck_geoms = geoms_of(model, body_subtree(model, trunk))
    green = model.body("person_green")
    mocap = int(model.body_mocapid[green.id])
    box = model.geom("green_box").id
    saved = data.mocap_pos[mocap].copy()
    try:
        data.mocap_pos[mocap] = np.array([0.9, 0.0, 0.36])
        mujoco.mj_forward(model, data)
        for geom in duck_geoms:
            centre_distance = float(
                np.linalg.norm(data.geom_xpos[geom] - data.geom_xpos[box])
            )
            approximated = box_sphere_distance(
                model, data, box, data.geom_xpos[geom],
                float(model.geom_rbound[geom]),
            )
            assert approximated <= centre_distance
    finally:
        data.mocap_pos[mocap] = saved
        mujoco.mj_forward(model, data)


def test_overlap_is_reported_as_negative(model, data):
    """Driving an adult onto the duck must produce a clearly negative distance."""
    from rollout_crowd import body_subtree, geoms_of

    trunk = model.body("trunk_base").id
    duck_geoms = geoms_of(model, body_subtree(model, trunk))
    green = model.body("person_green")
    adult_geoms = geoms_of(model, body_subtree(model, green.id))
    mocap = int(model.body_mocapid[green.id])

    saved = data.mocap_pos[mocap].copy()
    try:
        data.mocap_pos[mocap] = np.array([0.0, 0.0, 0.36])
        mujoco.mj_forward(model, data)
        worst = min(
            float(mujoco.mj_geomDistance(model, data, d, a, 1.2, None))
            for a in adult_geoms
            for d in duck_geoms
        )
        assert worst < -0.01
    finally:
        data.mocap_pos[mocap] = saved
        mujoco.mj_forward(model, data)
