#!/usr/bin/env python3
"""Contact geometry: exact clearance, and the broad phase that must not hide one.

WHY "ZERO CONTACTS" FROM MUJOCO WOULD BE VACUOUS HERE
-------------------------------------------------------
Both the adults and the plaza walls are non-colliding, so the physics engine
will happily let a person walk through the duck and register nothing.  The
honest gate is GEOMETRIC: the smallest surface-to-surface separation, measured
every control tick and required to stay positive.

That puts the whole weight of the safety claim on this module, and on one
optimisation inside it.  ``ContactProbe`` screens each person with a cheap
bounding-sphere lower bound before running the analytic narrow phase.  A broad
phase that ever over-estimated a distance would SKIP a real contact, so the
tests below check the bound's direction, not just its speed.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from contact_geometry import (WallProbe, ContactProbe, body_subtree,
                              bounding_sphere_distance, box_sphere_distance,
                              capsule_sphere_distance, cylinder_sphere_distance,
                              duck_planar_radius, exact_lateral_half_width,
                              exact_planar_radius, geoms_of,
                              primitive_sphere_distance,
                              sphere_sphere_distance)
from pps_cast import ALL_NAMES
from pps_states import DUCK_PLANAR_RADIUS
from rollout_pps import SCENERY_PREFIXES, scenery_names

# The scenery this behavior grades clearance against.
EXPECTED_SCENERY = ("wall_e", "wall_w", "wall_n", "wall_s", "obs_kiosk_e",
                    "obs_lamp_nw", "obs_planter_s", "obs_bench_w",
                    "obs_bench_ne", "obs_bollard_se", "obs_bollard_sw")


@pytest.fixture(scope="module")
def trunk(model):
    return model.body("trunk_base").id


@pytest.fixture(scope="module")
def contact_probe(model, trunk):
    return ContactProbe(model, trunk, ALL_NAMES, prefix="actor_")


@pytest.fixture(scope="module")
def wall_probe(model, trunk):
    return WallProbe(model, trunk, scenery_names(model))


def geom_named(model, name):
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom >= 0, name
    return geom


# -- the pinned duck radius --------------------------------------------------
def test_the_declared_planar_radius_matches_the_compiled_model(model, data,
                                                               trunk):
    """The constant every station and clearance figure is derived from.

    MEASURED at the STAND pose from each geom's BOUNDING SPHERE, so it
    over-states the robot - which is the safe direction for every gate about
    the duck's own footprint.
    """
    measured = duck_planar_radius(model, data, trunk)
    assert measured == pytest.approx(DUCK_PLANAR_RADIUS, abs=5e-4)
    assert DUCK_PLANAR_RADIUS == 0.1162


def test_the_declared_radius_over_states_the_real_robot(model, data, trunk):
    """Conservative by construction, and measurably so."""
    exact = exact_planar_radius(model, data, trunk)
    assert exact < DUCK_PLANAR_RADIUS
    assert exact_lateral_half_width(model, data, trunk) <= exact + 1e-9


def test_the_robot_subtree_is_found_completely(model, trunk):
    bodies = body_subtree(model, trunk)
    assert trunk in bodies
    assert len(bodies) > 5
    assert len(geoms_of(model, bodies)) > 5


# -- the analytic primitives -------------------------------------------------
def test_box_distance_is_exact_outside_and_signed_inside(model, data):
    """Every wall and bench is a box, so this path carries the scenery gate."""
    geom = geom_named(model, "obs_bench_ne")
    centre = data.geom_xpos[geom].copy()
    half = model.geom_size[geom]
    outside = centre + np.array([float(half[0]) + 0.5, 0.0, 0.0])
    assert box_sphere_distance(model, data, geom, outside, 0.0) == \
        pytest.approx(0.5)
    assert box_sphere_distance(model, data, geom, outside, 0.2) == \
        pytest.approx(0.3)
    assert box_sphere_distance(model, data, geom, centre, 0.0) < 0.0


def test_cylinder_distance_is_exact_radially_and_axially(model, data):
    geom = geom_named(model, "obs_bollard_se")
    centre = data.geom_xpos[geom].copy()
    radius = float(model.geom_size[geom][0])
    half_height = float(model.geom_size[geom][1])
    radial = centre + np.array([radius + 0.4, 0.0, 0.0])
    assert cylinder_sphere_distance(model, data, geom, radial, 0.0) == \
        pytest.approx(0.4)
    axial = centre + np.array([0.0, 0.0, half_height + 0.3])
    assert cylinder_sphere_distance(model, data, geom, axial, 0.0) == \
        pytest.approx(0.3)
    assert cylinder_sphere_distance(model, data, geom, centre, 0.0) < 0.0


def test_capsule_distance_measures_from_the_spine(model, data,
                                                  contact_probe):
    """Every adult's torso and legs are capsules."""
    capsule = next(
        g for g in contact_probe.person_geoms["aina"]
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE))
    centre = data.geom_xpos[capsule].copy()
    radius = float(model.geom_size[capsule][0])
    probe = centre + np.array([radius + 0.25, 0.0, 0.0])
    assert capsule_sphere_distance(model, data, capsule, probe, 0.0) == \
        pytest.approx(0.25)
    assert capsule_sphere_distance(model, data, capsule, probe, 0.1) == \
        pytest.approx(0.15)


def test_sphere_distance_subtracts_both_radii(model, data, contact_probe):
    """Every adult's head is a sphere."""
    head = geom_named(model, "aina_head")
    centre = data.geom_xpos[head].copy()
    radius = float(model.geom_size[head][0])
    probe = centre + np.array([radius + 0.6, 0.0, 0.0])
    assert sphere_sphere_distance(model, data, head, probe, 0.0) == \
        pytest.approx(0.6)
    assert sphere_sphere_distance(model, data, head, probe, 0.25) == \
        pytest.approx(0.35)


def test_the_bounding_sphere_fallback_under_reports_rather_than_over(model,
                                                                     data):
    """It can raise a false alarm; it cannot hide a contact.

    That asymmetry is the whole reason a conservative fallback is acceptable in
    a safety measurement.
    """
    geom = geom_named(model, "obs_bench_ne")
    probe = data.geom_xpos[geom] + np.array([1.5, 0.0, 0.0])
    conservative = bounding_sphere_distance(model, data, geom, probe, 0.0)
    exact = box_sphere_distance(model, data, geom, probe, 0.0)
    assert conservative <= exact + 1e-12


@pytest.mark.parametrize("name", ["obs_bench_ne", "obs_bollard_se",
                                  "aina_head", "aina_torso"])
def test_the_dispatcher_routes_each_shape_to_its_exact_form(model, data,
                                                            name):
    geom = geom_named(model, name)
    probe = data.geom_xpos[geom] + np.array([1.2, 0.0, 0.0])
    routed = primitive_sphere_distance(model, data, geom, probe, 0.05)
    kind = int(model.geom_type[geom])
    exact = {int(mujoco.mjtGeom.mjGEOM_BOX): box_sphere_distance,
             int(mujoco.mjtGeom.mjGEOM_CYLINDER): cylinder_sphere_distance,
             int(mujoco.mjtGeom.mjGEOM_CAPSULE): capsule_sphere_distance,
             int(mujoco.mjtGeom.mjGEOM_SPHERE): sphere_sphere_distance}[kind]
    assert routed == pytest.approx(exact(model, data, geom, probe, 0.05))


# -- the probes refuse the shapes they cannot measure honestly ----------------
def test_the_wall_probe_refuses_a_mesh(model, trunk):
    """``mj_geomDistance`` cannot be trusted for mesh-versus-primitive here.

    The probe is only valid for primitives, so being handed a mesh has to be a
    loud failure rather than a silently wrong number.
    """
    mesh_geom = next(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in range(model.ngeom)
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g))
    with pytest.raises(RuntimeError, match="mesh"):
        WallProbe(model, trunk, (mesh_geom,))


def test_the_wall_probe_refuses_a_geom_that_does_not_exist(model, trunk):
    with pytest.raises(RuntimeError, match="not found"):
        WallProbe(model, trunk, ("wall_that_is_not_there",))


def test_every_person_is_all_primitives(model, trunk):
    """The contact probe asserts it, so an actor gaining a mesh fails loudly."""
    probe = ContactProbe(model, trunk, ALL_NAMES, prefix="actor_")
    for name in ALL_NAMES:
        for geom in probe.person_geoms[name]:
            assert int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH)


# -- the scenery set ---------------------------------------------------------
def test_the_graded_scenery_is_every_wall_and_fixture(model):
    assert set(scenery_names(model)) == set(EXPECTED_SCENERY)
    assert SCENERY_PREFIXES == ("obs_", "wall_")


def test_the_floor_and_markers_are_not_graded_as_scenery(model):
    """Clearance to the floor the duck stands on is not a safety claim.

    Neither are the mocap discs that visualise stations, which the duck walks
    over on purpose.
    """
    graded = set(scenery_names(model))
    assert "plaza_floor" not in graded
    assert "ground_plane" not in graded
    assert not any(n.endswith("_disc") for n in graded)


# -- the broad phase ---------------------------------------------------------
@pytest.mark.parametrize("name", list(ALL_NAMES))
def test_the_broad_phase_never_over_states_a_distance(model, data,
                                                      contact_probe, name):
    """The one direction that would be unsafe, checked per person.

    Bounding-sphere surface distance is never larger than the true primitive
    distance, so a screen that passed could not have skipped a real contact.
    """
    duck_positions = data.geom_xpos[contact_probe.duck_geoms]
    person_positions = data.geom_xpos[contact_probe.person_geoms[name]]
    delta = duck_positions[:, None, :] - person_positions[None, :, :]
    centre = np.linalg.norm(delta, axis=2)
    lower = centre - contact_probe._duck_radii[:, None] \
        - contact_probe._person_radii[name][None, :]

    for duck_index, duck_geom in enumerate(contact_probe.duck_geoms):
        for person_index, person_geom in enumerate(
                contact_probe.person_geoms[name]):
            exact = primitive_sphere_distance(
                model, data, person_geom, data.geom_xpos[duck_geom],
                contact_probe.duck_rbound[duck_geom])
            assert lower[duck_index, person_index] <= exact + 1e-9


def test_the_broad_phase_returns_the_cutoff_when_everybody_is_far(model, data,
                                                                  contact_probe):
    """Parked actors are 3 m below the floor, so every pair is beyond it."""
    for name in ALL_NAMES:
        assert contact_probe.distance(data, name, cutoff=1.5) == 1.5


def test_a_person_brought_close_defeats_the_screen_and_is_measured(model,
                                                                   trunk):
    """The screen has to stop screening once somebody is actually near.

    Otherwise the optimisation would be a way of never noticing a contact.
    """
    private = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, private, model.key("STAND").id)
    mujoco.mj_forward(model, private)
    probe = ContactProbe(model, trunk, ALL_NAMES, prefix="actor_")
    mocap = int(model.body_mocapid[model.body("actor_dario").id])
    duck = private.xpos[trunk].copy()
    private.mocap_pos[mocap] = (float(duck[0]) + 0.35, float(duck[1]), 0.374)
    mujoco.mj_forward(model, private)
    close = probe.distance(private, "dario", cutoff=1.5)
    assert close < 1.5, "the narrow phase ran"
    assert 0.0 < close < 0.5, close


def test_the_screen_and_the_narrow_phase_agree_when_somebody_is_near(model,
                                                                     trunk):
    """The optimisation must not change the answer, only the cost."""
    private = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, private, model.key("STAND").id)
    mujoco.mj_forward(model, private)
    probe = ContactProbe(model, trunk, ALL_NAMES, prefix="actor_")
    mocap = int(model.body_mocapid[model.body("actor_dario").id])
    duck = private.xpos[trunk].copy()
    private.mocap_pos[mocap] = (float(duck[0]) + 0.35, float(duck[1]), 0.374)
    mujoco.mj_forward(model, private)

    exhaustive = min(
        primitive_sphere_distance(model, private, person_geom,
                                  private.geom_xpos[duck_geom],
                                  probe.duck_rbound[duck_geom])
        for person_geom in probe.person_geoms["dario"]
        for duck_geom in probe.duck_geoms)
    assert probe.distance(private, "dario", cutoff=1.5) == pytest.approx(
        exhaustive)


def test_overlapping_bodies_report_a_negative_distance(model, trunk):
    """Negative means real geometric overlap, which is what the gate refuses.

    The people are non-colliding, so MuJoCo reports nothing here; this number
    is the only thing standing between the run and an unnoticed walk-through.
    """
    private = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, private, model.key("STAND").id)
    mujoco.mj_forward(model, private)
    probe = ContactProbe(model, trunk, ALL_NAMES, prefix="actor_")
    mocap = int(model.body_mocapid[model.body("actor_dario").id])
    duck = private.xpos[trunk].copy()
    private.mocap_pos[mocap] = (float(duck[0]), float(duck[1]),
                                float(duck[2]) + 0.104)
    mujoco.mj_forward(model, private)
    assert probe.distance(private, "dario", cutoff=1.5) < 0.0


@pytest.mark.parametrize("cutoff", [0.5, 1.0, 1.5])
def test_the_cutoff_bounds_the_reported_distance(model, data, contact_probe,
                                                 cutoff):
    for name in ALL_NAMES:
        assert contact_probe.distance(data, name, cutoff=cutoff) <= cutoff


def test_the_wall_probe_reports_a_limiting_geom_by_name(model, data,
                                                        wall_probe):
    distance, limiting = wall_probe.distance(data, cutoff=5.0)
    assert distance < 5.0
    assert limiting in EXPECTED_SCENERY
    assert np.isfinite(distance)


def test_the_wall_probe_names_the_wall_it_is_pressed_against(model, trunk):
    private = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, private, model.key("STAND").id)
    mujoco.mj_forward(model, private)
    probe = WallProbe(model, trunk, scenery_names(model))
    private.qpos[0:2] = (0.0, -2.98)
    mujoco.mj_forward(model, private)
    distance, limiting = probe.distance(private, cutoff=1.0)
    assert limiting == "wall_s", limiting
    assert distance < 0.2


def test_the_wall_probe_saturates_at_its_cutoff_in_the_open(model, trunk):
    private = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, private, model.key("STAND").id)
    mujoco.mj_forward(model, private)
    probe = WallProbe(model, trunk, scenery_names(model))
    private.qpos[0:2] = (0.0, 0.0)
    mujoco.mj_forward(model, private)
    distance, limiting = probe.distance(private, cutoff=0.4)
    assert distance == 0.4
    assert limiting == "", "nothing limited it"


def test_the_cutoff_is_fixed_rather_than_fed_back(model, data, wall_probe):
    """The first measured trap: feeding the running minimum back collapses it.

    Once any pair returned ``x`` every later pair would be clamped to ``x``, so
    a single zero would report contact for the whole frame.  A larger cutoff
    must therefore never make the answer smaller.
    """
    tight, _ = wall_probe.distance(data, cutoff=0.3)
    loose, _ = wall_probe.distance(data, cutoff=3.0)
    assert loose >= tight - 1e-12


def test_no_geom_distance_call_survives_in_the_measured_path():
    """The second measured trap: mesh-versus-primitive returns spurious zeros.

    Sibling behaviors measured ``mj_geomDistance`` returning exactly 0.0 for
    pairs more than a metre apart, state-dependently.  This module answers by
    not calling it at all, and that is checked on the source rather than
    promised in a docstring.
    """
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "scripts" /
              "contact_geometry.py").read_text()
    body = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith(("#", "*", '"""')))
    assert "mj_geomDistance(" not in body
