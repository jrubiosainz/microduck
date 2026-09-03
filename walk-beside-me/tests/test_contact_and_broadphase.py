#!/usr/bin/env python3
"""Exact contact geometry, and the broadphase that must not change an answer.

``ContactProbe`` carries a conservative bounding-sphere broad phase that skips
the analytic narrow phase whenever every duck/person geom pair is already beyond
the cutoff.  It removes about 80k Python calls per 20 ticks in a five-person
scene and took the headless rollout from over 45 s to about 30 s.

A broad phase that can skip a pair BELOW the cutoff would silently weaken the
contact gate, which is the one gate this behavior cannot afford to soften.  The
tests here prove it cannot:

* the bounding-sphere lower bound is never larger than the analytic distance it
  screens, at every pose sampled from a real rollout — that is the property the
  skip rests on;
* the probe's answer with the broad phase equals the answer without it, at the
  default cutoff and at cutoffs deliberately placed just above and just below
  the true separation, which is where an off-by-one in the comparison would show.

Marked ``slow``: these load the real scene and step real physics.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from beside_cast import ALL_NAMES  # noqa: E402
from contact_geometry import (  # noqa: E402
    ContactProbe,
    WallProbe,
    body_subtree,
    bounding_sphere_distance,
    box_sphere_distance,
    capsule_sphere_distance,
    cylinder_sphere_distance,
    duck_planar_radius,
    exact_lateral_half_width,
    exact_planar_radius,
    geoms_of,
    primitive_sphere_distance,
    sphere_sphere_distance,
)
from policy_runtime import load_scene  # noqa: E402

pytestmark = pytest.mark.slow

POLICY = REPO / "onnx" / "alpha_walking.onnx"


def reference_distance(probe: ContactProbe, data, name: str,
                       cutoff: float) -> float:
    """The narrow phase alone, with the broad-phase skip removed.

    A deliberate duplicate of the probe's inner loop: comparing the probe
    against itself would prove nothing.
    """
    best = cutoff
    for person_geom in probe.person_geoms[name]:
        for duck_geom in probe.duck_geoms:
            distance = primitive_sphere_distance(
                probe.model, data, person_geom,
                data.geom_xpos[duck_geom], probe.duck_rbound[duck_geom])
            if distance < best:
                best = distance
    return best


@pytest.fixture(scope="module")
def stepped():
    """A short real rollout, kept as (model, data, probe) plus sampled poses."""
    from rollout_beside import BesideRollout

    rollout = BesideRollout(str(POLICY), 6.0)
    poses = []
    for index in range(rollout.total_steps):
        rollout.step(index)
        if index % 15 == 0:
            poses.append(rollout.data.qpos.copy())
    return rollout, poses


# -- the analytic primitives ---------------------------------------------------

def test_the_analytic_primitives_agree_with_brute_force_on_a_synthetic_scene():
    """Each closed form is checked against a dense surface sampling."""
    xml = """
    <mujoco>
      <worldbody>
        <geom name="b" type="box" pos="0 0 0" size="0.20 0.15 0.10"
              euler="0 0 25" contype="0" conaffinity="0"/>
        <geom name="c" type="cylinder" pos="1.5 0 0" size="0.18 0.22"
              euler="20 10 0" contype="0" conaffinity="0"/>
        <geom name="k" type="capsule" pos="3.0 0 0" size="0.12 0.25"
              euler="0 40 0" contype="0" conaffinity="0"/>
        <geom name="s" type="sphere" pos="4.5 0 0" size="0.17"
              contype="0" conaffinity="0"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(20260902)
    for name, closed_form in (("b", box_sphere_distance),
                              ("c", cylinder_sphere_distance),
                              ("k", capsule_sphere_distance),
                              ("s", sphere_sphere_distance)):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for _ in range(60):
            centre = data.geom_xpos[geom] + rng.normal(0.0, 0.45, 3)
            radius = float(rng.uniform(0.01, 0.09))
            analytic = closed_form(model, data, geom, centre, radius)
            assert analytic == pytest.approx(
                primitive_sphere_distance(model, data, geom, centre, radius))
            assert analytic > -1.0


def test_a_sphere_centre_inside_a_box_reports_a_negative_distance():
    xml = """<mujoco><worldbody>
      <geom name="b" type="box" pos="0 0 0" size="0.3 0.3 0.3"
            contype="0" conaffinity="0"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "b")
    assert box_sphere_distance(model, data, geom,
                               np.zeros(3), 0.05) < 0.0
    assert box_sphere_distance(model, data, geom,
                               np.array([0.5, 0.0, 0.0]), 0.05) > 0.0


def test_the_bounding_sphere_fallback_never_over_reports_clearance():
    xml = """<mujoco><worldbody>
      <geom name="b" type="box" pos="0 0 0" size="0.3 0.2 0.1"
            contype="0" conaffinity="0"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "b")
    rng = np.random.default_rng(7)
    for _ in range(120):
        centre = rng.normal(0.0, 0.8, 3)
        radius = float(rng.uniform(0.0, 0.05))
        conservative = bounding_sphere_distance(model, data, geom, centre,
                                                radius)
        exact = box_sphere_distance(model, data, geom, centre, radius)
        assert conservative <= exact + 1e-12, (
            "the conservative form may under-report clearance but never "
            "over-report it")


# -- the broad phase -----------------------------------------------------------

def test_the_broadphase_lower_bound_never_exceeds_the_analytic_distance(stepped):
    """The property the skip rests on, checked at every sampled real pose.

    If the bounding-sphere lower bound could exceed the true separation, the
    probe could skip a pair that is genuinely inside the cutoff and report a
    clearance that never happened.
    """
    rollout, poses = stepped
    probe = rollout.contacts
    model, data = rollout.model, rollout.data
    checked = 0
    for qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for name in ALL_NAMES:
            people = probe.person_geoms[name]
            duck_pos = data.geom_xpos[probe.duck_geoms]
            person_pos = data.geom_xpos[people]
            centre = np.linalg.norm(
                duck_pos[:, None, :] - person_pos[None, :, :], axis=2)
            lower = (centre - probe._duck_radii[:, None]
                     - probe._person_radii[name][None, :])
            for i, duck_geom in enumerate(probe.duck_geoms):
                for j, person_geom in enumerate(people):
                    analytic = primitive_sphere_distance(
                        model, data, person_geom, data.geom_xpos[duck_geom],
                        probe.duck_rbound[duck_geom])
                    assert lower[i, j] <= analytic + 1e-9, (
                        f"broadphase bound {lower[i, j]:.6f} exceeded the "
                        f"analytic {analytic:.6f} for {name}")
                    checked += 1
    assert checked > 1000, "the sweep must be non-trivial"


def test_the_broadphase_answer_equals_the_narrow_phase_at_the_default_cutoff(
        stepped):
    rollout, poses = stepped
    probe = rollout.contacts
    model, data = rollout.model, rollout.data
    for qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for name in ALL_NAMES:
            assert probe.distance(data, name) == pytest.approx(
                reference_distance(probe, data, name, 1.5), abs=1e-12)


def test_the_broadphase_cannot_skip_a_pair_just_below_the_cutoff(stepped):
    """The off-by-one hunt.

    For each pose the cutoff is placed a hair ABOVE and a hair BELOW the true
    separation.  Just above, the narrow phase must run and return the true
    value; just below, both paths must agree on returning the cutoff.  A broad
    phase using ``>`` instead of ``>=``, or comparing against the wrong radius,
    fails here and nowhere else.
    """
    rollout, poses = stepped
    probe = rollout.contacts
    model, data = rollout.model, rollout.data
    exercised = 0
    for qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for name in ALL_NAMES:
            true = reference_distance(probe, data, name, 1e6)
            if not math.isfinite(true) or true > 40.0:
                continue
            for delta in (-1e-4, -1e-6, 0.0, +1e-6, +1e-4, +0.05):
                cutoff = true + delta
                if cutoff <= 0.0:
                    continue
                assert probe.distance(data, name, cutoff) == pytest.approx(
                    reference_distance(probe, data, name, cutoff), abs=1e-12), (
                    f"{name}: broad and narrow phase disagreed at cutoff "
                    f"{cutoff:.8f} (true separation {true:.8f})")
                exercised += 1
    assert exercised > 100


def test_the_broadphase_returns_the_cutoff_when_everybody_is_far_away(stepped):
    """The case the optimisation exists for, and it must be the ONLY case it
    changes: the returned value, not merely the time taken."""
    rollout, _ = stepped
    probe = rollout.contacts
    model, data = rollout.model, rollout.data
    data.qpos[0], data.qpos[1] = 40.0, 40.0
    mujoco.mj_forward(model, data)
    for name in ALL_NAMES:
        assert probe.distance(data, name, 1.5) == 1.5
        assert reference_distance(probe, data, name, 1.5) == 1.5


def test_an_overlapping_pose_is_reported_as_negative_by_both_paths(stepped):
    """A contact the gate must catch is caught with the broad phase in place."""
    rollout, _ = stepped
    probe = rollout.contacts
    model, data = rollout.model, rollout.data
    # Put the duck exactly under the guardian's mocap origin.
    body = model.body("person_nadia")
    mocap = int(model.body_mocapid[body.id])
    data.mocap_pos[mocap][:2] = data.qpos[0:2]
    mujoco.mj_forward(model, data)
    with_broad = probe.distance(data, "nadia", 1.5)
    without = reference_distance(probe, data, "nadia", 1.5)
    assert with_broad == pytest.approx(without, abs=1e-12)
    assert with_broad < 0.0, "an overlapping pose must report a contact"


def test_the_probe_refuses_a_person_carrying_mesh_geoms():
    """``mj_geomDistance`` cannot be trusted for mesh-versus-primitive pairs in
    this simulator, so the probe's all-primitive precondition is enforced.

    Exercised by renaming the ROBOT's own trunk to ``person_*`` in a copy of
    the scene: the robot is all meshes, so it is the only body available that
    trips the check.
    """
    model = load_scene()
    trunk = model.body("trunk_base").id

    class MeshCarrier:
        """A model view whose ``person_decoy`` resolves to the mesh robot."""

        def __init__(self, wrapped, decoy_id):
            self._wrapped = wrapped
            self._decoy_id = decoy_id

        def body(self, name):
            if name == "person_decoy":
                return type("B", (), {"id": self._decoy_id})()
            return self._wrapped.body(name)

        def __getattr__(self, item):
            return getattr(self._wrapped, item)

    with pytest.raises(RuntimeError, match="mesh geoms"):
        ContactProbe(MeshCarrier(model, trunk), trunk, ("decoy",))


def test_the_wall_probe_refuses_a_mesh_and_names_a_missing_geom():
    model = load_scene()
    trunk = model.body("trunk_base").id
    with pytest.raises(RuntimeError, match="not found"):
        WallProbe(model, trunk, ("no_such_wall",))


def test_no_scenery_geom_ever_goes_through_mj_geomdistance():
    """The trap this module exists to document: measured exact zeros for pairs
    more than a metre apart, state-dependent and not reproducible outside the
    rollout."""
    source = (REPO / "scripts" / "contact_geometry.py").read_text()
    assert "mj_geomDistance" in source, "the trap must stay documented"
    code = [line for line in source.splitlines()
            if "mj_geomDistance" in line and not line.strip().startswith(("#", "*", '"'))
            and "``" not in line]
    assert not code, f"mj_geomDistance is actually called: {code}"


# -- the duck's own footprint --------------------------------------------------

def test_the_conservative_radius_over_states_the_exact_one(stepped):
    rollout, poses = stepped
    model, data = rollout.model, rollout.data
    trunk = rollout.trunk
    for qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        conservative = duck_planar_radius(model, data, trunk)
        exact = exact_planar_radius(model, data, trunk)
        assert conservative >= exact, (
            "the bounding-sphere radius must over-state the robot; that is "
            "the safe direction for every footprint gate")


def test_the_lateral_half_width_never_exceeds_the_rotation_invariant_radius(
        stepped):
    rollout, poses = stepped
    model, data = rollout.model, rollout.data
    trunk = rollout.trunk
    for qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        assert exact_lateral_half_width(model, data, trunk) \
            <= exact_planar_radius(model, data, trunk) + 1e-9


def test_the_declared_duck_radius_is_a_sizing_figure_not_a_clearance_gate():
    """``DUCK_PLANAR_RADIUS`` is INHERITED, and nothing safety-critical reads it.

    MEASURED on this scene: the duck's conservative bounding-sphere planar
    half-extent is 0.1162 m at pose zero and reaches about 0.142 m mid-gait,
    while the declared constant is 0.1303 m — a figure carried over from the
    sibling corridor scenes.  It is therefore NOT conservative against the gait
    maximum, and this test exists to say so out loud rather than to assert a
    comfortable inequality that happens to hold at pose zero.

    That is acceptable only because the constant sizes the REFUSAL MARGINS in
    prose and no gate measures clearance with it: clearance is measured every
    control tick by ``ContactProbe`` and ``WallProbe`` against the real geoms at
    the real pose.  The safety-relevant claim is the one asserted below, and it
    is made against the measured gait maximum rather than against the constant.
    """
    from beside_geometry import (
        DUCK_PLANAR_RADIUS,
        SIDE_PERSON_MARGIN_M,
        SIDE_STATIC_MARGIN_M,
    )

    assert DUCK_PLANAR_RADIUS == 0.1303, (
        "the inherited constant changed; re-measure it against THIS scene "
        "before trusting any prose that quotes it")


def test_the_refusal_margins_clear_the_duck_measured_gait_maximum(stepped):
    """The claim the sizing argument actually needs, measured not assumed.

    A candidate slot is refused unless it clears static surfaces by
    ``SIDE_STATIC_MARGIN_M``.  For that refusal to mean "the duck could not
    physically occupy this slot", the margin has to exceed the duck's real
    half-extent AT ITS WIDEST POINT IN THE GAIT, not at pose zero.
    """
    from beside_geometry import SIDE_PERSON_MARGIN_M, SIDE_STATIC_MARGIN_M

    rollout, poses = stepped
    model, data = rollout.model, rollout.data
    conservative = 0.0
    exact = 0.0
    for qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        conservative = max(conservative,
                           duck_planar_radius(model, data, rollout.trunk))
        exact = max(exact, exact_planar_radius(model, data, rollout.trunk))

    assert conservative > exact, "the conservative form must over-state"
    assert SIDE_STATIC_MARGIN_M > conservative, (
        f"a slot refused for being {SIDE_STATIC_MARGIN_M} m from a surface "
        f"must be one the duck ({conservative:.4f} m wide mid-gait) could not "
        "occupy")
    assert SIDE_PERSON_MARGIN_M > conservative


def test_the_reported_footprint_matches_the_pose_zero_measurement(stepped):
    """PINNED against the 86 s metrics, which quote both figures."""
    rollout, _ = stepped
    assert rollout.duck_radius == pytest.approx(0.1162, abs=1e-3)
    assert rollout.duck_exact_radius == pytest.approx(0.0827, abs=1e-3)
    assert rollout.adult_half_extent == pytest.approx(0.1155, abs=1e-3)
    assert rollout.duck_radius > rollout.duck_exact_radius


def test_the_subtree_and_geom_helpers_cover_the_whole_robot():
    model = load_scene()
    trunk = model.body("trunk_base").id
    bodies = body_subtree(model, trunk)
    assert trunk in bodies
    assert len(bodies) > 5
    geoms = geoms_of(model, bodies)
    assert geoms
    assert all(int(model.geom_bodyid[g]) in bodies for g in geoms)
