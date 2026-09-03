#!/usr/bin/env python3
"""The facility, the population and the compiled scene.

Pure geometry and pure choreography: no physics stepping, no policy.  These are
the tests that catch a scenario bug where it IS rather than where its
consequence shows up.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from patrol_actors import (
    APPEARANCES,
    ROUTES,
    bodies_at,
    max_heading_step,
    moving_fraction,
    zone_occupancy,
)
from patrol_cast import (
    ALL_NAMES,
    ANOMALY_NAMES,
    BY_NAME,
    EXPECTED_VERDICTS,
    OBJECT_NAMES,
    PERSON_NAMES,
)
from patrol_facility import (
    CHECKPOINTS,
    CHECKPOINT_NAMES,
    CIRCUIT,
    FIXTURES,
    FLOOR_HALF,
    HOME,
    OCCLUDERS,
    RESTRICTED_ZONE,
    occluder_between,
    static_gap,
    stowed_on,
)
from patrol_markers import (
    MEMORY_DISCS,
    ROUTE_DISCS,
    STANDOFF_DISCS,
    TRAIL_DISCS,
)
from patrol_plan import circuit_length_m, corner_turns_deg
from patrol_states import (
    ATTENDED_RADIUS_M,
    DETECT_MAX_RANGE_M,
    DUCK_PLANAR_RADIUS,
)


# -- the circuit -------------------------------------------------------------
def test_five_checkpoints_are_declared_in_a_fixed_order():
    assert len(CHECKPOINTS) == 5
    assert CHECKPOINT_NAMES == ("dock-gate", "east-aisle", "north-bay",
                                "server-door", "west-stair")
    assert len(set(CHECKPOINT_NAMES)) == 5


def test_the_circuit_returns_to_the_guard_post():
    assert CIRCUIT[-1] is HOME
    assert CIRCUIT[:5] == CHECKPOINTS


def test_every_circuit_corner_is_a_left_turn_the_measured_yaw_can_carry():
    """The circuit runs counter-clockwise into the policy's WEAK yaw sign.

    That is deliberate: a clockwise loop would have the policy's own measured
    right bias doing the turning, and "the duck walked its circuit" would be
    partly a fact about the policy rather than about the controller.
    """
    turns = corner_turns_deg()
    assert len(turns) == 5
    assert all(t > 0.0 for t in turns), turns
    assert all(55.0 <= t <= 65.0 for t in turns), turns


def test_the_circuit_clears_every_fixture_by_more_than_the_duck_is_wide():
    worst, where = float("inf"), None
    previous = HOME.position
    for checkpoint in CIRCUIT:
        for index in range(121):
            point = previous + (checkpoint.position - previous) * (index / 120)
            _, gap = static_gap(point)
            if gap < worst:
                worst, where = gap, point
        previous = checkpoint.position
    assert worst > DUCK_PLANAR_RADIUS, (worst, where)


def test_the_circuit_length_matches_its_own_geometry():
    """Six legs of the hexagon's side length, computed two different ways."""
    from patrol_facility import LOOP_RADIUS_M
    assert circuit_length_m() == pytest.approx(6.0 * LOOP_RADIUS_M, abs=1e-3)


# -- the restricted zone ------------------------------------------------------
def test_only_the_intruder_ever_enters_the_restricted_zone():
    occupancy = zone_occupancy(150.0)
    inside = sorted(k for k, v in occupancy.items() if v > 0.0)
    assert inside == ["visitor"], occupancy


def test_the_intruder_ends_well_inside_the_zone_not_clipping_its_edge():
    position = bodies_at(150.0)["visitor"].pos
    assert RESTRICTED_ZONE.contains(position)
    assert RESTRICTED_ZONE.depth_inside(position) >= 0.10


def test_the_zone_posts_are_generated_from_the_zone_itself():
    """The stanchions a viewer sees and the rectangle the detector tests cannot
    drift apart, because the posts are built FROM the corners."""
    posts = {f.name: f.center for f in FIXTURES
             if f.name.startswith("obs_zone_post_")}
    assert len(posts) == 4
    assert set(posts.values()) == set(RESTRICTED_ZONE.corners())


def test_depth_inside_is_signed_and_agrees_with_contains():
    assert RESTRICTED_ZONE.depth_inside(RESTRICTED_ZONE.center) > 0.0
    outside = (RESTRICTED_ZONE.center[0] + 5.0, RESTRICTED_ZONE.center[1])
    assert RESTRICTED_ZONE.depth_inside(outside) < 0.0
    assert not RESTRICTED_ZONE.contains(outside)


# -- the population -----------------------------------------------------------
def test_the_facility_is_populated_by_people_and_objects():
    assert len(ALL_NAMES) == 6
    assert len(PERSON_NAMES) == 4
    assert len(OBJECT_NAMES) == 2
    assert set(ANOMALY_NAMES) == set(EXPECTED_VERDICTS)


def test_three_bodies_move_for_a_real_fraction_of_the_run():
    fractions = moving_fraction(150.0)
    moving = [n for n, f in fractions.items() if f > 0.02]
    assert len(moving) >= 3, fractions


def test_objects_never_move_which_is_what_makes_stationary_time_meaningful():
    fractions = moving_fraction(150.0)
    for name in OBJECT_NAMES:
        assert fractions[name] == 0.0, (name, fractions[name])


def test_every_scripted_person_has_a_continuous_heading():
    worst, name, when = max_heading_step(150.0)
    assert worst <= 6.0, (worst, name, when)


def test_no_body_stands_inside_a_fixture_except_an_object_on_its_stow_area():
    """The trolley stands ON the stow pallet - that is the rule that makes it
    benign - so the exemption is narrow and uses the SAME predicate the duck's
    classifier uses."""
    worst = (float("inf"), "", "", 0.0)
    for step in range(0, 1501):
        t = step * 0.1
        for name, state in bodies_at(t).items():
            if not state.present:
                continue
            if not BY_NAME[name].is_person and stowed_on(state.pos):
                continue
            fixture, gap = static_gap(state.pos)
            if gap < worst[0]:
                worst = (gap, name, fixture, t)
    assert worst[0] > 0.0, worst


def test_a_body_that_has_not_appeared_is_reported_absent():
    """An object before its appearance time must not be visible to anything."""
    for name, entry in APPEARANCES.items():
        before = bodies_at(max(entry["at_s"] - 1.0, 0.0))[name]
        after = bodies_at(entry["at_s"] + 1.0)[name]
        assert not before.present, name
        assert after.present, name


# -- the three cases the duck has to tell apart --------------------------------
def test_the_trolley_satisfies_both_benign_rules():
    """Two independent rules, so the dismissal is robust rather than a coin flip."""
    states = bodies_at(150.0)
    trolley = states["trolley"].pos
    assert stowed_on(trolley), "the trolley must stand on a stow area"
    gap = float(np.linalg.norm(states["emil"].pos - trolley))
    assert gap <= ATTENDED_RADIUS_M, gap


def test_nobody_comes_near_the_crate_while_the_patrol_is_running():
    """The suspicious call rests on 'nobody is with it', which is a MEASUREMENT
    at the time - so the scenario has to keep it true."""
    worst = (float("inf"), "", 0.0)
    for step in range(0, 1101):
        t = step * 0.1
        states = bodies_at(t)
        if not states["crate"].present:
            continue
        for person in PERSON_NAMES:
            gap = float(np.linalg.norm(
                states[person].pos - states["crate"].pos))
            if gap < worst[0]:
                worst = (gap, person, t)
    assert worst[0] > ATTENDED_RADIUS_M, worst


def test_the_crate_is_not_on_any_stow_area():
    assert stowed_on(bodies_at(150.0)["crate"].pos) == ""


def test_the_three_anomalies_are_far_apart():
    states = bodies_at(150.0)
    for a in ANOMALY_NAMES:
        for b in ANOMALY_NAMES:
            if a >= b:
                continue
            gap = float(np.linalg.norm(states[a].pos - states[b].pos))
            assert gap >= 1.5, (a, b, gap)


def test_every_anomaly_is_within_the_camera_gate_of_some_checkpoint():
    states = bodies_at(150.0)
    for name in ANOMALY_NAMES:
        best = min(float(np.linalg.norm(c.position - states[name].pos))
                   for c in CIRCUIT)
        assert best <= DETECT_MAX_RANGE_M, (name, best)


# -- occlusion is real on this scene ------------------------------------------
def test_the_facility_has_real_occluders_unlike_its_siblings():
    """The central rack is 0.72 m against a 0.20 m eye, so the occlusion
    predicate FIRES on this scene rather than being vacuously false."""
    assert len(OCCLUDERS) >= 3
    across = occluder_between((-1.2, 0.0), (1.2, 0.0))
    assert across == "obs_rack_core", across


def test_a_clear_sightline_reports_no_occluder():
    assert occluder_between((0.0, -1.8), (0.4, -1.9)) is None


# -- the compiled scene --------------------------------------------------------
def test_the_scene_compiles_with_meshes_and_the_stock_actuators(model):
    assert model.nmesh > 0
    assert model.nu == 14


def test_the_duck_planar_radius_matches_the_built_model(model, data):
    """The constant the clearance gates use is pinned to the real geometry."""
    from contact_geometry import duck_planar_radius
    trunk = model.body("trunk_base").id
    assert duck_planar_radius(model, data, trunk) == pytest.approx(
        DUCK_PLANAR_RADIUS, abs=5e-4)


def test_the_scene_carries_every_marker(model):
    import mujoco
    for prefix, count in (("route_", ROUTE_DISCS), ("trail_", TRAIL_DISCS),
                          ("memory_", MEMORY_DISCS),
                          ("standoff_", STANDOFF_DISCS)):
        for index in range(count):
            name = f"{prefix}{index}"
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, name
    for name in ("target_marker", "checkpoint_marker", "patrol_rig"):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, name


def test_every_fixture_has_a_geom_the_clearance_gate_can_find(model):
    import mujoco
    for fixture in FIXTURES:
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, fixture.name) >= 0, fixture.name


def test_the_scenery_probe_collects_every_wall_and_fixture(model):
    from rollout_patrol import scenery_geom_names
    names = scenery_geom_names(model)
    assert len(names) >= len(FIXTURES) + 4
    for fixture in FIXTURES:
        assert fixture.name in names


def test_the_checkpoints_and_the_zone_are_painted_on_the_floor(model):
    """A viewer must be able to SEE the duck stop on a checkpoint and see that
    it never crossed the marked rectangle."""
    import mujoco
    for index in range(len(CHECKPOINTS)):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"cp_pad_{index}") >= 0
    for name in ("home_pad", "zone_floor", "zone_tape_n", "zone_tape_s",
                 "zone_tape_e", "zone_tape_w"):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0, name


def test_every_body_is_non_colliding_scenery(model):
    """Nothing but the robot may touch the robot, so every clearance claim is a
    property of the SOFTWARE rather than of the contact solver."""
    for name in ALL_NAMES:
        body = model.body(f"actor_{name}")
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) == body.id:
                assert int(model.geom_contype[geom]) == 0
                assert int(model.geom_conaffinity[geom]) == 0


def test_the_floor_contains_every_checkpoint_and_fixture():
    for checkpoint in CIRCUIT:
        assert abs(checkpoint.xy[0]) < FLOOR_HALF[0]
        assert abs(checkpoint.xy[1]) < FLOOR_HALF[1]
    for fixture in FIXTURES:
        assert abs(fixture.center[0]) < FLOOR_HALF[0]
        assert abs(fixture.center[1]) < FLOOR_HALF[1]


def test_routes_are_filleted_with_real_bends():
    for name, route in ROUTES.items():
        if name == "visitor":
            continue
        bends = route.corner_report()
        assert bends, name
        assert all(b["radius_m"] > 0.0 for b in bends), name
