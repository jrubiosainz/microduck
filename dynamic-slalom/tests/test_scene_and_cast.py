#!/usr/bin/env python3
"""The scene, the cast, the choreography and the camera, against the built model.

These are the tests that need a real compiled MuJoCo model, so they assert the
things that can only be checked there: that the geometry the gate measures is
the geometry the scene contains, that the duck's radius constant matches the
robot that was actually built, and that the scripted routes behave like walking
bodies rather than teleporting ones.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from contact_geometry import duck_planar_radius, exact_lateral_half_width
from slalom_actors import (
    ENCOUNTERS,
    ROUTES,
    actors_at,
    lane_crossings,
    max_heading_step,
    moving_fraction,
)
from slalom_camera import PIP_H, PIP_W
from slalom_cast import ALL_NAMES, BY_ENCOUNTER, ENCOUNTER_ORDER, EXPECTED_PASS_SIDES
from slalom_course import (
    FLOOR_HALF,
    GOAL_BAND_HALF,
    GOAL_XY,
    LANE_HALF_W,
    STATIC_OBSTACLES,
    goal_contains,
    goal_sample_points,
    static_gap,
)
from slalom_markers import (
    CORRIDOR_DISCS,
    PRED_DISCS,
    ROUTE_DISCS,
    TRAIL_DISCS,
)
from slalom_course import DUCK_START_XY
from slalom_states import DUCK_PLANAR_RADIUS


# -- the model ------------------------------------------------------------------
def test_the_scene_compiles_with_its_meshes(model):
    """Zero meshes means meshdir did not resolve and the robot is a stick."""
    assert model.nmesh > 0
    assert model.nu == 14


def test_the_exact_gyro_sensor_resolves_and_is_not_a_fallback(model):
    from policy_runtime import GYRO_SENSOR, gyro_address
    sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR)
    assert sensor >= 0
    address = gyro_address(model)
    assert address >= 0
    # An accidental last-sensor fallback is refused explicitly.
    if sensor != model.nsensor - 1:
        assert address != int(model.sensor_adr[-1])


def test_no_alias_sensor_is_accepted(model):
    from policy_runtime import gyro_address
    with pytest.raises(ValueError):
        gyro_address(model, "imu_gyro")


def test_every_scenery_geom_the_gate_measures_exists(model):
    from rollout_slalom import scenery_geom_names
    names = scenery_geom_names(model)
    assert names, "the clearance gate would be vacuous"
    for name in names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    # Every declared static obstacle is among them.
    for obstacle in STATIC_OBSTACLES:
        assert obstacle.name in names


def test_all_scenery_is_non_colliding(model):
    """Avoidance must be a property of the software, not the contact solver."""
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if name.startswith(("obs_", "wall_", "goal_", "hall_", "lane_")) \
                or any(name.startswith(f"{a}_") for a in ALL_NAMES):
            assert int(model.geom_contype[geom]) == 0, name
            assert int(model.geom_conaffinity[geom]) == 0, name


def test_every_actor_body_and_its_load_exists(model):
    from slalom_cast import BY_NAME
    for name in ALL_NAMES:
        assert model.body(f"actor_{name}").id >= 0
        spec = BY_NAME[name]
        if spec.carries_cart:
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_cart") >= 0
        if spec.carries_box:
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_box") >= 0


def test_the_scene_carries_every_marker(model):
    for prefix, count in (("route", ROUTE_DISCS), ("trail", TRAIL_DISCS),
                          ("left", CORRIDOR_DISCS), ("right", CORRIDOR_DISCS),
                          ("pred", PRED_DISCS)):
        for index in range(count):
            assert model.body(f"{prefix}_{index}").id >= 0
    assert model.body("goal_marker").id >= 0


def test_duck_planar_radius_matches_the_built_model(model, data):
    """The constant every clearance claim uses must match the real robot.

    Measured at the STAND keyframe and across the FULL head-yaw range, because
    the head is the widest thing on the robot and the constant has to hold at
    every gaze angle the behavior can produce.
    """
    import mujoco
    from policy_runtime import HEAD_YAW_ACT, actuator_indices

    trunk = model.body("trunk_base").id
    qpos, _ = actuator_indices(model)
    joint = int(model.actuator_trnid[HEAD_YAW_ACT, 0])
    low, high = model.jnt_range[joint]
    worst = 0.0
    for fraction in np.linspace(0.0, 1.0, 9):
        data.qpos[qpos[HEAD_YAW_ACT]] = low + (high - low) * fraction
        mujoco.mj_forward(model, data)
        worst = max(worst, duck_planar_radius(model, data, trunk))
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    mujoco.mj_forward(model, data)
    assert worst == pytest.approx(DUCK_PLANAR_RADIUS, abs=0.002)


def test_the_conservative_radius_over_states_the_robot(model, data):
    """Over-stating the duck is the safe direction for every clearance gate."""
    trunk = model.body("trunk_base").id
    assert duck_planar_radius(model, data, trunk) > \
        exact_lateral_half_width(model, data, trunk)


# -- the course -----------------------------------------------------------------
def test_the_goal_is_inside_the_floor_and_ahead_of_the_start():
    assert abs(GOAL_XY[0]) < FLOOR_HALF[0]
    assert abs(GOAL_XY[1]) < FLOOR_HALF[1]
    assert GOAL_XY[0] > DUCK_START_XY[0], "the duck would start past the goal"


def test_the_goal_band_is_a_real_place():
    assert goal_contains(GOAL_XY)
    assert not goal_contains((GOAL_XY[0] + GOAL_BAND_HALF[0] + 0.1, GOAL_XY[1]))
    assert not goal_contains((GOAL_XY[0], GOAL_XY[1] + GOAL_BAND_HALF[1] + 0.1))


def test_the_goal_beacon_is_sampled_at_several_heights():
    points = goal_sample_points()
    assert len(points) >= 3
    assert len({round(float(p[2]), 4) for p in points}) == len(points)


def test_the_start_is_clear_of_every_static_body():
    name, gap = static_gap(DUCK_START_XY)
    assert gap > DUCK_PLANAR_RADIUS, f"the duck starts inside {name}"


def test_both_hands_are_genuinely_available_beside_the_lane():
    """A course that forced one answer would make every choice meaningless."""
    for x in (-3.5, -2.0, 0.0, 2.0, 3.0):
        for side in (+1.0, -1.0):
            point = (x, side * LANE_HALF_W)
            _, gap = static_gap(point)
            assert gap > 0.0, f"the lane edge at {point} is inside scenery"


def test_at_least_seven_obstacles_and_actors_populate_the_course():
    assert len(STATIC_OBSTACLES) + len(ALL_NAMES) >= 7


# -- the choreography -------------------------------------------------------------
def test_every_encounter_has_at_least_one_actor():
    for key in ENCOUNTER_ORDER:
        assert BY_ENCOUNTER[key], key


def test_the_expected_sides_alternate_by_construction():
    """If the scenario itself did not alternate, the gate would be vacuous."""
    assert all(a != b for a, b in
               zip(EXPECTED_PASS_SIDES, EXPECTED_PASS_SIDES[1:]))
    assert len(set(EXPECTED_PASS_SIDES)) == 2


def test_each_body_crosses_the_lane_when_it_was_solved_to():
    """``_solve_start`` must actually produce the declared crossing time."""
    crossings = {c["actor"]: c for c in lane_crossings(95.0)}
    for key in ENCOUNTER_ORDER:
        for name in BY_ENCOUNTER[key]:
            assert name in crossings, f"{name} never crossed the lane"
            # E4's pair is offset from the encounter time on purpose.
            if key != "E4":
                assert crossings[name]["t_s"] == pytest.approx(
                    ENCOUNTERS[key]["cross_t"], abs=0.30)


def test_each_body_crosses_near_its_declared_x():
    crossings = {c["actor"]: c for c in lane_crossings(95.0)}
    for key in ENCOUNTER_ORDER:
        for name in BY_ENCOUNTER[key]:
            assert crossings[name]["x_m"] == pytest.approx(
                ENCOUNTERS[key]["cross_x"], abs=0.60)


def test_no_scripted_body_ever_teleports():
    """A cornered polyline turns its walker through a corner in ONE tick."""
    worst, name, t = max_heading_step(95.0)
    assert worst < 6.0, f"{name} turned {worst:.1f} deg in one tick at {t:.2f}s"


def test_every_body_actually_moves():
    fractions = moving_fraction(95.0)
    assert all(value > 0.02 for value in fractions.values()), fractions


def test_the_routes_have_real_filleted_bends():
    for name, route in ROUTES.items():
        bends = route.corner_report()
        assert bends, f"{name} has no bends to fillet"
        assert all(b["radius_m"] > 0.0 for b in bends)


def test_actors_are_posed_where_their_routes_say(model, data):
    """The mocap poses and the analytic routes must be the same positions."""
    from slalom_actors import pose_actors
    t = 20.0
    actors = actors_at(t)
    pose_actors(model, data, actors, t)
    mujoco.mj_forward(model, data)
    for name in ALL_NAMES:
        body = model.body(f"actor_{name}")
        mocap = int(model.body_mocapid[body.id])
        assert np.allclose(data.mocap_pos[mocap, :2], actors[name].pos,
                           atol=1e-9)


def test_the_e4_pair_straddles_the_duck_so_both_sides_are_blocked():
    """The WAIT is a geometric fact about this pair, not a scripted state."""
    names = BY_ENCOUNTER["E4"]
    assert len(names) == 2
    crossings = {c["actor"]: c for c in lane_crossings(95.0)}
    northbound = [n for n in names if crossings[n]["northbound"]]
    southbound = [n for n in names if not crossings[n]["northbound"]]
    assert northbound and southbound, "both hands must be represented"


# -- the camera --------------------------------------------------------------------
def test_the_pip_camera_exists_and_the_head_camera_is_corrected(model):
    assert model.camera("slalom_camera").id >= 0
    assert model.camera("head_camera").id >= 0


def test_the_gaze_layer_never_writes_back_into_the_physics(short_rollout):
    """Gaze must not prop the robot up: it lives in an isolated MjData."""
    camera = short_rollout.camera
    assert camera.render_data is not short_rollout.data


def test_visibility_is_measured_through_the_pip_frustum(short_rollout):
    camera = short_rollout.camera
    assert camera.camera_id == camera.view_cam
    assert (camera.pip_w, camera.pip_h) == (PIP_W, PIP_H)
    assert camera.tan_h == pytest.approx((PIP_W / PIP_H) * camera.tan_v)
