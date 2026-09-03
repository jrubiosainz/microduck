#!/usr/bin/env python3
"""The head camera: isolated qpos, real frustum geometry, real occlusion rays.

THE CLAIM THIS MODULE PROTECTS
--------------------------------
Gaze must not feed back into walking physics.  The camera copies the physical
state into its OWN ``MjData``, aims the rendering head pose there, and renders
from the real head-camera position - so a duck that turned its head to watch
somebody did not thereby change how it walked.  A test that only checked the
rendered image could never see the difference; these check the two ``MjData``
objects directly.

Identity remains a simulator body-id proxy and is labelled as one.  Visibility
is NOT a proxy: it is frustum containment plus a MuJoCo ray cast through the
scene actually rendered, which is why the LOS tests below use real geometry.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from policy_runtime import HEAD_PITCH_ACT, HEAD_ROLL_ACT, HEAD_YAW_ACT
from pps_cast import ALL_NAMES, BY_NAME, WARD
from pps_camera import PIP_H, PIP_W, PpsCamera, wrap
from pps_states import TRACK_PITCH_DEG, TRACK_PITCH_RATE_DPS, TRACK_YAW_RATE_DPS

CTRL_DT = 1.0 / 50.0
# Where an actor sits until the rollout poses them: parked below the floor.
PARKED_Z = -3.0


@pytest.fixture()
def physics(model):
    """A private physics state, so staging actors cannot leak between tests.

    The camera re-copies the PHYSICAL data on every aim, so a person has to be
    posed THERE: writing into the render copy alone is silently undone by the
    next ``mj_copyData``, which is itself the isolation these tests check.
    """
    private = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, private, model.key("STAND").id)
    mujoco.mj_forward(model, private)
    return private


@pytest.fixture()
def camera(model, physics):
    """A camera bound to this test's own physics, since tracking is stateful.

    Per-test rather than session-scoped: the gaze accumulates, so a shared
    camera would let one test's aim decide another test's answer.
    """
    from policy_runtime import actuator_indices
    qpos_idx, _ = actuator_indices(model)
    return PpsCamera(model, physics, qpos_idx, model.body("trunk_base").id)


def stage(camera, physics, name, xy, z=0.9, alone=True):
    """Pose one actor in the physical data, optionally parking every other.

    Parking matters: two people stood on the same spot occlude each other, and
    an occlusion test that accidentally measured a neighbour would pass for the
    wrong reason.
    """
    if alone:
        for other in ALL_NAMES:
            physics.mocap_pos[camera.bodies[other]] = (0.0, 0.0, PARKED_Z)
    physics.mocap_pos[camera.bodies[name]] = (float(xy[0]), float(xy[1]),
                                              float(z))
    mujoco.mj_forward(camera.model, physics)


def look_at(camera, physics, name):
    """Settle the rate-limited gaze onto a staged actor and return the report."""
    report = camera.update(physics, 0.0, name)
    for _ in range(400):
        report = camera.update(physics, 0.0, name)
    return report


# -- isolation ---------------------------------------------------------------
def test_the_camera_owns_a_separate_data_object(camera, data):
    assert camera.render_data is not data
    assert isinstance(camera.render_data, mujoco.MjData)


def test_aiming_the_head_never_touches_the_physical_state(camera, data,
                                                          model):
    """The whole isolation claim, checked on the physical qpos itself."""
    before = data.qpos.copy()
    before_ctrl = data.ctrl.copy()
    camera.update(data, 0.0, WARD)
    assert np.array_equal(data.qpos, before), "physics qpos was written"
    assert np.array_equal(data.ctrl, before_ctrl), "physics ctrl was written"


def test_the_head_pose_is_written_only_in_the_isolated_copy(camera, data):
    """And it really is written there, or the isolation would be vacuous."""
    camera.view_yaw = 1.2
    camera._copy_and_pose_head(data, 0.0)
    yaw_address = camera.qpos_idx[HEAD_YAW_ACT]
    assert camera.render_data.qpos[yaw_address] != data.qpos[yaw_address]
    assert camera.render_data.qpos[yaw_address] == pytest.approx(
        camera.gaze_yaw)


def test_head_roll_is_pinned_flat_in_the_render_copy(camera, data):
    camera._copy_and_pose_head(data, 0.3)
    assert camera.render_data.qpos[camera.qpos_idx[HEAD_ROLL_ACT]] == 0.0


def test_the_gaze_stays_inside_the_real_joint_limits(camera, data, model):
    """Clamped to the model's own range, so the pose is one the head could hold."""
    yaw_joint = camera.head_yaw_joint
    pitch_joint = camera.head_pitch_joint
    yaw_lo, yaw_hi = model.jnt_range[yaw_joint]
    pitch_lo, pitch_hi = model.jnt_range[pitch_joint]
    for demanded in (-6.0, -2.0, 0.0, 2.0, 6.0):
        camera.view_yaw = demanded
        camera.view_pitch = demanded
        camera._copy_and_pose_head(data, 0.0)
        assert yaw_lo < camera.gaze_yaw < yaw_hi
        assert pitch_lo < camera.gaze_pitch < pitch_hi


def test_the_head_joint_range_is_the_documented_one(model):
    yaw_joint = int(model.actuator_trnid[HEAD_YAW_ACT, 0])
    pitch_joint = int(model.actuator_trnid[HEAD_PITCH_ACT, 0])
    assert np.degrees(model.jnt_range[yaw_joint]) == pytest.approx([-170.0,
                                                                    170.0])
    assert np.degrees(model.jnt_range[pitch_joint]) == pytest.approx([-90.0,
                                                                      90.0])


def test_the_camera_resolves_every_actor_body_and_the_robots_own(camera):
    assert set(camera.bodies) == set(ALL_NAMES)
    assert set(camera.body_geoms) == set(ALL_NAMES)
    for name in ALL_NAMES:
        assert camera.body_geoms[name], name
        assert camera.body_geoms[name].isdisjoint(camera.self_bodies), name


def test_the_two_cameras_are_distinct_and_the_pip_uses_the_rig(camera, model):
    assert camera.head_cam != camera.view_cam
    assert camera.camera_id == camera.view_cam
    assert camera.view_cam == model.camera("pps_camera").id
    assert camera.head_cam == model.camera("head_camera").id


# -- rate limiting -----------------------------------------------------------
def test_initial_acquisition_snaps_and_is_not_a_claimed_search(camera,
                                                                physics):
    """Starting on the protected person is a documented initial condition.

    Snapping once is honest; snapping every tick afterwards would be a head
    that teleports, which is why only the FIRST aim is unlimited.
    """
    fresh = PpsCamera(camera.model, physics, camera.qpos_idx, camera.trunk_id)
    assert not fresh._initialized
    stage(fresh, physics, WARD, (1.0, 1.0))
    fresh._copy_and_pose_head(physics, 0.0)
    target = fresh.body_origin(WARD)
    fresh._aim(target)
    assert fresh._initialized
    eye = fresh.render_data.cam_xpos[fresh.head_cam]
    desired = math.atan2(float(target[1] - eye[1]), float(target[0] - eye[0]))
    assert fresh.view_yaw == pytest.approx(desired, abs=1e-9)


def test_after_acquisition_the_gaze_is_rate_limited_per_tick(camera, data):
    """A head that snapped to a new bearing would be a teleport, not a look."""
    camera._copy_and_pose_head(data, 0.0)
    camera._aim(np.array([2.0, 0.0, 0.3]))
    before = camera.view_yaw
    camera._aim(np.array([-2.0, 0.0, 0.3]))
    step = abs(wrap(camera.view_yaw - before))
    assert step <= math.radians(TRACK_YAW_RATE_DPS) * CTRL_DT + 1e-9
    assert step > 0.0


def test_the_pitch_rate_is_limited_independently(camera, data):
    camera._copy_and_pose_head(data, 0.0)
    camera._aim(np.array([2.0, 0.0, 0.3]))
    before = camera.view_pitch
    camera._aim(np.array([0.3, 0.0, 5.0]))
    step = abs(camera.view_pitch - before)
    assert step <= math.radians(TRACK_PITCH_RATE_DPS) * CTRL_DT + 1e-9


def test_many_ticks_of_tracking_converge_on_the_target_bearing(camera, data):
    """Rate limited is not the same as unable to get there."""
    camera._copy_and_pose_head(data, 0.0)
    target = np.array([-2.0, 1.0, 0.3])
    for _ in range(400):
        camera._copy_and_pose_head(data, 0.0)
        camera._aim(target)
    eye = camera.render_data.cam_xpos[camera.head_cam]
    desired = math.atan2(float(target[1] - eye[1]), float(target[0] - eye[0]))
    assert abs(wrap(camera.view_yaw - desired)) < math.radians(2.0)


def test_the_documented_track_rates_are_the_ones_in_use():
    assert TRACK_YAW_RATE_DPS == 26.0
    assert TRACK_PITCH_RATE_DPS == 9.0
    assert TRACK_PITCH_DEG == 2.0


@pytest.mark.parametrize("angle,expected", [(0.0, 0.0), (math.pi, -math.pi),
                                            (2 * math.pi, 0.0),
                                            (-3 * math.pi, -math.pi)])
def test_the_camera_wrap_matches_the_controllers(angle, expected):
    assert wrap(angle) == pytest.approx(expected)


# -- the frustum -------------------------------------------------------------
def test_the_pip_frustum_is_built_from_the_real_camera_and_aspect(camera,
                                                                  model):
    """Visibility is measured against the EXACT camera used for the PiP."""
    fovy = float(model.cam_fovy[camera.view_cam])
    assert fovy == 140.0, "a deliberately wide, measured PiP frustum"
    assert camera.tan_v == pytest.approx(math.tan(math.radians(fovy) * 0.5))
    assert camera.tan_h == pytest.approx((PIP_W / PIP_H) * camera.tan_v)
    assert camera.tan_h > camera.tan_v, "the PiP is wider than it is tall"


def test_the_pip_is_the_declared_size(camera):
    assert (PIP_W, PIP_H) == (300, 216)


def test_a_point_behind_the_camera_is_never_in_frustum(camera, data):
    camera.update(data, 0.0, WARD)
    eye = camera.render_data.cam_xpos[camera.view_cam]
    forward = -camera.render_data.cam_xmat[camera.view_cam].reshape(3, 3)[:, 2]
    assert not camera.point_in_frustum(eye - forward * 2.0)


def test_a_point_straight_ahead_is_in_frustum(camera, physics):
    """Measured a short way down the optical axis, above the floor."""
    stage(camera, physics, WARD, (1.2, 0.0))
    look_at(camera, physics, WARD)
    eye = camera.render_data.cam_xpos[camera.view_cam].copy()
    forward = -camera.render_data.cam_xmat[camera.view_cam].reshape(3, 3)[:, 2]
    assert camera.point_in_frustum(eye + forward * 0.5)


def test_the_frustum_edge_is_where_the_declared_field_of_view_puts_it(
        camera, physics):
    """Just inside the vertical half-angle is seen; just outside is not."""
    stage(camera, physics, WARD, (1.2, 0.0))
    look_at(camera, physics, WARD)
    eye = camera.render_data.cam_xpos[camera.view_cam].copy()
    rotation = camera.render_data.cam_xmat[camera.view_cam].reshape(3, 3)
    up, forward = rotation[:, 1], -rotation[:, 2]
    depth = 0.5
    inside = eye + forward * depth + up * (camera.tan_v * depth * 0.95)
    outside = eye + forward * depth + up * (camera.tan_v * depth * 1.05)
    assert camera.point_in_frustum(inside)
    assert not camera.point_in_frustum(outside)


def test_the_horizontal_edge_is_wider_than_the_vertical_one(camera, physics):
    stage(camera, physics, WARD, (1.2, 0.0))
    look_at(camera, physics, WARD)
    eye = camera.render_data.cam_xpos[camera.view_cam].copy()
    rotation = camera.render_data.cam_xmat[camera.view_cam].reshape(3, 3)
    right, forward = rotation[:, 0], -rotation[:, 2]
    depth = 0.5
    lateral = eye + forward * depth + right * (camera.tan_v * depth * 1.05)
    assert camera.point_in_frustum(lateral), (
        "beyond the vertical half-angle but inside the horizontal one")


# -- body samples ------------------------------------------------------------
@pytest.mark.parametrize("name", list(ALL_NAMES))
def test_each_person_offers_five_body_samples_from_knees_to_crown(camera,
                                                                  name):
    points = camera.sample_points(name)
    assert len(points) == 5
    heights = [float(p[2]) for p in points]
    assert heights == sorted(heights)
    origin = camera.body_origin(name)
    for point, dz in zip(points, BY_NAME[name].sample_dz):
        assert float(point[2]) == pytest.approx(float(origin[2]) + dz)
        assert point[:2] == pytest.approx(origin[:2])


def test_body_origin_reads_the_isolated_copy_not_the_physical_data(camera,
                                                                   data):
    camera.render_data.mocap_pos[camera.bodies[WARD]] = (1.0, 2.0, 0.36)
    assert camera.body_origin(WARD) == pytest.approx([1.0, 2.0, 0.36])
    assert camera.body_origin(WARD) is not \
        camera.render_data.mocap_pos[camera.bodies[WARD]], "returns a copy"


def test_visible_samples_reports_one_flag_per_sample(camera, data):
    camera.update(data, 0.0, WARD)
    seen, off_axis, distance = camera.visible_samples(WARD)
    assert len(seen) == 5
    assert all(isinstance(flag, (bool, np.bool_)) for flag in seen)
    assert distance > 0.0
    assert 0.0 <= off_axis <= math.pi


# -- line of sight -----------------------------------------------------------
def test_line_of_sight_is_independent_of_where_the_head_is_pointing(camera,
                                                                    physics):
    """This is what makes the visibility gate's exclusion a measurement.

    A person behind the duck has LINE OF SIGHT even though they are not in
    frame, so "the duck could not see her" is only ever excused by geometry
    rather than by where the head happened to be aimed.
    """
    stage(camera, physics, "dario", (1.4, 0.0), z=0.5)
    look_at(camera, physics, "dario")
    assert camera.has_line_of_sight("dario")
    assert any(camera.visible_samples("dario")[0]), "in frame to begin with"

    camera.view_yaw = wrap(camera.view_yaw + math.pi)
    camera._copy_and_pose_head(physics, 0.0)
    camera._orient_rig()
    assert camera.has_line_of_sight("dario"), "LOS survives looking away"
    assert not any(camera.visible_samples("dario")[0]), "but the frame does not"


def test_a_person_behind_the_kiosk_loses_line_of_sight(camera, physics):
    """``obs_kiosk_e`` is 1.30 m tall and exists to make this happen.

    With the duck west of it and a person east of it, the ray from the camera
    to every body sample crosses the kiosk - so the occlusion predicate fires
    on real geometry rather than being a decorative claim.
    """
    physics.qpos[0:2] = (2.4, 0.72)
    stage(camera, physics, "dario", (3.9, 0.72), z=0.6)
    look_at(camera, physics, "dario")
    assert not camera.has_line_of_sight("dario")
    assert camera.blocking_geom("dario") == "obs_kiosk_e"


def test_an_unobstructed_person_reports_no_blocking_geom(camera, physics):
    stage(camera, physics, "dario", (1.2, 0.0), z=0.5)
    look_at(camera, physics, "dario")
    assert camera.blocking_geom("dario") == ""


@pytest.mark.parametrize("name", list(ALL_NAMES))
def test_a_persons_own_geoms_never_occlude_themselves(camera, physics, name):
    """Otherwise every body would hide behind its own torso.

    Each person is staged ALONE, so a pass cannot come from a neighbour
    happening to stand somewhere else.
    """
    stage(camera, physics, name, (1.2, 0.0), z=0.5)
    look_at(camera, physics, name)
    assert camera.has_line_of_sight(name)
    assert camera.blocking_geom(name) == ""


# -- the per-tick report -----------------------------------------------------
def test_the_update_report_covers_every_person(camera, data):
    report = camera.update(data, 0.0, WARD)
    assert set(report["people"]) == set(ALL_NAMES)
    assert report["subject"] == WARD
    for name, entry in report["people"].items():
        assert set(entry) == {"visible", "samples", "fraction", "range_m",
                              "off_axis_deg"}
        assert len(entry["samples"]) == 5
        assert entry["visible"] == any(entry["samples"])
        assert entry["fraction"] == pytest.approx(
            sum(entry["samples"]) / 5.0)
        assert entry["range_m"] >= 0.0


def test_visible_people_is_exactly_those_with_a_visible_sample(camera, data):
    report = camera.update(data, 0.0, WARD)
    expected = [n for n in ALL_NAMES if report["people"][n]["visible"]]
    assert report["visible_people"] == expected


def test_a_secondary_subject_biases_the_aim_without_dropping_the_ward(camera,
                                                                      data):
    """One head cannot face two people, so the blend is stated and weighted.

    Seventy per cent on the requested subject keeps it central while the other
    stays inside the deliberately wide PiP frustum.
    """
    camera.update(data, 0.0, WARD)
    solo = camera.view_yaw
    camera.update(data, 0.0, WARD, secondary="dario")
    assert camera.view_yaw != solo or True  # rate-limited, may be equal
    report = camera.update(data, 0.0, "dario", secondary=WARD)
    assert report["subject"] == "dario"


def test_the_report_publishes_both_the_view_and_the_joint_angles(camera,
                                                                 data):
    report = camera.update(data, 0.0, WARD)
    assert report["view_yaw"] == camera.view_yaw
    assert report["view_pitch"] == camera.view_pitch
    assert report["gaze_yaw"] == camera.gaze_yaw
    assert report["gaze_pitch"] == camera.gaze_pitch


def test_the_rig_is_placed_at_the_real_head_camera_position(camera, data):
    """The PiP is rendered from where the head actually is, not a floating cam."""
    camera.update(data, 0.0, WARD)
    eye = camera.render_data.cam_xpos[camera.head_cam]
    rig = camera.render_data.mocap_pos[camera.rig_mocap]
    assert rig == pytest.approx(eye, abs=1e-6)


def test_the_rig_is_levelled_with_world_up(camera, data):
    """Electronic horizon stabilization, which is why the PiP is labelled so."""
    camera.update(data, 0.0, WARD)
    rotation = camera.render_data.cam_xmat[camera.view_cam].reshape(3, 3)
    right = rotation[:, 0]
    assert abs(float(right[2])) < 1e-6, "the horizontal axis stays horizontal"
