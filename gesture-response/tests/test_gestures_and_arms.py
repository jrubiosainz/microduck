#!/usr/bin/env python3
"""The gesture classifier and the arm animation, tested against each other.

These are the two halves that share nothing but the MuJoCo model: ``gest_arm``
drives joint angles, ``gest_pose`` reads world positions, and a sign error in
one shows up here rather than as a gesture that classifies wrongly at runtime.

Every threshold assertion compares a MEASURED value against a DIFFERENT
constant.  A test that asserts a constant equals itself passes forever and
protects nothing, so where a bound is checked it is checked against the
measurement it was derived from.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gest_arm import (
    ANIMATED,
    BACK_UP,
    COME,
    JOINT_KEYS,
    PARTIAL,
    POINT_LEFT_ARM,
    POINT_RIGHT_ARM,
    RAISE_S,
    REST,
    STOP,
    WAVE,
    arm_targets,
    envelope,
)
from gest_gesture import (
    BY_TEMPLATE,
    MIN_CONFIDENCE,
    MOTION_WINDOW_S,
    MOVING_BAR,
    OSCILLATION_WANDER_BAR,
    STILL_BAR,
    TEMPLATES,
    classify,
    command_for,
    score,
)
from gest_pose import measure_body
from gest_script import TEMPLATE_FOR_GESTURE, command_for_gesture


# -- the template table itself -------------------------------------------------
def test_every_template_maps_to_a_distinct_command():
    commands = [t.command for t in TEMPLATES]
    assert len(commands) == len(set(commands)), (
        f"two templates share a command: {commands}")


def test_pointing_templates_map_to_opposite_turns():
    """The instructor faces the duck, so her arms mirror the duck's turns.

    This is the single place the person frame and the robot frame meet, and
    getting it backwards would make every turn claim in the run exactly wrong
    while every other test still passed.
    """
    assert BY_TEMPLATE["POINT_LEFT_ARM"].command == "TURN_RIGHT"
    assert BY_TEMPLATE["POINT_RIGHT_ARM"].command == "TURN_LEFT"


def test_pointing_windows_are_genuinely_one_sided():
    """The two POINT templates must not accept each other's lateral value."""
    left = BY_TEMPLATE["POINT_LEFT_ARM"].lateral
    right = BY_TEMPLATE["POINT_RIGHT_ARM"].lateral
    assert left.contains(0.999) and not left.contains(-0.999)
    assert right.contains(-0.999) and not right.contains(0.999)


def test_animation_names_all_bridge_to_a_template():
    """Every animated gesture reaches a real command through its template.

    THE REGRESSION THIS PINS.  ``gest_arm`` calls the pointing animations
    ``POINT_L_ARM``/``POINT_R_ARM`` while ``gest_gesture`` calls the templates
    ``POINT_LEFT_ARM``/``POINT_RIGHT_ARM``.  Passing an animation name straight
    to ``command_for`` returned ``\"\"`` for exactly those two, so the acceptance
    gate silently asked for two EMPTY commands and would have been satisfied by
    a duck that executed neither turn.
    """
    for gesture in ANIMATED:
        if gesture == PARTIAL:
            continue
        assert command_for_gesture(gesture), (
            f"{gesture} maps to no command")
    assert command_for_gesture(POINT_LEFT_ARM) == "TURN_RIGHT"
    assert command_for_gesture(POINT_RIGHT_ARM) == "TURN_LEFT"


def test_raw_animation_name_is_not_a_template_name():
    """The bug above was possible because the two namespaces overlap partly."""
    assert command_for(POINT_LEFT_ARM) == "", (
        "an animation name must NOT resolve as a template name, or the bridge "
        "in gest_script is pointless")
    assert TEMPLATE_FOR_GESTURE[POINT_LEFT_ARM] == "POINT_LEFT_ARM"


# -- the motion window ----------------------------------------------------------
def test_motion_window_is_longer_than_every_oscillation_period():
    """The window must contain a WHOLE cycle of the gestures that oscillate.

    THE REGRESSION THIS PINS, and it is the one that stopped the behavior dead.
    The window was 0.60 s while the beckon runs at 1.15 Hz (0.870 s per cycle),
    so it could only ever hold a half-swing - a one-way hand movement whose
    ``wander`` collapses to about 1.0, which is indistinguishable from an arm on
    its way up.  MEASURED consequence: COME's confidence fell below the bar
    twice a second, the confirm window reset every ~0.7 s, and the duck read
    COME thirteen times in twelve seconds without ever confirming it.
    """
    from gest_arm import BECKON_HZ, PUSH_HZ, WAVE_HZ

    for name, rate in (("beckon", BECKON_HZ), ("wave", WAVE_HZ),
                       ("push", PUSH_HZ)):
        period = 1.0 / rate
        assert MOTION_WINDOW_S >= period, (
            f"the {name} period is {period:.3f}s but the motion window is "
            f"{MOTION_WINDOW_S:.3f}s, so the window can only ever contain a "
            "half-swing")


def test_still_and_moving_bars_do_not_overlap():
    assert STILL_BAR < MOVING_BAR, (
        f"a pose could be both still ({STILL_BAR}) and moving ({MOVING_BAR})")


# -- posing the real model ------------------------------------------------------
class ArmPoser:
    """Drives one person's arms on the compiled model and reads world keypoints."""

    def __init__(self, model, data, person: str = "mira",
                 yaw_deg: float = -90.0):
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.person = person
        self.yaw = math.radians(yaw_deg)
        mocap = int(model.body_mocapid[model.body(f"actor_{person}").id])
        data.mocap_pos[mocap] = (0.0, 1.30, 0.36)
        data.mocap_quat[mocap] = np.array(
            [math.cos(self.yaw / 2.0), 0.0, 0.0, math.sin(self.yaw / 2.0)])
        self.joints = {k: model.joint(f"{person}_{k}").id for k in JOINT_KEYS}

    def keypoints(self, gesture: str, elapsed: float, span: float) -> dict:
        targets, _ = arm_targets(gesture, elapsed, span, 0.0)
        for key, joint in self.joints.items():
            self.data.qpos[int(self.model.jnt_qposadr[joint])] = targets[key]
        self.mujoco.mj_forward(self.model, self.data)
        out = {}
        for side in ("l", "r"):
            out[f"{side}_shoulder"] = self.data.xpos[
                self.model.body(f"{self.person}_shoulder_{side}").id].copy()
            out[f"{side}_elbow"] = self.data.xpos[
                self.model.body(f"{self.person}_fore_{side}").id].copy()
            out[f"{side}_hand"] = self.data.xpos[
                self.model.body(f"{self.person}_hand_{side}").id].copy()
        return out

    def measure(self, gesture: str, elapsed: float, span: float):
        return measure_body(self.person, self.yaw,
                            self.keypoints(gesture, elapsed, span),
                            {"l": True, "r": True})

    def motion(self, gesture: str, elapsed: float, span: float,
               window_s: float = MOTION_WINDOW_S, dt: float = 0.02):
        from gest_cast import BY_NAME

        steps = int(round(window_s / dt))
        frames = [self.keypoints(gesture, max(elapsed - window_s + i * dt, 0.0),
                                 span)
                  for i in range(steps + 1)]
        best_path, best_net = 0.0, 0.0
        for side in ("l", "r"):
            track = [f[f"{side}_hand"] for f in frames]
            path = sum(float(np.linalg.norm(b - a))
                       for a, b in zip(track, track[1:]))
            if path >= best_path:
                best_path = path
                best_net = float(np.linalg.norm(track[-1] - track[0]))
        return (best_path / BY_NAME[self.person].arm_span,
                best_path / max(best_net, 1e-6))


@pytest.fixture(scope="module")
def poser(model, data):
    return ArmPoser(model, data)


STEADY_ELAPSED = RAISE_S + MOTION_WINDOW_S + 0.40
SPAN = 6.4


@pytest.mark.parametrize("gesture,template", [
    (COME, "COME"),
    (STOP, "STOP"),
    (POINT_LEFT_ARM, "POINT_LEFT_ARM"),
    (POINT_RIGHT_ARM, "POINT_RIGHT_ARM"),
    (BACK_UP, "BACK_UP"),
    (WAVE, "WAVE"),
])
def test_each_gesture_classifies_as_its_own_template(poser, gesture, template):
    """Posed on the REAL model, each animation reads as its own template."""
    travel, wander = poser.motion(gesture, STEADY_ELAPSED, SPAN)
    pose = poser.measure(gesture, STEADY_ELAPSED, SPAN)
    reading = classify(pose, travel, wander)
    assert reading.template == template, (
        f"{gesture} read as {reading.template!r} (conf {reading.confidence:.2f})")
    assert reading.confidence >= MIN_CONFIDENCE


@pytest.mark.parametrize("gesture", [REST, PARTIAL])
def test_non_commands_are_refused_at_every_instant(poser, gesture):
    """The ambiguous partial and the rest pose must never classify.

    Checked across the WHOLE hold including the raise transient: there is no
    instant at which either may be accepted, which is a stronger claim than
    checking one sample.
    """
    span = 0.0 if gesture == REST else 5.2
    for index in range(0, 120, 3):
        elapsed = 0.0 if gesture == REST else RAISE_S + index * 0.02
        travel, wander = poser.motion(gesture, elapsed, span)
        pose = poser.measure(gesture, elapsed, span)
        reading = classify(pose, travel, wander)
        assert not reading.accepted, (
            f"{gesture} was accepted as {reading.template} at elapsed "
            f"{elapsed:.2f}s with confidence {reading.confidence:.2f}")


def test_partial_is_refused_on_its_own_margin_not_on_arm_count(poser):
    """The partial has ONE raised arm, like COME and STOP.

    That matters: if it were refused because no arm was raised, the rejection
    would prove nothing about the classifier's windows.  It is refused because
    its measured extension sits below every one-armed template's floor.
    """
    pose = poser.measure(PARTIAL, STEADY_ELAPSED, 5.2)
    assert pose.raised_count == 1, (
        "the partial must present a raised arm, or its rejection is vacuous")
    primary = pose.primary
    assert primary is not None
    assert not BY_TEMPLATE["STOP"].extension.contains(primary.extension), (
        f"the partial's extension {primary.extension:.3f} is inside STOP's "
        "window; it is not actually ambiguous-but-refusable")


def test_come_and_stop_are_separated_by_motion_alone(poser):
    """Both are one raised forward arm; only the hand motion tells them apart.

    Verified by feeding the STOP pose with COME's motion features and requiring
    the answer to change - which proves the motion rule is load-bearing rather
    than decorative.
    """
    stop_pose = poser.measure(STOP, STEADY_ELAPSED, SPAN)
    still_travel, still_wander = poser.motion(STOP, STEADY_ELAPSED, SPAN)
    assert classify(stop_pose, still_travel, still_wander).template == "STOP"

    moving = classify(stop_pose, MOVING_BAR + 0.5, OSCILLATION_WANDER_BAR + 2.0)
    assert moving.template != "STOP", (
        "a moving hand still read as STOP, so the 'still' rule does nothing")


def test_back_up_is_the_only_two_armed_template(poser):
    pose = poser.measure(BACK_UP, STEADY_ELAPSED, SPAN)
    assert pose.raised_count == 2
    two_armed = [t.name for t in TEMPLATES if t.arms == 2]
    assert two_armed == ["BACK_UP"]


def test_wave_is_separated_from_come_by_extension(poser):
    """MEASURED: their elevation windows overlap almost exactly.

    So the separation has to come from the arm being straight, and this asserts
    the measured extensions really do sit on opposite sides of WAVE's floor.
    """
    come = poser.measure(COME, STEADY_ELAPSED, SPAN).primary
    wave = poser.measure(WAVE, STEADY_ELAPSED, SPAN).primary
    floor = BY_TEMPLATE["WAVE"].extension.low
    assert come.extension < floor <= wave.extension, (
        f"COME {come.extension:.3f} and WAVE {wave.extension:.3f} do not "
        f"straddle WAVE's extension floor {floor}")


def test_every_template_holds_its_window_over_the_whole_steady_hold(poser):
    """The confirm window needs a SUSTAINED reading, not a lucky instant.

    THE REGRESSION THIS PINS.  Window edges were originally set from nine
    sampled instants, and COME's real extension peaks at 0.948 against a
    then-ceiling of 0.97 with a 0.10 margin - scoring 0.22, under the bar, twice
    per beckon.  Every template must clear the bar on EVERY steady tick.
    """
    for gesture, template_name in (
            (COME, "COME"), (STOP, "STOP"), (BACK_UP, "BACK_UP"),
            (WAVE, "WAVE"), (POINT_LEFT_ARM, "POINT_LEFT_ARM"),
            (POINT_RIGHT_ARM, "POINT_RIGHT_ARM")):
        template = BY_TEMPLATE[template_name]
        worst = 1.0
        for index in range(0, 90, 3):
            elapsed = RAISE_S + MOTION_WINDOW_S + index * 0.02
            travel, wander = poser.motion(gesture, elapsed, SPAN)
            pose = poser.measure(gesture, elapsed, SPAN)
            worst = min(worst, score(template, pose, travel, wander))
        assert worst >= MIN_CONFIDENCE, (
            f"{gesture} fell to {worst:.3f} during its own hold, below the "
            f"{MIN_CONFIDENCE} bar - the confirm window would reset forever")


# -- the animation envelope ------------------------------------------------------
def test_envelope_rises_and_falls_within_the_span():
    assert envelope(0.0, 5.0) == pytest.approx(0.0, abs=1e-6)
    assert envelope(RAISE_S, 5.0) == pytest.approx(1.0, abs=1e-6)
    assert envelope(5.0, 5.0) == pytest.approx(0.0, abs=1e-6)


def test_confirm_window_fits_inside_every_scripted_hold():
    """A gesture too brief to confirm would be refused by the clock.

    The wrong-person and partial claims both depend on the refusal being about
    identity and measurement rather than about a gesture being too short, so
    every scripted gesture must be comfortably longer than the confirm window
    plus the raise it takes to get there.
    """
    from gest_script import CUES
    from gest_states import CONFIRM_S

    for cue in CUES:
        usable = cue.span_s - RAISE_S - MOTION_WINDOW_S
        assert usable >= CONFIRM_S, (
            f"{cue.person}'s {cue.gesture} holds {cue.span_s:.1f}s, leaving "
            f"{usable:.2f}s of readable window against a {CONFIRM_S:.2f}s "
            "confirm requirement")
