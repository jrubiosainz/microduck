#!/usr/bin/env python3
"""Reading an arm from WORLD GEOMETRY: the six keypoints and what they mean.

This is the PERCEPTION side, and it shares nothing with :mod:`gest_arm` except
the model those joint angles moved.  The features below are computed from the
real world positions MuJoCo produced for the shoulder, elbow and hand bodies -
the same positions the camera ray-casts against - so a sign error in the
animation shows up as a gesture that fails to classify, never as one that
classifies wrongly.

WHAT IS REAL HERE AND WHAT IS A PROXY, STATED ONCE
----------------------------------------------------
* **Real:** the arm is a genuine three-link kinematic chain in the MuJoCo model.
  Every keypoint position is forward kinematics on real joint angles.  Whether
  the duck can SEE each keypoint is real camera geometry - frustum containment
  plus an occlusion ray cast through actual scene geometry.
* **A semantic proxy:** *which* bodies are the shoulder, elbow and hand comes
  from the simulator by body name, not from an RGB keypoint detector, and *which*
  person is the instructor comes from body identity.  So this module stands in
  for a pose estimator, and the classifier above it stands in for a gesture
  recogniser.  Both are labelled as proxies everywhere they surface.

THE FEATURES ARE EXPRESSED IN THE PERSON'S OWN FRAME
------------------------------------------------------
``forward`` and ``lateral`` are components along the person's facing and their
own left, which is what makes POINT_LEFT and POINT_RIGHT genuine mirror images
rather than two labels.  Using the DUCK's frame instead would make the same
physical gesture read differently depending on where the robot happened to be
standing, which is exactly the confusion the behavior has to avoid: the
instructor's "left" is a property of the instructor.

Reading the person's facing is part of the pose proxy and is stated as such.

EVERY LENGTH IS NORMALISED BY THE PERSON'S OWN ARM SPAN
--------------------------------------------------------
``extension``, ``forward`` and ``lateral`` are all divided by the arm span that
person's stature gives them, so a taller distractor and a shorter instructor
produce the same numbers for the same pose.  A classifier tuned on absolute
metres would silently key on stature, which would be a body-identity cue
smuggled in as a gesture feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gest_cast import BY_NAME

# An arm counts as RAISED when its hand sits above this elevation relative to
# the shoulder.  MEASURED: an arm hanging at rest with the idle sway reports
# about -80 deg, and the lowest raised template (BACK_UP at the bottom of its
# push cycle) reports about +12 deg.  The bar sits far from both.
RAISED_ELEVATION_DEG = -25.0


@dataclass(frozen=True)
class ArmPose:
    """One arm's measured pose, in the person's own normalised frame."""

    side: str                 # "l" | "r"
    extension: float          # |hand - shoulder| / arm span, 0..1
    elevation_deg: float      # angle of the hand above the shoulder plane
    forward: float            # component along the person's facing / arm span
    lateral: float            # component along the person's LEFT / arm span
    shoulder: tuple[float, float, float]
    elbow: tuple[float, float, float]
    hand: tuple[float, float, float]

    @property
    def raised(self) -> bool:
        return self.elevation_deg >= RAISED_ELEVATION_DEG

    def as_record(self) -> dict:
        return {
            "side": self.side,
            "extension": round(float(self.extension), 4),
            "elevation_deg": round(float(self.elevation_deg), 2),
            "forward": round(float(self.forward), 4),
            "lateral": round(float(self.lateral), 4),
            "raised": bool(self.raised),
        }


@dataclass(frozen=True)
class BodyPose:
    """Both arms of one person, plus which of them the camera could read."""

    name: str
    yaw: float
    left: ArmPose
    right: ArmPose
    left_readable: bool
    right_readable: bool

    @property
    def raised_arms(self) -> tuple[ArmPose, ...]:
        return tuple(arm for arm in (self.left, self.right) if arm.raised)

    @property
    def raised_count(self) -> int:
        return len(self.raised_arms)

    def readable(self, side: str) -> bool:
        return self.left_readable if side == "l" else self.right_readable

    @property
    def primary(self) -> ArmPose | None:
        """The raised arm a single-arm template would be read from.

        The HIGHER one when both are raised, so a two-armed pose still has a
        well-defined primary arm and the two-arm templates are separated by an
        explicit arm COUNT rather than by which arm happened to be picked.
        """
        raised = self.raised_arms
        if not raised:
            return None
        return max(raised, key=lambda arm: arm.elevation_deg)

    @property
    def fully_readable(self) -> bool:
        """Could the camera read every arm this pose is being judged on?

        A pose with no raised arm still has to be readable, because "the arm was
        down" is itself a reading.  This is the quantity the confirm gate
        requires, and it is what makes an acceptance depend on the duck having
        SEEN the whole pose rather than on the pose merely existing.
        """
        if self.raised_count == 0:
            return self.left_readable and self.right_readable
        return all(self.readable(arm.side) for arm in self.raised_arms)

    def as_record(self) -> dict:
        return {
            "name": self.name,
            "yaw_deg": round(math.degrees(float(self.yaw)), 2),
            "left": self.left.as_record(),
            "right": self.right.as_record(),
            "left_readable": bool(self.left_readable),
            "right_readable": bool(self.right_readable),
            "raised_count": int(self.raised_count),
            "fully_readable": bool(self.fully_readable),
        }


def _frame(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """The person's own forward and left unit vectors, planar."""
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    return forward, left


def measure_arm(side: str, shoulder, elbow, hand, yaw: float,
                arm_span: float) -> ArmPose:
    """Turn three world keypoints into the normalised features above.

    ``arm_span`` is the person's own fully-straight reach, so ``extension``
    is 1.0 for a straight arm at any stature.
    """
    shoulder = np.asarray(shoulder, dtype=np.float64)
    elbow = np.asarray(elbow, dtype=np.float64)
    hand = np.asarray(hand, dtype=np.float64)
    span_vec = hand - shoulder
    span = float(np.linalg.norm(span_vec))
    forward_axis, left_axis = _frame(yaw)
    elevation = math.degrees(math.asin(
        float(np.clip(span_vec[2] / max(span, 1e-9), -1.0, 1.0))))
    return ArmPose(
        side=side,
        extension=span / max(arm_span, 1e-9),
        elevation_deg=elevation,
        forward=float(span_vec @ forward_axis) / max(arm_span, 1e-9),
        lateral=float(span_vec @ left_axis) / max(arm_span, 1e-9),
        shoulder=tuple(float(v) for v in shoulder),
        elbow=tuple(float(v) for v in elbow),
        hand=tuple(float(v) for v in hand),
    )


def measure_body(name: str, yaw: float, keypoints: dict[str, np.ndarray],
                 readable: dict[str, bool]) -> BodyPose:
    """Both arms of one person from six world keypoints.

    ``keypoints`` is keyed ``"{side}_{joint}"`` for ``side`` in ``l``/``r`` and
    ``joint`` in ``shoulder``/``elbow``/``hand``, exactly as
    :meth:`gest_camera.GestureCamera.arm_keypoints` returns them.
    """
    span = BY_NAME[name].arm_span
    return BodyPose(
        name=name,
        yaw=float(yaw),
        left=measure_arm("l", keypoints["l_shoulder"], keypoints["l_elbow"],
                         keypoints["l_hand"], yaw, span),
        right=measure_arm("r", keypoints["r_shoulder"], keypoints["r_elbow"],
                          keypoints["r_hand"], yaw, span),
        left_readable=bool(readable.get("l", False)),
        right_readable=bool(readable.get("r", False)),
    )
