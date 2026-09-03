#!/usr/bin/env python3
"""The formation frame: what "beside" MEANS, in numbers.

Single source of truth for the slot geometry, shared by the side chooser, the
controller, the state machine and the acceptance gate, so all four mean the same
thing by "beside" and by "behind".

THE GUARDIAN-RELATIVE FRAME
---------------------------
Everything is expressed in a right-handed frame carried by the guardian:

* ``forward`` is her unit heading;
* ``left`` is ``forward`` rotated +90 deg;
* **lateral** is the duck's offset along ``left`` — positive on her left, and
  its SIGN is the side the duck is on;
* **longitudinal** is the duck's offset along ``forward`` — positive AHEAD of
  her, negative behind.

That single sign convention is what makes every claim in this behavior
checkable.  "The duck joined the left side" is ``lateral > 0``; "the duck fell
behind to cross" is ``longitudinal <= -CROSS_BEHIND_M``; "the duck never cut in
front of her" is ``longitudinal <= FORWARD_HALF_PLANE_M`` at every tick.

WHY THE BAND IS 0.45-0.75 m, JUSTIFIED RATHER THAN CHOSEN
----------------------------------------------------------
The duck's conservative planar half-extent is MEASURED at 0.1303 m from the
built scene (bounding-sphere based, so it over-states the robot; its exact
planar half-extent is 0.0978 m).  An adult's exact planar half-extent in this
cast runs about 0.14 m at pose zero to 0.27 m mid-stride, arms included.

* **0.45 m** leaves roughly 0.45 - 0.098 - 0.27 = 0.082 m of surface gap against
  a mid-stride arm swing at the near edge, using the duck's exact width.
  Positive, but tight enough that the duck reads as genuinely walking WITH her
  rather than near her.
* **0.75 m** is the far edge: beyond it the duck is no longer in formation, it
  is merely in the same corridor.

THOSE ARE SIZING ARGUMENTS, NOT SAFETY GUARANTEES.  The actual safety claim is
the per-tick ``ContactProbe`` measurement of real surface clearance against the
real geoms at the real pose, and it is graded separately.

WHY THE CROSSOVER GOES BEHIND AND NEVER IN FRONT
-------------------------------------------------
A companion that cuts across somebody's forward half-plane to change sides walks
through the space they are about to occupy.  It is the single most antisocial
thing this behavior could do, so it is not merely discouraged — the crossover
target is CONSTRUCTED behind her, the acceptance gate measures the longitudinal
offset at every tick of the crossing, and a run in which the duck passed in
front of her fails regardless of how good the final formation looked.
"""

from __future__ import annotations

import math

import numpy as np

# -- the duck ---------------------------------------------------------------
# Conservative planar half-extent from each geom's BOUNDING SPHERE, which
# over-states the robot.  That is the safe direction for every gate here.
# ``test_the_duck_planar_radius_constant_matches_the_model`` pins it against the
# built scene.
DUCK_PLANAR_RADIUS = 0.1303

# -- the formation band -----------------------------------------------------
# The lateral band the duck must hold while it is beside her.
BESIDE_MIN_M = 0.45
BESIDE_MAX_M = 0.75
BESIDE_TARGET_M = 0.58
# Longitudinal station: slightly behind her shoulder, which is where a companion
# walks.  Zero would put the duck exactly abreast, which looks like a race.
BESIDE_LONG_TARGET_M = -0.12
# The longitudinal error the formation is allowed before it stops counting as
# "beside" at all.  Wider than the target offset because the duck's forward
# speed is quantised by the gait-onset cliff and cannot be trimmed continuously.
BESIDE_LONG_TOLERANCE_M = 0.55

# -- the forward half-plane -------------------------------------------------
# The duck may never be more than this far AHEAD of the guardian's lateral axis.
# Slightly positive rather than exactly zero: the guardian's own arm swing and
# the duck's gait mean an exactly-abreast formation crosses zero by a few
# centimetres every stride, and a gate at exact zero would be measuring the
# stride rather than the behavior.  Any real overtake is far beyond this.
FORWARD_HALF_PLANE_M = 0.22

# -- the crossover ----------------------------------------------------------
# How far behind her the duck must be before it is allowed to start crossing.
# It must clear her trailing leg with margin, so it is measured against her body
# rather than picked: an adult's exact planar half-extent reaches 0.27 m
# mid-stride, and the duck's own exact half-extent is 0.098 m, so 0.62 m leaves
# about 0.25 m of surface gap at the moment of the crossing.
CROSS_BEHIND_M = 0.62
# The longitudinal offset the crossing waypoint itself sits at.  Deeper than the
# entry gate so the duck is still behind her when it reaches the far side, even
# though she keeps walking during the crossing.
CROSS_WAYPOINT_LONG_M = -0.78
# Lateral tolerance for having reached the crossing waypoint.
CROSS_ARRIVE_M = 0.20
# The lateral offset at which the duck is considered to have committed to the
# far side and can start closing back up into formation.
CROSS_COMMIT_M = 0.20

# -- refusal margins --------------------------------------------------------
# A candidate slot must clear every STATIC surface by at least this much.
# Measured against the duck's own bounding radius (0.1303 m on this scene) plus
# a working margin, so a slot the duck could not physically occupy is refused
# before it is ever walked to.
SIDE_STATIC_MARGIN_M = 0.22
# A candidate slot must stay at least this far from any PREDICTED pedestrian
# position over the lookahead window.  Larger than the static margin because a
# person moves and their arms swing.
SIDE_PERSON_MARGIN_M = 0.55
# How far ahead the side chooser predicts pedestrian motion.  At the guardian's
# 0.130 m/s and an oncoming 0.30 m/s the closing speed is 0.43 m/s, so 3.0 s is
# 1.29 m of approach: long enough to decide before the encounter, short enough
# that a linear prediction of a walking person still means something.
SIDE_LOOKAHEAD_S = 3.0
# How many samples the lookahead is evaluated at.
SIDE_LOOKAHEAD_SAMPLES = 7

# -- where the duck starts --------------------------------------------------
# Behind the guardian and off to her right, NOT in either slot.  The initial
# join is therefore a real traverse into formation rather than a spawn already
# in place, and it starts from the side the hedge will force it to reject.
DUCK_START_XY = (-4.65, -2.72)
DUCK_START_YAW_DEG = 24.0


def frame(guardian_xy, guardian_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """The guardian's ``(forward, left)`` unit axes."""
    forward = np.array([math.cos(guardian_yaw), math.sin(guardian_yaw)])
    left = np.array([-math.sin(guardian_yaw), math.cos(guardian_yaw)])
    return forward, left


def relative(duck_xy, guardian_xy, guardian_yaw: float) -> tuple[float, float]:
    """``(lateral, longitudinal)`` of the duck in the guardian's frame.

    ``lateral`` is positive on her LEFT.  ``longitudinal`` is positive AHEAD.
    Every side claim in this behavior is a statement about these two numbers.
    """
    forward, left = frame(guardian_xy, guardian_yaw)
    delta = np.asarray(duck_xy, dtype=np.float64) - np.asarray(
        guardian_xy, dtype=np.float64)
    return float(delta @ left), float(delta @ forward)


def slot_point(guardian_xy, guardian_yaw: float, side: int,
               lateral: float = BESIDE_TARGET_M,
               longitudinal: float = BESIDE_LONG_TARGET_M) -> np.ndarray:
    """World point of the beside slot on ``side`` (+1 left, -1 right)."""
    if side not in (-1, 1):
        raise ValueError(f"side must be +1 (left) or -1 (right), got {side!r}")
    forward, left = frame(guardian_xy, guardian_yaw)
    return (np.asarray(guardian_xy, dtype=np.float64)
            + left * (side * lateral) + forward * longitudinal)


def cross_point(guardian_xy, guardian_yaw: float, side: int) -> np.ndarray:
    """The rear waypoint the duck crosses through, on its way to ``side``.

    Placed BEHIND her by :data:`CROSS_WAYPOINT_LONG_M` and only slightly to the
    target side, so the path bends round her stern instead of sweeping across
    her front.  This construction, not a rule applied afterwards, is what makes
    the crossing rear-going.
    """
    return slot_point(guardian_xy, guardian_yaw, side,
                      lateral=0.5 * BESIDE_TARGET_M,
                      longitudinal=CROSS_WAYPOINT_LONG_M)


def side_of(lateral: float) -> int:
    """+1 if the duck is on her left, -1 on her right.  Zero counts as right."""
    return 1 if lateral > 0.0 else -1


def side_name(side: int) -> str:
    return "left" if side == 1 else "right"


def in_band(lateral_abs: float) -> bool:
    return BESIDE_MIN_M <= lateral_abs <= BESIDE_MAX_M


def band_verdict(lateral_abs: float) -> str:
    if lateral_abs < BESIDE_MIN_M:
        return "too close"
    if lateral_abs > BESIDE_MAX_M:
        return "too far"
    return "in band"


def formation_ok(lateral: float, longitudinal: float, side: int) -> bool:
    """Is the duck genuinely in the beside slot on ``side`` right now?

    Requires the correct SIDE, a lateral offset inside the band, and a bounded
    longitudinal error.  The longitudinal term is what stops a duck trailing two
    metres behind her from claiming to be walking beside anybody, which is the
    failure mode this whole behavior exists to avoid.
    """
    if side_of(lateral) != side:
        return False
    if not in_band(abs(lateral)):
        return False
    return abs(longitudinal - BESIDE_LONG_TARGET_M) <= BESIDE_LONG_TOLERANCE_M


def crossed_forward_half_plane(longitudinal: float) -> bool:
    """Did the duck get ahead of the guardian's lateral axis?"""
    return longitudinal > FORWARD_HALF_PLANE_M
