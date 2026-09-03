#!/usr/bin/env python3
"""States, thresholds and the geometry of door and lift etiquette, in one place.

Every threshold here is either MEASURED on this scene with this model (the
locomotion constants, produced by ``tools/sweep_commands.py``) or DERIVED from
one of those measurements.  Nothing is inherited from a sibling behavior without
being re-measured, because the building, the cast and the route are different.

THE MEASUREMENT THAT SHAPED THE BUILDING
------------------------------------------
**The duck cannot turn on the spot.**  MEASURED at ``vx = 0`` over 3 s, across
the whole command range:

    wz = -0.42 -> -1.6 deg/s, 0.0016 m drift
    wz = -0.16 -> -0.5 deg/s, 0.0001 m drift
    wz = +0.16 -> +0.7 deg/s, 0.0017 m drift
    wz = +0.42 -> +1.4 deg/s, 0.0031 m drift

The best turn-in-place command yields **1.6 deg/s**, so squaring up 180 deg
inside a lift car would take **113 seconds**.  A single-entry cabin is therefore
not a scenario this robot can perform at all, and that is why the lift in this
behavior is a **through-car** with doors front and rear: it makes the whole
route monotonically forward.  The building was shaped by the measurement, not
the other way round.  There is no turn-in-place command anywhere in this
behavior, and :data:`SPIN_BEST_RATE_DPS` records the figure so the absence stays
a finding rather than an oversight.

**The turning circle is asymmetric, and the LEFT one is the binding
constraint.**  MEASURED at ``vx = 0.34`` with each sign at its ceiling:
0.353 m travelled in 3 s against 18.9 deg/s turning left, and 0.361 m against
17.4 deg/s turning right, which is a **0.36 m** minimum left-hand radius and a
**0.40 m** right-hand one.  Every bend in this behavior's route is checked
against its own sign's figure by ``tools/check_layout.py``, and the ceiling
itself had to be re-measured before the route would fit: see the yaw block
below.

**Forward gait onset is a cliff, and it is at 0.24 on this scene.**  MEASURED
over 6 s:

    vx = 0.20 -> 0.008 m   (no gait at all)
    vx = 0.22 -> 0.009 m   (no gait at all)
    vx = 0.24 -> 0.525 m
    vx = 0.26 -> 0.578 m

A robot that "edged forward slowly" while somebody came out of a door would emit
0.22, stand still, and log a nonzero command.  **So yielding is a state, not a
speed.**  The duck walks or it holds exactly zero.

**Holding exact zero really is holding still.**  MEASURED: 10 s of exact-zero
command from STAND drifts 0.0006 m with 0.0055 m of path and 0.11 deg of yaw.
That is what makes "it stood still while they came out" and "it did not shuffle
inside the lift" claims about the floor rather than about the metrics.

**The yaw axis carries a right-hand bias that a small left command cannot
overcome.**  MEASURED at ``vx = 0.34``: ``wz = -0.10`` gives -7.9 deg/s but
``wz = +0.10`` gives only +0.9 deg/s.  Each sign therefore carries its own gain,
ceiling and dead band, and the left dead band sits above the bias.  MEASURED at
``wz = 0``: 6 s of straight walking at ``vx = 0.34`` drifts **-13.0 deg**, so
the controller must close the loop on heading even to walk in a straight line
through a 0.66 m doorway.
"""

from __future__ import annotations

import math

# -- forward speeds (MEASURED, 6 s runs on this scene) ----------------------
#   vx    net m   speed m/s
#   0.18  0.007   0.001   <- no gait
#   0.20  0.008   0.001   <- no gait
#   0.22  0.009   0.002   <- no gait
#   0.24  0.525   0.088
#   0.26  0.578   0.096
#   0.30  0.686   0.114
#   0.34  0.776   0.129
#   0.38  0.889   0.148
#   0.42  1.012   0.169
#   0.46  1.245   0.207
VX_ONSET = 0.24
# The cruising command for open floor.
VX_WALK = 0.34
SPEED_AT_WALK = 0.129
# Used inside the cabin and through both apertures: slower, so a heading error
# in a 0.66 m opening costs less lateral travel before the controller corrects
# it.  It is NOT below the onset; there is nothing between the onset and zero.
VX_CAREFUL = 0.26
SPEED_AT_CAREFUL = 0.096
# The slowest walking command this behavior uses: easing into a holding point.
# It is the gait-onset command itself, because there is nothing below it.
VX_SETTLE = 0.24
SPEED_AT_SETTLE = 0.088

# -- yaw, per sign (MEASURED at vx=0.26, 0.34 and 0.42 over 3 s) ------------
#   vx=0.34: wz=-0.34 -> -14.2 deg/s   wz=+0.34 ->  +9.7 deg/s
#            wz=-0.16 ->  -9.3 deg/s   wz=+0.16 ->  +3.5 deg/s
#            wz=-0.10 ->  -7.9 deg/s   wz=+0.10 ->  +0.9 deg/s
#
# THE CEILING WAS RE-MEASURED, AND THAT CHANGED THE ROUTE.
# A first draft capped both signs at 0.55 by inheritance from a sibling
# behavior, which gave a 0.76 m left-hand turning circle and REJECTED every
# bend in this route.  Sweeping further out shows the axis has not saturated
# there at all:
#
#   vx=0.26: wz=+0.42 ->  +9.6   wz=+0.58 -> +13.6   wz=+0.68 -> +16.7 deg/s
#   vx=0.34: wz=+0.42 -> +12.7   wz=+0.58 -> +18.9   wz=+0.68 -> +22.8 deg/s
#   vx=0.34: wz=-0.42 -> -15.7   wz=-0.50 -> -17.4 deg/s
#
# The ceiling is therefore 0.58: the largest command whose measured minimum
# trunk height (0.1130 m at vx=0.34) is indistinguishable from the straight-line
# figure.  0.68 turns faster still and is left on the table deliberately - it is
# the first command in the sweep whose gait degrades, and a behavior about
# walking politely should not ride the edge of its own stability to make a
# corner.
WZ_MAX_RIGHT = 0.58
WZ_MAX_LEFT = 0.58
# Dead bands.  MEASURED: wz=+0.10 produced +0.9 deg/s because the policy's own
# right bias very nearly swallowed it, while wz=-0.10 produced -7.9 deg/s.
WZ_MIN_RIGHT = 0.10
WZ_MIN_LEFT = 0.16
KP_YAW_RIGHT = 0.90
KP_YAW_LEFT = 1.50

# -- the turning circle, DERIVED from the yaw sweep AT THE CEILING ----------
# radius = speed / rate, both taken from the same 3 s run at vx = VX_WALK with
# wz at the ceiling: 0.353 m travelled against 18.9 deg/s turning left, and
# 0.361 m against 17.4 deg/s turning right.  Per SIGN, because they differ by
# 9%, and ``tools/check_layout.py`` refuses any bend that needs more than its
# own sign can deliver.
MIN_LEFT_TURN_RADIUS_M = (0.353 / 3.0) / math.radians(18.9)
MIN_RIGHT_TURN_RADIUS_M = (0.361 / 3.0) / math.radians(17.4)

# -- turning in place: MEASURED TO BE UNAVAILABLE ---------------------------
# The whole command range at vx = 0 produces at most 1.6 deg/s.  No turn-in-place
# command is emitted anywhere in this behavior; these figures exist so the
# absence is a MEASUREMENT rather than an omission, and they are why the cabin is
# a through-car (see the module docstring and ``lobby_layout``).
SPIN_BEST_RATE_DPS = 1.6
SPIN_BEST_COMMAND = 0.42
SPIN_180_SECONDS = 180.0 / SPIN_BEST_RATE_DPS

# -- coasting and drift (MEASURED) -----------------------------------------
# Distance travelled in 1.5 s after the command becomes exactly zero.
COAST_AT_WALK_M = 0.0106
COAST_AT_FAST_M = 0.0075
# Drift over 10 s of exact zero from STAND: 0.0006 m of net displacement,
# 0.0055 m of path, 0.11 deg of yaw.
ZERO_DRIFT_10S_M = 0.0006
ZERO_PATH_10S_M = 0.0055

# -- head tracking (MEASURED joint range +/-170 deg yaw, +/-90 deg pitch) ---
TRACK_YAW_RATE_DPS = 26.0
TRACK_PITCH_RATE_DPS = 9.0
TRACK_PITCH_DEG = 5.0

# -- where the duck starts --------------------------------------------------
# West of the divider, on the approach line, already roughly pointing at the
# door: the duck cannot turn in place, so its start heading has to be walkable
# into the first leg.
DUCK_START_XY = (-3.30, -0.42)
DUCK_START_YAW_DEG = 8.0

# -- following the route ----------------------------------------------------
PURSUIT_LOOKAHEAD_M = 0.38
PROJECTION_WINDOW_M = 1.10
ARRIVE_RADIUS_M = 0.26

# -- the states -------------------------------------------------------------
STATES = (
    "APPROACH_DOOR", "YIELD_EXITERS", "FOLLOW_THROUGH", "APPROACH_LIFT",
    "WAIT_SIDE", "DOORS_OPEN", "LET_OCCUPANTS_EXIT", "FOLLOW_GUARDIAN_IN",
    "POSITION_INSIDE", "RIDE", "DOORS_OPEN_TARGET", "FOLLOW_OUT", "DONE",
)
# States in which the duck is actively walking a leg of its route.
WALKING_STATES = ("APPROACH_DOOR", "FOLLOW_THROUGH", "APPROACH_LIFT",
                  "FOLLOW_GUARDIAN_IN", "POSITION_INSIDE", "FOLLOW_OUT")
# States in which the forward command MUST be exactly zero.  This is the
# behavior's strongest claim and it is checked literally, per tick.
#
# THE RIDE IS IN THIS LIST, and it is the one that matters most: a duck that
# shuffled inside a moving lift would be a duck that could not be trusted in one.
# So is DOORS_OPEN_TARGET: the doors are still travelling and the guardian has
# not moved yet, so anything but a standstill would be pushing past her.
ZERO_COMMAND_STATES = ("YIELD_EXITERS", "WAIT_SIDE", "DOORS_OPEN",
                       "LET_OCCUPANTS_EXIT", "RIDE", "DOORS_OPEN_TARGET",
                       "DONE")
# States in which the duck is watching people rather than merely walking, and
# where the visibility gate is therefore conditioned.
MONITOR_STATES = ("YIELD_EXITERS", "WAIT_SIDE", "DOORS_OPEN",
                  "LET_OCCUPANTS_EXIT")
# States that must NOT appear at all.  PUSH_THROUGH and BOARD_FIRST are the two
# named failures this behavior exists to avoid, and declaring them means a run
# that produced either would fail loudly rather than pass quietly.
FORBIDDEN_STATES = ("PUSH_THROUGH", "BOARD_FIRST")

# -- yielding at the door ---------------------------------------------------
# How far past the aperture plane an exiter must be before it counts as CLEAR of
# the doorway.  Derived from the aperture box's own half-depth plus a body: a
# person one full pace beyond the opening is no longer in it, and a shallower
# figure would let the duck start walking while somebody was still on the sill.
EXITER_CLEAR_M = 0.55
# And how far clear of the duck's own route corridor.  A person who has left the
# doorway but is still standing in the duck's line is still someone to wait for.
EXITER_LATERAL_CLEAR_M = 0.34
# How long every exiter must be CONTINUOUSLY clear before the duck moves off, so
# a single tick of a swinging arm is not a green light.
CLEAR_CONFIRM_S = 0.60
# How long the duck must have held its stop before FOLLOW_THROUGH may begin at
# all.  A yield that lasted two ticks is not a yield.
MIN_YIELD_S = 2.00

# -- following the guardian -------------------------------------------------
# The duck must stay at least this far BEHIND the guardian along its own route,
# measured as arc length, so "it entered behind her" is a claim about the path
# rather than about a camera angle.
MIN_FOLLOW_GAP_M = 0.42
# And no further than this, or it has not followed her through at all.
MAX_FOLLOW_GAP_M = 2.60
# The guardian must be this far through an aperture before the duck may enter
# it, so the two are never inside the same opening.
GUARDIAN_THROUGH_M = 0.30

# -- the lift ---------------------------------------------------------------
# How long the duck holds beside the doors before they open.  The schedule opens
# them at a fixed instant; this is the minimum the gate requires to have elapsed
# with the duck standing there, so "it waited" is a duration and not a frame.
MIN_WAIT_SIDE_S = 2.50
# Minimum number of occupants who must have fully exited before the duck enters.
MIN_OCCUPANTS_EXITED = 2
# How far outside the cabin an occupant must be to count as having exited.
OCCUPANT_EXITED_M = 0.45
# The ride itself: how long the duck must hold exactly still inside the moving
# car.  Long enough to be unambiguous in the video.
MIN_RIDE_S = 8.00
# How near its holding spot inside the cabin the duck must be for POSITION_INSIDE
# to be complete.
CABIN_HOLD_RADIUS_M = 0.30

# -- phase ceilings ---------------------------------------------------------
# Ceilings so a stuck phase fails loudly instead of hanging.  Each one
# TRANSITIONS rather than merely logging: a ceiling that does not move the
# machine is not a ceiling.
APPROACH_DOOR_MAX_S = 40.0
YIELD_MAX_S = 30.0
FOLLOW_THROUGH_MAX_S = 30.0
APPROACH_LIFT_MAX_S = 40.0
WAIT_SIDE_MAX_S = 40.0
DOORS_OPEN_MAX_S = 12.0
LET_OCCUPANTS_EXIT_MAX_S = 30.0
FOLLOW_GUARDIAN_IN_MAX_S = 30.0
POSITION_INSIDE_MAX_S = 25.0
RIDE_MAX_S = 30.0
DOORS_OPEN_TARGET_MAX_S = 12.0
FOLLOW_OUT_MAX_S = 30.0

# -- geometry ---------------------------------------------------------------
# The duck's conservative planar half-extent, MEASURED on this scene's own built
# model at the STAND pose and across the full head-yaw range (the widest sample
# is the same at every head angle).  It is BOUNDING-SPHERE based, so it already
# over-states the robot: the exact planar half-extent is 0.0827 m and the exact
# lateral half-width 0.0710 m.  Using the over-stated figure is the safe
# direction for every zone claim in this behavior at once - a fatter robot finds
# it HARDER to stay out of a passage, harder to fit inside the cabin bounds and
# harder to keep clear of a jamb.  ``test_duck_planar_radius_matches_model``
# pins it against the built scene.
DUCK_PLANAR_RADIUS = 0.1162
# The exact figures, kept beside it so the conservatism is visible rather than
# implied.  ``exact_lateral_half_width`` is what the abreast-budget arithmetic
# uses, because two bodies passing without turning are as wide as their lateral
# half-widths and no wider.
DUCK_EXACT_PLANAR_RADIUS = 0.0827
DUCK_EXACT_LATERAL_HALF_WIDTH = 0.0710
