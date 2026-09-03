#!/usr/bin/env python3
"""MEASURED locomotion constants for this scene, and the timing built on them.

Every number here was produced by ``tools/sweep_commands.py`` on THIS scene with
THIS model.  Nothing is inherited from a sibling behavior.

THE MEASUREMENT THAT SHAPED THE WHOLE BEHAVIOR
-----------------------------------------------
**The duck cannot strafe, so "move over one lane" is not a primitive.**
MEASURED over 6 s at ``vx = 0``:

    vy = -0.22 -> 0.000 m left, 0.5 deg     (no gait at all)
    vy = +0.22 -> 0.004 m left, -2.9 deg    (no gait at all)
    vy = -0.28 -> 0.255 m right, 51.2 deg   (a turn wearing a strafe's clothes)
    vy = +0.34 -> 0.334 m left, -12.8 deg

and at ``vx = 0.30`` a ``vy`` of +/-0.18 still costs 19 deg of unwanted yaw for
0.19-0.29 m of lateral travel.  There is no command that moves this robot
sideways without turning it, so **changing sides has to be a PATH**: fall behind,
cross through a point astern of the guardian, and come up the other side.  The
controller never emits ``vy``, and the state machine's CROSS_BEHIND state exists
because of this measurement rather than for narrative effect.

**Forward gait onset is a cliff.**  MEASURED over 6 s:

    vx = 0.20 -> 0.010 m   (no gait at all)
    vx = 0.22 -> 0.409 m
    vx = 0.24 -> 0.515 m

Nothing between zero and 0.22 is ever commanded, because a command in that band
appears in the metrics and produces no motion on the floor.  This is also why
station-keeping beside a walking person is a walk-or-stand policy with
hysteresis rather than a continuously trimmed servo.

**Holding exact zero really is holding still.**  MEASURED: 10 s of exact-zero
command from STAND drifts 0.0014 m, min trunk z 0.1155, final z 0.1163.

**The yaw axis is strongly asymmetric and carries a right-hand bias.**
MEASURED at ``vx = 0.34`` over 3 s: ``wz = -0.10`` gives -6.3 deg/s but
``wz = +0.10`` gives 0.0 deg/s — the bias swallows it completely.  ``wz = +0.16``
gives only +2.0 deg/s where ``wz = -0.16`` gives -7.8 deg/s.  Each sign therefore
gets its own gain, ceiling and dead band, and the left dead band sits above the
bias so a small left command cannot be swallowed by it.
"""

from __future__ import annotations

# -- forward speeds (MEASURED, 6 s runs) ------------------------------------
#   vx    net m   speed m/s
#   0.20  0.010   0.002   <- no gait
#   0.22  0.409   0.068
#   0.24  0.515   0.086
#   0.26  0.587   0.098
#   0.30  0.693   0.116
#   0.34  0.777   0.130
#   0.38  0.898   0.150
#   0.42  1.031   0.172
#   0.46  1.254   0.209
#   0.52  1.513   0.252
VX_ONSET = 0.22
# Cruise while holding formation, matched just above the guardian's 0.112 m/s so
# a duck exactly in station drifts forward slowly rather than backward.
VX_CRUISE = 0.30
# Closing a longitudinal gap that has opened up, and the crossing itself.
VX_CLOSE = 0.42
# The fastest command this behavior uses: recovering station after a crossover,
# when the guardian has walked on during the manoeuvre.
VX_SPRINT = 0.46
# Easing into the slot, still comfortably above the onset cliff.
VX_SETTLE = 0.24
SPEED_AT_CRUISE = 0.116
SPEED_AT_CLOSE = 0.172
SPEED_AT_SPRINT = 0.209
SPEED_AT_SETTLE = 0.086

# -- yaw, per sign (MEASURED at vx=0.26, 0.34 and 0.42 over 3 s) ------------
# Negative (right) is much the stronger sense and is helped by the policy's own
# right bias; positive (left) must overcome it.
WZ_MAX_RIGHT = 0.55
WZ_MAX_LEFT = 0.55
# Dead bands.  MEASURED at vx=0.34: wz=+0.10 produced 0.0 deg/s (the bias won
# outright), wz=+0.16 produced +2.0 deg/s.  On the right, wz=-0.10 already
# produced -6.3 deg/s.
WZ_MIN_RIGHT = 0.10
WZ_MIN_LEFT = 0.16
KP_YAW_RIGHT = 0.90
KP_YAW_LEFT = 1.50

# -- coasting and drift (MEASURED) -----------------------------------------
# Distance travelled in 1.5 s after the command becomes exactly zero.
COAST_AT_CRUISE_M = 0.0086
COAST_AT_CLOSE_M = 0.0109
COAST_AT_SPRINT_M = 0.0143
# Drift over 10 s of exact zero from STAND.
ZERO_DRIFT_10S_M = 0.0014

# -- head tracking (MEASURED joint range +/-170 deg yaw, +/-90 deg pitch) ---
# The head is aimed at the guardian throughout; there is no search in this
# behavior.  Rates are slow enough that the PiP is readable and fast enough to
# hold a companion walking at arm's length.
TRACK_YAW_RATE_DPS = 26.0
TRACK_PITCH_RATE_DPS = 9.0
TRACK_PITCH_DEG = 5.0

# -- the decision gates -----------------------------------------------------
# How long a side must be CONTINUOUSLY measured as unusable before the duck
# commits to leaving it.  At the 0.42 m/s closing speed of an oncoming walker
# this is 0.34 m of approach: long enough that a single tick of a swinging arm
# is not a blockage, short enough that the duck decides well before the
# encounter.
BLOCK_CONFIRM_S = 0.80
# How long the far side must be CONTINUOUSLY usable before the duck will cross
# to it.  Deliberately longer than the blockage window: crossing is expensive
# and crossing into a lane that is about to close would be worse than staying.
CLEAR_CONFIRM_S = 1.00
# After completing a switch, the duck will not consider another for this long.
# Without it a duck standing on the boundary of two marginal lanes ping-pongs,
# and the "side switches" count becomes a count of ticks rather than decisions.
SWITCH_COOLDOWN_S = 6.0
# How long the duck must hold the formation before a join counts as complete.
JOIN_SETTLE_S = 0.50

# -- phase ceilings ---------------------------------------------------------
# Ceilings so a stuck phase fails loudly instead of hanging.
ACQUIRE_MAX_S = 40.0
JOIN_MAX_S = 30.0
FALL_BACK_MAX_S = 14.0
CROSS_MAX_S = 22.0
JOIN_OTHER_MAX_S = 30.0

# -- states -----------------------------------------------------------------
STATES = ("ACQUIRE", "JOIN_SIDE", "BESIDE_LEFT", "BESIDE_RIGHT", "SIDE_BLOCKED",
          "FALL_BACK", "CROSS_BEHIND", "JOIN_OTHER_SIDE", "BESIDE", "DONE")
# The two states in which the duck is holding a formation it has already joined.
BESIDE_STATES = ("BESIDE_LEFT", "BESIDE_RIGHT")
# Every state in which the duck is walking under its own steam.  There is no
# stationary state in this behavior except DONE: a companion that stops walking
# is no longer walking beside anybody.
MOVING_STATES = ("ACQUIRE", "JOIN_SIDE", "BESIDE_LEFT", "BESIDE_RIGHT",
                 "SIDE_BLOCKED", "FALL_BACK", "CROSS_BEHIND",
                 "JOIN_OTHER_SIDE")
# States that must NOT appear at all.  The scenario runs to its end in
# formation; a rollout that reaches DONE has stopped early, and one that emits a
# HOLD has invented a state this behavior does not define.
FORBIDDEN_STATES = ("HOLD", "DONE")
