#!/usr/bin/env python3
"""MEASURED locomotion constants for this scene, and the timing built on them.

Every number here was produced by ``tools/sweep_commands.py`` and
``tools/measure_scan.py`` on THIS scene with THIS model.  Nothing is inherited
from a sibling behavior.

THE MEASUREMENT THAT SHAPED THE WHOLE BEHAVIOR
-----------------------------------------------
**The duck cannot turn on the spot.**  MEASURED over 6 s at ``vx = 0``:

    wz = -0.55  ->  -6.0 deg   (-1.0 deg/s)
    wz = -0.40  ->  -4.5 deg
    wz = +0.40  ->  +3.7 deg
    wz = +0.55  ->  +5.0 deg   (+0.8 deg/s)

A stationary body scan is therefore not merely inelegant, it is impossible: a
five-second sweep would turn the duck about four degrees.  The search has to be
performed by the HEAD, whose yaw joint spans a measured +/-170 deg, while the
locomotion command stays at exactly zero.  That is fortunate rather than
awkward, because the acceptance gate requires exactly zero command while lost —
so the only physically available search is also the only permissible one.

**Holding exact zero really is holding still.**  MEASURED: 10 s of exact-zero
command from STAND drifts 0.0014 m, min trunk z 0.1155, final z 0.1163.  So
"the duck did not move while it was lost" is a claim this policy can actually
support.

**Forward gait onset is a cliff.**  MEASURED over 6 s:

    vx = 0.20 -> 0.010 m   (no gait at all)
    vx = 0.22 -> 0.409 m

Nothing between zero and 0.22 is ever commanded, because a command in that band
appears in the HUD and produces no motion on the floor.

**The yaw axis is strongly asymmetric**, and it carries a right-hand bias:
straight-line ``wz = 0`` runs drift about -6 deg over 6 s.  MEASURED at
``vx = 0.30``: ``wz = -0.16`` gives -9.0 deg/s but ``wz = +0.16`` gives only
+4.0 deg/s.  Each sign therefore gets its own gain, ceiling and dead band, and
the left dead band is set above the bias so a small left command cannot be
swallowed by it.

**Lateral commands are not usable.**  MEASURED at ``vy = -0.28``: 0.184 m left
but 33 deg of unwanted yaw; at ``vy = +/-0.18`` nothing happens at all.  The
controller never emits ``vy``.
"""

from __future__ import annotations

# -- forward speeds (MEASURED, 6 s runs) ------------------------------------
#   vx    net m   speed m/s
#   0.20  0.010   0.002   <- no gait
#   0.22  0.409   0.068
#   0.26  0.587   0.098
#   0.30  0.693   0.116
#   0.34  0.777   0.130
#   0.38  0.898   0.150
#   0.42  1.031   0.172
#   0.46  1.254   0.209
#   0.52  1.513   0.252
VX_ONSET = 0.22
# Following the guardian in the open, matched to her measured 0.175 m/s cruise.
VX_FOLLOW = 0.42
# Closing a gap that has opened up, and the rejoin march.
VX_CLOSE = 0.46
# The last stretch into the standoff band: slow, but comfortably above onset.
VX_SETTLE = 0.28
SPEED_AT_FOLLOW = 0.172
SPEED_AT_CLOSE = 0.209

# -- yaw, per sign (MEASURED at vx=0.30 and vx=0.42 over 3 s) ---------------
# Negative (right) is much the stronger sense and is helped by the policy's own
# right bias; positive (left) must overcome it.
WZ_MAX_RIGHT = 0.55
WZ_MAX_LEFT = 0.55
# Dead bands.  MEASURED at vx=0.30: wz=+0.06 produced -1.0 deg/s (the bias won),
# wz=+0.10 produced +0.8 deg/s, wz=+0.14 produced +2.5 deg/s.  On the right,
# wz=-0.10 already produced -6.7 deg/s.
WZ_MIN_RIGHT = 0.10
WZ_MIN_LEFT = 0.16
KP_YAW_RIGHT = 0.95
KP_YAW_LEFT = 1.45

# -- coasting and drift (MEASURED) -----------------------------------------
# Distance travelled in 1.5 s after the command becomes exactly zero.
COAST_AT_FOLLOW_M = 0.0112
COAST_AT_CLOSE_M = 0.0236
# Drift over 10 s of exact zero from STAND.
ZERO_DRIFT_10S_M = 0.0014

# -- head scan (MEASURED joint range +/-170 deg yaw, +/-90 deg pitch) -------
# The sweep uses a little over half the available yaw either way, which covers
# every bearing the guardian can plausibly reappear on without driving the joint
# into its stop, where it would stick and stop being a sweep.
SCAN_AMPLITUDE_DEG = 104.0
# Sweep rate.  Slow enough that a person crossing the far side of the hall is
# inside the frustum for several consecutive ticks rather than being clipped by
# one fast pass, fast enough to cover the amplitude in about 3.5 s.
SCAN_RATE_DPS = 62.0
SCAN_PITCH_DEG = 6.0
# Rates at which the head tracks a subject once it has one.
TRACK_YAW_RATE_DPS = 10.0
TRACK_PITCH_RATE_DPS = 4.5

# -- the loss gate ----------------------------------------------------------
# How long the guardian must be continuously invisible in the EXACT PiP camera
# before the duck declares itself lost.  Long enough that a single stride of
# somebody crossing the sightline is not a loss; short enough that the duck is
# not still walking blindly seconds after she has gone.  At the follow speed of
# 0.172 m/s the duck covers 0.10 m in this window, which is why the gate also
# requires the command to fall to zero the instant LOST is entered rather than
# at the end of a ramp.
LOSS_CONFIRM_S = 0.60
# How long the guardian must be continuously VISIBLE and identity-confirmed
# before the duck accepts a reacquisition.  Deliberately longer than the loss
# window: a false lock costs far more than a slow one.
REACQUIRE_CONFIRM_S = 0.90
# A candidate must be visible for at least this long before it is scored at all,
# so a body clipped by one sweep tick never becomes a candidate.
CANDIDATE_MIN_S = 0.24
# How long a rejection verdict is displayed and honoured before the duck resumes
# sweeping.  Also the minimum time a REJECT state occupies, so it is a visible
# decision rather than a single-tick flicker.
REJECT_HOLD_S = 0.70
# Once rejected, a candidate is not re-scored for this long.  Without it the
# sweep re-acquires the same look-alike every time it passes over her and the
# rejection count becomes a count of ticks rather than of people.
REJECT_COOLDOWN_S = 6.0

# -- identity thresholds ----------------------------------------------------
# Above this an observation is worth evaluating as a candidate at all.
CANDIDATE_SCORE = 0.55
# At or above this, AND with every feature readable, the duck accepts identity.
ACCEPT_SCORE = 0.90
# How well centred a subject must be for its appearance to be read reliably.
READ_CONE_DEG = 30.0

# -- phase ceilings ---------------------------------------------------------
# Ceilings so a stuck phase fails loudly instead of hanging.
FOLLOW_MAX_S = 30.0
SEARCH_MAX_S = 26.0
REJOIN_MAX_S = 30.0
# Settle window before an arrival counts, chosen against the measured coast.
SETTLE_S = 0.60
ARRIVE_TOLERANCE_M = 0.10

# -- states -----------------------------------------------------------------
STATES = ("FOLLOW", "LOST", "STOP", "SEARCH_SWEEP", "CANDIDATE", "REJECT",
          "REACQUIRED", "REJOIN", "SAFE", "DONE")
# Every state in which the locomotion command must be EXACTLY zero.  This is the
# behavior's central safety claim: the duck never moves while it does not know
# where its guardian is.
STATIONARY_STATES = ("LOST", "STOP", "SEARCH_SWEEP", "CANDIDATE", "REJECT",
                     "REACQUIRED", "SAFE", "DONE")
MOVING_STATES = ("FOLLOW", "REJOIN")
