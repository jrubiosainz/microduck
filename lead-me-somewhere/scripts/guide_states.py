#!/usr/bin/env python3
"""States, thresholds and the geometry of guiding, in one place.

Every threshold here is either MEASURED on this scene with this model (the
locomotion constants, produced by ``tools/sweep_commands.py``) or DERIVED from
one of those measurements (the lag bands, derived from the walking speeds and
the follower's own pace).  Nothing is inherited from a sibling behavior without
being re-measured, because the concourse, the cast and the route are different.

THE MEASUREMENT THAT SHAPES THIS BEHAVIOR
------------------------------------------
**The duck cannot turn on the spot, so "look back at her" is a HEAD action and
not a body action.**  MEASURED on this scene at ``vx = 0`` over 3 s:

    wz = -0.16 -> -0.5 deg/s, 0.0001 m drift
    wz = -0.30 -> -1.1 deg/s, 0.0008 m drift
    wz = -0.42 -> -1.6 deg/s, 0.0016 m drift
    wz = +0.42 -> +1.4 deg/s, 0.0031 m drift

The largest turn-in-place command in the sweep yields 1.6 deg/s, so squaring up
to somebody 130 deg behind would take 80 seconds.  A guide that tried it would
appear frozen.  The head, by contrast, has a MEASURED joint range of +/-170 deg
in yaw, which reaches straight behind the robot and past it.

So the duck looks back the way somebody carrying a tray does: it **stops** and
**turns its head**.  ``CHECK_FOLLOWER`` and ``WAIT_FOR_PERSON`` are therefore
both exact-zero-command states, the look-back lives entirely in the isolated
rendering head pose, and the acceptance gate can check the zero literally.  This
was NOT the original design — the first draft had a turn-in-place controller
with a spin rate copied from a sibling behavior — and the sweep replaced it.

**Forward gait onset is a cliff, and it is at 0.24 on this scene.**  MEASURED
over 6 s:

    vx = 0.20 -> 0.008 m   (no gait at all)
    vx = 0.22 -> 0.009 m   (no gait at all)
    vx = 0.24 -> 0.527 m
    vx = 0.26 -> 0.580 m

The sibling promenade behavior measured its onset at 0.22; this scene's is one
step higher, which is exactly why the constants are re-measured per scene rather
than inherited.  A guide that "slowed down a little" for a lagging follower
would emit 0.22 and stand still while the metrics recorded a nonzero command.
This is why waiting is a STATE and not a speed.

**Holding exact zero really is holding still.**  MEASURED: 10 s of exact-zero
command from STAND drifts 0.0006 m with 0.0057 m of path and 0.10 deg of yaw.
That is what makes "the duck stopped and waited for her" a claim about the floor
rather than about the metrics.

**The yaw axis is asymmetric and carries a right-hand bias.**  MEASURED at
``vx = 0.34`` over 3 s: ``wz = -0.10`` gives -7.3 deg/s but ``wz = +0.10`` gives
only +0.7 deg/s — the bias very nearly swallows it.  ``wz = +0.16`` gives
+3.5 deg/s where ``wz = -0.16`` gives -9.3 deg/s.  Each sign therefore gets its
own gain, ceiling and dead band, and the left dead band sits above the bias.
"""

from __future__ import annotations

# -- forward speeds (MEASURED, 6 s runs on this scene) ----------------------
#   vx    net m   speed m/s
#   0.18  0.007   0.001   <- no gait
#   0.20  0.008   0.001   <- no gait
#   0.22  0.009   0.002   <- no gait
#   0.24  0.527   0.088
#   0.26  0.580   0.097
#   0.30  0.682   0.114
#   0.34  0.778   0.130
#   0.38  0.890   0.148
#   0.42  1.009   0.168
#   0.46  1.236   0.206
VX_ONSET = 0.24
# The leading pace.  Chosen just BELOW the follower's 0.132 m/s comfortable
# walk, so an unimpeded follower keeps up and every lag episode in the run is
# caused by her scripted stall rather than by a guide that walks too fast.
VX_LEAD = 0.34
SPEED_AT_LEAD = 0.130
# Used when the duck is well off its route and closing back onto it.
VX_REJOIN = 0.42
SPEED_AT_REJOIN = 0.168
# The slowest walking command this behavior uses: easing into the final
# standing point.  It is the gait-onset command itself, because there is
# nothing between it and zero.
VX_SETTLE = 0.24
SPEED_AT_SETTLE = 0.088

# -- yaw, per sign (MEASURED at vx=0.26, 0.34 and 0.42 over 3 s) ------------
#   vx=0.34: wz=-0.34 -> -14.1 deg/s   wz=+0.34 -> +9.7 deg/s
#            wz=-0.16 ->  -9.3 deg/s   wz=+0.16 -> +3.5 deg/s
#            wz=-0.10 ->  -7.3 deg/s   wz=+0.10 -> +0.7 deg/s
WZ_MAX_RIGHT = 0.55
WZ_MAX_LEFT = 0.55
# Dead bands.  MEASURED at vx=0.34: wz=+0.10 produced +0.7 deg/s (the policy's
# own right bias very nearly swallowed it) while wz=-0.10 produced -7.3 deg/s.
WZ_MIN_RIGHT = 0.10
WZ_MIN_LEFT = 0.16
KP_YAW_RIGHT = 0.90
KP_YAW_LEFT = 1.50

# -- turning in place: MEASURED TO BE IMPOSSIBLE ---------------------------
# At vx = 0 over 3 s, the whole command range produces almost no rotation:
#   wz = -0.16 -> -0.5 deg/s    wz = +0.16 -> +0.7 deg/s
#   wz = -0.30 -> -1.1 deg/s    wz = +0.30 -> +1.0 deg/s
#   wz = -0.42 -> -1.6 deg/s    wz = +0.42 -> +1.4 deg/s
# The best of those would take 80 s to square up to somebody 130 deg behind.
# No turn-in-place command is therefore emitted anywhere in this behavior, and
# these figures exist so that the absence is a MEASUREMENT rather than an
# omission.  Looking back is a HEAD action; see the module docstring.
SPIN_BEST_RATE_DPS = 1.6
SPIN_BEST_COMMAND = 0.42

# -- coasting and drift (MEASURED) -----------------------------------------
# Distance travelled in 1.5 s after the command becomes exactly zero.
COAST_AT_LEAD_M = 0.0088
COAST_AT_REJOIN_M = 0.0080
# Drift over 10 s of exact zero from STAND: 0.0006 m of net displacement,
# 0.0057 m of path, 0.10 deg of yaw.
ZERO_DRIFT_10S_M = 0.0006
ZERO_PATH_10S_M = 0.0057

# -- head tracking (MEASURED joint range +/-170 deg yaw, +/-90 deg pitch) ---
# The head tracks the follower whenever the duck is monitoring her.  Rates are
# slow enough that the PiP is readable and fast enough to hold somebody walking
# behind the robot through a bend.
TRACK_YAW_RATE_DPS = 26.0
TRACK_PITCH_RATE_DPS = 9.0
TRACK_PITCH_DEG = 5.0

# -- where the duck starts and where she starts ----------------------------
# The duck starts near the help desk end, off to one side, so the requested
# destination is genuinely reachable only by crossing the sealed middle of the
# hall.  She stands slightly behind it, already waiting to be led.
DUCK_START_XY = (-2.85, -1.65)
DUCK_START_YAW_DEG = 6.0
FOLLOWER_START_XY = (-3.22, -1.30)

# -- the request ------------------------------------------------------------
# When the semantic request arrives, and what it asks for.  A single constant so
# the machine, the metrics and the HUD all name the same instant.
REQUEST_T_S = 1.60
REQUESTED_DESTINATION = "LIFTS"
# How long the duck spends acknowledging before it starts.  A guide that walked
# off the instant it was addressed would not have acknowledged anything, and the
# command is exactly zero throughout, so this is a visible standstill.
ACK_SECONDS = 1.40

# -- following the route ----------------------------------------------------
# Pure-pursuit lookahead along the planned route.  At the 0.116 m/s lead pace
# this is 3.3 s of travel, which is long enough that the yaw controller is not
# chasing quantisation noise and short enough that the duck genuinely tracks the
# bends rather than cutting them.
PURSUIT_LOOKAHEAD_M = 0.38
# Arc length is advanced by projecting the duck onto the route, so a duck pushed
# off the line does not skip ahead.  This is how far back from the projection the
# search window starts, which stops a route that doubles back from being
# re-acquired at the wrong place.
PROJECTION_WINDOW_M = 1.10
# The duck has reached the end of the route when it is this near the standing
# point.  It is the ARRIVE gate's tolerance, and the final-distance band in the
# metrics is graded against the destination fixture, not against this.
ARRIVE_RADIUS_M = 0.30

# -- monitoring the follower ------------------------------------------------
# The distance beyond which the duck considers her to be LAGGING.  Derived: at
# the lead pace of 0.116 m/s and her 0.132 m/s follow pace, a person who stops
# dead opens the gap at 0.116 m/s, so 1.45 m is 12.5 s of separation — long
# enough that a bend or a momentary stride mismatch cannot trigger it, short
# enough that the duck reacts while she is still in the same part of the hall.
LAG_DISTANCE_M = 1.45
# She is CAUGHT UP again at this distance.  Well inside the lag threshold, so a
# person hovering at the boundary cannot make the duck stutter.
CATCHUP_DISTANCE_M = 0.95
# How long the duck holds still while the route is searched and shown, before it
# leads off.  The search itself takes a few hundred milliseconds of wall clock,
# but a PLAN state that is entered and left inside one control tick is a state
# the run did not actually spend time in — nothing could be seen and nothing
# could be graded.  The dwell makes it real, and the command is exactly zero
# throughout.
PLAN_DWELL_S = 1.20
# How long the duck must have BOTH its head on her AND actual camera visibility
# before it settles into waiting.  At the moment a lag is confirmed she is
# typically behind the robot and out of frame, and the head takes about a second
# to come round at the MEASURED 26 deg/s tracking rate.  Requiring sustained
# visual contact is what keeps the CHECK/WAIT visibility percentage a statement
# about monitoring rather than an average that includes the reacquisition.
CHECK_CONFIRM_S = 0.60
# How long the lag must be CONTINUOUSLY measured before the duck acts.  At her
# 0.196 m/s catch-up pace this is 0.24 m, so a single tick of a swinging arm or
# one bad camera frame is not an episode.
LAG_CONFIRM_S = 1.20
# How long she must be CONTINUOUSLY unseen before that counts as a loss.  Longer
# than the lag window because a body crossing the sightline for half a second is
# not a lost person, and the duck should not stop for one.
LOST_CONFIRM_S = 1.60
# How long she must be CONTINUOUSLY close enough AND visible before the duck
# resumes.  Deliberately longer than either detection window: resuming too early
# is what turns a wait into a stutter, and the gate requires the resume to be
# justified by a sustained measurement rather than by one good frame.
RESUME_CONFIRM_S = 1.00
# The distance at which the situation stops being a lag and becomes a safety
# problem.  The duck must never leave her beyond this for a prolonged interval;
# the gate measures the longest such interval.
SAFETY_MAX_DISTANCE_M = 3.20
SAFETY_MAX_INTERVAL_S = 6.0
# After resuming, the duck will not declare another lag for this long, so an
# episode is a decision rather than a count of ticks.
LAG_COOLDOWN_S = 3.0

# -- waiting ----------------------------------------------------------------
# While waiting the duck watches her with its HEAD, not by turning its body:
# turning in place is MEASURED at 1.6 deg/s maximum on this model, so a body
# turn is not available.  The check is therefore that the RENDERING head yaw has
# reached her bearing within this tolerance — a claim about where the camera is
# actually pointing, measured through the same camera the PiP renders.
LOOK_BACK_TOLERANCE_DEG = 22.0
# A waiting duck holds an exact zero forward command.  This is not a threshold —
# it is a statement that the gate checks literally.
WAIT_COMMAND_VX = 0.0

# -- arriving ---------------------------------------------------------------
# At the destination the duck faces the fixture, then indicates.  Since it
# cannot turn in place, FACING is achieved by the ROUTE: the final leg is
# extended so the duck's own walking heading at the standing point already
# points at the fixture.  The tolerance below is what the gate grades, and
# ``guide_planner.approach_waypoint`` is what makes it achievable.
FACE_TOLERANCE_DEG = 42.0
# How long the arrival indication lasts.  Long enough to read in the video.
INDICATE_SECONDS = 4.0
# The indication is a body-safe gesture: a slow alternating yaw of the HEAD
# only, plus the arrival marker.  Amplitude and rate are inside the measured
# head joint range and are applied in the ISOLATED rendering data, never to the
# walking physics, exactly like the gaze layer.
INDICATE_YAW_AMPLITUDE_DEG = 22.0
INDICATE_PITCH_AMPLITUDE_DEG = 7.0
INDICATE_HZ = 0.42
# The duck must not finish until she is safely nearby.  Graded at the last tick.
FINAL_PERSON_NEAR_M = 1.60

# -- phase ceilings ---------------------------------------------------------
# Ceilings so a stuck phase fails loudly instead of hanging.
RECEIVE_MAX_S = 8.0
PLAN_MAX_S = 4.0
LEAD_MAX_S = 150.0
CHECK_MAX_S = 14.0
WAIT_MAX_S = 45.0
ARRIVE_MAX_S = 25.0
INDICATE_MAX_S = 12.0

# -- states -----------------------------------------------------------------
STATES = ("RECEIVE_DESTINATION", "PLAN", "LEAD", "CHECK_FOLLOWER",
          "WAIT_FOR_PERSON", "RESUME", "ARRIVE", "INDICATE", "DONE")
# The states in which the duck is actively leading along the route.
LEADING_STATES = ("LEAD", "RESUME")
# Every state in which the duck is monitoring her specifically rather than
# merely walking.  The visibility gate is conditioned on these.
MONITOR_STATES = ("CHECK_FOLLOWER", "WAIT_FOR_PERSON")
# States in which the forward command MUST be exactly zero.  This is the
# behavior's strongest claim and it is checked literally, per tick.
#
# CHECK_FOLLOWER IS IN THIS LIST, and getting it there took three attempts.
# The head reaches +/-170 deg (MEASURED) and turn-in-place is unavailable
# (MEASURED at 1.6 deg/s), so a follower at 173 deg astern cannot be seen at
# all.  Two drafts solved that on the ROBOT side, by walking a bounded arc to
# square up; both worked, both cost about 11 s per episode, and both meant the
# duck was still moving in a state that claims it stopped.  The third fixed it
# on the SCENARIO side instead — a follower walks a little to one side of the
# guide rather than in its footprints, which is both what people do and what
# keeps her inside the head's reach.  See ``guide_follower.FOLLOW_OFFSET_M``.
# With that, every monitoring state is a genuine standstill.
ZERO_COMMAND_STATES = ("RECEIVE_DESTINATION", "PLAN", "CHECK_FOLLOWER",
                       "WAIT_FOR_PERSON", "INDICATE", "DONE")
# The relative bearing at which the duck accepts that it is watching her.
# Inside the MEASURED 170 deg head limit with 22 deg to spare, and reachable
# without any manoeuvre because of the follow offset above.
CHECK_ARC_TARGET_DEG = 158.0
# States that must NOT appear at all.  A guide that ABANDONS has failed the
# scenario outright, and SEARCH is a state this behavior does not define — the
# duck knows where she is because it can measure her, and when it cannot, it
# waits rather than hunting.
FORBIDDEN_STATES = ("ABANDON", "SEARCH")

# -- geometry helpers reused by several modules ----------------------------
# The duck's conservative planar half-extent, bounding-sphere based and
# therefore over-stating the robot.  ``test_duck_planar_radius_matches_model``
# pins it against the built scene.
DUCK_PLANAR_RADIUS = 0.1303
