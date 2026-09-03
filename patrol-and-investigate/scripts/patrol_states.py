#!/usr/bin/env python3
"""States, thresholds and the measured locomotion contract, in one place.

Every constant here is either MEASURED on THIS scene with THIS model (by
``tools/sweep_commands.py``) or DERIVED from one of those measurements.  Nothing
is inherited from a sibling behavior without being re-measured, because the
floor, the route and the task are different.

THE MEASUREMENT THIS BEHAVIOR TURNS ON: A CHECKPOINT STOP IS A STATE
----------------------------------------------------------------------
Forward gait onset on this scene is a CLIFF, not a ramp.  MEASURED over 6 s:

    vx = 0.20 -> 0.008 m   (no gait at all)
    vx = 0.22 -> 0.009 m   (no gait at all)
    vx = 0.24 -> 0.508 m
    vx = 0.26 -> 0.576 m
    vx = 0.34 -> 0.770 m

There is NOTHING between zero and a walk.  A guard robot that "slowed to a
crawl" at each checkpoint would emit 0.22, stand perfectly still, and log a
nonzero command - the appearance of care with none of the physics.  So every
checkpoint stop, every observation hold and every classification in this
behavior is an EXACT ZERO, and the gate checks that literally, per tick.

**Holding exact zero really is holding still.**  MEASURED: 10 s of exact-zero
command drifts 0.0006 m with 0.0054 m of path and 0.11 deg of yaw.  That is what
makes "it stopped at the checkpoint and scanned" a claim about the floor.

TURNING IN PLACE IS UNAVAILABLE, WHICH SHAPES THE WHOLE ROUTE
---------------------------------------------------------------
MEASURED at ``vx = 0`` across the whole command range: at most **1.6 deg/s**.
The duck cannot pivot to face a checkpoint, cannot spin to scan, and cannot turn
round to go back.  Three consequences:

1. the patrol circuit is a HEXAGON, whose 60 deg corners the duck carves while
   walking at its MEASURED yaw ceiling;
2. a checkpoint scan is done with the HEAD, not the body - the head yaw joint
   has a MEASURED +/-170 deg range, so a stopped duck can still sweep its camera
   across the facility;
3. returning to an interrupted checkpoint is a real WALK back along a real
   route, not a pivot - which is what makes route memory a physical claim.

THE YAW AXIS IS ASYMMETRIC AND BIASED RIGHT, AND THE CIRCUIT TURNS THE WEAK WAY
--------------------------------------------------------------------------------
MEASURED at ``vx = 0.34`` over 3 s: ``wz = -0.10`` gives -8.7 deg/s while
``wz = +0.10`` gives only +0.7 deg/s - the policy's own right bias very nearly
swallows a small left command.  Each sign carries its own gain, ceiling and dead
band, and the left dead band sits above the bias.  MEASURED at ``wz = 0``: 6 s of
straight walking at ``vx = 0.34`` drifts **-16.7 deg**, so the heading loop must
be closed even to walk a straight leg.

THE PATROL CIRCUIT RUNS COUNTER-CLOCKWISE, WHICH MEANS EVERY CORNER IS A LEFT
TURN INTO THE WEAK SIGN AND AGAINST THE BIAS.  That is deliberate: a clockwise
loop would have the policy's own drift doing the turning, and "the duck walked
its circuit" would be partly a fact about the policy rather than about the
controller.  MEASURED at the ceiling, the left sign still delivers +19.7 deg/s,
which carries a 60 deg hexagon corner in about 3.0 s of walking.
"""

from __future__ import annotations

# -- forward speeds (MEASURED, 6 s runs on this scene) ----------------------
#   vx    net m   speed m/s   yaw drift deg
#   0.18  0.007   0.001   <- no gait
#   0.20  0.008   0.001   <- no gait
#   0.22  0.009   0.002   <- no gait
#   0.24  0.508   0.085        -9.4
#   0.26  0.576   0.096       -18.1
#   0.30  0.677   0.113       -16.9
#   0.34  0.770   0.128       -16.7
#   0.38  0.874   0.146       -20.8
#   0.42  1.087   0.181        +8.6
#   0.46  1.247   0.208       +18.4
VX_ONSET = 0.24
# The patrol cruise: the command the duck walks its circuit at.
#
# 0.38 rather than 0.34, and the choice is a MEASURED trade against the yaw the
# hexagon needs.  At 0.38 the duck covers 0.146 m/s and the LEFT yaw ceiling -
# the weak sign, which every corner of this counter-clockwise circuit turns into
# - still delivers +19.3 deg/s for a 0.536 m turning radius, comfortably inside
# the 0.86 m circuit.  0.42 was rejected on its own measurement rather than on
# taste: there the open-loop yaw drift REVERSES SIGN, from -20.8 deg to +8.6 deg
# over 6 s, and a gait whose heading bias flips direction with speed is one the
# heading loop has to fight rather than trim.
VX_PATROL = 0.38
SPEED_AT_PATROL = 0.146
# The approach command.  Slower, because closing on an unidentified object is
# where a heading error costs the most, and because a slower approach lands
# inside the standoff band rather than through it.  NOT below the onset; there
# is nothing between the onset and zero.
VX_APPROACH = 0.26
SPEED_AT_APPROACH = 0.096  # MEASURED 0.576 m in 6 s
# The slowest walking command this behavior uses: easing into a checkpoint or a
# standoff.  It is the gait-onset command itself, because there is nothing
# below it.
VX_SETTLE = 0.24
SPEED_AT_SETTLE = 0.085  # MEASURED 0.508 m in 6 s

# -- yaw, per sign (MEASURED at vx=0.26, 0.34, 0.42 over 3 s) ---------------
#   vx=0.34: wz=-0.34 -> -13.9 deg/s   wz=+0.34 -> +10.4 deg/s
#            wz=-0.22 -> -12.5 deg/s   wz=+0.22 ->  +6.9 deg/s
#            wz=-0.16 -> -10.3 deg/s   wz=+0.16 ->  +4.0 deg/s
#            wz=-0.10 ->  -8.7 deg/s   wz=+0.10 ->  +0.7 deg/s
#
# THE CEILING WAS SWEPT PAST ITS OWN EDGE.  At 0.58 both signs are still
# responding almost linearly and the gait is untouched (min trunk z 0.1130 m):
#
#   vx=0.34: wz=-0.42 -> -15.7   wz=-0.50 -> -17.9   wz=-0.58 -> -20.6 deg/s
#            wz=+0.42 -> +13.7   wz=+0.50 -> +16.5   wz=+0.58 -> +19.7 deg/s
#            wz=-0.68 -> -25.3   wz=-0.75 -> -29.1
#            wz=+0.68 -> +23.1   wz=+0.75 -> +25.8
#
# 0.58 is the ceiling this behavior uses.  MEASURED at the patrol cruise of
# ``vx = 0.38`` it delivers -26.2 deg/s right and +19.3 deg/s left, for turning
# radii of 0.393 m and 0.536 m - both inside the 0.86 m circuit radius, which is
# what makes the hexagon walkable at all.  0.68 and 0.75 turn faster still and
# are left on the table deliberately: their measured minimum trunk height is
# indistinguishable, but a patrol is about walking a known route accurately, and
# the extra rate buys a tighter corner at the cost of a visibly harsher gait in
# a video whose whole subject is a robot moving deliberately.
WZ_MAX_RIGHT = 0.58
WZ_MAX_LEFT = 0.58
# Dead bands.  MEASURED: wz=+0.10 produced +0.7 deg/s because the policy's own
# right bias very nearly swallowed it, while wz=-0.10 produced -8.7 deg/s.
WZ_MIN_RIGHT = 0.10
WZ_MIN_LEFT = 0.16
KP_YAW_RIGHT = 0.90
KP_YAW_LEFT = 1.50

# -- the turning circle, DERIVED from the ceiling sweep ---------------------
# radius = (path / t) / rate, both from the same 3 s run at the patrol cruise
# ``vx = 0.38`` with wz at the ceiling: 0.393 m turning right, 0.536 m turning
# left.  Per SIGN, because they differ by 36 %.  Both are inside the 0.86 m
# circuit radius, which is what makes the hexagon walkable.
MIN_RIGHT_TURN_RADIUS_M = 0.393
MIN_LEFT_TURN_RADIUS_M = 0.536

# -- turning in place: MEASURED TO BE UNAVAILABLE ---------------------------
# The whole command range at vx = 0 produces at most 1.6 deg/s, and 0.0031 m of
# drift.  No turn-in-place command is emitted anywhere in this behavior; these
# figures exist so the absence is a MEASUREMENT rather than an omission, and
# they are why a checkpoint scan is done with the HEAD.
SPIN_BEST_RATE_DPS = 1.6
SPIN_BEST_COMMAND = 0.42

# -- coasting and drift (MEASURED) -----------------------------------------
# Walk 4 s, then hold exact zero for 1.5 s.
COAST_AT_PATROL_M = 0.0091   # from vx = 0.30
COAST_AT_FAST_M = 0.0161     # from vx = 0.42
# Drift over 10 s of exact zero: 0.0006 m of net displacement, 0.0054 m of path,
# +0.11 deg of yaw.  Over 3 s: 0.0003 m and 0.0013 m.
ZERO_DRIFT_10S_M = 0.0006
ZERO_PATH_10S_M = 0.0054

# -- the head (MEASURED joint range +/-170 deg yaw, +/-90 deg pitch) --------
# The head is the ONLY thing that can sweep while the body is stopped, so the
# scan rate is the rate this whole behavior's observation states run at.
TRACK_YAW_RATE_DPS = 26.0
TRACK_PITCH_RATE_DPS = 9.0
TRACK_PITCH_DEG = 4.0
# A checkpoint scan sweeps this far either side of the checkpoint's own outward
# watch bearing.  MEASURED against the head yaw limit (+/-170 deg) and the
# camera's own 58 deg vertical FOV: +/-46 deg of sweep plus a 34 deg half-width
# frustum covers a 160 deg arc of the facility from a standstill.
SCAN_SWEEP_DEG = 46.0
# The arc the head must actually TRAVEL for a sweep to count as complete: out to
# one extreme and back across to the other, which is ``3 x SCAN_SWEEP_DEG``.
# DERIVED from the sweep rather than chosen, and measured on the pose the head
# actually reached - so a sweep cut short by a detection reports the shorter arc
# and is not counted as a complete scan.
SCAN_ARC_COMPLETE_DEG = 3.0 * SCAN_SWEEP_DEG
# The time that arc takes at the MEASURED 26 deg/s head rate, kept as the
# fallback bound so a scan cannot run forever if the head is rate-limited.
SCAN_PERIOD_S = 4.0 * SCAN_SWEEP_DEG / TRACK_YAW_RATE_DPS

# -- following the route ----------------------------------------------------
PURSUIT_LOOKAHEAD_M = 0.34
# Within this of a checkpoint the duck eases in, so it stops ON it rather than
# walking through it.  DERIVED from the MEASURED 0.0091 m coast plus the settle
# command's own 0.085 m/s.
SETTLE_REMAINING_M = 0.28
# The duck has ARRIVED at a checkpoint when its trunk is this near it.
CHECKPOINT_ARRIVE_M = 0.20
# The measured ground speed below which the duck counts as STOPPED.  DERIVED
# from the MEASURED zero-command drift: 0.0006 m over 10 s is 0.00006 m/s, so
# anything under 0.01 m/s is the gait having unwound rather than the robot still
# walking.  The gait-onset cliff means there is no genuine walking speed below
# 0.085 m/s to confuse it with.
SETTLED_MPS = 0.010

# -- the safe observation standoff ------------------------------------------
# THE BAND THE DUCK MUST STOP IN WHEN IT APPROACHES AN ANOMALY.  It is a REQUIRED
# band, not a measured one: close enough that the camera resolves the target,
# far enough that the robot never makes contact and a person is not crowded.
STANDOFF_MIN_M = 0.45
STANDOFF_MAX_M = 0.75
# What the approach controller aims for: the middle of the band, so the MEASURED
# stopping error has room on both sides.  DERIVED from the MEASURED 0.0091 m
# coast: aiming at the centre leaves 0.15 m of margin against a coast an order
# of magnitude smaller.
STANDOFF_TARGET_M = 0.60
# The approach STOPS when the measured range is inside the band.  Checked on
# MEASURED range every tick, independently of the planned standoff point, so a
# standoff point that was badly placed still cannot produce a close approach.
STANDOFF_STOP_M = 0.68

# -- observation ------------------------------------------------------------
# A multi-angle observation is a HOLD at each of several bearings on the target.
# The duck cannot orbit it - that would be minutes of walking - so the angles are
# swept by the HEAD from the standoff point, which is what a guard robot with a
# pan-tilt camera actually does.  Three bearings, each held long enough for the
# sweep to arrive and settle.
OBSERVE_ANGLES_DEG: tuple[float, ...] = (-26.0, 0.0, 26.0)
# DERIVED from the MEASURED 26 deg/s head yaw rate: 26 deg between adjacent
# angles is 1.0 s of travel, so 2.2 s per angle leaves the camera settled and
# pointing for more than half of each hold.
OBSERVE_HOLD_S = 2.2
OBSERVE_TOTAL_S = OBSERVE_HOLD_S * len(OBSERVE_ANGLES_DEG)
# How long the classification is held.  Long enough to be read in the video.
CLASSIFY_S = 2.2
# How long a checkpoint stop lasts before the scan begins, and how long the
# result is shown after it.
CHECKPOINT_STOP_S = 1.4
CHECKPOINT_RESULT_S = 1.2

# -- detection --------------------------------------------------------------
# THE CAMERA GATE.  An anomaly may be detected ONLY while the duck can actually
# see it through the real head camera - frustum containment plus a real MuJoCo
# occlusion ray cast.  This is the constant that makes detection a perception
# claim rather than a schedule.
DETECT_MAX_RANGE_M = 2.60
# And it must stay visible for this long before the duck acts, so a single
# frame of a body clipping the frustum edge is not a detection.  DERIVED from
# the MEASURED head sweep rate: at 26 deg/s a target crossing the 68 deg frustum
# is inside it for 2.6 s, so 0.40 s is a real dwell rather than a graze.
DETECT_CONFIRM_S = 0.40
# An unattended object is SUSPICIOUS when it has been stationary this long
# outside a designated stow area.  A number the duck measures for itself from
# its own successive observations of the object.
#
# 5.0 s rather than the traffic's own crossing time, because the rule this bar
# protects is "nobody has come back for it", and the MEASURED evidence the duck
# has by the time it acts is far stronger than the bar: the crate has stood
# untouched since it appeared, with the nearest person 2.58 m away.  A bar the
# evidence only just clears would make the confidence proxy report 0.50 - a coin
# flip - for a case that is not close.
UNATTENDED_S = 5.0
# How near a person must be to an object for it to count as attended.
ATTENDED_RADIUS_M = 0.90

# -- the states -------------------------------------------------------------
STATES = (
    "PATROL", "CHECKPOINT_STOP", "SCAN", "CLEAR", "DETECT",
    "INVESTIGATE_PLAN", "APPROACH", "OBSERVE", "CLASSIFY",
    "RETURN_TO_PATROL", "RESUME", "HOME", "DONE",
)
# States in which the duck is actively walking toward a target.
WALKING_STATES = ("PATROL", "APPROACH", "RETURN_TO_PATROL", "RESUME")
# States in which the forward command MUST be exactly zero.  This is the
# behavior's strongest stillness claim and it is checked literally, per tick.
#
# CHECKPOINT_STOP and SCAN are the ones that matter: a guard robot that rolled
# through its own checkpoints while nodding its head has not stopped and
# scanned.  OBSERVE and CLASSIFY are the other half: the multi-angle observation
# happens from a standstill at a safe standoff, never while closing.
ZERO_COMMAND_STATES = (
    "CHECKPOINT_STOP", "SCAN", "CLEAR", "DETECT", "INVESTIGATE_PLAN",
    "OBSERVE", "CLASSIFY", "HOME", "DONE",
)
# States in which the duck is watching an anomaly, and where the visibility gate
# is therefore conditioned.  APPROACH is included deliberately: a robot that
# closed on something while looking away would be a robot that had stopped
# investigating.
MONITOR_STATES = ("DETECT", "INVESTIGATE_PLAN", "APPROACH", "OBSERVE",
                  "CLASSIFY")
# States in which the head is sweeping the facility rather than tracking a
# body: the checkpoint scan.
SCAN_STATES = ("SCAN",)
# States that must NOT appear at all.  These are the two named failures this
# behavior exists to avoid - abandoning the patrol after an investigation, and
# walking up to an unidentified object without stopping.  Declaring them means a
# run that produced either would fail loudly rather than pass quietly.
FORBIDDEN_STATES = ("ABANDON_PATROL", "CONTACT_TARGET")

# -- phase ceilings ---------------------------------------------------------
# Ceilings so a stuck phase fails loudly instead of hanging.  Each one
# TRANSITIONS rather than merely logging: a ceiling that does not move the
# machine is not a ceiling.
#
# EACH IS DERIVED FROM THE DISTANCE THE STATE HAS TO COVER AT A MEASURED SPEED,
# NOT CHOSEN.  A ceiling shorter than the walk it bounds is not a safety net -
# it is a bug that fires on a healthy run, which is exactly what a 34 s PATROL
# ceiling did: a leg RESUMED from an investigation standoff is much longer than
# a plain 0.86 m circuit leg, because the duck rejoins its route from wherever
# it happened to be standing, and it tripped at 154.68 s on an otherwise
# healthy run.  These are sized against the MEASURED worst case with margin.
#
#   longest patrol leg     ~1.30 m at 0.128 m/s = 10.2 s
#   longest approach       ~0.56 m at 0.096 m/s =  5.8 s
#   longest return         ~1.53 m at 0.128 m/s = 12.0 s
PATROL_MAX_S = 44.0
CHECKPOINT_STOP_MAX_S = 6.0
SCAN_MAX_S = 14.0
CLEAR_MAX_S = 4.0
DETECT_MAX_S = 6.0
INVESTIGATE_PLAN_MAX_S = 5.0
APPROACH_MAX_S = 26.0
OBSERVE_MAX_S = 16.0
CLASSIFY_MAX_S = 6.0
RETURN_MAX_S = 40.0
RESUME_MAX_S = 6.0
HOME_MAX_S = 10.0

# -- geometry ---------------------------------------------------------------
# The duck's conservative planar half-extent, MEASURED on this scene's own built
# model at the STAND pose and across the full head-yaw range.  It is
# BOUNDING-SPHERE based, so it already over-states the robot: the exact planar
# half-extent is 0.0827 m and the exact lateral half-width 0.0781 m.  Using the
# over-stated figure is the safe direction for every clearance claim at once.
# ``test_duck_planar_radius_matches_model`` pins it against the built scene.
DUCK_PLANAR_RADIUS = 0.1162
DUCK_EXACT_PLANAR_RADIUS = 0.0827
DUCK_EXACT_LATERAL_HALF_WIDTH = 0.0781
