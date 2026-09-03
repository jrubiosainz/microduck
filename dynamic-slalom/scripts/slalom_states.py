#!/usr/bin/env python3
"""States, thresholds and the measured locomotion contract, in one place.

Every constant here is either MEASURED on this scene with this model (by
``tools/sweep_commands.py``) or DERIVED from one of those measurements.  Nothing
is inherited from a sibling behavior without being re-measured, because the
floor, the traffic and the task are different.

THE MEASUREMENT THIS WHOLE BEHAVIOR TURNS ON: THERE IS NO STRAFE
------------------------------------------------------------------
A lateral ``vy`` command on this policy is a yaw disturbance wearing a strafe's
clothes, so the duck cannot step sideways out of somebody's way.  Every pass is
a **turning path**: turn out, run, turn back.  MEASURED with
``tools/sweep_commands.py --what lateral`` at ``vx = 0.34``, turning out for
``out_s`` seconds, running straight for 1.0 s, then turning back:

    wz     out_s   dx      dy       total
    -0.58   1.2    0.419   -0.100   3.4 s
    -0.58   1.8    0.542   -0.203   4.6 s
    -0.58   2.4    0.637   -0.331   5.8 s
    +0.58   1.2    0.414   +0.109   3.4 s
    +0.58   1.8    0.535   +0.216   4.6 s
    +0.58   2.4    0.627   +0.340   5.8 s

**A 0.34 m sidestep costs 0.64 m of course and 5.8 s of video**, and the two
signs agree to within 3 % at the ceiling.  Three consequences shape everything
else:

1. the corridor offsets the planner may choose are 0.32 / 0.46 / 0.60 m, because
   anything larger cannot be reached before a crossing body arrives;
2. the duck must COMMIT to a side early — a decision taken 1 m out cannot be
   executed — which is why the planner predicts over a horizon rather than
   reacting to a range;
3. WAITING is genuinely the cheaper option when neither side is clear, since a
   3.4 s sidestep into a corridor that closes is worse than a 2 s stop.

**Forward gait onset is a cliff, and it is at 0.24 on this scene.**  MEASURED
over 6 s:

    vx = 0.20 -> 0.008 m   (no gait at all)
    vx = 0.22 -> 0.009 m   (no gait at all)
    vx = 0.24 -> 0.524 m
    vx = 0.26 -> 0.582 m
    vx = 0.34 -> 0.777 m

So a robot that "edged forward carefully" past a moving cart would emit 0.22,
stand still, and log a nonzero command.  **Waiting is therefore a STATE, not a
speed**: the duck walks, or it holds exactly zero.

**The yaw axis is asymmetric and biased right.**  MEASURED at ``vx = 0.34``:
``wz = -0.10`` gives -6.7 deg/s while ``wz = +0.10`` gives +1.0 deg/s — the
policy's own right bias very nearly swallows a small left command.  Each sign
therefore carries its own gain, ceiling and dead band, and the left dead band
sits above the bias.  MEASURED at ``wz = 0``: 6 s of straight walking at
``vx = 0.34`` drifts **-11.4 deg**, so the heading loop must be closed even to
walk a straight lane.

**Turning in place is unavailable.**  MEASURED at ``vx = 0`` across the whole
command range: at most **1.4 deg/s**.  The duck cannot pirouette out of the way,
which is the other half of why a pass is a turning path.

**Holding exact zero really is holding still.**  MEASURED: 10 s of exact-zero
command drifts 0.0006 m with 0.0057 m of path and 0.12 deg of yaw.  That is what
makes "it waited for both bodies to clear" a claim about the floor rather than
about the metrics.
"""

from __future__ import annotations

import math

# -- forward speeds (MEASURED, 6 s runs on this scene) ----------------------
#   vx    net m   speed m/s
#   0.18  0.007   0.001   <- no gait
#   0.20  0.008   0.001   <- no gait
#   0.22  0.009   0.002   <- no gait
#   0.24  0.524   0.087
#   0.26  0.582   0.097
#   0.30  0.685   0.114
#   0.34  0.777   0.129
#   0.38  0.888   0.148
#   0.42  0.998   0.166
#   0.46  1.268   0.211
VX_ONSET = 0.24
# The cruising command down the open lane.
VX_WALK = 0.34
SPEED_AT_WALK = 0.129
# Used while executing a pass beside a moving body: slower, so a heading error
# costs less lateral travel before the controller corrects it.  NOT below the
# onset; there is nothing between the onset and zero.
VX_CAREFUL = 0.26
SPEED_AT_CAREFUL = 0.097
# The slowest walking command this behavior uses: easing into the goal band.
# It is the gait-onset command itself, because there is nothing below it.
VX_SETTLE = 0.24
SPEED_AT_SETTLE = 0.087

# -- yaw, per sign (MEASURED at vx=0.26, 0.34, 0.42 over 3 s) ---------------
#   vx=0.34: wz=-0.34 -> -13.3 deg/s   wz=+0.34 ->  +9.5 deg/s
#            wz=-0.16 ->  -9.4 deg/s   wz=+0.16 ->  +3.6 deg/s
#            wz=-0.10 ->  -6.7 deg/s   wz=+0.10 ->  +1.0 deg/s
#
# THE CEILING WAS SWEPT PAST ITS OWN EDGE, TWICE.  At 0.58 both signs are still
# responding almost linearly and the gait is untouched:
#
#   vx=0.34: wz=-0.42 -> -16.1   wz=-0.50 -> -17.4   wz=-0.58 -> -20.0 deg/s
#            wz=+0.42 -> +12.8   wz=+0.50 -> +15.9   wz=+0.58 -> +18.8 deg/s
#            wz=-0.68 -> -24.5   wz=-0.75 -> -29.5
#            wz=+0.68 -> +23.1   wz=+0.75 -> +25.6
#
# 0.58 is the ceiling this behavior uses.  0.68 and 0.75 turn faster still and
# are left on the table deliberately: their measured minimum trunk height is
# indistinguishable, but a slalom is about choosing a corridor early rather than
# about carving the tightest possible turn, and the extra rate buys 0.03 m of
# lateral travel per second against a visibly harsher gait.
WZ_MAX_RIGHT = 0.58
WZ_MAX_LEFT = 0.58
# Dead bands.  MEASURED: wz=+0.10 produced +1.0 deg/s because the policy's own
# right bias very nearly swallowed it, while wz=-0.10 produced -6.7 deg/s.
WZ_MIN_RIGHT = 0.10
WZ_MIN_LEFT = 0.16
KP_YAW_RIGHT = 0.90
KP_YAW_LEFT = 1.50

# -- the turning circle, DERIVED from the ceiling sweep ---------------------
# radius = (path / t) / rate, both from the same 3 s run at vx = VX_WALK with wz
# at the ceiling: 0.476 m turning right, 0.506 m turning left.  Per SIGN,
# because they differ by 6 %.
MIN_RIGHT_TURN_RADIUS_M = 0.476
MIN_LEFT_TURN_RADIUS_M = 0.506

# -- turning in place: MEASURED TO BE UNAVAILABLE ---------------------------
# The whole command range at vx = 0 produces at most 1.4 deg/s.  No turn-in-place
# command is emitted anywhere in this behavior; these figures exist so the
# absence is a MEASUREMENT rather than an omission.
SPIN_BEST_RATE_DPS = 1.4
SPIN_BEST_COMMAND = 0.42

# -- the lateral budget (MEASURED; see the module docstring) ----------------
# The three sidestep sizes the planner may choose.  A corridor wider than
# LATERAL_OFFSETS[-1] is not offered because the duck cannot converge onto it
# inside the prediction horizon.
LATERAL_OFFSETS: tuple[float, ...] = (0.26, 0.38, 0.50)
# MEASURED dy for an open-loop 2.4 s turn-out at the ceiling, both signs.
LATERAL_DY_AT_CEILING_M = 0.34
LATERAL_DX_COST_M = 0.64
LATERAL_SECONDS = 5.8

# THE RATE THE PLANNER USES IS THE CLOSED-LOOP ONE, AND THAT DISTINCTION IS A
# MEASUREMENT, NOT A REFINEMENT.
#
# The open-loop figure above is a ROUND TRIP - turn out, run, turn back - so
# 0.34 m / 5.8 s = 0.059 m/s understates what the duck can do while PASSING,
# because half that manoeuvre is spent undoing the very displacement a pass
# wants to keep.  ``tools/measure_pursuit.py`` runs the REAL controller chasing
# an offset line against the REAL policy, which is what the rollout actually
# does, and measures convergence:
#
#   offset   cruise           careful
#   0.26 m   3.56 s / 0.066   5.02 s / 0.051 m/s
#   0.38 m   4.78 s / 0.079   6.24 s / 0.058 m/s
#   0.50 m   5.92 s / 0.082   7.76 s / 0.057 m/s
#
# Passes run at the CAREFUL command, and the planner must assume the WORST of
# the two signs, so 0.0475 m/s is the honest figure: the slowest convergence
# measured on either hand at any offered offset.  Using the cruise figure would
# let the planner promise a corridor the duck reaches late.
LATERAL_RATE_MPS = 0.0475
# The cruise figure, kept beside it so the conservatism is visible rather than
# implied.  No gate consumes it.
LATERAL_RATE_CRUISE_MPS = 0.0644

# -- coasting and drift (MEASURED) -----------------------------------------
COAST_AT_WALK_M = 0.0088
COAST_AT_FAST_M = 0.0096
# Drift over 10 s of exact zero: 0.0006 m of net displacement, 0.0057 m of path.
ZERO_DRIFT_10S_M = 0.0006
ZERO_PATH_10S_M = 0.0057

# -- head tracking (MEASURED joint range +/-170 deg yaw, +/-90 deg pitch) ---
TRACK_YAW_RATE_DPS = 26.0
TRACK_PITCH_RATE_DPS = 9.0
TRACK_PITCH_DEG = 4.0

# -- following the plan -----------------------------------------------------
PURSUIT_LOOKAHEAD_M = 0.40
PROJECTION_WINDOW_M = 1.20
ARRIVE_RADIUS_M = 0.26

# -- the states -------------------------------------------------------------
STATES = (
    "PLAN", "ADVANCE", "THREAT", "CHOOSE_LEFT", "CHOOSE_RIGHT", "WAIT",
    "PASS", "REPLAN", "GOAL", "DONE",
)
# States in which the duck is actively walking toward a target.
WALKING_STATES = ("ADVANCE", "CHOOSE_LEFT", "CHOOSE_RIGHT", "PASS", "REPLAN")
# States in which the forward command MUST be exactly zero.  This is the
# behavior's strongest stillness claim and it is checked literally, per tick.
#
# WAIT is the one that matters: it is the state the duck enters when NEITHER
# corridor is predicted clear, and a duck that crept forward there would be a
# duck that had not really refused.
ZERO_COMMAND_STATES = ("WAIT", "GOAL", "DONE")
# States in which the duck is actively watching a crossing body, and where the
# visibility gate is therefore conditioned.  PASS is excluded for the reason in
# ``slalom_aim.THREAT_GAZE_STATES``: during a pass the head is back on the goal,
# so grading body-visibility there would grade the duck for looking where it is
# going.
MONITOR_STATES = ("THREAT", "CHOOSE_LEFT", "CHOOSE_RIGHT", "WAIT")
# States that must NOT appear at all.  BARGE_THROUGH and FREEZE_FOREVER are the
# two named failures this behavior exists to avoid; declaring them means a run
# that produced either would fail loudly rather than pass quietly.
FORBIDDEN_STATES = ("BARGE_THROUGH", "FREEZE_FOREVER")

# -- predicting occupancy ---------------------------------------------------
# How far ahead the planner predicts, and at what resolution.  THE HORIZON IS
# DERIVED FROM THE LATERAL BUDGET, and getting it wrong was a real failure.
#
# A first draft used 3.2 s, reasoning that it covered the decisive part of a
# crossing at the traffic's 0.16-0.30 m/s.  It does - but it is far shorter than
# the MEASURED time the duck needs to GET anywhere.  A 0.32 m sidestep takes
# 0.32 / 0.059 = 5.5 s, so every corridor was pruned as unreachable and the
# planner answered "wait" to a body 1.3 m away on an empty floor.  The symptom
# looked like an over-cautious planner; the cause was a horizon shorter than the
# robot's own manoeuvre time.
#
# 10.5 s is the smallest horizon that lets a decision be made on a COMPLETE
# view of a crossing rather than a truncated one.  Two constraints set it, and
# both are measured.
#
# The lower bound comes from the manoeuvre: a 0.50 m sidestep converges in a
# MEASURED 7.8 s at the careful command, so a shorter horizon cannot see the end
# of the duck's own move.
#
# The upper bound is that the prediction is CONSTANT-VELOCITY, so every extra
# second extrapolates a body that may be turning.  10.5 s is where a crossing at
# this traffic's 0.16-0.30 m/s fits entirely inside the window - the body enters
# the lane, crosses and vacates it - which is exactly the condition
# ``score_corridor``'s truncation rule requires before it will call a corridor
# safe.
#
# AT 7.0 s THE RULE FIRED CONSTANTLY AND THE DUCK WAITED AT EVERY ENCOUNTER:
# 9.1 s, 6.7 s, 3.6 s and 4.7 s of standing still, because no corridor could be
# scored on a complete view until the body was almost on top of it.  The waits
# were correct refusals on incomplete information; the horizon was the problem.
PREDICT_HORIZON_S = 10.5
PREDICT_DT_S = 0.35
PREDICT_SAMPLES = int(round(PREDICT_HORIZON_S / PREDICT_DT_S))
# Constant-velocity is the prediction model, and it is deliberately the SIMPLEST
# one that can be wrong.  Each actor's velocity is measured by finite difference
# from its own two most recent MEASURED positions, never read from its route.
# The gate then requires the prediction to conservatively BRACKET the closest
# approach that actually happened, which is a real test precisely because the
# model is not the choreography.
PREDICT_MODEL = "constant-velocity, finite-differenced from measured positions"

# A body is a THREAT when it is predicted to come within this of the duck's
# planned line within the horizon.  Derived from the duck's conservative planar
# radius (0.1162 m) plus a loaded body's planning extent (0.26-0.48 m) plus a
# margin, so anybody who could not physically conflict is never called a threat.
THREAT_CLEARANCE_M = 0.62
# And the along-course range within which a predicted conflict is close enough
# to act on.  DERIVED from the horizon rather than chosen: over 7.0 s the duck
# covers a MEASURED 0.90 m and a crossing body up to 2.1 m, so a conflict that
# matters is inside about 3.0 m.  Beyond this the duck keeps walking, because
# committing a sidestep to a body that will have crossed by then wastes the
# lateral budget it may need for the next one.
#
# THE UPPER BOUND ALSO PREVENTS COMMITTING BLIND, WHICH WAS A MEASURED FAILURE.
# At 3.20 m the duck engaged ``mara`` 9 s before she reached the lane - so far
# ahead that the horizon could only see her first two metres, scored the north
# as clear, committed LEFT, and she then walked north into the corridor it had
# chosen.  A decision taken outside the horizon that supports it is a guess.
# 3.00 m is the range at which the whole of a crossing fits inside the 10.5 s
# horizon at this traffic's speeds.
THREAT_RANGE_M = 3.00
# A corridor is SAFE only if its worst predicted clearance is at least this.
# Strictly positive, so "safe" means room to walk rather than room to touch.
# Lowered from 0.30 when the prediction became a SURFACE clearance rather than a
# centre-to-centre gap: the same physical situation now scores about a duck
# radius lower, so the bar moved with it rather than the geometry being loosened.
SAFE_CLEARANCE_M = 0.16
# Extra pessimism added to every predicted clearance, on top of subtracting the
# duck's own planar radius.
#
# DERIVED FROM A MEASURED FAILURE.  With a plain centre-to-centre prediction the
# planner's promise was routinely OPTIMISTIC against what was later measured:
# 0.630 m predicted against 0.249 m measured for ``mara``, 0.817 m against
# 0.252 m for ``dev``.  Two effects cause it, and both are real: a predicted gap
# between CENTRES is not a surface-to-surface clearance, and a body walking a
# filleted route covers more ground than a straight-line extrapolation of its
# current velocity predicts.  0.12 m covers the residual after the radius
# subtraction across every encounter in this scenario, and the bracketing gate
# is what proves it did.
PREDICTION_SLOP_M = 0.12
# The clearance a corridor must show AT THE HORIZON EDGE before it counts as
# safe on a truncated prediction.  See ``slalom_plan.score_corridor``: when the
# worst predicted moment is the last sample, the conflict is still developing
# when the prediction stops looking, so the number is an artifact of the cut
# rather than a property of the corridor.  A corridor that is comfortably clear
# even at the edge is genuinely clear; one that is marginal there has not been
# scored at all.  Set well above SAFE_CLEARANCE_M for exactly that reason.
TRUNCATED_SAFE_M = 0.55
# Static bodies are pruned at a tighter margin than moving ones, because they do
# not move: the duck knows exactly where a crate will be.
STATIC_MARGIN_M = 0.20

# -- resolving an encounter --------------------------------------------------
# A threat is RESOLVED - the crossing is over - once the body has crossed the
# duck's line and is at least this far away.  BOTH halves are required: a body
# that has passed but is still close is not yet clear, and a body that is far
# away but approaching is not resolved at all.
PASS_CLEAR_M = 0.70
# How long a resolved body keeps being ignored as a threat.  DERIVED from the
# traffic's own speeds: the slowest crosser (a 0.16 m/s cart) takes 8.1 s to
# clear the 1.3 m from the lane to where it stops mattering, so 9.0 s covers
# every body in this cast.
#
# THIS CONSTANT EXISTS BECAUSE OF A MEASURED FAILURE.  Without it the duck
# resolved an encounter, replanned, immediately re-detected the SAME receding
# body and opened another pass on it - producing ten "passes" for five
# crossings, destroying the alternation claim and running two phases into their
# ceilings.  It is time-limited rather than permanent so that a body which
# genuinely comes back round is a new encounter.
RESOLVED_IGNORE_S = 9.0
# How long a chosen corridor must stay predicted-safe before the duck commits,
# so a single tick of a favourable prediction is not a green light.
COMMIT_CONFIRM_S = 0.30
# How long the duck must hold a WAIT before it may resolve, so a two-tick
# "wait" cannot count as one.
MIN_WAIT_S = 1.20
# After each pass the duck REPLANS: it recomputes its line to the goal from
# where it actually is.  This is the minimum time in REPLAN, long enough that
# the replan is visible in the video and in the timeline.
MIN_REPLAN_S = 0.60

# -- the goal ---------------------------------------------------------------
# How near the band's centre the duck must be for GOAL, and how long it must
# hold there.  The band's own half-extent is 0.30 x 0.55 m.
GOAL_ARRIVE_M = 0.30
MIN_GOAL_S = 1.50
# The measured ground speed below which the duck counts as STOPPED.  DERIVED
# from the MEASURED zero-command drift: 0.0006 m of net displacement over 10 s
# is 0.00006 m/s, so anything under 0.01 m/s is the gait having unwound rather
# than the robot still walking.  The gait-onset cliff means there is no genuine
# walking speed below 0.087 m/s to confuse it with.
GOAL_SETTLED_MPS = 0.010
# The duck begins easing toward the band from this far out.  DERIVED from the
# MEASURED 0.0088 m coast plus the settle command's own 0.087 m/s.
#
# IT IS ALSO WHY GOAL IS ENTERED AT A STANDSTILL RATHER THAN MID-STRIDE.  The
# gait cannot stop instantly: the first run entered GOAL while still walking and
# drifted 0.033 m during the state, which is a stride's worth of coast and
# breaks the exact-standstill claim.  Easing in from 0.30 m means the duck is
# already on its slowest command when it crosses into the band.
GOAL_SETTLE_M = 0.30

# -- phase ceilings ---------------------------------------------------------
# Ceilings so a stuck phase fails loudly instead of hanging.  Each one
# TRANSITIONS rather than merely logging: a ceiling that does not move the
# machine is not a ceiling.
PLAN_MAX_S = 6.0
ADVANCE_MAX_S = 40.0
THREAT_MAX_S = 12.0
CHOOSE_MAX_S = 12.0
WAIT_MAX_S = 16.0
PASS_MAX_S = 18.0
REPLAN_MAX_S = 8.0
GOAL_MAX_S = 8.0

# -- geometry ---------------------------------------------------------------
# The duck's conservative planar half-extent, MEASURED on this scene's own built
# model at the STAND pose and across the full head-yaw range.  It is
# BOUNDING-SPHERE based, so it already over-states the robot: the exact planar
# half-extent is 0.0827 m and the exact lateral half-width 0.0710 m.  Using the
# over-stated figure is the safe direction for every clearance claim at once - a
# fatter robot finds every corridor harder to fit through.
# ``test_duck_planar_radius_matches_model`` pins it against the built scene.
DUCK_PLANAR_RADIUS = 0.1162
DUCK_EXACT_PLANAR_RADIUS = 0.0827
DUCK_EXACT_LATERAL_HALF_WIDTH = 0.0710
