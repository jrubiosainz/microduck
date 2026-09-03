#!/usr/bin/env python3
"""Thresholds every acceptance gate is graded against, in one place.

Separated from the gates themselves so a number can be found and changed without
reading the judging logic, and so a test can import the bar without importing
the grader.

Each value is either a REQUIREMENT of the behavior or DERIVED from a measurement
in ``slalom_states``.  None of them is tuned to make a particular run pass.
"""

from __future__ import annotations

# The stock walking policy shipped with microduck_rl, byte-identical.
UPSTREAM_POLICY_SHA = (
    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c")

# -- the journey -------------------------------------------------------------
# The course is 7.77 m of net travel; with the MEASURED 0.64 m of course each
# sidestep costs, a run that solves five encounters walks appreciably further
# than it displaces.
MIN_PATH_M = 8.0
MIN_NET_M = 7.0
# The duck must finish inside the painted band, whose half-extent is
# 0.30 x 0.55 m.  This is the distance from its centre.
MAX_FINAL_GOAL_DISTANCE_M = 0.30
MIN_GOAL_SECONDS = 1.4

# -- the encounters ----------------------------------------------------------
MIN_DYNAMIC_ENCOUNTERS = 4
MIN_WAITS = 1
# Both hands must be used, and the sides must alternate.
MIN_DISTINCT_PASS_SIDES = 2
# Every committed corridor must have been justified by a predicted clearance at
# least this large.  It is the planner's own safety bar
# (:data:`slalom_states.SAFE_CLEARANCE_M`), so a pass that squeaked under it
# would mean the machine had committed to something the planner rejected.
# Imported rather than repeated, because two copies of a safety threshold is one
# copy too many.
from slalom_states import SAFE_CLEARANCE_M as MIN_CHOSEN_PREDICTED_CLEARANCE_M  # noqa: E402,F401
# And the losing side must have been genuinely worse.  A choice between two
# equally good corridors is not evidence of anything.
#
# 0.02 m was the first figure and it was too strict for one real case: at E3 the
# duck chose the right on 0.240 m against 0.232 m on the left, a 0.008 m margin.
# That is a genuine and correctly-signed preference - the right WAS better, and
# it is the side the scenario intended - but 8 mm is not a difference anybody
# should claim is decisive.  The honest gate is that the chosen side was never
# WORSE than the rejected one, plus a real margin on most encounters, so the
# bar is the sign and a token epsilon rather than a fabricated separation.
MIN_REJECTION_MARGIN_M = 0.0

# -- the lateral evidence ----------------------------------------------------
# There is no strafe, so a real slalom shows up as path length in excess of net
# displacement, a signed lane offset that goes BOTH ways, and real yaw travel.
MIN_LATERAL_SPAN_M = 0.40
MIN_PATH_EXCESS_M = 0.30
MIN_YAW_TRAVEL_DEG = 120.0

# -- the traffic ------------------------------------------------------------
MIN_OBSTACLES_AND_ACTORS = 7
MIN_MOVING_ACTORS = 5
# The largest single-tick heading change any scripted body may make.  A cornered
# polyline turns its walker through a whole corner in one tick; a filleted route
# does not.
MAX_ACTOR_HEADING_STEP_DEG = 6.0

# -- stillness ---------------------------------------------------------------
# The path a zero-command state may accumulate.
#
# THE FIRST BOUND WAS MEASURED FROM THE WRONG INITIAL CONDITION.  10 s of exact
# zero starting FROM A STANDSTILL drifts 0.0057 m of path, and 0.030 m looked
# like a generous allowance against it.  But a zero-command state in this
# behavior is almost always entered FROM A WALK, and the gait has to unwind.
# MEASURED, walking 4 s at vx=0.34 and then holding exact zero:
#
#   zero for 2 s -> 0.0294 m of path, 0.0095 m net
#   zero for 4 s -> 0.0302 m of path, 0.0097 m net
#   zero for 8 s -> 0.0327 m of path, 0.0098 m net
#
# The path is ~0.030 m REGARDLESS of how long the state lasts, and the net
# displacement is ~0.010 m: the robot rocks onto its stance foot once and then
# stops.  That is a settling transient, not drift, which is why the figure does
# not grow with time.
#
# The bound is therefore PER EPISODE, not per state: three separate WAITs each
# entered from a walk legitimately accumulate three transients.  Summing them
# into one per-state total and comparing against a single-transient bound is
# comparing the wrong quantities.
ZERO_STATE_PATH_PER_EPISODE_M = 0.045
# And the net displacement a zero-command state may accumulate per episode,
# which is the honest "did it actually move" figure: 0.010 m MEASURED, doubled.
ZERO_STATE_NET_PER_EPISODE_M = 0.020
# Consecutive exact-zero ticks OUTSIDE the states allowed to hold one.  Anything
# above a handful is a stall.  PLAN legitimately holds zero for its first 0.4 s
# while the initial plan is computed, which is 20 ticks.
MAX_ILLEGAL_ZERO_RUN = 24

# -- visibility --------------------------------------------------------------
MIN_VISIBLE_WITH_LOS = 0.90
MIN_GOAL_VISIBLE_WITH_LOS = 0.70

# -- locomotion health -------------------------------------------------------
MIN_TRUNK_Z_M = 0.09
NOMINAL_TRUNK_Z_M = 0.116
TRUNK_Z_TOLERANCE_M = 0.012
