#!/usr/bin/env python3
"""Thresholds every acceptance gate is graded against, in one place.

Separated from the gates themselves so a number can be found and changed without
reading the judging logic, and so a test can import the bar without importing the
grader.

Each value is either a REQUIREMENT of the behavior or DERIVED from a measurement
in ``patrol_states``.  None of them is tuned to make a particular run pass.
"""

from __future__ import annotations

# The stock walking policy shipped with microduck_rl, byte-identical.
UPSTREAM_POLICY_SHA = (
    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c")

# -- the patrol --------------------------------------------------------------
# All five checkpoints, in the declared order, exactly once each.
REQUIRED_CHECKPOINTS = 5
# The circuit is 5.16 m and two investigations add their own approaches and
# returns, so a completed run walks appreciably further than one loop.
MIN_PATH_M = 9.0
# How near a checkpoint's centre the duck must stop for the visit to count.
# The painted pad is 0.38 m across, so this is the pad's own half-extent.
MAX_CHECKPOINT_ERROR_M = 0.22
# The duck must finish on the guard-post pad, whose half-extent is 0.30 m.
MAX_HOME_DISTANCE_M = 0.30
MIN_HOME_SECONDS = 1.2

# -- the checkpoint scans -----------------------------------------------------
# Every checkpoint must be a real stop and a real sweep.  The arc is MEASURED
# from the pose the head actually reached, so a sweep cut short reports less.
#
# A scan interrupted by a detection legitimately sweeps less than a full one -
# that is the point of leaving early - so the arc bar applies to the scans that
# ran to completion, and the stop bar applies to all of them.
MIN_SCAN_ARC_DEG = 100.0
MIN_CHECKPOINT_STOP_S = 1.2
# The path a checkpoint stop and its scan may accumulate on the floor.  See
# ZERO_STATE_PATH_PER_EPISODE_M below: this is the same settling transient.
MAX_STILL_PATH_M = 0.045

# -- the anomalies ------------------------------------------------------------
# Two distinct anomalies must be detected and investigated, and the benign
# distractor must be explicitly dismissed.
REQUIRED_INVESTIGATIONS = 2
REQUIRED_DISMISSALS = 1
REQUIRED_VERDICTS = 3
# Every detection must have happened inside the camera gate, so this is the
# minimum number of ticks a body must have been visible before being acted on.
# DERIVED from DETECT_CONFIRM_S at 50 Hz: 0.40 s is 20 ticks.
MIN_CAMERA_GATE_TICKS = 20

# -- the investigations -------------------------------------------------------
# Each approach must physically reduce the range by at least this.  DERIVED from
# the layout: the shorter of the two approaches is solved to close 0.52 m, and
# ``tools/check_layout.py`` fails the LAYOUT if either falls below 0.45 m.
MIN_RANGE_REDUCTION_M = 0.30
# And it must be a real walk on the floor, not a range that closed because the
# target moved.
MIN_APPROACH_PATH_M = 0.25
# The safe observation standoff band, imported rather than repeated: two copies
# of a safety threshold is one copy too many.
from patrol_states import (  # noqa: E402
    STANDOFF_MAX_M as MAX_STANDOFF_M,
)
from patrol_states import (  # noqa: E402
    STANDOFF_MIN_M as MIN_STANDOFF_M,
)
# Every observation must hold every declared angle.
REQUIRED_OBSERVE_ANGLES = 3
# And the target must be visible for most of each hold, or the angle was not an
# observation of anything.
MIN_ANGLE_VISIBLE_FRACTION = 0.60

# -- the route memory ---------------------------------------------------------
# The duck must return to within this of the point it broke off at.  It is the
# machine's own RESUME_TOLERANCE_M, imported for the same reason as the band.
from patrol_branch import RESUME_TOLERANCE_M as MAX_RETURN_ERROR_M  # noqa: E402

# -- safety -------------------------------------------------------------------
# The duck must never enter the marked restricted rectangle.  Strictly positive:
# the gate is that its trunk stayed outside, measured every control tick.
MIN_ZONE_GAP_M = 0.0
# And it must never touch anything.
MIN_BODY_CLEARANCE_M = 0.0

# -- stillness ----------------------------------------------------------------
# The path a zero-command state may accumulate, PER EPISODE.
#
# MEASURED on this scene: walking 4 s at vx=0.30 and then holding exact zero
# coasts 0.0091 m, and 10 s of exact zero from a standstill drifts 0.0006 m net
# with 0.0054 m of path.  A zero-command state entered FROM A WALK therefore
# accumulates one settling transient of about 0.03 m regardless of how long it
# lasts, which is why the bound is per episode rather than per state: this
# behavior enters CHECKPOINT_STOP five times and SCAN five times, and summing
# ten transients into two per-state totals and comparing them against a
# single-transient bound compares the wrong quantities.
ZERO_STATE_PATH_PER_EPISODE_M = 0.045
ZERO_STATE_NET_PER_EPISODE_M = 0.020
# Consecutive exact-zero ticks OUTSIDE the states allowed to hold one.  Anything
# above a handful is a stall.
MAX_ILLEGAL_ZERO_RUN = 24

# -- visibility ---------------------------------------------------------------
# The camera must be active on a target - watching something it is meant to be
# watching - in almost every tick where line of sight to it existed.
MIN_CAMERA_ACTIVE = 0.95
# And the investigated body must be visible in most monitoring steps where line
# of sight existed.
MIN_VISIBLE_WITH_LOS = 0.90

# -- the facility -------------------------------------------------------------
MIN_BODIES = 6
MIN_MOVING_BODIES = 3
MIN_FIXTURES = 10
# The largest single-tick heading change any scripted person may make.  A
# cornered polyline turns its walker through a whole corner in one tick; a
# filleted route does not.
MAX_ACTOR_HEADING_STEP_DEG = 6.0

# -- locomotion health --------------------------------------------------------
MIN_TRUNK_Z_M = 0.09
NOMINAL_TRUNK_Z_M = 0.116
TRUNK_Z_TOLERANCE_M = 0.012
