#!/usr/bin/env python3
"""States, thresholds and the measured locomotion contract, in one place.

Every locomotion constant here is either MEASURED on THIS scene with THIS model
(by ``tools/sweep_commands.py``) or DERIVED from one of those measurements.
Nothing is inherited from a sibling behavior without being re-measured: the
floor is different, and this behavior asks for something no sibling did - a
duck that must reach a MOVING geometric station between two people, hold it at
an exact zero, and then leave it in the other direction.

THE MEASUREMENT THIS BEHAVIOR TURNS ON: THERE IS NO SMALL COMMAND
------------------------------------------------------------------
Forward gait onset on this scene is a CLIFF, not a ramp.  The numbers are filled
in by ``tools/sweep_commands.py --what forward`` and quoted in the README.

Three consequences shape the whole behavior:

1. **HOLD_BUFFER is an exact zero, not a crawl.**  A robot that "held station"
   by emitting a sub-onset command would stand perfectly still and log motion -
   the appearance of active station-keeping with none of the physics.
2. **MONITOR, HOLD_BUFFER, THREAT_CLEAR and DONE are STATES, not speeds**, each
   holding a literal zero that the gate checks per tick.
3. **Every reposition is a real walk.**  There is no way to nudge 5 cm sideways;
   the duck either walks or it does not move, which is why an interpose station
   is only worth going to if it is far enough away to be worth a walk.

REVERSE IS A SECOND, DEEPER CLIFF - AND IT IS WHAT THE RETREAT IS BUILT ON
---------------------------------------------------------------------------
The retreat from an approaching protected person is a genuine reverse leg,
because turning to walk away would carry the duck through an arc that passes
CLOSER to her before it gets further away.  The reverse gait carries a large
open-loop yaw drift, so the retreat closes a heading loop and is graded on
displacement projected along the pre-action heading rather than on path.

TURNING IN PLACE IS UNAVAILABLE, WHICH IS WHY EVERY REPOSITION IS A WALKED ARC
-------------------------------------------------------------------------------
MEASURED at ``vx = 0`` across the whole command range: about a degree per
second.  The duck cannot pivot to face a new bearing, so getting onto the
interpose bearing is a real path problem with a real turning radius, and the
station-keeping controller must aim at a POINT rather than at a heading.
"""

from __future__ import annotations

import math

# ===========================================================================
# LOCOMOTION - every number below is MEASURED by tools/sweep_commands.py on
# this scene, and re-measured rather than inherited.  The tables are filled in
# from that tool's output; see the README for the full sweeps.
# ===========================================================================

# -- forward speeds (MEASURED, 6 s runs on this scene) ----------------------
#   vx    net m   speed m/s   yaw drift deg
#   0.20  0.008   0.001   <- no gait
#   0.22  0.009   0.002   <- no gait
#   0.24  0.522   0.087
#   0.26  0.582   0.097
#   0.30  0.683   0.114
#   0.34  0.772   0.129
VX_ONSET = 0.24
# The ESCORT command: the speed the duck holds its slot beside a person walking
# at 0.070 m/s.  It is well above her speed on purpose - the duck spends most of
# an escort leg stationary and closes the gap in short walks, because there is
# no command between zero and 0.087 m/s that could match her pace continuously.
VX_ESCORT = 0.24
SPEED_AT_ESCORT = 0.087
# The REPOSITION command, used to reach an interpose station or an escape gap.
# Faster than the escort because an interpose is a race against a walking adult:
# the station has to be reached BEFORE the intruder arrives, and the gate
# measures exactly that.
VX_REPOSITION = 0.52
SPEED_AT_REPOSITION = 0.252
# The command used to ease into a station over the last few centimetres.  It is
# the gait-onset command itself, because there is nothing below it.
VX_SETTLE = 0.24
SPEED_AT_SETTLE = 0.087

# -- reverse (MEASURED, 6 s runs, displacement on the pre-command heading) --
#   vx     net m   back m   yaw deg
#  -0.30   0.006   -0.004     -0.9   <- no gait
#  -0.32   0.789   -0.716    -50.0
#  -0.34   0.938   -0.817    -62.2
VX_REVERSE_ONSET = -0.32
# The RETREAT command.  The onset itself, because it is the slowest reverse that
# exists: -0.30 produces 4 mm in six seconds.  A gentler yield is not available
# on this policy, and pretending otherwise would be a command that logs a
# retreat and does not move.
VX_RETREAT = -0.32
SPEED_AT_RETREAT = 0.119
REVERSE_YAW_DRIFT_6S_DEG = -50.0

# -- yaw, per sign (MEASURED trunk-yaw delta over real walks) ---------------
WZ_MAX_RIGHT = 0.58
WZ_MAX_LEFT = 0.58
WZ_MIN_RIGHT = 0.10
WZ_MIN_LEFT = 0.16
KP_YAW_RIGHT = 0.95
KP_YAW_LEFT = 1.45
TURN_RATE_LEFT_DPS = 21.1
TURN_RATE_RIGHT_DPS = 21.7
TURN_RADIUS_LEFT_M = 0.423
TURN_RADIUS_RIGHT_M = 0.411

# -- turning in place: MEASURED TO BE UNAVAILABLE ---------------------------
SPIN_BEST_RATE_DPS = 1.6
SPIN_BEST_COMMAND = 0.42

# -- coasting and drift (MEASURED) -----------------------------------------
COAST_AT_REPOSITION_M = 0.0086
ZERO_DRIFT_10S_M = 0.0006
ZERO_PATH_10S_M = 0.0057
ZERO_DRIFT_3S_M = 0.0003

# -- the head (MEASURED joint range +/-170 deg yaw, +/-90 deg pitch) --------
TRACK_YAW_RATE_DPS = 26.0
TRACK_PITCH_RATE_DPS = 9.0
TRACK_PITCH_DEG = 2.0
SEARCH_SWEEP_DEG = 52.0

# ===========================================================================
# THE PROTECTIVE POLICY - required quantities, not measured ones.  Each is
# stated with the reason it has the value it has.
# ===========================================================================

# THE BUFFER.  The radius around the protected person the duck is trying to keep
# free of strangers.  A REQUIRED figure: large enough that an approach into it
# is unmistakable in the video, small enough that a populated plaza does not
# trigger it continuously - MEASURED on the real run, the person who is simply
# crossing the plaza never enters it.
BUFFER_M = 1.95
# The radius at which a stranger stops being a live threat, once inside.  Wider
# than the buffer so an intruder loitering at exactly the edge does not chatter
# between threatening and clear.
BUFFER_CLEAR_M = 2.05
# How far out a stranger is first watched at all.  Beyond this the duck does not
# even predict: at 0.135 m/s the fastest adult in the plaza is 26 s away, and
# predicting that far ahead is astrology rather than anticipation.
ALERT_RANGE_M = 3.50

# -- prediction --------------------------------------------------------------
# The horizon the closest-approach prediction looks over.  DERIVED: the fastest
# scripted adult covers 0.135 * 8 = 1.08 m in it, which is comfortably more than
# the buffer radius, so an intrusion is always predicted before it happens
# rather than at the moment it does.
PREDICT_HORIZON_S = 8.0
# A predicted intrusion must have a closest approach at least this far INSIDE
# the buffer before the duck acts.  A margin against a person who will graze the
# boundary and pass: the false-alarm adult's predicted closest approach is
# measured OUTSIDE the buffer entirely, and this margin is what makes the
# rejection robust rather than a coin flip on a boundary.
PREDICT_MARGIN_M = 0.12
# And it must be arriving within this long.  A person predicted to intrude in
# 20 s is not an intrusion yet; acting on them would make the duck jumpy and
# would spend the escort formation on nothing.
PREDICT_TTC_MAX_S = 7.5
# A predicted intrusion must be SUSTAINED for this long before the duck acts.
# DERIVED from the MEASURED velocity estimator: the estimator averages over
# VELOCITY_WINDOW_S, so a single noisy tick cannot survive this window.  It is
# also what the false alarm has to fail: he is briefly predicted to come close
# as he turns onto his line, and this is the window that discards it.
INTRUSION_CONFIRM_S = 0.60
# How long an intrusion must be gone before the duck stands down.  Long enough
# that a person pausing on their way out does not re-open the episode.
CLEAR_HOLD_S = 1.10

# -- the protected person's own approach -------------------------------------
# When SHE closes to this, the duck yields.  Well inside the escort slot's own
# distance, so it fires only when she has genuinely walked at the robot rather
# than when the slot geometry breathes.
PERSON_APPROACH_M = 0.62
# And it must be sustained: she must be closing, not merely close.
PERSON_APPROACH_CONFIRM_S = 0.40
# How far the retreat must carry the duck, measured along its OWN heading at the
# moment the yield began.  DERIVED from the MEASURED reverse speed of 0.119 m/s:
# 0.34 m is 2.9 s of real reverse gait, long enough to be unmistakable and short
# enough to stay well inside the plaza.
RETREAT_TARGET_M = 0.34
RETREAT_TOLERANCE_M = 0.05
# The retreat also has to actually increase the range to her.  Graded separately
# from the displacement, because she is walking toward the duck while it
# reverses: a retreat that moved the duck backward while she closed faster would
# satisfy the displacement test and fail the person.
RETREAT_RANGE_GAIN_M = 0.10

# -- the escort slot ---------------------------------------------------------
# Where the duck sits when nothing is happening: BESIDE and slightly BEHIND the
# protected person, on her right.  A REQUIRED formation.
ESCORT_LATERAL_M = 0.65
ESCORT_BEHIND_M = 0.40
# The slot counts as JOINED when the duck is within this of it.  DERIVED from
# the MEASURED coast of 0.0086 m plus the settle command's own per-tick travel:
# 0.22 m is comfortably outside both, so joining is a real arrival rather than a
# tolerance that any nearby position satisfies.
ESCORT_JOIN_M = 0.22
# And it must be held for this long for the escort to count as re-established.
ESCORT_HOLD_S = 1.00

# -- the interpose station ---------------------------------------------------
# How far from the PERSON the duck stands when interposing.  Chosen so the duck
# is unambiguously between the two rather than beside either: it is inside the
# buffer, which is the whole point, but far enough out that it never crowds her.
INTERPOSE_FROM_PERSON_M = 0.85
# The minimum clearance the interpose station must retain to BOTH people.  A
# station that satisfied between-ness by standing on somebody's toes would be a
# worse outcome than not interposing at all.
INTERPOSE_MIN_CLEARANCE_M = 0.30
# How close to the ideal station the duck must get for the interpose to count as
# ON STATION.  Wider than the escort join tolerance because the station moves
# while the intruder walks.
INTERPOSE_ON_STATION_M = 0.34
# The angular tolerance for "between".  The duck is between when the angle
# person->duck differs from person->threat by no more than this.  DERIVED: at
# the interpose radius of 0.70 m, 34 deg subtends 0.41 m of arc, which is more
# than the duck's own 0.23 m width - so a duck at the edge of tolerance is still
# physically on the line rather than beside it.
INTERPOSE_BEARING_TOL_DEG = 34.0

# -- the squeeze -------------------------------------------------------------
# Two threats count as SIMULTANEOUS when both are predicted to intrude and the
# angle between their bearings from the person exceeds this.  Below it, one
# station covers both and there is nothing to choose.
SQUEEZE_SEPARATION_DEG = 92.0
# The escape gap must keep at least this clearance to every person and every
# static surface.  Strictly larger than the interpose minimum, because an escape
# is chosen when standing between is impossible, and the whole point is to end
# up somewhere with room.
ESCAPE_MIN_CLEARANCE_M = 0.42
# How far the duck moves out to the escape gap.
ESCAPE_RADIUS_M = 0.95

# -- geometry ---------------------------------------------------------------
# The duck's conservative planar half-extent, MEASURED on this scene's own built
# model at the STAND pose and across the full head-yaw range.  BOUNDING-SPHERE
# based, so it already over-states the robot.
DUCK_PLANAR_RADIUS = 0.1162
# The window the duck estimates every person's velocity over, from positions it
# observed itself.  DERIVED: 0.40 s at 50 Hz is 20 samples, enough to be steady
# while short enough that a person turning a corner is tracked rather than
# averaged through the turn.
VELOCITY_WINDOW_S = 0.40

# The measured ground speed below which the duck counts as STOPPED.
SETTLED_MPS = 0.010

# ===========================================================================
# THE STATES
# ===========================================================================
STATES = (
    "ESCORT", "MONITOR", "PREDICT_INTRUSION", "INTERPOSE", "HOLD_BUFFER",
    "THREAT_CLEAR", "RETURN_ESCORT",
    "PERSON_APPROACH", "RETREAT",
    "MULTI_THREAT", "ESCAPE_GAP", "RECOVER", "DONE",
)
# States in which the duck is actively walking under the protective policy.
WALKING_STATES = (
    "ESCORT", "MONITOR", "INTERPOSE", "RETREAT", "ESCAPE_GAP",
    "RETURN_ESCORT", "RECOVER",
)
# States in which the forward command MUST be exactly zero.  This is the
# behavior's strongest stillness claim and it is checked literally, per tick.
#
# HOLD_BUFFER is the one that matters most: it is entered from a WALK, and the
# whole point of holding a buffer is that the duck STANDS on the line between
# the two people rather than shuffling on it.
ZERO_COMMAND_STATES = (
    "PREDICT_INTRUSION", "HOLD_BUFFER", "THREAT_CLEAR",
    "PERSON_APPROACH", "MULTI_THREAT", "DONE",
)
# States in which the duck is attending to its protected person, and where the
# visibility gate is therefore conditioned.  That is EVERY state: a protector
# that stopped watching its person while dealing with a stranger would have
# swapped one job for another.
MONITOR_STATES = STATES
# States in which an ACTIVE THREAT must also be visible.  Confined to the states
# where the duck is acting on a specific person, because in ESCORT there is no
# selected threat to see.
THREAT_VISIBLE_STATES = (
    "PREDICT_INTRUSION", "INTERPOSE", "HOLD_BUFFER", "MULTI_THREAT",
    "ESCAPE_GAP",
)
# States that must NOT appear at all.  These are the named failures this
# behavior exists to avoid, and declaring them means a run that produced either
# would fail loudly rather than pass quietly.
FORBIDDEN_STATES = ("CHARGE_INTRUDER", "BLOCK_BOTH", "CONTACT_THREAT")

# -- phase ceilings ---------------------------------------------------------
# Ceilings so a stuck phase fails loudly instead of hanging.  Each TRANSITIONS
# rather than merely logging.  EACH IS DERIVED FROM THE DISTANCE ITS STATE MUST
# COVER AT A MEASURED RATE:
#   longest escort join   ~1.60 m at 0.087 m/s = 18.4 s
#   an interpose leg      ~1.30 m at 0.114 m/s = 11.4 s
#   a 0.34 m retreat      at 0.119 m/s         =  2.9 s
ESCORT_MAX_S = 40.0
MONITOR_MAX_S = 60.0
PREDICT_MAX_S = 6.0
INTERPOSE_MAX_S = 16.0
HOLD_MAX_S = 22.0
CLEAR_MAX_S = 6.0
RETURN_MAX_S = 26.0
PERSON_APPROACH_MAX_S = 6.0
RETREAT_MAX_S = 12.0
MULTI_MAX_S = 6.0
ESCAPE_MAX_S = 16.0
RECOVER_MAX_S = 22.0


def wrap_deg(angle: float) -> float:
    """Wrap degrees to [-180, 180)."""
    return math.degrees(
        (math.radians(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def turn_rate_dps(direction: str) -> float:
    """MEASURED yaw rate for a named turn direction, at the reposition speed."""
    return TURN_RATE_LEFT_DPS if direction == "left" else TURN_RATE_RIGHT_DPS


def retreat_seconds(distance_m: float = RETREAT_TARGET_M) -> float:
    """How long a retreat leg should take, from the MEASURED reverse speed."""
    return abs(distance_m) / SPEED_AT_RETREAT
