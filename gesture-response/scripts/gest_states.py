#!/usr/bin/env python3
"""States, thresholds and the measured locomotion contract, in one place.

Every constant here is either MEASURED on THIS scene with THIS model (by
``tools/sweep_commands.py``) or DERIVED from one of those measurements.  Nothing
is inherited from a sibling behavior without being re-measured, because the
floor and the task are different - and this behavior needs two commands no
sibling needed: a real REVERSE and a pair of genuinely opposite TURNS.

THE MEASUREMENT THIS BEHAVIOR TURNS ON: THERE IS NO SMALL COMMAND
------------------------------------------------------------------
Forward gait onset on this scene is a CLIFF, not a ramp.  MEASURED over 6 s:

    vx = 0.20 -> 0.008 m   (no gait at all)
    vx = 0.22 -> 0.009 m   (no gait at all)
    vx = 0.24 -> 0.522 m
    vx = 0.26 -> 0.582 m
    vx = 0.30 -> 0.683 m

There is NOTHING between zero and a walk.  Three consequences shape the whole
behavior:

1. **A STOP is an exact zero, not a slow-down.**  A robot that "eased off" on
   being told to stop would emit 0.22, stand perfectly still, and log a nonzero
   command - the appearance of compliance with none of the physics.
2. **READY, OBSERVE, CONFIRM and ACK are STATES, not speeds.**  Each holds an
   exact zero, and the gate checks that literally, per tick.
3. **No sub-gait decorative command can exist anywhere**, because any command
   below the onset is indistinguishable from zero on the floor.  The gate
   requires every nonzero forward command to be at or above the onset.

**Holding exact zero really is holding still.**  MEASURED: 10 s of exact-zero
command drifts 0.0006 m with 0.0057 m of path and 0.12 deg of yaw.  Over 3 s:
0.0003 m and 0.0012 m.  That is what makes "it stopped when told" a claim about
the floor rather than about the command register.

REVERSE IS A SECOND, DEEPER CLIFF - AND IT IS NOT THE FORWARD ONE MIRRORED
---------------------------------------------------------------------------
MEASURED over 6 s, with displacement projected on the duck's own pre-command
heading:

    vx = -0.26 -> -0.002 m   (no gait at all)
    vx = -0.30 -> -0.004 m   (no gait at all)
    vx = -0.32 -> -0.716 m   and -50.0 deg of yaw
    vx = -0.34 -> -0.817 m   and -62.2 deg of yaw
    vx = -0.38 -> -0.982 m   and -56.4 deg of yaw

Backward onset sits at **-0.32**, far deeper than the forward -0.24 would
suggest, so a BACK_UP implemented by negating the approach command would produce
a robot that logged a reverse and did not move.  And the reverse gait carries an
enormous open-loop yaw drift - **-50 deg in 6 s** - so a reverse leg MUST close
a heading loop or it curls away from the straight line it claims to walk.

THE TURNS ARE MEASURED PER SIGN, AND THE SPEED WAS CHOSEN ON THAT MEASUREMENT
------------------------------------------------------------------------------
MEASURED trunk-yaw delta over a real walk, which is the only thing that makes a
named turn a physical claim rather than a label:

    vx = 0.26, wz = -0.58 -> -21.7 deg/s      vx = 0.26, wz = +0.58 -> +13.4 deg/s
    vx = 0.30, wz = -0.58 -> -21.7 deg/s      vx = 0.30, wz = +0.58 -> +21.1 deg/s

**The turn speed is 0.30 rather than 0.26 because of the LEFT sign.**  At 0.26
the policy's own right bias eats most of a left command - 13.4 deg/s against
21.7 - so a left turn would take 1.6x as long as its mirror and the two would
not be comparable manoeuvres.  At 0.30 the two signs deliver +21.1 and -21.7
deg/s, within 3 % of each other, which is what lets LEFT and RIGHT be executed
by the same controller with the same gains and compared as opposites.

TURNING IN PLACE IS UNAVAILABLE, WHICH IS WHY EVERY TURN IS A WALKED ARC
-------------------------------------------------------------------------
MEASURED at ``vx = 0`` across the whole command range: at most **1.6 deg/s**,
with 0.0032 m of drift.  The duck cannot pivot to face where it was pointed, so
a commanded TURN_LEFT is a real arc of MEASURED radius 0.423-0.436 m and a
TURN_RIGHT an arc of 0.411-0.412 m.  Both fit the open training area, which is
what ``tools/check_layout.py`` verifies against the instructor's own position.
"""

from __future__ import annotations

import math

# -- forward speeds (MEASURED, 6 s runs on this scene) ----------------------
#   vx    net m   speed m/s   yaw drift deg
#   0.18  0.007   0.001   <- no gait
#   0.20  0.008   0.001   <- no gait
#   0.22  0.009   0.002   <- no gait
#   0.24  0.522   0.087        -2.4
#   0.26  0.582   0.097        -6.6
#   0.30  0.683   0.114        -8.3
#   0.34  0.772   0.129       -11.9
#   0.38  0.889   0.148       -14.3
#   0.42  0.999   0.166       -12.8
#   0.46  1.264   0.211        +2.7
VX_ONSET = 0.24
# The APPROACH command, used to execute COME.  0.26 rather than faster: closing
# on the person who is training the robot is where a heading error costs the
# most, and a slower approach lands inside the standoff band rather than through
# it.  NOT below the onset; there is nothing between the onset and zero.
VX_APPROACH = 0.26
SPEED_AT_APPROACH = 0.097  # MEASURED 0.582 m in 6 s
# The command used to ease into the standoff over the last few centimetres.  It
# is the gait-onset command itself, because there is nothing below it.
VX_SETTLE = 0.24
SPEED_AT_SETTLE = 0.087  # MEASURED 0.522 m in 6 s
# The TURN command.  See the module docstring: chosen on the LEFT sign's
# measured rate, not on speed.
VX_TURN = 0.30
SPEED_AT_TURN = 0.114  # MEASURED 0.683 m in 6 s

# -- reverse (MEASURED, 6 s runs, displacement on the pre-command heading) --
#   vx     net m   back m   yaw deg
#  -0.26   0.003   -0.002     -0.2   <- no gait
#  -0.30   0.006   -0.004     -0.9   <- no gait
#  -0.32   0.789   -0.716    -50.0
#  -0.34   0.938   -0.817    -62.2
#  -0.38   1.082   -0.982    -56.4
#  -0.42   1.217   -1.044    -61.7
VX_REVERSE_ONSET = -0.32
# The BACK_UP command.  The onset itself, because it is the slowest reverse that
# exists: -0.30 produces 4 mm in six seconds.  A gentler retreat is not
# available on this policy, and pretending otherwise would be a command that
# logs a reverse and does not move.
VX_BACK_UP = -0.32
SPEED_AT_BACK_UP = 0.119  # MEASURED 0.716 m of BACKWARD travel in 6 s
# MEASURED open-loop yaw drift while reversing at the onset, over 6 s.  Quoted
# because it is why the reverse leg closes a heading loop rather than running
# open-loop like a short forward leg could.
REVERSE_YAW_DRIFT_6S_DEG = -50.0

# -- yaw, per sign (MEASURED trunk-yaw delta over real walks) ---------------
#   vx=0.26: wz=-0.58 -> -21.7 deg/s   wz=+0.58 -> +13.4 deg/s   (asymmetric)
#            wz=-0.50 -> -17.0 deg/s   wz=+0.50 -> +10.7 deg/s
#   vx=0.30: wz=-0.58 -> -21.7 deg/s   wz=+0.58 -> +21.1 deg/s   (symmetric)
#            wz=-0.50 -> -19.4 deg/s   wz=+0.50 -> +17.3 deg/s
#
# THE CEILING IS 0.58 AND THE TURN SPEED IS 0.30 FOR THE SAME REASON: that pair
# is where the two signs deliver the same rate.  Every other combination
# measured makes a left turn slower than its mirror, which would make "LEFT and
# RIGHT are opposite equal manoeuvres" false by construction.
WZ_MAX_RIGHT = 0.58
WZ_MAX_LEFT = 0.58
# Dead bands, per sign.  MEASURED at vx=0.26 the left sign needs more command to
# produce the same rate, so it carries the wider band.
WZ_MIN_RIGHT = 0.10
WZ_MIN_LEFT = 0.16
KP_YAW_RIGHT = 0.95
KP_YAW_LEFT = 1.45
# MEASURED yaw rate each sign delivers at the turn speed and the ceiling.  These
# are what the turn ceilings below are derived from.
TURN_RATE_LEFT_DPS = 21.1
TURN_RATE_RIGHT_DPS = 21.7
# MEASURED turning radius per sign at ``VX_TURN`` and the ceiling.
TURN_RADIUS_LEFT_M = 0.423
TURN_RADIUS_RIGHT_M = 0.411

# -- turning in place: MEASURED TO BE UNAVAILABLE ---------------------------
# The whole command range at vx = 0 produces at most 1.6 deg/s and 0.0032 m of
# drift.  No turn-in-place command is emitted anywhere in this behavior; these
# figures exist so the absence is a MEASUREMENT rather than an omission, and
# they are why every commanded turn is a walked arc and every look is a HEAD
# movement.
SPIN_BEST_RATE_DPS = 1.6
SPIN_BEST_COMMAND = 0.42

# -- coasting and drift (MEASURED) -----------------------------------------
# Walk 4 s, then hold exact zero for 1.5 s.
COAST_AT_APPROACH_M = 0.0086   # from vx = 0.30
COAST_AT_FAST_M = 0.0097       # from vx = 0.42
# Drift over 10 s of exact zero: 0.0006 m net, 0.0057 m of path, +0.12 deg.
# Over 3 s: 0.0003 m and 0.0012 m.
ZERO_DRIFT_10S_M = 0.0006
ZERO_PATH_10S_M = 0.0057
ZERO_DRIFT_3S_M = 0.0003

# -- the head (MEASURED joint range +/-170 deg yaw, +/-90 deg pitch) --------
TRACK_YAW_RATE_DPS = 26.0
TRACK_PITCH_RATE_DPS = 9.0
TRACK_PITCH_DEG = 4.0
# A SEARCH sweeps this far either side of the duck's own heading while it looks
# for the instructor.  MEASURED against the head yaw limit and the camera's own
# 58 deg vertical FOV: +/-52 deg of sweep plus a 34 deg half-width frustum
# covers a 172 deg arc from a standstill.
SEARCH_SWEEP_DEG = 52.0
SEARCH_PERIOD_S = 4.0 * SEARCH_SWEEP_DEG / TRACK_YAW_RATE_DPS

# -- the gesture gate --------------------------------------------------------
# THE CONFIRM WINDOW.  A gesture must be classified as the SAME command, from
# the SAME person, fully readable, for this long before the duck acts on it.
#
# DERIVED FROM THE ANIMATION'S OWN MEASURED ENVELOPE rather than chosen: the arm
# takes 0.70 s to travel from rest into the pose, so a window shorter than that
# could be satisfied entirely by an arm on its way somewhere else.  0.90 s is
# the raise time plus a margin, and it is comfortably inside the shortest
# gesture the instructor holds (5.2 s), so a real command is never missed for
# being too brief.
CONFIRM_S = 0.90
# The classification must hold at this rate over the window: a single dropped
# tick - the head lagging, a distractor crossing the sightline - must not reset
# a gesture the person is plainly still making.  MEASURED on the real run, the
# worst accepted gesture held 100.0 % of its confirm window.
CONFIRM_MIN_FRACTION = 0.85
# How long a rejected reading is remembered, so a pose that flickers in and out
# of a template does not produce a burst of separate rejections in the log.
REJECT_COOLDOWN_S = 1.2
# A gesture may be read only within this range.  DERIVED from the camera: at
# 3.2 m an adult's hand is a few pixels across in a 300x216 PiP, and the arm
# keypoints stop being separable.  Every gesture in the run is read well inside
# it.
GESTURE_MAX_RANGE_M = 3.20
# And the instructor must be confirmed visible for this long before the duck
# will lock onto her at all.  DERIVED from the MEASURED head sweep rate: at
# 26 deg/s a target crossing the 68 deg frustum is inside it for 2.6 s, so
# 0.40 s is a real dwell rather than a graze.
ACQUIRE_CONFIRM_S = 0.40

# -- executing the commands --------------------------------------------------
# THE SAFE STANDOFF BAND the COME command must leave the duck in.  A REQUIRED
# band, not a measured one: close enough to be plainly "come here", far enough
# that the robot never crowds or touches the person.
STANDOFF_MIN_M = 0.45
STANDOFF_MAX_M = 0.75
# What the approach aims for: the middle of the band, so the MEASURED stopping
# error has room on both sides.  DERIVED from the MEASURED 0.0086 m coast:
# aiming at the centre leaves 0.15 m of margin against a coast an order of
# magnitude smaller.
STANDOFF_TARGET_M = 0.60
# The approach STOPS when the MEASURED surface clearance is inside the band.
# Checked on measured clearance every tick, independently of the planned
# standoff point, so a badly placed target still cannot produce a close approach.
STANDOFF_STOP_M = 0.66

# How far the duck must turn on a TURN_LEFT or TURN_RIGHT.  A REQUIRED heading
# change, and the gate measures the trunk yaw delta that actually resulted.
TURN_TARGET_DEG = 72.0
# The turn is complete when the MEASURED yaw delta is within this of the target.
TURN_TOLERANCE_DEG = 8.0
# How far back a BACK_UP must carry the duck, measured along its OWN heading at
# the moment the command was accepted.  DERIVED from the MEASURED reverse speed
# of 0.119 m/s: 0.40 m is 3.4 s of real reverse gait, long enough to be
# unmistakable in the video and short enough to stay well inside the area.
BACK_UP_TARGET_M = 0.40
BACK_UP_TOLERANCE_M = 0.06

# How long the acknowledgment is held after each executed command, at an exact
# zero, before the duck returns to READY.  Long enough to be read in the video.
ACK_S = 1.6
# The final goodbye acknowledgment, held longer because it ends the session.
GOODBYE_S = 2.6
# How long the duck must be measurably stopped for a STOP to count as executed.
STOP_HOLD_S = 2.0
# The measured ground speed below which the duck counts as STOPPED.  DERIVED
# from the MEASURED zero-command drift: 0.0006 m over 10 s is 0.00006 m/s, so
# anything under 0.01 m/s is the gait having unwound rather than the robot still
# walking.  The gait-onset cliff means there is no genuine walking speed below
# 0.087 m/s to confuse it with.
SETTLED_MPS = 0.010
# How promptly a STOP must produce an exact zero command once confirmed.  It is
# ONE CONTROL TICK: the command goes to zero on the tick the confirmation
# completes, with no ramp, and the gate measures the tick index rather than
# trusting the controller.
STOP_ZERO_WITHIN_TICKS = 1

# The ONE command that may be accepted while the duck is already carrying out
# another one.  A STOP that can only be given to a robot that is already
# standing still is a formality rather than a stop: the whole purpose of the
# command is to interrupt motion already under way, and the scenario requires it
# to be given WHILE THE DUCK IS STILL WALKING.  Every other command is refused
# mid-manoeuvre, so the duck cannot be handed a new destination halfway through
# one - but it can always be told to stop.
#
# The confirm gate is NOT relaxed for it: same locked person, arm fully
# readable, sustained for the whole CONFIRM_S window.  Only the SET of
# acceptable commands narrows, never the standard of proof.
INTERRUPT_COMMAND = "STOP"

# -- the states -------------------------------------------------------------
STATES = (
    "READY", "OBSERVE", "CONFIRM",
    "EXECUTE_APPROACH", "EXECUTE_STOP", "EXECUTE_TURN_LEFT",
    "EXECUTE_TURN_RIGHT", "EXECUTE_BACK_UP",
    "ACK", "GOODBYE", "DONE",
)
EXECUTE_STATES = (
    "EXECUTE_APPROACH", "EXECUTE_STOP", "EXECUTE_TURN_LEFT",
    "EXECUTE_TURN_RIGHT", "EXECUTE_BACK_UP",
)
# The command each execute state carries out.  Kept as a mapping so the machine
# never has to parse its own state name.
STATE_FOR_COMMAND: dict[str, str] = {
    "COME": "EXECUTE_APPROACH",
    "STOP": "EXECUTE_STOP",
    "TURN_LEFT": "EXECUTE_TURN_LEFT",
    "TURN_RIGHT": "EXECUTE_TURN_RIGHT",
    "BACK_UP": "EXECUTE_BACK_UP",
    "WAVE": "GOODBYE",
}
# States in which the duck is actively walking under a command.
WALKING_STATES = (
    "EXECUTE_APPROACH", "EXECUTE_TURN_LEFT", "EXECUTE_TURN_RIGHT",
    "EXECUTE_BACK_UP",
)
# States in which the forward command MUST be exactly zero.  This is the
# behavior's strongest stillness claim and it is checked literally, per tick.
#
# EXECUTE_STOP is the one that matters most: it is entered from a WALK, and the
# whole point of a STOP command is that the nonzero command it interrupted goes
# to exactly zero within one control tick and stays there.
ZERO_COMMAND_STATES = (
    "READY", "OBSERVE", "CONFIRM", "EXECUTE_STOP", "ACK", "GOODBYE", "DONE",
)
# States in which the duck is watching the instructor, and where the visibility
# gate is therefore conditioned.  EXECUTE_APPROACH is included deliberately: a
# robot that closed on a person while looking away would have stopped attending
# to the person who is training it.
MONITOR_STATES = (
    "OBSERVE", "CONFIRM", "EXECUTE_APPROACH", "ACK", "GOODBYE",
)
# States in which the duck is sweeping for the instructor rather than tracking
# her.  READY is included because that is where an unlocked duck searches.
SEARCH_STATES = ("READY",)
# States that must NOT appear at all.  These are the named failures this
# behavior exists to avoid, and declaring them means a run that produced either
# would fail loudly rather than pass quietly.
FORBIDDEN_STATES = ("EXECUTE_UNCONFIRMED", "OBEY_STRANGER")

# -- phase ceilings ---------------------------------------------------------
# Ceilings so a stuck phase fails loudly instead of hanging.  Each TRANSITIONS
# rather than merely logging: a ceiling that does not move the machine is not a
# ceiling.  EACH IS DERIVED FROM THE DISTANCE OR ANGLE ITS STATE MUST COVER AT A
# MEASURED RATE, NOT CHOSEN.
#
#   longest approach   ~1.90 m at 0.097 m/s = 19.6 s
#   a 72 deg turn      at 21.1 deg/s        =  3.4 s
#   a 0.40 m reverse   at 0.119 m/s         =  3.4 s
READY_MAX_S = 30.0
OBSERVE_MAX_S = 14.0
CONFIRM_MAX_S = 4.0
APPROACH_MAX_S = 30.0
STOP_MAX_S = 6.0
TURN_MAX_S = 12.0
BACK_UP_MAX_S = 12.0
ACK_MAX_S = 4.0
GOODBYE_MAX_S = 6.0

# -- geometry ---------------------------------------------------------------
# The duck's conservative planar half-extent, MEASURED on this scene's own built
# model at the STAND pose and across the full head-yaw range.  It is
# BOUNDING-SPHERE based, so it already over-states the robot: the exact planar
# half-extent is smaller.  Using the over-stated figure is the safe direction
# for every clearance claim at once.  ``test_duck_planar_radius_matches_model``
# pins it against the built scene.
DUCK_PLANAR_RADIUS = 0.1162


def turn_rate_dps(direction: str) -> float:
    """MEASURED yaw rate for a named turn direction, at the turn speed."""
    return TURN_RATE_LEFT_DPS if direction == "left" else TURN_RATE_RIGHT_DPS


def turn_seconds(direction: str, degrees: float = TURN_TARGET_DEG) -> float:
    """How long a named turn should take, from its own MEASURED rate."""
    return abs(degrees) / turn_rate_dps(direction)


def back_up_seconds(distance_m: float = BACK_UP_TARGET_M) -> float:
    """How long a reverse leg should take, from the MEASURED reverse speed."""
    return abs(distance_m) / SPEED_AT_BACK_UP


def wrap_deg(angle: float) -> float:
    """Wrap degrees to [-180, 180)."""
    return math.degrees(
        (math.radians(angle) + math.pi) % (2.0 * math.pi) - math.pi)
