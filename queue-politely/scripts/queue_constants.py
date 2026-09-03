#!/usr/bin/env python3
"""MEASURED locomotion constants, and the phase timing built on them.

Every number here was measured on THIS scene with THIS model by
``tools/sweep_commands.py`` and ``tools/measure_advance.py``.  Nothing is
inherited from a sibling behavior, because two properties of the stock walking
policy make inheriting numbers unsafe:

* **Gait onset is a cliff, not a ramp.**  A command below onset produces no
  motion at all, so "creep gently up the queue" cannot be expressed by shrinking
  the command.
* **The axes are not symmetric.**  Each direction has its own onset and its own
  left/right asymmetry, and the turn this behavior depends on had never been
  measured in this lab before.

They live in their own module so the state machine and the controller can both
read them without either owning them, and so a re-measurement changes exactly
one file.
"""

from __future__ import annotations

# -- forward speeds ---------------------------------------------------------
# Forward gait onset is a CLIFF: vx=0.20 -> 0.010 m in 6 s (no gait at all);
# vx=0.22 -> 0.409 m.  Nothing between zero and this is ever commanded.
VX_ONSET = 0.22
# Crossing the hall, and the queue advance itself.
#
# An advance is 0.55 m of queue and it must COMPLETE before the next person is
# served, or consecutive advances merge into one long chase and the behavior
# stops being a queue.  MEASURED: vx=0.38 walks at 0.150 m/s, so 0.55 m plus the
# settle window takes 4.6 s, and the straggler's first closure of 0.90 m takes
# 6.9 s against a 6 s service interval.  That is exactly what happened - four
# services merged into a single 2.711 m advance and only two WAIT->ADVANCE
# cycles were logged.  vx=0.46 walks at 0.207 m/s, clearing the same 0.90 m in
# 5.2 s with margin in hand.
VX_APPROACH = 0.46
VX_ADVANCE = 0.46
# The last stretch into the standoff band.  MEASURED 0.638 m in 6 s, and
# comfortably above the 0.22 onset.
VX_SETTLE = 0.28

# -- yaw, per sign ----------------------------------------------------------
# The two directions are NOT symmetric on this policy.  MEASURED over 3 s
# windows at vx=0.34:  wz=-0.18 -> R=1.119 m, but wz=+0.18 -> R=3.689 m.  The
# negative sense is by far the stronger, which is why the queue's fold is laid
# out to turn that way.  Each sign therefore carries its own gain, ceiling and
# dead band; a command below the dead band is emitted as exact zero rather than
# as a small number that does nothing.
WZ_MAX_RIGHT = 0.55        # negative wz, the STRONG direction
WZ_MAX_LEFT = 0.42         # positive wz, the weak direction
WZ_MIN_RIGHT = 0.14
WZ_MIN_LEFT = 0.16
KP_YAW_RIGHT = 1.05
KP_YAW_LEFT = 1.30

# -- path following ---------------------------------------------------------
# Pure-pursuit lookahead along the path, in metres of arc.  Long enough that the
# steering is smooth, short enough that the duck does not aim across the fold's
# chord.  MEASURED: shortening it on the bend to 0.22 m made the duck turn IN
# too hard and then swing wider on the exit (peak |cross-track| went from
# 0.133 m to 0.155 m), which is ordinary pure-pursuit over-steer.  The single
# longer lookahead tracks the fold better than a curvature-scaled one.
LOOKAHEAD_M = 0.42
LOOKAHEAD_MIN_M = 0.18
# Cross-track correction folded into the heading setpoint, clamped hard: a large
# correction inside a 0.84 m lane swings the nose into a rope faster than it
# fixes the offset.
KP_CROSS = 1.35
CROSS_SETPOINT_LIMIT_DEG = 28.0

# -- phase timing -----------------------------------------------------------
# Long enough to be a real look at the queue rather than a formality, and the
# whole of it is spent with the command at exactly zero.
OBSERVE_S = 2.4
IDENTIFY_S = 1.6
EVALUATE_S = 2.0
# Ceilings, so a stuck phase fails loudly instead of hanging.
APPROACH_MAX_S = 26.0
JOIN_MAX_S = 26.0
ADVANCE_MAX_S = 22.0
WAIT_MAX_S = 30.0
# Arrival tolerance on the target arc, and the settle window before a state
# counts as complete.  The coast after an exact-zero command was MEASURED at
# 0.011-0.020 m, so a duck that merely touched the setpoint is genuinely
# stopped a fraction of a second later.
ARRIVE_TOLERANCE_M = 0.06
SETTLE_S = 0.7
# The duck starts an advance once this much slack has opened between where it
# stands and the standoff it wants to hold.
ADVANCE_TRIGGER_M = 0.22

STATES = ("APPROACH", "OBSERVE_QUEUE", "IDENTIFY_TAIL", "EVALUATE_GAPS",
          "JOIN", "WAIT", "ADVANCE", "AT_COUNTER", "DONE")
STATIONARY_STATES = ("OBSERVE_QUEUE", "IDENTIFY_TAIL", "EVALUATE_GAPS",
                     "WAIT", "AT_COUNTER", "DONE")
# How long the duck stands at the counter being served before the run ends.
AT_COUNTER_S = 2.4
