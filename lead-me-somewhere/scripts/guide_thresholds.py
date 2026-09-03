#!/usr/bin/env python3
"""Every acceptance threshold, in one place, with the reason it has that value.

Separated from ``guide_metrics`` so that the numbers a run is graded against can
be read, reviewed and imported without loading the code that computes them — and
so that a test can assert a threshold is what the README claims without pulling
in a rollout.

Nothing here is a round number chosen for looks.  Each one is either MEASURED on
this scene (see ``tools/sweep_commands.py``) or DERIVED from something that was.
"""

from __future__ import annotations

# The stock walking policy this whole behavior is measured against.
UPSTREAM_POLICY_SHA = (
    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c")

# -- what the run must contain ----------------------------------------------
MIN_EPISODES = 2
MIN_BENDS = 3
MIN_DESTINATION_CANDIDATES = 3
MIN_MOVING_ADULTS = 5
# The lead must be a real walk, not a shuffle.
MIN_LEAD_PATH_M = 3.50
MIN_LEAD_NET_M = 3.00
# The follower must genuinely have been led somewhere.
MIN_FOLLOWER_WALKED_M = 3.00
# Each wait must be a real stop that achieved something.
MIN_WAIT_SECONDS = 1.50
MIN_CLOSED_DISTANCE_M = 0.35
# Visibility while monitoring, conditioned on line of sight existing at all.
MIN_VISIBLE_WITH_LOS = 0.95
# How near the destination fixture the duck must finish.
FINAL_DISTANCE_BAND_M = (0.30, 0.95)
# The planner's crowd term must have bitten.
MIN_CROWD_BLOCKED_CELLS = 1
# CHECK_FOLLOWER emits a literal zero, so the only path it may accumulate is the
# MEASURED zero-command drift: 0.0057 m over 10 s.  This bound is that figure
# with headroom for a longer check, and it is what proves the state is a
# standstill rather than a slow walk.
CHECK_STILL_PATH_M = 0.030
