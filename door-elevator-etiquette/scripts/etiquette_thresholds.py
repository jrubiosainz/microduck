#!/usr/bin/env python3
"""Every acceptance threshold, in one place, with the reason it has that value.

Separated from ``etiquette_metrics`` so that the numbers a run is graded against
can be read, reviewed and imported without loading the code that computes them -
and so a test can assert a threshold is what the README claims without pulling in
a rollout.

Nothing here is a round number chosen for looks.  Each one is either MEASURED on
this scene (see ``tools/sweep_commands.py``) or DERIVED from something that was.
"""

from __future__ import annotations

# The stock walking policy this whole behavior is measured against.
UPSTREAM_POLICY_SHA = (
    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c")

# -- what the run must contain ----------------------------------------------
# The scenario requires at least five adults besides the guardian.  There are
# seven; the gate checks the number that actually MOVED, so a body parked
# off-camera cannot pad the count.
MIN_MOVING_ADULTS = 5
# Both exiters must use the doorway, and at least this many occupants must
# leave the lift before the duck may board.
MIN_DOOR_EXITERS = 2
MIN_OCCUPANTS_OUT = 2
# The journey must be a real walk, not a shuffle.
MIN_PATH_M = 6.00
MIN_NET_M = 5.50
# Each yield must be a real stop.
MIN_YIELD_SECONDS = 2.00
# The ride must be long enough to be unambiguous.
MIN_RIDE_SECONDS = 8.00
# And the duck must have stood beside the lift, not in front of it, for this
# long before the doors opened.
MIN_WAIT_SIDE_SECONDS = 2.50

# -- the zone claims ---------------------------------------------------------
# THRESHOLD ENCROACHMENT IS GRADED AS AN EXACT ZERO, not a tolerance.
# The duck's conservative planar radius (0.1162 m) already over-states its
# footprint, and the holding points sit 0.11 m clear of every band, so any
# penetration at all means the behavior moved when it should not have.  A
# tolerance here would be a licence to creep.
MAX_EARLY_ZONE_STEPS = 0
# Ticks in which the duck shared an aperture with anybody.  Also exactly zero:
# the openings are wide enough for two (0.304 m of slack at the door), so this
# is a claim about the robot rather than about the wall.
MAX_SHARED_APERTURE_STEPS = 0
# The duck must never be ahead of the guardian along the shared route.
MAX_OVERTAKE_STEPS = 0
# And it must stay within this far behind her, or it has not followed her at
# all.  Generous, because the yields legitimately open the gap.
MAX_GUARDIAN_GAP_M = 4.20

# -- the doors ---------------------------------------------------------------
# The open fraction each aperture must have had at the instant the duck's own
# footprint first entered it.  This is the "no movement through closed doors"
# gate, and it is graded on the MEASURED fraction at first entry so a door that
# opened later cannot retroactively excuse an early crossing.
MIN_OPEN_FRACTION_AT_CROSSING = 0.55
# The clear gap that fraction must correspond to, in metres.  Derived: the
# duck's conservative planar diameter is 0.2324 m, so a gap below 0.30 m is not
# a passage it should be taking.
MIN_EFFECTIVE_GAP_AT_CROSSING_M = 0.30

# -- the cabin ---------------------------------------------------------------
# The duck's footprint must clear every interior face by this much while it
# rides.  Positive and non-trivial: standing with a toe over the sill is not
# being inside a lift.
MIN_CABIN_MARGIN_M = 0.02
# And it must actually have been in there for a real interval.
MIN_CABIN_SECONDS = 8.00

# -- visibility --------------------------------------------------------------
# Visibility of the ACTIVE subject while monitoring, conditioned on line of
# sight existing at all.
MIN_VISIBLE_WITH_LOS = 0.95

# -- stillness ---------------------------------------------------------------
# A zero-command state may accumulate only the MEASURED zero-command drift:
# 0.0055 m of path over 10 s.  This bound is that figure scaled to the longest
# such state in the run with headroom, and it is what proves those states are
# standstills rather than slow walks.
ZERO_STATE_PATH_M = 0.040
