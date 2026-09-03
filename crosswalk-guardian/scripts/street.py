#!/usr/bin/env python3
"""The street's geometry, as a single source of truth.

Every other module reads its numbers from here: the scene generator that emits
the MJCF, the conflict predictor that decides when it is safe to cross, the
camera that measures which road sector is visible, the metrics that grade the
rollout, and the overlay that draws it.  A crossing decision computed against
one set of lane edges and drawn against another would be unfalsifiable, so the
edges exist exactly once.

Coordinate convention
---------------------
The duck crosses along **+X**.  It starts on the near pavement facing +X, so
its LEFT hand is **+Y** and its RIGHT hand is **−Y** throughout — the duck
never turns around, which is what makes "look left" a fixed world direction
rather than something that has to be recomputed from trunk yaw.

Traffic runs along **Y**, right-hand drive:

* the **NEAR lane** (``x = -0.275``) carries vehicles travelling **−Y**, so
  they arrive from the duck's **LEFT**;
* the **FAR lane** (``x = +0.275``) carries vehicles travelling **+Y**, so
  they arrive from the duck's **RIGHT**.

That is why ``LOOK_LEFT → LOOK_RIGHT → LOOK_LEFT_AGAIN`` is the correct order
and not a decoration: the first lane the duck enters is the one whose traffic
comes from the left, and the last check before stepping off is a re-check of
that same lane because it is the one that will reach it first.

    y ↑                        far pavement
      │        ┌─────────────────────────────────┐
      │  ····· │ FAR  lane  →+Y  ················│  x = +0.275
      │  ───── │ ─────────────────────────────── │  centre line
      │  ····· │ NEAR lane  →−Y  ················│  x = -0.275
      │        └─────────────────────────────────┘
      │                       near pavement
      └────────────────────────────────────────────→ x
         start        wait line   road    wait line   safe zone
         -2.05          -0.78   ±0.55       +0.78       +1.075
"""

from __future__ import annotations

# --- carriageway -------------------------------------------------------------
ROAD_HALF_WIDTH: float = 0.55        # road spans x in [-0.55, +0.55]
ROAD_HALF_LENGTH: float = 46.0       # exceeds traffic.LOOP_HALF_Y (42 m)
LANE_OFFSET: float = 0.275           # lane centres at x = -/+0.275
LANE_HALF_WIDTH: float = 0.275
CENTRE_LINE_X: float = 0.0

NEAR_LANE_X: float = -LANE_OFFSET
FAR_LANE_X: float = +LANE_OFFSET
# Lane extents along x, used to compute when the duck occupies each lane.
NEAR_LANE_SPAN: tuple[float, float] = (-ROAD_HALF_WIDTH, CENTRE_LINE_X)
FAR_LANE_SPAN: tuple[float, float] = (CENTRE_LINE_X, ROAD_HALF_WIDTH)
LANE_SPANS: dict[str, tuple[float, float]] = {
    "near": NEAR_LANE_SPAN,
    "far": FAR_LANE_SPAN,
}
# Which hand each lane's traffic arrives from.  Used by the scan gate to decide
# which sector must be visible in which LOOK phase.
LANE_SIDE: dict[str, str] = {"near": "left", "far": "right"}
# Sign of travel along y for each lane.
LANE_DIRECTION: dict[str, float] = {"near": -1.0, "far": +1.0}

# --- kerbs, markings, zones --------------------------------------------------
KERB_X: float = 0.61
WAIT_LINE_X: float = 0.78            # painted on both sides, at -/+0.78
PAVEMENT_OUTER: float = 2.60
SAFE_ZONE_X: float = 1.075
SAFE_ZONE_HALF: float = 0.275
SAFE_ZONE_SPAN: tuple[float, float] = (
    SAFE_ZONE_X - SAFE_ZONE_HALF, SAFE_ZONE_X + SAFE_ZONE_HALF)
CROSSWALK_HALF_SPAN: float = 0.55    # zebra bars span y in [-0.55, +0.55]

# --- the duck's own places ---------------------------------------------------
START_X: float = -2.05
START_Y: float = 0.0
# Where the trunk centre must come to rest before the scan.  The duck's planar
# half-extent is 0.1303 m (measured), so the leading surface sits 0.1303 m
# ahead of the trunk centre and the wait line is at -0.78.
#
# MEASURED OVERSHOOT: the approach releases its command when the trunk reaches
# the release point, but the policy does not stop on the tick it is told to.
# Run 1 released at -0.960 and came to rest at **-0.900**: 60 mm of overshoot,
# an order of magnitude more than the 5-9 mm coast the other behaviors measured
# after a slow cruise, because this approach runs at vx=0.52 rather than 0.28.
# The first draft placed the stop at -0.95 and the duck's leading edge finished
# 10 mm PAST the wait line — a gate failure that is entirely real: a pedestrian
# whose toes are over the line has not stopped before it.
#
# The target is therefore set from the line backwards: leading edge at -0.86
# leaves 80 mm of paint-to-toe margin, which absorbs the measured 60 mm
# overshoot and still reads as "stopped at the kerb" rather than "stopped in
# the middle of the pavement".
CURB_STOP_X: float = -1.055
# The far-side goal.  The crossing ends when the trunk centre reaches the safe
# zone, whose near edge is at x=0.80, so a goal at 1.10 keeps the duck walking
# to the MIDDLE of the zone rather than stopping the instant it clips the edge.
# The extra 0.30 m also puts the duck's trailing surface at 0.67 m — clear of
# the road edge at 0.55 — which is what makes "reached the opposite safe zone"
# mean the whole robot, not just its centre point.
CROSS_GOAL_X: float = 1.10

# --- what counts as "in the road" -------------------------------------------
# The duck's planar half-extent, MEASURED on this scene by
# ``contact_geometry.duck_planar_radius`` (tools/measure_crossing.py prints it)
# and pinned by a test.  Occupancy is graded on the trunk centre inflated by
# it, so a gate that says "never in the lane" means no part of the duck's
# footprint, not just its centre point.
#
# MEASURED VALUE: 0.1303 m.  The first draft of this file inherited 0.090 from
# come-here-recall, which is that behavior's *torso* half-width rather than the
# full geom envelope; it under-reported the footprint by 45% and would have let
# the duck's real outline sit inside a lane while every occupancy gate called
# it clear.
DUCK_PLANAR_RADIUS: float = 0.1303
ROAD_ENTRY_X: float = -ROAD_HALF_WIDTH - DUCK_PLANAR_RADIUS   # -0.6803
ROAD_EXIT_X: float = +ROAD_HALF_WIDTH + DUCK_PLANAR_RADIUS    # +0.6803


def duck_span(trunk_x: float) -> tuple[float, float]:
    """The duck's planar footprint along x, at a given trunk centre."""
    return (trunk_x - DUCK_PLANAR_RADIUS, trunk_x + DUCK_PLANAR_RADIUS)


def in_road(trunk_x: float) -> bool:
    """True while any part of the duck's footprint overlaps the carriageway."""
    low, high = duck_span(trunk_x)
    return high > -ROAD_HALF_WIDTH and low < ROAD_HALF_WIDTH


def in_lane(trunk_x: float, lane: str) -> bool:
    """True while the duck's footprint overlaps one particular lane."""
    low, high = duck_span(trunk_x)
    lane_low, lane_high = LANE_SPANS[lane]
    return high > lane_low and low < lane_high


def encroaches_wait_line(trunk_x: float) -> bool:
    """True once the duck's leading surface has crossed the near wait line."""
    return duck_span(trunk_x)[1] > -WAIT_LINE_X


def in_safe_zone(trunk_x: float) -> bool:
    """True while the trunk centre is inside the painted far-side safe zone."""
    return SAFE_ZONE_SPAN[0] <= trunk_x <= SAFE_ZONE_SPAN[1]


# --- road sectors the duck has to look at ------------------------------------
# Sample points the LOOK phases are graded against.  They sit on the LANE the
# corresponding traffic actually uses, at eye-catching ranges up the road, at
# roughly a vehicle's cabin height.  Visibility is measured through the exact
# camera the PiP renders from, with occlusion ray casts — so "the duck looked
# left" means the left approach was genuinely inside its camera, not that a
# head joint reached an angle.
SECTOR_SAMPLE_Y: tuple[float, ...] = (1.30, 2.20, 3.40, 4.80)
SECTOR_SAMPLE_Z: float = 0.14

LEFT_SECTOR = tuple(
    (NEAR_LANE_X, +y, SECTOR_SAMPLE_Z) for y in SECTOR_SAMPLE_Y
)
RIGHT_SECTOR = tuple(
    (FAR_LANE_X, -y, SECTOR_SAMPLE_Z) for y in SECTOR_SAMPLE_Y
)
SECTORS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "left": LEFT_SECTOR,
    "right": RIGHT_SECTOR,
}
# Fraction of a sector's sample points that must be visible for the phase to
# count as a genuine look.  Two of four is a real sightline up the road, and it
# does not demand that a 58 deg camera swallow a 4.8 m span in one pose.
SECTOR_VISIBLE_MIN_FRACTION: float = 0.50


def sector_of_lane(lane: str) -> str:
    """Which sector a lane's traffic appears in."""
    return LANE_SIDE[lane]
