#!/usr/bin/env python3
"""The corridor's geometry, as a single source of truth.

Every other module reads its numbers from here: the generator that emits the
MJCF, the encounter predictor that decides when to pull over, the alcove scorer
that refuses the unusable bays, the camera that measures what the duck can see,
the metrics that grade the rollout, and the overlay that draws it.  A pull-over
decided against one set of wall lines and drawn against another would be
unfalsifiable, so the walls exist exactly once.

Coordinate convention
---------------------
The corridor runs along **+X**.  The duck starts at the near end facing +X and
never turns around, so its LEFT hand is **+Y** and its RIGHT hand is **−Y** for
the whole rollout.  The corridor centreline is ``y = 0``.

    y ↑
      │  ███████████        ████████            ██████    wall  y = +0.21
      │             ▏bay_crates▕    ▏ bay_far ▕
      │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   centreline y = 0
      │      ▏bay_shallow▕   ▏bay_open▕
      │  ██████           █████       ███████████████     wall  y = −0.21
      └───────────────────────────────────────────────→ x
        start                                  dest    lobby

Why the corridor is genuinely too narrow to share
-------------------------------------------------
Two bodies of lateral half-widths ``h1`` and ``h2``, both confined to a strip
of half-width ``W`` and neither of them turning, can be separated by at most
``2W − h1 − h2`` between their centres.  They fit side by side at all only when
that exceeds ``h1 + h2``; they fit *safely*, with a clear surface gap ``g``,
only when it exceeds ``h1 + h2 + g``.

``corridor_passing_geometry()`` evaluates exactly that from the MEASURED
half-widths and reports every term, so "too narrow to pass safely" is
arithmetic a test can check rather than a claim in a README.

It is graded on each body's **exact lateral half-width**, not on the
conservative bounding-sphere radius used everywhere else.  Bounding spheres
badly over-state a body that is long in x and narrow in y, and over-stating the
bodies here would make the corridor look narrower than it is — the one
direction in which conservatism would flatter this behavior instead of testing
it.  Every *other* gate — wall clearance, passage intrusion, alcove fit — uses
the conservative radius, because there an over-wide robot makes the gate
harder.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- the corridor itself -----------------------------------------------------
CORRIDOR_HALF_WIDTH: float = 0.21       # inner wall faces at y = ±0.21
CORRIDOR_WIDTH: float = 2.0 * CORRIDOR_HALF_WIDTH
WALL_THICKNESS: float = 0.05
WALL_HEIGHT: float = 0.60
CORRIDOR_X_MIN: float = -2.70            # doorway from the room behind
CORRIDOR_X_MAX: float = +2.20            # doorway into the lobby
LOBBY_X_MAX: float = +12.0
LOBBY_HALF_WIDTH: float = 1.90
CENTRELINE_Y: float = 0.0

# --- the duck's own places ---------------------------------------------------
START_X: float = -1.95
START_Y: float = 0.0
DESTINATION_X: float = +1.60
DESTINATION_HALF: float = 0.18          # painted threshold half-length in x

# --- footprint radii, all MEASURED on this scene -----------------------------
# The duck's CONSERVATIVE planar half-extent: every robot geom's bounding
# sphere, projected into the plane.  MEASURED by
# ``contact_geometry.duck_planar_radius`` and pinned by a test.  This is the
# value every gate about the duck's own footprint uses, because over-stating
# the robot makes each of those gates harder to satisfy.
#
# The measurement varies slightly with pose - 0.1162 m in the STAND keyframe
# with the legs together, more mid-stride - so the constant is set ABOVE what
# the model reports at rest and a test requires it to stay there.  Rounding a
# footprint radius down is the one direction that quietly weakens every
# clearance gate at once.
DUCK_PLANAR_RADIUS: float = 0.1303
# The duck's EXACT lateral half-width, MEASURED by
# ``contact_geometry.exact_lateral_half_width``: the largest |y| any robot
# surface reaches from the trunk origin, with no bounding-sphere inflation.
DUCK_LATERAL_HALF: float = 0.0705
# The same measurement applied to a pedestrian body.  Both are pinned by tests
# against the generated geometry.
ADULT_LATERAL_HALF: float = 0.1040
# The adult's rotation-invariant planar half-extent, used for the along-corridor
# reach in the encounter predictor.
ADULT_PLANAR_RADIUS: float = 0.1155

# The clear surface-to-surface gap a side-by-side pass would need to count as
# safe.
#
# JUSTIFIED FROM MEASUREMENT, not taste.  ``tools/measure_pullover.py`` runs the
# real closed-loop corridor cruise and reports the duck's own peak lateral
# excursion from the centreline it is trying to hold: MEASURED 0.0634 m over a
# 12 s cruise.  A nominal passing gap smaller than the robot's own tracking
# error can be closed by tracking error alone, so it is not a gap.  0.10 m
# carries that measurement with margin, and a test requires the constant to
# stay at or above the measured excursion.
SAFE_PASSING_GAP_M: float = 0.10

# --- the passage the duck has to vacate --------------------------------------
# The strip of corridor an adult needs in order to walk through unimpeded: the
# adult's own lateral half-width plus the same clear gap.  A pull-over only
# counts when the duck's ENTIRE footprint is outside this strip.
CENTER_PASSAGE_HALF: float = ADULT_LATERAL_HALF + SAFE_PASSING_GAP_M   # 0.204
# Where the duck's trunk centre must get to for its footprint to clear the
# passage, using the CONSERVATIVE radius.
CLEAR_ABS_Y: float = CENTER_PASSAGE_HALF + DUCK_PLANAR_RADIUS          # 0.3343
# Rejoin tolerance: how close to the centreline "back on the centreline" means.
REJOIN_TOLERANCE_M: float = 0.10
# Margin past the point where the duck's footprint first fits between an
# alcove's cheeks, at which it may begin its lateral step.  Small, because the
# mouths are far longer than the footprint; nonzero, so the step never begins
# exactly on the geometric boundary.
ENTRY_MARGIN_M: float = 0.05

# NOTE WHAT THOSE TWO NUMBERS IMPLY.  ``CLEAR_ABS_Y`` is 0.3343 m, while the
# plain corridor only lets the trunk centre reach ``CORRIDOR_HALF_WIDTH −
# DUCK_PLANAR_RADIUS`` = 0.0797 m.  **There is nowhere in the plain corridor
# where the duck can clear the passage.**  Pulling over is not an optimisation
# here; it is the only way the adult gets through, and
# ``plain_corridor_max_trunk_abs_y()`` states that as a checkable number.


def plain_corridor_max_trunk_abs_y(
    half_width: float = CORRIDOR_HALF_WIDTH,
    radius: float = DUCK_PLANAR_RADIUS,
) -> float:
    """Largest |y| the trunk centre can reach in the corridor outside a recess."""
    return half_width - radius


@dataclass(frozen=True)
class Alcove:
    """One side recess in the corridor wall.

    ``depth`` is how far the recess extends beyond the corridor's inner wall
    face.  ``blocked_from`` is the |y| at which an obstruction (a stack of
    crates) begins; everything beyond it is unusable, so the alcove's usable
    outer limit is ``min(wall + depth, blocked_from)``.
    """

    name: str
    center_x: float
    half_length_x: float
    side: int                 # +1 = the +Y wall, −1 = the −Y wall
    depth: float
    blocked_from: float | None = None
    label: str = ""

    @property
    def outer_y(self) -> float:
        """|y| of the alcove's own back wall."""
        return CORRIDOR_HALF_WIDTH + self.depth

    @property
    def usable_outer_y(self) -> float:
        """|y| of the first surface the duck cannot pass, wall or obstruction."""
        if self.blocked_from is None:
            return self.outer_y
        return min(self.outer_y, self.blocked_from)

    @property
    def blocked(self) -> bool:
        return self.blocked_from is not None and self.blocked_from < self.outer_y

    @property
    def max_trunk_abs_y(self) -> float:
        """Largest |y| the trunk centre can reach with its footprint inside."""
        return self.usable_outer_y - DUCK_PLANAR_RADIUS

    @property
    def clears_passage(self) -> bool:
        """Can the duck's whole footprint get out of the centre passage here?"""
        return self.max_trunk_abs_y >= CLEAR_ABS_Y

    @property
    def clearance_headroom_m(self) -> float:
        """How much further out than strictly required the duck could stand."""
        return self.max_trunk_abs_y - CLEAR_ABS_Y

    @property
    def park_y(self) -> float:
        """Signed trunk-centre y the duck parks at inside this alcove.

        Half-way between the minimum that clears the passage and the maximum
        the recess allows, so the pull-over is shaved against neither the
        passage nor the back wall.
        """
        return self.side * 0.5 * (CLEAR_ABS_Y + self.max_trunk_abs_y)

    @property
    def x_span(self) -> tuple[float, float]:
        return (self.center_x - self.half_length_x,
                self.center_x + self.half_length_x)

    @property
    def x_headroom_m(self) -> float:
        """Slack along the corridor axis for the duck's footprint in the mouth."""
        return self.half_length_x - DUCK_PLANAR_RADIUS

    @property
    def entry_x(self) -> float:
        """The station at which the duck may begin stepping into this recess.

        The first point at which the whole footprint is between the two cheeks,
        plus a small margin.  The reachability estimate is scored against this
        station because it is where the lateral leg starts; the duck keeps
        walking to :attr:`park_x` underneath the step, which is what makes the
        two legs concurrent.
        """
        low, _high = self.x_span
        return low + DUCK_PLANAR_RADIUS + ENTRY_MARGIN_M

    @property
    def sightline_half_span_m(self) -> float:
        """Half-width of the corridor an occupant of this recess can actually see.

        A duck parked at the back of a recess looks out through its mouth, and
        the two cheeks are opaque walls.  By similar triangles, a sightline from
        ``park_y`` to a point on the corridor centreline crosses the wall line a
        fraction ``(|park_y| - CORRIDOR_HALF_WIDTH) / |park_y|`` of the way, so
        the mouth's ``half_length_x`` projects to ``half_length_x / fraction``
        at the centreline.

        This is a PROPERTY OF THE ARCHITECTURE, not a tuning parameter: with a
        0.80 m mouth and a park point 0.397 m off-centre it comes to +/-0.849 m,
        and no amount of neck travel changes it.  The tracking gate is graded
        over exactly this window, because requiring the duck to see a person it
        is physically walled off from would be a gate no robot could pass.
        Outside the window the adult is simply behind a wall.
        """
        park = abs(self.park_y)
        if park <= CORRIDOR_HALF_WIDTH:
            return float("inf")
        fraction = (park - CORRIDOR_HALF_WIDTH) / park
        return self.half_length_x / fraction

    @property
    def park_x(self) -> float:
        """The station the duck comes to rest at inside this recess.

        The middle of the mouth, so the parked footprint is shaved against
        neither cheek.  MEASURED NECESSITY: driving only to ``entry_x`` left
        the duck 5 cm inside the mouth's near cheek, and the settle drift after
        the lateral command was released pushed its bounding footprint back
        across that boundary — the wall-clearance gate measured -0.0235 m
        against ``bay_far_cheek_lo`` for the whole of a yield.  The park station
        is the centre, which leaves 0.27 m of slack at each cheek.
        """
        return self.center_x

    def contains_x(self, x: float) -> bool:
        low, high = self.x_span
        return low <= x <= high

    def footprint_inside(self, x: float, y: float,
                         radius: float = DUCK_PLANAR_RADIUS) -> bool:
        """Is a footprint at ``(x, y)`` wholly within this recess's mouth?"""
        low, high = self.x_span
        if x - radius < low or x + radius > high:
            return False
        return abs(y) + radius <= self.usable_outer_y + 1e-9


# THE FOUR BAYS.  Two are unusable for reasons a viewer can SEE, and the
# scorer refuses them from the same geometry the scene was generated from:
#
#   bay_shallow  a 0.10 m niche.  Usable outer |y| 0.31, so the trunk centre
#                can reach 0.1797 against the 0.3343 it needs.  Too shallow by
#                0.155 m, whatever the duck does.
#   bay_crates   a full-depth 0.32 m recess with a stack of crates in it.  The
#                crates begin at |y| = 0.34, leaving 0.13 m of usable depth, so
#                the trunk centre can reach 0.2097.  Short by 0.125 m.
#   bay_open     a full-depth 0.32 m recess, empty.  Trunk centre reaches
#                0.3997: clears the passage with 0.065 m to spare.
#   bay_far      the same on the opposite wall, further down the corridor.
# THE FOUR BAYS.  Two are unusable for reasons a viewer can SEE, and the
# scorer refuses them from the same geometry the scene was generated from:
#
#   bay_shallow  a 0.10 m niche.  Usable outer |y| 0.31, so the trunk centre
#                can reach 0.1797 against the 0.3343 it needs.  Too shallow by
#                0.155 m, whatever the duck does.
#   bay_crates   a full-depth 0.38 m recess with a stack of crates in it.  The
#                crates begin at |y| = 0.34, leaving 0.13 m of usable depth, so
#                the trunk centre can reach 0.2097.  Short by 0.125 m.
#   bay_open     a full-depth 0.38 m recess, empty.  Trunk centre reaches
#                0.4597: clears the passage with 0.125 m to spare.
#   bay_far      the same on the opposite wall, further down the corridor, for
#                the second encounter.
#
# THE MOUTHS ARE LONG ENOUGH TO ENTER WITHOUT SCRAPING A CHEEK.  The duck's
# footprint is 0.26 m across, and the lateral step may only begin once that
# footprint is wholly between the mouth's two cheeks, so a mouth barely longer
# than the robot would force the entry to be stop-then-step rather than the
# single diagonal move the measured timings are built on.  A 0.80 m mouth
# leaves 0.54 m of room for the two legs to overlap.
#
# THE STATIONS ARE CHOSEN SO THE FIRST TWO REJECTIONS ARE NOT WON BY DISTANCE.
# At the moment of the first decision the duck sits at x ≈ -1.17, and both
# ``bay_shallow`` and ``bay_crates`` lie AHEAD of it and are comfortably
# reachable in time — more comfortably than the bay it actually chooses.  Their
# refusals are therefore pure physical-clearance failures, which is the claim
# this behavior exists to make.  Consecutive mouths on the SAME wall never
# overlap, or the wall-segment builder would emit one merged opening instead of
# two distinct bays; the shallow niche and the open recess are 0.08 m apart on
# the -Y wall, and the crates bay sits opposite them on the +Y wall.
ALCOVES: tuple[Alcove, ...] = (
    Alcove("bay_shallow", center_x=-0.98, half_length_x=0.24, side=-1,
           depth=0.10, label="shallow niche"),
    Alcove("bay_crates", center_x=-0.92, half_length_x=0.34, side=+1,
           depth=0.38, blocked_from=0.34, label="blocked by crates"),
    Alcove("bay_open", center_x=-0.30, half_length_x=0.40, side=-1,
           depth=0.38, label="open recess"),
    Alcove("bay_far", center_x=+0.95, half_length_x=0.40, side=+1,
           depth=0.38, label="open recess"),
)
ALCOVE_BY_NAME: dict[str, Alcove] = {a.name: a for a in ALCOVES}
ALCOVE_NAMES: tuple[str, ...] = tuple(a.name for a in ALCOVES)


# --- footprint helpers -------------------------------------------------------
def duck_span_y(trunk_y: float,
                radius: float = DUCK_PLANAR_RADIUS) -> tuple[float, float]:
    """The duck's footprint along y at a given trunk centre."""
    return (trunk_y - radius, trunk_y + radius)


def clears_center_passage(trunk_y: float,
                          radius: float = DUCK_PLANAR_RADIUS) -> bool:
    """True when NO part of the duck's footprint is inside the centre passage."""
    low, high = duck_span_y(trunk_y, radius)
    return low >= CENTER_PASSAGE_HALF or high <= -CENTER_PASSAGE_HALF


def center_passage_intrusion(trunk_y: float,
                             radius: float = DUCK_PLANAR_RADIUS) -> float:
    """How far the duck's footprint reaches into the passage; <=0 means clear.

    Reported as a signed depth so the HUD can show the pull-over closing on
    zero rather than flipping a boolean.
    """
    low, high = duck_span_y(trunk_y, radius)
    if low >= CENTER_PASSAGE_HALF:
        return -(low - CENTER_PASSAGE_HALF)
    if high <= -CENTER_PASSAGE_HALF:
        return -(-CENTER_PASSAGE_HALF - high)
    return min(high, CENTER_PASSAGE_HALF) - max(low, -CENTER_PASSAGE_HALF)


def local_half_width(x: float, side: int) -> float:
    """Free half-width on one side of the corridor at station ``x``.

    Inside an alcove's mouth this is the recess's usable limit; everywhere else
    it is the plain corridor wall.  Obstructions count as wall.
    """
    if x > CORRIDOR_X_MAX:
        return LOBBY_HALF_WIDTH
    limit = CORRIDOR_HALF_WIDTH
    for alcove in ALCOVES:
        if alcove.side == side and alcove.contains_x(x):
            limit = max(limit, alcove.usable_outer_y)
    return limit


def wall_clearance(x: float, y: float,
                   radius: float = DUCK_PLANAR_RADIUS) -> float:
    """Signed clearance from a footprint at ``(x, y)`` to the nearest wall.

    A coarse geometric companion to the exact per-geom wall probe: the
    controller uses it to keep a pull-over inside its recess, while the
    acceptance gate grades the real thing against the scene's actual geoms.
    """
    side = +1 if y >= 0.0 else -1
    return local_half_width(x, side) - abs(y) - radius


def at_destination(trunk_x: float) -> bool:
    """True once the trunk centre has reached the painted threshold."""
    return trunk_x >= DESTINATION_X


def corridor_passing_geometry(
    duck_half: float = DUCK_LATERAL_HALF,
    adult_half: float = ADULT_LATERAL_HALF,
    half_width: float = CORRIDOR_HALF_WIDTH,
    safe_gap: float = SAFE_PASSING_GAP_M,
) -> dict:
    """Can the duck and an adult pass side by side in the plain corridor?

    Returns the full arithmetic rather than a verdict alone, so the README, the
    HUD and the tests all quote the same numbers the scenario was built on.
    """
    max_separation = 2.0 * half_width - duck_half - adult_half
    touching_separation = duck_half + adult_half
    safe_separation = touching_separation + safe_gap
    return {
        "corridor_half_width_m": half_width,
        "corridor_width_m": 2.0 * half_width,
        "duck_lateral_half_m": duck_half,
        "adult_lateral_half_m": adult_half,
        "max_centre_separation_m": max_separation,
        "separation_to_touch_m": touching_separation,
        "separation_to_pass_safely_m": safe_separation,
        "best_possible_surface_gap_m": max_separation - touching_separation,
        "fits_at_all": max_separation >= touching_separation,
        "fits_safely": max_separation >= safe_separation,
        "safe_gap_m": safe_gap,
        "shortfall_m": safe_separation - max_separation,
    }


def counterfactual_pass_clearance(duck_y: float, adult_y: float,
                                  duck_half: float = DUCK_LATERAL_HALF,
                                  adult_half: float = ADULT_LATERAL_HALF
                                  ) -> float:
    """Surface gap the pass WOULD have had at these two lateral offsets.

    Negative means the two bodies would have overlapped.  This is the honest
    counterfactual for "what if the duck had not pulled over": it is evaluated
    at the duck's own lateral position when it first detected the encounter, so
    it reports the outcome of doing nothing rather than a hypothetical.
    """
    return abs(duck_y - adult_y) - duck_half - adult_half


# --- scenery the video needs -------------------------------------------------
# Doors and pipes along the walls, purely as depth cues so the corridor reads
# as an indoor space rather than two grey slabs.  Non-colliding, like every
# other piece of scenery in this lab.
WALL_DOORS: tuple[tuple[str, float, int], ...] = (
    ("door_a", -2.25, +1),
    ("door_b", +0.25, -1),
    ("door_c", +1.80, +1),
)
CEILING_LIGHTS: tuple[float, ...] = (-2.3, -1.6, -0.9, -0.2, 0.5, 1.2, 1.9)
