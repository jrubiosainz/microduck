#!/usr/bin/env python3
"""The zones etiquette is graded in: thresholds, door-centre passages, the cabin,
and the two places the duck is required to stand out of the way.

Every zone here is DERIVED from an aperture in ``lobby_layout``, never declared
separately, so widening a door widens the passage it must not block and moves
the threshold it must not encroach on at the same time.  A zone that could drift
out of step with the geometry it grades would make the whole gate meaningless.

THE FOUR ZONES, AND WHAT EACH ONE IS FOR
------------------------------------------
* **Threshold band** — a slab :data:`THRESHOLD_DEPTH_M` deep on the approach
  side of an aperture, spanning the clear width plus a margin.  Entering it is
  what "encroaching on the doorway" means, and the gate requires the duck to
  stay out of it until the last exiter is clear.  It is a band rather than a
  line because a robot that stops with its nose over the line has encroached.

* **Aperture box** — the opening itself, extruded
  :data:`APERTURE_DEPTH_M` either side of the plane.  Two bodies inside it at
  once is the "side by side through a narrow door" failure, and the gate checks
  occupancy per tick rather than trusting the choreography.

* **Door-centre passage** — the strip of floor directly in front of an
  aperture, :data:`PASSAGE_DEPTH_M` deep and only as wide as the clear opening
  plus :data:`PASSAGE_MARGIN_M`.  This is the corridor exiting occupants need,
  and "waiting beside the lift rather than in front of it" is exactly the claim
  that the duck's own body never intersects it while it waits.

* **Cabin interior** — the four inner faces of the car, inset by the duck's own
  conservative planar radius.  "Inside the cabin" is then a statement about the
  robot's whole footprint rather than about its origin, so a duck whose trunk
  point is inside but whose body overhangs the sill does not count as in.

WHY THE STANDING SPOTS ARE COMPUTED AND NOT TYPED
---------------------------------------------------
:data:`WAIT_SIDE_XY` and :data:`CABIN_HOLD_XY` are derived from the passage and
the cabin so that they are provably outside the passage and provably inside the
cabin.  A hand-typed coordinate that happened to satisfy both would satisfy them
by luck, and a later geometry edit would silently break it.  ``tools/check_layout.py``
re-derives and re-checks both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lobby_doors import APERTURES
from lobby_layout import CABIN_X, CABIN_Y, LIFT_CLEAR_HALF, LIFT_FRONT_Y

# How deep the "do not encroach" band in front of an aperture is.  Derived from
# the duck's own conservative planar diameter (0.2606 m) with headroom: a robot
# a full body length back from the opening is unambiguously not in the way, and
# a shallower band would let it hover at the jamb and still pass.
THRESHOLD_DEPTH_M = 0.62
# Half-depth of the aperture box either side of the plane.  A body within this
# of the plane, inside the clear width, is IN the doorway.
APERTURE_DEPTH_M = 0.22
# How far in front of the lift doors the exit corridor extends.  People stepping
# out need room to clear the sill and turn; a robot standing anywhere in this
# strip is standing in front of the doors.
PASSAGE_DEPTH_M = 1.05
# Extra width the passage carries beyond the clear opening, per side.  An
# occupant does not exit through a slot exactly as wide as the door frame.
PASSAGE_MARGIN_M = 0.16
# Extra width the threshold band carries beyond the clear opening, per side.
THRESHOLD_MARGIN_M = 0.10


@dataclass(frozen=True)
class Band:
    """An axis-aligned planar region, in world coordinates."""

    name: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    def contains(self, xy, radius: float = 0.0) -> bool:
        """Does a disc of ``radius`` at ``xy`` intersect this band at all?"""
        point = np.asarray(xy, dtype=np.float64)
        return bool(
            self.x_range[0] - radius <= float(point[0]) <= self.x_range[1] + radius
            and self.y_range[0] - radius <= float(point[1]) <= self.y_range[1] + radius)

    def depth_into(self, xy, radius: float = 0.0) -> float:
        """How far a disc of ``radius`` at ``xy`` penetrates this band.

        Zero outside, positive inside.  Reported rather than only tested so a
        violation names how badly it violated, and so a gate can require an
        EXACT zero rather than a boolean somebody could weaken later.
        """
        point = np.asarray(xy, dtype=np.float64)
        dx = min(float(point[0]) + radius - self.x_range[0],
                 self.x_range[1] - (float(point[0]) - radius))
        dy = min(float(point[1]) + radius - self.y_range[0],
                 self.y_range[1] - (float(point[1]) - radius))
        return float(max(0.0, min(dx, dy)))

    def center(self) -> np.ndarray:
        return np.array([0.5 * (self.x_range[0] + self.x_range[1]),
                         0.5 * (self.y_range[0] + self.y_range[1])])


def threshold_band(name: str, approach_sign: float) -> Band:
    """The do-not-encroach band on the approach side of an aperture.

    ``approach_sign`` is -1 when the duck approaches from lower x and +1 when it
    approaches from higher x, so the band is always on the side the duck is
    coming FROM.
    """
    spec = APERTURES[name]
    plane = float(spec["plane_x"])
    half_w = 0.5 * float(spec["clear_w"]) + THRESHOLD_MARGIN_M
    near = plane + approach_sign * THRESHOLD_DEPTH_M
    return Band(f"{name}_threshold", (min(plane, near), max(plane, near)),
                (float(spec["center_y"]) - half_w,
                 float(spec["center_y"]) + half_w))


def aperture_box(name: str) -> Band:
    """The opening itself, extruded either side of its plane."""
    spec = APERTURES[name]
    plane = float(spec["plane_x"])
    half_w = 0.5 * float(spec["clear_w"])
    return Band(f"{name}_aperture",
                (plane - APERTURE_DEPTH_M, plane + APERTURE_DEPTH_M),
                (float(spec["center_y"]) - half_w,
                 float(spec["center_y"]) + half_w))


def passage_band(name: str, approach_sign: float) -> Band:
    """The exit corridor in front of an aperture, on the approach side."""
    spec = APERTURES[name]
    plane = float(spec["plane_x"])
    half_w = 0.5 * float(spec["clear_w"]) + PASSAGE_MARGIN_M
    far = plane + approach_sign * PASSAGE_DEPTH_M
    return Band(f"{name}_passage", (min(plane, far), max(plane, far)),
                (float(spec["center_y"]) - half_w,
                 float(spec["center_y"]) + half_w))


# The duck approaches the concourse door from the west (lower x) and the lift
# from the west too; it leaves the cabin eastward through the rear doors.
DOOR_THRESHOLD = threshold_band("concourse_door", -1.0)
DOOR_APERTURE = aperture_box("concourse_door")
LIFT_THRESHOLD = threshold_band("lift_front", -1.0)
LIFT_APERTURE = aperture_box("lift_front")
LIFT_PASSAGE = passage_band("lift_front", -1.0)
REAR_APERTURE = aperture_box("lift_rear")

# The cabin, inset by the duck's conservative planar radius so that "inside"
# is a claim about the whole footprint.  The radius is MEASURED on the built
# model by ``etiquette_states`` and imported rather than re-declared, so the two
# cannot drift apart.
from etiquette_states import DUCK_PLANAR_RADIUS  # noqa: E402

CABIN_INTERIOR = Band("cabin_interior",
                      (CABIN_X[0] + DUCK_PLANAR_RADIUS,
                       CABIN_X[1] - DUCK_PLANAR_RADIUS),
                      (CABIN_Y[0] + DUCK_PLANAR_RADIUS,
                       CABIN_Y[1] - DUCK_PLANAR_RADIUS))


def _spot_beside_passage(clear_m: float) -> np.ndarray:
    """A waiting spot beside the lift doors, provably out of the exit corridor.

    Placed ``clear_m`` beyond the passage's north edge, past the duck's own
    footprint, and set back along the corridor by
    :data:`WAIT_SIDE_BACK_FRACTION`.  Computed rather than typed: see the module
    docstring, and see that constant for why the set-back is a measurement
    rather than a preference.
    """
    y = LIFT_PASSAGE.y_range[1] + DUCK_PLANAR_RADIUS + clear_m
    x = (LIFT_PASSAGE.x_range[1]
         - WAIT_SIDE_BACK_FRACTION * (LIFT_PASSAGE.x_range[1]
                                      - LIFT_PASSAGE.x_range[0]))
    return np.array([x, y])


# How much clear air the duck holds between its own footprint and the exit
# corridor while waiting.  Not zero: standing exactly on the edge of a passage is
# not standing beside it, and a person stepping out at an angle needs slack.
WAIT_SIDE_CLEAR_M = 0.20
# How far BACK from the doors the holding spot sits, as a fraction of the
# passage depth.  0.0 would put the duck level with the doors and 1.0 at the far
# end of the corridor.  0.62 was chosen by SWEEPING the route's own worst static
# clearance over this parameter and the approach corner together: at the front
# of the passage the lobby leg's fillet grazed the north lift jamb with only
# 0.058 m of clear air for a 0.1303 m robot, and pulling the spot back to 0.62
# opens that to 0.1735 m.  A holding spot is not just a place to stand; it sets
# the shape of the corner the duck has to walk out of it.
WAIT_SIDE_BACK_FRACTION = 0.62
WAIT_SIDE_XY = _spot_beside_passage(WAIT_SIDE_CLEAR_M)

# Where the duck stands INSIDE the car: away from both apertures' centrelines,
# so it is out of the way of everybody riding and of the guardian's own exit.
#
# THE POSITION IS SOLVED AGAINST THE ROUTE, NOT POSED.  A first draft put it in
# the far corner at (1.98, 0.71), which is a perfectly good place to stand and an
# unwalkable one to route through: the turn out of it towards the rear doors came
# to 120 deg and needed 0.71 m of cutback against 0.42 m of available leg, so
# ``etiquette_route._build`` refused it.  Sweeping the position against the whole
# route's worst static clearance and its bend list gives (2.10, 0.30): every bend
# walkable, 0.534 m to the nearest cabin face, and 0.21 m of clear air at the
# route's tightest point.  Standing out of the way and being able to get there
# are different constraints, and both are real.
CABIN_HOLD_XY = np.array([2.10, 0.30])


def cabin_contains(xy, radius: float = 0.0) -> bool:
    """Is a disc of ``radius`` at ``xy`` wholly inside the cabin interior?"""
    point = np.asarray(xy, dtype=np.float64)
    return bool(
        CABIN_INTERIOR.x_range[0] + radius <= float(point[0])
        <= CABIN_INTERIOR.x_range[1] - radius
        and CABIN_INTERIOR.y_range[0] + radius <= float(point[1])
        <= CABIN_INTERIOR.y_range[1] - radius)


def cabin_margin_m(xy) -> float:
    """Smallest distance from ``xy`` to any cabin interior face.

    Negative outside.  This is what the "the duck's cabin position is real and
    inside the bounds" gate reports, so the claim carries a number.
    """
    point = np.asarray(xy, dtype=np.float64)
    return float(min(float(point[0]) - CABIN_INTERIOR.x_range[0],
                     CABIN_INTERIOR.x_range[1] - float(point[0]),
                     float(point[1]) - CABIN_INTERIOR.y_range[0],
                     CABIN_INTERIOR.y_range[1] - float(point[1])))


ZONES: dict[str, Band] = {
    band.name: band for band in (
        DOOR_THRESHOLD, DOOR_APERTURE, LIFT_THRESHOLD, LIFT_APERTURE,
        LIFT_PASSAGE, REAR_APERTURE, CABIN_INTERIOR)
}
