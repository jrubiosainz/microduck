#!/usr/bin/env python3
"""The duck's own physical route through the building, and the holding points
along it.

The route is one continuous filleted path from the start, through the automatic
door, across the lobby to the holding spot beside the lift, into the car, across
it to the side, and out through the rear doors to the arrival point.  It is
built once, at start-up, from geometry the duck could have measured: the
apertures' own centres, the zone module's derived holding spots, and its own
starting pose.

WHY ONE ROUTE AND NOT A SEARCH
-------------------------------
The sibling ``lead-me-somewhere`` behavior searches its route with A\\* because
the question there is whether the robot can find a way through a hall it has not
been told about.  That is not the question here.  Here the building offers
exactly one way through — a single doorway, a single lift — and the behavior
under test is the ETIQUETTE of using them: when to stop, whom to let past, and
in what order to move.  Inventing a search over a corridor with one option would
be theatre, so the route is CONSTRUCTED and its properties are then MEASURED and
gated: every bend must fit the duck's measured turning circle, every waypoint
must sit at a positive clearance from the real scenery, and the aperture
crossings must pass through the middle of the openings.

THE ROUTE IS SPLIT INTO LEGS, AND THE MACHINE ADVANCES ALONG THEM
-------------------------------------------------------------------
A single arc-length cursor over the whole path would let the duck sail through
the door while the state machine was still in YIELD_EXITERS.  Instead each leg
ends at a holding point, and the controller's pursuit target is CLAMPED to the
end of the leg the current state authorises.  Stopping is then structural: the
duck cannot walk past a holding point the machine has not released, because
there is no target beyond it.

EVERY BEND IS CHECKED AGAINST THE MEASURED TURNING CIRCLE
-----------------------------------------------------------
The duck's MEASURED minimum radius is 0.40 m turning right and 0.36 m turning
left, both taken at each sign's own command ceiling.
:func:`route_bend_report` reports every arc with the radius it needs and the
radius that sign can actually deliver, and ``tools/check_layout.py`` refuses a
route containing one that cannot be walked.  The route's own
:data:`CORNER_RADIUS_M` sits above both, with the tightest corner in the whole
journey — the 98 deg turn inside the car — checked against the right-hand
figure it uses.
"""

from __future__ import annotations

import numpy as np

from etiquette_route import Route
from etiquette_states import (
    DUCK_START_XY,
    MIN_LEFT_TURN_RADIUS_M,
    MIN_RIGHT_TURN_RADIUS_M,
)
from etiquette_zones import CABIN_HOLD_XY, WAIT_SIDE_XY
from lobby_doors import APERTURES
from lobby_layout import (
    DOOR_CENTER_Y,
    DOOR_WALL_X,
    LIFT_FRONT_Y,
    LIFT_WALL_X,
    REAR_WALL_X,
    REAR_Y,
)

# Corner radius for the duck's own route.  Above BOTH measured minima (0.40 m
# right, 0.36 m left) with margin, and small enough that every corner in the
# journey - including the 88 deg turn out of the lift holding spot and the 62 deg
# turn inside a 1.64 x 1.95 m car - can actually be filleted.
#
# 0.46 m was the first choice and ``etiquette_route._build`` rejected it: the
# turn off the lift holding point needed 0.4433 m of cutback against 0.4159 m of
# available leg.  Before that exception existed the corner was silently left as
# a HARD VERTEX, which turns the walker through the whole corner in one control
# tick.  ``tools/check_layout.py`` checks every resulting arc against its own
# sign's measured minimum rather than against this constant.
CORNER_RADIUS_M = 0.41

# How far short of each aperture plane the approach holding point sits.  Derived
# from the threshold band's own depth so the duck stops OUTSIDE it rather than
# on its edge: the band is 0.62 m deep, so a holding point 0.86 m out leaves the
# duck's 0.13 m footprint 0.11 m clear of the band it must not enter.
DOOR_HOLD_BACK_M = 0.86

# Where the journey ends, beyond the rear doors on the target floor.  Far enough
# past the sill that the duck is unambiguously off the lift and the video shows
# it standing on the target floor, and no further: each extra 0.11 m of this leg
# is a MEASURED second of video at the careful pace it leaves the cabin on.
ARRIVAL_XY = (3.52, -0.48)

# The arc length either side of an aperture plane over which the duck uses its
# CAREFUL command instead of its cruise.  Derived from the aperture box's own
# half-depth (0.22 m) plus a fillet's worth of run-in, so the duck is already
# slowed before it reaches an opening and does not accelerate away until it is
# clear of one.
#
# THIS BAND IS EXPENSIVE AND THE COST WAS MEASURED.  Three apertures at 2 x
# 0.55 m is 3.3 m of the 8.12 m route walked at the MEASURED 0.096 m/s instead
# of 0.129 m/s, which is 8.8 s of the video.  A first draft used 0.70 m and cost
# 11.2 s.  It is kept - walking carefully through a doorway is the behavior, not
# an overhead - but trimmed to the smallest band that still covers the whole
# aperture box with a fillet's margin either side.
CAREFUL_BAND_M = 0.55


def door_hold_xy() -> np.ndarray:
    """Where the duck stops to let the exiters out.

    On the door's own axis, ``DOOR_HOLD_BACK_M`` west of the plane: square to
    the opening, so it is visibly waiting AT the door rather than loitering
    beside it, and far enough back that its whole footprint is outside the
    threshold band.
    """
    return np.array([DOOR_WALL_X - DOOR_HOLD_BACK_M, DOOR_CENTER_Y])


# The five legs.  Each ends at a holding point the machine must release before
# the controller is given a target beyond it.
#
# LEG 0  APPROACH_DOOR    start           -> door hold
# LEG 1  FOLLOW_THROUGH   door hold       -> through the aperture into the lobby
# LEG 2  APPROACH_LIFT    lobby           -> the holding spot beside the doors
# LEG 3  FOLLOW_...IN     wait side       -> through the lift aperture, cabin hold
# LEG 4  FOLLOW_OUT       cabin hold      -> through the rear aperture, arrival
LEG_NAMES: tuple[str, ...] = (
    "approach_door", "through_door", "approach_lift", "board_cabin",
    "leave_cabin",
)

# Which leg each state is allowed to walk.  A state absent from this map has no
# walking target at all, which is how every zero-command state gets its zero
# structurally rather than by the controller remembering to return one.
STATE_LEG: dict[str, int] = {
    "APPROACH_DOOR": 0,
    "FOLLOW_THROUGH": 1,
    "APPROACH_LIFT": 2,
    "FOLLOW_GUARDIAN_IN": 3,
    "POSITION_INSIDE": 3,
    "FOLLOW_OUT": 4,
}


def _corners() -> list[tuple[float, float]]:
    """The whole route's corner list, in order.

    Built from the apertures' own centres and the zone module's derived holding
    spots, so a geometry edit moves the route with the building.  The two points
    either side of each aperture are placed ON the aperture axis so the duck
    crosses through the MIDDLE of the opening rather than clipping a jamb.
    """
    hold = door_hold_xy()
    door_x = float(APERTURES["concourse_door"]["plane_x"])
    lift_x = float(APERTURES["lift_front"]["plane_x"])
    rear_x = float(APERTURES["lift_rear"]["plane_x"])
    return [
        (float(DUCK_START_XY[0]), float(DUCK_START_XY[1])),
        (float(hold[0]), float(hold[1])),
        # Straight through the doorway on its own axis, and far enough past it
        # that the fillet into the lobby leg starts outside the opening.
        (door_x, DOOR_CENTER_Y),
        (door_x + 0.62, DOOR_CENTER_Y),
        # Across the lobby to the holding spot beside the lift doors.  The
        # 0.70 m run-in is what lets the fillet into that spot complete before
        # the duck arrives at it, so it reaches the holding point already
        # square rather than mid-turn.
        (float(WAIT_SIDE_XY[0]) - 0.70, float(WAIT_SIDE_XY[1])),
        (float(WAIT_SIDE_XY[0]), float(WAIT_SIDE_XY[1])),
        # Round onto the lift axis, then STRAIGHT through the front aperture
        # and over the sill before any turn begins.  This corner and the one
        # past the plane are both MEASURED rather than chosen.
        #
        # TWO CONSTRAINTS FIGHT HERE AND BOTH ARE REAL.  Sweeping the corner
        # against the route's worst static clearance, a first draft's fillet
        # grazed the north lift jamb with 0.058 m of clear air for a 0.1162 m
        # robot.  Pulling the corner WEST opens that up - but the 88 deg turn
        # off the holding point then needs more cutback than the leg has room
        # for, and ``etiquette_route._build`` refuses to leave a hard vertex.
        # x = 0.72 gives that turn enough run-in for a 0.41 m fillet while
        # keeping the jamb clearance comfortable, and the duck still crosses the
        # sill square to the doors as a person does.
        #
        # THERE IS NO WAYPOINT ON THE APERTURE PLANE ITSELF, AND THAT IS LOAD
        # BEARING.  A draft placed one at ``(lift_x, LIFT_FRONT_Y)``, collinear
        # with the two either side of it, which looked harmless.  It is not:
        # ``_build`` allows a corner only half of the following leg as cutback
        # room, so an extra collinear vertex HALVED the room available to the
        # 88 deg turn before it and made the route unbuildable.  The straight
        # run through the opening is one leg, and the crossing is verified by
        # ``aperture_crossings`` rather than by a waypoint sitting on it.
        (0.72, LIFT_FRONT_Y),
        (lift_x + 0.34, LIFT_FRONT_Y),
        # Across the car to the holding spot at its side.
        (float(CABIN_HOLD_XY[0]), float(CABIN_HOLD_XY[1])),
        # Round onto the rear axis and out.
        (rear_x - 0.40, REAR_Y),
        (rear_x, REAR_Y),
        (float(ARRIVAL_XY[0]), float(ARRIVAL_XY[1])),
    ]


# Index into :func:`_corners` at which each leg ENDS.  The route is one Route
# object; the legs are arc-length cuts through it, computed once below.
_LEG_END_CORNER: tuple[int, ...] = (1, 3, 5, 8, 11)


def build_route() -> Route:
    """The duck's whole physical route, filleted and constant-speed."""
    return Route("duck", tuple(_corners()), speed=1.0,
                 radius=CORNER_RADIUS_M)


def leg_bounds(route: Route) -> list[float]:
    """Arc length at which each leg ends.

    Computed by projecting each leg's own end CORNER onto the filleted route,
    rather than by summing declared lengths: the fillet shortens the path near
    every bend, so a sum of leg lengths would not agree with the curve the duck
    actually walks.  A holding point that is itself a corner is unaffected by
    its own fillet only when the route is straight through it, which is why the
    corner list places a straight run either side of each holding point.
    """
    corners = _corners()
    bounds: list[float] = []
    for index in _LEG_END_CORNER:
        target = np.asarray(corners[index], dtype=np.float64)
        best_s, best_d = 0.0, float("inf")
        samples = 4000
        for step in range(samples + 1):
            s = route.length * step / samples
            position, _ = route.pose_at_arc(s)
            distance = float(np.linalg.norm(position - target))
            if distance < best_d:
                best_d, best_s = distance, s
        bounds.append(best_s)
    return bounds


def route_bend_report(route: Route) -> list[dict]:
    """Every bend, with the radius its SIGN can actually deliver.

    A left bend needs :data:`MIN_LEFT_TURN_RADIUS_M` and a right bend
    :data:`MIN_RIGHT_TURN_RADIUS_M`; both are DERIVED from the yaw sweep rather
    than declared.  ``walkable`` is what ``tools/check_layout.py`` refuses on.
    """
    report = []
    for bend in route.corner_report():
        needed = (MIN_LEFT_TURN_RADIUS_M if bend["hand"] == "left"
                  else MIN_RIGHT_TURN_RADIUS_M)
        report.append({
            **bend,
            "min_radius_for_hand_m": round(needed, 4),
            "walkable": bend["radius_m"] >= needed,
        })
    return report


def aperture_crossings(route: Route, samples: int = 4000) -> list[dict]:
    """Where the route crosses each aperture plane, and how centred it is.

    Reported rather than assumed so a test can require the duck's path to pass
    through the MIDDLE of every opening: a route that clipped a jamb would show
    up here as an offset approaching the clear half width.
    """
    crossings = []
    for name, spec in APERTURES.items():
        plane = float(spec["plane_x"])
        half_w = 0.5 * float(spec["clear_w"])
        previous = None
        hit = None
        for step in range(samples + 1):
            s = route.length * step / samples
            position, _ = route.pose_at_arc(s)
            side = float(position[0]) - plane
            if previous is not None and previous[1] < 0.0 <= side:
                hit = (s, position)
                break
            previous = (s, side)
        if hit is None:
            crossings.append({"aperture": name, "crossed": False})
            continue
        s, position = hit
        offset = float(position[1]) - float(spec["center_y"])
        crossings.append({
            "aperture": name,
            "crossed": True,
            "arc_s_m": round(s, 4),
            "xy": [round(float(position[0]), 4), round(float(position[1]), 4)],
            "offset_from_centre_m": round(offset, 4),
            "clear_half_w_m": round(half_w, 4),
            "margin_m": round(half_w - abs(offset), 4),
        })
    return crossings


def careful_bands(route: Route) -> list[tuple[float, float]]:
    """Arc-length windows in which the duck walks at its CAREFUL command.

    One band per aperture, centred on the crossing this route makes of it.
    Returned as arc lengths rather than as positions so the controller can test
    membership with a single comparison against the cursor it already has.
    """
    bands: list[tuple[float, float]] = []
    for crossing in aperture_crossings(route):
        if not crossing.get("crossed"):
            continue
        s = float(crossing["arc_s_m"])
        bands.append((max(0.0, s - CAREFUL_BAND_M),
                      min(route.length, s + CAREFUL_BAND_M)))
    return bands


def in_careful_band(bands, arc_s: float) -> bool:
    return any(lo <= arc_s <= hi for lo, hi in bands)
