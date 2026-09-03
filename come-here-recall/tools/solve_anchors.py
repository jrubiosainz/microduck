#!/usr/bin/env python3
"""Solve the five pacing anchors against the duck's own predicted recall path.

Hand-placing anchors on a ring is the obvious approach and it is wrong here,
for a reason that is only visible once the gait has been measured:

* the stock policy CANNOT turn in place (``wz=+/-0.85`` at ``vx=0`` moves the
  trunk 7.8 deg / -9.5 deg in six seconds), so every call bearing is flown as
  an arc and costs both time and ground;
* right turns are measurably faster than left (-31.0 vs +26.8 deg/s at
  ``vx=0.28``), so a layout that happens to demand left turns is slower;
* after each recall the duck STOPS one standoff short of that caller, so the
  next approach starts from there, not from the origin.  A symmetric ring
  therefore produces approaches of wildly different length.

This script searches anchor placements so that, replaying the recall sequence
geometrically, every approach range lands in a target band, every consecutive
pair of call bearings differs by a large angle, the turns are predominantly to
the faster side, and no two adults ever pace within a minimum separation.

    python tools/solve_anchors.py

It prints the best layout found and the measured properties of that layout.
The chosen numbers are then pasted into ``scripts/people_routes.py`` as
literals, so the behavior itself never depends on this search.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from people_routes import Station  # noqa: E402

CALL_ORDER = ("red", "yellow", "green")
NON_CALLERS = ("blue", "purple")
# Approach range band.  Long enough that the approach is a real walk with a
# real turn, short enough that three of them fit inside a 35-55 s video at the
# measured path speed of about 0.26 m/s.
RANGE_LO, RANGE_HI = 1.75, 2.25
STANDOFF = 0.60
MIN_BEARING_SEPARATION_DEG = 110.0
MIN_PERSON_SEPARATION = 1.20
# Keep everyone inside the wide shot.
MAX_ANCHOR_RADIUS = 2.60
# The duck must never be standing inside somebody.  This is checked against the
# duck's ACTUAL resting points - the origin plus the three standoff points the
# recall sequence produces - rather than as a ring around the origin, because
# the duck does not stay at the origin.  A ring constraint rejected 97% of
# otherwise-valid layouts while protecting a place the duck only occupies for
# the first few seconds.
#
# The caller currently being approached is exempt: closing to the standoff
# distance is the entire point of the behavior.
MIN_BYSTANDER_CLEARANCE = 0.85
CALL_TIMES = (3.0, 17.0, 31.0)
ROLLOUT_S = 52.0

RADII = {
    "red": (0.30, 0.20), "yellow": (0.32, 0.19), "green": (0.31, 0.21),
    "blue": (0.33, 0.21), "purple": (0.29, 0.24),
}
PERIOD = {"red": 29.0, "yellow": 31.0, "green": 26.0, "blue": 24.0, "purple": 27.5}
ROTATION = {"red": -20.0, "yellow": 105.0, "green": 35.0, "blue": 150.0,
            "purple": 70.0}
PHASE = {"red": 60.0, "yellow": 310.0, "green": 200.0, "blue": 120.0, "purple": 25.0}
REVERSE = {"red": True, "yellow": False, "green": False, "blue": True,
           "purple": False}


def make_station(name: str, anchor: tuple[float, float]) -> Station:
    return Station(
        name=name, anchor=anchor, radii=RADII[name], period=PERIOD[name],
        rotation=math.radians(ROTATION[name]), phase=math.radians(PHASE[name]),
        reverse=REVERSE[name],
    )


def evaluate(anchors: dict[str, tuple[float, float]]) -> dict | None:
    """Replay the recall sequence geometrically; ``None`` if a hard rule fails."""
    stations = {name: make_station(name, xy) for name, xy in anchors.items()}

    # Everyone must stay inside the wide shot for the whole rollout.
    for station in stations.values():
        for t in np.arange(0.0, ROLLOUT_S, 0.5):
            if float(np.linalg.norm(station.at(float(t))[0])) > MAX_ANCHOR_RADIUS:
                return None

    # Nobody may ever pace into anybody else.  0.5 s sampling with 0.04-0.09 m/s
    # pacing speeds moves any adult under 45 mm between samples, far below the
    # 1.20 m separation floor, so the minimum cannot hide between samples.
    separation = float("inf")
    for t in np.arange(0.0, ROLLOUT_S, 0.5):
        positions = {n: s.at(float(t))[0] for n, s in stations.items()}
        names = list(positions)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                separation = min(
                    separation,
                    float(np.linalg.norm(positions[first] - positions[second])),
                )
    if separation < MIN_PERSON_SEPARATION:
        return None

    duck = np.zeros(2)
    yaw = 0.0
    legs = []
    bearings = []
    rest_points = [(np.zeros(2), None)]
    for name, call_t in zip(CALL_ORDER, CALL_TIMES):
        target = stations[name].at(call_t)[0]
        delta = target - duck
        distance = float(np.linalg.norm(delta))
        if not RANGE_LO <= distance <= RANGE_HI:
            return None
        bearing = math.atan2(float(delta[1]), float(delta[0]))
        turn = math.degrees((bearing - yaw + math.pi) % (2 * math.pi) - math.pi)
        legs.append({"caller": name, "range_m": distance,
                     "bearing_deg": math.degrees(bearing), "turn_deg": turn})
        bearings.append(math.degrees(bearing))
        duck = target - STANDOFF * delta / distance
        yaw = bearing
        rest_points.append((duck.copy(), name))

    # The duck must never come to rest inside a BYSTANDER.  Closing on the
    # caller it was summoned by is the behavior; brushing anyone else is not.
    bystander_clearance = float("inf")
    for rest, exempt in rest_points:
        for name, station in stations.items():
            if name == exempt:
                continue
            for t in np.arange(0.0, ROLLOUT_S, 0.5):
                bystander_clearance = min(
                    bystander_clearance,
                    float(np.linalg.norm(station.at(float(t))[0] - rest)),
                )
    if bystander_clearance < MIN_BYSTANDER_CLEARANCE:
        return None

    for i in range(len(bearings)):
        for j in range(i + 1, len(bearings)):
            gap = abs((bearings[i] - bearings[j] + 180.0) % 360.0 - 180.0)
            if gap < MIN_BEARING_SEPARATION_DEG:
                return None

    # Prefer right turns (measurably faster) and bigger, more legible turns.
    right_turns = sum(1 for leg in legs[1:] if leg["turn_deg"] < 0)
    turn_size = min(abs(leg["turn_deg"]) for leg in legs[1:])
    score = (
        right_turns * 100.0
        + turn_size
        + separation * 8.0
        + bystander_clearance * 6.0
        - 20.0 * max(0.0, max(leg["range_m"] for leg in legs) - 2.1)
    )
    return {"legs": legs, "separation_m": separation, "score": score,
            "right_turns": right_turns, "anchors": anchors,
            "bystander_clearance_m": bystander_clearance}


def _loop_positions(station: Station, times: np.ndarray) -> np.ndarray:
    return np.array([station.at(float(t))[0] for t in times])


def _min_pair_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between two adults sampled at the SAME instants."""
    return float(np.min(np.linalg.norm(a - b, axis=1)))


def place_non_callers(
    caller_anchors: dict[str, tuple[float, float]], times: np.ndarray
) -> dict[str, tuple[float, float]] | None:
    """Place the two non-callers GREEDILY instead of drawing them at random.

    Random placement is what made the search fail: five adults pacing 0.3 m
    loops inside a 2.6 m disc with a 1.20 m separation floor is a tight packing,
    and two independent uniform draws land on top of somebody 97% of the time
    (measured: 584 of 600 candidate layouts rejected on separation alone).

    Sweeping a polar grid and keeping the position with the largest clearance
    turns that into a deterministic placement that succeeds whenever the caller
    geometry leaves room at all.
    """
    loops = {
        name: _loop_positions(make_station(name, xy), times)
        for name, xy in caller_anchors.items()
    }
    for i, first in enumerate(CALL_ORDER):
        for second in CALL_ORDER[i + 1:]:
            if _min_pair_distance(loops[first], loops[second]) < MIN_PERSON_SEPARATION:
                return None

    placed: dict[str, tuple[float, float]] = {}
    for name in NON_CALLERS:
        best_xy = None
        best_clearance = 0.0
        for radius in np.arange(1.35, 2.35, 0.05):
            for degrees in np.arange(0.0, 360.0, 4.0):
                angle = math.radians(float(degrees))
                anchor = (round(float(radius) * math.cos(angle), 2),
                          round(float(radius) * math.sin(angle), 2))
                candidate = _loop_positions(make_station(name, anchor), times)
                if float(np.max(np.linalg.norm(candidate, axis=1))) > MAX_ANCHOR_RADIUS:
                    continue
                clearance = min(
                    _min_pair_distance(candidate, other) for other in loops.values()
                )
                if clearance > best_clearance:
                    best_clearance = clearance
                    best_xy = anchor
        if best_xy is None or best_clearance < MIN_PERSON_SEPARATION:
            return None
        placed[name] = best_xy
        loops[name] = _loop_positions(make_station(name, best_xy), times)
    return placed


def solve_callers(rng: random.Random) -> dict | None:
    """CONSTRUCT the three caller anchors instead of guessing them.

    Rejection sampling over five free anchors does not work here: the three
    approach ranges are CHAINED (each starts where the previous recall stopped),
    so three independent draws almost never land three ranges inside a 0.50 m
    band at once.  Measured: 120k random layouts produced no feasible sample.

    Constructing forward is exact.  Walk the sequence: from the duck's current
    pose choose the next call BEARING directly (which is the quantity the
    scenario actually cares about), choose a RANGE inside the band, place the
    caller there, and advance the duck to the standoff point.  Every range and
    every bearing is then correct by construction, and only the cheap global
    rules - anchor radius, pacing separation - can still reject a candidate.

    The bearings are drawn so each successive turn is a large RIGHT turn, which
    the measured yaw asymmetry makes the faster direction.
    """
    duck = np.zeros(2)
    yaw = 0.0
    anchors: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(CALL_ORDER):
        # A right turn of 100-150 deg from the current heading.
        turn = math.radians(-rng.uniform(100.0, 150.0))
        if index == 0:
            # The first call comes from ahead-ish, so the opening search is a
            # short sweep rather than a full rear scan; the later calls are the
            # ones that force a big turn.
            turn = math.radians(rng.uniform(-40.0, 40.0))
        bearing = yaw + turn
        distance = rng.uniform(RANGE_LO, RANGE_HI)
        # The anchor is where the caller must BE at their call time; the pacing
        # loop offsets them from the anchor, so invert that offset.
        station = make_station(name, (0.0, 0.0))
        offset = station.at(CALL_TIMES[index])[0]
        target = duck + distance * np.array([math.cos(bearing), math.sin(bearing)])
        anchor = target - offset
        anchors[name] = (round(float(anchor[0]), 2), round(float(anchor[1]), 2))
        duck = target - STANDOFF * np.array([math.cos(bearing), math.sin(bearing)])
        yaw = bearing
    for name in NON_CALLERS:
        angle = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(1.40, 2.30)
        anchors[name] = (round(radius * math.cos(angle), 2),
                         round(radius * math.sin(angle), 2))
    return anchors


def main() -> int:
    rng = random.Random(20260901)
    times = np.arange(0.0, ROLLOUT_S, 0.5)
    best = None
    attempts = 0
    for _ in range(260):
        caller_anchors = solve_callers(rng)
        if caller_anchors is None:
            continue
        caller_anchors = {k: v for k, v in caller_anchors.items() if k in CALL_ORDER}
        non_callers = place_non_callers(caller_anchors, times)
        if non_callers is None:
            continue
        attempts += 1
        anchors = {**caller_anchors, **non_callers}
        result = evaluate(anchors)
        if result is not None and (best is None or result["score"] > best["score"]):
            best = result
            print(f"score={result['score']:.2f} sep={result['separation_m']:.3f} "
                  f"right={result['right_turns']} "
                  + "  ".join(f"{leg['caller']}:{leg['range_m']:.2f}m/"
                              f"{leg['turn_deg']:+.0f}deg" for leg in result["legs"]))
    if best is None:
        print(f"no layout satisfied every constraint ({attempts} reached scoring)")
        return 1
    print("\nBEST LAYOUT")
    for name, xy in best["anchors"].items():
        print(f'    Station("{name}", anchor=({xy[0]:+.2f}, {xy[1]:+.2f}), '
              f'radii={RADII[name]}, period={PERIOD[name]}, '
              f'rotation=math.radians({ROTATION[name]}), '
              f'phase=math.radians({PHASE[name]}), reverse={REVERSE[name]}),')
    print(f"\nmin person separation {best['separation_m']:.3f} m")
    print(f"min bystander clearance at rest {best['bystander_clearance_m']:.3f} m")
    for leg in best["legs"]:
        print(f"  {leg['caller']:7s} range {leg['range_m']:.2f} m  "
              f"bearing {leg['bearing_deg']:+7.1f}  turn {leg['turn_deg']:+7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
