#!/usr/bin/env python3
"""Where everybody walks, and when the guardian goes behind the kiosk.

Every adult in this hall moves for the whole rollout.  There is no "crowd
freezes so the duck can think" moment, because a crowd that stops is not an
occlusion problem — it is a diagram.

THE GUARDIAN'S ROUTE IS THE SCENARIO
------------------------------------
She walks a fixed polyline chosen so that the KIOSK, a 1.10 m box, comes
between her and the duck twice, for a sustained period each time, without
anybody scripting the word "hidden".  The duck loses her because a solid body
is in the way, and the acceptance gate measures that with a real MuJoCo ray
cast rather than with a schedule lookup.

* **Cycle 1.**  She rounds the kiosk's north-east corner at about t = 12 s and
  continues west along its north face.  The duck is still on the east side, so
  the whole kiosk stands in the sightline.
* **Cycle 2.**  Having been rejoined, she walks west and passes behind
  ``column_w``, then holds station beyond it.

The look-alikes are timed, not placed: each is scheduled to walk into the duck's
search sweep DURING a loss, so a false candidate becomes camera-visible while
the duck is looking for its guardian, which is when a mistake would actually
cost something.

THE CROWD IS NOT DECORATION EITHER.  Six adults cross the hall on independent
loops at independent speeds and phases; three of them cross the duck's rejoin
corridor, which is what makes "the route stayed clear of people" a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lost_cast import ALL_NAMES, BY_NAME

STRIDE_HZ: float = 1.05


def _smootherstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


@dataclass(frozen=True)
class Waypoint:
    """Be at ``xy`` at time ``t``.  Between waypoints, motion is smoothed."""

    t: float
    xy: tuple[float, float]


@dataclass(frozen=True)
class Walker:
    """One adult following a timed polyline, held at the ends.

    Positions are interpolated with a smootherstep in each leg, so nobody
    accelerates instantaneously and ``max_visible_jump`` stays below one tick of
    ordinary walking — pinned by a test rather than asserted.
    """

    name: str
    waypoints: tuple[Waypoint, ...]

    def _leg(self, t: float):
        points = self.waypoints
        if t <= points[0].t:
            return points[0], points[0], 0.0
        for before, after in zip(points, points[1:]):
            if t <= after.t:
                span = max(after.t - before.t, 1e-9)
                return before, after, (t - before.t) / span
        return points[-1], points[-1], 0.0

    def pos_at(self, t: float) -> np.ndarray:
        before, after, u = self._leg(t)
        a = np.asarray(before.xy, dtype=np.float64)
        b = np.asarray(after.xy, dtype=np.float64)
        return a + (b - a) * _smootherstep(u)

    def velocity_at(self, t: float, dt: float = 0.01) -> np.ndarray:
        return (self.pos_at(t + dt) - self.pos_at(max(t - dt, 0.0))) / (
            dt + min(t, dt))

    def speed_at(self, t: float) -> float:
        return float(np.linalg.norm(self.velocity_at(t)))

    def yaw_at(self, t: float) -> float:
        velocity = self.velocity_at(t)
        if float(np.linalg.norm(velocity)) < 1e-4:
            # Standing still: keep facing the way the last leg was heading, and
            # add a slow look-around so a stationary adult is alive.
            _, after, _ = self._leg(t)
            index = self.waypoints.index(after)
            reference = self.waypoints[max(index - 1, 0)]
            delta = (np.asarray(after.xy, dtype=np.float64)
                     - np.asarray(reference.xy, dtype=np.float64))
            base = (math.atan2(float(delta[1]), float(delta[0]))
                    if float(np.linalg.norm(delta)) > 1e-6 else 0.0)
            return base + math.radians(11.0) * math.sin(
                2.0 * math.pi * 0.16 * t + len(self.name))
        return math.atan2(float(velocity[1]), float(velocity[0]))


def _wp(*pairs) -> tuple[Waypoint, ...]:
    return tuple(Waypoint(t, xy) for t, xy in pairs)


# -- THE GUARDIAN -----------------------------------------------------------
# East leg (the duck follows her along it, in full view), then behind the
# kiosk's north-east corner, west along its north face, out to the far west of
# the hall, then behind column_w, then a stand at the west wall.
GUARDIAN_ROUTE = _wp(
    (0.0,  (2.05, -1.05)),
    (4.0,  (2.05, -0.35)),
    (8.6,  (1.95,  0.45)),
    (12.4, (1.30,  0.92)),      # rounding the kiosk's NE corner
    (17.4, (0.10,  1.02)),      # behind the kiosk, moving west
    # Stay behind the kiosk long enough for the search to evaluate two false
    # candidates, then emerge around its north-west corner and continue slowly
    # in the open.  A route that remains near y=1.0 is hidden forever from the
    # duck's east-side stop pose; racing on to the west wall also makes a
    # policy-driven rejoin impossible before its 30 s ceiling.
    (22.6, (-0.55, 1.08)),      # still geometrically behind the kiosk
    (26.0, (-0.90, 1.75)),      # emerges into the search sightline
    (35.0, (-1.00, 1.80)),
    (46.0, (-1.10, 1.84)),
    (52.0, (-1.143, 1.857)),
    # Hold after the second rejoin so the final measured standoff remains in
    # the 0.45–0.75 m band instead of drifting just beyond it after success.
    (60.0, (-1.143, 1.857)),
)

# -- THE LOOK-ALIKES --------------------------------------------------------
# Timed to walk through the duck's search sweep while it is LOST, not parked
# where they happen to be convenient.
LOOKALIKE_ROUTES: dict[str, tuple[Waypoint, ...]] = {
    # Enters the duck's sweep from the north-east during the first loss,
    # crosses in front of the kiosk, then leaves to the south.
    "mira": _wp(
        (0.0,  (2.70, 1.55)),
        (9.0,  (2.35, 1.30)),
        (15.0, (1.55, 0.72)),
        # Cross the active search cone after Sofia is refused, so the second
        # authored look-alike—not an arbitrary crowd member—is evaluated.
        (21.5, (0.95, 0.95)),
        (25.0, (0.30, -0.85)),
        (32.0, (0.95, -1.75)),
        (42.0, (2.30, -1.85)),
        (60.0, (2.85, -1.30)),
    ),
    # Comes up the west side during the first loss and again during the second.
    "sofia": _wp(
        (0.0,  (-1.10, -1.85)),
        (8.0,  (-0.35, -1.55)),
        (14.5, (0.55, -1.10)),
        (18.5, (1.05, -0.55)),
        (23.0, (0.85, 0.05)),
        (29.0, (-0.20, -0.35)),
        (36.0, (-1.15, -0.95)),
        (44.0, (-2.20, -1.35)),
        (60.0, (-2.80, -1.70)),
    ),
}

# -- THE CROWD --------------------------------------------------------------
# Six independent loops.  arun, dahl and faruq cross the duck's rejoin
# corridor; costa and eze work the west half; bekele patrols the north.
CROWD_ROUTES: dict[str, tuple[Waypoint, ...]] = {
    "arun": _wp((0.0, (-2.80, 1.75)), (11.0, (-0.90, 1.45)),
                (21.0, (0.85, 1.60)), (31.0, (2.45, 1.15)),
                (43.0, (2.60, -0.65)), (55.0, (1.35, -1.70)),
                (60.0, (0.65, -1.80))),
    "bekele": _wp((0.0, (1.25, 1.85)), (10.0, (-0.35, 1.72)),
                  (20.0, (-1.85, 1.55)), (30.0, (-2.85, 0.95)),
                  (40.0, (-1.65, 1.30)), (52.0, (0.35, 1.80)),
                  (60.0, (1.55, 1.72))),
    "costa": _wp((0.0, (-2.55, -1.55)), (12.0, (-2.05, -0.35)),
                 (23.0, (-2.35, 0.85)), (34.0, (-1.05, 1.15)),
                 (45.0, (-0.35, 0.20)), (60.0, (-0.85, -1.05))),
    "dahl": _wp((0.0, (0.95, -1.85)), (9.0, (1.85, -1.15)),
                (18.0, (2.55, -0.15)), (27.0, (2.60, -0.80)),
                (37.0, (0.45, -1.50)), (48.0, (-0.75, -1.65)),
                (60.0, (-1.75, -1.80))),
    "eze": _wp((0.0, (-1.95, 0.95)), (13.0, (-2.75, -0.15)),
               (24.0, (-1.45, -1.15)), (36.0, (-0.15, -1.45)),
               (47.0, (-1.25, -0.55)), (60.0, (-2.35, 0.35))),
    "faruq": _wp((0.0, (2.85, 0.95)), (10.0, (2.15, 1.65)),
                 (19.0, (0.85, 1.15)), (28.0, (1.55, -0.35)),
                 (38.0, (2.35, -1.35)), (50.0, (1.05, -1.55)),
                 (60.0, (-0.15, -1.15))),
}

WALKERS: dict[str, Walker] = {
    "priya": Walker("priya", GUARDIAN_ROUTE),
    **{n: Walker(n, r) for n, r in LOOKALIKE_ROUTES.items()},
    **{n: Walker(n, r) for n, r in CROWD_ROUTES.items()},
}


@dataclass(frozen=True)
class PersonState:
    name: str
    pos: np.ndarray
    yaw: float
    speed: float
    role: str


def people_at(t: float) -> dict[str, PersonState]:
    """Every adult's world state at ``t``.

    The duck's perception layer never receives this dictionary.  It receives
    only what its camera can see, resolved through ``lost_camera``.
    """
    states: dict[str, PersonState] = {}
    for name in ALL_NAMES:
        walker = WALKERS[name]
        states[name] = PersonState(
            name=name, pos=walker.pos_at(t), yaw=walker.yaw_at(t),
            speed=walker.speed_at(t), role=BY_NAME[name].role)
    return states


def max_visible_jump(seconds: float, dt: float = 0.02) -> tuple[float, str, float]:
    """Largest single-tick position jump any adult makes, over the rollout."""
    worst = (0.0, "", 0.0)
    previous = {n: s.pos.copy() for n, s in people_at(0.0).items()}
    for index in range(1, int(seconds / dt) + 1):
        t = index * dt
        for name, state in people_at(t).items():
            jump = float(np.linalg.norm(state.pos - previous[name]))
            if jump > worst[0]:
                worst = (jump, name, t)
            previous[name] = state.pos.copy()
    return worst


def moving_fraction(seconds: float, threshold: float = 0.05,
                    dt: float = 0.10) -> dict[str, float]:
    """Fraction of the rollout each adult spends actually walking."""
    counts = {name: 0 for name in ALL_NAMES}
    steps = int(seconds / dt)
    for index in range(steps):
        for name, state in people_at(index * dt).items():
            if state.speed >= threshold:
                counts[name] += 1
    return {name: counts[name] / max(steps, 1) for name in ALL_NAMES}


def pose_people(model, data, people: dict[str, PersonState], t: float) -> None:
    """Write mocap poses and animate the gait.  Kinematic scenery, no contacts."""
    for order, name in enumerate(ALL_NAMES):
        person = people[name]
        spec = BY_NAME[name]
        body = model.body(f"person_{name}")
        mocap = int(model.body_mocapid[body.id])
        phase = t + 0.41 * order
        data.mocap_pos[mocap, :2] = person.pos
        data.mocap_pos[mocap, 2] = spec.origin_z + 0.008 * abs(
            math.sin(2.0 * math.pi * STRIDE_HZ * phase))
        data.mocap_quat[mocap] = np.array(
            [math.cos(person.yaw / 2.0), 0.0, 0.0, math.sin(person.yaw / 2.0)])
        if person.speed > 1e-3:
            amplitude = math.radians(18.0 + 78.0 * min(person.speed, 0.26))
            rate = STRIDE_HZ
        else:
            amplitude = math.radians(3.6)
            rate = 0.31
        stride = amplitude * math.sin(2.0 * math.pi * rate * phase)
        swing = {
            f"{name}_hip_l": stride, f"{name}_hip_r": -stride,
            f"{name}_shoulder_l": -0.62 * stride,
            f"{name}_shoulder_r": 0.62 * stride,
        }
        for joint_name, value in swing.items():
            joint = model.joint(joint_name).id
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.qvel[int(model.jnt_dofadr[joint])] = 0.0
