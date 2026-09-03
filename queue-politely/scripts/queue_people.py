#!/usr/bin/env python3
"""The queue's inhabitants: five queueing adults, two bystanders, one clerk.

Every body here is MOCAP and ``contype="0" conaffinity="0"``: kinematic
scenery, posed analytically each tick, adding no degree of freedom to the
robot's floating base.  The walking policy therefore sees exactly the robot it
was trained on, and no advance the duck makes can be the result of somebody
nudging it forward.

Design constraints this file exists to satisfy
----------------------------------------------
* **Nobody reacts to the duck, ever.**  Each adult advances when the person in
  front of them is served, on a schedule fixed before the rollout starts.  There
  is no branch anywhere that holds the queue for the duck, and
  ``test_the_queue_never_waits_for_the_duck`` pins that by replaying the whole
  schedule with the duck absent and requiring identical positions.
* **The queue is a PATH occupancy, not a formation.**  Positions are stored as
  ARC LENGTH along :data:`queue_path.PATH` and converted to world coordinates
  only for posing.  That is what makes an adult standing on the bend an
  ordinary queue member rather than a special case.
* **One adult is a straggler, and that is the whole point.**  ``eriksson``
  stands 0.90 m behind ``dubois`` instead of the nominal 0.55 m.  The resulting
  gap is comfortably wide enough for the duck to stand in - measured, not
  asserted, by ``queue_model.gap_fits_duck`` - so the duck's refusal to take it
  is a judgement about ORDER rather than an observation about width.  The gap
  closes on the first advance, exactly as a real queue closes up.
* **Two bystanders are near the queue but not in it.**  They are placed where a
  careless reading would interleave them into the order: ``nakamura`` beside the
  bend would rank between the 4th and 5th places, and ``okafor`` near the
  counter would rank 2nd.  Both are excluded by cross-track distance from the
  path, which is a property a max-coordinate ordering cannot express at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from queue_path import PATH, SERVICE_XY, SLOT_SPACING_M

# Stride/idle animation frequency.
STRIDE_HZ: float = 1.05
# How fast an adult closes up when the queue advances, and the per-place delay
# that makes the advance ripple backward instead of teleporting the whole line.
ADULT_ADVANCE_MPS: float = 0.62
ADVANCE_STAGGER_S: float = 0.42
# Planar half-extent of an adult: the LARGEST value over a full gait cycle,
# because the arms and legs swing and a person is widest mid-stride.  MEASURED
# over 200 poses of the real scene: min 0.1195, mean 0.1394, max 0.1647.
#
# The maximum is the honest number here, and it is the conservative one for the
# claim this behavior turns on.  The gaps the duck refuses have to be gaps it
# could genuinely have stood in, so the people bounding those gaps must be
# taken at their WIDEST; using the mean would make a refused gap look roomier
# than it ever really is.  ``test_the_adult_half_extent_constant_matches_the_model``
# pins this against the built scene over the same gait cycle.
ADULT_HALF_EXTENT_M: float = 0.1647

# WHEN EACH HEAD-OF-QUEUE PERSON FINISHES AND LEAVES.
#
# Fixed before the rollout and never consulted by the duck's controller.  The
# first service completes AFTER the duck has joined, so the order it infers is
# read against a queue that is standing still - the honest way to grade "did it
# get the order right" - and every later advance is then a genuine reaction to
# somebody else leaving.
DEPARTURE_TIMES: tuple[float, ...] = (22.0, 28.0, 34.0, 40.0, 46.0)
# Where a served person goes: away from the counter and away from the queue,
# so nobody ever walks back through the line.
DEPART_DIR = np.array([0.34, 0.94], dtype=np.float64) / np.linalg.norm(
    np.array([0.34, 0.94]))
DEPART_MPS: float = 0.55


@dataclass(frozen=True)
class Adult:
    """One queueing adult, positioned by ARC LENGTH along the queue path."""

    name: str
    rgba: str
    index: int                 # initial place in line, 0 = being served
    initial_arc: float
    label: str = ""

    def rank_at(self, t: float) -> int:
        """Places from the head at time ``t``; negative once departed."""
        served = sum(1 for d in DEPARTURE_TIMES[: self.index + 1] if t >= d)
        return self.index - served

    def departed_at(self) -> float:
        return DEPARTURE_TIMES[self.index]

    def arc_at(self, t: float) -> float:
        """Arc length along the queue path, closing up one place per departure.

        Applied as a sequence of completed moves rather than a single formula:
        the m-th departure ahead of this person sends them to station
        ``(index - m) * SLOT_SPACING`` starting ``ADVANCE_STAGGER_S`` per
        remaining place after that departure.  Each move finishes long before
        the next departure, so the sequence is exact rather than approximate.
        """
        arc = self.initial_arc
        for m, departure in enumerate(DEPARTURE_TIMES[: self.index], start=1):
            target = (self.index - m) * SLOT_SPACING_M
            start = departure + ADVANCE_STAGGER_S * (self.index - m)
            arc = _ease(arc, target, t, start, ADULT_ADVANCE_MPS)
        return arc

    def advancing_at(self, t: float) -> bool:
        for m, departure in enumerate(DEPARTURE_TIMES[: self.index], start=1):
            target = (self.index - m) * SLOT_SPACING_M
            start = departure + ADVANCE_STAGGER_S * (self.index - m)
            previous = self.initial_arc if m == 1 else (
                self.index - m + 1) * SLOT_SPACING_M
            duration = abs(previous - target) / ADULT_ADVANCE_MPS
            if start <= t < start + duration:
                return True
        return False

    def pos_at(self, t: float) -> np.ndarray:
        if t >= self.departed_at():
            return (np.asarray(SERVICE_XY, dtype=np.float64)
                    + DEPART_DIR * DEPART_MPS * (t - self.departed_at()))
        return PATH.point_at(self.arc_at(t))

    def yaw_at(self, t: float) -> float:
        if t >= self.departed_at():
            return math.atan2(float(DEPART_DIR[1]), float(DEPART_DIR[0]))
        return PATH.travel_heading_at(self.arc_at(t))

    def in_queue_at(self, t: float) -> bool:
        return t < self.departed_at()

    def speed_at(self, t: float) -> float:
        if t >= self.departed_at():
            return DEPART_MPS
        return ADULT_ADVANCE_MPS if self.advancing_at(t) else 0.0


def _ease(current: float, target: float, t: float, start: float,
          speed: float) -> float:
    """Move ``current`` toward ``target`` on a smooth ramp beginning at ``start``.

    Smootherstep rather than a linear ramp so an adult accelerates out of stand
    still and settles, which is what a queue closing up actually looks like and
    also keeps ``max_visible_jump`` small.
    """
    distance = abs(target - current)
    if distance < 1e-9 or t <= start:
        return current
    duration = distance / speed
    if t >= start + duration:
        return target
    u = (t - start) / duration
    blend = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
    return current + (target - current) * blend


@dataclass(frozen=True)
class Bystander:
    """Somebody near the queue who is not in it, and never joins."""

    name: str
    rgba: str
    home: tuple[float, float]
    sway_deg: float
    label: str = ""

    def pos_at(self, t: float) -> np.ndarray:
        # A slow shuffle in place: alive, but never advancing along the path.
        wobble = 0.035 * math.sin(2.0 * math.pi * 0.21 * t + hash(self.name) % 7)
        return np.array([self.home[0], self.home[1] + wobble],
                        dtype=np.float64)

    def yaw_at(self, t: float) -> float:
        return math.radians(
            self.sway_deg + 9.0 * math.sin(2.0 * math.pi * 0.17 * t))


# THE QUEUE.  Five adults; ``eriksson`` is the straggler.
QUEUE: tuple[Adult, ...] = (
    Adult("alvarez", "0.88 0.30 0.24 1", 0, 0.00, "being served"),
    Adult("bianchi", "0.24 0.46 0.86 1", 1, 0.55, "2nd"),
    Adult("chandra", "0.94 0.72 0.20 1", 2, 1.10, "3rd"),
    Adult("dubois", "0.32 0.70 0.38 1", 3, 1.65, "4th, on the bend"),
    Adult("eriksson", "0.72 0.38 0.82 1", 4, 2.55, "5th, straggler"),
)
# Two adults who are NOT in the queue, placed where a careless reading would
# interleave them: beside the bend, and beside the counter.
BYSTANDERS: tuple[Bystander, ...] = (
    # MEASURED PLACEMENT, not decoration.  The first draft stood this bystander
    # at (-1.72, -0.62), which ``tools/probe_camera.py`` showed occupying the
    # lower-left foreground of the wide shot in 80 % of sampled frames - a body
    # sweeping across the camera for most of the rollout.  Moved to (-1.20,
    # 0.55): still beside the bend, still 0.80 m off the path (well outside the
    # 0.30 m membership band), still where a range-sorted reading would
    # interleave it into the queue, and now in the lower-left corner in 1 % of
    # frames.
    Bystander("nakamura", "0.55 0.55 0.58 1", (-1.20, 0.55), -60.0,
              "bystander, beside the bend"),
    Bystander("okafor", "0.40 0.62 0.66 1", (0.16, -0.86), 150.0,
              "bystander, beside the counter"),
)
CLERK = Bystander("mensah", "0.20 0.24 0.34 1", (0.78, 0.00), 180.0, "clerk")

QUEUE_NAMES: tuple[str, ...] = tuple(a.name for a in QUEUE)
BYSTANDER_NAMES: tuple[str, ...] = tuple(b.name for b in BYSTANDERS)
ADULT_NAMES: tuple[str, ...] = QUEUE_NAMES + BYSTANDER_NAMES
ALL_NAMES: tuple[str, ...] = ADULT_NAMES + (CLERK.name,)
QUEUE_BY_NAME: dict[str, Adult] = {a.name: a for a in QUEUE}
PERSON_RGBA: dict[str, str] = {
    **{a.name: a.rgba for a in QUEUE},
    **{b.name: b.rgba for b in BYSTANDERS},
    CLERK.name: CLERK.rgba,
}


@dataclass(frozen=True)
class PersonState:
    name: str
    pos: np.ndarray
    yaw: float
    speed: float
    in_queue: bool
    label: str


def people_at(t: float) -> dict[str, PersonState]:
    """Every person's world state at ``t``, queue members and bystanders alike.

    The duck's perception layer receives THIS and nothing else - in particular
    it does not receive anybody's place in line.  Working the order out from
    these positions is the behavior.
    """
    states: dict[str, PersonState] = {}
    for adult in QUEUE:
        states[adult.name] = PersonState(
            name=adult.name, pos=adult.pos_at(t), yaw=adult.yaw_at(t),
            speed=adult.speed_at(t), in_queue=adult.in_queue_at(t),
            label=adult.label)
    for person in (*BYSTANDERS, CLERK):
        states[person.name] = PersonState(
            name=person.name, pos=person.pos_at(t), yaw=person.yaw_at(t),
            speed=0.0, in_queue=False, label=person.label)
    return states


def departures(seconds: float) -> list[dict]:
    """Every service completion inside the rollout, reported rather than assumed."""
    events = []
    for adult in QUEUE:
        when = adult.departed_at()
        if when <= seconds:
            events.append({"person": adult.name, "served_at_s": when,
                           "initial_place": adult.index + 1})
    return sorted(events, key=lambda e: e["served_at_s"])


def max_visible_jump(seconds: float, dt: float = 0.02) -> tuple[float, str, float]:
    """Largest single-tick position jump any person makes.

    The honest claim is not "the schedule is continuous by inspection" but "no
    adult ever moves discontinuously", and a test pins this below one tick of
    ordinary walking.
    """
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


def pose_people(model, data, people: dict[str, PersonState], t: float) -> None:
    """Write every mocap pose and animate gait, idle sway and the service mime.

    Kinematic scenery written straight into mocap slots: none of it is simulated
    and none of it can touch the robot.
    """
    for order, name in enumerate(ALL_NAMES):
        person = people[name]
        body = model.body(f"person_{name}")
        mocap = int(model.body_mocapid[body.id])
        phase = t + 0.47 * order
        data.mocap_pos[mocap, :2] = person.pos
        data.mocap_pos[mocap, 2] = 0.36 + 0.008 * abs(
            math.sin(2.0 * math.pi * STRIDE_HZ * phase))
        data.mocap_quat[mocap] = np.array(
            [math.cos(person.yaw / 2.0), 0.0, 0.0, math.sin(person.yaw / 2.0)])
        # Walking swings the legs; standing still leaves a small weight shift,
        # so a stationary queue is visibly alive rather than a row of statues.
        if person.speed > 1e-6:
            amplitude = math.radians(16.0 + 90.0 * min(person.speed, 0.24))
            rate = STRIDE_HZ
        else:
            amplitude = math.radians(3.4)
            rate = 0.33
        stride = amplitude * math.sin(2.0 * math.pi * rate * phase)
        # The person at the counter mimes a transaction with one arm.
        served = person.in_queue and float(
            np.linalg.norm(person.pos - np.asarray(SERVICE_XY))) < 0.18
        arm_l = -0.6 * stride
        if served or name == CLERK.name:
            arm_l = math.radians(34.0) * (
                0.5 + 0.5 * math.sin(2.0 * math.pi * 0.7 * phase))
        swing = {
            f"{name}_hip_l": stride,
            f"{name}_hip_r": -stride,
            f"{name}_shoulder_l": arm_l,
            f"{name}_shoulder_r": 0.6 * stride,
        }
        for joint_name, value in swing.items():
            joint = model.joint(joint_name).id
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.qvel[int(model.jnt_dofadr[joint])] = 0.0
