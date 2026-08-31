#!/usr/bin/env python3
"""Threat prediction, avoidance state machine and evade controller.

Pure geometry and pure Python/numpy: no MuJoCo, no ONNX, no rendering.  This is
the decision layer and it is fully unit-tested in ``tests/``.

What "threat" means here
------------------------
A threat score is computed from the SIMULATOR'S geometric state — each adult's
world position and velocity, and the duck's position — via a constant-velocity
predicted closest approach.  It is an honest **semantic/geometric proxy for
pedestrian threat assessment, not RGB pedestrian detection**.  There is no
detector, no tracker and no classifier anywhere in this behavior.  The camera
work that follows is real camera geometry (frustum + occlusion ray casts), but
the identity and kinematics of each adult come from the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- measured gait constants -------------------------------------------------
# All of these were MEASURED on scene_move_away_crowd.xml with the stock
# alpha_walking policy at action scale 0.9 and the real imu_ang_vel gyro
# (tools/sweep_commands.py, 4 s and 6 s rollouts).  The forward gait has a HARD
# ONSET: vx=0.20 produces 10 mm in 6 s (no gait), vx=0.24 produces 516 mm.
# Never assume a smaller command gives smaller motion.
VX_WALK: float = 0.28          # 6 s: 0.638 m forward, yaw drift -3.6 deg
VY_EVADE: float = 0.30         # with VX_WALK: 4 s: dx +0.38, dy +0.31, yaw +12 deg
VX_EVADE: float = 0.28
# Turning while walking. wz=+0.60 -> +68 deg/4 s; wz=-0.60 -> -89 deg/4 s.
# MEASURED TRAP: vx=0.24 with wz=+0.35 does NOT cross gait onset (15 mm in 4 s),
# while wz=+0.45 does (255 mm).  Yaw authority is therefore only used at or
# above WZ_MIN_EFFECTIVE, and any smaller correction is rounded to zero.
WZ_SCAN_TURN: float = 0.60
WZ_MIN_EFFECTIVE: float = 0.45
WZ_MAX: float = 0.85
# Backward escape.  MEASURED (tools/sweep_commands.py, 3 s): the backward gait
# has its own hard onset between -0.30 (5 mm, no gait) and -0.34, and above it
# reverses freely: vx=-0.36 gives 0.503 m and vx=-0.40 gives 0.565 m, both
# upright with min trunk z >= 0.114.  Adding lateral works too: (-0.40, +0.34)
# gives 0.599 m and (-0.40, -0.34) gives 0.632 m.
#
# WHY IT EXISTS: without a backward option the only way to escape a threat
# BEHIND the duck was to turn while walking FORWARD, and the measured turn rate
# (~50 deg/s at wz=0.85) needs ~1.5 s to swing 130 deg - all of it spent moving
# toward the threat.  In run 5 that produced the single genuine contact of the
# whole experiment: green locked with a -131.7 deg escape heading at t=10.46 s,
# and the duck walked forward into it, reaching -1.9 mm at t=12.96 s.  Backing
# out along the escape direction removes the turn entirely.
VX_BACK: float = 0.40
VY_BACK: float = 0.34
# Heading error beyond which walking forward is the wrong tool: past this the
# escape lies behind the duck and reversing reaches it without turning through
# the threat.  115 deg leaves the forward+turn primitive in charge of every
# escape it can reach in well under a second.
BACKWARD_ERROR_DEG: float = 115.0
# Command low-pass. 0.25 s is too slow to start the gait; 0.08 s is the value
# carried over from the validated bases and reproduced here.
CMD_TAU: float = 0.08

# --- threat geometry ---------------------------------------------------------
# Horizon over which closest approach is predicted, and the clearance below
# which an approach counts as a genuine stepping/collision threat.  The duck's
# trunk is ~0.09 m across and an adult's torso is 0.078 m, so 0.42 m is a real
# "they are about to walk into me" margin, not a cosmetic one.
PREDICT_HORIZON: float = 5.0
THREAT_CLEARANCE: float = 0.42
# Hysteresis: a threat must be predicted for this long before locking, and the
# lock is held until the encounter is genuinely resolved.
LOCK_CONFIRM_S: float = 0.30
# An encounter is over once the adult is receding AND past this clearance.
RESOLVED_CLEARANCE: float = 0.62
# Timings.
SCAN_MIN_S: float = 1.4
# SCAN_MIN_S makes the duck visibly look around before committing, which is a
# presentation requirement, not a safety one.  MEASURED (run 2): cycle 3 ended
# at t=26.2 s and the scan minimum blocked the next lock until 27.6 s, by which
# time red had closed to 0.16 m with 0.63 s to closest approach - the gait
# needs ~1.0 s just to leave standstill, so the encounter was unavoidable
# before it was ever locked, and the duck was overlapped by 0.073 m.
# An approach this urgent therefore bypasses the scan minimum: looking around
# politely is not worth being walked into.  Confirmation is still required, so
# a single noisy frame cannot trigger the bypass.
URGENT_TTC_S: float = 2.2
LOCK_MIN_S: float = 0.9
# Once an adult's actual rendered geometry is already close, holding still for
# another full second defeats the avoidance.  The corrected RED encounter made
# contact 1.9 s after lock while the duck spent 0.9 s of that interval
# stationary; its carried box was much closer than the centre-point predictor
# represented.  Preserve a visible commitment beat, but shorten it when the
# exact MuJoCo surface clearance says contact is imminent.
URGENT_LOCK_MIN_S: float = 0.30
URGENT_SURFACE_CLEARANCE_M: float = 0.35
# MEASURED (run 2): the forward gait spends its first ~1.0 s crossing onset, so
# a 1.8 s evasion produced only 0.249 m of path - below the 0.25 m the metrics
# gate requires as evidence that a maneuver physically happened.  The measured
# primitive sweep (tools/sweep_commands.py, 2.8 s) gives 0.408-0.493 m of path
# for every escape command used here, and the two run-2 evasions that lasted
# 2.92 s and 3.78 s produced 0.436 m and 0.546 m.  2.6 s therefore clears the
# evidence threshold with margin on the slowest primitive.
EVADE_MIN_S: float = 2.6
EVADE_MAX_S: float = 5.0
SETTLE_S: float = 1.6
CLEAR_S: float = 1.2

STATES = ("SCANNING", "THREAT_LOCK", "EVADING", "SETTLING", "CLEAR")


# Genuine threats always outrank non-threats, whatever their raw urgency.  A
# receding adult standing 0.3 m away is not more important than someone who is
# actually going to walk into the duck in four seconds, and the ranking must say
# so directly instead of relying on the caller to filter.
THREAT_PRIORITY: float = 1000.0


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


@dataclass(frozen=True)
class Approach:
    """Predicted constant-velocity closest approach between an adult and a point."""

    name: str
    time_to_closest: float
    min_clearance: float
    current_range: float
    closing_speed: float
    closest_point: np.ndarray
    bearing: float
    score: float

    @property
    def is_threat(self) -> bool:
        return self.min_clearance < THREAT_CLEARANCE and self.closing_speed > 0.0


def predict_approach(
    adult_pos: np.ndarray,
    adult_vel: np.ndarray,
    duck_pos: np.ndarray,
    *,
    name: str = "",
    horizon: float = PREDICT_HORIZON,
    duck_vel: np.ndarray | None = None,
) -> Approach:
    """Constant-velocity closest approach of one adult to the duck.

    Solves ``min_t |(p + v t) - q|`` for ``t in [0, horizon]`` in closed form.
    ``duck_vel`` lets the caller ask the counterfactual question "what would the
    clearance be if the duck kept doing what it is doing", which is what the
    validation gate compares an evasion against.
    """
    relative_p = np.asarray(adult_pos, dtype=np.float64) - np.asarray(
        duck_pos, dtype=np.float64
    )
    relative_v = np.asarray(adult_vel, dtype=np.float64)
    if duck_vel is not None:
        relative_v = relative_v - np.asarray(duck_vel, dtype=np.float64)
    speed_squared = float(relative_v @ relative_v)
    current_range = float(np.linalg.norm(relative_p))
    if speed_squared < 1e-12:
        time_to_closest = 0.0
    else:
        time_to_closest = clamp(-float(relative_p @ relative_v) / speed_squared,
                                0.0, horizon)
    closest_relative = relative_p + relative_v * time_to_closest
    min_clearance = float(np.linalg.norm(closest_relative))
    # Positive closing speed = the gap is shrinking right now.
    closing_speed = (
        -float(relative_p @ relative_v) / current_range if current_range > 1e-9 else 0.0
    )
    closest_point = np.asarray(duck_pos, dtype=np.float64) + closest_relative
    bearing = math.atan2(float(relative_p[1]), float(relative_p[0]))
    # Urgency: tight predicted clearance and little time to react score highest.
    # Bounded and strictly decreasing in both, so ordering is well defined.
    score = 1.0 / ((min_clearance + 0.08) * (time_to_closest + 0.45))
    is_threat = min_clearance < THREAT_CLEARANCE and closing_speed > 0.0
    if is_threat:
        score += THREAT_PRIORITY
    else:
        score *= 0.15  # ranked, but never able to outrank a real approach
    return Approach(
        name=name,
        time_to_closest=time_to_closest,
        min_clearance=min_clearance,
        current_range=current_range,
        closing_speed=closing_speed,
        closest_point=closest_point,
        bearing=bearing,
        score=score,
    )


def rank_threats(crowd: dict, duck_pos: np.ndarray, **kwargs) -> list[Approach]:
    """Every adult's predicted approach, most urgent first."""
    approaches = [
        predict_approach(
            adult.pos, adult.vel, duck_pos, name=name, **kwargs
        )
        for name, adult in crowd.items()
    ]
    approaches.sort(key=lambda approach: -approach.score)
    return approaches


def most_urgent(crowd: dict, duck_pos: np.ndarray, **kwargs) -> Approach | None:
    """The most urgent genuine threat, or ``None`` when the duck is clear."""
    for approach in rank_threats(crowd, duck_pos, **kwargs):
        if approach.is_threat:
            return approach
    return None


# Below this predicted clearance the escape direction is DEGENERATE: the vector
# from the duck to the predicted impact point has almost no length, so its
# direction is dominated by numerical noise and flips 180 deg from one tick to
# the next.  MEASURED (run 6, green at t=11.50 s): the predicted clearance
# passed through 0.000 m and the escape heading jumped +178.6 -> -1.3 deg in a
# single tick, abandoning a backward escape mid-maneuver and turning the duck
# straight back into the adult.  Inside this band the escape is taken as
# directly away from the adult instead, which is well conditioned.
DEGENERATE_CLEARANCE: float = 0.05


def escape_heading(
    approach: Approach,
    duck_pos: np.ndarray,
    adult_vel: np.ndarray | None = None,
) -> float:
    """World heading that opens the predicted closest approach fastest.

    Away from the point where the threat is predicted to arrive.

    This is ALREADY the normal to the threat's approach line whenever the
    closest approach falls strictly inside the prediction horizon: at an
    interior minimum ``(p + v t*) · v = 0``, so the vector from the duck to the
    closest point is exactly perpendicular to the adult's velocity.  The duck
    therefore sidesteps out of their lane rather than retreating down it, which
    is what a child actually has to do, and no velocity term is needed to get
    that behavior.

    An explicit velocity-normal variant was implemented and MEASURED against
    this one.  It changes nothing for interior minima (they are identical by
    the argument above) and it is strictly WORSE in the only regime where it
    differs — a threat whose closest approach lies beyond the horizon — because
    there the duck (0.28 m/s) is faster than the adults (0.22-0.23 m/s) and
    simply retreating opens the gap faster than stepping aside: predicted
    clearance 3.00 m versus 2.27 m on the same geometry.  The simpler form is
    kept, and ``tests/`` pins that comparison so the idea is not retried.
    """
    to_closest = np.asarray(approach.closest_point, dtype=np.float64) - np.asarray(
        duck_pos, dtype=np.float64
    )
    if float(np.linalg.norm(to_closest)) >= DEGENERATE_CLEARANCE:
        # Away from where they are predicted to arrive.  For an interior
        # minimum this is already normal to their path, so it clears the lane.
        return math.atan2(-float(to_closest[1]), -float(to_closest[0]))

    # DEGENERATE: the duck is on the predicted impact point, so the away-vector
    # has no reliable direction and flips 180 deg between ticks.
    #
    # MEASURED (run 7, green): "flee directly away from the adult" is the wrong
    # answer here.  A head-on approach makes that heading nearly ANTI-PARALLEL
    # to the adult's travel, so the duck retreats DOWN their lane and is simply
    # followed: green closed from 0.30 m to 0.11 m while the duck reversed
    # 0.89 m, and the predicted clearance improved by only 0.024 m.
    #
    # Stepping normal to their path leaves the lane instead.  Which side is
    # chosen by the duck's current offset from that path, so the duck commits
    # to the side it is already on rather than crossing in front of them.
    if adult_vel is not None:
        velocity = np.asarray(adult_vel, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-6:
            heading = velocity / speed
            normal = np.array([-heading[1], heading[0]])
            # Which side of their line of travel the duck is already on.
            offset = np.asarray(duck_pos, dtype=np.float64) - np.asarray(
                approach.closest_point, dtype=np.float64
            )
            side = float(offset @ normal)
            if abs(side) < 1e-9:
                # Exactly on their line: both sides are equivalent, so take the
                # one nearer to fleeing away and stay deterministic.
                away = np.array([
                    math.cos(approach.bearing + math.pi),
                    math.sin(approach.bearing + math.pi),
                ])
                side = float(away @ normal) or 1.0
            direction = math.copysign(1.0, side) * normal
            return math.atan2(float(direction[1]), float(direction[0]))
    # No velocity available: fall back to fleeing directly away.
    return wrap_angle(approach.bearing + math.pi)


@dataclass
class AvoidanceMachine:
    """SCANNING -> THREAT_LOCK -> EVADING -> SETTLING -> CLEAR, repeated.

    ``SCANNING``, ``SETTLING`` and ``CLEAR`` are stationary states: the
    locomotion command is exactly zero in all three.  ``THREAT_LOCK`` is also
    stationary — the duck stops, looks and commits before it moves.
    """

    ctrl_hz: float = 50.0
    state: str = "SCANNING"
    state_since: float = 0.0
    locked: str | None = None
    cycles: list[dict] = field(default_factory=list)
    current: dict = field(default_factory=dict)
    _confirming: str | None = None
    _confirmed_for: float = 0.0

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_hz

    @property
    def moving(self) -> bool:
        return self.state == "EVADING"

    @property
    def confirming(self) -> str | None:
        """The candidate currently accumulating confirmation time, if any.

        Exposed because a lock is a decision taken over ``LOCK_CONFIRM_S``, not
        at the instant of the transition: a caller grading "was this the right
        adult to lock" must evaluate the world as it was when confirmation of
        this candidate began.
        """
        return self._confirming

    def update(
        self, t: float, threat: Approach | None, locked_view: dict | None = None
    ) -> tuple[str, bool]:
        """Advance one tick. Returns ``(state, changed)``.

        An ``Approach`` that is not a genuine threat is treated exactly like
        ``None``: the machine never locks onto someone who is going to miss.
        """
        if threat is not None and not threat.is_threat:
            threat = None
        elapsed = t - self.state_since
        previous = self.state

        if self.state == "SCANNING":
            # Confirm a candidate for LOCK_CONFIRM_S before committing, so a
            # single noisy frame cannot trigger a lock.
            if threat is not None:
                if threat.name == self._confirming:
                    self._confirmed_for += self.dt
                else:
                    self._confirming = threat.name
                    self._confirmed_for = 0.0
            else:
                self._confirming = None
                self._confirmed_for = 0.0
            if (
                threat is not None
                and self._confirmed_for >= LOCK_CONFIRM_S
                and (elapsed >= SCAN_MIN_S
                     or threat.time_to_closest <= URGENT_TTC_S)
            ):
                self.locked = threat.name
                self.state = "THREAT_LOCK"
                self.current = {
                    "cycle": len(self.cycles) + 1,
                    "threat": threat.name,
                    "scan_start_s": self.state_since,
                    "lock_s": t,
                    "lock_clearance_m": threat.min_clearance,
                    "lock_ttc_s": threat.time_to_closest,
                    "lock_range_m": threat.current_range,
                    "lock_bearing_deg": math.degrees(threat.bearing),
                }
        elif self.state == "THREAT_LOCK":
            surface_clearance = (locked_view or {}).get(
                "surface_clearance_m", float("inf")
            )
            lock_hold = (
                URGENT_LOCK_MIN_S
                if surface_clearance <= URGENT_SURFACE_CLEARANCE_M
                else LOCK_MIN_S
            )
            if elapsed >= lock_hold:
                self.state = "EVADING"
                self.current["evade_start_s"] = t
        elif self.state == "EVADING":
            resolved = threat is None or threat.name != self.locked
            if not resolved and threat is not None:
                resolved = (
                    threat.closing_speed <= 0.0
                    and threat.min_clearance >= RESOLVED_CLEARANCE
                )
            # A maneuver must actually be performed: resolving instantly would
            # leave no physical displacement for the validation gate to measure.
            if elapsed < EVADE_MIN_S:
                resolved = False
            if resolved or elapsed >= EVADE_MAX_S:
                self.state = "SETTLING"
                self.current["evade_end_s"] = t
                self.current["evade_duration_s"] = t - self.current["evade_start_s"]
                self.current["evade_timeout"] = bool(elapsed >= EVADE_MAX_S)
        elif self.state == "SETTLING":
            if elapsed >= SETTLE_S:
                self.state = "CLEAR"
                self.current["settle_end_s"] = t
        elif self.state == "CLEAR":
            if elapsed >= CLEAR_S:
                self.current["cycle_end_s"] = t
                self.cycles.append(dict(self.current))
                self.current = {}
                self.locked = None
                self._confirming = None
                self._confirmed_for = 0.0
                self.state = "SCANNING"

        changed = self.state != previous
        if changed:
            self.state_since = t
        return self.state, changed


@dataclass
class EvadeController:
    """Produce a filtered ``(vx, vy, wz)`` from the machine state.

    Only ``EVADING`` ever produces a nonzero command.  The command itself is
    assembled exclusively from MEASURED constants, and any yaw correction below
    ``WZ_MIN_EFFECTIVE`` is snapped to zero because the measured sweep shows an
    intermediate wz can suppress the gait entirely.

    The controller also COMMITS to a maneuver once it has started one.  The
    escape heading is a function of the predicted impact point, which is
    genuinely unstable while the predicted clearance is small: it can swing far
    within a fraction of a second, and re-deciding forward-versus-backward every
    tick turns a committed evasion into indecision.  Run 6 measured exactly
    that: a backward escape was abandoned one tick after it began and the duck
    walked forward into the adult it was avoiding.  ``reset`` clears the
    commitment at the start of each evasion.
    """

    ctrl_hz: float = 50.0
    command: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    _reversing: bool | None = None

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_hz

    def reset(self) -> None:
        """Forget the committed maneuver; called when an evasion begins."""
        self._reversing = None

    def raw_command(self, state: str, heading_error: float) -> tuple[float, float, float]:
        """The unfiltered target command for this state and heading error."""
        if state != "EVADING":
            return (0.0, 0.0, 0.0)
        # Decide forward-versus-backward ONCE per evasion, then keep steering
        # within that choice.  Hysteresis is not enough here: the flip is a
        # genuine 180 deg reversal of the underlying geometry, not chatter
        # around a threshold.
        reversing = self._reversing
        if reversing is None:
            reversing = abs(heading_error) > math.radians(BACKWARD_ERROR_DEG)
            self._reversing = reversing
        if reversing:
            # Escape lies behind: back out along it rather than turning through
            # the threat.  Project the desired local heading onto the lateral
            # axis: sin(error) has the correct sign in the robot frame even
            # while vx is negative.  The old ``error +/- pi`` formulation
            # inverted that sign and made the green encounter reverse mostly
            # *along* the adult's lane, so the carried box caught the duck
            # after it stopped.
            lateral = clamp(
                VY_BACK * math.sin(heading_error),
                -VY_BACK, VY_BACK,
            )
            return (-VX_BACK, lateral, 0.0)
        # Large heading error: turn while walking, at a wz that provably crosses
        # gait onset.  Small heading error: walk forward with a lateral
        # component, which the sweep shows is the most stable escape.
        if abs(heading_error) > math.radians(35.0):
            wz = math.copysign(
                clamp(abs(1.6 * heading_error), WZ_SCAN_TURN, WZ_MAX), heading_error
            )
            return (VX_WALK, 0.0, wz)
        lateral = clamp(
            VY_EVADE * (heading_error / math.radians(35.0)), -VY_EVADE, VY_EVADE
        )
        if abs(lateral) < 0.5 * VY_EVADE:
            lateral = math.copysign(0.5 * VY_EVADE, lateral) if lateral != 0.0 else 0.0
        return (VX_EVADE, lateral, 0.0)

    def update(self, state: str, heading_error: float) -> np.ndarray:
        target = np.asarray(self.raw_command(state, heading_error), dtype=np.float32)
        if state != "EVADING":
            # Stationary states are EXACTLY zero, not a decaying tail: the gate
            # requires zero locomotion command while scanning or settled.
            self.command[:] = 0.0
            return self.command.copy()
        alpha = min(1.0, self.dt / CMD_TAU)
        self.command += alpha * (target - self.command)
        return self.command.copy()
