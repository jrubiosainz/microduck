#!/usr/bin/env python3
"""Recall state machine, caller selection and the approach controller.

Pure geometry and pure Python/numpy: no MuJoCo, no ONNX, no rendering.  This is
the decision layer and it is fully unit-tested in ``tests/``.

What "a call" means here
------------------------
A call is a scripted EVENT carrying a caller identity, an onset time and a
duration.  It is an honest **simulator semantic proxy** for "an adult calls the
robot": there is no audio, no keyword spotter and no sound propagation
anywhere in this behavior.  What IS real is the camera geometry that follows —
the duck may only lock a caller it can actually see through the exact frustum
the PiP renders from, so a call from behind forces a genuine visual search.

State machine
-------------
``LISTEN -> SEARCH -> CALLER_LOCK -> APPROACH -> ARRIVED -> LISTEN``

Only ``APPROACH`` ever emits a nonzero locomotion command.  ``LISTEN``,
``SEARCH``, ``CALLER_LOCK`` and ``ARRIVED`` are exactly zero — not a decaying
tail — because the acceptance gate requires it.

The no-steal rule
-----------------
A call that arrives while the duck is already committed to someone does NOT
take over.  Once ``CALLER_LOCK`` is entered, ``active_call`` is pinned until
``ARRIVED`` completes.  This is deliberate for v1: an interruption rule that
can retarget mid-approach needs its own acceptance evidence (which cycle was
abandoned, whether the abandoned caller was ever reached), and shipping it
untested would weaken exactly the gate this behavior exists to prove.  The
scenario issues one such interrupting call so the refusal is measured rather
than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- measured gait constants -------------------------------------------------
# MEASURED on scene_come_here_recall.xml with the stock alpha_walking policy at
# action scale 0.9 and the real imu_ang_vel sensor (tools/sweep_commands.py and
# tools/measure_approach.py; 3-6 s rollouts).  Nothing here is inherited from
# the pre-PR-#22 corrupted-observation constants.
#
# FORWARD GAIT ONSET IS A CLIFF, NOT A RAMP:
#   vx=0.16 -> 0.008 m in 6 s      (no gait at all)
#   vx=0.20 -> 0.010 m in 6 s      (no gait at all)
#   vx=0.24 -> 0.515 m in 6 s      (walking)
# A command below onset produces NO motion, so "slow down as you arrive" cannot
# be done by shrinking vx toward zero - it is done by stopping.
VX_MIN_EFFECTIVE: float = 0.24
VX_CRUISE: float = 0.46        # 6 s: 1.254 m, path 1.396 m, min z 0.112
VX_APPROACH: float = 0.32      # 6 s: 0.742 m, gentler for the final metres
VX_TURN: float = 0.36          # 4 s: turning arcs, min z 0.112

# TURN IN PLACE IS IMPOSSIBLE with this policy.  MEASURED at vx=0.0 over SIX
# seconds: wz=+0.85 -> +7.8 deg, wz=-0.85 -> -9.5 deg, wz=+/-0.45 -> +4.1/-5.1
# deg.  Yaw authority exists ONLY while walking, so every heading change is
# flown as an arc and the duck necessarily covers ground while turning.
#
# THE POLICY IS STRONGLY ASYMMETRIC.  MEASURED yaw rate over 3 s:
#   vx=0.28 wz=-0.85 -> -31.0 deg/s     vx=0.28 wz=+0.85 -> +26.8 deg/s
#   vx=0.28 wz=-0.60 -> -23.7 deg/s     vx=0.28 wz=+0.60 -> +16.4 deg/s
#   vx=0.28 wz=-0.45 -> -18.6 deg/s     vx=0.28 wz=+0.45 -> +11.1 deg/s
#   vx=0.24 wz=-0.25 ->  -8.0 deg/s     vx=0.24 wz=+0.25 ->  +0.7 deg/s
# The right (negative) side turns 1.2-11x faster for the same |wz|, so the two
# signs are tuned INDEPENDENTLY rather than mirrored.
WZ_MAX_RIGHT: float = 0.85
WZ_MAX_LEFT: float = 0.85
# Below these the measured yaw rate collapses toward zero on the LEFT side
# (+0.7 deg/s at wz=+0.25), so a small left correction is worthless: it burns a
# control tick without turning.  Snap anything smaller to straight-ahead.
WZ_MIN_RIGHT: float = 0.25
WZ_MIN_LEFT: float = 0.45
# Proportional gains, per side, chosen so a 120 deg turn is flown in about 4 s
# without overshooting past the caller.
KP_RIGHT: float = 1.05
KP_LEFT: float = 1.45

# COAST AFTER STOP IS NEGLIGIBLE.  MEASURED: cruising 4 s then commanding
# exactly zero, the trunk drifts 4.5 mm (vx=0.24), 6.7 mm (vx=0.28) and 8.9 mm
# (vx=0.28, wz=-0.45), and the drift is FLAT from +0.5 s to +2.5 s.  The duck
# therefore stops essentially where it is told to, and the standoff band does
# not need a braking-distance term - it needs the command released at the band.
COAST_M: float = 0.010

# --- standoff geometry -------------------------------------------------------
# Distance from the duck's trunk centre to the adult's mocap origin at which
# the approach is complete.
#
# The band is justified from the SCENE'S OWN geometry, not picked for looks:
#   adult torso capsule radius              0.078 m
#   duck planar half-extent (measured)      ~0.09 m
#   => bodies touch at about                0.17 m
# A standoff of 0.45-0.75 m therefore leaves 0.28-0.58 m of real clearance:
# close enough to read as "the duck came when called" and to be inside the
# adult's reach, far enough that neither the pacing loop (+/-0.32 m amplitude,
# ~0.06 m/s) nor a stopping error can close the gap to contact.
STANDOFF_TARGET: float = 0.60
STANDOFF_MIN: float = 0.45
STANDOFF_MAX: float = 0.75
# Release the walking command this far out, so the residual coast lands the
# duck near the band centre instead of at its inner edge.
STOP_RANGE: float = STANDOFF_TARGET + COAST_M

# --- acquisition gate --------------------------------------------------------
# A caller may be locked only while inside this cone of the attention camera's
# optical axis AND geometrically visible through that same camera.  The camera
# is 58 deg vertical FOV on a 300x220 PiP, i.e. +/-29.0 deg vertical and
# +/-37.1 deg horizontal, so a 12 deg acquisition cone sits comfortably inside
# the frame rather than accepting a target clinging to the frame edge.
ACQUIRE_CONE_DEG: float = 12.0
# The caller must stay inside the gate this long before the lock commits, so a
# single frame of the search sweep clipping past them cannot trigger a lock.
ACQUIRE_CONFIRM_S: float = 0.24

# --- timings -----------------------------------------------------------------
LISTEN_MIN_S: float = 0.9
LOCK_HOLD_S: float = 0.9
ARRIVED_HOLD_S: float = 2.0
# An approach that cannot finish is abandoned rather than running forever.
APPROACH_MAX_S: float = 22.0

STATES = ("LISTEN", "SEARCH", "CALLER_LOCK", "APPROACH", "ARRIVED")
STATIONARY_STATES = ("LISTEN", "SEARCH", "CALLER_LOCK", "ARRIVED")


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


@dataclass(frozen=True)
class Call:
    """One scripted call event.

    ``expected`` marks the calls the scenario intends the duck to serve.  A
    call with ``expected=False`` is an interruption issued while the duck is
    already busy; the no-steal rule must refuse it, and the metrics gate checks
    that refusal explicitly.
    """

    caller: str
    start_s: float
    duration_s: float
    expected: bool = True

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def active_at(self, t: float) -> bool:
        return self.start_s <= t < self.end_s


def calls_active_at(calls: tuple[Call, ...], t: float) -> list[Call]:
    """Every call sounding at ``t``, earliest first."""
    return [call for call in calls if call.active_at(t)]


@dataclass
class RecallMachine:
    """LISTEN -> SEARCH -> CALLER_LOCK -> APPROACH -> ARRIVED, repeated.

    The machine owns WHICH caller is being served and WHEN the phase changes.
    It does not own geometry: the caller's range and bearing are supplied by
    the caller, and whether the caller is inside the acquisition gate is
    supplied by the camera.  That split is what lets the whole machine be
    tested without MuJoCo.
    """

    ctrl_hz: float = 50.0
    state: str = "LISTEN"
    state_since: float = 0.0
    active_call: Call | None = None
    locked: str | None = None
    cycles: list[dict] = field(default_factory=list)
    current: dict = field(default_factory=dict)
    refused_calls: list[dict] = field(default_factory=list)
    served_calls: set[tuple[str, float]] = field(default_factory=set)
    _gated_for: float = 0.0
    _gated_on: str | None = None

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_hz

    @property
    def busy(self) -> bool:
        """True once the duck has committed to a caller."""
        return self.state in ("CALLER_LOCK", "APPROACH", "ARRIVED")

    @property
    def moving(self) -> bool:
        return self.state == "APPROACH"

    def update(
        self,
        t: float,
        *,
        calls: tuple[Call, ...],
        caller_range: float | None,
        gate_open: bool,
        caller_visible: bool,
    ) -> tuple[str, bool]:
        """Advance one tick. Returns ``(state, changed)``.

        ``gate_open`` is the camera's verdict: the ACTIVE caller is inside the
        acquisition cone and geometrically visible through the attention
        camera.  The machine never infers visibility from geometry it computed
        itself - if the camera cannot see the caller, no lock happens, however
        obvious the caller's position is in world coordinates.
        """
        elapsed = t - self.state_since
        previous = self.state
        # A call that has ALREADY been served is not a new call.  Without this,
        # a generous call duration keeps sounding after the duck arrives and
        # the machine immediately re-serves the same person: run 2 completed
        # red -> yellow -> yellow instead of red -> yellow -> green, and the
        # third "recall" was a 0.02 s cycle that never moved because the duck
        # was already standing at the standoff distance.
        #
        # A call is identified by (caller, start_s), so the SAME adult calling
        # again later is a genuinely new call and is served normally.
        sounding = [
            call for call in calls_active_at(calls, t)
            if (call.caller, call.start_s) not in self.served_calls
        ]

        # A call arriving while committed is REFUSED, not queued and not
        # obeyed.  Record it so the gate can prove the refusal happened.
        if self.busy:
            for call in sounding:
                if self.active_call is not None and call is self.active_call:
                    continue
                already = any(
                    entry["caller"] == call.caller
                    and abs(entry["call_start_s"] - call.start_s) < 1e-9
                    for entry in self.refused_calls
                )
                if not already:
                    self.refused_calls.append({
                        "caller": call.caller,
                        "call_start_s": call.start_s,
                        "refused_at_s": t,
                        "busy_with": self.locked,
                        "state": self.state,
                    })

        if self.state == "LISTEN":
            self.locked = None
            self.active_call = None
            self._gated_for = 0.0
            self._gated_on = None
            if sounding and elapsed >= LISTEN_MIN_S:
                # Serve the earliest call that is sounding.
                self.active_call = min(sounding, key=lambda call: call.start_s)
                self.state = "SEARCH"
                self.current = {
                    "cycle": len(self.cycles) + 1,
                    "caller": self.active_call.caller,
                    "call_start_s": self.active_call.start_s,
                    "listen_start_s": self.state_since,
                    "search_start_s": t,
                }
        elif self.state == "SEARCH":
            call = self.active_call
            if call is None or not call.active_at(t):
                # The caller gave up before being found; go back to listening.
                self.state = "LISTEN"
                self.current = {}
            else:
                if gate_open and caller_visible:
                    if self._gated_on == call.caller:
                        self._gated_for += self.dt
                    else:
                        self._gated_on = call.caller
                        self._gated_for = 0.0
                else:
                    self._gated_on = None
                    self._gated_for = 0.0
                if self._gated_for >= ACQUIRE_CONFIRM_S:
                    self.locked = call.caller
                    self.state = "CALLER_LOCK"
                    self.current["lock_s"] = t
                    self.current["search_duration_s"] = t - self.current[
                        "search_start_s"]
                    self.current["lock_range_m"] = caller_range
        elif self.state == "CALLER_LOCK":
            if elapsed >= LOCK_HOLD_S:
                self.state = "APPROACH"
                self.current["approach_start_s"] = t
                self.current["approach_start_range_m"] = caller_range
        elif self.state == "APPROACH":
            arrived = caller_range is not None and caller_range <= STOP_RANGE
            if arrived or elapsed >= APPROACH_MAX_S:
                self.state = "ARRIVED"
                self.current["approach_end_s"] = t
                self.current["approach_duration_s"] = t - self.current[
                    "approach_start_s"]
                self.current["arrival_range_m"] = caller_range
                self.current["approach_timeout"] = bool(not arrived)
        elif self.state == "ARRIVED":
            if elapsed >= ARRIVED_HOLD_S:
                self.current["arrived_end_s"] = t
                self.current["final_range_m"] = caller_range
                self.cycles.append(dict(self.current))
                if self.active_call is not None:
                    self.served_calls.add(
                        (self.active_call.caller, self.active_call.start_s))
                self.current = {}
                self.locked = None
                self.active_call = None
                self.state = "LISTEN"

        changed = self.state != previous
        if changed:
            self.state_since = t
        return self.state, changed


@dataclass
class ApproachController:
    """Turn a range and bearing error into a measured ``(vx, vy, wz)``.

    Three properties are enforced here rather than hoped for:

    * **Zero means zero.**  Every state except ``APPROACH`` returns exactly
      ``(0, 0, 0)`` with no filter tail, because the gate tests for exact zero.
    * **No decorative commands.**  ``vx`` is never emitted between zero and
      ``VX_MIN_EFFECTIVE``: the measured gait onset means such a command looks
      like motion in the HUD and produces none on the floor.
    * **Independent turn signs.**  Right and left use different gains and
      different dead zones, because the measured yaw rates are not mirror
      images of each other.
    """

    ctrl_hz: float = 50.0
    command: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_hz

    def reset(self) -> None:
        self.command[:] = 0.0

    def raw_command(
        self, state: str, heading_error: float, caller_range: float | None
    ) -> tuple[float, float, float]:
        """Unfiltered target command for this state, heading error and range."""
        if state != "APPROACH":
            return (0.0, 0.0, 0.0)
        if caller_range is not None and caller_range <= STOP_RANGE:
            return (0.0, 0.0, 0.0)

        # Yaw first: the two signs are tuned independently against measurement.
        if heading_error < 0.0:
            wz = -clamp(KP_RIGHT * abs(heading_error), 0.0, WZ_MAX_RIGHT)
            if abs(wz) < WZ_MIN_RIGHT:
                wz = 0.0
        else:
            wz = clamp(KP_LEFT * heading_error, 0.0, WZ_MAX_LEFT)
            if wz < WZ_MIN_LEFT:
                wz = 0.0

        # Speed: cruise while pointed at the caller, ease off for the final
        # metre, and slow down while flying a large arc so the turn closes.
        far = caller_range is None or caller_range > 1.10
        if abs(heading_error) > math.radians(35.0):
            vx = VX_TURN
        elif far:
            vx = VX_CRUISE
        else:
            vx = VX_APPROACH
        # Never emit a sub-onset command: it is a decorative number.
        if vx < VX_MIN_EFFECTIVE:
            vx = VX_MIN_EFFECTIVE
        return (vx, 0.0, wz)

    def update(
        self, state: str, heading_error: float, caller_range: float | None
    ) -> np.ndarray:
        target = np.asarray(
            self.raw_command(state, heading_error, caller_range), dtype=np.float32
        )
        if state != "APPROACH":
            self.command[:] = 0.0
            return self.command.copy()
        # Inside APPROACH the command is applied directly.  A low-pass filter
        # here would spend the first ticks BELOW the measured gait onset, which
        # is not a gentle start - it is no motion at all followed by a jump.
        self.command[:] = target
        return self.command.copy()
