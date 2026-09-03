#!/usr/bin/env python3
"""The state machine: receive, plan, lead, notice she is lagging, wait, resume.

    RECEIVE_DESTINATION -> PLAN -> LEAD -> ARRIVE -> INDICATE -> DONE
                                    |  ^
                                    v  |
                            CHECK_FOLLOWER
                                    |
                                    v
                            WAIT_FOR_PERSON -> RESUME -+
                                                        |
                                    +-------------------+
                                    v
                                  LEAD

The machine never touches physics and never emits a command; ``guide_control``
does that from the state.  Keeping the two apart is what lets every transition
rule be unit-tested on hand-built inputs, with no MuJoCo anywhere.

FIVE INVARIANTS ARE STRUCTURAL RATHER THAN CHECKED AFTERWARDS
---------------------------------------------------------------
* **The destination is resolved once, by exact lookup, and never revised.**
  :meth:`receive` resolves the requested key against the registry and stores the
  entry.  There is no fuzzy match, no default and no re-resolution, so the route
  the duck walks is a consequence of the key it was given.  A second request
  with a different key raises rather than silently re-targeting.

* **A wait is CAUSED, never scheduled.**  ``CHECK_FOLLOWER`` is entered only
  after the follower has been continuously measured too far away for
  ``LAG_CONFIRM_S``, or continuously unseen for ``LOST_CONFIRM_S``.  The machine
  records the measurement it was given at that instant.  There is no timer, no
  waypoint index and no schedule lookup anywhere in this file: the scenario's
  own stall windows live in ``guide_follower`` and the machine cannot see them.

* **A resume is JUSTIFIED, never assumed.**  ``WAIT_FOR_PERSON`` ends only after
  she has been continuously close enough AND visible for ``RESUME_CONFIRM_S``.
  One good frame is not a catch-up, and the episode record carries the distance
  and the visibility that justified it.

* **Waiting is a state, not a speed.**  Forward gait onset on this scene is a
  MEASURED cliff between ``vx = 0.20`` (no gait at all) and ``vx = 0.22``, so a
  guide that "slowed down" for a lagging follower would emit a command that
  appears in the metrics and produces nothing on the floor.  The duck walks or
  it holds exactly zero, and ``WAIT_FOR_PERSON`` is where the zero lives.

* **An episode cannot chatter.**  After a resume the machine refuses to declare
  another lag for ``LAG_COOLDOWN_S``.  Without it, a follower hovering at the
  threshold produces an episode count that is really a count of ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from guide_layout import Destination, resolve_destination
from guide_states import (
    ACK_SECONDS,
    ARRIVE_MAX_S,
    CATCHUP_DISTANCE_M,
    CHECK_CONFIRM_S,
    CHECK_MAX_S,
    INDICATE_MAX_S,
    INDICATE_SECONDS,
    LAG_CONFIRM_S,
    LAG_COOLDOWN_S,
    LAG_DISTANCE_M,
    LEAD_MAX_S,
    LOST_CONFIRM_S,
    PLAN_DWELL_S,
    PLAN_MAX_S,
    RECEIVE_MAX_S,
    RESUME_CONFIRM_S,
    WAIT_MAX_S,
)


@dataclass
class GuideMachine:
    """Transitions, lag episodes and the arrival record.  No physics, no MuJoCo."""

    ctrl_hz: float = 50.0
    state: str = "RECEIVE_DESTINATION"
    state_since: float = 0.0
    follower: str = ""
    destination: Destination | None = None
    requested_key: str = ""
    request_t: float | None = None
    candidates: tuple[str, ...] = ()

    transitions: list[dict] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)
    timeouts: list[str] = field(default_factory=list)
    arrival: dict = field(default_factory=dict)

    _lagging_for: float = 0.0
    _unseen_for: float = 0.0
    _recovered_for: float = 0.0
    _contact_for: float = 0.0
    _episode: dict = field(default_factory=dict)
    _plan: object = None
    _last_resume_t: float | None = None
    _planned: bool = False
    _arrived: bool = False

    def __post_init__(self) -> None:
        self.dt = 1.0 / self.ctrl_hz

    # -- helpers ---------------------------------------------------------
    def _go(self, t: float, state: str, **detail) -> None:
        self.transitions.append(
            {"t": round(t, 3), "from": self.state, "to": state, **detail})
        self.state = state
        self.state_since = t

    def _elapsed(self, t: float) -> float:
        return t - self.state_since

    def _timeout(self, t: float, limit: float, label: str) -> bool:
        if self._elapsed(t) >= limit:
            self.timeouts.append(f"{label}@{t:.2f}s")
            return True
        return False

    def set_follower(self, name: str) -> None:
        """Record the identity being guided.  Callable once, by design."""
        if self.follower and self.follower != name:
            raise ValueError(
                f"the follower is {self.follower!r} and cannot be reassigned to "
                f"{name!r}: a guide leads one person")
        self.follower = name

    # -- the request ------------------------------------------------------
    def receive(self, t: float, key: str, candidates: tuple[str, ...]) -> None:
        """Resolve a semantic destination request by EXACT lookup.

        Raises on an unknown key and on a second, different request.  A guide
        that silently walked somewhere plausible when it did not understand
        would be worse than one that refused, and the gate could not tell the
        two apart.
        """
        if self.destination is not None:
            if key != self.requested_key:
                raise ValueError(
                    f"already leading to {self.requested_key!r}; refusing to "
                    f"re-target to {key!r} mid-route")
            return
        self.destination = resolve_destination(key)
        self.requested_key = key
        self.request_t = t
        self.candidates = tuple(candidates)

    def note_plan(self, t: float, plan) -> None:
        """Record that a route was searched.  The transition happens later.

        Storing rather than transitioning is what gives PLAN a real duration:
        see :meth:`_plan_state`.
        """
        self._planned = True
        self._plan = plan

    # -- the machine -----------------------------------------------------
    def update(self, t: float, *, distance_m: float, visible: bool,
               los_available: bool, route_remaining_m: float,
               facing_ok: bool) -> tuple[str, bool]:
        """Advance one control tick.

        Every argument is a MEASUREMENT the duck took this tick:
        ``distance_m`` from the same contact probe every clearance uses,
        ``visible`` from the real head camera, ``los_available`` from the planar
        occluder test, ``route_remaining_m`` from its own projection onto the
        route it planned, ``facing_ok`` from its own yaw against the destination
        bearing.  Nothing here reads the scenario.
        """
        before = self.state

        if self.state == "RECEIVE_DESTINATION":
            self._receive_state(t)
        elif self.state == "PLAN":
            self._plan_state(t)
        elif self.state in ("LEAD", "RESUME"):
            self._lead_state(t, distance_m, visible, los_available,
                             route_remaining_m)
        elif self.state == "CHECK_FOLLOWER":
            self._check_state(t, distance_m, visible)
        elif self.state == "WAIT_FOR_PERSON":
            self._wait_state(t, distance_m, visible)
        elif self.state == "ARRIVE":
            self._arrive_state(t, facing_ok, distance_m)
        elif self.state == "INDICATE":
            self._indicate_state(t)

        return self.state, self.state != before

    # -- per-state rules -------------------------------------------------
    def _receive_state(self, t: float) -> None:
        """Stand still, acknowledge, then plan.

        The duck holds an exact zero command throughout, so the acknowledgement
        is a visible standstill rather than a caption.  A guide that walked off
        the instant it was addressed would not have acknowledged anything.
        """
        if self.destination is None:
            self._timeout(t, RECEIVE_MAX_S, "RECEIVE_DESTINATION")
            return
        if self.request_t is not None and t - self.request_t >= ACK_SECONDS:
            self._go(t, "PLAN", destination=self.requested_key,
                     acknowledged_for_s=round(t - self.request_t, 3),
                     reason="request acknowledged")

    def _plan_state(self, t: float) -> None:
        """Hold still while the route is searched, then lead off.

        The search itself happens in ``rollout_guide`` because it needs the
        duck's measured pose and the measured crowd; :meth:`note_plan` hands the
        result back.  The transition is deliberately deferred to a LATER tick
        than the one the plan arrived on, so PLAN is a state the run actually
        spends time in and a viewer can see the route appear before the duck
        moves.  A state that is entered and left within one tick is a state that
        did not happen.
        """
        if self._plan is not None and self._elapsed(t) >= PLAN_DWELL_S:
            self._go(t, "LEAD", destination=self.requested_key,
                     route_m=round(self._plan.length_m, 4),
                     bends=len(self._plan.bends),
                     searched_for_s=round(self._elapsed(t), 3),
                     reason="route planned; leading off")
            return
        self._timeout(t, PLAN_MAX_S, "PLAN")

    def _lead_state(self, t: float, distance_m: float, visible: bool,
                    los_available: bool, route_remaining_m: float) -> None:
        """Walk the route, and watch the person behind at every tick.

        RESUME is a distinct state from LEAD so the timeline can show that the
        duck started again BECAUSE she caught up, rather than merely continuing.
        It behaves identically otherwise and falls back to LEAD once it has run
        long enough to be legible.
        """
        # Arrival wins over everything: a duck at the destination should stop,
        # not start a fresh lag episode about a person who is about to arrive
        # too.
        if route_remaining_m <= 0.0:
            self._go(t, "ARRIVE", remaining_m=round(route_remaining_m, 4),
                     reason="reached the end of the planned route")
            return

        lagging = distance_m > LAG_DISTANCE_M
        # A follower who is out of sight while LINE OF SIGHT EXISTS is genuinely
        # unaccounted for.  While an occluder is in the way, the duck is not
        # blind through its own fault, but it still cannot see her — so both
        # count, and the episode records which it was.
        self._lagging_for = self._lagging_for + self.dt if lagging else 0.0
        self._unseen_for = 0.0 if visible else self._unseen_for + self.dt

        if self._last_resume_t is not None \
                and t - self._last_resume_t < LAG_COOLDOWN_S:
            return

        cause = ""
        if self._lagging_for >= LAG_CONFIRM_S:
            cause = "lag"
        elif self._unseen_for >= LOST_CONFIRM_S:
            cause = "loss"
        if not cause:
            if self.state == "RESUME" and self._elapsed(t) >= 2.0:
                self._go(t, "LEAD", reason="resumed and back on the route")
            return

        self._episode = {
            "index": len(self.episodes),
            "cause": cause,
            "detected_at_s": round(t, 3),
            "distance_at_detect_m": round(distance_m, 4),
            "visible_at_detect": bool(visible),
            "los_available_at_detect": bool(los_available),
            "lagging_for_s": round(self._lagging_for, 3),
            "unseen_for_s": round(self._unseen_for, 3),
            "lag_threshold_m": LAG_DISTANCE_M,
            "lost_threshold_s": LOST_CONFIRM_S,
            "route_remaining_at_detect_m": round(route_remaining_m, 4),
        }
        self._go(t, "CHECK_FOLLOWER", cause=cause,
                 distance_m=round(distance_m, 4), visible=bool(visible),
                 reason=f"{cause} confirmed by measurement")

    def _check_state(self, t: float, distance_m: float, visible: bool) -> None:
        """Pull round until the duck can actually SEE her, then stop and wait.

        CHECK_FOLLOWER exists separately from WAIT_FOR_PERSON because squaring
        up is a different act from waiting, and on this robot it is the harder
        one.  MEASURED head yaw range is +/-170 deg, and a follower walking the
        duck's own trail approaches from exactly astern — the first full run put
        her at 173 deg for 1.4 s, 3 deg BEYOND the head's reach, and the camera
        correctly could not see her.

        The duck cannot turn on the spot (MEASURED at 1.6 deg/s), so it does
        what somebody carrying a tray does: it walks a small arc round.  That is
        why this state is NOT exact-zero while WAIT_FOR_PERSON is — stopping
        somewhere you can watch from is part of stopping, and a guide frozen with
        its back to the person it is waiting for is not watching.

        The ceiling here TRANSITIONS rather than merely logging.  An earlier
        draft only appended to ``timeouts``, so when the arc budget ran out the
        duck emitted zero for ever and the run spent 71 s in this state: a
        ceiling that does not move the machine is not a ceiling.  Reaching it now
        starts the wait anyway and records that the squaring-up was incomplete,
        which is the honest outcome — a guide that cannot get round should still
        stop and wait rather than walk on.
        """
        if self._timeout(t, CHECK_MAX_S, "CHECK_FOLLOWER"):
            self._episode["checked_at_s"] = round(t, 3)
            self._episode["distance_at_check_m"] = round(distance_m, 4)
            self._episode["squaring_up_incomplete"] = True
            self._contact_for = 0.0
            self._recovered_for = 0.0
            self._go(t, "WAIT_FOR_PERSON",
                     distance_m=round(distance_m, 4), visible=bool(visible),
                     reason="squaring-up ceiling reached; waiting here anyway")

    def confirm_check(self, t: float, *, looking_back: bool,
                      distance_m: float, visible: bool,
                      bearing_ok: bool) -> None:
        """Called by the rollout with what the camera can measure this tick.

        Separated from :meth:`update` because ``looking_back`` and
        ``bearing_ok`` are claims about where the rendering head is pointing and
        where she sits relative to the trunk, which the machine does not
        otherwise need to know.

        THE HEAD IS ALLOWED TO STILL BE MOVING.  Requiring visual contact is
        right; requiring the tracker to have SETTLED is not, and it cost the
        behavior most of its waiting.  At the MEASURED 26 deg/s tracking rate the
        head takes about 4 s to swing from ahead to 155 deg astern, and an
        earlier draft held CHECK_FOLLOWER for all of it plus the confirm window —
        11 s per episode, by which time she had already caught up and
        WAIT_FOR_PERSON lasted a single second.  The duck was genuinely stopped
        and genuinely watching throughout; it was simply doing it in the state
        that does not claim to be waiting.

        So the exit needs only that the camera CAN SEE HER, held for
        :data:`CHECK_CONFIRM_S`.  ``looking_back`` and ``bearing_ok`` are still
        measured and still recorded, because they are what the visibility gate
        is graded against, but they do not gate the transition.
        """
        if self.state != "CHECK_FOLLOWER":
            return
        self._contact_for = self._contact_for + self.dt if visible else 0.0
        if self._contact_for < CHECK_CONFIRM_S:
            return
        self._episode["checked_at_s"] = round(t, 3)
        self._episode["distance_at_check_m"] = round(distance_m, 4)
        self._episode["contact_established_for_s"] = round(self._contact_for, 3)
        self._episode["head_on_her_at_check"] = bool(looking_back)
        self._episode["bearing_inside_head_reach_at_check"] = bool(bearing_ok)
        self._contact_for = 0.0
        self._recovered_for = 0.0
        self._go(t, "WAIT_FOR_PERSON",
                 distance_m=round(distance_m, 4), visible=bool(visible),
                 reason="visual contact established; waiting here for her")

    def _wait_state(self, t: float, distance_m: float, visible: bool) -> None:
        """Hold exactly still until she is BOTH near enough AND visible.

        Both conditions, continuously, for ``RESUME_CONFIRM_S``.  Requiring only
        distance would let the duck set off again while she was behind a
        partition; requiring only visibility would let it set off while she was
        still three metres back.
        """
        recovered = distance_m <= CATCHUP_DISTANCE_M and visible
        self._recovered_for = self._recovered_for + self.dt if recovered else 0.0
        if self._recovered_for >= RESUME_CONFIRM_S:
            self._episode.update({
                "resumed_at_s": round(t, 3),
                "distance_at_resume_m": round(distance_m, 4),
                "visible_at_resume": bool(visible),
                "recovered_for_s": round(self._recovered_for, 3),
                "catchup_threshold_m": CATCHUP_DISTANCE_M,
                "resume_confirm_s": RESUME_CONFIRM_S,
                "wait_duration_s": round(t - self._episode["checked_at_s"], 3),
                "episode_duration_s": round(
                    t - self._episode["detected_at_s"], 3),
            })
            self.episodes.append(self._episode)
            self._episode = {}
            self._last_resume_t = t
            self._lagging_for = 0.0
            self._unseen_for = 0.0
            self._recovered_for = 0.0
            self._go(t, "RESUME", distance_m=round(distance_m, 4),
                     reason="she caught up and is visible; leading on")
            return
        self._timeout(t, WAIT_MAX_S, "WAIT_FOR_PERSON")

    def _arrive_state(self, t: float, facing_ok: bool,
                      distance_m: float) -> None:
        """Stop, turn to face the destination, and only then indicate.

        A guide that announced arrival while still walking, or while facing away
        from the thing it had led somebody to, has not arrived in any useful
        sense.
        """
        if facing_ok:
            self._arrived = True
            self.arrival = {
                "arrived_at_s": round(t, 3),
                "destination": self.requested_key,
                "follower_distance_at_arrival_m": round(distance_m, 4),
            }
            self._go(t, "INDICATE", reason="facing the destination; indicating")
            return
        self._timeout(t, ARRIVE_MAX_S, "ARRIVE")

    def _indicate_state(self, t: float) -> None:
        """Hold the arrival gesture for its declared duration, then finish."""
        if self._elapsed(t) >= INDICATE_SECONDS:
            self.arrival["indicated_for_s"] = round(self._elapsed(t), 3)
            self._go(t, "DONE", reason="arrival indicated")
            return
        self._timeout(t, INDICATE_MAX_S, "INDICATE")

    # -- bookkeeping ------------------------------------------------------
    @property
    def completed_episodes(self) -> int:
        return len(self.episodes)

    @property
    def planned(self) -> bool:
        return self._planned

    @property
    def arrived(self) -> bool:
        return self._arrived

    @property
    def waiting(self) -> bool:
        return self.state == "WAIT_FOR_PERSON"

    def summary(self) -> dict:
        return {
            "follower": self.follower,
            "state": self.state,
            "requested_key": self.requested_key,
            "request_t_s": self.request_t,
            "candidates": list(self.candidates),
            "destination": None if self.destination is None
            else self.destination.key,
            "transitions": list(self.transitions),
            "episodes": list(self.episodes),
            "arrival": dict(self.arrival),
            "timeouts": list(self.timeouts),
        }
