#!/usr/bin/env python3
"""WHO the duck is listening to, and the hand history it reads them with.

Two things the confirm gate depends on but which are not the confirm gate, kept
here so :mod:`gest_detect` stays about ONE question - has this person sustained
a readable command for long enough to act on.

* :class:`Acquisition` is the explicit ``search -> found -> locked`` walk that
  decides WHO.  Only the requested identity can satisfy it, and once locked no
  other person can become the subject for the rest of the session.
* :class:`HandTrack` is the rolling window of hand positions the duck observed
  itself, from which the motion features are derived.

THE HAND HISTORY IS THE DUCK'S OWN, ACCUMULATED TICK BY TICK
--------------------------------------------------------------
Motion features are never read from the animation.  Each tick the detector
stores the world position of each hand it could see and accumulates PATH between
successive ticks over a rolling window.  That is a quantity a real robot with a
pose estimator would have, and it is what makes an oscillation distinguishable
from an arm on its way up - see ``gest_gesture.OSCILLATION_WANDER_BAR``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from gest_gesture import MOTION_WINDOW_S
from gest_states import ACQUIRE_CONFIRM_S



@dataclass
class HandTrack:
    """One person's recent hand positions, and the motion features from them.

    A rolling window of world positions the duck actually observed, from which
    PATH and NET displacement are derived.  Both are needed: path alone cannot
    tell a beckon from an arm rising, because over a short window a rising arm
    also travels a long way.
    """

    window_s: float = MOTION_WINDOW_S
    dt: float = 0.02
    _left: deque = field(default_factory=deque)
    _right: deque = field(default_factory=deque)

    def _capacity(self) -> int:
        return max(2, int(round(self.window_s / max(self.dt, 1e-9))) + 1)

    def push(self, left, right) -> None:
        for track, point in ((self._left, left), (self._right, right)):
            track.append(np.asarray(point, dtype=np.float64).copy())
            while len(track) > self._capacity():
                track.popleft()

    def clear(self) -> None:
        self._left.clear()
        self._right.clear()

    @staticmethod
    def _path_and_net(track: deque) -> tuple[float, float]:
        if len(track) < 2:
            return 0.0, 0.0
        path = 0.0
        for before, after in zip(track, list(track)[1:]):
            path += float(np.linalg.norm(after - before))
        net = float(np.linalg.norm(track[-1] - track[0]))
        return path, net

    def features(self, arm_span: float) -> tuple[float, float, bool]:
        """Normalised hand path, wander, and whether the window is full.

        ``wander`` is path divided by net displacement: about 1 for a hand
        travelling one way, far more for one that swung out and came back.  The
        window must be FULL before either is trusted, because a partial window
        under-reports path and would make a real oscillation look still.
        """
        left_path, left_net = self._path_and_net(self._left)
        right_path, right_net = self._path_and_net(self._right)
        if left_path >= right_path:
            path, net = left_path, left_net
        else:
            path, net = right_path, right_net
        full = len(self._left) >= self._capacity()
        return (path / max(arm_span, 1e-9),
                path / max(net, 1e-6),
                full)


@dataclass
class Candidate:
    """An accumulating confirmation of one command from the locked person."""

    command: str
    template: str
    began_at_s: float
    ticks: int = 0
    matching: int = 0
    readable_ticks: int = 0
    best_confidence: float = 0.0
    last_confidence: float = 0.0
    rule: str = ""
    features: dict = field(default_factory=dict)

    @property
    def held_s(self) -> float:
        return self.ticks * 0.02

    @property
    def fraction(self) -> float:
        return self.matching / self.ticks if self.ticks else 0.0


@dataclass
class Acquisition:
    """The explicit ``search -> found -> locked`` walk that decides WHO.

    Only the requested identity can satisfy it, and the requested identity is a
    body-identity SEMANTIC PROXY rather than an RGB recognition - stated here
    and everywhere it surfaces.  What is real is the camera gate it must pass:
    frustum containment plus an occlusion ray cast, sustained for a MEASURED
    dwell.
    """

    wanted: str
    state: str = "search"
    locked: str = ""
    visible_s: float = 0.0
    found_at_s: float = 0.0
    locked_at_s: float = 0.0
    # Every person the camera confirmed as visible during the search, so
    # "it locked onto the right one" is a choice among several rather than the
    # only option it ever had.
    seen: list[str] = field(default_factory=list)

    def feed(self, t: float, dt: float, visibility: dict) -> str:
        for name, entry in visibility.items():
            if entry.get("visible") and name not in self.seen:
                self.seen.append(name)
        if self.state == "locked":
            return self.locked
        entry = visibility.get(self.wanted, {})
        if entry.get("visible"):
            if self.state == "search":
                self.state = "found"
                self.found_at_s = t
            self.visible_s += dt
            if self.visible_s >= ACQUIRE_CONFIRM_S:
                self.state = "locked"
                self.locked = self.wanted
                self.locked_at_s = t
        else:
            self.visible_s = 0.0
            if self.state == "found":
                self.state = "search"
        return self.locked

    def as_record(self) -> dict:
        return {
            "wanted": self.wanted,
            "state": self.state,
            "locked": self.locked,
            "found_at_s": round(float(self.found_at_s), 3),
            "locked_at_s": round(float(self.locked_at_s), 3),
            "people_seen_during_search": list(self.seen),
        }
