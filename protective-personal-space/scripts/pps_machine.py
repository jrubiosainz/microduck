#!/usr/bin/env python3
"""Pure state machine for measured personal-space events."""
from __future__ import annotations
from dataclasses import dataclass, field
from pps_states import CLEAR_HOLD_S


@dataclass
class PpsMachine:
    state: str = "ESCORT"
    state_since: float = 0.0
    selected: str | None = None
    secondary: str | None = None
    target: list[float] | None = None
    transitions: list[dict] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)
    handled: set[str] = field(default_factory=set)
    _episode: dict = field(default_factory=dict)

    def go(self, t, state, **detail):
        self.transitions.append({"t": round(t,3), "from": self.state,
                                 "to": state, **detail})
        self.state, self.state_since = state, t

    def start_episode(self, t, kind, selected, target, secondary=None, **evidence):
        self.selected, self.secondary = selected, secondary
        self.target = None if target is None else [float(v) for v in target]
        self._episode = {"index": len(self.episodes), "kind": kind,
                         "started_at_s": round(t,3), "selected": selected,
                         "secondary": secondary, "target": self.target,
                         **evidence}

    def close_episode(self, t, outcome="recovered"):
        if self._episode:
            self._episode.update({"ended_at_s": round(t,3), "outcome": outcome,
                                  "duration_s": round(t-self._episode["started_at_s"],3)})
            self.episodes.append(self._episode)
            if self.selected: self.handled.add(self.selected)
            if self.secondary: self.handled.add(self.secondary)
        self._episode = {}
        self.selected = self.secondary = None
        self.target = None

    def update(self, t, *, escort_joined=False, threat=None, squeeze=None,
               ward_approach=False, target_reached=False, threat_clear=False,
               retreat_complete=False, finish=False):
        before = self.state
        # The protected person always has priority. If she walks directly at
        # the duck during an interpose/return, abandon that station and yield;
        # continuing to body-block her would invert the behavior's purpose.
        if (squeeze and self.state in
                ("PREDICT_INTRUSION", "HOLD_BUFFER",
                 "THREAT_CLEAR", "RETURN_ESCORT", "MONITOR")):
            self.close_episode(t, "superseded_by_squeeze")
            first, second, target, evidence = squeeze
            self.start_episode(t, "squeeze", first, target, second, **evidence)
            self.go(t, "MULTI_THREAT", primary=first, secondary=second)
        elif (ward_approach and self.state in
                ("INTERPOSE", "HOLD_BUFFER", "THREAT_CLEAR", "RETURN_ESCORT")):
            self.close_episode(t, "yielded_to_ward")
            self.start_episode(t, "ward_approach", "aina", None)
            self.go(t, "PERSON_APPROACH", reason="protected person closing")
        elif self.state == "ESCORT":
            if escort_joined: self.go(t, "MONITOR", reason="neutral escort established")
        elif self.state == "MONITOR":
            if finish:
                self.go(t, "DONE", reason="session complete in escort")
            elif ward_approach:
                self.start_episode(t,"ward_approach","aina",None)
                self.go(t,"PERSON_APPROACH",reason="protected person closing")
            elif squeeze:
                first, second, target, evidence = squeeze
                self.start_episode(t,"squeeze",first,target,second,**evidence)
                self.go(t,"MULTI_THREAT",primary=first,secondary=second)
            elif threat:
                name,target,evidence = threat
                self.start_episode(t,"intrusion",name,target,**evidence)
                self.go(t,"PREDICT_INTRUSION",threat=name,**evidence)
        elif self.state == "PREDICT_INTRUSION":
            self.go(t,"INTERPOSE",threat=self.selected)
        elif self.state == "INTERPOSE":
            if target_reached: self.go(t,"HOLD_BUFFER",threat=self.selected)
        elif self.state == "HOLD_BUFFER":
            if threat_clear: self.go(t,"THREAT_CLEAR",threat=self.selected)
        elif self.state == "THREAT_CLEAR":
            if t-self.state_since >= CLEAR_HOLD_S:
                self.go(t,"RETURN_ESCORT",reason="threat clear held")
        elif self.state == "RETURN_ESCORT":
            if escort_joined:
                self.close_episode(t); self.go(t,"MONITOR",reason="escort restored")
        elif self.state == "PERSON_APPROACH":
            self.go(t,"RETREAT",reason="yielding to protected person")
        elif self.state == "RETREAT":
            if retreat_complete: self.go(t,"RECOVER",reason="retreat range gained")
        elif self.state == "MULTI_THREAT":
            self.go(t,"ESCAPE_GAP",primary=self.selected,secondary=self.secondary)
        elif self.state == "ESCAPE_GAP":
            if target_reached: self.go(t,"RECOVER",reason="safe gap reached")
        elif self.state == "RECOVER":
            if escort_joined:
                self.close_episode(t); self.go(t,"MONITOR",reason="escort recovered")
        return self.state, self.state != before

    def finish(self,t):
        if self.state=="MONITOR": self.go(t,"DONE",reason="session complete")
