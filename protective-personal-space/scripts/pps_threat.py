#!/usr/bin/env python3
"""Measured constant-velocity intrusion prediction; no scenario schedule imports."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from pps_states import (ALERT_RANGE_M, BUFFER_M, PREDICT_HORIZON_S,
                        PREDICT_MARGIN_M, PREDICT_TTC_MAX_S,
                        SQUEEZE_SEPARATION_DEG)


@dataclass(frozen=True)
class Prediction:
    name: str
    range_m: float
    cpa_m: float
    ttc_s: float
    bearing_deg: float
    closing_mps: float
    intrusion: bool

    def record(self):
        return {"name": self.name, "range_m": round(self.range_m, 4),
                "cpa_m": round(self.cpa_m, 4), "ttc_s": round(self.ttc_s, 3),
                "bearing_deg": round(self.bearing_deg, 2),
                "closing_mps": round(self.closing_mps, 4),
                "intrusion": self.intrusion}


def predict_one(name, ward_xy, ward_velocity, person_xy, person_velocity) -> Prediction:
    relative = np.asarray(person_xy)-np.asarray(ward_xy)
    velocity = np.asarray(person_velocity)-np.asarray(ward_velocity)
    range_m = float(np.linalg.norm(relative))
    vv = float(velocity @ velocity)
    ttc = 0.0 if vv < 1e-9 else float(np.clip(-(relative @ velocity)/vv, 0.0, PREDICT_HORIZON_S))
    cpa = float(np.linalg.norm(relative + velocity*ttc))
    closing = 0.0 if range_m < 1e-9 else -float(relative @ velocity)/range_m
    bearing = math.degrees(math.atan2(float(relative[1]), float(relative[0])))
    intrusion = (range_m <= ALERT_RANGE_M and cpa <= BUFFER_M-PREDICT_MARGIN_M
                 and 0.0 < ttc <= PREDICT_TTC_MAX_S and closing > 0.015)
    return Prediction(name, range_m, cpa, ttc, bearing, closing, intrusion)


def predict_all(ward, people: dict, exclude: set[str] | None = None) -> list[Prediction]:
    exclude = exclude or set()
    found = []
    for name, person in people.items():
        if name in exclude or not getattr(person, "present", True):
            continue
        found.append(predict_one(name, ward.pos, ward.velocity, person.pos, person.velocity))
    return sorted(found, key=lambda p: (not p.intrusion, p.ttc_s, p.cpa_m))


def active(predictions: list[Prediction]) -> list[Prediction]:
    return [p for p in predictions if p.intrusion]


def priority(predictions: list[Prediction]) -> Prediction | None:
    live = active(predictions)
    return min(live, key=lambda p: (p.ttc_s, p.cpa_m)) if live else None


def angle_separation(a: float, b: float) -> float:
    return abs((a-b+180.0)%360.0-180.0)


def squeeze_pair(predictions: list[Prediction]):
    # Callers pass the already confirmed live set. A person already inside the
    # buffer can have TTC=0 and must remain eligible for a simultaneous squeeze.
    live = list(predictions)
    best = None
    for i, first in enumerate(live):
        for second in live[i+1:]:
            separation = angle_separation(first.bearing_deg, second.bearing_deg)
            if separation >= SQUEEZE_SEPARATION_DEG:
                score = min(first.ttc_s, second.ttc_s)
                if best is None or score < best[0]: best = (score, first, second, separation)
    return None if best is None else best[1:]
