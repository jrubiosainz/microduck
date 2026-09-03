#!/usr/bin/env python3
"""Measured locomotion controller for protective station changes.

The stock policy cannot strafe or turn in place. Every side change is therefore
pure pursuit along a walked arc; every hold is literal zero. Reverse is reserved
for yielding to the protected person and is projected on its starting heading.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from pps_states import (KP_YAW_LEFT, KP_YAW_RIGHT, VX_ESCORT, VX_REPOSITION,
                        VX_RETREAT, VX_SETTLE, WALKING_STATES,
                        WZ_MAX_LEFT, WZ_MAX_RIGHT, WZ_MIN_LEFT, WZ_MIN_RIGHT,
                        ZERO_COMMAND_STATES)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value, low, high):
    return float(min(max(value, low), high))


@dataclass
class PpsController:
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def reset(self):
        self.command[:] = 0.0

    def yaw_command(self, desired: float, current: float) -> float:
        error = wrap(desired-current)
        if error >= 0:
            value = clamp(KP_YAW_LEFT*error, 0, WZ_MAX_LEFT)
            return 0.0 if value < WZ_MIN_LEFT else value
        value = clamp(KP_YAW_RIGHT*abs(error), 0, WZ_MAX_RIGHT)
        return 0.0 if value < WZ_MIN_RIGHT else -value

    def raw(self, state: str, duck_xy, duck_yaw: float, target_xy=None,
            settle: bool = False, retreat_heading: float | None = None):
        if state in ZERO_COMMAND_STATES or state == "DONE":
            return (0.0, 0.0, 0.0)
        if state == "RETREAT":
            heading = duck_yaw if retreat_heading is None else retreat_heading
            yaw = self.yaw_command(heading, duck_yaw)
            return (VX_RETREAT, 0.0, yaw)
        if state not in WALKING_STATES or target_xy is None:
            return (0.0, 0.0, 0.0)
        delta = np.asarray(target_xy)-np.asarray(duck_xy)
        distance = float(np.linalg.norm(delta))
        desired = math.atan2(float(delta[1]), float(delta[0]))
        yaw = self.yaw_command(desired, duck_yaw)
        if distance <= 0.08:
            return (0.0, 0.0, yaw)
        if state == "ESCORT":
            vx = VX_SETTLE if distance < 0.30 else VX_ESCORT
        else:
            vx = VX_SETTLE if settle or distance < 0.28 else VX_REPOSITION
        return (vx, 0.0, yaw)

    def update(self, state: str, duck_xy, duck_yaw: float, **kwargs):
        self.command[:] = np.asarray(
            self.raw(state, duck_xy, duck_yaw, **kwargs), dtype=np.float32)
        return self.command.copy()
