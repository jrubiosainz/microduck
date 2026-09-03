#!/usr/bin/env python3
"""Per-cycle and per-state bookkeeping during a rollout.

Accumulators only: what happened during each JOIN/ADVANCE, what the command
peaked at in each stationary state, and whether the order read matched the
truth on every tick.  Held apart from the integration loop so that loop reads as
the five ordered steps it is.

ONE RULE HERE IS SUBSTANTIVE rather than clerical.  Tracking is graded only
while the subject is still IN THE QUEUE: a predecessor who has been served walks
away across the hall, and requiring the duck to keep them in frame would grade
it on watching somebody who is no longer the person in front - the queueing
equivalent of demanding sight through a wall.  Those steps are recorded
separately rather than dropped.
"""

from __future__ import annotations

import numpy as np

from queue_constants import STATIONARY_STATES
from queue_people import QUEUE_NAMES


class RolloutTracker:
    """Every accumulator a rollout fills in, and nothing else."""

    def __init__(self):
        self.cycle_path: dict[int, float] = {}
        self.cycle_start_arc: dict[int, float] = {}
        self.cycle_tracking: dict[int, list[bool]] = {}
        self.cycle_tracking_after_service: dict[int, list[bool]] = {}
        self.cycle_command_max: dict[int, float] = {}
        self.cycle_cross_track: dict[int, list[float]] = {}
        self.stationary_command_max: dict[str, float] = {}
        self.order_samples: list[dict] = []

    def note_step(self, *, state, command, cycle_index, arc_before, cross,
                  travelled, subject, camera_state, display_reading):
        """Fold one control tick into the accumulators."""
        command_peak = float(np.max(np.abs(command)))
        if state in STATIONARY_STATES:
            self.stationary_command_max[state] = max(
                self.stationary_command_max.get(state, 0.0), command_peak)
        if state not in ("JOIN", "ADVANCE"):
            return
        self.cycle_path[cycle_index] = (
            self.cycle_path.get(cycle_index, 0.0) + travelled)
        self.cycle_start_arc.setdefault(cycle_index, float(arc_before))
        self.cycle_cross_track.setdefault(cycle_index, []).append(float(cross))
        self.cycle_command_max[cycle_index] = max(
            self.cycle_command_max.get(cycle_index, 0.0), command_peak)
        if subject is None:
            return
        seen = bool(camera_state["people"].get(subject, {}).get("visible", False))
        if subject in display_reading.members:
            self.cycle_tracking.setdefault(cycle_index, []).append(seen)
        else:
            self.cycle_tracking_after_service.setdefault(
                cycle_index, []).append(seen)

    def note_order(self, *, display_t, state, display_reading, display_people):
        """Record this tick's ordering verdict against the ground truth.

        Graded over the WHOLE rollout rather than at one instant, so a lucky
        sample cannot carry the claim.
        """
        truth = [name for name in QUEUE_NAMES
                 if display_people[name].in_queue
                 and name in display_reading.members]
        self.order_samples.append({
            "t": display_t, "state": state,
            "inferred": list(display_reading.order),
            "truth": truth,
            "correct": list(display_reading.order) == truth,
            "tail": display_reading.tail,
            "true_tail": truth[-1] if truth else None,
            "tail_correct": bool(truth and display_reading.tail == truth[-1]),
        })
        return truth
