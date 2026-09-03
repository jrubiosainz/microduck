#!/usr/bin/env python3
"""Everything the rollout ACCUMULATES, in one object with no physics in it.

Split out of ``rollout_etiquette`` so that the tick loop stays about ORDER - what
is measured before what - and this stays about TALLYING.  The rollout owns one of
these and forwards each tick's measurements into it; ``etiquette_summary`` reads
it afterwards.

FOUR ACCUMULATORS HERE ENCODE A DECISION THAT IS EASY TO GET WRONG
--------------------------------------------------------------------
* :meth:`note_zone` records the WORST encroachment into each etiquette zone
  together with the tick it happened on, not merely a boolean.  A gate that only
  knew "it entered the threshold" could not say by how much or when, and a
  violation with no number attached is a violation nobody can act on.

* :meth:`note_aperture` counts, per aperture, the ticks on which the duck shared
  it with somebody.  That is the side-by-side failure, and it is counted rather
  than flagged so the metrics can report zero out of a real denominator.

* :meth:`note_order` tracks the MINIMUM gap to the guardian along the shared
  route, signed.  A single negative tick means the duck overtook her, which is
  why the minimum is kept rather than an average.

* :meth:`note_crossing` records, for each aperture, the door's measured open
  fraction at the instant the duck's own footprint first entered it.  This is
  what makes "it never moved through a closed door" a number rather than a
  claim, and it is taken at the FIRST entry so a door that opened later cannot
  retroactively excuse an early crossing.
"""

from __future__ import annotations

import numpy as np


class RolloutTally:
    """Per-run and per-phase accumulators.  No MuJoCo, no policy, no time."""

    def __init__(self, dt: float, initial_trunk_z: float):
        self.dt = float(dt)

        # -- locomotion health ------------------------------------------
        self.min_trunk_z = float(initial_trunk_z)
        self.path_m = 0.0
        self.fallen_steps = 0
        self.contact_steps = 0

        # -- clearance ---------------------------------------------------
        self.min_person_clearance = float("inf")
        self.min_person_name = ""
        self.min_scenery_clearance = float("inf")
        self.min_scenery_geom = ""
        self.min_clearance_by_person: dict[str, float] = {}

        # -- per-state ----------------------------------------------------
        self.state_steps: dict[str, int] = {}
        self.state_command_max: dict[str, float] = {}
        self.state_path_m: dict[str, float] = {}
        self.zero_command_violations: list[dict] = []

        # -- walking ------------------------------------------------------
        self.walk_path_m = 0.0
        self.walk_steps = 0
        self.max_cross_track_m = 0.0

        # -- the etiquette zones ------------------------------------------
        # name -> {"worst_m": float, "at_s": float, "steps": int}
        self.zone_worst: dict[str, dict] = {}
        # Ticks in each zone that happened BEFORE that zone was released.
        self.zone_violation_steps: dict[str, int] = {}

        # -- apertures -----------------------------------------------------
        self.aperture_steps: dict[str, int] = {}
        self.aperture_shared_steps: dict[str, int] = {}
        self.aperture_shared_with: dict[str, set] = {}
        self.crossings: dict[str, dict] = {}

        # -- order relative to the guardian --------------------------------
        self.min_guardian_gap_m = float("inf")
        self.max_guardian_gap_m = -float("inf")
        self.overtake_steps = 0
        self.guardian_gap_samples = 0

        # -- the cabin ------------------------------------------------------
        self.cabin_steps = 0
        self.min_cabin_margin_m = float("inf")
        self.cabin_outside_steps = 0

        # -- visibility, conditioned on line of sight ----------------------
        self.visible_steps = 0
        self.los_steps = 0
        self.visible_with_los = 0
        self.monitor_steps = 0
        self.monitor_los_steps = 0
        self.monitor_visible_with_los = 0
        self.blocked_by: dict[str, int] = {}
        self.subject_steps: dict[str, int] = {}
        self.subject_sequence: list[dict] = []
        self.subject_visible_los: dict[str, int] = {}
        self.subject_los: dict[str, int] = {}

        # -- the interlock ---------------------------------------------------
        self.interlock_holds = 0
        self.interlock_reasons: dict[str, int] = {}

    # -- locomotion --------------------------------------------------------
    def note_pose(self, trunk_z: float, travelled: float) -> None:
        self.min_trunk_z = min(self.min_trunk_z, float(trunk_z))
        if float(trunk_z) < 0.09:
            self.fallen_steps += 1
        self.path_m += float(travelled)

    def note_command(self, state: str, peak: float, travelled: float) -> None:
        self.state_command_max[state] = max(
            self.state_command_max.get(state, 0.0), float(peak))
        self.state_steps[state] = self.state_steps.get(state, 0) + 1
        self.state_path_m[state] = \
            self.state_path_m.get(state, 0.0) + float(travelled)

    def note_zero_violation(self, t: float, state: str, command) -> None:
        self.zero_command_violations.append(
            {"t": round(float(t), 3), "state": state,
             "command": [float(v) for v in command]})

    def note_walk(self, travelled: float, cross_track: float) -> None:
        self.walk_steps += 1
        self.walk_path_m += float(travelled)
        self.max_cross_track_m = max(self.max_cross_track_m, float(cross_track))

    # -- clearance ---------------------------------------------------------
    def note_clearance(self, clearances: dict, nearest: str,
                       scenery_gap: float, scenery_geom: str) -> None:
        if clearances[nearest] < self.min_person_clearance:
            self.min_person_clearance = clearances[nearest]
            self.min_person_name = nearest
        if clearances[nearest] <= 0.0:
            self.contact_steps += 1
        for name, gap in clearances.items():
            current = self.min_clearance_by_person.get(name, float("inf"))
            if gap < current:
                self.min_clearance_by_person[name] = float(gap)
        if scenery_gap < self.min_scenery_clearance:
            self.min_scenery_clearance = scenery_gap
            self.min_scenery_geom = scenery_geom

    # -- the zones ---------------------------------------------------------
    def note_zone(self, name: str, depth_m: float, t: float,
                  released: bool) -> None:
        """Record one tick's penetration into one zone.

        ``released`` is whether the behavior was ENTITLED to be in that zone at
        this instant.  Ticks before release are counted separately, which is
        what the gate reads: being in a doorway is not a failure, being in it
        early is.
        """
        entry = self.zone_worst.setdefault(
            name, {"worst_m": 0.0, "at_s": None, "steps": 0})
        if depth_m > 0.0:
            entry["steps"] += 1
            if depth_m > entry["worst_m"]:
                entry["worst_m"] = float(depth_m)
                entry["at_s"] = round(float(t), 3)
            if not released:
                self.zone_violation_steps[name] = \
                    self.zone_violation_steps.get(name, 0) + 1

    # -- apertures ----------------------------------------------------------
    def note_aperture(self, name: str, duck_inside: bool,
                      others_inside: list[str]) -> None:
        if not duck_inside:
            return
        self.aperture_steps[name] = self.aperture_steps.get(name, 0) + 1
        if others_inside:
            self.aperture_shared_steps[name] = \
                self.aperture_shared_steps.get(name, 0) + 1
            self.aperture_shared_with.setdefault(name, set()).update(
                others_inside)

    def note_crossing(self, name: str, t: float, open_fraction: float,
                      effective_gap_m: float, duck_xy) -> None:
        """First entry into one aperture, with the door state at that instant."""
        if name in self.crossings:
            return
        self.crossings[name] = {
            "aperture": name,
            "entered_at_s": round(float(t), 3),
            "open_fraction_at_entry": round(float(open_fraction), 4),
            "effective_gap_at_entry_m": round(float(effective_gap_m), 4),
            "duck_xy": [round(float(duck_xy[0]), 4),
                        round(float(duck_xy[1]), 4)],
        }

    # -- order --------------------------------------------------------------
    def note_order(self, gap_m: float, counts: bool) -> None:
        """The signed arc gap to the guardian along the duck's own route.

        ``counts`` excludes the ticks before she has set off at all, when the
        duck is legitimately ahead of a person who is not yet walking and the
        gap carries no information about overtaking.
        """
        if not counts:
            return
        self.guardian_gap_samples += 1
        self.min_guardian_gap_m = min(self.min_guardian_gap_m, float(gap_m))
        self.max_guardian_gap_m = max(self.max_guardian_gap_m, float(gap_m))
        if gap_m < 0.0:
            self.overtake_steps += 1

    # -- the cabin -----------------------------------------------------------
    def note_cabin(self, inside: bool, margin_m: float, riding: bool) -> None:
        if inside:
            self.cabin_steps += 1
            self.min_cabin_margin_m = min(self.min_cabin_margin_m,
                                          float(margin_m))
        elif riding:
            # Riding a lift from outside it is the failure this counts.
            self.cabin_outside_steps += 1

    # -- visibility -----------------------------------------------------------
    def note_visibility(self, *, subject: str, visible: bool, los_ok: bool,
                        monitoring: bool, blocker: str, t: float) -> None:
        self.subject_steps[subject] = self.subject_steps.get(subject, 0) + 1
        if not self.subject_sequence or \
                self.subject_sequence[-1]["subject"] != subject:
            self.subject_sequence.append(
                {"subject": subject, "from_s": round(float(t), 3),
                 "to_s": round(float(t), 3)})
        else:
            self.subject_sequence[-1]["to_s"] = round(float(t), 3)

        if visible:
            self.visible_steps += 1
        if los_ok:
            self.los_steps += 1
            self.subject_los[subject] = self.subject_los.get(subject, 0) + 1
            if visible:
                self.visible_with_los += 1
                self.subject_visible_los[subject] = \
                    self.subject_visible_los.get(subject, 0) + 1
        if monitoring:
            self.monitor_steps += 1
            if los_ok:
                self.monitor_los_steps += 1
                if visible:
                    self.monitor_visible_with_los += 1
        if not visible:
            key = blocker or "out_of_frustum"
            self.blocked_by[key] = self.blocked_by.get(key, 0) + 1

    # -- the interlock --------------------------------------------------------
    def note_interlock(self, blocked: bool, reason: str) -> None:
        if not blocked:
            return
        self.interlock_holds += 1
        self.interlock_reasons[reason] = \
            self.interlock_reasons.get(reason, 0) + 1
