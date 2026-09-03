#!/usr/bin/env python3
"""Per-cycle analysis: what each JOIN and ADVANCE actually achieved.

Separated from the gate evaluation so the two can be read apart: this module
turns the raw per-step records into one row per cycle, and ``queue_metrics``
decides whether those rows pass.

TWO MEASUREMENT SUBTLETIES LIVE HERE
------------------------------------
* **Corner cutting is the POSITIVE cross-track sense, and only on the bend.**
  MEASURED on the path itself: a point displaced 0.10 m from the fold TOWARD the
  arc's centre - i.e. inside the bend - projects to ``cross = +0.100``, and a
  point displaced away projects to ``cross = -0.100``.  An earlier version
  negated this and therefore graded swinging WIDE as corner cutting, which is
  the opposite failure and not one a queue cares about.  Restricting the measure
  to the bend matters too: on a straight there is no corner to cut, and any
  lateral offset there is ordinary tracking error already covered by
  ``max_cross_track_m``.
* **A standoff is only gradeable while the person it is measured against is
  still in the queue.**  Once they have been served and walked away there is no
  standoff to hold, and the cycle is graded on its arc progress alone.  The
  steps after their departure are recorded separately rather than silently
  dropped.
"""

from __future__ import annotations

from queue_control import _bendiness
from queue_geometry import STANDOFF_MAX_M, STANDOFF_MIN_M

# How much of the path must be turning for a sample to count toward the
# corner-cut measure.
BEND_THRESHOLD = 0.25


def cycle_rows(rollout, records: list[dict]) -> list[dict]:
    """One row per JOIN/ADVANCE cycle, carrying everything the gates need."""
    rows: list[dict] = []
    for index, cycle in enumerate(rollout.machine.cycles):
        if cycle.get("kind") not in ("advance", "to_counter"):
            continue
        cycle_records = [r for r in records
                         if r["cycle"] == index and r["state"] == "ADVANCE"]
        if not cycle_records:
            continue

        cross = [abs(r["duck_cross_track_m"]) for r in cycle_records]
        inside = [r["duck_cross_track_m"] for r in cycle_records
                  if _bendiness(r["duck_arc_m"]) > BEND_THRESHOLD]
        tracked = rollout.cycle_tracking.get(index, [])
        after_service = rollout.cycle_tracking_after_service.get(index, [])
        start_arc = rollout.cycle_start_arc.get(index)
        end_arc = cycle_records[-1]["duck_arc_m"]
        standoff = cycle_records[-1]["standoff_m"]
        gradeable = bool(cycle.get("kind") == "advance" and standoff is not None)

        rows.append({
            "index": index,
            "kind": cycle.get("kind"),
            "behind": cycle.get("behind"),
            "started_s": cycle.get("started_s"),
            "completed_s": cycle.get("completed_s"),
            "duration_s": cycle.get("duration_s"),
            "path_m": round(rollout.cycle_path.get(index, 0.0), 4),
            "arc_progress_m": (round(float(start_arc - end_arc), 4)
                               if start_arc is not None else None),
            "final_arc_m": round(float(end_arc), 4),
            "final_standoff_m": (round(float(standoff), 4)
                                 if standoff is not None else None),
            "standoff_gradeable": gradeable,
            "standoff_in_band": bool(
                gradeable and STANDOFF_MIN_M <= standoff <= STANDOFF_MAX_M),
            "max_cross_track_m": round(max(cross), 4) if cross else 0.0,
            "max_inside_cut_m": round(max(inside), 4) if inside else 0.0,
            "bend_samples": len(inside),
            "tracked_fraction": (round(sum(tracked) / len(tracked), 4)
                                 if tracked else None),
            "tracked_steps": len(tracked),
            "steps_after_subject_served": len(after_service),
            "command_peak": round(rollout.cycle_command_max.get(index, 0.0), 4),
        })
    return rows
