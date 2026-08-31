# microduck

A dedicated workspace built on Pollen Robotics' Mini BDX simulator (`microduck` / `microduck_rl`).

The approach is simple: every behavior that genuinely works in simulation is frozen in
its own subfolder, and further improvements are made incrementally from that baseline.

> **Latest behavior:** [`follow-me/`](follow-me/) — a person leads a complete
> forward/left/stop/right/backward route while Microduck follows and keeps them
> visible. The frozen avoidance baseline remains
> [`move-away-head-tracking/`](move-away-head-tracking/).

## Behaviors

| Folder | Status | Description |
|---|---|---|
| [`move-away/`](move-away/) | ✅ working | Frozen baseline: detects an approaching person, backs up, turns about 90°, and moves out of the way. |
| [`move-away-early-camera/`](move-away-early-camera/) | ✅ validated | Stable maneuver triggered at 1.15 m, with the duck's real camera shown in a 225×165 PiP; validated over 19 seconds with the person walking for 3 additional seconds. |
| [`move-away-head-tracking/`](move-away-head-tracking/) | 🏆 avoidance baseline | Extends the sequence to 22 seconds and keeps the person in view for all 1100/1100 control steps through an independent kinematic gaze layer. |
| [`follow-me/`](follow-me/) | ✅ validated | Follows an animated person through forward, left-turn, stop, screen-right and backward phases; includes measured tracking metrics and a stabilized head-camera PiP. |

## Conventions

- One behavior per subfolder. A validated behavior is never modified to test the
  next idea: copy it and iterate in the copy.
- Every subfolder includes a `README.md` with MEASURED parameters—not assumptions—
  and the validated video under `media/`.
- The ONNX policies are the stock policies from `microduck_rl`. Nothing is trained
  here yet; a behavior layer drives locomotion through velocity commands.

## Upstream

- Simulator and policies: https://github.com/pollen-robotics/microduck_rl
- Robot firmware: https://github.com/pollen-robotics/microduck
