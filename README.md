# microduck

A dedicated workspace built on Pollen Robotics' Mini BDX simulator (`microduck` / `microduck_rl`).

The approach is simple: every behavior that genuinely works in simulation is frozen in
its own subfolder, and further improvements are made incrementally from that baseline.

> **Current follow behavior:**
> [`follow-me-among-others/`](follow-me-among-others/) — Microduck searches a
> moving crowd by shirt color and repeats `BUSCO → ENCUENTRO → SIGO → PARO` for
> blue, green, red and blue. The frozen avoidance baseline remains
> [`move-away-head-tracking/`](move-away-head-tracking/).

## Behaviors

| Folder | Status | Description |
|---|---|---|
| [`move-away/`](move-away/) | ✅ working | Frozen baseline: detects an approaching person, backs up, turns about 90°, and moves out of the way. |
| [`move-away-early-camera/`](move-away-early-camera/) | ✅ validated | Stable maneuver triggered at 1.15 m, with the duck's real camera shown in a 225×165 PiP; validated over 19 seconds with the person walking for 3 additional seconds. |
| [`move-away-head-tracking/`](move-away-head-tracking/) | 🏆 avoidance baseline | Extends the sequence to 22 seconds and keeps the person in view for all 1100/1100 control steps through an independent kinematic gaze layer. |
| [`follow-me/`](follow-me/) | ⚠️ comparison v1 | Original follow demo. Locomotion works, but it mirrors the leader's current pose and therefore cuts corners instead of following the same footsteps. |
| [`follow-me-footsteps/`](follow-me-footsteps/) | ✅ validated path queue | Uses a 0.65 m world-space trail, but retains the original ambiguous turn labels/directions for comparison. |
| [`follow-me-left-right-turns/`](follow-me-left-right-turns/) | ✅ validated turn base | Keeps the delayed footprint queue and replaces the old one-direction/strafe pair with true opposite curves: leader `+90°/−90°`, duck `+86.4°/−84.0°`. |
| [`follow-me-among-others/`](follow-me-among-others/) | 🏆 current follow | Five independently moving colored people; camera-gated acquisition and queued-footprint following in the exact sequence blue → green → red → blue. |

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
