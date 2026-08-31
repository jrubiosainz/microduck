# microduck

A dedicated workspace built on Pollen Robotics' Mini BDX simulator (`microduck` / `microduck_rl`).

The approach is simple: every behavior that genuinely works in simulation is frozen in
its own subfolder, and further improvements are made incrementally from that baseline.

Each folder holds the final validated implementation of one behavior — its scene,
scripts, stock policy, measured parameters and the video that was actually produced.

## Behaviors

| Folder | Status | Description |
|---|---|---|
| [`move-away/`](move-away/) | ✅ validated | Detects an approaching person, backs up, turns about 90° and clears the path, keeping the person in view for all 1100/1100 control steps through an independent kinematic gaze layer. |
| [`follow-me/`](follow-me/) | ✅ validated | Follows a leader along a delayed 0.65 m world-space footprint queue, with true opposite turns: leader `+90°/−90°`, duck `+86.4°/−84.0°`. |
| [`follow-me-among-others/`](follow-me-among-others/) | ✅ validated | Five independently moving colored people; camera-gated acquisition and queued-footprint following in the exact sequence blue → green → red → blue. |

`follow-me-among-others/` is built on top of [`follow-me/`](follow-me/), which is in
turn built on the gaze layer validated in [`move-away/`](move-away/).

## Conventions

- One behavior per subfolder. A validated behavior is never modified to test the
  next idea: copy it and iterate in the copy.
- Every subfolder includes a `README.md` with MEASURED parameters—not assumptions—
  and the validated video under `media/`.
- The ONNX policies are the stock policies from `microduck_rl`. Nothing is trained
  here yet; a behavior layer drives locomotion through velocity commands.
- Intermediate exploration folders are consolidated into the behavior they matured
  into once a version is final. The superseded steps remain reachable in the Git
  history rather than at the repository root.

## Upstream

- Simulator and policies: https://github.com/pollen-robotics/microduck_rl
- Robot firmware: https://github.com/pollen-robotics/microduck
