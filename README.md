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
| [`move-away-crowd/`](move-away-crowd/) | ✅ validated | Scans eight independently moving adults, predicts the most urgent near-pass, and completes four contact-free evasions from different bearings; five adults carry boxes. |
| [`follow-me/`](follow-me/) | ✅ validated | Follows a leader along a delayed 0.65 m world-space footprint queue, with true opposite turns: leader `+90°/−90°`, duck `+86.4°/−84.0°`. |
| [`follow-me-among-others/`](follow-me-among-others/) | ✅ validated | Five independently moving colored people; camera-gated acquisition and queued-footprint following in the exact sequence blue → green → red → blue. |
| [`come-here-recall/`](come-here-recall/) | ✅ validated | Comes when called: five adults, three calling in turn from bearings 111.6° apart, camera-gated `LISTEN→SEARCH→CALLER_LOCK→APPROACH→ARRIVED` cycles stopping at a 0.485–0.508 m standoff, with an interrupting call refused. |
| [`crosswalk-guardian/`](crosswalk-guardian/) | ✅ validated | Crosses an unsignalled two-lane street: stops at the kerb, scans left→right→left through the real head camera, refuses four unsafe gaps against seven non-stopping road users, and crosses continuously on a +10.5 s margin. |
| [`narrow-corridor-etiquette/`](narrow-corridor-etiquette/) | ✅ validated | Steps aside in a 0.42 m corridor it cannot share: rejects a too-shallow and a crate-filled bay on measured clearance while both are reachable, parks fully out of the centre passage with the command at exactly zero, tracks each person past at 100 %, and rejoins for two complete cycles. |
| [`queue-politely/`](queue-politely/) | ✅ validated | Joins the correct end of a moving queue, preserves order and personal space, advances only when the person ahead moves, and refuses a queue-jumper without contact. |
| [`lost-child-find-person/`](lost-child-find-person/) | ✅ validated | Detects two real camera occlusions of Priya, stops, searches and rejects three distractors before reacquiring the same person and physically rejoining her. |
| [`walk-beside-me/`](walk-beside-me/) | ✅ validated | Joins Nadia on her free side, detects an obstructed lane, drops 1.15 m behind, crosses physically behind her and maintains the opposite side through three alternating turns. |
| [`lead-me-somewhere/`](lead-me-somewhere/) | ✅ validated | Selects the requested destination among three, leads along an eight-bend route, stops twice when Lila falls behind and resumes only after she catches up. |
| [`door-elevator-etiquette/`](door-elevator-etiquette/) | ✅ validated | Yields at a narrow doorway, waits beside an elevator, lets occupants exit, follows the guardian inside and remains behind her through the target-floor exit. |
| [`dynamic-slalom/`](dynamic-slalom/) | ✅ validated | Crosses a busy depot to a visible destination through 7 static and 7 moving bodies: predicts short-horizon occupancy and resolves five crossings **right, left, right, left, right**, waiting three times when neither corridor is safe. |
| [`patrol-and-investigate/`](patrol-and-investigate/) | ✅ validated | Patrols five checkpoints, investigates an unattended object and a restricted-zone intruder from safe standoffs, rejects a benign distractor and returns to the exact interrupted route. |
| [`gesture-response/`](gesture-response/) | ✅ validated | Accepts the instructor-only sequence COME→STOP→LEFT→RIGHT→BACK UP→WAVE, rejects an ambiguous partial gesture and ignores a sustained command from the wrong person. |
| [`protective-personal-space/`](protective-personal-space/) | ✅ validated | Maintains Aina's escort slot, resolves four alternating intrusions plus a simultaneous squeeze, retreats when Aina approaches directly, dismisses a false alarm and returns to escort with 27/27 gates and zero contacts. |

`follow-me-among-others/` is built on top of [`follow-me/`](follow-me/), which is in
turn built on the gaze layer validated in [`move-away/`](move-away/).
`come-here-recall/` is built on the corrected locomotion runtime, contact
measurement and camera isolation of [`move-away-crowd/`](move-away-crowd/).
`crosswalk-guardian/` and `narrow-corridor-etiquette/` both build on
`come-here-recall/`'s runtime; the corridor behavior additionally reuses the
analytic contact geometry that `crosswalk-guardian/` needed after MuJoCo's
mesh-versus-primitive narrowphase was measured returning false contacts.
`dynamic-slalom/` builds on `door-elevator-etiquette/`'s locomotion runtime,
contact geometry and camera isolation, re-measuring every locomotion constant on
its own scene — including the closed-loop lateral rate, which is the measurement
a behavior about choosing sides turns on.

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
