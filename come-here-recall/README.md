# come-here-recall — coming when called, to the right person

A deterministic MuJoCo scenario in which Microduck behaves like a small robot
being summoned across a room. Five adults stand around it, pacing slowly on
independent loops. One at a time, a different adult calls the duck over. The
duck listens, sweeps its head to find who is calling, locks onto that person,
walks to them, stops at a safe standoff distance facing them, holds, and goes
back to listening:

```text
LISTEN → SEARCH → CALLER_LOCK → APPROACH → ARRIVED → LISTEN
```

This is a new behavior folder. It does not modify the validated
[`move-away/`](../move-away/), [`move-away-crowd/`](../move-away-crowd/),
[`follow-me/`](../follow-me/) or
[`follow-me-among-others/`](../follow-me-among-others/) baselines it was
derived from. The locomotion runtime, contact measurement and camera isolation
come from [`move-away-crowd/`](../move-away-crowd/), which carries the
corrected PR&nbsp;#22 sensor path.

## Demo

▶️ **[Watch or download the full MP4 (54 s, 960×640, 50 fps)](media/come-here-recall.mp4)**

The wide view shows the whole room. The upper-right PiP is a stabilized view
taken from the duck's physical head-camera position; the yellow circle drawn on
it is the acquisition cone, **to scale** against the camera's real vertical
FOV, so the circle a viewer sees is the gate the lock is actually tested
against. The HUD reports the state, who is calling, whether the acquisition
gate is open, the exact velocity command, the standoff band, clearance and
trunk height. The bottom strip is the state pipeline; the bar below it shows
each call as a hollow bar where it sounds and each completed recall as a solid
bar — so the refused call is visibly a call that never became an approach.

## Scenario

- **Five adults:** blue, green, red, yellow and purple, each with a matching
  cap so identity survives the small PiP resolution and rear views.
- **Three of them call**, in order: **red → yellow → green**, from three widely
  separated bearings.
- **Blue and purple never call.** They exist so acquisition has to reject four
  wrong identities.
- **The active caller is unmistakable without the HUD**: they turn to face the
  duck, raise one arm overhead and wave it, stand inside a yellow floor ring,
  and carry a floating yellow beacon.
- Every adult paces its own slow ellipse (0.04–0.09 m/s) for the whole rollout.
  Nobody freezes, nobody teleports, and the caller keeps drifting while the
  duck walks to them, so the approach tracks a moving goal.
- Measured minimum separation between any two adults over the rollout is
  **1.464 m**, so they never interpenetrate. Pedestrian geometry does not
  collide with the robot in MuJoCo, so an arrival cannot succeed by physically
  bumping into anybody; clearance is measured geometrically instead.

## What a "call" actually is

A call is a scripted **event carrying a caller identity, an onset and a
duration**. It is an honest **simulator semantic proxy** for "an adult calls
the robot": there is no audio, no keyword spotter, no sound propagation and no
speaker localization anywhere in this behavior.

What **is** real is everything that happens after the call:

- The **camera geometry is real.** Visibility is measured through the exact
  frustum of the same stabilized camera drawn in the PiP, with occlusion ray
  casts against the actual scene geometry.
- The **acquisition gate is real.** A lock is permitted only while the caller
  is geometrically visible through that camera *and* within 12° of its optical
  axis, held continuously for 0.24 s. World-frame position alone can never open
  it — a caller standing 1 m away but outside the frustum is not acquired.
- The **search is real.** The duck cannot turn in place (measured below), and a
  caller can be behind it, so the only way to acquire them is to sweep the head
  through its ±170° yaw range. That is why `SEARCH` can hold the locomotion
  command at exactly zero and still be a genuine search.

A physical robot would need sound-source localization plus an image-based
person detector and re-identification to replace the simulator identities.

### Search is glimpse-then-fixate

The first implementation swept the head at a constant rate and required the
caller to sit inside the 12° cone for 0.24 s. It **locked nobody in a 46 s
rollout**, with the caller plainly visible the whole time: a sweep fast enough
to scan 300° in a few seconds crosses a 12° cone in about 0.20 s, which is less
than the confirmation window. Slowing the sweep enough to fix that alone would
have made every search take the better part of ten seconds.

Search is therefore two stages, which is also what an active-vision system
actually does:

1. **Sweep** until the caller falls anywhere inside the 74°×58° frustum — a
   genuine glimpse, measured with occlusion.
2. **Fixate** the glimpsed caller, bringing them toward the optical axis so the
   12° cone can be satisfied continuously.

The cone is then a statement about *how well centred* the caller is when the
lock commits, rather than a lottery on sweep phase. Measured off-axis angle at
lock: **1.2°, 1.5°, 1.0°**.

## Measured locomotion constants

Every constant was measured on **this scene**, with the stock
`alpha_walking.onnx` at action scale `0.9`, the real `imu_ang_vel` sensor and a
61-D observation. Nothing is inherited from the pre-PR-#22 corrupted-observation
numbers.

**Forward gait onset is a cliff, not a ramp** (6 s rollouts):

| `vx` | displacement in 6 s |
|---:|---:|
| 0.16 | 0.008 m — no gait |
| 0.20 | 0.010 m — no gait |
| **0.24** | **0.515 m — walking** |
| 0.28 | 0.640 m |
| 0.46 | 1.254 m |

A command below onset produces *no* motion, so "ease off as you arrive" cannot
be done by shrinking `vx` toward zero. It is done by stopping. The controller
never emits a `vx` strictly between zero and 0.24.

**Turning in place is impossible with this policy.** At `vx=0`, over *six*
seconds: `wz=+0.85` → **+7.8°**, `wz=−0.85` → **−9.5°**, `wz=±0.45` → +4.1°/−5.1°.
Yaw authority exists only while walking, so every heading change is flown as an
arc and the duck necessarily covers ground while turning.

**The policy is strongly asymmetric** (3 s windows, so no ±180° wrapping):

| `wz` at `vx=0.28` | yaw rate | | `wz` at `vx=0.24` | yaw rate |
|---:|---:|---|---:|---:|
| −0.85 | **−31.0 °/s** | | −0.25 | **−8.0 °/s** |
| +0.85 | +26.8 °/s | | +0.25 | +0.7 °/s |
| −0.60 | −23.7 °/s | | −0.45 | −14.0 °/s |
| +0.60 | +16.4 °/s | | +0.45 | +5.2 °/s |

Right turns are 1.2–11× faster for the same `|wz|`, so the two signs get
independent gains and independent dead zones. A shared dead zone would emit
left commands the policy cannot act on.

**Coast after stopping is negligible.** Cruising 4 s and then commanding exactly
zero, the trunk drifts **4.5 mm** (`vx=0.24`), **6.7 mm** (`vx=0.28`) and
**8.9 mm** (`vx=0.28, wz=−0.45`), and the drift is flat from +0.5 s to +2.5 s.
The standoff therefore needs no braking-distance term, only a 10 mm release
margin.

## The standoff band

The duck must stop **0.45–0.75 m** from the caller's centre, targeting 0.60 m.
That band is justified from the scene's own geometry, not chosen for looks:

| Quantity | Value |
|---|---|
| adult torso capsule radius | 0.078 m |
| duck planar half-extent (measured) | 0.0900 m |
| bodies would touch at | ≈ 0.17 m |
| **band minimum** | **0.45 m** → ≥ 0.28 m real clearance |
| **band maximum** | **0.75 m** → ≤ 0.58 m, still within arm's reach |

The band is 0.30 m wide, which comfortably absorbs both the measured ≤9 mm
coast and the caller's own pacing drift (±0.32 m amplitude at 0.04–0.09 m/s).

## Why the anchors were solved, not placed

Putting five people on a ring is the obvious layout and it is wrong here,
because the three approach ranges are **chained**: each recall starts from
wherever the previous one stopped. A ring produces wildly uneven approaches.

`tools/solve_anchors.py` walks the sequence forward instead — choose the next
call bearing, choose a range in the target band, place the caller there, advance
the duck to its standoff point — so every range and bearing is correct by
construction. Rejection sampling over five free anchors was tried first and
produced **zero** feasible layouts in 120,000 draws; the two non-callers are
placed by a greedy clearance sweep rather than at random, because random draws
violated the 1.20 m separation floor in 97% of candidates.

Solved result:

| Leg | Caller | Range | Call bearing | Turn required |
|---:|---|---:|---:|---:|
| 1 | red | 1.94 m | −1.8° | −1.8° |
| 2 | yellow | 1.89 m | −120.7° | −118.9° (right) |
| 3 | green | 2.23 m | +114.1° | −125.2° (right) |

Both large turns are to the measurably faster side. The first call comes from
nearly straight ahead on purpose, so the opening search is a short sweep and the
later calls are the ones that force a genuine rear search.

## Validated 54-second rollout

54.0 s · 2700 control steps at 50 Hz · decimation 10 · stock
`alpha_walking.onnx` at action scale `0.9`. **No policy was trained for this
behavior.**

| Cycle | Caller | Call at | Search | Lock at | Off-axis at lock | Approach | Path | Net | Range r₀ → min | Final standoff | Facing error | Visible during approach | Visible at arrival |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | red | 1.50 s | 2.84 s | 4.34 s | 1.2° | 6.26 s | 1.25 m | 1.03 m | 1.93 → 0.61 m | **0.485 m** | +23.2° | 100 % | 100 % |
| 2 | yellow | 15.50 s | 0.74 s | 16.24 s | 1.5° | 6.62 s | 1.21 m | 0.87 m | 1.59 → 0.61 m | **0.508 m** | −17.4° | 100 % | 100 % |
| 3 | green | 30.00 s | 4.52 s | 34.52 s | 1.0° | 12.86 s | 2.66 m | 2.04 m | 2.44 → 0.61 m | **0.501 m** | +9.4° | 100 % | 100 % |

Blue's call at **t = 22.0 s** landed while the duck was mid-approach to yellow.
It was **refused**, recorded, and never served.

State occupancy: `LISTEN` 322 steps, `SEARCH` 405, `CALLER_LOCK` 135,
`APPROACH` 1287, `ARRIVED` 551.

All nineteen acceptance gates pass:

| Gate | Result |
|---|---|
| ≥ 3 completed recalls | 3 |
| ≥ 3 distinct callers | red, yellow, green |
| callers served in the intended order | red → yellow → green |
| call bearings widely separated | minimum gap **111.6°** |
| wrong-caller locks | **0** |
| every lock happened inside the camera acquisition gate | 3/3, off-axis 1.0–1.5° |
| caller visible during APPROACH | **100 %** every cycle (limit 95 %) |
| caller visible at ARRIVED | **100 %** every cycle |
| every approach physically moved | path 1.21–2.66 m, net 0.87–2.04 m |
| every approach closed the range | reduction 0.98–1.83 m |
| final standoff inside 0.45–0.75 m | 0.485 / 0.508 / 0.501 m |
| trunk faces the caller at arrival | 9.4–23.2° (limit 30°) |
| locomotion command outside APPROACH | exactly **0.000000** over 1413 steps |
| caller identity changed mid-cycle | never |
| person contact | **0**; minimum geometric clearance **0.1603 m** |
| falls | **0** |
| minimum trunk height | **0.1112 m** (limit 0.09 m) |
| final trunk height | **0.1163 m** (nominal 0.116 m) |
| interrupting call refused | 1 (blue, t = 22.0 s) |

Exact values are in
[`media/come-here-recall-metrics.json`](media/come-here-recall-metrics.json).
The test suite has **80 passing tests**.

## Defects found during validation

Three integration defects survived a green unit suite and were caught only by
the full physical gate:

1. **The acquisition gate could never open.** The head swept at 120 °/s and the
   confirmation window was 0.24 s, so the caller spent only ~0.20 s inside the
   12° cone. Zero locks in 46 s, with the caller visible throughout. Fixed by
   glimpse-then-fixate (above), not by widening the cone — widening it would
   have weakened the very claim the gate exists to make.

2. **MuJoCo mesh-vs-box narrowphase reported a false contact.** At `t=28.02 s`
   `mj_geomDistance` returned exactly `0.00000` against yellow while the steps
   on either side measured 0.21–0.23 m. The offending pair was a duck mesh
   against `yellow_brim`, whose geom centres were **0.5526 m** apart. This is
   the same artifact `move-away-crowd/` measured (65 spurious zeros in 264,000
   samples). Box geoms are now handled analytically against each robot geom's
   bounding sphere: exact for the box, conservative for the mesh, so it can only
   under-report clearance and cannot hide a real contact.

3. **A still-sounding call re-served the same caller.** Call durations are
   generous enough that a call is still active when the duck arrives, so the
   machine immediately restarted the same recall: run 2 completed
   red → yellow → **yellow**, and the third "recall" lasted 0.02 s with zero
   path because the duck was already at the standoff distance. Calls are now
   keyed `(caller, start_s)` in a served ledger, so the same adult calling again
   *later* is still a genuinely new call.

## The no-steal rule

A call arriving while the duck is already committed does **not** take over.
Once `CALLER_LOCK` is entered, the active call is pinned until `ARRIVED`
completes; the interrupting call is recorded as refused.

This is deliberate for v1. An emergency-priority rule that can retarget
mid-approach needs its own acceptance evidence — which cycle was abandoned,
whether the abandoned caller was ever served, whether the abandoned approach
still counts as a recall — and shipping it untested would weaken exactly the
gates this behavior exists to prove. The scenario issues one interrupting call
so the refusal is *measured* rather than assumed, and
`test_mutation_stealing_lock_breaks_the_order` builds a stealing machine and
requires it to behave differently, so the rule is not vacuous.

## Physical versus kinematic

This distinction matters for reading every number above.

- **Physical, authoritative, simulated:** the duck's floating base and 14
  actuated joints, driven by the stock ONNX policy through MuJoCo physics.
  Trunk height, path length, displacement, yaw and every stability figure come
  from this state and nothing else.
- **Kinematic, scripted, non-physical:** all five adults. They are mocap bodies
  with `contype=0 conaffinity=0`, posed analytically each tick. They add no
  degrees of freedom to the robot, cannot push it, and never react to it.
- **Rendering-only, isolated:** the head gaze and the stabilized camera rig.
  The head pose is applied to a **separate `MjData` copy**; it is never written
  back into the locomotion state. Gaze therefore cannot stabilize the robot,
  and equally the robot's real head actuators are not exercised here.
  `test_gaze_never_touches_the_authoritative_state` asserts the physical `qpos`
  is bit-identical across a camera update.

The machine also consumes the camera verdict from the **previous** control
tick. Measuring and deciding within one tick would let a lock be justified by a
camera pose that only exists after the decision. One tick at 50 Hz is 20 ms,
which is honest and is also what a real perception pipeline would incur.

## Limitations

- **Calling is a semantic proxy, not audio.** No sound, no localization, no
  speech. Identity and world pose come from the simulator.
- **Person recognition is a semantic proxy, not RGB classification.** The
  frustum, occlusion and acquisition geometry are real; the identity behind
  them is a simulator lookup.
- **Adults are scripted and kinematic.** Fixed loops, no reaction to the duck,
  no physical interaction.
- **No emergency interruption.** By design for v1; see above.
- **Facing is graded on the trunk at arrival**, and the measured errors
  (9.4–23.2°) reflect that the policy cannot turn in place — the final heading
  is whatever the closing arc produced.
- **Simulation only.** No hardware validation of this behavior.
- The PiP is informational; the behavior's decisions are not taken from pixels.

## Reproduce

Use the environment from a local `microduck_rl` checkout:

```bash
cd /path/to/microduck_rl
export MICRODUCK_RL_ROBOT_DIR=$PWD/src/mjlab_microduck/robot/microduck

# Regenerate the scene from its generator (optional; the XML is committed)
python /path/to/come-here-recall/tools/build_scene.py

# Physics and acceptance gates, no rendering dependencies at all
uv run python /path/to/come-here-recall/scripts/render_come_here_recall.py \
  --no-render --seconds 54 --metrics /tmp/come-here-recall-metrics.json

# Low-fps preview for visual inspection
uv run python /path/to/come-here-recall/scripts/render_come_here_recall.py \
  --seconds 54 --fps 4 --width 960 --height 640 \
  --out /tmp/preview-frames --metrics /tmp/preview-metrics.json

# Final 2700 frames at 50 fps
uv run python /path/to/come-here-recall/scripts/render_come_here_recall.py \
  --seconds 54 --fps 50 --width 960 --height 640 \
  --out /tmp/come-here-recall-frames \
  --metrics media/come-here-recall-metrics.json

ffmpeg -framerate 50 -i /tmp/come-here-recall-frames/f%05d.png \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart \
  media/come-here-recall.mp4

# Measurements behind the constants above
uv run python /path/to/come-here-recall/tools/sweep_commands.py \
  --policy /path/to/come-here-recall/onnx/alpha_walking.onnx --seconds 6
uv run python /path/to/come-here-recall/tools/measure_approach.py \
  --policy /path/to/come-here-recall/onnx/alpha_walking.onnx

# Tests
uv run python -m pytest /path/to/come-here-recall/tests -q
```

## Contents

- `assets/scene_come_here_recall.xml` — robot, five adults, markers, cameras.
- `scripts/policy_runtime.py` — scene loading, 61-D observation, ONNX rollout.
- `scripts/people_routes.py` — pacing loops, caller wave and gait animation.
- `scripts/recall_model.py` — call events, recall state machine, approach
  controller, measured gait constants.
- `scripts/attention_camera.py` — isolated gaze, search sweep, frustum,
  occlusion and the acquisition gate.
- `scripts/rollout_recall.py` — authoritative policy/physics integration and
  per-cycle evidence.
- `scripts/contact_geometry.py` — exact duck-vs-person surface distance and the
  two MuJoCo narrowphase traps it has to survive.
- `scripts/recall_metrics.py` — the nineteen acceptance gates.
- `scripts/render_frames.py`, `scripts/video_overlay.py` — video presentation.
- `scripts/render_come_here_recall.py` — entry point and the call script.
- `tools/build_scene.py` — scene generator.
- `tools/sweep_commands.py`, `tools/measure_approach.py` — command sweeps,
  gait-onset, coast and yaw-rate measurement.
- `tools/solve_anchors.py` — constructive anchor solver.
- `tests/` — pure state/control tests plus real MuJoCo scene and gate tests,
  including mutation counterexamples for every meaningful gate.
- `onnx/alpha_walking.onnx` — byte-identical stock walking policy
  (`sha256:e36332d3…daa6c`, matching the other behaviors and the upstream
  checkout).
