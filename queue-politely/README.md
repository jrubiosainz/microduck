# queue-politely — reading a queue's order, and taking the last place in it

A deterministic MuJoCo scenario in which Microduck walks into a service hall,
finds a **hairpin rope-barrier queue of five adults**, works out the order of the
line by **projecting each person onto an explicit curved world-space queue
path**, identifies the true tail, **refuses two places it could physically have
taken**, joins behind the last person, and then advances one station at a time
as each person ahead is served — reaching the counter only after every one of
them has gone:

```text
APPROACH → OBSERVE_QUEUE → IDENTIFY_TAIL → EVALUATE_GAPS → JOIN
         → WAIT → ADVANCE → WAIT → ADVANCE → … → AT_COUNTER → DONE
```

`WAIT` and `ADVANCE` alternate once per service, so the cycle is a loop rather
than a one-shot script.

This is a new behavior folder. It does not modify the validated
[`move-away/`](../move-away/), [`move-away-crowd/`](../move-away-crowd/),
[`follow-me/`](../follow-me/), [`follow-me-among-others/`](../follow-me-among-others/),
[`come-here-recall/`](../come-here-recall/), [`crosswalk-guardian/`](../crosswalk-guardian/)
or [`narrow-corridor-etiquette/`](../narrow-corridor-etiquette/) baselines it was
derived from. The locomotion runtime and the analytic contact geometry come from
[`narrow-corridor-etiquette/`](../narrow-corridor-etiquette/), which carries the
corrected PR&nbsp;#22 sensor path.

## Demo

▶️ **[Watch or download the full MP4 (58 s, 960×640, 50 fps)](media/queue-politely.mp4)**

The wide shot looks down on the hall from high and to one side — an angle chosen
by measurement, not taste (see *Filming a hairpin*). The upper-right PiP is a
stabilized view from the duck's **physical head-camera position**, aimed at
whoever it is standing behind. The left panels show the inferred order with the
**two naive readings struck through**, a live scorecard of every candidate
standing place, and the running state. The plan view draws the real queue path,
the real barrier lines and every real footprint.

## The problem this behavior actually solves

A queue is **not** a set of people sorted by distance from the counter, and it is
**not** a set of people sorted by a coordinate. It is an *ordered occupancy of a
path*. The scene is built so that this distinction is arithmetic rather than
rhetorical: the queue folds through 180°, and on that geometry both naive
readings fail — and they fail *differently*.

MEASURED, on the five stations as the duck sees them at decision time:

| reading | order it produces | tail it names | |
|---|---|---|---|
| distance from counter | alvarez bianchi chandra **eriksson dubois** | `dubois` (4th) | ❌ |
| largest −x | alvarez **eriksson** bianchi dubois **chandra** | `chandra` (3rd) | ❌ |
| **arc length along the path** | alvarez bianchi chandra dubois eriksson | **`eriksson`** | ✅ |

| person | arc *s* | world (x, y) | ‖p‖ from counter | −x |
|---|---:|---|---:|---:|
| alvarez | 0.00 | (0.000, 0.000) | 0.000 | 0.000 |
| bianchi | 0.55 | (−0.549, −0.018) | 0.549 | 0.549 |
| chandra | 1.10 | (−0.960, −0.355) | 1.024 | 0.960 |
| dubois | 1.65 | (−0.960, −0.887) | 1.307 | 0.960 |
| **eriksson** | 2.55 | (−0.198, −1.240) | **1.256** | 0.198 |

The fold puts the true tail **nearer the counter** (1.256 m) than the person two
places ahead of it (1.307 m), and back at almost the same *x* as the head of the
queue. That is not a contrived edge case — it is what every folded queue in
every airport does.

The same projection settles membership, which a coordinate reading cannot
express at all: **two bystanders** stand near the line but not in it, at measured
off-path distances of **0.797 m** and **0.385 m** against a 0.30 m band. A
range-sorted reading would have interleaved both into the order.

The duck's own join station is at ‖p‖ = **1.298 m** — nearer the counter than
`dubois`, whom it is queueing behind. A distance-sorted reading would rank the
newly joined duck ahead of somebody it is genuinely behind.

## The refusals are the behavior

Being **wide enough** is a geometric fact. Being **yours** is a social one, and
this behavior is about the second. The two are kept in separate functions
(`gap_fits_duck` vs `classify_gap`) precisely so the scene can make them
disagree.

| candidate | kind | separation | duck fits? | verdict |
|---|---|---:|:---:|---|
| `beside_counter` | beside the person being served | 0.67 m of open floor | **yes** | ❌ REJECT |
| `alvarez–bianchi` | cut-in | 0.55 m | no | ❌ REJECT |
| `bianchi–chandra` | cut-in | 0.55 m | no | ❌ REJECT |
| `chandra–dubois` | cut-in | 0.55 m | no | ❌ REJECT |
| **`dubois–eriksson`** | cut-in (the straggler's hole) | **0.90 m** | **yes** | ❌ REJECT |
| **`behind_tail`** | behind the last person | 0.58 m | yes | ✅ **JOIN** |

Two of those refusals are of places the duck **could physically have stood in**,
and the gate counts only those. Refusing a gap too narrow to occupy would
demonstrate nothing.

The straggler's hole is the interesting one. `eriksson` stands 0.90 m behind
`dubois` instead of the nominal 0.55 m, which leaves **0.155 m of surface slack**
around the duck's footprint even with the adults measured at their **widest
point in the gait cycle**. It is a real, comfortable, tempting gap — and it
closes on the first advance, exactly as a real queue closes up.

## Measured locomotion constants

Every constant is measured on **this** scene with **this** model. Nothing is
inherited from a sibling behavior, because gait onset is a cliff and the axes
are not symmetric.

| quantity | measured |
|---|---|
| forward gait onset | **a cliff**: `vx=0.20` → 0.010 m in 6 s; `vx=0.22` → 0.409 m |
| approach / advance command | `vx=0.46`, 0.207 m/s |
| settle command | `vx=0.28`, above onset with margin |
| coast after **exact zero** | 0.020 m (`vx=0.46`), 0.011 m (`vx=0.38`) |
| turn radius, fold direction | `vx=0.34, wz=−0.42` → **R = 0.630 m** |
| turn asymmetry at `vx=0.34` | `wz=−0.18` → R = 1.119 m; `wz=+0.18` → R = **3.689 m** |
| duck planar radius (conservative) | 0.1303 m |
| adult half-extent (widest in gait) | 0.1647 m |

**The scene geometry follows the measurement, not the other way round.** The
fold radius is **0.62 m** because the policy was measured holding R = 0.630 m at
the advance speed. A 0.40 m fold would have demanded a turn rate the stock
walking policy does not deliver.

The turn asymmetry is why the queue folds in the **negative-`wz`** sense: that
is the strong direction, by nearly a factor of three at small commands.

## Results

| gate | measured |
|---|---|
| order inferred correctly | **2299 / 2299** sampled ticks |
| true tail identified | **2299 / 2299**; wrong locks **0** |
| bystanders excluded | both, every tick, on measured distance |
| invalid gaps refused **that fit** | **2** |
| join behind tail | longitudinal **0.590 m**, lateral **0.007 m** |
| WAIT→ADVANCE cycles | **3**, plus the final run to the counter |
| standoff at each stop | **0.598 / 0.594 / 0.595 m** (band 0.45–0.75) |
| advance path / arc progress | 0.738 / 0.727 m path, 0.554 / 0.550 m arc |
| max corner cut on the bend | **0.038 m** (limit 0.13) |
| max cross-track | 0.148 m (limit 0.20) |
| person ahead visible | **100 %** of every advance |
| counter reached | 50.16 s, after the last service at 46.0 s |
| min clearance to a person | **0.2155 m** |
| min clearance to scenery | **0.0978 m** |
| contacts | **0** |
| command in stationary states | **exactly 0.0** in all six |
| falls | **0**; min trunk z **0.1112 m**; final **0.1163 m** |

Exact values are in
[`media/queue-politely-metrics.json`](media/queue-politely-metrics.json).

## Defects found during validation

Six defects survived design review and were caught only by measurement.

1. **The arrival test was inverted, and it wrecked the whole behavior.** The duck
   travels toward arc length *zero*, so it approaches every target from above.
   The first version tested `duck_arc >= target_arc`, which is true from the
   instant the rollout starts — the duck begins near arc 4.5 and every target is
   smaller. It therefore "arrived" before taking a step, joined **1.948 m**
   behind the tail instead of 0.58 m, and dribbled forward in 0.17 m advances
   for the rest of the run: **31 cycles logged where there should have been
   five**. `join_band` and `advances_are_real` both failed, and both were this
   one comparison.

2. **The advance target was frozen at trigger time.** A queue advance is not a
   move to a station fixed when the trigger fired — the person in front is still
   walking. Freezing it left the duck stopping **1.149 m** behind the tail on
   four consecutive cycles against a 0.45–0.75 m band. The setpoint is now
   re-derived every tick from the predecessor's *current* arc, which is also the
   honest description of following somebody up a queue.

3. **The corner-cutting gate had its sign inverted**, so it graded *swinging
   wide* as cutting in. Measured on the path itself: displacing a point on the
   fold toward the arc's centre yields a **positive** cross-track. The gate now
   grades the positive sense, and only where the path actually bends —
   `test_positive_cross_track_is_inside_the_bend` pins the convention.

4. **Curvature-scaled lookahead made the tracking worse, not better.** Shortening
   the pure-pursuit lookahead on the bend (0.42 → 0.22 m) was an obvious-looking
   fix for corner cutting; measured, it produced over-steer and a *wider* peak
   excursion (0.133 → 0.155 m). The single longer lookahead tracks the fold
   better and was kept.

5. **The counter was built on top of the queue.** The first scene placed it at
   negative *x*, directly across the queue's own first leg; the scenery probe
   measured **−0.200 m** of overlap at arc 0.55 — the counter was standing where
   the second person in line stands. Separately, the barrier run initially
   enclosed the lane completely, so the duck would have had to pass *through* a
   rope to join at all; it now stops 0.60 m short, leaving an entrance.

6. **Three presentation defects the preview caught.** The wide shot left the duck
   underneath the HUD panels for **89 of 290** sampled frames; `tools/probe_camera.py`
   now projects the duck and five queue stations into pixel space for every
   candidate camera and scores the framing, and the chosen angle scores
   **1.000** clear-of-HUD with **1.000** queue visibility. The gap scorecard drew
   its "fits" and verdict columns at fixed offsets and they overlapped on every
   row; they are now laid out from measured text width. And the counter sign hung
   dead on the sightline from the head camera to the clerk, so the last seconds
   of PiP were a blank green board.

## Filming a hairpin

A folded queue is a hard thing to film: a low angle hides the fold behind the
people standing on it, and a top-down shot turns a 25 cm robot into a dot.
Rather than re-render and squint, `tools/probe_camera.py` projects the duck and
five queue stations into output-pixel coordinates for each candidate camera and
reports how often the duck is on screen *and* clear of the HUD regions. The
chosen shot (azimuth 38°, elevation −34°, distance 4.5 m) is the measured
optimum, not a preference.

## Physical versus kinematic

This distinction matters for reading every number above.

- **Physical, authoritative, simulated:** the duck's floating base and 14
  actuated joints, driven by the stock ONNX policy through MuJoCo physics. Trunk
  height, path length, arc progress, yaw and every stability figure come from
  this state and nothing else.
- **Kinematic, scripted, non-physical:** all eight people. They are mocap bodies
  with `contype=0 conaffinity=0`, posed analytically each tick. They add **no
  degrees of freedom** to the robot (the model has exactly 7 free-joint qpos + 14
  hinges + 4 hinges per person), cannot push it, and never react to it. **An
  advance can therefore never be the result of somebody nudging the duck
  forward**, and because they cannot collide, MuJoCo's own contact count is
  vacuous — the honest gate is the geometric clearance, measured every tick.
  `test_the_queue_never_waits_for_the_duck` parses the schedule module's AST with
  docstrings stripped and requires that the word "duck" appears nowhere in its
  executable code: the robot is not an input to the queue.
- **The barriers are non-colliding too, and that is deliberate.** If they
  collided, "the duck stayed in the lane and followed the bend" would be enforced
  by the contact solver rather than demonstrated by the controller, and a duck
  that scraped around the fold against a rope would still pass. With
  non-colliding barriers the lane is a constraint the *controller* respects, and
  the gate measures real surface distance to all 46 post, rope, counter, shelf
  and wall geoms every tick.
- **Rendering-only, isolated:** the head gaze and the stabilized camera rig. The
  head pose is applied to a **separate `MjData` copy** and never written back, so
  gaze cannot stabilize the robot.
  `test_gaze_never_touches_the_authoritative_walking_state` asserts `qpos`,
  `qvel` and `ctrl` are bit-identical across 120 camera updates, and a companion
  test proves the head genuinely moves in the isolated copy, so isolation is not
  achieved by doing nothing.

The machine also consumes the queue reading from the **previous** control tick.
Reading and deciding within one tick would let a join be authorised, or an
advance released, by a state that only exists after the decision. One tick at
50 Hz is 20 ms, which is honest and is what a real perception pipeline would
incur.

## Limitations

- **Person perception is a semantic proxy, not sensing.** Each person's world
  position comes from the simulator. There is no detector, no tracker and no RGB
  classification anywhere in this behavior. What is *derived* — and what the
  behavior actually is — is membership, order, tail and the verdict on every
  candidate place. A physical robot would need multi-person detection and
  tracking to replace the positions.
- **The queue path is known a priori.** The duck projects onto a path the scene
  generator and the decision layer share; it does not discover the queue's shape.
  A real robot would have to infer the lane from the barriers or from the
  people's own arrangement. The *projection and ordering* are real geometry; the
  *path* is given.
- **People are scripted and kinematic.** They advance on a fixed schedule, never
  react to the duck, and never contest a place. A real queue involves negotiation
  this behavior does not model.
- **Nobody joins behind the duck.** The order the duck must maintain only ever
  shrinks. A person arriving after it would not change any decision, but that
  case carries no acceptance evidence here.
- **The duck never reverses.** Every manoeuvre reduces arc length. There is no
  measured reverse primitive in this behavior.
- **The PiP is informational.** The behavior's decisions are not taken from
  pixels — but the visibility gate that grades each advance *is* measured through
  that exact camera.
- **Simulation only.** No hardware validation of this behavior.

## Reproduce

Use the environment from a local `microduck_rl` checkout:

```bash
cd /path/to/microduck_rl
export MICRODUCK_RL_ROBOT_DIR=$PWD/src/mjlab_microduck/robot/microduck

# Regenerate the scene from its generator (optional; the XML is committed)
python /path/to/queue-politely/tools/build_scene.py

# Physics and acceptance gates, no rendering dependencies at all
python /path/to/queue-politely/scripts/render_queue_politely.py \
  --no-render --seconds 58 --metrics /tmp/qp-metrics.json

# Low-fps preview for visual inspection
python /path/to/queue-politely/scripts/render_queue_politely.py \
  --seconds 58 --fps 2 --width 960 --height 640 \
  --out /tmp/qp-preview --metrics /tmp/qp-preview.json

# Final 2900 frames at 50 fps
python /path/to/queue-politely/scripts/render_queue_politely.py \
  --seconds 58 --fps 50 --width 960 --height 640 \
  --out /tmp/qp-final --metrics media/queue-politely-metrics.json

ffmpeg -framerate 50 -i /tmp/qp-final/f%05d.png \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart \
  media/queue-politely.mp4

# Measurements behind the constants above
python /path/to/queue-politely/tools/sweep_commands.py \
  --policy /path/to/queue-politely/onnx/alpha_walking.onnx --seconds 6
python /path/to/queue-politely/tools/measure_advance.py \
  --policy /path/to/queue-politely/onnx/alpha_walking.onnx
python /path/to/queue-politely/tools/probe_camera.py --records /tmp/qp-records.json

# Tests
python -m pytest /path/to/queue-politely/tests -q
```

## Contents

- `assets/scene_queue_politely.xml` — hall, counter, painted lane, rope
  barriers, eight people, cameras. Generated.
- `scripts/queue_path.py` — the curved world-space queue path, arc-length
  projection, and the two naive orderings reproduced on purpose.
- `scripts/queue_geometry.py` — scene layout, the duck's footprint, and the
  separation of "fits" from "allowed".
- `scripts/queue_people.py` — the eight people, the service schedule and the
  measurements that keep it honest.
- `scripts/queue_model.py` — membership, ordering, tail, and gap judgement.
  Pure; no MuJoCo.
- `scripts/queue_machine.py` — the nine-state machine and the pure-pursuit path
  controller, with every measured locomotion constant.
- `scripts/queue_camera.py` — isolated gaze, the PiP camera, and the visibility
  measurement.
- `scripts/rollout_queue.py` — authoritative policy/physics integration.
- `scripts/contact_geometry.py` — exact duck-vs-person and duck-vs-scenery
  surface distance, and the MuJoCo narrowphase traps it has to survive.
- `scripts/queue_metrics.py` — the twenty-four acceptance gates.
- `scripts/render_frames.py`, `scripts/video_overlay.py` — video presentation.
- `scripts/render_queue_politely.py` — entry point.
- `tools/build_scene.py` — scene generator.
- `tools/sweep_commands.py` — command sweeps, including the turn-radius table
  the fold's geometry was chosen from.
- `tools/measure_advance.py` — gait onset, coast after exact zero, turn radius.
- `tools/probe_camera.py` — measured camera framing.
- `tests/` — **86 tests**: pure decision logic, real MuJoCo scene and camera
  invariants, and **29 synthetic counterexamples proving every gate can fail**,
  guarded by a baseline fixture that must pass all of them.
- `onnx/alpha_walking.onnx` — byte-identical stock walking policy
  (`sha256:e36332d3…daa6c`, matching the other behaviors and the upstream
  checkout).
