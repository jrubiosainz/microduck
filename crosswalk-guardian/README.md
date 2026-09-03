# crosswalk-guardian — stopping, looking both ways, and waiting for a real gap

A deterministic MuJoCo scenario in which Microduck behaves like a small robot
crossing an unsignalled two-lane street. It walks to the kerb, **stops short of
the wait line**, performs an explicit **left → right → left** visual scan from
its physical head-camera position, geometrically predicts when each road user
will occupy the crossing, **refuses every gap that is too tight**, and only then
commits — crossing continuously, without stopping in a live traffic lane, to the
safe zone on the far pavement:

```text
APPROACH_CURB → STOP → LOOK_LEFT → LOOK_RIGHT → LOOK_LEFT_AGAIN
              → WAIT_FOR_GAP → CROSSING → SAFE
```

This is a new behavior folder. It does not modify the validated
[`move-away/`](../move-away/), [`move-away-crowd/`](../move-away-crowd/),
[`follow-me/`](../follow-me/), [`follow-me-among-others/`](../follow-me-among-others/)
or [`come-here-recall/`](../come-here-recall/) baselines it was derived from.
The locomotion runtime and the camera isolation come from
[`come-here-recall/`](../come-here-recall/) and
[`move-away-crowd/`](../move-away-crowd/), which carry the corrected PR&nbsp;#22
sensor path; the geometric approach predictor is the same idea as
`move-away-crowd`'s closest-approach model, re-derived here as **1-D lane
occupancy intervals** because a crossing decision is about *time in a lane*, not
distance to a point.

## Demo

▶️ **[Watch or download the full MP4 (46 s, 960×640, 50 fps)](media/crosswalk-guardian.mp4)**

The wide shot follows the crossing and leans up the road toward whichever
vehicle is arriving next. The upper-right PiP is a stabilized view from the
duck's **physical head-camera position** — during each LOOK phase it shows the
road sector that phase is graded against, and the overlay reports the measured
fraction of that sector actually inside the frustum. The left panels are a plan
view of the street and a **conflict chart**: one bar per road user showing when
its body is predicted to occupy the pedestrian corridor, drawn behind the duck's
own predicted occupancy, so the overlap the duck refuses is visible directly
rather than inferred from a number.

## Scenario

- **A marked zebra crossing** with a dropped kerb, painted wait lines on both
  sides, a lane divider, and a green safe zone on the far pavement.
- **Seven road users** — four cars, a scooter and a bicycle, plus a late courier
  — on continuous scripted trajectories at 0.86–1.55 m/s, which is **3–5× the
  duck's measured walking speed**.
- **Two travel directions.** Right-hand drive, so the near lane runs −Y (traffic
  from the duck's **left**) and the far lane runs +Y (traffic from its
  **right**). That is what makes left → right → left the *correct* scan order
  rather than a decoration: the first lane the duck enters is the one whose
  traffic comes from the left, and the last check before stepping off re-checks
  that same lane.
- **Nobody stops, nobody yields, nobody teleports.** Every vehicle drives its
  lane at constant speed for the whole rollout. There is no branch anywhere that
  pauses traffic so the duck can cross.
- **Six vehicles cross in the first 24.5 s**, alternating lanes, and a seventh
  arrives at t≈40 s. The gap the duck takes is therefore **bounded at both
  ends** — without the late courier, "waited for a safe gap" would be
  indistinguishable from "waited for the traffic to end".

## What the crossing decision actually is

The predictor is an honest **simulator semantic/geometric proxy for vehicle
perception**. Each road user's lane, position, velocity and body size come from
the simulator. There is no detector, no tracker, no radar, and no
time-to-collision estimated from pixels anywhere in this behavior.

What **is** real is everything the decision is built on:

- **The camera geometry is real.** Each LOOK phase is graded against sample
  points on the *lane* that phase is about, tested through the exact frustum of
  the same camera the PiP renders from, with occlusion ray casts against actual
  scene geometry. A car blocking the view up the road counts as an occlusion.
- **The scan gate is real.** A phase advances only after its dwell time **and**
  after its sector has been continuously visible for 0.50 s. Turning the head is
  not looking.
- **The occupancy arithmetic is real**, and it is deliberately conservative in
  three independent ways (below).

Both the duck and every vehicle are reduced to **1-D occupancy intervals**, and
the question is whether they are disjoint by a margin:

- the **duck** occupies each lane over `[t_enter, t_exit]` relative to stepping
  off, with both ends inflated by its planar radius;
- a **vehicle** occupies the pedestrian corridor over `[t_in, t_out]`, inflated
  by its own half-length plus the duck's radius and a lateral buffer.

The judgement is **per lane**. A far-lane vehicle is irrelevant while the duck is
still in the near lane — and the predictor says so directly instead of treating
the road as one undivided block. That is both more realistic *and strictly
harder* to satisfy at the moment of commitment, because the duck must be clear
of the far lane **later**, when it is most exposed.

## Measured locomotion constants

Every constant was measured on **this scene**, with the stock
`alpha_walking.onnx` at action scale `0.9`, the real `imu_ang_vel` sensor and a
61-D observation.

**Forward gait onset is a cliff, not a ramp** (6 s rollouts):

| `vx` | displacement in 6 s | ground speed |
|---:|---:|---:|
| 0.16 | 0.008 m — no gait | — |
| 0.20 | 0.010 m — no gait | — |
| **0.24** | **0.516 m — walking** | 0.086 m/s |
| 0.52 | 1.516 m | 0.253 m/s |
| **0.58** | **1.794 m** | **0.299 m/s** |
| 0.65 | 2.188 m | 0.365 m/s |
| 0.75 | 2.730 m | 0.455 m/s |

A command below onset produces *no* motion, so a crossing cannot be walked
slowly — it is walked at speed or not at all. The controller never emits a `vx`
strictly between zero and 0.24.

**The policy drifts right at speed, and not by a little.** Measured yaw over 5–6 s
of straight-line walking: −18.0° at `vx=0.46`, −17.7° at 0.58, −23.4° at 0.60,
−24.5° at 0.70. An open-loop crossing curves tens of degrees off the zebra, so
**the crossing is flown closed-loop on heading**. Measured effect, from the same
start:

| | lateral drift | peak yaw excursion |
|---|---:|---:|
| open loop | **0.660 m** | 29.7° |
| closed loop | **0.204 m** | 19.1° |

**Yaw authority is strongly asymmetric** at `vx=0.58` over 5 s: `wz=+0.10` gives
+2.4 °/s while `wz=−0.10` gives **−8.7 °/s**; `wz=+0.18` gives +7.2 °/s against
−11.0 °/s. The right side is 3–5× stronger for the same magnitude, so the two
signs get independent gains and independent dead zones. Mirroring one onto the
other would make every left correction a violent over-correction.

**Stopping overshoots by 60 mm**, not by the 5–9 mm the sibling behaviors
measured after a slower cruise — this approach runs at `vx=0.52`. The release
point is therefore placed 60 mm short of the stop target and the residual
overshoot lands the duck on it.

## The crossing-duration estimate, and why one pessimism factor is wrong

`tools/measure_crossing.py` runs the **exact crossing primitive with the exact
heading controller** from the real kerb stop and times the true lane
occupancies. Measured: **0.2977 m/s**, near lane occupied `[1.16, 3.72] s`, far
lane `[2.88, 5.50] s`.

The first implementation divided that speed by a single pessimism factor of 1.30
and used the result for both ends of every interval. It produced near-lane
`[1.38, 4.63]` — the predicted **exit** was safely late, but the predicted
**entry was 0.10 s later than the duck actually entered**. A vehicle clearing the
lane in that window would have been judged clear while the duck was already in
it. **Stretching a schedule uniformly is not conservatism**: it moves both ends
the same way, and only one of them is the safe way.

The interval is now widened *outward* from both ends — entry computed at a speed
1.15× **faster** than measured, exit at 1.30× **slower** plus the measured 0.55 s
gait-onset dead time:

| lane | predicted | measured | contains |
|---|---|---|:--:|
| near | `[0.79, 5.27]` | `[1.16, 3.72]` | ✅ |
| far | `[2.39, 7.67]` | `[2.88, 5.50]` | ✅ |

The safety margin on top of that is **1.50 s** per road user, per lane, across
the whole estimated crossing.

## Validated 46-second rollout

46.0 s · 2300 control steps at 50 Hz · decimation 10 · stock
`alpha_walking.onnx` at action scale `0.9`. **No policy was trained for this
behavior.**

| Phase | Window | Evidence |
|---|---|---|
| `APPROACH_CURB` | 0.00–3.82 s | walks 1.10 m to the kerb |
| `STOP` | 3.82–5.14 s | rests at x = −1.052, leading edge −0.922 |
| `LOOK_LEFT` | 5.16–7.54 s | left sector visible **95.8 %** of steps, peak fraction 1.00 |
| `LOOK_RIGHT` | 7.56–9.96 s | right sector visible **74.4 %**, peak 1.00 |
| `LOOK_LEFT_AGAIN` | 9.98–11.88 s | left sector visible **74.0 %**, peak 1.00 |
| `WAIT_FOR_GAP` | 11.88–24.08 s | **12.20 s** waiting, 4 distinct gaps refused |
| `CROSSING` | 24.08–31.32 s | 7.24 s, path 2.271 m, net 2.177 m |
| `SAFE` | 31.32–46.00 s | rests at x = +1.180, inside the safe zone |

The sub-100 % scan figures are the honest head-slew ramp: the head takes ~0.4 s
to swing 148° between phases, and the sector is measured from the first tick of
the phase. Steady-state visibility in every phase is 0.75–1.00.

### The four refused gaps

| Refused | Worst margin | Limiting vehicle | Why |
|---|---:|---|---|
| 11.90–13.12 s | **−1.00 s** | bike (near) | would be in the near lane with the duck |
| 13.14–14.52 s | **−0.85 s** | scooter (far) | far lane occupied during the duck's far-lane window |
| 14.54–17.76 s | **−0.90 s** | van (far) | same, while taxi cleared by only +0.87 s |
| 17.78–23.68 s | **−1.11 s** | taxi (near) | near lane occupied at step-off |

Negative margins are **overlap durations**, not near misses: in each case the
vehicle's predicted corridor window intersects the duck's predicted lane window.

**Committed at t = 24.08 s** with a worst margin of **+10.48 s**, limited by
`courier` — a vehicle that had not yet arrived and would not reach the crossing
until t ≈ 40 s. Predicted crossing duration 8.12 s; actual **7.24 s**, comfortably
inside the estimate that authorised it.

### All nineteen acceptance gates pass

| Gate | Result |
|---|---|
| states occur exactly once, in order | `APPROACH_CURB→STOP→LOOK_LEFT→LOOK_RIGHT→LOOK_LEFT_AGAIN→WAIT_FOR_GAP→CROSSING→SAFE` |
| stops before the wait line | leading edge −0.922 m vs line −0.780 m, margin **+0.142 m** |
| zero encroachment before `CROSSING` | worst margin **+0.135 m** |
| all three scan phases, in order | 3/3 |
| required sector genuinely seen in the PiP camera | 3/3, peak fraction 1.00 each |
| at least one unsafe gap explicitly rejected | **4** distinct gaps |
| commit margin ≥ 1.50 s for every road user | **+10.48 s** |
| continuous crossing, no lane stop | longest zero-command run in road **0 steps**; worst 0.5 s advance in road **0.147 m** |
| crossing physically happened | path **2.271 m**, net **2.177 m** |
| reaches the opposite safe zone | final x **+1.180 m** |
| final state `SAFE` | yes |
| does not reverse after arriving | **0.0070 m** |
| vehicle contact | **0** steps |
| minimum geometric clearance | **+0.5531 m** |
| command exactly zero in `STOP`/`LOOK`/`WAIT`/`SAFE` | **0.000000** over 1849 steps |
| falls | **0** |
| minimum trunk height | **0.1100 m** (limit 0.09 m) |
| final trunk height | **0.1163 m** (nominal 0.116 m) |
| phase timeouts | **none** |

Exact values are in
[`media/crosswalk-guardian-metrics.json`](media/crosswalk-guardian-metrics.json).

## Defects found during validation

Five defects survived design review and were caught only by measurement.

1. **The duck's planar radius was 45 % too small.** `street.py` inherited
   `0.090 m` from a sibling behavior; that is a *torso* half-width, not the geom
   envelope. Measured on this scene: **0.1303 m**. Lane occupancy, wait-line
   encroachment and the crossing-duration estimate are all graded on the trunk
   centre inflated by this radius, so the stale value would have let the duck's
   real outline sit inside a lane while every occupancy gate called it clear. A
   test now pins the constant against `duck_planar_radius`.

2. **Uniform time-stretching made the prediction optimistic** on exactly the
   unsafe side (above). Replaced with a two-sided outward bracket, pinned by a
   test that reproduces the historical bug and requires the current form to
   avoid it.

3. **MuJoCo's mesh-vs-cylinder narrowphase returned exact zeros for pairs over a
   metre apart.** `mj_geomDistance` reported `0.000000` between a duck mesh and
   `sedan_wheel_fl` with the geom centres **1.0629 m** apart, and against
   `scooter_wheel_f`/`_r` at **1.1817 m** and **1.1602 m** — 7 of 2300 control
   steps claimed contact with nothing near. This is the same artifact
   `move-away-crowd` measured against boxes (65 spurious zeros in 264,000
   samples) and `come-here-recall` hit against a cap brim, now on a new geom
   type. It is also **state-dependent**: reconstructing the identical poses in a
   fresh `MjData` returns the correct distance and no spurious pair, so it cannot
   be reproduced outside the rollout or screened by pose. Every vehicle geom is
   now measured analytically — exact box, cylinder, capsule and sphere forms
   against each robot geom's bounding sphere, conservative for the mesh side, so
   it can only under-report clearance and can never hide a real contact.
   `mj_geomDistance` is not called at all on this scene.

4. **The traffic loop wrap was inside the predictor's horizon.** The wrap at
   |y| = 26 m looked comfortably off-screen, but the fastest vehicle covers that
   in **16.8 s** — inside the 22 s prediction horizon. A wrap would have appeared
   to the predictor as a road user materialising inside its reach, able to flip a
   gap decision between two consecutive ticks. The loop is now 42 m and a test
   pins the relation `LOOP_HALF_Y > max_speed × PREDICT_HORIZON_S` rather than the
   number. The longer loop then let the van catch the scooter in its own lane
   (**−0.433 m overlap**, i.e. one vehicle driving through another); the van is
   now slower than the vehicle ahead of it and the measured minimum same-lane gap
   is **+5.36 m**.

5. **Two presentation defects the preview caught.** The conflict chart was drawn
   straight onto the street, where the bright pavement and white zebra made every
   label unreadable — the same defect `come-here-recall` hit with its legend — and
   its bars were offset by a double-added origin. Separately, the crossing PiP
   aimed level along +x and framed a grey building wall during the most important
   phase of the behavior; the fix needed a **negative** pitch, because
   `view_pitch` feeds `forward.z = sin(view_pitch)` and the first attempt at +20°
   framed even more sky.

## Physical versus kinematic

This distinction matters for reading every number above.

- **Physical, authoritative, simulated:** the duck's floating base and 14
  actuated joints, driven by the stock ONNX policy through MuJoCo physics. Trunk
  height, path length, displacement, yaw and every stability figure come from
  this state and nothing else.
- **Kinematic, scripted, non-physical:** all seven road users. They are mocap
  bodies with `contype=0 conaffinity=0`, posed analytically each tick. They add
  **no degrees of freedom** to the robot (the model has exactly 7 free-joint qpos
  + 14 hinges), cannot push it, and never react to it. **A completed crossing can
  therefore never be the result of a vehicle nudging the duck**, and because they
  cannot collide, MuJoCo's own contact count is vacuous here — the honest gate is
  the geometric clearance, measured every tick and required to stay positive.
- **Rendering-only, isolated:** the head gaze and the stabilized camera rig. The
  head pose is applied to a **separate `MjData` copy** and never written back, so
  gaze cannot stabilize the robot — and equally, the robot's real head actuators
  are not exercised here. `test_gaze_never_touches_the_authoritative_walking_state`
  asserts `qpos`, `qvel` and `ctrl` are bit-identical across 120 camera updates,
  and a companion test proves the head genuinely moves in the isolated copy, so
  isolation is not achieved by doing nothing.

The machine also consumes the camera verdict **and the traffic state** from the
*previous* control tick. Measuring and deciding within one tick would let a scan
phase be satisfied, or a gap authorised, by a state that only exists after the
decision. One tick at 50 Hz is 20 ms, which is honest and is also what a real
perception pipeline would incur.

## Limitations

- **Vehicle perception is a semantic proxy, not sensing.** Lane, position,
  velocity and body size come from the simulator. No detector, no tracker, no
  radar, no vision-based time-to-collision. A physical robot would need
  multi-object detection, tracking and velocity estimation to replace this.
- **Road users are scripted and kinematic.** Constant speed, fixed lanes, no
  reaction to the duck, no physical interaction. A real driver might brake, or
  might not see a 25 cm robot at all.
- **The crossing is never re-decided once committed.** This is deliberate:
  stopping in a live traffic lane is the worst response to a surprise, and the
  commitment already covers the whole crossing with margin. An abort-and-retreat
  rule needs its own acceptance evidence and shipping it untested would weaken
  the gates this behavior exists to prove.
- **The scan is a fixed left → right → left script**, not an adaptive
  information-gathering policy. Which sectors matter is derived from the lane
  directions, but the order and dwell times are scripted.
- **The wrap is how a finite road models an endless stream.** Vehicles wrap at
  |y| = 42 m. That is a genuine discontinuity, placed far outside both the camera's
  useful range and the predictor's horizon; `max_visible_jump` measures the claim
  (**0.031 m**, exactly one tick of travel) rather than asserting it.
- **The PiP is informational.** The behavior's decisions are not taken from
  pixels — but the sector-visibility gate that authorises each scan phase *is*
  measured through that exact camera.
- **Simulation only.** No hardware validation of this behavior.

## Reproduce

Use the environment from a local `microduck_rl` checkout:

```bash
cd /path/to/microduck_rl
export MICRODUCK_RL_ROBOT_DIR=$PWD/src/mjlab_microduck/robot/microduck

# Regenerate the scene from its generator (optional; the XML is committed)
python /path/to/crosswalk-guardian/tools/build_scene.py

# Physics and acceptance gates, no rendering dependencies at all
uv run python /path/to/crosswalk-guardian/scripts/render_crosswalk_guardian.py \
  --no-render --seconds 46 --metrics /tmp/crosswalk-guardian-metrics.json

# Low-fps preview for visual inspection
uv run python /path/to/crosswalk-guardian/scripts/render_crosswalk_guardian.py \
  --seconds 46 --fps 4 --width 960 --height 640 \
  --out /tmp/preview-frames --metrics /tmp/preview-metrics.json

# Final 2300 frames at 50 fps
uv run python /path/to/crosswalk-guardian/scripts/render_crosswalk_guardian.py \
  --seconds 46 --fps 50 --width 960 --height 640 \
  --out /tmp/crosswalk-guardian-frames \
  --metrics media/crosswalk-guardian-metrics.json

ffmpeg -framerate 50 -i /tmp/crosswalk-guardian-frames/f%05d.png \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart \
  media/crosswalk-guardian.mp4

# Measurements behind the constants above
uv run python /path/to/crosswalk-guardian/tools/sweep_commands.py \
  --policy /path/to/crosswalk-guardian/onnx/alpha_walking.onnx --seconds 6
uv run python /path/to/crosswalk-guardian/tools/measure_crossing.py \
  --policy /path/to/crosswalk-guardian/onnx/alpha_walking.onnx

# Tests
uv run --with pytest python -m pytest /path/to/crosswalk-guardian/tests -q
```

## Contents

- `assets/scene_crosswalk_guardian.xml` — street, crossing, seven road users,
  cameras.
- `scripts/street.py` — the street's geometry as a single source of truth: lane
  edges, wait lines, safe zone, footprint occupancy, road sectors.
- `scripts/traffic.py` — the traffic schedule and its kinematics, plus the
  measurements that keep it honest (arrivals, visible jump, same-lane gap).
- `scripts/conflict.py` — occupancy intervals, the gap decision, and the measured
  locomotion constants.
- `scripts/guardian_model.py` — the eight-state machine and the two locomotion
  controllers.
- `scripts/guardian_camera.py` — isolated gaze, the PiP camera, sector visibility
  and occlusion.
- `scripts/rollout_guardian.py` — authoritative policy/physics integration.
- `scripts/contact_geometry.py` — exact duck-vs-vehicle surface distance and the
  MuJoCo narrowphase traps it has to survive.
- `scripts/guardian_metrics.py` — the nineteen acceptance gates.
- `scripts/render_frames.py`, `scripts/video_overlay.py` — video presentation.
- `scripts/render_crosswalk_guardian.py` — entry point.
- `tools/build_scene.py` — scene generator.
- `tools/sweep_commands.py` — command sweeps and gait-onset measurement.
- `tools/measure_crossing.py` — the crossing primitive, timed lane by lane.
- `tests/` — 98 tests: pure decision logic, real MuJoCo scene and camera
  invariants, and **21 synthetic counterexamples proving every gate can fail**.
- `onnx/alpha_walking.onnx` — byte-identical stock walking policy
  (`sha256:e36332d3…daa6c`, matching the other behaviors and the upstream
  checkout).
