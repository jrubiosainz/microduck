# narrow-corridor-etiquette — stepping aside in a corridor too tight to share

A deterministic MuJoCo scenario in which Microduck walks down a **narrow indoor
corridor**, works out that it cannot pass an approaching adult side by side,
picks a side recess that is actually usable, **steps out of the way and stops
completely**, watches the person walk past, waits until they have genuinely
cleared, returns to the centreline and carries on to its destination — twice,
for two different people, on opposite walls:

```text
CRUISE → DETECT → SELECT_ALCOVE → PULL_OVER → YIELD
       → CLEAR → REJOIN → RESUME → DONE
```

`RESUME` re-enters `DETECT` when the next person appears, so the cycle is a
loop rather than a one-shot script.

This is a new behavior folder. It does not modify the validated
[`move-away/`](../move-away/), [`move-away-crowd/`](../move-away-crowd/),
[`follow-me/`](../follow-me/), [`follow-me-among-others/`](../follow-me-among-others/),
[`come-here-recall/`](../come-here-recall/) or
[`crosswalk-guardian/`](../crosswalk-guardian/) baselines it was derived from.
The locomotion runtime, the camera isolation and the analytic contact geometry
come from [`crosswalk-guardian/`](../crosswalk-guardian/), which carries the
corrected PR&nbsp;#22 sensor path.

## Demo

▶️ **[Watch or download the full MP4 (50 s, 960×640, 50 fps)](media/narrow-corridor-etiquette.mp4)**

The wide shot looks down the corridor from high and behind — the only angle
that shows both walls, both alcove mouths and a 25 cm robot at the same time.
The upper-right PiP is a stabilized view from the duck's **physical head-camera
position**; during a yield it is aimed at the passing adult, and the overlay
reports the measured fraction of that person's body actually inside the
frustum. The left panels are a plan view of the corridor — with the duck drawn
as its real footprint against the real wall lines — and a **live scorecard of
every candidate bay**, so the two the duck refuses are visible as refusals
rather than inferred from a number.

## Scenario

- **A 0.42 m corridor**, 5.6 m long, with plain walls, doors, and a painted
  centreline. The duck starts at one end and has a marked destination at the
  other.
- **Four side recesses**, two of them useless for reasons a viewer can see:

  | Bay | Depth | Usable depth | Trunk can reach | Verdict |
  |---|---:|---:|---:|---|
  | `bay_shallow` | 0.10 m | 0.10 m | 0.180 m | ❌ too shallow |
  | `bay_crates` | 0.38 m | **0.13 m** (crates) | 0.210 m | ❌ obstructed |
  | `bay_open` | 0.38 m | 0.38 m | 0.460 m | ✅ usable |
  | `bay_far` | 0.38 m | 0.38 m | 0.460 m | ✅ usable |

  The duck needs its trunk centre at **0.334 m** off the centreline before its
  whole footprint is out of the way, so the first two cannot work however
  patiently it tries.
- **Two adults**, walking the corridor head-on at 0.42 m/s — **three times the
  duck's measured 0.140 m/s** — one after the other, meeting the duck at
  different points and requiring recesses on **opposite walls**.
- **Nobody stops, nobody steps aside, nobody teleports.** Each adult holds a
  constant velocity and a fixed lateral offset within 0.02 m of the centreline
  for the whole of its pass. There is no branch anywhere that pauses a person
  so the duck can get out of the way, and `test_an_adult_never_yields_to_the_duck`
  pins that by sampling the velocity across the whole traverse.

## Why the duck has to move at all

This is the premise of the behavior, and it is **arithmetic rather than art
direction**. Two bodies of lateral half-widths `h₁` and `h₂` confined to a strip
of half-width `W`, neither of them turning, can be separated by at most
`2W − h₁ − h₂` between their centres. They fit safely, with a clear surface gap
`g`, only when that exceeds `h₁ + h₂ + g`.

Measured on this scene:

| | value |
|---|---:|
| corridor width | 0.420 m |
| duck lateral half-width (measured, exact) | 0.0705 m |
| adult lateral half-width (measured, exact) | 0.1040 m |
| best possible surface gap, both hugging their walls | **0.0710 m** |
| gap required to count as a safe pass | 0.1000 m |
| **shortfall** | **0.0290 m** |

So a side-by-side pass is possible only in the sense that the two bodies would
not quite overlap — it is not a pass anybody would take. And in the *plain*
corridor the duck's trunk centre can reach only **0.0797 m** off the centreline
against the 0.334 m it needs, so **there is nowhere outside a recess where it
can get out of the way at all**. Pulling over is not an optimisation here; it is
the only thing that works.

Both figures are computed with each body's **exact lateral half-width**, not the
conservative bounding-sphere radius used by every other gate. Bounding spheres
badly over-state a body that is long in x and narrow in y, and over-stating the
bodies here would make the corridor look narrower than it is — the one direction
in which conservatism would flatter this behavior instead of testing it. Every
*other* gate uses the conservative radius, where an over-wide robot makes the
gate harder.

The 0.10 m safe gap is itself measured, not chosen: the duck's own peak lateral
excursion while trying to hold the centreline is **0.0634 m** over a 12 s
closed-loop cruise. A nominal gap smaller than the robot's own tracking error
can be closed by tracking error alone, so it is not a gap.

## What the decision actually is

The predictor is an honest **simulator semantic/geometric proxy for pedestrian
perception**. Each adult's position, velocity and body size come from the
simulator. There is no detector, no tracker, and no time-to-collision estimated
from pixels anywhere in this behavior.

What **is** real is everything else:

- **The camera geometry is real.** Tracking is graded against sample points at
  the adult's knee, chest and head height, tested through the exact frustum of
  the same camera the PiP renders from, with occlusion ray casts against actual
  scene geometry — including the alcove's own cheek, which is a genuine wall.
- **The counterfactual is real, and it is recorded before the duck acts.** At
  the moment of detection the behavior records the surface clearance the pass
  *would* have had if the duck had simply kept walking: **−0.117 m** and
  **−0.166 m** for the two encounters. Negative means the two bodies would have
  overlapped. Reconstructing that number later, from a duck already tucked into
  a recess, would report the clearance of the manoeuvre rather than the
  clearance of doing nothing — which is the only thing that justifies acting.
- **The alcove arithmetic is real**, and each candidate must satisfy two
  independent requirements:
  - **physical clearance** — the whole footprint must fit inside the recess
    *and* out of the centre passage, computed from the recess's **usable** depth
    so an obstruction fails on geometry rather than by name;
  - **reachability** — the duck must arrive, settle and be stationary at least
    0.80 s before the adult's body reaches the mouth.

### The rejections are about judgement, not distance

At the first decision the duck sits at x = −1.09 and **all four bays are ahead
of it**. Both unusable bays are comfortably *reachable* — more comfortably than
the one it picks:

| Bay | Clears passage | Reachable | Travel | Available | Margin | Verdict |
|---|:--:|:--:|---:|---:|---:|---|
| `bay_shallow` | ❌ | ✅ | 3.26 s | 10.09 s | +6.83 s | refused on **clearance** |
| `bay_crates` | ❌ | ✅ | 3.41 s | 9.71 s | +6.30 s | refused on **clearance** |
| `bay_open` | ✅ | ✅ | 6.56 s | 8.09 s | **+1.53 s** | **selected** |
| `bay_far` | ✅ | ❌ | 19.56 s | 5.11 s | −14.44 s | out of reach |

The duck walks *past* two bays it could easily reach, because neither of them
would get its body out of the way. That is the claim this behavior exists to
make, and the layout is arranged specifically so it cannot be won by an alcove
happening to be behind the robot.

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
| 0.36 | 0.828 m | 0.138 m/s |
| 0.52 | 1.515 m | 0.253 m/s |

**Lateral gait onset is also a cliff, and it is asymmetric:**

| `vy` | sideways travel in 4–6 s | |
|---:|---:|---|
| +0.20 / −0.20 | 0.003 m / 0.002 m | no gait |
| +0.28 / −0.24 | 0.005 m / 0.000 m | no gait |
| **+0.30** | **0.186 m** | walking |
| **−0.26** | **0.149 m** | walking |

The two signs cross onset at different magnitudes — about −0.26 to the right and
+0.30 to the left — so a single symmetric threshold would either stall one
direction or over-drive the other.

**The lateral command is strongly yaw-coupled, and only on one side.** Measured
over 4 s of pure lateral command:

| command | sideways travel | yaw change |
|---|---:|---:|
| `vy=+0.46, wz=0` | +0.476 m | −1.3° |
| `vy=−0.46, wz=0` | −0.402 m | **+93.6°** |
| `vy=−0.46, wz=−0.45` | −0.388 m | −5.8° |
| `vy=−0.60, wz=−0.45` | −0.522 m | +5.3° |

Stepping **right** with no yaw command spins the duck through ninety degrees in
four seconds; stepping **left** barely rotates it at all. In a 0.42 m corridor a
ninety-degree spin puts the robot's nose into a wall, so the right-hand step
carries a large feed-forward yaw term (`wz=−0.45`) and the left-hand step almost
none (`wz=−0.05`). The two signs never share a gain.

## The manoeuvre model, and the mistake it replaced

`tools/measure_pullover.py` times the **exact `PULL_OVER` controller** entering
each usable bay from a lead distance short of its mouth, to a parked, stationary
duck:

| lead | `bay_open` parked | ratio to longer leg | `bay_far` parked | ratio |
|---:|---:|---:|---:|---:|
| 0.10 m | 2.52 s | 0.88 | 3.68 s | **1.28** |
| 0.30 m | 2.50 s | 0.87 | 3.18 s | 1.11 |
| 0.60 m | 4.60 s | 0.96 | 5.20 s | 1.08 |
| 0.90 m | 6.56 s | 0.91 | 7.42 s | 1.03 |
| 1.30 m | 9.44 s | 0.91 | 10.26 s | 0.99 |
| 1.80 m | 13.26 s | 0.92 | 13.94 s | 0.97 |
| 2.40 m | 17.56 s | 0.91 | 18.22 s | 0.95 |

The first implementation charged the forward leg and the lateral leg **one after
the other**. That describes a manoeuvre the controller never performs — it drives
both axes together — and it over-estimated every candidate by several seconds,
refusing `bay_open` outright and leaving the duck with nowhere to go. The
right-hand entry into `bay_far` is the expensive one, exactly as the command
sweep predicted: that side spends part of its budget fighting the yaw coupling.

The estimate is now the **longer** of the two legs scaled by 1.30, which exceeds
every one of the fourteen measured ratios, plus the measured gait-onset dead time
and settle. Two tests pin it: one requires the factor to clear the worst measured
ratio, the other reproduces the historical sequential model and shows it refusing
a bay the duck demonstrably reaches.

## Validated 50-second rollout

50.0 s · 2500 control steps at 50 Hz · decimation 10 · stock
`alpha_walking.onnx` at action scale `0.9`. **No policy was trained for this
behavior.**

| Phase | Window | Evidence |
|---|---|---|
| `CRUISE` | 0.00–5.60 s | walks 0.766 m down the centreline |
| `DETECT` → `SELECT_ALCOVE` | 5.60–6.72 s | keeps walking while deciding |
| `PULL_OVER` | 6.72–12.82 s | 1.097 m path, 0.299 m lateral, into `bay_open` |
| `YIELD` | 12.82–17.14 s | 216 steps, command **exactly 0.000000** |
| `CLEAR` → `REJOIN` | 17.14–20.06 s | returns to y = −0.095 m |
| `RESUME` | 20.06–25.32 s | walks on |
| `DETECT` → `PULL_OVER` | 25.32–30.84 s | 0.798 m path, 0.361 m lateral, into `bay_far` |
| `YIELD` | 30.84–37.56 s | 336 steps, command **exactly 0.000000** |
| `CLEAR` → `REJOIN` | 37.56–40.02 s | returns to y = +0.096 m |
| `RESUME` → `DONE` | 40.02–50.00 s | reaches the destination at x = +1.605 m |

### The two etiquette cycles

| | cycle 1 | cycle 2 |
|---|---|---|
| person | `chen` | `diaz` |
| detected at | 5.60 s, range 4.80 m | 25.32 s, range 4.79 m |
| nearest body at detection | 1.500 m | 1.500 m |
| counterfactual clearance | **−0.117 m** | **−0.166 m** |
| bays scored | 4 | 4 |
| refused on clearance | `bay_shallow`, `bay_crates` | — |
| selected | `bay_open` (−Y wall) | `bay_far` (+Y wall) |
| commit margin | +2.02 s | +4.29 s |
| pull-over | 6.10 s, 1.097 m path, 0.299 m lateral | 4.42 s, 0.798 m path, 0.361 m lateral |
| parked at | (−0.323, −0.342) | (+0.881, +0.347) |
| passage intrusion while parked | **−0.014 m** (clear) | **−0.044 m** (clear) |
| yield | 4.32 s, cmd max **0.000000** | 6.72 s, cmd max **0.000000** |
| adult offsets during yield | +1.129 m → −0.675 m | +2.159 m → −0.647 m |
| tracked in the PiP | **100 %** in sightline | **100 %** in sightline |
| released at range | 0.606 m | 0.610 m |
| rejoin | 2.42 s, 0.252 m lateral → y = −0.095 m | 1.96 s, 0.273 m lateral → y = +0.096 m |

The adult offsets crossing from positive to negative is what makes "the person
walked past" evidence rather than narration: the signed gap along the corridor
changes sign, which cannot happen unless they went by.

### All twenty-five acceptance gates pass

| Gate | Result |
|---|---|
| states occur in order, ending in `DONE` | `CRUISE→(DETECT→SELECT→PULL_OVER→YIELD→CLEAR→REJOIN→RESUME)×2→DONE` |
| real forward path before the first encounter | **0.766 m** |
| detects before unsafe proximity | nearest body **1.500 m** at both detections (limit 0.70 m) |
| counterfactual clearance recorded, and unsafe | **−0.117 m**, **−0.166 m** |
| at least two alcoves evaluated | **4** per decision |
| an alcove refused for physical clearance while reachable | **2** (`bay_shallow`, `bay_crates`) |
| every selection satisfies reachability **and** clearance | 2/2 |
| pull-over produced real path and lateral displacement | 1.097 / 0.798 m path, 0.299 / 0.361 m lateral |
| whole footprint left the centre passage | intrusion **−0.014 m**, **−0.044 m** |
| command exactly zero throughout `YIELD` | **0.000000** over 552 steps |
| adult passed completely during the yield | 2/2, offsets change sign |
| no early rejoin | released at 0.606 m / 0.610 m (limit 0.55 m) |
| rejoin returns near the centreline | y = **−0.095 m**, **+0.096 m** (limit 0.10 m) |
| real forward progress after the last rejoin | **0.698 m** |
| reaches the destination | final x **+1.607 m** (threshold +1.60 m) |
| minimum duck/adult clearance | **+0.1738 m** |
| minimum duck/wall clearance | **+0.0157 m** (`bay_far_cheek_hi`, 21 geoms) |
| contacts | **0** |
| adult tracked in the PiP during yield | **100 %** of in-sightline steps, both cycles |
| decorative commands below gait onset | **0** |
| command exactly zero in every stationary state | **0.000000** over 851 steps |
| falls | **0** |
| minimum trunk height | **0.1095 m** (limit 0.09 m) |
| final trunk height | **0.1163 m** (nominal 0.116 m) |
| phase timeouts | **none** |

Exact values are in
[`media/narrow-corridor-etiquette-metrics.json`](media/narrow-corridor-etiquette-metrics.json).

## Why tracking is graded over a sightline window

During a yield the duck is at the back of a recess looking out through its
mouth, and the two cheeks are opaque walls. By similar triangles a sightline
from a park point 0.397 m off-centre through a 0.80 m mouth reaches
**±0.849 m** of corridor at the centreline. Beyond that the adult is *behind a
wall*, and no amount of neck travel changes it — the duck's head yaw spans
±170°, and it still cannot see through masonry.

The gate therefore requires ≥95 % tracking over the steps in which the person is
inside that window, and the measured figure is **100 % for both cycles**. The
raw whole-yield figures (92.6 % and 59.8 %) are recorded alongside, and the
difference is entirely the tail of each yield in which the duck is waiting for a
person it can no longer see. Demanding sight through a wall would be a gate no
robot could pass.

## Defects found during validation

Five defects survived design review and were caught only by measurement.

1. **The pull-over drove the robot into a wall.** The first controller began its
   sideways step as soon as `PULL_OVER` began, regardless of whether the duck had
   reached the recess. It reached its park `y` before it had walked far enough
   along the corridor to be inside the bay at all, and the wall gate measured
   **−0.109 m** of overlap against `bay_open_cheek_lo`. The lateral step now
   begins only once the whole footprint is between the two cheeks, and the
   completion test requires the footprint to be inside the mouth as well as out
   of the passage.

2. **Parking at the mouth's edge failed the same gate more subtly.** With the
   entry fixed, the duck drove only to `entry_x` — 5 cm inside the near cheek —
   and the settle drift after the lateral command was released pushed its
   bounding footprint back across the boundary, holding **−0.0235 m** against
   `bay_far_cheek_lo` for an entire yield. The park station is now the middle of
   the mouth, which leaves 0.27 m of slack at each cheek.

3. **The tracking gate was measuring self-occlusion.** The ray cast to a point on
   an adult's centreline necessarily strikes that adult's own torso first — a
   0.078 m capsule against a 0.02 m tolerance — so a perfectly centred,
   completely unobstructed person scored **0.00** visibility. Hitting the
   target's own geometry is what seeing them *means*; those hits now end the cast
   successfully, and tracking went from 25 % to 100 %.

4. **Charging the two manoeuvre legs sequentially made every bay look
   unreachable** (above). Replaced with the measured concurrency model, pinned by
   two tests.

5. **Two presentation defects the preview caught.** Pipe runs at z = 0.545 and
   ceiling lights on the corridor centreline sat directly between the only usable
   camera angle and the subject, hiding the duck for most of the rollout; both
   now sit over the walls. `test_no_scenery_obstructs_the_corridor_itself` pins
   that as a scene invariant rather than a memory. Separately, the alcove
   scorecard drew its verdict column at a fixed offset and overlapped the bay
   name on every row, because "too shallow" is far wider than "viable"; it is now
   right-aligned by measured text width.

## Physical versus kinematic

This distinction matters for reading every number above.

- **Physical, authoritative, simulated:** the duck's floating base and 14
  actuated joints, driven by the stock ONNX policy through MuJoCo physics. Trunk
  height, path length, displacement, yaw and every stability figure come from
  this state and nothing else.
- **Kinematic, scripted, non-physical:** both adults. They are mocap bodies with
  `contype=0 conaffinity=0`, posed analytically each tick. They add **no degrees
  of freedom** to the robot (the model has exactly 7 free-joint qpos + 14 hinges
  + 4 hinges per adult), cannot push it, and never react to it. **A completed
  pull-over can therefore never be the result of a person nudging the duck into
  a recess**, and because they cannot collide, MuJoCo's own contact count is
  vacuous — the honest gate is the geometric clearance, measured every tick.
- **The walls are non-colliding too, and that is deliberate.** If they collided,
  "the duck stayed inside the corridor" would be enforced by the contact solver
  rather than demonstrated by the controller, and a robot that scraped along a
  wall for ten seconds would still pass. With non-colliding walls the corridor is
  a constraint the *controller* has to respect, and the gate measures the real
  surface distance to all 21 wall, cheek, crate and back-wall geoms every tick.
- **Rendering-only, isolated:** the head gaze and the stabilized camera rig. The
  head pose is applied to a **separate `MjData` copy** and never written back, so
  gaze cannot stabilize the robot — and equally, the robot's real head actuators
  are not exercised here.
  `test_gaze_never_touches_the_authoritative_walking_state` asserts `qpos`,
  `qvel` and `ctrl` are bit-identical across 120 camera updates, and a companion
  test proves the head genuinely moves in the isolated copy, so isolation is not
  achieved by doing nothing.

The machine also consumes the encounter prediction **and** the camera verdict
from the *previous* control tick. Measuring and deciding within one tick would
let a pull-over be authorised, or a yield released, by a state that only exists
after the decision. One tick at 50 Hz is 20 ms, which is honest and is also what
a real perception pipeline would incur.

## Limitations

- **Pedestrian perception is a semantic proxy, not sensing.** Position, velocity
  and body size come from the simulator. No detector, no tracker, no
  vision-based motion estimation. A physical robot would need multi-person
  detection, tracking and velocity estimation to replace this.
- **The corridor's own geometry is known a priori.** The duck does not discover
  the alcoves — it scores a list the scene generator and the decision layer share.
  A real robot would need to map or perceive the recesses, and the depth of a bay
  and the presence of the crates would both have to be sensed. The *scoring* is
  real geometry, but the *inventory* is given.
- **Adults are scripted and kinematic.** Constant speed, fixed lateral offset, no
  reaction to the duck, no physical interaction. A real person would step aside
  too, which is a negotiation this behavior does not model.
- **A pull-over is never re-decided once started.** This is deliberate: changing
  target halfway leaves the robot in the middle of the passage at the worst
  possible moment, and the commitment already covers the whole manoeuvre with
  margin. An abort-and-reselect rule needs its own acceptance evidence.
- **The duck never reverses.** An alcove it has already walked past is refused as
  "behind the duck" rather than reconsidered, because this behavior has no
  measured reverse primitive. That is why the second encounter has only one
  viable candidate.
- **The PiP is informational.** The behavior's decisions are not taken from
  pixels — but the tracking gate that grades each yield *is* measured through
  that exact camera.
- **Simulation only.** No hardware validation of this behavior.

## Reproduce

Use the environment from a local `microduck_rl` checkout:

```bash
cd /path/to/microduck_rl
export MICRODUCK_RL_ROBOT_DIR=$PWD/src/mjlab_microduck/robot/microduck

# Regenerate the scene from its generator (optional; the XML is committed)
python /path/to/narrow-corridor-etiquette/tools/build_scene.py

# Physics and acceptance gates, no rendering dependencies at all
uv run python /path/to/narrow-corridor-etiquette/scripts/render_narrow_corridor.py \
  --no-render --seconds 50 --metrics /tmp/nce-metrics.json

# Low-fps preview for visual inspection
uv run python /path/to/narrow-corridor-etiquette/scripts/render_narrow_corridor.py \
  --seconds 50 --fps 3 --width 960 --height 640 \
  --out /tmp/preview-frames --metrics /tmp/preview-metrics.json

# Final 2500 frames at 50 fps
uv run python /path/to/narrow-corridor-etiquette/scripts/render_narrow_corridor.py \
  --seconds 50 --fps 50 --width 960 --height 640 \
  --out /tmp/narrow-corridor-frames \
  --metrics media/narrow-corridor-etiquette-metrics.json

ffmpeg -framerate 50 -i /tmp/narrow-corridor-frames/f%05d.png \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart \
  media/narrow-corridor-etiquette.mp4

# Measurements behind the constants above
uv run python /path/to/narrow-corridor-etiquette/tools/sweep_commands.py \
  --policy /path/to/narrow-corridor-etiquette/onnx/alpha_walking.onnx --seconds 6
uv run python /path/to/narrow-corridor-etiquette/tools/measure_pullover.py \
  --policy /path/to/narrow-corridor-etiquette/onnx/alpha_walking.onnx

# Tests
uv run --with pytest python -m pytest /path/to/narrow-corridor-etiquette/tests -q
```

## Contents

- `assets/scene_narrow_corridor.xml` — corridor, four recesses, crates, lobby,
  two adults, cameras.
- `scripts/corridor.py` — the corridor's geometry as a single source of truth:
  wall lines, alcove definitions and their usable depth, the centre passage,
  footprint helpers, and the side-by-side passing arithmetic.
- `scripts/people.py` — the pedestrian schedule and its kinematics, plus the
  measurements that keep it honest (traverses, visible jump, separation).
- `scripts/encounter.py` — encounter prediction, alcove scoring and rejection,
  and every measured locomotion constant.
- `scripts/etiquette_model.py` — the nine-state machine and the corridor
  controllers.
- `scripts/etiquette_camera.py` — isolated gaze, the PiP camera, and the
  person-tracking gate.
- `scripts/rollout_etiquette.py` — authoritative policy/physics integration.
- `scripts/contact_geometry.py` — exact duck-vs-adult and duck-vs-wall surface
  distance, and the MuJoCo narrowphase traps it has to survive.
- `scripts/etiquette_metrics.py` — the twenty-five acceptance gates.
- `scripts/render_frames.py`, `scripts/video_overlay.py` — video presentation.
- `scripts/render_narrow_corridor.py` — entry point.
- `tools/build_scene.py` — scene generator.
- `tools/sweep_commands.py` — command sweeps, including the lateral primitive
  no sibling behavior had measured.
- `tools/measure_pullover.py` — the pull-over primitive, timed end to end.
- `tests/` — 161 tests: pure decision logic, real MuJoCo scene and camera
  invariants, and **32 synthetic counterexamples proving every gate can fail**.
- `onnx/alpha_walking.onnx` — byte-identical stock walking policy
  (`sha256:e36332d3…daa6c`, matching the other behaviors and the upstream
  checkout).
