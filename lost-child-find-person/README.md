# lost-child-find-person — losing your person in a crowd, and finding *her* again

A deterministic MuJoCo scenario in which Microduck follows a specific adult
across a busy concourse, **loses her behind a solid kiosk**, stops dead,
searches the hall with its head while its feet do not move, **refuses three
people who are not her — two of them built to be mistaken for her**, confirms
her identity, walks a planned route back to her, and does the whole thing twice:

```text
FOLLOW → LOST → STOP → SEARCH_SWEEP → CANDIDATE ⇄ REJECT
       → CANDIDATE → REACQUIRED → REJOIN → FOLLOW → … → SAFE
```

`CANDIDATE ⇄ REJECT` is a loop, not a one-shot: in the first cycle it runs three
times, once per refused person, before the sweep finds the guardian.

This is a new behavior folder. It does not modify the validated
[`move-away/`](../move-away/), [`move-away-crowd/`](../move-away-crowd/),
[`follow-me/`](../follow-me/),
[`follow-me-among-others/`](../follow-me-among-others/),
[`come-here-recall/`](../come-here-recall/),
[`crosswalk-guardian/`](../crosswalk-guardian/),
[`narrow-corridor-etiquette/`](../narrow-corridor-etiquette/) or
[`queue-politely/`](../queue-politely/) baselines it was derived from.

## Demo

▶️ **[Watch or download the full MP4 (60 s, 960×640, 50 fps)](media/lost-child-find-person.mp4)**

The wide shot looks down on the hall from the south-west at an angle chosen by
measurement rather than taste (see *Filming a 25 cm robot in a 6.6 m hall*). The
upper-right PiP is the **exact camera every visibility number is measured
through**, placed at the duck's physical head-camera position and stabilized to
a level horizon. The left panels carry the state, the guardian's visibility and
what is blocking her, the live identity scorecard with its four descriptor terms
broken out, and the running log of refusals. The plan view draws every real
footprint, the real obstacle shapes, the **sightline coloured by whether the
camera can actually see her**, and the planned rejoin route.

## The problem this behavior actually solves

"Follow a person" is easy while you can see them. This behavior is about the
five seconds after you cannot, and it is built so that three separate things
that are usually conflated have to be solved separately:

1. **The loss must be geometric, not scripted.** Nothing in the code says
   "hidden at t = 16". A 1.10 m kiosk comes between the duck and the guardian
   because of where they both walked, and the duck notices because a real
   MuJoCo ray cast to five points on her body returns nothing for 0.60 s.
2. **Not moving is a decision.** A robot that keeps walking toward where it last
   saw someone is dangerous in a crowd. The command is *exactly* zero in every
   state where the duck does not know where its guardian is, and that is graded
   per tick rather than asserted.
3. **Finding "a person" is not finding "your person".** The hall contains eight
   other adults. Two of them were built to score high: same teal shirt, same
   shoulder bag, differing in exactly one feature each. The duck has to decline
   them.

The scene is arranged so these are measurements rather than claims:

| the duck must | and the scene makes it | measured evidence |
|---|---|---|
| lose her for a real reason | kiosk 1.10 m tall, taller than the topmost body sample at *z* = 0.66 | 7.48 s continuous occlusion attributed to `obs_kiosk` |
| not simply wait it out | she keeps walking while hidden | 3.69 m of world-space trail retained |
| refuse near-misses | `mira` and `sofia` differ in exactly one feature | scores 0.862 and 0.819, both refused |
| refuse strangers too | six more adults on independent loops | `faruq` refused at 0.687 |
| accept only her | her descriptor matches on all four terms | 2433 accept-grade sightings, **all `priya`** |

## Cast

The identity layer compares **four features**: shirt colour (weight 0.45),
standing height (0.35), headwear (0.10) and shoulder bag (0.10).

| | who | shirt | height | cap | bag | how they differ from the guardian |
|---|---|---|---|---|---|---|
| 🎯 | **priya** — the guardian | teal | 1.72 m | no | yes | — |
| ⚠️ | **mira** — look-alike | teal | 1.70 m | **yes** | yes | **the cap, and nothing else** |
| ⚠️ | **sofia** — look-alike | teal | **1.60 m** | no | yes | **12 cm shorter, and nothing else** |
| | arun, bekele, costa, dahl, eze, faruq | various | ~1.7 m | mixed | mixed | obviously different colours |

A distractor in a different-coloured shirt proves nothing — the shirt term alone
rejects it. Both look-alikes are within a hair of the guardian's teal and carry
the same bag, so the shirt and bag terms are effectively tied and the decision
falls on one remaining feature each. Both score **above** the candidate
threshold of 0.55 and **below** the accept threshold of 0.90, which is exactly
what a tempting false positive is.

## The state machine

Ten states, of which **eight forbid any locomotion command at all**:

| state | what it means | command | seconds |
|---|---|---|---:|
| `FOLLOW` | guardian in camera, holding ~0.85 m behind her | walking | 33.68 |
| `LOST` | unseen for 0.60 s — the loss is declared | **exactly 0** | 0.04 |
| `STOP` | halted; her position is unknown | **exactly 0** | 1.60 |
| `SEARCH_SWEEP` | head sweeping ±104°, feet still | **exactly 0** | 4.38 |
| `CANDIDATE` | somebody is being scored | **exactly 0** | 3.52 |
| `REJECT` | a candidate has been refused, and is held on screen | **exactly 0** | 2.16 |
| `REACQUIRED` | identity confirmed for 0.90 s continuously | **exactly 0** | 0.04 |
| `REJOIN` | walking a route planned once, at reacquisition | walking | 14.58 |
| `SAFE` | rejoined, inside the standoff band | **exactly 0** | — |
| `DONE` | terminal | **exactly 0** | — |

MEASURED peak command magnitude in each stationary state: `LOST` 0.0, `STOP`
0.0, `SEARCH_SWEEP` 0.0, `CANDIDATE` 0.0, `REJECT` 0.0, `REACQUIRED` 0.0.
**Blind movement steps: 0.**

### Why the search is done with the head

Because a body scan is *physically impossible* for this robot, which is a
measurement and not a design preference. MEASURED over 6 s at `vx = 0`:

| `wz` | yaw change | rate |
|---:|---:|---:|
| −0.55 | −6.0° | −1.0 °/s |
| −0.40 | −4.5° | |
| +0.40 | +3.7° | |
| +0.55 | +5.0° | +0.8 °/s |

A five-second stationary sweep would turn the duck about four degrees. The head
yaw joint, by contrast, spans a measured **±170°**. So the only search this
robot can perform is a head sweep at an exactly-zero locomotion command — which
is also the only search the acceptance gate permits. The constraint and the
safety requirement happen to agree.

Holding zero really is holding still: MEASURED, 10 s of exact-zero command from
a standing start drifts **0.0014 m**.

### Why the loss and the reacquisition windows differ

`LOSS_CONFIRM_S = 0.60` but `REACQUIRE_CONFIRM_S = 0.90`. The asymmetry is
deliberate: a single stride of somebody crossing the sightline must not count as
losing your guardian, and — far more importantly — **a false lock costs more
than a slow one**. Confirmation is a *duration*, so a lucky glimpse cannot
authorise a reacquisition.

## What actually happened

60.0 s · 3000 control steps at 50 Hz · policy `alpha_walking.onnx`
(SHA-256 `e36332d3…141daa6c`) · 61-D observation · action scale 0.9 ·
gyro sensor `imu_ang_vel`.

### Timeline

| t (s) | transition | detail |
|---:|---|---|
| 16.58 | `FOLLOW → LOST` | guardian not visible for 0.60 s |
| 16.60 | `LOST → STOP` | halting; position unknown |
| 17.40 | `STOP → SEARCH_SWEEP` | beginning head sweep |
| 17.68 | `SEARCH_SWEEP → CANDIDATE` | **sofia**, score 0.819 |
| 18.58 | `CANDIDATE → REJECT` | *stands 1.60 m, guardian 1.72 m* |
| 21.46 | `SEARCH_SWEEP → CANDIDATE` | **mira**, score 0.862 |
| 21.52 | `CANDIDATE → REJECT` | *wears a cap; guardian does not* |
| 22.26 | `SEARCH_SWEEP → CANDIDATE` | **faruq**, score 0.587 |
| 23.18 | `CANDIDATE → REJECT` | *shirt colour differs from the guardian's teal* |
| 24.30 | `SEARCH_SWEEP → CANDIDATE` | **priya**, score 1.000 |
| 25.06 | `CANDIDATE → REACQUIRED` | confirmed 0.90 s |
| 25.08 | `REACQUIRED → REJOIN` | route planned |
| 37.18 | `REJOIN → FOLLOW` | standoff reached |
| 46.34 | `FOLLOW → LOST` | **second cycle begins** |
| 47.16 | `STOP → SEARCH_SWEEP` | |
| 48.68 | `SEARCH_SWEEP → CANDIDATE` | **priya**, score 1.000 |
| 49.56 | `CANDIDATE → REACQUIRED` | confirmed 0.90 s |
| 52.06 | `REJOIN → FOLLOW` | standoff reached |
| 60.00 | `FOLLOW → SAFE` | complete, at safe standoff |

### The occlusion is real geometry

| | |
|---|---|
| first loss declared | **16.58 s** (the `FOLLOW → LOST` transition; the first record carrying the `LOST` label is stamped one 20 ms tick later at 16.60 s, which is what the metrics report as `first_loss_at_s`) |
| longest continuous occlusion | **7.48 s** |
| attributed to | **`obs_kiosk`** (341 ticks), with `faruq_torso` 29 and out-of-frustum 4 |
| pre-loss visibility while following | 95.78 % |
| world-space trail retained | **3.6885 m**, last seen at (−1.143, 1.857) |

Occlusion is measured by casting a real MuJoCo ray to five stature-scaled points
on the body — knees, waist, chest, head, crown — through actual scene geometry.
The kiosk is 1.10 m tall and the topmost sample sits at *z* = 0.66 with the
duck's eye near *z* = 0.19, so nothing 0.90 m or taller can be seen over.

### The refusals

| t (s) | who | score | shirt | stature | cap | bag | verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 18.58 | sofia | 0.819 | 0.03 | **0.48** | 0.00 | 0.00 | *stands 1.60 m, guardian 1.72 m* |
| 21.52 | mira | 0.862 | 0.03 | 0.07 | **1.00** | 0.00 | *wears a cap; guardian does not* |
| 23.18 | faruq | 0.687 | **0.37** | 0.14 | 0.00 | **1.00** | *shirt colour differs from the teal* |

Columns are per-feature **mismatch**; the failing term is bolded. Every refusal
was made on a **complete descriptor** — all four features readable — so none of
them is an artefact of a partial view. The duck refused three distinct people
and accepted **zero** wrong targets.

### Both rejoins are real physical progress

| | cycle 1 | cycle 2 |
|---|---:|---:|
| lost at | 16.58 s | 46.34 s |
| refusals | **3** | 0 |
| rejoined at | 37.18 s | 52.06 s |
| path walked | **2.7435 m** | **0.4322 m** |
| net displacement | 2.0319 m | 0.3340 m |
| range at reacquisition | 2.7286 m | 1.0273 m |
| range on arrival | **0.8947 m** | **0.7092 m** |
| guardian visible during rejoin | **100.0 %** of 605 steps | **100.0 %** of 124 steps |
| route | 3 waypoints, direct line blocked by kiosk | 2 waypoints, direct line clear |
| min person clearance | 0.281 m | 0.389 m |

A nonzero command does not prove the policy crossed its gait-onset threshold, so
each rejoin is graded on **physical path length, net displacement and a reduced
range** independently.

### Safety and stability

| | measured | requirement |
|---|---:|---|
| falls | **0** | 0 |
| contacts | **0** | 0 |
| min trunk height | **0.11137 m** | > 0.09 m |
| final trunk height | **0.11627 m** | ≈ 0.116 m nominal |
| min clearance to any person | **0.1050 m** (`dahl`) | > 0 |
| min clearance to scenery | **0.1875 m** (`obs_kiosk`) | > 0 |
| final range to guardian | **0.7057 m** | 0.45–0.75 m |
| guardian visible at the end | **yes** | yes |
| total path walked | 6.2767 m | — |

**All 25 acceptance gates pass. 205 tests pass.**

## Limitations — read this before believing the video

This is a **simulation result**. Four things in it are weaker than they look,
and each is stated here in the same place rather than left to be inferred.

### 1. Identity is a semantic proxy, not an RGB classifier

The duck does **not** run a person re-identification network over pixels. It
reads an appearance descriptor — shirt colour, standing height, headwear,
shoulder bag — **out of the simulator** for whichever body its camera can
currently see, and compares it against the descriptor recorded of its guardian.
When the HUD says *"looking at mira, 0.862, wears a cap"*, the cap was read from
a data structure, not from the image in the PiP.

What is **not** faked is *which* features are readable. Each person is sampled
at five stature-scaled heights, and a feature can only be read if the camera can
actually see the body parts carrying it:

* shirt and bag need the torso samples;
* the cap needs the head and crown samples;
* stature needs **the knees and the head**, because you cannot judge someone's
  height from their shoulders up.

So a candidate standing half behind a column yields an **incomplete descriptor**
and can never be confirmed, however well the visible half matches. That is why
the confirmation gate is a duration. A real system would swap the descriptor
lookup for a network over the visible pixels; **the visibility logic, and
therefore every claim here about occlusion, confirmation duration and false
candidates, would be unchanged.**

### 2. There is no audio, and no calling out

A lost child says *"where are you?"* and a guardian answers. Nothing in this
scenario models sound: no speech, no localisation by ear, no attention-getting.
The duck is mute and deaf, and the guardian never helps it. Everything here is
vision and geometry alone, which is the **harder** case but not the complete
one.

### 3. The gaze is rendering-only, and deliberately so

The head is a large fraction of this robot's mass, and the stock walking policy
was never trained to compensate an imposed head trajectory. So the head pose is
applied in an **isolated `MjData` copy** of the authoritative walking state and
**never written back**. Gaze cannot prop the robot up, and the reported trunk
heights are the physics the policy actually produced.

The PiP camera rig is **electronically stabilized**: it sits exactly where the
physical head camera sits, but holds a level horizon so the picture is readable
while the trunk pitches through its gait. It is labelled *(stabilized)* in the
overlay for that reason. Visibility is measured through **that exact camera**,
so the percentages and the picture cannot disagree.

### 4. The other adults are scripted, and the robot is the only agent

All nine adults follow fixed timed polylines. They do not react to the duck, do
not avoid it, and would walk through it if the geometry allowed — they are
non-colliding kinematic scenery. Consequently:

* **"Zero contacts" from the physics engine would be vacuous here**, so the gate
  does not use it. Clearance is a geometric measurement taken every control tick
  with `ContactProbe` against the real geoms at the real pose.
* The crowd is *timed*, not merely placed: the look-alikes are scheduled to walk
  into the search sweep **during** a loss, which is when a mistake would cost
  something. That makes them a fair test of the identity layer, but it is still
  an authored encounter rather than an emergent one.

### 5. No hardware validation

Nothing here has run on a physical Microduck. Sim-to-real for the locomotion is
inherited from the stock policy; the perception, identity and planning layers
above it have never met a real camera, real lighting, or a real crowd.

### A documentation defect that was found and corrected

`ADULT_HALF_EXTENT_M = 0.1647` was previously documented as "the widest planar
half-extent of an adult over a full gait cycle, MEASURED on the built scene by
`tools/measure_scene.py` and pinned by a test". **Every part of that was
wrong.** MEASURED on this scene, the guardian's exact planar half-extent is:

| | value |
|---|---:|
| pose zero, arms down (*t* = 0) | **0.1375 m** |
| mean over 300 poses | 0.1945 m |
| widest, mid-stride | **0.2629 m** |
| widest of any adult over the full 60 s | 0.2709 m (`eze`) |

0.1647 is neither the minimum, the mean, nor the maximum, and
`tools/measure_scene.py` does not exist in this project.

**The constant has been left at its value on purpose.** It is read by exactly
one function, `surface_gap`, which is explanatory prose about how the
0.45–0.75 m standoff band was sized; changing it would silently move a
documented figure without improving any safety property. It is now labelled a
**legacy nominal reference** in `lost_geometry.py`, and the true spread is
pinned by `test_the_adult_half_extent_constant_is_a_legacy_nominal`.

**No acceptance gate consumes it.** Every clearance gate measures real surface
separation per tick with `ContactProbe`, which accounts for the arm swing
exactly. That claim is itself enforced:
`test_no_acceptance_gate_consumes_the_legacy_half_extent` scans the shipped
source and fails if the constant ever leaks into a controller, planner or gate
— verified by mutation, not merely asserted.

The rollout's reported `adult_half_extent_m` is **0.1375 m**, a pose-zero sample
and explicitly not a bound; the metrics JSON now carries an
`adult_half_extent_basis` field saying so.

## Filming a 25 cm robot in a 6.6 m hall

The one thing this video may never do is hide the duck behind the very kiosk
whose occlusion it is demonstrating. So the framing was **scored, not chosen**:
the recorded 3000-step trace was replayed through candidate cameras — with the
renderer's own easing, azimuth swing and look-at clamp — and every sampled frame
was checked for whether the duck was on screen, clear of the HUD panels,
unoccluded in 3D against the real solid volumes, and how large it actually
appeared, plus whether the guardian and the kiosk were in shot.

| camera | duck unoccluded | clear of panels | guardian unoccluded | kiosk in shot | duck height |
|---|---:|---:|---:|---:|---:|
| az 290, el −28, d 4.9 | 1.000 ⚠️ | 1.000 | 1.000 | 1.000 | — |
| az 210, el −34, d 3.6 | 0.833 | 0.992 | **0.500** | 0.958 | 54.4 px |
| az 225, el −40, d 4.1 | 0.758 | 0.992 | 0.475 | 0.950 | 46.6 px |
| **az 210, el −40, d 3.8** | **1.000** | **0.992** | **1.000** | **1.000** | **50.9 px** |

⚠️ **The first row is why this table exists.** An early probe scored only the
*obstacles* and returned azimuth 290 with a perfect 1.000 — which then put the
hall's own **1.24 m north wall** between the camera and the duck for the whole
of the second cycle. The walls are now included as solid volumes, which moved
the answer to azimuth 210. Elevation −34 at 3.6 m makes the duck larger still,
but drops guardian visibility to 0.500, so it is rejected: the guardian must be
on screen for the follow and rejoin phases to be gradeable at all.

Two further rendering notes, both of which were bugs first:

* **The contact sheet selects frames through a manifest**, not by assuming
  `frame == t × fps`. At 4 fps the writer emits every 12th control tick, so
  frame 200 is *t* = 48.02 s, not 50.00 s — which captioned a `SEARCH_SWEEP`
  frame as "REJOIN cycle 2" until it was fixed.
* **The overlay's presentation memory is updated every control tick**, before
  the frame rate is applied, and is keyed by person. `mira` is scored for only
  three ticks (21.46–21.50 s); a 4 fps preview steps straight over them, and her
  refusal panel rendered blank while the final 50 fps render showed it. Keying
  by name also stops the sweep's *next* candidate being drawn under the
  *previous* one's name — at 21.76 s the tracker is already scoring `faruq`
  while `mira` is still inside her refusal hold.

## Reproduction

All commands from this directory, using the shared MuJoCo virtualenv.

```bash
# 1. the headless gate — no rendering dependencies at all (~82 s)
../../microduck_rl/.venv/bin/python scripts/validate_lost.py \
    --seconds 60 --json /tmp/metrics.json

# 2. the test suite (205 tests, ~9 s)
../../microduck_rl/.venv/bin/python -m pytest -q

# 3. a low-fps preview for visual inspection
../../microduck_rl/.venv/bin/python scripts/render_lost_child.py \
    --seconds 60 --fps 4 --out /tmp/lcfp-preview \
    --metrics /tmp/lcfp-preview-metrics.json
../../microduck_rl/.venv/bin/python tools/contact_sheet.py \
    --frames /tmp/lcfp-preview --out /tmp/preview-sheet.jpg

# 4. the final render (3000 frames, ~20 min)
../../microduck_rl/.venv/bin/python scripts/render_lost_child.py \
    --seconds 60 --fps 50 --width 960 --height 640 \
    --out /tmp/lcfp-final --metrics media/lost-child-find-person-metrics.json

# 5. encode
ffmpeg -y -framerate 50 -i /tmp/lcfp-final/f%05d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow -movflags +faststart \
    media/lost-child-find-person.mp4
```

The gate and the render run the **same** `LostRollout` and are graded by the
**same** `summarize`/`report`, so the video cannot disagree with the metrics; the
render path exits non-zero if any gate fails. Rendering imports `PIL` and
`imageio` only inside the render branch, so step 1 has no rendering dependency
— a claim pinned by `test_no_render_imports`.

## Layout

```text
assets/scene_lost_child.xml      the concourse, the cast, the markers
onnx/alpha_walking.onnx          stock walking policy (unmodified)
scripts/
  plaza_layout.py                obstacles: one source of truth for 3 consumers
  lost_cast.py                   the nine adults and the descriptor terms
  lost_people.py                 timed polylines; the guardian's route
  lost_geometry.py               standoff band, clearances, the legacy nominal
  contact_geometry.py            exact surface distance, and the MuJoCo traps
  lost_camera.py                 head pose, frustum, ray casts, readability
  lost_identity.py               scoring, candidates, refusals, acceptance
  lost_machine.py                the ten-state machine
  lost_control.py                command generation, per-sign yaw gains
  lost_memory.py                 world-space trail and the rejoin planner
  lost_record.py                 one control tick, flattened
  lost_metrics.py                summary and the 25 acceptance gates
  rollout_lost.py                integration: the strict per-tick order
  policy_runtime.py              scene loading, ONNX, observation assembly
  validate_lost.py               headless gate entrypoint
  render_lost_child.py           render entrypoint (gate still authoritative)
  render_frames.py               wide shot + PiP + measured camera framing
  video_overlay.py               frame composition and PiP chrome
  hud_style.py / hud_panels.py / hud_views.py    the overlay
tools/
  build_scene.py                 paints the scene from plaza_layout
  sweep_commands.py              gait onset and yaw asymmetry
  measure_scan.py                yaw dead band, lateral authority, head reach
  contact_sheet.py               labelled contact sheet, manifest-driven
  verify.sh                      compile, tests, gate
tests/                           205 tests
media/
  lost-child-find-person.mp4                 60 s, 960×640, 50 fps
  lost-child-find-person-telegram.mp4        ≤5 MB derivative, same 60 s/res/fps
  lost-child-find-person-contact-sheet.jpg   the nine moments
  lost-child-find-person-metrics.json        full metrics and gate results
```

## Credit

Locomotion is the **stock** `alpha_walking.onnx` from `microduck_rl`, unmodified
(SHA-256 `e36332d3…141daa6c`). Nothing here retrains or fine-tunes it. Every
behavior in this folder is built by choosing velocity commands for a policy that
already knew how to walk — the contribution is the perception, identity, memory
and planning above it, and the measurements that justify each number.
