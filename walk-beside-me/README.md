# Walk Beside Me

The duck walks **beside** a person rather than behind her, holds a side, and
changes side **only when it measures that its own side has gone**.

Behavior 6 of 12 in the Microduck lab. Locomotion is the **stock walking
policy**, byte-identical to the upstream checkpoint; nothing here is retrained.

```
media/walk-beside-me.mp4          86 s, 960x640, H.264 yuv420p, 50 fps, 4300 frames
media/walk-beside-me-metrics.json every gate, with its evidence
```

---

## What it does

The guardian, `nadia`, walks a 10.96 m filleted route across an open promenade
at 0.118 m/s. The duck starts **behind her and off to her right**, not in a
slot, and has to walk into formation.

| t (s) | state | what happens |
|---|---|---|
| 0.00 | `JOIN_SIDE` | The **right** slot is refused — the hedge is 0.09 m *inside* it. Goes left. |
| 6.26 | `BESIDE_LEFT` | Formation established on her left at 0.515 m lateral. |
| 8.86 | `SIDE_BLOCKED` | The **kiosk** enters the left lane: measured static gap **0.1082 m** against a 0.22 m margin. |
| 8.88 | `FALL_BACK` | Drops astern rather than cutting across her. |
| 13.32 | `CROSS_BEHIND` | Reaches **−0.6215 m** astern; begins the lateral transit. |
| 17.84 | `JOIN_OTHER_SIDE` | Committed to the far side; closing up. |
| 29.36 | `BESIDE_RIGHT` | Formation re-established on her right, held for the remaining 56.64 s. |

The route then bends **left 42.9°**, **right −42.9°** and **left 81.3°**, and
the formation follows all three.

---

## Measured results

Every figure below is read from `media/walk-beside-me-metrics.json`, which
`scripts/validate_beside.py` writes and the render reproduces byte-identically.

### The formation

| quantity | measured |
|---|---|
| time in formation (`BESIDE_LEFT` + `BESIDE_RIGHT`) | **59.24 s** of 86 s |
| distance walked while in formation | **10.848 m** |
| lateral offset while beside her | **0.4757 – 0.7237 m** (mean 0.5784) against a 0.45–0.75 m band |
| longitudinal error while beside her | **≤ 0.6444 m** against a 0.55 m tolerance |
| total path | 16.1122 m |
| first `BESIDE` | 6.28 s, on her **left**, at 0.515 m lateral |
| walked into formation before it | path 1.312 m, net 1.140 m |

### The side decision

| quantity | measured |
|---|---|
| side decisions | **2** — initial → left; blocked → right |
| completed physical switches | **1** |
| ticks with the duck's own side measured unusable | **498** |
| cause of the switch | `static:kiosk`, held for **1.0 s** before acting |
| gap that caused it | static **0.1082 m** (margin 0.22 m); nearest person 1.0028 m |

### The crossing — behind her, never in front

| quantity | measured |
|---|---|
| path walked during the switch | **3.9514 m** |
| net displacement | **2.5871 m** |
| lateral travel | **+0.7269 → −0.6276 m** |
| deepest point astern | **−1.1511 m** |
| **most forward point during the switch** | **−0.4200 m** (limit +0.22 m) |
| **most forward point at ANY tick of the whole 86 s** | **+0.0938 m** (limit +0.22 m) |
| minimum clearance to her legs during the switch | **0.5628 m** |

The duck never entered her forward half-plane, and that is measured **every
tick of the entire rollout**, not only during the crossing.

### The bends

| bend | turn | beside | max lateral | followed |
|---|---|---|---|---|
| 1 | left 42.879° | 5.72 s | 0.6249 m | ✅ |
| 2 | right −42.879° | 5.70 s | 0.6027 m | ✅ |
| 3 | left 81.254° | 10.82 s | 0.5714 m | ✅ |

### Safety and locomotion

| quantity | measured |
|---|---|
| contacts | **0** |
| falls | **0** |
| minimum trunk height | **0.11113 m** (floor 0.09 m) |
| final trunk height | **0.1193 m** (nominal 0.116 m) |
| minimum clearance to any person | **0.2846 m** (to `nadia`) |
| minimum clearance to any obstacle or wall | **0.0040 m** (to `obs_hedge_s`) |
| guardian visible, of steps where line of sight exists | **100.00 %** of 4300 |
| phases that hit a ceiling | **none** |
| `HOLD` / `DONE` steps | **0 / 0** |

**All 30 acceptance gates pass.** 260 tests pass.

---

## Semantic proxy, camera, and gaze — three different claims

These are routinely conflated. They are kept separate here because only one of
them is a perception result, and it is not the interesting-sounding one.

**1. Identity is a semantic proxy, not recognition.**
Every person is a non-colliding mocap body posed analytically each control tick.
The duck knows *which body is `nadia`* because the simulator says so. There is
no RGB classifier, no re-identification, no appearance model anywhere in this
behavior. Nothing is claimed about recognising a person from pixels.

**2. The camera geometry is real, and it is the camera the PiP shows.**
Visibility is *not* a proxy. It is measured through an actual frustum —
containment plus a MuJoCo ray cast through real scene geometry — from the
**exact camera the picture-in-picture renders from**, at the same 300×216 pixel
geometry, which is what sets its horizontal FOV. Five sample points per person,
scaled by that person's stature. The reported 100 % visibility and the picture
in the video are the same measurement. `beside_camera.PIP_W/PIP_H` is imported
by both the headless gate and the renderer for exactly this reason.

**3. Gaze is rendering-only and cannot prop the robot up.**
The head is posed in an **isolated `MjData`** that is a copy of the walking
state, and is never written back. The head is a large fraction of this robot's
mass and the stock policy was never trained to compensate an imposed head
trajectory, so feeding gaze into the physics would quietly change the
locomotion. `test_the_camera_never_writes_back_into_the_walking_state` parses
the camera's AST and fails on any simulator write outside the render copy.

The PiP rig is additionally **electronically stabilized**: it sits exactly where
the physical head camera sits, but holds a level horizon so a human can read it
while the trunk pitches through its gait. Both disclosures are drawn into the
PiP chrome in the video, not left to this file.

---

## Corrections this work discovered

Documentation must match what the code measures. Two claims in the inherited
prose did not, and are corrected here rather than quietly left alone.

### `DUCK_PLANAR_RADIUS = 0.1303` is an inherited nominal, and it is not conservative

`beside_geometry.DUCK_PLANAR_RADIUS` is **0.1303 m**, a figure carried over from
the sibling corridor scenes. Measured on *this* scene:

| figure | measured |
|---|---|
| conservative planar half-extent at pose zero | **0.1162 m** |
| conservative planar half-extent, **gait maximum** | **0.1421 m** |
| exact planar half-extent at pose zero | 0.0827 m |
| exact planar half-extent, gait maximum | 0.1155 m |

So the declared constant **over-states the robot at pose zero and under-states
it mid-gait**. It is not a conservative bound on the real footprint, and calling
it one would be wrong.

That is acceptable only because **no safety gate reads it**. The constant sizes
the refusal margins in prose. Clearance is measured *every control tick* by
`ContactProbe` and `WallProbe` against the real geoms at the real pose, which
accounts for arm swing and gait exactly. The safety-relevant claim is asserted
against the **measured gait maximum**, not against the constant:
`SIDE_STATIC_MARGIN_M` (0.22 m) and `SIDE_PERSON_MARGIN_M` (0.55 m) both exceed
0.1421 m, so a slot refused for being too near a surface is genuinely one the
duck could not occupy.

`test_the_declared_duck_radius_is_a_sizing_figure_not_a_clearance_gate` exists
to say this out loud rather than to assert a comfortable inequality that only
holds at pose zero. The HUD plan view draws the duck at the **measured 0.1162
m**, not at the nominal.

### Rafa's 168° hairpin drops its fillet

`rafa`'s background loop doubles back through **168°**. A 0.90 m fillet there
would need an 8.6 m cutback and his legs are 1.8 m, so `beside_route._build`
**leaves it as a plain vertex** rather than silently accepting a bad arc. His
heading therefore **steps discontinuously once**, by **141.02° at t = 23.80 s** —
the only such step in the scene, and the worst that
`beside_actors.max_heading_step` finds over the whole rollout.
`ROUTES["rafa"].corner_report()` returns 2 corners, not 3.

(The 168° is the *turn angle at the vertex*; the 141.02° is the *heading step
actually taken in one 20 ms tick*. They are different quantities and the fillet
drop is why the second one is nonzero at all.)

This is stated rather than hidden because a scripted actor that teleports its
heading could, in principle, corrupt a side decision. It cannot here, and that
is **measured, not argued**: over the whole 86 s rollout `rafa`'s closest
approach to *either* candidate slot is **4.4002 m** (at t = 44.14 s), against a
0.55 m pedestrian margin — a factor of **8.00**.
`test_the_one_dropped_hairpin_belongs_to_a_walker_who_decides_nothing`
measures that distance and fails if his route is ever moved near the guardian.

The two oncoming walkers who *can* reach a decision, `tomas` and `iris`, plus
the guardian herself, are held to continuous headings with no exception.

---

## How the video is framed

The wide camera was **measured, not eyeballed**. `tools/probe_framing.py`
replays the real 4300-tick recorded trace through candidate cameras — with the
renderer's own easing, swing and clamp — and scores each sampled frame on
whether the duck is on screen, clear of the HUD panels, and unoccluded in 3D
against real solid volumes (every obstacle at its true footprint *and* height,
the four perimeter walls, every person as a standing cylinder), plus guardian
visibility, kiosk visibility during the switch, and apparent sizes.

Chosen: **azimuth 10°, elevation −26°, distance 2.8 m, look-at bias 1.00**.

| score | value |
|---|---|
| duck on screen | 0.998 |
| duck unoccluded | 0.998 |
| camera eye inside the promenade | 1.000 |
| duck clear of the HUD panels | 0.984 |
| guardian in shot | 1.000 |
| kiosk in shot through the switch window | 0.892 |
| duck apparent height | 32.7 px |
| **formation separation on screen** | **180 px mean / 136 px min** |

Three things the probe caught that inspection alone would not have:

- **Formation separation had to become a scored term.** "Beside" is a *lateral*
  offset, so a camera looking along her direction of travel projects the whole
  formation onto a few pixels and the video stops showing the thing it is about.
  Adding that term moved the answer off the family the first sweep preferred.
- **The camera's own eye must stay inside the hall.** A free camera orbiting a
  look-at near the perimeter puts its eye beyond the wall, and MuJoCo renders
  the near wall as a slab across the shot. The first rendered frame showed
  exactly that, so containment became a scored term.
- **The look-at bounds are derived, not chosen.** Since
  `eye = lookat − forward · distance`, eye containment is an *asymmetric* box
  constraint on the look-at. Every symmetric clamp had to be tightened to
  whichever side was worst, which stopped the camera following the duck down a
  12.4 m promenade: the best symmetric setting scored **0.712** panel clearance
  against **0.984** here, at identical containment.

The look-at easing advances **once per control tick, not once per written
frame**, so the 4 fps preview, the 50 fps final render and the probe all fly the
same camera path.

---

## Reproduce

The headless gate imports no rendering stack at all — no PIL, no imageio, no
GPU. `test_the_headless_gate_imports_no_rendering_stack` pins that.

```bash
cd projects/microduck-lab/walk-beside-me

# 260 tests
../../microduck_rl/.venv/bin/python -m pytest -q

# the 86 s headless gate; writes the metrics this README quotes
../../microduck_rl/.venv/bin/python scripts/validate_beside.py \
    --seconds 86 --json media/walk-beside-me-metrics.json

# low-fps preview, for inspecting every phase
../../microduck_rl/.venv/bin/python scripts/render_walk_beside.py \
    --seconds 86 --fps 4 --out /tmp/wbm-preview

# the final render: 4300 frames at 50 fps, same gates, same numbers
../../microduck_rl/.venv/bin/python scripts/render_walk_beside.py \
    --seconds 86 --fps 50 --out /tmp/wbm-final \
    --metrics media/walk-beside-me-metrics.json

ffmpeg -y -framerate 50 -i /tmp/wbm-final/f%05d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow \
    media/walk-beside-me.mp4

# re-measure the framing against a recorded trace
../../microduck_rl/.venv/bin/python tools/probe_framing.py \
    --trace /tmp/wbm/trace.json --stage shortlist

# contact sheet (selects frames by TIME through the manifest)
../../microduck_rl/.venv/bin/python tools/build_contact_sheet.py \
    --frames /tmp/wbm-final --out media/walk-beside-me-contact-sheet.jpg

# <= 5 MB derivative, preserving duration, resolution, fps and frame count
ffmpeg -y -framerate 50 -i /tmp/wbm-final/f%05d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 28 -preset veryslow \
    -movflags +faststart media/walk-beside-me-telegram.mp4
```

The render path and the headless path run the **same** `BesideRollout` and the
**same** `summarize`/`report`. Both exit non-zero if any gate fails, so the
video cannot disagree with the numbers.

---

## Files

| path | what it is |
|---|---|
| `scripts/rollout_beside.py` | the integration loop: actors, camera, side choice, machine, controller, physics, in a strict per-tick order |
| `scripts/side_choice.py` | is this side usable? A measurement with a named cause, every tick |
| `scripts/beside_machine.py` | the state machine; no physics, no MuJoCo |
| `scripts/beside_control.py` | pure pursuit of a slot, speed chosen by longitudinal error |
| `scripts/beside_geometry.py` | what "beside" MEANS, in numbers |
| `scripts/beside_camera.py` | isolated-gaze head camera and the visibility measurement |
| `scripts/validate_beside.py` | the headless gate; no rendering imports |
| `scripts/hud_style.py` · `hud_panels.py` · `hud_views.py` | HUD palette, panels, plan view and timeline |
| `scripts/video_overlay.py` | frame layout and the PiP chrome that carries both disclosures |
| `scripts/render_frames.py` | the measured wide camera and the frame writer |
| `scripts/render_walk_beside.py` | render entry point; same rollout, same gates |
| `tools/framing_geometry.py` | pinhole camera, scene solid volumes, occlusion predicates |
| `tools/probe_framing.py` | the framing search and its scoring policy |
| `tools/build_contact_sheet.py` | contact sheet, selecting frames by time via the manifest |

No source file exceeds 300 lines.

---

## Design notes worth keeping

**The duck cannot strafe, and that is why the behavior has the shape it has.**
Measured on this model at `vx = 0`: `vy = ±0.22` produces under 4 mm of lateral
motion, and `vy = −0.28` produces 0.255 m sideways together with **51° of
unwanted yaw**. There is no command that moves this robot sideways without
turning it. "Move over one lane" is therefore *not a primitive*, and changing
sides has to be a **path** — fall behind, cross astern, come up the other side.
`CROSS_BEHIND` exists because of that measurement, not for narrative effect.

**Forward gait onset is a cliff, not a ramp.** `vx = 0.20` gives 0.010 m over
6 s (no gait at all); `vx = 0.22` gives 0.409 m. The controller emits either a
walking command or exact zero, never anything between, because a command in that
band appears in the metrics and produces nothing on the floor. Station-keeping
beside a walking person is therefore a walk-or-stand policy with hysteresis.

**The yaw axis is asymmetric and biased.** At `vx = 0.34`: `wz = −0.10` gives
−6.3 °/s while `wz = +0.10` gives **0.0 °/s** — the policy's own right bias
swallows a small left command entirely. Each sign carries its own gain, ceiling
and dead band.

**Decisions are made on measurements taken *before* the physics step.** Grading
a side and acting on that grade within the same tick would let a decision be
authorised by a world state that only exists after the decision was made. One
control tick at 50 Hz is 20 ms, which is honest and is what a real perception
pipeline incurs anyway.

**The crossing waypoint cursor is monotonic.** A stateless "nearest unreached
waypoint" selector re-targets a waypoint the duck has already passed as soon as
its distance grows again — that is how the sibling `lost-child-find-person`
produced an endless loop around a corner. `_advance_cross_cursor` is the only
thing permitted to move it, and it never decreases.

## Limitations

- Simulation only. No hardware validation.
- Every person is a **scripted mocap actor**, not a reactive agent. They do not
  avoid the duck; the duck does all the avoiding.
- Identity is a **semantic proxy** (see above). No appearance-based recognition.
- Pedestrian prediction is a **linear extrapolation** over a 3 s window. It is a
  stated proxy for a tracker, not a tracker.
- One completed side switch in this 86 s scenario. The machine supports repeated
  switches with a cooldown, but only one is exercised here.
