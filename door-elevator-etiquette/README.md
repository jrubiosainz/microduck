# door-elevator-etiquette

**Waiting its turn.** The duck walks up to a narrow automatic door and **stops
outside the threshold** while two people come out towards it. It watches them,
lets them clear, and only then follows its guardian through — behind her, never
alongside. At the lift it waits **beside** the doors rather than in front of
them, holds still while **three** occupants step out, boards after the guardian,
moves to the side of the car, rides **exactly still** in a sealed lift, and when
the doors open at the target floor it lets her out first before following.

Behavior 8 of 12 in the Microduck lab. Locomotion is the **stock walking
policy**, byte-identical to the upstream checkpoint; nothing here is retrained.

Everything below is MEASURED from the run in `media/`, by
`scripts/validate_etiquette.py`, which imports no rendering stack at all.

## The run

| | |
|---|---|
| Duration | 110.0 s, 5500 control steps at 50 Hz |
| Route | **8.08 m**, 9 bends, through 3 apertures |
| Physical walk | **11.19 m** of path, 6.74 m net, 68.1 s of it walking |
| Yield at the door | **7.64 s**, forward command **exactly 0.0** |
| Occupants out before boarding | **3** of 3 |
| Ride | **9.88 s** sealed, command **exactly 0.0**, drifted 0.032 m |
| Order | **0** overtaking steps in 5087 samples; gap 0.33–2.65 m |
| Shared apertures | **0** ticks of 1016 the duck spent inside one |
| Visibility | **100 %** of monitored line-of-sight steps |
| Safety | **0 contacts, 0 falls**, +0.237 m to people, +0.191 m to scenery |

**40/40 acceptance gates pass. 191/191 tests pass**, including **45 mutation
counterexamples** that each prove one gate rejects its own violation.

## The video

`media/door-elevator-etiquette.mp4` — H.264, **960×640 @ 50 fps, 5500 frames,
110.0 s**. A derivative under 5 MB at identical resolution, frame rate and
duration is in `media/door-elevator-etiquette-telegram.mp4`; both decode cleanly
end to end. A per-phase contact sheet is in
`media/door-elevator-etiquette-contact-sheet.jpg`.

## The two sequences

### A — the doorway

| t (s) | state | what happens |
|---|---|---|
| 0.00 | `APPROACH_DOOR` | walks 1.40 m up to its holding point |
| 9.32 | `YIELD_EXITERS` | **stops 0.86 m short of the plane**, 2 exiters pending |
| 16.96 | `FOLLOW_THROUGH` | both measured clear for 0.6 s continuously; moves off |
| 22.84 | — | enters the aperture, door **1.00 open**, 0.66 m clear |

`tomas` and `leila` walk **west through the opening towards the duck**, in
opposite lanes of the same 0.66 m door, 0.8 s apart. The duck's own footprint
never enters the threshold band until the yield ends: **0 early steps**, against
a band it does enter for 390 ticks later in the run — so the gate is measuring
something the duck could have got wrong.

### B — the lift

| t (s) | state | what happens |
|---|---|---|
| 31.30 | `APPROACH_LIFT` | crosses the lobby |
| 43.40 | `WAIT_SIDE` | **beside** the doors, 0.20 m clear of the exit passage |
| 48.94 | `DOORS_OPEN` | the doors begin to move; still exactly zero |
| 49.46 | `LET_OCCUPANTS_EXIT` | 3 occupants inside; holds for **11.14 s** |
| 60.60 | `FOLLOW_GUARDIAN_IN` | all 3 measured clear, sustained 0.6 s |
| 75.06 | `POSITION_INSIDE` | inside the car, guardian already aboard, gap 1.03 m |
| 77.10 | `RIDE` | at its holding spot; **9.88 s of exact zero** |
| 86.98 | `DOORS_OPEN_TARGET` | rear doors opening; **waits for her** |
| 90.36 | `FOLLOW_OUT` | she is 0.30 m through; follows |
| 106.22 | `DONE` | 0.35 m beyond the rear plane, on the target floor |

The duck is inside the cabin for **24.76 s** with a minimum clearance of
**0.116 m** to the nearest interior face, and **0** ticks outside it while
riding.

## What is real, and what is a proxy

Stated plainly, because a robot demo about doors invites the reader to assume
the robot operated them.

**Real:** the walking. Every step is the stock `alpha_walking.onnx` policy
(SHA-256 `e36332…daa6c`, byte-identical to `microduck_rl`) driven at 50 Hz
through the exact `imu_ang_vel` sensor with a 61-D observation and the shipped
0.9 action scale. Nothing is trained here.

**Real:** the etiquette. Every stop is caused by a measurement the duck took —
somebody's position against an aperture plane, a door's open fraction, its own
footprint against a zone — and every release requires that measurement to hold
continuously for 0.6 s. There is no schedule anywhere in the decision path.

**Real:** the geometry. Clearances are exact analytic surface distances against
every wall, jamb, cabin panel and **moving door leaf**, measured every control
tick. The zone claims use the duck's conservative bounding-sphere radius
(0.1162 m), which over-states the robot: its exact planar half-extent is
0.0827 m. Over-stating makes every zone claim *harder*, which is the safe
direction.

**Real:** the camera. Visibility is frustum containment plus a MuJoCo ray cast
through actual scene geometry, measured through the *exact* camera the PiP is
rendered from.

**A scripted kinematic proxy:** the doors. There is no door controller, no
motion sensor and no call button. Each door's open fraction is a smootherstepped
function of time declared in `lobby_doors.DOOR_SCHEDULE`. What is **not** a
proxy is the consequence: the duck reads the fraction through the interface a
real sensor would provide, and "it never moved through a closed door" is graded
on the fraction the door actually had at the tick the duck's footprint entered
the aperture — **1.00 at all three crossings**.

**A semantic proxy:** person identity. Bodies are identified by MuJoCo body id,
not by an RGB classifier. The *geometry* that decides whether the camera can see
them is real; the label on them is given.

**Scripted:** all eight people. None of them reacts to the duck, none can be
pushed by it, and none stops because it arrived — deliberately. Traffic that
waited for the robot would make every etiquette claim vacuous: "it let them out
first" would be true of a duck that walked straight in.

**A cutaway:** the shaft and cabin walls are 1.35 m and 1.15 m rather than full
height, so the overhead camera can see into the car. Both are far above the
0.90 m occlusion threshold and the 0.70 m topmost camera sample on an adult, so
the duck's **own** head camera is occluded by them exactly as it would be by
full-height walls. Nothing the robot measures changes.

**Not validated:** hardware. This is simulation only.

## The measurements that shaped the design

Constants are re-measured per scene (`tools/sweep_commands.py`), never inherited.
Three findings changed the building itself.

### The duck cannot turn on the spot — so the lift is a through-car

MEASURED at `vx = 0` over 3 s, across the whole command range:

| `wz` | rate | drift |
|---|---|---|
| −0.42 | −1.6 °/s | 0.0016 m |
| −0.16 | −0.5 °/s | 0.0001 m |
| +0.16 | +0.7 °/s | 0.0017 m |
| +0.42 | +1.4 °/s | 0.0031 m |

The best turn-in-place command yields **1.6 °/s**, so squaring up 180° inside a
lift car would take **113 seconds**. A single-entry cabin is therefore not a
scenario this robot can perform at all. The lift in this behavior is a
**through-car** with doors front and rear — a real service-lift configuration —
which makes the whole route monotonically forward. The building was shaped by
the measurement, not the other way round. There is no turn-in-place command
anywhere in this behavior.

### The yaw ceiling was inherited, and it was wrong

A first draft capped both signs at `wz = 0.55` by inheritance from a sibling
behavior. That gives a 0.76 m left-hand turning circle and **rejected every bend
in the route**. The axis has not saturated there at all:

| `wz` at `vx=0.34` | rate | radius |
|---|---|---|
| +0.34 | +9.7 °/s | 0.71 m |
| +0.42 | +12.7 °/s | 0.54 m |
| **+0.58** | **+18.9 °/s** | **0.36 m** |
| +0.68 | +22.8 °/s | 0.29 m |

The ceiling is **0.58**: the largest command whose measured minimum trunk height
(0.1130 m) is indistinguishable from the straight-line figure. 0.68 turns faster
still and is left on the table deliberately — it is the first command whose gait
degrades, and a behavior about walking politely should not ride the edge of its
own stability to make a corner.

### Gait onset is a cliff, and it is at 0.24 here

MEASURED over 6 s: `vx = 0.22 → 0.009 m` (no gait at all), `vx = 0.24 →
0.525 m`. A robot that "edged forward slowly" while somebody came out of a door
would emit 0.22, stand still, and log a nonzero command. **So yielding is a
state, not a speed.** The duck walks or it holds exactly zero — and the seven
zero-command states in this run drift between **0.0003 m and 0.032 m** total,
against a MEASURED 0.0055 m per 10 s of exact zero.

## The apertures are wide enough for two, on purpose

| aperture | clear | duck + adult abreast leaves |
|---|---|---|
| concourse door | 0.66 m | **0.301 m** |
| lift | 0.72 m | **0.361 m** |

This is the arithmetic that makes the etiquette gate *mean* something. Squeezing
the doorway until two bodies physically could not fit would turn "the duck never
went through side by side" into a fact about the wall rather than about the
robot, and the gate would pass no matter what the state machine did. The
openings are deliberately wide enough to misuse, and the gate checks **per tick**
that the duck and another body were never inside the same aperture box: **0
shared ticks of the 1016** the duck spent inside one.

## Two independent refusals

The duck is held out of an aperture by **two mechanisms computed from different
quantities**, so a mistake in either alone cannot produce a side-by-side pass:

1. **The leg clamp.** Each state may walk exactly one leg of the route, and the
   pursuit point is clipped to that leg's end. While a state has not been
   released there is simply no target beyond the holding point — stopping is
   structural, not a remembered zero.
2. **The interlock.** `etiquette_sense.build_interlock` refuses to advance
   whenever this tick's raw occupancy shows anybody in the aperture ahead, or
   the door is not open far enough to pass. It never reads the state machine.

The interlock held the duck for **0 ticks** in this run — the machine never
needed overriding — and a test mutates the machine to release early and requires
the controller to hold anyway.

## States

```
APPROACH_DOOR → YIELD_EXITERS → FOLLOW_THROUGH → APPROACH_LIFT → WAIT_SIDE
    → DOORS_OPEN → LET_OCCUPANTS_EXIT → FOLLOW_GUARDIAN_IN → POSITION_INSIDE
    → RIDE → DOORS_OPEN_TARGET → FOLLOW_OUT → DONE
```

| State | Seconds | Command | What it is |
|---|---|---|---|
| `APPROACH_DOOR` | 9.32 | walking | up to the holding point outside the threshold |
| `YIELD_EXITERS` | 7.64 | **0.0** | two people are coming out; standing still |
| `FOLLOW_THROUGH` | 14.34 | walking | through the doorway, behind her |
| `APPROACH_LIFT` | 12.10 | walking | across the lobby |
| `WAIT_SIDE` | 5.54 | **0.0** | beside the doors, out of the exit passage |
| `DOORS_OPEN` | 0.52 | **0.0** | the doors begin to move |
| `LET_OCCUPANTS_EXIT` | 11.14 | **0.0** | three occupants step out |
| `FOLLOW_GUARDIAN_IN` | 14.46 | walking | boarding after her |
| `POSITION_INSIDE` | 2.04 | walking | crossing to the side of the car |
| `RIDE` | 9.88 | **0.0** | sealed car, exactly still |
| `DOORS_OPEN_TARGET` | 3.38 | **0.0** | she steps out first |
| `FOLLOW_OUT` | 15.86 | walking | following her off |
| `DONE` | 3.78 | **0.0** | on the target floor |

Seven of the thirteen states hold an exact zero. **41.9 s of the 110 s run is
the robot holding still**, 38.1 s of it waiting on somebody else — which is what
the behavior is about.

`PUSH_THROUGH` and `BOARD_FIRST` are declared as forbidden states and asserted
to have zero steps — the two named failures this behavior exists to avoid.

## The choreography is solved, not tuned

`tools/measure_legs.py` walks the whole route with the real policy and **no state
machine**, and reports when the duck actually arrives at each holding point.
`tools/tune_phasing.py` turns those arrivals into the door schedule and the
people's start times, and **fails** if any edge falls on the wrong side of one.
Both are re-run after any geometry change.

That tooling caught four bugs that a rendered video would have shown only as
something looking subtly wrong:

- **the guardian walked through sealed doors** at 40.3 s and 69.7 s, which would
  have made the duck's own no-closed-doors gate look arbitrary;
- **the guardian stood between the duck's head camera and the exiters** it was
  waiting for, blocking 323 ticks and dropping the visibility figure to 28.7 %;
- **the guardian's own body overlapped the duck's** at the door hold — the
  measured surface clearance went to **−0.0609 m over 163 control steps**;
- **the guardian stopped just inside the cabin door**, at arc 5.746 on the
  duck's route against the duck's own holding point at 6.009, so the order
  measurement reported an overtake on a run where the duck passed nobody.

A fifth was caught by the route builder itself. `etiquette_route._build` used to
skip any corner whose fillet did not fit and leave a **hard vertex** — which
turns a walker through the whole corner in one control tick. Four routes had
one. It now raises with the radius that would fit, and the exception found a
sixth case in the duck's own route immediately.

## Reproducing

```bash
cd projects/microduck-lab/door-elevator-etiquette

# the building: bends, clearances, zones, and the abreast non-vacuity check
../../microduck_rl/.venv/bin/python tools/check_layout.py

# the locomotion constants, re-measured on this scene
../../microduck_rl/.venv/bin/python tools/sweep_commands.py

# how long each leg really takes, with no state machine at all
../../microduck_rl/.venv/bin/python tools/measure_legs.py

# the choreography, solved against those measured arrivals
../../microduck_rl/.venv/bin/python tools/tune_phasing.py

# the headless gate — no PIL, no imageio, no GPU
../../microduck_rl/.venv/bin/python scripts/validate_etiquette.py --seconds 110

# tests, including the mutation suite
../../microduck_rl/.venv/bin/python -m pytest tests/ -q

# the video
../../microduck_rl/.venv/bin/python scripts/render_door_lift.py \
    --seconds 110 --fps 50 --out /tmp/dee-final \
    --metrics media/door-elevator-etiquette-metrics.json
```

## Layout

33 modules, none over 300 lines of code, kept there by
`test_every_module_stays_under_300_code_lines`.

| Path | What |
|---|---|
| `scripts/lobby_layout.py` | the building — one source of truth for every surface |
| `scripts/lobby_doors.py` | the three doors: schedule, open fraction, clear gap |
| `scripts/etiquette_zones.py` | thresholds, passages and the cabin, DERIVED from the apertures |
| `scripts/etiquette_path.py` | the duck's route, its legs and its aperture crossings |
| `scripts/etiquette_actors.py` | the eight scripted people |
| `scripts/etiquette_sense.py` | the boundary: world → what the duck measured |
| `scripts/etiquette_machine.py` | the state machine; no physics, no MuJoCo, fully unit-tested |
| `scripts/etiquette_control.py` | commands; never sub-onset, never `vy`, never a spin |
| `scripts/etiquette_aim.py` | who the head watches in each state; pure geometry |
| `scripts/etiquette_camera.py` | isolated-`MjData` gaze; real ray-cast visibility |
| `scripts/etiquette_tally.py` | every accumulator, with no physics in it |
| `scripts/etiquette_summary.py` | flattens a run into the summary the gates read |
| `scripts/etiquette_thresholds.py` | every acceptance threshold, data only |
| `scripts/etiquette_metrics.py` | the gates themselves |
| `scripts/rollout_etiquette.py` | the tick loop — ORDER, and nothing else |
| `scripts/validate_etiquette.py` | the headless gate |
| `tools/check_layout.py` | proves the building is walkable and the gates non-vacuous |
| `tools/measure_legs.py` | how long each leg really takes |
| `tools/tune_phasing.py` | solves the choreography against those measurements |
| `tools/probe_framing.py` | scores the wide camera by replaying the real trace |

Built on the runtime, contact geometry and camera isolation validated in
[`lead-me-somewhere/`](../lead-me-somewhere/), which is unmodified.

## Limitations

- **Simulation only.** No hardware validation.
- **The doors are scripted.** The robot did not call the lift, did not press a
  button, and cannot hold a door open for somebody.
- **Identity is a MuJoCo body id**, not perception. There is no
  re-identification: if the guardian were swapped for another person mid-route
  the duck would not notice.
- **The people are kinematic constructions**, not walking policies. They cannot
  trip, cannot choose a different route, and cannot refuse.
- **The lift is a goods lift.** A through-car is what makes the route walkable
  for a robot that cannot turn in place; a passenger cabin with one door is not
  a scenario this policy can perform.
- **The cabin walls are cutaway height** so the spectator camera can see in. The
  duck's own camera is occluded by them exactly as by full-height walls.
- **The line-of-sight conditioning barely bites in this run.** 612 of 1242
  monitoring ticks had line of sight, and the duck was visible in **100 %** of
  them. The excluded ticks are ones where a jamb or a body was genuinely in the
  way; the conditioning is doing real work here, but the raw figure is not far
  behind and should not be read as if the gate were straining.
