# lead-me-somewhere

**A guide.** A person asks the duck to take her somewhere. The duck acknowledges,
searches a route through a divided and populated concourse, leads her along it —
and when she stops to look at a departures board, or slows to a crawl rounding a
partition, it *notices*, stops at a safe spot, turns its head to watch her, waits
until she has actually caught up, and only then leads on. At the lifts it stops,
facing what it brought her to, and indicates arrival.

Everything below is MEASURED from the run in `media/`, by
`scripts/validate_guide.py`, which imports no rendering stack at all.

## The run

| | |
|---|---|
| Duration | 95.0 s, 4750 control steps at 50 Hz |
| Requested destination | `LIFTS`, out of `LIFTS` / `CAFE` / `HELPDESK` |
| Planned route | **7.81 m, 5 bends**, ×1.36 the straight line |
| Physical lead | **9.93 m** of path, 5.50 m net displacement |
| The person led | walked **15.26 m** along the duck's own trail |
| Lag episodes | **2**, both detected from measurement |
| Waiting | 10.4 s + 12.7 s, forward command **exactly 0.0** |
| Arrival | 0.865 m from the fixture, facing it within **0.5°** |
| Her final distance | 0.40 m — safely alongside |
| Safety | **0 contacts, 0 falls**, +0.144 m to people, +0.168 m to scenery |

**39/39 acceptance gates pass. 218/218 tests pass**, including 46 mutation
counterexamples that each prove one gate rejects its own violation.

## The video

`media/lead-me-somewhere.mp4` — H.264, **960×640 @ 50 fps, 4750 frames, 95.0 s**,
12.3 MB. A 3.4 MB derivative at identical resolution, frame rate and duration is
in `media/lead-me-somewhere-telegram.mp4`; both decode cleanly end to end.

The wide camera is **measured, not eyeballed**. `tools/probe_framing.py` replays
the real recorded trace through the renderer's own easing, distance ramp and
look-at clamp, and scores 360 candidates on whether the duck and the follower
are on screen, unoccluded in 3D against the partitions and every person as a
solid, clear of the HUD panels, and whether the camera's own eye stays out of
the scenery.

The answer was **azimuth 180°, elevation −56°**, and the elevation is the
interesting part: every candidate above 0.90 duck-visibility sits at −56°,
because the hall is divided by two 2.05 m slabs and the camera has to fly *over*
them. The first attempt inherited −27° from the promenade behavior and produced
a video in which the duck is behind a partition for much of the route. The cost
is a smaller duck — 15 px against the promenade's 33 — which is the right trade:
a small duck you can see beats a large one behind a wall.

## What is real, and what is a proxy

Stated plainly, because a guide robot demo invites more credit than it deserves.

**Real:** the walking. Every step is the stock `alpha_walking.onnx` policy
(SHA-256 `e36332…daa6c`, byte-identical to `microduck_rl`) driven at 50 Hz
through the exact `imu_ang_vel` sensor with a 61-D observation and the shipped
0.9 action scale. Nothing is trained here.

**Real:** the route. It is *searched* — A\* over a 0.09 m grid of inflated free
space, then line-of-sight shortcutted, then filleted — from the duck's own
measured pose and the crowd's measured positions and velocities, at run time,
against whichever of the three destinations was requested. A different request
produces a different walk.

**Real:** the camera. Visibility is frustum containment plus a MuJoCo ray cast
through actual scene geometry, measured through the *exact* camera the PiP is
rendered from.

**A semantic proxy:** the request. There is no speech recognition. At t = 1.6 s
a simulator event delivers one of three destination keys, and the machine
resolves it by exact dictionary lookup — no fuzzy matching, no default. What is
demonstrated is that the duck *went where it was asked* out of three real
options, not that it understood English.

**A semantic proxy:** person identity. Bodies are identified by MuJoCo body id,
not by an RGB classifier. The *geometry* that decides whether the camera can see
them is real; the label on them is given.

**Scripted:** the six other adults, and the follower's *speed*. Her stall
windows are declared in `guide_follower.STALLS` and are the scenario. What is
**not** scripted is anything the duck knows: it never reads that module. It
measures her range with the same probe it measures everyone with, and her
visibility through the real head camera, and decides from those two numbers
alone. The gate *compares* declared stalls against detected episodes rather than
conflating them — which is why "the duck noticed" is falsifiable.

**Not validated:** hardware. This is simulation only.

## The measurements that shaped the design

Constants are re-measured per scene (`tools/sweep_commands.py`), never inherited.
Three findings changed the behavior itself.

### The duck cannot turn on the spot

MEASURED at `vx = 0` over 3 s, across the whole command range:

| `wz` | rate | drift |
|---|---|---|
| −0.16 | −0.5 °/s | 0.0001 m |
| −0.30 | −1.1 °/s | 0.0008 m |
| −0.42 | −1.6 °/s | 0.0016 m |
| +0.42 | +1.4 °/s | 0.0031 m |

The best turn-in-place command yields **1.6 °/s** — squaring up to somebody 130°
behind would take 80 seconds. The first draft had a spin controller with a rate
copied from a sibling behavior; the sweep deleted it. There is no turn-in-place
command anywhere in this behavior, and `guide_states.SPIN_BEST_RATE_DPS` records
the figure so the absence stays a *finding* rather than an oversight.

Two consequences follow. **Looking back is a head action**, against the MEASURED
±170° head yaw range. And **arrival facing is solved by the route**: the planner
appends an approach waypoint on the fixture's own axis, so the duck's walking
heading as it reaches the standing point already points at the lifts. It arrives
facing the right way because of where it walked, not because it turned.

### Gait onset is a cliff, and it is at 0.24 here

MEASURED over 6 s: `vx = 0.22 → 0.009 m` (no gait at all), `vx = 0.24 → 0.527 m`.
The sibling promenade measured its onset at 0.22; this scene's is one step
higher, which is exactly why constants are not inherited.

A guide that "slowed down a little" for a lagging follower would emit 0.22, stand
still, and log a nonzero command. **So waiting is a state, not a speed.** The duck
walks or it holds exactly zero — and 10 s of exact zero MEASURES 0.0006 m of
drift and 0.0057 m of path, which is what makes "it stopped and waited" a claim
about the floor.

### A follower in the guide's footprints cannot be seen

MEASURED: with the follower walking the duck's trail exactly, she sits at
**173–180° astern** — outside the head's ±170° reach. The camera correctly could
not see her, and the first full run lost her for 1.4 s of a wait.

Two robot-side fixes were built and both worked and both were wrong: they cost
11 s per episode of walking an arc, and meant the duck was *moving* in a state
that claims it stopped. The third fix was on the scenario side, and is also what
people actually do — **somebody following a guide walks a little to one side**,
because that is how you see past them. A 0.30 m offset puts her at ~154°,
comfortably inside the head's reach, with no manoeuvre asked of the robot.
`CHECK_FOLLOWER` then became an exact-zero state too, which is a *stronger*
claim than the design started with.

## States

```
RECEIVE_DESTINATION → PLAN → LEAD ⇄ CHECK_FOLLOWER → WAIT_FOR_PERSON → RESUME
                                                                          ↓
                                          ARRIVE → INDICATE → DONE ←──────┘
```

| State | Seconds | Command | What it is |
|---|---|---|---|
| `RECEIVE_DESTINATION` | 3.00 | **0.0** | asked; acknowledging, standing still |
| `PLAN` | 1.20 | **0.0** | searching the route, still standing |
| `LEAD` | 52.12 | walking | leading along the planned route |
| `CHECK_FOLLOWER` | 1.16 | **0.0** | stopped; head coming round to find her |
| `WAIT_FOR_PERSON` | 23.14 | **0.0** | waiting where she can be watched |
| `RESUME` | 6.00 | walking | she is back; leading on |
| `ARRIVE` | 0.02 | 0.0 | at the standing point, already facing the lifts |
| `INDICATE` | 4.00 | **0.0** | head gesture: this is the place |
| `DONE` | 4.36 | **0.0** | delivered |

`ARRIVE` lasting one tick is the design, not a bug: the route's final heading
already faces the fixture, so there is nothing to turn. A test asserts it rather
than excusing it.

## The two episodes

Neither is scheduled. Both are opened by the machine from measured distance and
measured visibility, with a confirm window so a single bad frame is not an event.

| | Episode 0 | Episode 1 |
|---|---|---|
| Declared stall | "stops to look at the departures board" 17–27 s | "slows to a crawl rounding the partition" 44–55 s |
| Detected at | **25.08 s** | **48.10 s** |
| Distance then | 1.60 m (threshold 1.45 m) | 1.59 m |
| Waited | **10.42 s** | **12.72 s** |
| Max command while waiting | **0.0** | **0.0** |
| Drift while waiting | 0.023 m | 0.029 m |
| She closed | 0.737 m | 0.736 m |
| Resumed at | 0.85 m, visible, sustained 1.0 s | 0.85 m, visible, sustained 1.0 s |
| She was visible | 90.4 % of LOS ticks | 100 % |

Longest continuous interval beyond the 3.20 m safety maximum: **0.0 s**.

## The route, and why the hall is shaped like this

Two 2.05 m full-height slabs divide the concourse, each sealed against an
opposite wall, so any route must pass *south* of the partition, climb the
corridor between them, and cross back *north* of the screen. The straight line
to the lifts is blocked by `hall_screen`; the searched route is ×1.36 longer and
carries 5 bends.

The crowd term is required to bite and is graded: **607 grid cells** were free of
every static body and refused *only* because a person's swept tube covered them,
attributed to six named adults. A planner that "avoids the crowd" in an empty
corridor has proved nothing.

Three passage widths were measured the hard way. Corridors of 1.37 m and 1.50 m
are geometrically fine for a 0.26 m robot and were **sealed by any one of the six
adults** — a coordinate-descent sweep over actor start offsets found *zero*
feasible planning instants, which is the correct answer to a badly sized hall
rather than a scheduling problem to tune away. All passages are now ≥ 2.15 m.

When a person does stand in the only way through, the planner retries at reduced
crowd margin and **reports which tier it needed**. The static inflation never
changes: clearance to walls and partitions is not negotiable.

## Reproducing

```bash
cd projects/microduck-lab/lead-me-somewhere

# geometry sanity: sealing, reachability, route distinctness
../../microduck_rl/.venv/bin/python tools/check_layout.py

# the locomotion constants, re-measured on this scene
../../microduck_rl/.venv/bin/python tools/sweep_commands.py

# the headless gate — no PIL, no imageio, no GPU
../../microduck_rl/.venv/bin/python scripts/validate_guide.py --seconds 95

# a different request produces a different walk
../../microduck_rl/.venv/bin/python scripts/validate_guide.py --destination CAFE

# tests, including the mutation suite
../../microduck_rl/.venv/bin/python -m pytest tests/ -q

# the video
../../microduck_rl/.venv/bin/python scripts/render_lead_me.py \
    --seconds 95 --fps 50 --out /tmp/lms-final \
    --metrics media/lead-me-somewhere-metrics.json
```

## Layout

29 modules, none over 300 lines of code — the three that outgrew it
(`guide_metrics`, `guide_planner`, `rollout_guide`) were split, and
`test_every_module_stays_under_300_code_lines` keeps them that way.

| Path | What |
|---|---|
| `scripts/guide_layout.py` | the concourse and the three destinations — one source of truth |
| `scripts/guide_planning_space.py` | inflation figures, swept crowd tubes, the `Plan` record |
| `scripts/guide_planner.py` | A\* search, line-of-sight shortcutting, fillet choice |
| `scripts/guide_machine.py` | the state machine; no physics, no MuJoCo, fully unit-tested |
| `scripts/guide_control.py` | commands; never sub-onset, never `vy`, never a spin |
| `scripts/guide_aim.py` | what the duck aims at in each state; pure geometry |
| `scripts/guide_follower.py` | she walks the duck's own trail — what makes "led" falsifiable |
| `scripts/guide_camera.py` | isolated-`MjData` gaze and gesture; real ray-cast visibility |
| `scripts/guide_tally.py` | every accumulator, with no physics in it |
| `scripts/guide_summary.py` | flattens a run into the summary the gates read |
| `scripts/guide_thresholds.py` | every acceptance threshold, data only |
| `scripts/guide_metrics.py` | the gates themselves |
| `scripts/rollout_guide.py` | the tick loop — ORDER, and nothing else |
| `scripts/validate_guide.py` | the headless gate |
| `tools/check_layout.py` | proves the hall is sealed, reachable and distinct |
| `tools/sweep_commands.py` | re-measures the locomotion constants on this scene |
| `tools/probe_framing.py` | scores the wide camera by replaying the real trace |

Built on the runtime, contact geometry and camera isolation validated in
[`walk-beside-me/`](../walk-beside-me/), which is unmodified.

## Limitations

- **Simulation only.** No hardware validation.
- **The request is a simulator event**, not speech.
- **Identity is a MuJoCo body id**, not perception. There is no re-identification:
  if she were swapped for another person mid-route the duck would not notice.
- **The line-of-sight conditioning never fires in this run.** The planar occluder
  test found no full-height body inside the 0.6–1.6 m segment to the follower at
  any of the 4750 ticks, so the LOS-conditioned visibility percentage is
  *identical* to the raw one. That is the strict direction — the duck is held
  responsible for every monitoring tick — but the conditioning is earning
  nothing here and should not be read as if it were. The real occlusions come
  from crowd actors crossing the sightline, and those ticks are graded.
- **The follower's motion is a kinematic construction**, not a walking policy.
  She cannot trip, cannot choose to take a different route, and cannot refuse.
- **The crowd margin degrades under pressure.** The static margin does not.
