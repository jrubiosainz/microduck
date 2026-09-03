# patrol-and-investigate

The duck walks a five-checkpoint security round of an indoor facility, stopping
and scanning at each post. Twice it finds something it cannot explain, breaks
off, walks up to it — stopping at a safe distance — looks at it from several
angles, records what it decided and why, walks **back to the exact point it
left the route**, and carries on. A third thing it finds, it explains away
without taking a step toward it.

The whole round completes in one continuous 148 s run: **five checkpoints in
the declared order, three classifications, two investigations, zero contacts,
and the restricted zone never entered.**

## Demo

`media/patrol-and-investigate.mp4` — 960×640, 50 fps, 148 s.

## What actually happened, measured

| | |
|---|---|
| Checkpoints | **5/5, in the declared order, once each** — `dock-gate → east-aisle → north-bay → server-door → west-stair` |
| Path walked | **15.95 m** over 90.6 s of walking, on a 5.16 m circuit |
| Stopped ON each post | worst arrival error **0.199 m**; **0.000 m** of path during any stop-and-scan |
| Classifications | **3, all correct**: `crate`→suspicious, `visitor`→intrusion, `trolley`→benign |
| Investigations | **2**, each with a full `DETECT→…→RESUME` cycle |
| Route memory | broke off toward `east-aisle` and `north-bay`; **resumed toward the same two** |
| Return accuracy | back within **0.158 m** and **0.160 m** of the point it left |
| Restricted zone | **never entered** — closest **+0.274 m**, 0 breaches in 7400 ticks |
| Min clearance to any body | **+0.517 m** (to the crate it was observing) |
| Min clearance to any fixture | **+0.116 m** (to a zone stanchion) |
| Contacts / falls | **0 / 0** |
| Camera active on target | **99.61 %** of 7400 ticks |
| Trunk height | never below **0.1119 m**, final **0.1163 m** |
| Home | reached **140.1 s**, stood there **7.9 s**, **0.194 m** from the pad centre |

### The five checkpoints

Every visit records the stop, the sweep the head actually travelled, and what
the camera resolved — so "it stopped and looked" is four measurements rather
than a caption.

| # | Checkpoint | Stopped | Head swept | Camera resolved | Result |
|---|---|---|---|---|---|
| 1 | `dock-gate` | 1.40 s | — | `trolley` | **DETECT** → dismissed |
| 2 | `east-aisle` | 1.42 s | 94.3° | `crate` | **DETECT** → investigated |
| 3 | `north-bay` | 1.42 s | 44.3° | `visitor` | **DETECT** → investigated |
| 4 | `server-door` | 1.40 s | **138.1°** | — | CLEAR |
| 5 | `west-stair` | 1.42 s | **138.2°** | — | CLEAR |

The three short sweeps are short **because the scan was cut off by a
detection**, which is the correct behaviour: a guard that finished its sweep
before reacting to something it had already identified would be following a
script rather than watching. The two scans that ran to completion swept 138°.

### The three things it had to tell apart

The scenario is a classification problem wearing a patrol's clothes. Each
verdict is a decision over features the duck measured itself, and each carries
the rule it fired on.

| Target | Verdict | Confidence | The rule it fired on |
|---|---|---|---|
| `trolley` | **benign** | 0.70 | standing in the designated stow area `obs_pallet_s` |
| `crate` | **suspicious** | 0.98 | stationary 18.7 s outside any stow area, nobody within 0.90 m |
| `visitor` | **intrusion** | 0.99 | inside the marked restricted zone 23.4 s, past the 2.5 s dwell |

**The trolley is the case that matters.** It appears during the patrol, at a
similar range and a similar size to the crate, and it is exactly as unexpected.
What separates it is two rules the duck can check from its own measurements: it
stands in a designated stow area, and a member of staff is beside it. A
threshold-only detector gets it wrong. The duck **dismissed it without taking a
single step toward it** — the patrol was never interrupted for it — and the
dismissal is logged exactly as an escalation is, which is what makes "it
explicitly dismissed the distractor" checkable rather than an absence of
evidence.

### The two investigations

| | `crate` | `visitor` |
|---|---|---|
| Detected at | 21.2 s, 1.422 m | 58.1 s, 2.306 m |
| Broke off toward | `east-aisle` | `north-bay` |
| Approach | 1.422 → 1.066 m (**+0.357 m**) | 2.306 → 1.126 m (**+1.180 m**) |
| Walked | **0.702 m** | **1.801 m** |
| Standoff planned / measured | 0.60 m → **0.517 m** | 0.60 m → **0.609 m** |
| Standoff candidates rejected | 3 of 10 | 3 of 10 |
| Closest measured clearance | **0.517 m** | **0.609 m** |
| Angles held | 3 × 2.2 s, target in frame **100 %** | 3 × 2.2 s, **100 %** || Command during observation | **exactly 0.00** | **exactly 0.00** |
| Resumed toward | **`east-aisle`** | **`north-bay`** |
| Back within | **0.158 m** | **0.160 m** |

Both approaches ended inside the required **0.45–0.75 m** safe observation
band, measured as a real surface clearance against the real geoms every control
tick — not derived from the standoff point that was planned.

## The measurement this behavior is built on: there is no turn in place

MEASURED with `tools/sweep_commands.py --what spin`, across the whole command
range at `vx = 0`:

| `wz` | drift | yaw rate |
|---|---|---|
| ±0.16 | 0.0001–0.0017 m | 0.5–0.6 °/s |
| ±0.34 | 0.0011–0.0030 m | 1.2 °/s |
| ±0.42 | 0.0016–0.0031 m | **1.6 °/s** |

**The duck cannot pivot.** It cannot turn to face a checkpoint, cannot spin to
scan a room, and cannot turn round to go back. Three consequences shape
everything else:

1. **The patrol circuit is a hexagon**, whose 60° corners the duck carves while
   walking. Measured corners: `[60.00, 60.00, 60.00, 60.00, 60.00]°`.
2. **Every scan is a HEAD sweep.** The head yaw joint has a measured ±170°
   range, so a stopped duck can still sweep its camera across the facility —
   and the arc reported is the arc the head *actually travelled*, accumulated
   from the pose it reached, not the amplitude that was commanded.
3. **Returning to an interrupted checkpoint is a real walk back**, not a pivot,
   which is what makes route memory a physical claim.

### The rest of the locomotion contract, measured on this scene

- **Gait onset is a cliff at `vx = 0.24`**: 0.22 → 0.009 m in 6 s (no gait at
  all), 0.24 → 0.508 m. There is nothing between zero and a walk, which is why
  **a checkpoint stop is a STATE, not a speed**. A robot that "slowed to a
  crawl" at each post would emit 0.22, stand perfectly still, and log a nonzero
  command — the appearance of care with none of the physics.
- **Patrol cruise** 0.146 m/s at `vx = 0.38`; **approach** 0.096 m/s at 0.26.
- **`vx = 0.42` was rejected on its own measurement**: there the open-loop yaw
  drift *reverses sign*, from −20.8° to +8.6° over 6 s. A gait whose heading
  bias flips direction with speed is one the heading loop has to fight rather
  than trim.
- **Yaw is asymmetric and biased right**: at `vx = 0.34`, `wz = −0.10` gives
  −8.7 °/s but `+0.10` gives only +0.7 °/s. Each sign carries its own gain,
  ceiling and dead band.
- **The circuit runs counter-clockwise on purpose**, so every corner is a LEFT
  turn into the *weak* sign and against the policy's own bias. A clockwise loop
  would have the policy's drift doing the turning, and "the duck walked its
  circuit" would be partly a fact about the policy rather than about the
  controller.
- **Exact zero really is still**: 0.0006 m of drift and 0.0054 m of path over
  10 s. The worst zero-command episode in the whole run accumulated **0.033 m
  of path and 0.009 m net**, and the controller emitted a lateral term of
  **exactly 0.0** on all 7400 control ticks.

## What the duck is NOT told

The duck never reads the choreography. It measures every body's position
through the same per-tick world state its contact probe uses, sees them through
the real head camera, and decides what they are from geometry it measured
itself. `tests/test_rollout_and_hygiene.py` parses the import graph with `ast`
and fails if `patrol_machine`, `patrol_branch`, `patrol_detect`,
`patrol_control` or `patrol_plan` ever imports `patrol_actors` — so "the duck
did not know" is structural rather than an honour-system claim.

**Detection is gated on the camera.** A body can become a candidate only while
it is inside the real frustum with a real MuJoCo occlusion ray cast, and only
after 0.40 s of confirmed visibility. Measured: `crate` was in the camera gate
for 1204 ticks before being acted on, `visitor` 1769, `trolley` 875.

## Six bugs this behavior found, and what each one taught

Each is now a regression test and a comment at the site of the fix.

1. **The detection subject was not latched.** `DETECT` is entered on tick *N*
   but the subject is assigned when the handler runs on *N+1* — and if the body
   left the frustum on exactly that tick, the subject was never assigned. The
   trolley was detected, dismissed with an **empty verdict**, never settled, and
   re-detected **three times**; `dock-gate` was visited **four times**. → The
   candidate is latched in the module that owns the world, at the moment the
   detection is made.

2. **A dismissal did not complete its checkpoint.** Only `CLEAR` and `RESUME`
   advanced the patrol, so a checkpoint where something was found and explained
   never advanced the target. → A dismissal completes the checkpoint too: the
   duck stopped, scanned, found something, explained it, and moved on.

3. **The zone interlock was a trap.** Refusing to advance on *heading* is
   permanent on a robot that cannot turn in place, because the yaw it needs
   comes from walking. Measured: after observing the intruder the duck stood at
   (−1.054, 0.970) facing the annex and did not move for the whole 40 s
   ceiling. → The refusal is about the *direction of travel*: walking away from
   a restricted area is always allowed.

4. **Two different zone margins were conflated.** The planner's generous margin
   was borrowed by the interlock, which governs whether the robot may move at
   all. The next 0.34 m crossed the *grown* rectangle by 7 mm and the duck
   froze — while the turn it needed would have stayed 0.298 m clear of the real
   rectangle. → The interlock forbids exactly what the rule forbids: the duck's
   own footprint inside the marked area.

5. **Stationary time was measured as time spent watching.** The crate had stood
   untouched since it appeared but was only in frustum for 5.58 s of one scan,
   so it missed a 6 s bar it had physically satisfied for far longer. → It is
   *elapsed time since it was first seen in this place*. A robot that glances
   twice, ten seconds apart, and sees the same box in the same place has
   evidence it stood there for ten seconds.

6. **A phase ceiling shorter than the walk it bounded.** A patrol leg resumed
   from an investigation standoff is much longer than a plain 0.86 m circuit
   leg, because the duck rejoins its route from wherever it was standing. The
   34 s ceiling tripped at 154.68 s on an otherwise healthy run. → Every ceiling
   is derived from the distance its state must cover at a measured speed.

And two more found by writing the tests and reading the HUD:

7. **Two metrics that could not fail.** "The camera was active" was true by
   construction (the head is always aimed *somewhere*), and "it reached home"
   fired at t = 0.02 s because the duck *starts* on the guard-post pad. → Camera
   activity is now whether the aim point is genuinely inside the frustum, which
   the rate-limited head really does fail at corners (99.61 %, not 100 %); home
   is only measured once the circuit is complete.

8. **The HUD contradicted the gate.** The safety panel plotted a
   centre-to-centre range against a *surface* standoff band — quantities that
   differ by both bodies' radii, about 0.29 m — so the tick read outside a
   window the duck was correctly inside. → The bar is drawn in surface
   clearance, the same units the gate grades.

## The scenario

An 11-fixture facility on a 5.9 × 4.4 m floor: a central racking island the
circuit runs round, two shelf stacks, two structural columns, a recycling bin, a
designated stow pallet and four stanchions marking the restricted annex. Six
bodies — three staff, one intruder, two objects.

| Body | Kind | Role |
|---|---|---|
| `rosa`, `nadia` | staff | work the north and west aisles throughout |
| `emil` | staff | crosses the south-east and **stands beside the trolley** |
| `visitor` | intruder | walks into the marked annex and stays: **115.6 s inside** |
| `crate` | object | appears at 10 s in the open north-east bay |
| `trolley` | object | appears at 3 s **on the stow pallet** |

Nobody reacts to the duck, nobody yields, and nobody stops because it arrived —
deliberately, since staff who got out of the way would make every patrol claim
vacuous. The largest single-tick heading change any of them makes is **1.66°**,
because every route is filleted.

**The layout is solved, not drawn.** `tools/check_layout.py` runs 22 geometric
checks and refuses a layout where an approach would not be a real walk, an
anomaly would sit outside the camera gate, or a body would stand inside a
fixture. It caught seven real scenario bugs by measurement, all invisible by
eye: approaches only 0.04 m and −0.06 m long, a staff member 2.17 m from the
trolley he was supposed to be attending, a route bringing another within
0.449 m of the crate that had to look unattended, and three routes clipping
fixtures by 13–47 mm.

## What the duck could see

Visibility is measured through the **exact camera the PiP is rendered from** —
frustum containment plus a real MuJoCo occlusion ray cast — so the picture and
the percentages agree.

- The investigated body was visible in **100 %** of the 2090 monitoring steps
  where line of sight existed.
- The camera's aim point was genuinely inside the frustum in **99.61 %** of all
  7400 ticks. It is not 100 % because the gaze is rate-limited to the measured
  26 °/s, so the head really does lag when the duck carves a 60° corner.

**Occlusion is real on this scene, unlike its siblings.** The central rack is
0.72 m tall against a head camera at about 0.20 m, so bodies behind it are
genuinely hidden and `occluder_between` returns a real name on the real run —
the visibility gate is conditioned on something that actually happens.

Gaze runs in an isolated `MjData` copied from the physics each tick and never
written back, so it cannot prop the robot up.

## Reproducing

```bash
cd projects/microduck-lab/patrol-and-investigate
V=../../microduck_rl/.venv/bin/python

$V tools/build_scene.py                      # regenerate assets/scene_patrol.xml
$V tools/check_layout.py                     # 22 geometric checks on the scenario
$V scripts/validate_patrol.py --seconds 148 --json /tmp/pt.json --trace /tmp/pt_trace.json
PATROL_SUMMARY=/tmp/pt.json PATROL_TRACE=/tmp/pt_trace.json $V -m pytest tests/ -q
$V scripts/render_patrol.py --seconds 148 --fps 50 --out media/patrol-and-investigate.mp4
```

Measurement tools, each of which produced a number quoted above:

```bash
$V tools/sweep_commands.py --what forward|yaw|ceiling|spin|zero
$V tools/probe_framing.py --trace /tmp/pt_trace.json   # the wide camera's framing
```

The headless gate imports no rendering stack at all — proved by blocking `PIL`,
`imageio` and `matplotlib` in `sys.meta_path` and importing the entry point.

## Tests

**174 tests**, including **55 gate counterexamples**: each takes the summary of
a real passing run, breaks exactly one thing, and requires the named gate to go
red *and no other gate to be repaired by the same mutation*. A meta-test parses
that file with `ast`, collects every needle asserted, and fails if any of the
**41** acceptance gates has no counterexample — so adding a gate without one
fails in CI rather than shipping.

## Limitations, stated plainly

- **Simulation only.** No hardware validation.
- **The population is scripted.** Bodies walk declared routes and never react
  to the duck. This is deliberate — reactive staff would make the patrol claims
  vacuous — but nothing here demonstrates interaction with a person who
  responds.
- **Identity and classification are a semantic proxy.** Bodies are identified
  by MuJoCo body id inside a real frustum with a real occlusion ray cast, not by
  an RGB classifier, and the verdicts are rules over measured geometry rather
  than learned perception. **The reported confidence is a rule-margin proxy** —
  how far the evidence sits past each rule's own threshold — not a probability,
  and it is labelled as such in the HUD, the metrics and here.
- **"Stationary for N seconds" assumes the object did not leave and return
  between two observations.** The conservative alternative — counting only
  observed time — was measured and rejected because it makes the verdict depend
  on how long the patrol happened to look rather than on the object.
- **Neither zone guard was load-bearing on this run.** The behavior guards the
  restricted zone twice independently — the standoff planner prunes points
  inside it, and the interlock refuses steps into it. Disabling each in turn and
  re-running changed the closest approach only from +0.274 m to +0.214 m, with
  no breach either way: on *this* geometry the chosen standoff was comfortably
  outside regardless. The guards are defence in depth against a standoff placed
  badly, and this run did not exercise them.
- **The actors are non-colliding kinematic proxies.** They cannot push the duck,
  so "zero contacts" from the physics engine would be vacuous. Every clearance
  quoted here is an *analytic surface distance* measured every control tick —
  `mj_geomDistance` is never called, because this simulator has been measured
  returning exact zeros for mesh-versus-primitive pairs more than a metre apart.
- **The gaze layer is rendering-only** and never feeds back into walking physics.
- **The policy is the stock walking policy**, unmodified: SHA-256
  `e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c`, driven at
  50 Hz through a 61-D observation with the shipped 0.9 action scale and the
  exact `imu_ang_vel` sensor. Nothing was trained here.

## Built on

The locomotion runtime, contact geometry and camera isolation come from
[`dynamic-slalom/`](../dynamic-slalom/), which is unmodified by this behavior.
Every locomotion constant was **re-measured on this scene** rather than
inherited — the floor, the route and the task are different, and the
turn-in-place measurement that shapes this behavior was not one the slalom
needed.
