# dynamic-slalom

The duck crosses a busy depot floor to a visible destination, choosing which
side to pass each moving obstacle on — and stopping when neither side is safe.

Five crossing encounters are resolved in one continuous 92 s run: a pedestrian,
a pushed cart, two carried boxes and a fast walker, each predicted, negotiated
and passed. The sides alternate **right, left, right, left, right** — not
because the planner was told to alternate, but because the traffic comes from
alternating sides and each corridor is scored on its own merits. Three of those
five begin with a full stop, because at the moment of decision *neither* corridor
was predicted safe.

The duck reaches the arrival band at **79.98 s** and stands in it. That is the
point: this behavior is about **getting somewhere**, not about avoiding contact.

## Demo

`media/dynamic-slalom.mp4` — 960×640, 50 fps, 92 s.

## What actually happened, measured

| | |
|---|---|
| Path walked | **11.50 m** over 71.3 s of walking |
| Net progress | **7.48 m**, from `(-4.05, 0.00)` to `(3.43, -0.02)` |
| Goal | reached **79.98 s**, stood in the band **12.0 s**, closest **0.288 m** to its centre |
| Encounters resolved | **5**, in 5 full `THREAT → CHOOSE/WAIT → PASS → REPLAN` cycles |
| Pass sides | **right, left, right, left, right** |
| Waits (neither side safe) | **3** — 3.60 s, 1.90 s, 2.76 s |
| Replans after a pass | **5 of 5** |
| Min clearance to any body | **+0.590 m** (to `mara`) |
| Min clearance to any static body | **+0.314 m** (to `obs_cone_mid`) |
| Contacts / falls | **0 / 0** |
| Trunk height | never below **0.1130 m**, final **0.1163 m** |

### The five decisions

Every choice records the corridor it **took**, the corridor it **refused**, and
both predicted clearances — so a decision can be checked rather than believed.

| # | Body | Kind | Chose | Predicted | Rejected | Waited | **Measured** |
|---|---|---|---|---|---|---|---|
| 0 | `mara` | pedestrian | **right** | +0.166 m | left +0.019 m | 3.60 s | **+0.590 m** |
| 1 | `tobin` | rolling cart | **left** | +0.166 m | right −0.000 m | 1.90 s | **+1.052 m** |
| 2 | `ines` | carried box | **right** | +0.240 m | left +0.232 m | — | **+0.947 m** |
| 3 | `dev` | rolling cart | **left** | +0.161 m | right −0.163 m | 2.76 s | **+0.978 m** |
| 4 | `karl` | pedestrian | **right** | +0.505 m | left +0.434 m | — | **+0.903 m** |

**Every prediction conservatively bracketed what happened.** The planner
promised between +0.161 m and +0.505 m; the measured closest approaches came out
between +0.590 m and +1.052 m, so it under-promised every time by +0.26 m to
+0.88 m. That is the direction a safety prediction has to err in, and it is a
real test — the predictor is a deliberately naive constant-velocity model, and
the bodies walk filleted routes it genuinely mispredicts.

## The measurement this behavior is built on: there is no strafe

A lateral `vy` command on this policy is a yaw disturbance wearing a strafe's
clothes. **The duck cannot step sideways.** Every pass is a *turning path* —
turn out, run, turn back — and that single fact shaped the course, the planner
and the clock.

MEASURED with `tools/sweep_commands.py --what lateral` at `vx = 0.34`:

| `wz` | turn-out | `dx` | `dy` | total |
|---|---|---|---|---|
| ±0.58 | 1.2 s | 0.42 m | ±0.10 m | 3.4 s |
| ±0.58 | 1.8 s | 0.54 m | ±0.21 m | 4.6 s |
| ±0.58 | 2.4 s | 0.63 m | ±0.34 m | 5.8 s |

**A 0.34 m sidestep costs 0.64 m of course and 5.8 s of video.**

That open-loop figure is a *round trip*, though, and the planner needs something
else: how fast does the duck converge onto a line it is *pursuing*?
`tools/measure_pursuit.py` runs the real controller against the real policy:

| offset | cruise | careful |
|---|---|---|
| 0.26 m | 3.56 s / 0.066 m/s | 5.02 s / 0.051 m/s |
| 0.38 m | 4.78 s / 0.079 m/s | 6.24 s / 0.058 m/s |
| 0.50 m | 5.92 s / 0.082 m/s | 7.76 s / 0.057 m/s |

Passes run at the careful command and the planner must assume the worst sign, so
**0.0475 m/s** is the figure it plans with.

The evidence that the duck really moved sideways, with no strafe available:
**11.50 m of path against 7.48 m of net** — 4.02 m of excess — a lane offset
spanning **−0.262 m to +0.316 m** on both hands, and **2295°** of accumulated
yaw, with `max |vy| = 0.0` over all 4600 control ticks.

### The rest of the locomotion contract, measured on this scene

- **Gait onset is a cliff at `vx = 0.24`**: 0.22 → 0.009 m in 6 s (no gait at
  all), 0.24 → 0.524 m. There is nothing between zero and a walk, which is why
  **waiting is a state, not a speed**.
- **Cruise** 0.129 m/s at `vx = 0.34`; **careful** 0.097 m/s at 0.26.
- **Yaw is asymmetric and biased right**: at `vx = 0.34`, `wz = −0.10` gives
  −6.7 °/s but `+0.10` gives only +1.0 °/s. Each sign carries its own gain,
  ceiling and dead band.
- **Turning in place is unavailable**: at most 1.4 °/s across the whole command
  range at `vx = 0`.
- **Exact zero really is still**: 0.0006 m of drift over 10 s from a standstill.

## How the decision is made

`slalom_plan` predicts every tracked body forward at constant velocity over a
**10.5 s horizon**, scores **six candidate corridors** (0.26 / 0.38 / 0.50 m on
each hand) against that predicted occupancy, and returns the survivor with the
greatest worst-case clearance — or `wait` when nothing survives on either hand.

Three things disqualify a corridor, and all three are recorded so a refusal can
be explained:

- **unsafe** — worst predicted clearance below the planner's own 0.16 m bar;
- **static** — the lane comes inside 0.20 m of a crate, pallet or cone;
- **truncated** — *the worst moment is the last horizon sample*, so the conflict
  is still developing when the prediction stops looking. See the scars below.

Both hands are always scored, even when the first one checked is fine, so every
decision record carries the rejected side's number beside the chosen one.

## What the duck is NOT told

The duck never reads the choreography. It measures every body's position through
the same per-tick world state its contact probe uses, **finite-differences its
own two most recent measurements** to get velocity, and sees bodies through the
real head camera. `tests/test_rollout_and_hygiene.py` parses the import graph
with `ast` and fails if `slalom_plan`, `slalom_machine` or `slalom_control` ever
imports `slalom_actors`, so "the duck did not know" is structural rather than an
honour-system claim.

## Six bugs this behavior found, and what each one taught

Each is now a regression test and a comment at the site of the fix.

1. **The reachability veto double-counted.** `duck_at` already ramps the lateral
   offset in at the measured rate, so an unreachable corridor already scores
   badly. Vetoing on it *as well* made the planner answer "wait" to a single
   body on an empty floor. → Report it, don't veto on it.

2. **Threat chatter: ten "passes" for five crossings.** After resolving an
   encounter the duck immediately re-detected the same receding body and opened
   another pass on it, destroying alternation. → A resolved body is ignored as a
   threat for 9 s, derived from the slowest crosser's own clearing time.

3. **A corridor line that receded forever.** The pursuit target was rebuilt from
   the duck's *current* pose each tick, so the "corridor" stayed 0.26 m to the
   side of wherever the duck was — a line it could never reach, and `CHOOSE_RIGHT`
   ran into its ceiling. → A `Corridor` now stores a **fixed world line**.

4. **Predictions were optimistic.** 0.630 m predicted against 0.249 m measured
   for `mara`; 0.817 m against 0.252 m for `dev`. A predicted gap between
   *centres* is not a surface clearance. → Subtract the duck's own radius and a
   0.12 m slop term. Bracketing went from 1/6 to 5/5 conservative.

5. **A pass never ended for a perpendicular crosser.** Ending on "the body is
   behind me" never fires for something crossing sideways ahead of you. → Resolve
   on **lane clearance**: the body is clear of the duck's line and still moving
   away from it.

6. **Committing on a truncated prediction — the one that caused a collision.**
   The duck engaged `mara` while she was 2.2 m south of the lane; both corridors
   bottomed out at the horizon edge, the north scored +0.332 m on that cut-off
   view, it committed left, and she walked north into it: **−0.085 m measured
   overlap**. Every bad pass in that run had its worst moment at the horizon
   edge. → A corridor whose worst moment is the last sample is *rejected* unless
   it is comfortably clear there.

And two more found by writing the tests:

7. **A gate that compared a constant to itself.** `the command carried no
   lateral term` asserted `all(x == 0.0 for x in [0.0])` and could never fail. →
   It now reads a measured per-tick maximum over all 4600 ticks.

8. **A docstring claiming a measurement nobody took.** The tracker's filter was
   described as rejecting the actors' gait bob — but the bob is written into `z`
   only, which the tracker never reads. Raw and filtered *speed* extremes are
   identical at 0.300 m/s. → Corrected to what the filter actually bounds: the
   **rate of change** of the estimate, 10.67 → 0.96 m/s² measured.

## The scenario

Seven static bodies (three crate stacks, a pallet, three cones) and seven moving
ones on a 10.0 × 5.7 m floor — **14 in total**. Up to **2 bodies were in the
duck's lane at once**, and the lane was occupied for **21.8 s** of the run.

| Actor | Kind | Encounter | Role |
|---|---|---|---|
| `mara` | pedestrian | E1 | northbound; the duck passes behind her, south |
| `tobin` | rolling cart | E2 | southbound; behind him is the north |
| `ines` | carried box | E3 | northbound |
| `dev` + `karl` | cart + pedestrian | E4 | **both hands at once** — the forced wait |
| `noor` | carried box | E5 | northbound |
| `pilar` | pedestrian | — | background, crosses the north side throughout |

All seven move (29 %–83 % of the run each). None reacts to the duck, none
yields, none stops because it arrived — deliberately, since traffic that waited
for the robot would make every claim here vacuous. The largest single-tick
heading change any of them makes is **1.15°**, because every route is filleted.

**The crossing times are solved, not chosen.** `tools/solve_leads.py` sweeps each
body's own geometry through the real planner and reports the smallest lead that
produces a decisive, correct-side, wait-free approach — and it scales with
planning radius, as one would expect:

| body | kind | planning radius | solved lead |
|---|---|---|---|
| `mara` | pedestrian | 0.26 m | 5.5 s |
| `ines` / `noor` | carried box | 0.36 m | 6.5 s |
| `tobin` | rolling cart | 0.48 m | 7.5 s |

`tools/tune_phasing.py` then solves each `cross_t` against the duck's own
**unimpeded** arrival — measured arrival minus time spent stopped. Solving
against the raw arrival *diverges*: a wait delays the duck, which pushes the
crossing later, which makes it wait longer. Measured over three iterations the
waits grew 3.3 s → 9.5 s while the crossings marched 15.2 s → 16.6 s.

## What the duck could see

Visibility is measured through the **exact camera the PiP is rendered from** —
frustum containment plus a real MuJoCo occlusion ray cast — so the picture and
the percentages agree.

- The negotiated body was visible in **99.80 %** of monitoring steps where line
  of sight existed.
- The **goal beacon** was visible in **75.91 %** of steps with line of sight to
  it, sampled at five heights through that same camera.

Gaze runs in an isolated `MjData` copied from the physics each tick and never
written back, so it cannot prop the robot up. During a `PASS` the head returns to
the goal, which is what a person does once they have committed to going round
somebody.

## Reproducing

```bash
cd projects/microduck-lab/dynamic-slalom
V=../../microduck_rl/.venv/bin/python

$V tools/build_scene.py                    # regenerate assets/scene_slalom.xml
$V scripts/validate_slalom.py --seconds 92 --json /tmp/sl.json --trace /tmp/tr.json
$V -m pytest tests/ -q                     # 124 tests
$V scripts/render_slalom.py --seconds 92 --fps 50 --out media/dynamic-slalom.mp4
```

Measurement tools, each of which produced a number quoted above:

```bash
$V tools/sweep_commands.py --what forward|yaw|ceiling|lateral|spin|zero
$V tools/measure_pursuit.py                # closed-loop lateral convergence
$V tools/solve_leads.py                    # per-body crossing leads
$V tools/tune_phasing.py --seconds 100     # crossing times vs measured progress
$V tools/probe_framing.py --trace /tmp/tr.json
```

The headless gate imports no rendering stack at all — proved by blocking `PIL`,
`imageio` and `matplotlib` in `sys.meta_path` and importing the entry point.

## Tests

**124 tests**, including **35 gate counterexamples**: each takes the summary of a
real passing run, breaks exactly one thing, and requires the named gate to go
red *and no other gate to be repaired by the same mutation*. A gate that cannot
fail is not a gate.

## Limitations, stated plainly

- **Simulation only.** No hardware validation.
- **The traffic is scripted.** Bodies walk declared routes and never react to
  the duck. This is deliberate — reactive traffic would make the avoidance
  claims vacuous — but it means nothing here demonstrates negotiation with an
  agent that also yields.
- **Identity is a semantic proxy.** Bodies are identified by MuJoCo body id
  inside a real frustum with a real occlusion ray cast, not by an RGB
  classifier. There is no perception stack.
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
- **The prediction is constant-velocity**, the simplest model that can be wrong.
  It is wrong wherever a body is turning; the bracketing gate is what shows the
  error stayed in the safe direction.

## Built on

The locomotion runtime, contact geometry and camera isolation come from
[`door-elevator-etiquette/`](../door-elevator-etiquette/), which is unmodified by
this behavior. Every locomotion constant was **re-measured on this scene** rather
than inherited — the floor, the traffic and the task are different.
