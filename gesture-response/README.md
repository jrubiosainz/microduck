# gesture-response

The duck stands on its pad and watches a training area with five adults in it.
One of them is its instructor. She gives it six hand commands in sequence — come
here, stop, point left, point right, back up, goodbye — and it carries out every
one of them, **in order, from her alone.**

Halfway through the first command she tells it to stop while it is still
walking, and it stops. Between two of the commands she gives it a half-lifted,
ambiguous gesture from the same mark, in full view, and it refuses. Three of the
other adults make full, well-formed gestures from the same vocabulary while the
duck is watching, and it ignores every one of them.

The whole session runs in one continuous 87 s rollout: **six commands accepted
in the exact required order, one ambiguous gesture refused, one stranger's
sustained command ignored, zero contacts, zero falls.**

## Demo

`media/gesture-response.mp4` — 960×640, 50 fps, 87.2 s (4360 frames).

`media/gesture-response-contact-sheet.jpg` — one still per act of the session.
`media/gesture-response-metrics.json` — all 48 gates, graded on the rendered run.

## What actually happened, measured

| | |
|---|---|
| Commands accepted | **6/6, in the declared order** — `COME → STOP → TURN_LEFT → TURN_RIGHT → BACK_UP → WAVE` |
| Every command from | **`mira` only** — 0 episodes opened on anybody else |
| Ambiguous gesture | **refused**, while she was **100 %** visible with her arm **100 %** readable |
| Stranger's command | `teo` gave a full COME, readable and sustained **2.44 s** — **ignored** |
| Camera on the instructor | **100.00 %** of monitoring ticks |
| Contacts / falls | **0 / 0** |
| Closest approach to anybody | **+0.5971 m** (to `mira`, at 61.68 s) |
| Closest approach to scenery | **+0.6277 m** (`obs_cone_w`) |
| Trunk height | never below **0.1131 m**, final **0.1163 m** |
| Lateral command | **exactly 0.0** on all 4360 control ticks |
| Sub-gait commands | **0 ticks** |
| Phase ceilings fired | **0** |

### The six commands, and what each one physically did

Every row is graded on a quantity measured from physics, never on a state name
or a command register.

| # | Gesture | Command | What the duck actually did |
|---|---|---|---|
| 1 | one arm beckoning | `COME` | closed **2.899 → 1.699 m** (**−1.200 m**), walking **1.8025 m** — then interrupted |
| 2 | open palm, held still | `STOP` | command **0.260 → exactly 0.000 in 0 ticks**, held **1.98 s**, drifted **2.8 mm** |
| 3 | straight arm to her right | `TURN_LEFT` | trunk yaw **+64.6°**, a walked arc of **0.5395 m** |
| 4 | straight arm to her left | `TURN_RIGHT` | trunk yaw **−64.5°**, a walked arc of **0.5116 m** |
| 5 | both arms pushing away | `BACK_UP` | **0.342 m backward along the pre-action heading**, at the measured reverse onset `−0.32` |
| 6 | high arm sweeping | `WAVE` | acknowledged and stood down |

Total path walked: **3.43 m**, of which **3.28 m** under a walking command.

**The pointing commands are mirrored on purpose.** She faces the duck, so her
raised *right* arm is on the duck's *left*. That mapping happens in exactly one
place — `gest_gesture.command_for` — and it is why the two turns are opposite
**real heading changes** rather than opposite labels.

### The STOP is the command this behavior is really about

A stop that can only be given to a robot already standing still is a formality.
So the scenario gives it **while the duck is walking**, and the duck has to be
able to hear it while walking.

- The command register read **0.260** on the tick before the stop was confirmed,
  so it interrupted a real gait rather than a state.
- It reached **exactly `(0.000, 0.000, 0.000)` on the very next tick** — no ramp,
  no filter, no decay — and the gate measures that by tick index.
- It then held **1.98 s** below the measured settled speed, accumulating
  **2.8 mm** of drift.

The interruption is logged from both sides: the `COME` episode records
`interrupted_by: STOP`, and the `STOP` episode records `interrupts: COME` at
**20.08 s**.

While the duck is walking, gesture reading is **narrowed rather than
suspended**: only `STOP` may be accepted, and it still has to pass the same
confirm gate as any other command — same locked person, arm fully readable,
sustained for the full window. The set of acceptable commands shrinks to one;
the standard of proof never moves.

## What the duck is NOT told

The duck never reads the choreography. It measures every person's position
through the same per-tick world state its contact probe uses, sees them through
the real head camera, and reads their arms from the world positions of **real
articulated keypoint bodies**. `tests/test_rollout_and_hygiene.py` parses the
import graph with `ast` and fails if `gest_machine`, `gest_detect`,
`gest_gesture`, `gest_pose`, `gest_control`, `gest_states` or `gest_episode`
ever imports `gest_actors` or `gest_script` — so "the duck did not know" is
structural rather than an honour-system claim.

**The arms are real kinematics, not decoration.** Each arm is three nested
bodies on three hinge joints. Writing `qpos` and calling `mj_forward` means
MuJoCo computes the shoulder, elbow and hand world positions — and those are the
*same* positions the camera ray-casts against and the *same* positions the
classifier measures its features from. A sign error in the animation therefore
shows up as a gesture that fails to classify, never as one that classifies
wrongly.

**The arm gate is strictly harder than the body gate.** A person can be
comfortably in frame with their raised hand outside it — the duck is 0.20 m tall
and stands close enough to read a gesture, so an adult's raised arm sits near the
top of the frustum exactly when their torso is centred. `arm_readable` therefore
tests all three keypoints of one arm individually, each with its own frustum
containment and its own occlusion ray cast.

## Seven bugs this behavior found, and what each one taught

Each is now a regression test and a comment at the site of the fix.

1. **Two commands silently mapped to nothing.** `gest_arm` names the pointing
   animations `POINT_L_ARM`/`POINT_R_ARM`; `gest_gesture` names the templates
   `POINT_LEFT_ARM`/`POINT_RIGHT_ARM`. Feeding an animation name to
   `command_for` returned `""` for exactly those two, so the required sequence
   was `('COME','STOP','','','BACK_UP','WAVE')` — **an acceptance gate that
   asked for two empty commands**, and which a duck that executed neither turn
   would have passed. → The bridge lives in the scenario, the one place that
   legitimately knows both namespaces, and it raises if any gesture maps to
   nothing.

2. **The motion window was shorter than the gesture it measured.** The beckon
   runs at 1.15 Hz — 0.870 s per cycle — against a 0.60 s window, so the window
   could only ever hold a *half*-swing. A half-swing is one-way hand travel whose
   `wander` collapses to ≈1.0, indistinguishable from an arm on its way up.
   MEASURED: COME was read **thirteen times in twelve seconds and never once
   confirmed**, the window resetting every ~0.7 s. → The window is 1.10 s,
   longer than every oscillation period in the vocabulary, and a test asserts
   that against the animation's own frequencies.

3. **Window edges set from sampled instants, not from the hold.** COME's real
   extension peaks at 0.948 against a ceiling of 0.97 with a 0.10 margin —
   scoring 0.22, under the bar, twice per beckon. A nine-sample check could not
   see it. → `tools/probe_templates.py` sweeps **every control tick** of every
   hold and reports the worst; every edge is set from that.

4. **A float32 round trip failed the reverse.** The command register is float32,
   so an exactly-onset `-0.32` comes back as `-0.3199999928474426` — strictly
   *greater* than the onset. The gate counted all **230 ticks** of a real reverse
   leg as sub-gait, while the same run measured 0.363 m of genuine backward
   travel. → Compared with one float32 epsilon of slack, far below the 0.02 gap
   to any command the controller emits.

5. **An interrupted episode never recorded what it did.** The interrupt closes
   one episode and opens another in a single transition, so the approach's
   measurements were frozen *after* it had already been replaced: the gate read
   **0.000 m of path for a walk that visibly covered 1.75 m**. → The episode that
   was open *before* the transition is the one frozen.

6. **A turn was graded after its own gait unwound.** The running fields keep
   updating through the ACK that follows an action, and during ACK the duck holds
   an exact zero while its gait unwinds — rotating the trunk back a few degrees.
   A turn that reached **+64.5°** and satisfied its own exit test was logged as
   **+57.8°**, so the episode contradicted the transition that closed it. → Each
   action's result is frozen the moment its execute state ends.

7. **The camera hid the subject, and the preview looked fine.** Two measured
   camera answers put the instructor behind the opaque left HUD column for the
   whole COME and STOP. The cause was `_aim_camera`'s look-at clamp, which keeps
   the eye out of the walls: inert at the sibling patrol's −52°, but firing on
   nearly every tick at the shallow angles this behavior wanted for arm
   legibility, shifting the look-at by a **whole camera distance**. → Verified by
   rendering a **MuJoCo segmentation frame**, which put her centroid at (32, 320)
   against the probe's analytic (21, 280) — agreement to ~10 px, proving the
   probe right and the eye wrong. The sweep now only offers cameras whose eye
   clears the walls, so the clamp is provably inert.

## The measurements this behavior is built on

Every locomotion constant was **re-measured on this scene** with
`tools/sweep_commands.py` rather than inherited from a sibling.

### There is no small command

Forward gait onset is a **cliff**, not a ramp. MEASURED over 6 s:

| `vx` | net travel |
|---|---|
| 0.20 | 0.008 m — no gait at all |
| 0.22 | 0.009 m — no gait at all |
| **0.24** | **0.522 m** |
| 0.30 | 0.683 m |

There is nothing between zero and a walk. Three consequences shape everything:

1. **A STOP is an exact zero, not a slow-down.** A robot that "eased off" would
   emit 0.22, stand perfectly still, and log a nonzero command — the appearance
   of compliance with none of the physics.
2. **READY, OBSERVE, CONFIRM and ACK are STATES, not speeds**, each holding a
   literal zero that the gate checks per tick.
3. **No decorative sub-gait command can exist**, and `is_sub_gait` exists so the
   gate can test that per tick rather than read the controller's branches.

### Reverse is a second, deeper cliff — and not the forward one mirrored

MEASURED over 6 s, projected on the duck's pre-command heading:

| `vx` | backward travel | yaw drift |
|---|---|---|
| −0.30 | −0.004 m — no gait | −0.9° |
| **−0.32** | **−0.716 m** | **−50.0°** |
| −0.38 | −0.982 m | −56.4° |

Backward onset sits at **−0.32**, far deeper than the forward −0.24 would
suggest, so a BACK_UP implemented by negating the approach command would log a
reverse and not move. And the reverse gait carries an enormous open-loop yaw
drift — **−50° in 6 s** — which is why the reverse leg closes a heading loop and
why its gate is graded on displacement **projected along the pre-action
heading** rather than on distance travelled.

### The turns are measured per sign

| | `wz = −0.58` | `wz = +0.58` |
|---|---|---|
| at `vx = 0.26` | −21.7 °/s | +13.4 °/s |
| at `vx = 0.30` | −21.7 °/s | **+21.1 °/s** |

**The turn speed is 0.30 because of the LEFT sign.** At 0.26 the policy's own
right bias eats most of a left command, so a left turn would take 1.6× as long
as its mirror and the two would not be comparable manoeuvres.

### Turning in place is unavailable

MEASURED at `vx = 0` across the whole command range: at most **1.6 °/s**, with
0.0032 m of drift. The duck cannot pivot. So every commanded turn is a **walked
arc**, and every look is a **head** movement — `spin_to` returns a constant zero
so the finding is discoverable from the controller rather than only a comment.

## Reproducing

```bash
cd projects/microduck-lab/gesture-response
V=../../microduck_rl/.venv/bin/python

$V tools/build_scene.py                    # regenerate assets/scene_gesture.xml
$V tools/check_gestures.py                 # every template, on the real model
$V tools/probe_templates.py                # every tick of every hold
$V scripts/validate_gesture.py --json /tmp/gr_final.json --trace /tmp/gr_trace.json
GESTURE_SUMMARY=/tmp/gr_final.json GESTURE_TRACE=/tmp/gr_trace.json \
  $V -m pytest tests/ -q
$V scripts/render_gesture.py --fps 50 --out media/gesture-response.mp4
```

The rollout is **deterministic**: the 4360-tick trace hashes identically across
runs, and the render re-grades all 48 gates on the run it actually drew, so the
video and the numbers are the same execution rather than two that agree.

Measurement tools, each of which produced a number quoted above:

```bash
$V tools/sweep_commands.py --what forward|reverse|yaw|turn|spin|zero
$V tools/probe_framing.py --trace /tmp/gr_trace.json   # the wide camera
```

The headless gate imports no rendering stack at all — proved by blocking `PIL`,
`imageio` and `matplotlib` in `sys.meta_path` and importing the entry point.

## Tests

**195 tests**, including **48 gate counterexamples**: each takes the summary of a
real passing run, breaks exactly one thing, and requires the named gate to go red
*and no other gate to be repaired by the same mutation*. A meta-test parses that
file with `ast`, collects every gate asserted, and fails if any of the **48**
acceptance gates has no counterexample — so adding a gate without one fails in
CI rather than shipping as an unchecked claim.

The suite also parses the import graph with `ast` to prove the decision layers
cannot reach the scenario, checks the module graph is acyclic, and pins
`contact_geometry.py` as **byte-identical** to the frozen sibling it was
inherited from.

## Limitations, stated plainly

- **Simulation only.** No hardware validation.
- **The people are scripted.** They walk declared routes, gesture at declared
  times, and never react to the duck. This is deliberate — an instructor who
  adjusted her timing to the robot would make "it responded" partly a fact about
  her — but nothing here demonstrates interaction with a person who responds.
- **Identity and gesture are a semantic proxy.** People are identified by MuJoCo
  body id inside a real frustum with a real occlusion ray cast, not by an RGB
  classifier, and a gesture is a **rule set over measured arm geometry** rather
  than learned perception. The reported confidence is a **rule-margin proxy** —
  how far the measured evidence sits past each window's own edge — not a
  probability, and it is labelled as such in the HUD, the metrics and here.
- **The vocabulary is six gestures.** The classifier refuses what it does not
  recognise, which is what the ambiguous pose demonstrates, but it has not been
  tested against arbitrary human motion.
- **The instructor never moves.** She stands on her mark for the whole session,
  so every range change is the robot's own doing. A moving instructor would make
  "the duck closed the range on COME" partly a fact about her.
- **The people are non-colliding kinematic proxies.** They cannot push the duck,
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
[`patrol-and-investigate/`](../patrol-and-investigate/), which is unmodified by
this behavior. Every locomotion constant was **re-measured on this scene** — the
floor, the task and the commands are different, and this behavior needed two
things no sibling did: a real **reverse** and a pair of genuinely **opposite**
turns.
