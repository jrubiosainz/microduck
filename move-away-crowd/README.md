# move-away-crowd — avoiding several adults in a busy plaza

A deterministic MuJoCo scenario in which Microduck behaves like a small child in
a crowd of adults who may not notice it. Eight independently animated pedestrians
walk continuous routes through a plaza; five of them carry boxes in front of
their chests, which puts a wide rigid obstacle well ahead of the person's own
body. The duck watches its surroundings, predicts which approach is genuinely
dangerous, locks onto the most urgent adult, steps out of that person's path,
settles, and returns to scanning:

```text
SCANNING → THREAT_LOCK → EVADING → SETTLING → CLEAR → SCANNING
```

This is a new behavior folder. It does not modify the validated
[`move-away/`](../move-away/) baseline it was derived from.

## Demo

▶️ **[Watch or download the full MP4 (52 s, 960×640, 50 fps)](media/move-away-crowd.mp4)**

The wide view shows the whole plaza. The upper-right PiP is a stabilized view
taken from the duck's physical head-camera position; its border and label
identify the adult currently treated as the threat. The HUD reports the state,
the locked adult, the predicted closest approach, the exact velocity command,
the nearest measured clearance, and trunk height. The bottom bar is the state
pipeline; the strip below it places one coloured block per completed encounter
on the 52 s timeline.

## Scenario

- **Eight adults:** blue, green, red, yellow, purple, orange, teal, and pink.
- **Five carried boxes:** green, red, yellow, orange, and teal.
- Every adult follows an independent continuous ellipse with its own direction,
  phase, speed and approach bearing. Nobody teleports, freezes, or waits for the
  duck; the pedestrians are fully scripted and never react to it.
- Six routes are tuned to create staggered near-passes. Blue and purple stay as
  moving background traffic and are never locked.
- Pedestrian geometry does not collide with the robot in MuJoCo, so an avoidance
  cannot succeed by physically pushing the duck out of the way. Clearance is
  measured geometrically instead, against exact surfaces including the boxes.

## What is actually perceived

Threat selection is a **geometric/semantic simulation proxy, not RGB person
recognition.** MuJoCo supplies each scripted adult's identity, position and
velocity; there is no detector, tracker or classifier anywhere in this behavior.

The **predictor** takes each adult's constant-velocity closest approach to the
duck over a 5 s horizon. An approach counts as a genuine threat when the
predicted clearance falls below `0.42 m`. Genuine threats always outrank
non-threats, so a receding adult standing nearby cannot displace someone who is
about to walk into the duck. A candidate must be confirmed for `0.30 s` before a
lock, which prevents a single noisy frame from triggering an evasion.

The **camera geometry is real**: visibility is measured through the exact
frustum of the same stabilized camera drawn in the PiP, with occlusion ray
casts. A physical robot would still need an image-based person detector,
tracker and motion estimator to replace the simulator identities and poses.

## Validated 52-second rollout

52.0 s · 2600 control steps at 50 Hz · decimation 10 · stock `alpha_walking.onnx`
at action scale `0.9`. No policy was trained for this behavior.

| Cycle | Threat | Box | Approach | Lock | Evasion | Path | Net | Predicted clearance gain | Actual minimum clearance | Lock visible | Evade visible |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | orange | yes | E | 1.52 s | 3.78 s | 0.546 m | 0.332 m | +0.139 m | 0.098 m | 98.3 % | 100 % |
| 2 | green | yes | S | 10.46 s | 4.10 s | 0.903 m | 0.750 m | +0.410 m | 0.128 m | 96.0 % | 100 % |
| 3 | red | yes | W | 19.66 s | 2.62 s | 0.533 m | 0.437 m | +0.468 m | 0.086 m | 93.2 % | 100 % |
| 4 | pink | no | NW | 41.58 s | 3.06 s | 0.443 m | 0.288 m | +0.132 m | 0.257 m | 98.5 % | 100 % |

The clearance gain compares each evasion against its own counterfactual: the
clearance the duck would have had at closest approach had it not moved.

State occupancy over the rollout: `SCANNING` 1210 steps, `THREAT_LOCK` 148,
`EVADING` 678 (26.1 %), `SETTLING` 322, `CLEAR` 242.

All twelve acceptance gates pass:

| Gate | Result |
|---|---|
| four complete avoidance cycles | 4 |
| distinct adults evaded | orange, green, red, pink |
| distinct approach sectors | E, S, W, NW |
| falls | 0 |
| minimum trunk height | 0.1126 m (limit 0.09 m) |
| final trunk height | 0.1163 m (nominal 0.116 m) |
| contact steps | 0; minimum geometric clearance 0.0858 m |
| locomotion command outside `EVADING` | exactly 0.000000 over 1922 steps |
| every evasion physically moved | yes, path and net displacement measured |
| every evasion improved its counterfactual | yes, 4/4 |
| wrong-target locks | 0; every lock top-ranked and tightest |
| locked-adult visibility | 93.2–98.5 % per lock, 100 % during every evasion |

Exact values are in
[`media/move-away-crowd-metrics.json`](media/move-away-crowd-metrics.json).
The test suite has **82 passing tests**.

## Controller corrections found during validation

Two integration defects survived a green unit-test suite and were caught only by
the full physical gate:

1. **Inverted lateral escape sign.** The reverse primitive backed the duck along
   roughly the adult's own lane. During the green encounter that let the carried
   box catch the duck after it had stopped, at −0.074 m of overlap. The command
   now projects the desired escape heading onto the robot's local lateral axis
   with `sin(heading_error)`. A test checks vector *alignment* with the requested
   escape direction, because merely having opposite left/right signs cannot
   detect that both signs were inverted.
2. **Box closer than the person.** A carried box is far ahead of the adult centre
   used by the predictor, so the commitment beat before evading could be spent
   while contact was already imminent. When exact MuJoCo surface clearance drops
   below `0.35 m`, `THREAT_LOCK` is shortened from `0.90 s` to `0.30 s`.

The backward escape exists because turning would be worse: at the measured turn
rate of about 50°/s, swinging 130° takes ~1.5 s spent moving *toward* the threat.

## Limitations

- **Gaze is rendering-only and kinematically isolated.** The head pose is applied
  to a separate `MjData` copy, so aiming the camera never feeds back into the
  walking policy or the physical state. Gaze cannot artificially stabilize the
  robot, and equally the robot's real head actuators are not exercised here.
- **Pedestrians are scripted and kinematic.** They follow fixed routes, do not
  react to the duck, and do not collide with it in physics.
- **Identity and kinematics come from the simulator,** not from perception.
- **Simulation only.** No hardware validation.
- The PiP is informational; the behavior's decisions are not taken from pixels.

## Reproduce

Use the environment from a local `microduck_rl` checkout:

```bash
cd /path/to/microduck_rl

# Physics and acceptance gates, no rendering dependencies at all
uv run python /path/to/move-away-crowd/scripts/render_move_away_crowd.py \
  --no-render --seconds 52 \
  --metrics /tmp/move-away-crowd-metrics.json

# Low-fps preview for visual inspection
uv run python /path/to/move-away-crowd/scripts/render_move_away_crowd.py \
  --seconds 52 --fps 5 --width 960 --height 640 \
  --out /tmp/preview-frames --metrics /tmp/preview-metrics.json

# Final 2600 frames at 50 fps
uv run python /path/to/move-away-crowd/scripts/render_move_away_crowd.py \
  --seconds 52 --fps 50 --width 960 --height 640 \
  --out /tmp/move-away-crowd-frames \
  --metrics media/move-away-crowd-metrics.json

ffmpeg -framerate 50 -i /tmp/move-away-crowd-frames/f%05d.png \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart \
  media/move-away-crowd.mp4

# Tests
uv run python -m pytest /path/to/move-away-crowd/tests -q
```

## Contents

- `assets/scene_move_away_crowd.xml` — robot, eight adults, boxes, markers, cameras.
- `scripts/policy_runtime.py` — scene loading, 61-D observation, ONNX rollout.
- `scripts/crowd_routes.py` — deterministic pedestrian routes and gait animation.
- `scripts/threat_model.py` — approach predictor, state machine, evade controller.
- `scripts/attention_camera.py` — isolated gaze, frustum and occlusion checks.
- `scripts/rollout_crowd.py` — authoritative policy/physics integration and evidence.
- `scripts/crowd_metrics.py` — per-cycle acceptance gates.
- `scripts/render_frames.py`, `scripts/video_overlay.py` — video presentation.
- `tools/build_scene.py`, `tools/sweep_commands.py` — scene generation, command sweeps.
- `tests/` — pure control tests plus real MuJoCo scene and contact tests.
- `onnx/alpha_walking.onnx` — byte-identical stock walking policy
  (`sha256:e36332d3…daa6c`, matching `move-away/` and the upstream checkout).
