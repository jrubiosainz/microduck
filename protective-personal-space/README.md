# Protective Personal Space — behavior 12/12

Microduck accompanies **Aina**, predicts when moving adults will enter her
personal-space buffer, and changes position without touching or chasing anyone.
It interposes on a measured bearing for single approaches, yields when Aina
walks directly toward it, chooses an escape gap during a simultaneous squeeze,
and returns to a neutral beside/rear escort slot after every episode.

## Validated scenario

The 190 s plaza run contains eight continuously scripted adults and seven static
fixtures. The measured sequence is:

1. Establish the escort slot beside and 0.30 m behind Aina.
2. Interpose for **Dario** from the east.
3. Interpose for **Noor** from the west.
4. Interpose for **Yara** from the north-east.
5. Abandon the station and reverse when **Aina approaches directly**.
6. Resolve the simultaneous **Kwame + Tomas** squeeze through a scored safe gap.
7. Interpose for **Liesl** from the opposite side.
8. Recover the escort slot and stop in `DONE`.

**Piet** is a deliberate near-pass false alarm: the predictor observes him but
never opens an intervention episode.

## Result

| Measurement | Validated value |
|---|---:|
| Acceptance gates | **27 / 27** |
| Tests | **774 passed**, 2 optional skips |
| Physical path | **22.1214 m** |
| Policy-command path | **21.7457 m** |
| Individual intrusion cycles | **4 distinct people** |
| Protective cycles including squeeze | **5** |
| Intrusion bearings | `-2.37°, +178.73°, +38.98°, -145.76°, +51.90°` |
| Aina visible when her camera phase has LOS | **97.94%** |
| Active threat visible during threat-camera phases | **98.89%** |
| Minimum person surface clearance | **+0.0077 m** |
| Minimum scenery surface clearance | **+0.2754 m** |
| Geometric contacts / falls | **0 / 0** |
| Minimum / final trunk height | **0.11058 / 0.11628 m** |
| Final escort-slot error | **0.0760 m** |
| Hold-state command peak | **exactly 0.0** |
| Lateral policy command | **exactly 0.0** |

The narrow +7.7 mm person clearance occurs in the deliberately difficult
simultaneous squeeze. Actors are kinematic and non-colliding, so this positive
clearance is not enforced by MuJoCo contacts; it is measured independently from
primitive geometry at every control tick.

## Architecture and honesty boundaries

- **Locomotion is physical.** The stock `alpha_walking.onnx` policy receives
  `[vx, vy, wz]` commands, drives 14 actuators, and advances the real MuJoCo
  robot at 50 Hz. Observation is exactly 61-D, action scale is `0.9`, and angular
  velocity comes from `imu_ang_vel`.
- **There is no strafe or decorative sub-gait.** Every side change is a walked
  arc; `vy` remains zero. Holds emit literal zero.
- **People are scripted non-colliding mocap proxies.** They do not avoid, wait
  for, or push the robot. This makes zero contact and clearance properties of
  the controller rather than the contact solver.
- **Identity and intent are semantic simulator proxies.** Aina's identity and
  the interpretation “intrusion” are not learned from pixels. Position,
  velocity, closest approach, time-to-contact, frustum and occlusion are
  measured from the simulated world.
- **The head camera is isolated.** Gaze yaw/pitch are applied only to a copied
  `MjData` used for PiP and visibility; they cannot inject torque into the
  validated walking dynamics.
- **One camera cannot face both directions simultaneously.** Threat visibility
  is graded while predicting/repositioning; Aina visibility is graded while
  holding, escorting and recovering. Both measurements use the exact rendered
  PiP with real frustum and MuJoCo ray casts.

## Reproduce

From `projects/microduck-lab`:

```bash
uv run --project ../microduck_rl \
  protective-personal-space/scripts/validate_pps.py \
  --seconds 190 \
  --json protective-personal-space/media/protective-personal-space-metrics.json \
  --quiet

uv run --project ../microduck_rl --with pytest \
  pytest protective-personal-space/tests -q
```

The canonical delivery is H.264, 960×640, 190.0 s and 50 fps. The physical
rollout and HUD data are evaluated at 50 Hz. To bound the unusually long render,
the validated visual pass was sampled at 5 fps and packaged at 50 fps without
inventing intermediate simulation states; the 5 fps source is retained beside
the delivery file.

## Artifacts

- `media/protective-personal-space.mp4` — canonical 50 fps delivery
- `media/protective-personal-space-telegram.mp4` — ≤5 MB delivery derivative
- `media/protective-personal-space-5fps.mp4` — validated visual source
- `media/protective-personal-space-contact-sheet.jpg` — 12 moments across run
- `media/protective-personal-space-metrics.json` — 27-gate headless metrics
- `media/protective-personal-space-render-metrics.json` — rendered-run metrics
