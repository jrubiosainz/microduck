# follow-me — following a walking person's actual path

The final validated implementation of the follow behavior. Earlier iterations —
the first pose-mirroring demo and the first footstep-queue version — were
consolidated into this folder; they remain reachable in the Git history.

It is built on the gaze layer validated in [`../move-away`](../move-away/), and is
in turn the base of [`../follow-me-among-others`](../follow-me-among-others/).

The demo uses Pollen Robotics' stock `alpha_walking.onnx` policy and the official
Velocity-task collision model (`robot_walk.xml`). A lightweight behavior layer
maps the leader's choreography to measured, stable policy commands; it does not
train or replace the locomotion network.

## Demo

▶️ **[Watch or download the full MP4](media/follow-me-left-right.mp4)**

The green disc marks the queued world-space footprint. The HUD separates the
leader's current phase from the delayed motion replayed by the duck. The
upper-right PiP is the stabilized head-camera view.

## Following footsteps, not poses

The first version selected the duck's command from the leader's **current** pose.
When the person began turning, the duck turned immediately from its own
coordinates and cut across the inside instead of arriving at the same corner.

This version follows a queue of world-space footsteps. At 50 Hz, the behavior
records the person's accumulated path in world coordinates, and the duck pursues
the interpolated point that the leader walked `0.65 m` earlier:

```text
leader position → append world-space sample → subtract 0.65 m of path length
                                               ↓
                                  queued footprint + stored motion phase
                                               ↓
                                      stock walking policy command
```

Consequently, when the leader starts the left turn, the duck keeps walking
straight and turns at the recorded corner instead: a measured **5.44 s delayed
turn**, not an immediate mirrored turn.

## True opposite turns

Two directional errors were also corrected in this version:

1. the phase labelled `LEFT TURN` used negative world yaw, so it was physically a
   right-hand curve;
2. the later `RIGHT` phase was a diagonal strafe, not an opposite turn.

The corrected route uses conventional actor-relative directions. Facing `+X`, a
left turn increases world yaw by `+90°`; the later right turn decreases yaw by
`−90°` and returns the leader to its original heading.

## Route

1. **READY** (`0–2 s`) — both stand.
2. **FORWARD** (`2–7 s`) — straight approach.
3. **LEFT TURN** (`7–15 s`) — a genuine `+90°` left arc.
4. **STOP** (`15–18 s`) — both stop and the path queue freezes.
5. **RIGHT TURN** (`18–35 s`) — a genuine `−90°` right arc followed by a straight
   exit, allowing the delayed duck to finish the same curve.
6. **BACKWARD** (`35–41 s`) — both reverse.
7. **DONE** (`41–44 s`) — stable final stand.

Because the duck follows a `0.65 m` spatial trail rather than the leader's
current pose, both turns occur at the same queued locations after the leader:

| Turn | Leader starts | Duck starts | Spatial delay | Leader yaw Δ | Duck yaw Δ |
|---|---:|---:|---:|---:|---:|
| Left | `7.00 s` | `12.44 s` | `5.44 s` | `+90.0°` | `+86.4°` |
| Right | `18.00 s` | `23.44 s` | `5.44 s` | `−90.0°` | `−84.0°` |

The signs are measured from the simulated trunk orientation, not inferred from
labels or camera projection. Thus the two duck turns are demonstrably opposite.

## Controller detail

The stock ONNX policy has strongly asymmetric turning authority. A single
mirrored command did not produce mirrored body motion. The validated controller
therefore closes the loop on the delayed footprint yaw and uses measured
asymmetric limits:

- positive-yaw / left correction: `wz=+0.60…+1.00`;
- negative-yaw / right correction: `wz=−0.18…−0.32`;
- deadband: `3°`;
- walking command during turns: `vx=+0.24`.

The stock policy has a sharp gait-onset threshold, so tiny continuous velocity
corrections are counterproductive. Policy actions use the shipped `0.9` action
scale. The angular-velocity sensor is resolved explicitly
(`imu_ang_vel`/aliases); silently reading the last sensor by index produced
unstable false results during development and is intentionally rejected now.

This is still a behavior layer over Pollen Robotics' stock
`alpha_walking.onnx`; no locomotion network was retrained.

## Validation

Exact measurements are in
[`media/follow-me-left-right-metrics.json`](media/follow-me-left-right-metrics.json).

Measured result:

- both leader turns have opposite signed `+90.0°/−90.0°` yaw changes;
- both duck turns have opposite signed `+86.4°/−84.0°` changes;
- both delayed turn starts occur `5.44 s` after the leader at the queued footprint;
- all seven phases complete over `44.0 s` / `2,200` control steps;
- `fallen_steps=0`, minimum trunk height `0.114 m`;
- final stable height `0.116 m`;
- person range mean `0.857 m` (`0.519–1.363 m`);
- person visible in the stabilized PiP for `2,200/2,200` steps;
- 960×640 H.264, 2,200 frames at 50 fps, fully decoded after encoding.

The queued-footprint error grows during the final reverse because reversing is an
intentional safety exception: the duck backs immediately rather than waiting for
the leader's reverse footprint to arrive, so the person cannot walk into it.
Cornering and lateral motion still come exclusively from the recorded world-space
trail.

## Files

- `assets/scene_follow_me_left_right.xml` — official walking collision model,
  animated leader, footprint marker and stabilized camera rig.
- `scripts/follow_motion.py` — S-shaped leader route, spatial trail, asymmetric
  closed-loop turn controller and reverse phase.
- `scripts/camera_tracking.py` — isolated gaze and camera visibility.
- `scripts/video_overlay.py` — leader/replay HUD, PiP and timeline.
- `scripts/render_follow_me_left_right.py` — ONNX rollout and metrics.
- `onnx/alpha_walking.onnx` — byte-identical stock policy.

The physical locomotion state remains authoritative. Gaze and camera
stabilization operate only in an isolated rendering `MjData` and never feed back
into walking dynamics.

## Reproduce

From the official `microduck_rl` checkout:

```bash
cp /path/to/follow-me/assets/scene_follow_me_left_right.xml \
  src/mjlab_microduck/robot/microduck/scene_follow_me_left_right.xml

uv run python /path/to/follow-me/scripts/render_follow_me_left_right.py \
  --policy /path/to/follow-me/onnx/alpha_walking.onnx \
  --fps 50 --out /tmp/follow-me-left-right-frames \
  --metrics /path/to/follow-me/media/follow-me-left-right-metrics.json

ffmpeg -framerate 50 -i /tmp/follow-me-left-right-frames/f%05d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  /path/to/follow-me/media/follow-me-left-right.mp4
```
