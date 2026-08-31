# follow-me — following a walking person

A new behavior built from the frozen
[`move-away-head-tracking`](../move-away-head-tracking/) baseline. The validated
versions remain untouched.

The demo uses Pollen Robotics' stock `alpha_walking.onnx` policy and the official
Velocity-task collision model (`robot_walk.xml`). A lightweight behavior layer
maps the leader's choreography to measured, stable policy commands; it does not
train or replace the locomotion network.

## Demo

▶️ **[Watch or download the full MP4](media/follow-me.mp4)**

The main view shows the leader and Microduck. The upper-right PiP is located at
the physical head-camera position, points at the leader, and applies electronic
horizon stabilization so the view remains readable during turns. The HUD reports
follow error, person range, policy commands, heading error, body height, camera
visibility, and the active choreography phase.

## Choreography

The 33-second deterministic sequence is:

1. **READY** (`0–2 s`) — both stand still.
2. **FORWARD** (`2–7 s`) — the person walks forward and the duck follows.
3. **LEFT TURN** (`7–15 s`) — the person curves left and the duck follows the turn.
4. **STOP** (`15–18 s`) — both stop.
5. **RIGHT** (`18–24 s`) — the person walks to screen-right and the duck follows.
6. **BACKWARD** (`24–30 s`) — the person walks backward and the duck follows backward.
7. **DONE** (`30–33 s`) — both stop again.

The humanoid is a scripted MuJoCo mocap body with animated opposing arm and leg
swings. Local `+X` is its forward direction.

## Validated result

The exact results of the committed video are stored in
[`media/follow-me-metrics.json`](media/follow-me-metrics.json).

Measured result:

- all seven requested phases complete over `33.0 s` / `1,650` control steps;
- no fall: `fallen_steps=0`, minimum trunk height `0.114 m`;
- final stable stand at `0.116 m` trunk height;
- person range mean `0.683 m`, bounded to `0.518–0.819 m`;
- person visible in the stabilized head camera for `1,650/1,650` steps;
- exact 0.50 m-behind follow-point error: `RMSE=0.623 m`, maximum `0.917 m`;
- 960×640 H.264 video, `1,650` frames at 50 fps, verified decodable.

The target follow point is `0.50 m` behind the person's heading. Error against
that exact point is reported honestly; it grows during the curved and lateral
legs because the stock walking policy has asymmetric turning and strafe
response. Person range is reported separately and is the more intuitive visual
spacing metric.

## Measured policy commands

The stock policy has a sharp gait-onset threshold, so tiny continuous velocity
corrections are counterproductive. This behavior uses phase-specific commands
measured in this simulator:

| Phase | `vx` | `vy` | `wz` | Purpose |
|---|---:|---:|---:|---|
| FORWARD | `+0.24` | `0.00` | `0.00` | Stable forward gait |
| LEFT TURN | `+0.24` | `0.00` | `-0.32` | Follow the leader's left arc |
| STOP / DONE | `0.00` | `0.00` | `0.00` | Stand still |
| RIGHT | `+0.24` | `-0.12` | `0.00` | Forward-right diagonal |
| BACKWARD | `-0.32` | `0.00` | `+0.20` | Backward gait with heading compensation |

Policy actions use the shipped `0.9` action scale. The angular-velocity sensor is
resolved explicitly (`imu_ang_vel`/aliases); silently reading the last sensor by
index produced unstable false results during development and is intentionally
rejected now.

## Files

- `assets/scene_follow_me.xml` — walking scene, animated humanoid and stabilized
  camera rig.
- `scripts/follow_motion.py` — leader trajectory and behavior commands.
- `scripts/camera_tracking.py` — independent head gaze, visibility and stabilized
  camera pose.
- `scripts/video_overlay.py` — metrics HUD, PiP and phase timeline.
- `scripts/render_follow_me.py` — ONNX rollout, metrics and frame renderer.
- `onnx/alpha_walking.onnx` — byte-identical stock policy inherited from the
  validated baseline.

## Reproduce

From an official `microduck_rl` checkout with its `uv` environment:

```bash
cp /path/to/follow-me/assets/scene_follow_me.xml \
  src/mjlab_microduck/robot/microduck/scene_follow_me.xml

uv run python /path/to/follow-me/scripts/render_follow_me.py \
  --policy /path/to/follow-me/onnx/alpha_walking.onnx \
  --fps 50 --out /tmp/follow-me-frames \
  --metrics /path/to/follow-me/media/follow-me-metrics.json

ffmpeg -framerate 50 -i /tmp/follow-me-frames/f%05d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  /path/to/follow-me/media/follow-me.mp4
```
