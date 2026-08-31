# follow-me-footsteps — following the leader's actual path

A semantic correction to [`../follow-me`](../follow-me/), created without
modifying that comparison baseline.

The first version selected the duck's command from the leader's **current**
pose. When the person began turning, the duck turned immediately from its own
coordinates and cut across the inside instead of arriving at the same corner.

This version follows a queue of world-space footsteps.

## What changed

At 50 Hz, the behavior records the person's accumulated path in world
coordinates. The duck pursues the interpolated point that the leader walked
`0.65 m` earlier:

```text
leader position → append world-space sample → subtract 0.65 m of path length
                                               ↓
                                  queued footprint + stored motion phase
                                               ↓
                                      stock walking policy command
```

Consequently, when the leader starts the left turn at `7.00 s`, the duck keeps
walking straight. It reaches the recorded corner at `12.44 s` and turns there:
a measured **5.44 s delayed turn**, not an immediate mirrored turn.

A green disc in the video marks the exact queued footprint. The HUD shows the
leader's current phase separately from the phase the duck is replaying.

## Demo

▶️ **[Watch or download the full MP4](media/follow-me-footsteps.mp4)**

The upper-right PiP remains at the physical head-camera position with electronic
horizon stabilization. The leader remains in view while the duck follows the
queued path.

## Choreography

1. **READY** (`0–2 s`) — both stand still.
2. **FORWARD** (`2–7 s`) — the person walks and the duck follows the straight
   footprints.
3. **LEFT TURN** (`7–15 s`) — the leader turns first; the duck continues straight
   until the old footprints reach the corner, then turns at that location.
4. **STOP** (`15–18 s`) — the spatial queue freezes and the duck also stops.
5. **RIGHT** (`18–28 s`) — the leader moves right; the duck finishes the queued
   turn and reaches/replays the right-hand leg at `25.96 s`.
6. **BACKWARD** (`28–34 s`) — both walk backward.
7. **DONE** (`34–37 s`) — both stop.

Backing toward a follower is the one deliberate safety exception to pure spatial
delay: the duck starts backing immediately (`28.02 s`) so the person cannot walk
into it. Cornering and lateral motion still come exclusively from the recorded
world-space trail.

## Validated result

Exact results are stored in
[`media/follow-me-footsteps-metrics.json`](media/follow-me-footsteps-metrics.json).

Measured result:

- complete 37-second / 1,850-step, seven-phase choreography;
- leader left turn `7.00 s` → duck left turn `12.44 s` (`+5.44 s`);
- leader right leg `18.00 s` → duck right leg `25.96 s` (`+7.96 s`);
- queued-footprint tracking RMSE `0.433 m` (maximum `1.177 m`);
- person range mean `0.605 m`, bounded to `0.389–0.961 m`;
- `fallen_steps=0`, minimum trunk height `0.114 m`;
- final stable stand at `0.116 m`;
- person visible in the stabilized PiP for `1,850/1,850` steps;
- 960×640 H.264, 1,850 frames at 50 fps, fully decoded after encoding.

## Architecture

- `assets/scene_follow_me_footsteps.xml` — official Velocity collision model,
  animated person, green footprint target and stabilized camera rig.
- `scripts/follow_motion.py` — leader trajectory, `FootstepTrail` spatial queue,
  measured policy commands and safe reverse behavior.
- `scripts/camera_tracking.py` — isolated gaze and camera visibility.
- `scripts/video_overlay.py` — dual leader/replay HUD, PiP and phase timeline.
- `scripts/render_follow_me_footsteps.py` — 61-D ONNX rollout, metrics and frames.
- `onnx/alpha_walking.onnx` — byte-identical stock policy inherited from v1.

The physical locomotion state remains authoritative. Gaze and camera
stabilization operate only in an isolated rendering `MjData` and never feed back
into walking dynamics.

## Reproduce

From the official `microduck_rl` checkout:

```bash
cp /path/to/follow-me-footsteps/assets/scene_follow_me_footsteps.xml \
  src/mjlab_microduck/robot/microduck/scene_follow_me_footsteps.xml

uv run python /path/to/follow-me-footsteps/scripts/render_follow_me_footsteps.py \
  --policy /path/to/follow-me-footsteps/onnx/alpha_walking.onnx \
  --fps 50 --out /tmp/follow-me-footsteps-frames \
  --metrics /path/to/follow-me-footsteps/media/follow-me-footsteps-metrics.json

ffmpeg -framerate 50 -i /tmp/follow-me-footsteps-frames/f%05d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  /path/to/follow-me-footsteps/media/follow-me-footsteps.mp4
```
