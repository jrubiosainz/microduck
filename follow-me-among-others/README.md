# follow-me-among-others — color-selective crowd following

A new behavior built from [`../follow-me`](../follow-me/) without modifying that
validated base.

Five pedestrians in blue, green, red, yellow and purple walk continuously on
independent smooth routes. Microduck must repeatedly stop following, search the
crowd through its camera, acquire the requested shirt color, follow that
person's queued footsteps, and stop again.

## Requested sequence

```text
BLUE → GREEN → RED → BLUE
SEARCH → FOUND → FOLLOW → STOP   (repeated four times)
```

▶️ **[Watch or download the full MP4](media/follow-me-among-others.mp4)**

The upper-right PiP is the view used for acquisition. During `SEARCH` it performs
a panoramic sweep from the physical head-camera position. A transition to
`FOUND` is permitted only when the requested color is in the camera field of
view and within 8° of the crosshair. Other visible colors are reported but do
not trigger the transition.

Color recognition is an explicit **MuJoCo semantic proxy**: actor identity plus
camera-frustum samples, not a newly trained RGB classifier. The camera geometry,
search sweep, target visibility and acquisition gate are real in the simulation;
replacing the semantic identity lookup with onboard vision is a separate task.

During `FOLLOW`, the green disc marks the selected pedestrian's queued
world-space footprint 0.55 m back along that person's path. The duck follows the
disc rather than mirroring the person's current pose. In `SEARCH`, `FOUND`
and `STOP`, the locomotion command is exactly zero.

## People and motion

- **Blue, green and red** are selectable targets.
- **Yellow and purple** are moving distractors.
- Each person walks an independent elliptical lane with a distinct direction,
  speed, phase and small unsynchronised wobble.
- All five continue moving throughout the four selections; nobody teleports or
  freezes to make acquisition easier.

The routes are intentionally busy but deterministic, so the same camera search
and follow sequence can be reproduced and measured.

## Measured sequence

| Selection | Target | Search starts | Found | Follow starts | Stop | Search time |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Blue | `0.00 s` | `2.44 s` | `3.44 s` | `12.44 s` | `2.44 s` |
| 2 | Green | `13.94 s` | `16.50 s` | `17.50 s` | `26.50 s` | `2.56 s` |
| 3 | Red | `28.00 s` | `29.34 s` | `30.34 s` | `39.34 s` | `1.34 s` |
| 4 | Blue | `40.84 s` | `41.92 s` | `42.92 s` | `51.92 s` | `1.08 s` |

Exact results are stored in
[`media/follow-me-among-others-metrics.json`](media/follow-me-among-others-metrics.json).

Acceptance gates:

- completed targets exactly `BLUE → GREEN → RED → BLUE`;
- four complete `SEARCH → FOUND → FOLLOW → STOP` cycles;
- every `FOUND` transition occurs while the target color is visible;
- `wrong_color_locks=0`;
- target visible for 100% of control steps during `FOLLOW`;
- every follow segment travels at least `0.40 m`, displaces at least `0.30 m`,
  and measurably approaches its selected target;
- maximum stationary-state locomotion command `0.0`;
- `fallen_steps=0`, minimum trunk height above `0.09 m`;
- final trunk height returns near nominal `0.116 m`;
- final H.264 fully decodes after encoding.

The mean selected-person range during following is `0.929 m`. Queued-footprint
RMSE is `1.028 m`; this includes the deliberately long first Blue approach,
rather than only the settled tail of each follow interval. In the corrected Red
segment, the duck travels `1.249 m`, displaces `0.718 m`, and reduces its
queued-footprint error from `0.397 m` to a minimum of `0.202 m`.

The original Red hand-off exposed a stock-policy gait threshold: `vx=0.16`
produced an ONNX command but no visible locomotion. The controller now retains
the measured `vx=0.24` walking command during large heading changes and the
validation gate rejects any future follow segment that fails to move or
approach its target.

## Architecture

- `assets/scene_follow_me_among_others.xml` — five colored pedestrians, target
  disc and stabilized camera rig on the official `robot_walk.xml` model.
- `scripts/crowd_motion.py` — independent pedestrian routes, five path queues,
  search/follow state machine and geometric path controller.
- `scripts/camera_search.py` — color-selective panoramic search and target
  tracking from the head-camera position.
- `scripts/video_overlay.py` — camera PiP, active target, state pipeline and
  four-selection progress HUD.
- `scripts/render_follow_me_among_others.py` — 61-D ONNX rollout, metrics and
  frame renderer.
- `onnx/alpha_walking.onnx` — byte-identical stock walking policy inherited
  from the validated base; no locomotion network was retrained.

As in the previous camera behaviors, gaze/perception uses an isolated rendering
`MjData`; it never modifies the authoritative walking state.

## Reproduce

From the official `microduck_rl` checkout:

```bash
cp /path/to/follow-me-among-others/assets/scene_follow_me_among_others.xml \
  src/mjlab_microduck/robot/microduck/scene_follow_me_among_others.xml

uv run python /path/to/follow-me-among-others/scripts/render_follow_me_among_others.py \
  --policy /path/to/follow-me-among-others/onnx/alpha_walking.onnx \
  --fps 50 --out /tmp/follow-me-among-others-frames \
  --metrics /path/to/follow-me-among-others/media/follow-me-among-others-metrics.json

ffmpeg -framerate 50 -i /tmp/follow-me-among-others-frames/f%05d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  /path/to/follow-me-among-others/media/follow-me-among-others.mp4
```
