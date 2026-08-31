# follow-me-among-others — color-selective crowd following

A new behavior built from
[`../follow-me-left-right-turns`](../follow-me-left-right-turns/) without
modifying that validated base.

Five pedestrians in blue, green, red, yellow and purple walk continuously on
independent smooth routes. Microduck must repeatedly stop following, search the
crowd through its camera, acquire the requested shirt color, follow that
person's queued footsteps, and stop again.

## Requested sequence

```text
BLUE  → GREEN → RED   → BLUE
BUSCO → ENCUENTRO → SIGO → PARO   (repeated four times)
```

▶️ **[Watch or download the full MP4](media/follow-me-among-others.mp4)**

The upper-right PiP is the view used for acquisition. During `BUSCO` it performs
a panoramic sweep from the physical head-camera position. A transition to
`ENCUENTRO` is permitted only when the requested color is in the camera field of
view and within 8° of the crosshair. Other visible colors are reported but do
not trigger the transition.

Color recognition is an explicit **MuJoCo semantic proxy**: actor identity plus
camera-frustum samples, not a newly trained RGB classifier. The camera geometry,
search sweep, target visibility and acquisition gate are real in the simulation;
replacing the semantic identity lookup with onboard vision is a separate task.

During `SIGO`, the green disc marks the selected pedestrian's queued
world-space footprint 0.55 m back along that person's path. The duck follows the
disc rather than mirroring the person's current pose. In `BUSCO`, `ENCUENTRO`
and `PARO`, the locomotion command is exactly zero.

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
| 1 | Blue | `0.00 s` | `2.34 s` | `3.34 s` | `12.34 s` | `2.34 s` |
| 2 | Green | `13.84 s` | `16.62 s` | `17.62 s` | `26.62 s` | `2.78 s` |
| 3 | Red | `28.12 s` | `28.54 s` | `29.54 s` | `38.54 s` | `0.42 s` |
| 4 | Blue | `40.04 s` | `44.14 s` | `45.14 s` | `54.14 s` | `4.10 s` |

Exact results are stored in
[`media/follow-me-among-others-metrics.json`](media/follow-me-among-others-metrics.json).

Acceptance gates:

- completed targets exactly `BLUE → GREEN → RED → BLUE`;
- four complete `BUSCO → ENCUENTRO → SIGO → PARO` cycles;
- every `ENCUENTRO` transition occurs while the target color is visible;
- `wrong_color_locks=0`;
- target visible for 100% of control steps during `SIGO`;
- maximum stationary-state locomotion command `0.0`;
- `fallen_steps=0`, minimum trunk height above `0.09 m`;
- final trunk height returns near nominal `0.116 m`;
- final H.264 fully decodes after encoding.

The mean selected-person range during following is `0.955 m`
(`0.456–1.558 m`). Queued-footprint RMSE is `0.972 m`; this includes each
initial cross-crowd approach after switching targets, rather than only the
settled tail of each follow interval.

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
