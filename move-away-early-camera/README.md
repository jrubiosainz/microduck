# move-away-early-camera — earlier reaction with onboard camera

A conservative evolution of [`../move-away`](../move-away), created on 2026-08-31.
The validated baseline remains untouched.

The behavior is unchanged:

```
IDLE → RETREAT → TURN → CLEAR → DONE
```

It introduces only two changes:

1. **A live view from the duck's camera**, shown in the upper-right corner of the video.
2. **Reaction at 1.15 m**, 20 cm earlier than the original 0.95 m threshold.

Validated video: [`media/move-away-early-camera.mp4`](media/move-away-early-camera.mp4).

## Running it

From a `microduck_rl` checkout with its environment (`mujoco`, `onnxruntime`,
`imageio`, Pillow, and ffmpeg):

```bash
python scripts/render_phase1.py --seconds 19 --out /tmp/move-away-early-camera --fps 50
ffmpeg -framerate 50 -i /tmp/move-away-early-camera/f%05d.png \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  media/move-away-early-camera.mp4
```

Place `assets/scene_yield.xml` under `src/mjlab_microduck/robot/microduck/`
and `onnx/alpha_walking.onnx` under `onnx/`.

## Why the person starts at 1.60 m

The baseline placed the person at 1.40 m and triggered at 0.95 m. This version
adds the same 20 cm to both the initial position and the trigger threshold:

- initial position: **1.40 → 1.60 m**
- trigger distance: **0.95 → 1.15 m**

This makes the duck start at exactly the same temporal phase of the ONNX policy
as in the validated version, but while the person is still 20 cm farther away.
That preserves the baseline's stable trajectory. Changing only the activation
time altered the gait phase, and some tests over-rotated or fell over.

## On-screen camera

A second `mujoco.Renderer` is attached to `head_camera`. Every frame receives a
225×165 px PiP—75% of the initial size—in the upper-right corner. Its border is
green while the person is visible and red when the person is outside the field
of view or occluded.

This is the model's real onboard camera, not an approximate external camera.

**Optical-frame correction:** the `head_camera` quaternion exported by the
upstream MJCF does not follow MuJoCo's camera convention (`-Z` is forward and
`+Y` is up). The initial render therefore showed internal/side geometry instead
of the person. At runtime, the script applies a −90° local rotation around Z
(`[√½, 0, 0, −√½]`), making `-Z` point forward and aligning the image vertically
with the head. Detection then uses those same optical axes and the PiP's real
frustum, so the image and `SEES PERSON` status agree throughout the turn.

## Inherited stability parameters — unchanged

- `VX_RETREAT = -0.28`
- `VX_ROTATE = -0.31`
- `RETREAT_HOLD = 5.0 s`
- `CLEAR_HOLD = 2.5 s`
- `TURN_CUT = 45°`
- `CMD_TAU = 0.08`

Ray-cast perception and every other constant from the validated `move-away`
version are unchanged.
