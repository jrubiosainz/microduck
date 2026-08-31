# move-away-head-tracking — apartarse sin perder de vista a la persona

Evolución conservadora de [`../move-away-early-camera`](../move-away-early-camera),
creada el 31-ago-2026. La mejor base anterior permanece intacta.

Mantiene exactamente la maniobra validada:

```
IDLE → RETREAT → TURN → CLEAR → DONE
```

Y añade una capa de mirada independiente que orienta yaw y pitch de la cabeza
hacia la persona durante toda la secuencia. El vídeo dura 22 segundos: la
persona continúa caminando 3 segundos más que en la base de 19 s.

Vídeo validado: [`media/move-away-head-tracking.mp4`](media/move-away-head-tracking.mp4).

## Resultado validado

- **Visibilidad:** `1100/1100` pasos de control; `lost_steps=0`.
- **Error angular máximo:** `1.2°` respecto al centro de la cámara.
- **Movimiento:** misma transición y trayectoria que la base validada.
- **Estabilidad final:** `trunk z=0.116`, erguido en `DONE`.
- **Vídeo:** 22 s, 960×640, 50 fps, H.264.
- **PiP:** 225×165 px, arriba a la derecha.

## Ejecutar

Desde un checkout de `microduck_rl` con su entorno (`mujoco`, `onnxruntime`,
`imageio`, Pillow y ffmpeg):

```bash
python scripts/render_phase1.py \
  --seconds 22 --out /tmp/move-away-head-tracking --fps 50
ffmpeg -framerate 50 -i /tmp/move-away-head-tracking/f%05d.png \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  media/move-away-head-tracking.mp4
```

`assets/scene_yield.xml` va en `src/mjlab_microduck/robot/microduck/` y
`onnx/alpha_walking.onnx` en `onnx/`.

## Cómo funciona el seguimiento

La dirección hacia el torso de la persona se proyecta sobre los ejes ópticos
reales de `head_camera`. Un servo visual corrige:

- `head_yaw`: error horizontal;
- `head_pitch`: error vertical.

La cámara usa la convención de MuJoCo (`-Z` = mirada, `+Y` = arriba) y conserva
la corrección de frame óptico de la versión anterior. La comprobación de
visibilidad usa el frustum real del PiP y ray-cast contra la escena.

### Separación entre mirada y locomoción

La cabeza representa una fracción grande de la masa del microduck. Dos pruebas
controlándola físicamente durante la marcha hicieron caer al robot, incluso
preservando la salida estabilizadora de la política. La ONNX de locomoción no
fue entrenada con una trayectoria externa de cabeza.

Por eso esta versión usa una **capa cinemática independiente de mirada**:

1. la física y la política de marcha avanzan en el `MjData` original sin ningún
   cambio respecto a la base estable;
2. percepción y render usan una copia aislada de `MjData` con yaw/pitch de
   cabeza orientados hacia la persona;
3. esa pose nunca se reinyecta en la dinámica de locomoción.

Esto equivale a separar el controlador de mirada del controlador de marcha,
como debe hacerse en el robot real, y evita presentar como estable una dinámica
que la política actual no sabe compensar.

## Parámetros heredados sin cambios

- reacción: `RETREAT_D = 1.15 m`, persona desde `x0 = 1.60 m`;
- `VX_RETREAT = -0.28`;
- `VX_ROTATE = -0.31`;
- `RETREAT_HOLD = 5.0 s`;
- `CLEAR_HOLD = 2.5 s`;
- `TURN_CUT = 45°`;
- `CMD_TAU = 0.08`.

Nunca se modificó `move-away-early-camera/`; esta variante vive íntegramente en
su propia carpeta.
