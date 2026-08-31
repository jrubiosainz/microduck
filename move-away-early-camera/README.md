# move-away-early-camera — apartarse antes + cámara a bordo

Evolución conservadora de [`../move-away`](../move-away), creada el 31-ago-2026.
La base validada permanece intacta.

El comportamiento es el mismo:

```
IDLE → RETREAT → TURN → CLEAR → DONE
```

Sólo incorpora dos cambios:

1. **Vista en tiempo real de la cámara del pato**, en un cuadro arriba a la derecha del vídeo.
2. **Reacción a 1,15 m**, 20 cm antes que el umbral original de 0,95 m.

Vídeo validado: [`media/move-away-early-camera.mp4`](media/move-away-early-camera.mp4).

## Ejecutar

Desde un checkout de `microduck_rl` con su entorno (`mujoco`, `onnxruntime`,
`imageio`, Pillow y ffmpeg):

```bash
python scripts/render_phase1.py --seconds 16 --out /tmp/move-away-early-camera --fps 50
ffmpeg -framerate 50 -i /tmp/move-away-early-camera/f%05d.png \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  media/move-away-early-camera.mp4
```

`assets/scene_yield.xml` va en `src/mjlab_microduck/robot/microduck/` y
`onnx/alpha_walking.onnx` en `onnx/`.

## Por qué la persona empieza a 1,60 m

La base comenzaba con la persona a 1,40 m y disparaba a 0,95 m. Esta versión
suma los mismos 20 cm tanto a la posición inicial como al umbral:

- inicio: **1,40 → 1,60 m**
- disparo: **0,95 → 1,15 m**

Así el pato comienza exactamente en la misma fase temporal de la política ONNX
que en la versión validada, pero cuando la persona aún está 20 cm más lejos.
Esto conserva la trayectoria estable de la base. Cambiar sólo el instante de
activación alteraba la fase de la marcha y algunas pruebas acababan girando de
más o cayéndose.

## Cámara en pantalla

Se usa un segundo `mujoco.Renderer` asociado a `head_camera`. Cada fotograma se
pega como PiP de 300×220 px en la esquina superior derecha, con borde verde si
la persona está visible y rojo si queda fuera del campo de visión u ocluida.
Es la cámara a bordo real del modelo, no una cámara exterior aproximada.

## Parámetros de estabilidad heredados (no tocados)

- `VX_RETREAT = -0.28`
- `VX_ROTATE = -0.31`
- `RETREAT_HOLD = 5.0 s`
- `CLEAR_HOLD = 2.5 s`
- `TURN_CUT = 45°`
- `CMD_TAU = 0.08`

Se mantienen también la percepción por ray-cast y todas las demás constantes
de la versión `move-away` validada.
