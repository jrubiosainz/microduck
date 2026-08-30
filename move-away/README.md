# move-away — apartarse de una persona que se acerca

Versión que **funciona** (30-ago-2026). Vídeo: [`media/move-away-latest.mp4`](media/move-away-latest.mp4).

El pato ve a una persona aproximarse por la cámara de la cabeza, retrocede en línea
recta, gira ~90° y sigue retrocediendo en el nuevo rumbo hasta quedar fuera de su paso.

```
IDLE  →  RETREAT  →  TURN  →  CLEAR  →  DONE
```

No hay política nueva entrenada: se conduce el `alpha_walking.onnx` de stock con un
comando de velocidad generado por una máquina de estados.

## Ejecutar

```bash
# desde un checkout de microduck_rl con su venv (mujoco + onnxruntime + imageio)
python scripts/render_phase1.py --seconds 16 --out /tmp/f_p1 --fps 50
```

`assets/scene_yield.xml` va en `src/mjlab_microduck/robot/microduck/`,
`onnx/alpha_walking.onnx` en `onnx/`.

## Parámetros MEDIDOS (no tocar a ciegas)

Todo esto salió de romper la simulación varias veces. Son observaciones, no ajustes estéticos:

- **La autoridad de guiñada sólo existe MIENTRAS camina.** Parado, un comando `wz`
  no hace prácticamente nada.
- **El signo de `wz` está INVERTIDO**: `wz` positivo produce velocidad de guiñada
  *negativa*. De ahí el menos en la ley de control.
- **Ventana útil de `vx` muy estrecha:** por debajo de ~−0.30 la marcha no arranca y
  el pato pisa en el sitio; por encima de ~−0.33 acumula deriva y se cae.
  - `VX_RETREAT = -0.28` → mantiene rumbo con 1.5° de error en 5 s.
  - `VX_ROTATE = -0.31` → pasado el escalón, autorrota ~110°/s.
- **El rumbo hay que cerrarlo en lazo cerrado en TODOS los estados que caminan.**
  Dejar `wz = 0.00` durante el retroceso era el bug real detrás de "se gira al otro
  lado": la marcha es asimétrica y, sin oposición, gira sola (medido: −52° en t=9 s,
  +117° en t=12 s con `wz` exactamente a cero).
- `YAW_KP = 0.9`, `WZ_MAX = 0.20`. Con 0.6 se descontrola y vuelca.
- `TURN_CUT = 45°` — se corta pronto porque la rotación sobrepasa hasta ~90°.
- `CMD_TAU = 0.08` — el filtro del comando; 0.25 es demasiado lento para arrancar la marcha.

## Percepción

Por **ray-cast** desde la cámara de la cabeza a la persona, no por segmentación.
La segmentación no servía: el *site* de la cámara está DENTRO de la geometría de la
propia mandíbula del pato, así que todos los rayos/píxeles chocan con `jaw_soft`
primero y la persona no se ve nunca. El ray-cast salta los cuerpos propios del pato
y responde a la pregunta real: ¿está la persona en el campo de visión y sin ocluir?

**El pato mira hacia +x.** (Una versión anterior movía a la persona hacia −x tras
leer mal la matriz de cámara; eso ponía a los dos espalda contra espalda.)

## Scripts

- `render_phase1.py` — **la versión buena**. Retroceso + giro de 90° + despeje.
- `render_yield.py` — intento previo (RETREAT → SIDESTEP → SETTLE). Sidestep frágil.
- `render_headless.py` — utilidad de render sin ventana.

## Siguiente paso (aún no hecho)

Volver a meter el *sidestep* y el "dejar pasar" encima de esta base, sin tocarla.
