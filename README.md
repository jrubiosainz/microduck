# microduck

Base de trabajo propia sobre el simulador del Mini BDX (microduck / microduck_rl de Pollen Robotics).

La idea: cada comportamiento que llegue a funcionar de verdad en simulación se congela
en su propia subcarpeta, y se va incrementando desde ahí sólo con pasos seguros.

> **Mejor versión actual:** [`move-away-early-camera/`](move-away-early-camera/) —
> nueva base recomendada para cualquier incremento futuro.

## Comportamientos

| Carpeta | Estado | Qué hace |
|---|---|---|
| [`move-away/`](move-away/) | ✅ funciona | Base congelada: detecta a una persona, retrocede, gira 90° y se aparta. |
| [`move-away-early-camera/`](move-away-early-camera/) | 🏆 mejor actual | Maniobra estable iniciada a 1,15 m; cámara real del pato en PiP 225×165; validación de 19 s con la persona caminando 3 s adicionales. |
| [`move-away-head-tracking/`](move-away-head-tracking/) | 🧪 candidata validada | Extiende a 22 s y mantiene a la persona en la cámara durante 1100/1100 pasos mediante una capa cinemática independiente de mirada. |

## Convenciones

- Una subcarpeta por comportamiento. Nunca se toca un comportamiento ya validado
  para probar el siguiente: se copia y se itera en la copia.
- Cada subcarpeta lleva su `README.md` con los parámetros MEDIDOS (no supuestos)
  y el vídeo de la versión buena en `media/`.
- Las políticas ONNX son las de stock de `microduck_rl`; aquí no se entrena nada
  todavía, se conduce la marcha con comandos de velocidad desde una capa de
  comportamiento.

## Upstream

- Simulador / políticas: https://github.com/pollen-robotics/microduck_rl
- Firmware del robot: https://github.com/pollen-robotics/microduck
