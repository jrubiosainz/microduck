# microduck

Base de trabajo propia sobre el simulador del Mini BDX (microduck / microduck_rl de Pollen Robotics).

La idea: cada comportamiento que llegue a funcionar de verdad en simulación se congela
en su propia subcarpeta, y se va incrementando desde ahí sólo con pasos seguros.

## Comportamientos

| Carpeta | Estado | Qué hace |
|---|---|---|
| [`move-away/`](move-away/) | ✅ funciona | El pato detecta a una persona que se le acerca, retrocede, gira 90° y se aparta de su camino. |

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
