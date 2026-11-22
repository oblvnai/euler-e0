# Planck Silicon — Euler E0

Primer chip de Planck Silicon, la división de chips de Oblivion: un motor de
multiplicación de matrices INT8 con arreglo sistólico y ReLU, es decir, una capa de
red neuronal en hardware. Se fabrica con Tiny Tapeout (SkyWater 130 nm, tanda SKY26d).

## Qué hay aquí

| Ruta | Qué es |
| --- | --- |
| `src/euler_pe.v` | La celda: un multiplicador-acumulador INT8 |
| `src/euler_core.v` | El arreglo N×N, los búferes y el control de 8 comandos |
| `src/project.v` | Los pines de Tiny Tapeout (`tt_um_planck_e0`) |
| `test/test.py`, `test/tb.v`, `test/Makefile` | Pruebas cocotb que corre Tiny Tapeout |
| `info.yaml`, `docs/info.md` | Información del proyecto y hoja de datos |

## Resultados de verificación (22 nov 2026)

- Simulación del circuito: 301 pruebas en 2×2 y 4×4, 0 errores.
- Pruebas cocotb: 2 de 2 pasan (64 multiplicaciones verificadas).
- Síntesis 2×2 en SKY130: 2,473 celdas, 18,584 µm².
- Simulación con compuertas reales: 151 pruebas, 0 errores.
- Plano físico propio con OpenROAD: 211 × 211 µm, 52% de uso, holgura de +11.1 ns a 50 MHz.

El GDS oficial, la verificación de reglas (DRC/LVS) y la prueba a nivel de compuertas
los genera la acción `gds` de este repositorio.

Software: el chip se programa con Hilbert, la plataforma de software de Planck Silicon.
