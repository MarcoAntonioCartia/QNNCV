# Demostración en FPGA: capa digital de control y lectura

Reproducción de una tabla de fases en los dos canales de un Pmod DA2, con un pulso de compuerta por paso, sobre una Arty A7. Base del Apéndice C de la tesis (sección 5.5).

**Estado (18 nov 2026):** diseño escrito y probado en simulación (Icarus Verilog 12.0 + cocotb 2.1.0, 14 pruebas, todas pasan). **No se ha sintetizado con Vivado ni probado en la placa.** Los datos marcados *por verificar* no se pudieron contrastar con la hoja de datos original.

## 1. Qué demuestra y qué no

Demuestra que la electrónica digital que acompañaría a un procesador fotónico programable se puede escribir y verificar a partir de los pesos entrenados:

- los ángulos de la malla de interferómetros (descomposición tipo Clements de `scripts/hw_sweep.py`) se convierten en códigos de 12 bits y se cargan en una memoria;
- un secuenciador recorre la tabla capa a capa y fija cada par de fases (interna y externa de un Mach–Zehnder) en dos salidas analógicas, en escalón o en rampa;
- en cada paso, una vez asentadas las tensiones, se emite un pulso de compuerta (ventana de bombeo o de lectura) con una posición y una anchura fijas respecto al cambio de fase.

No demuestra nada óptico:

- no hay fotones, ni desfasadores, ni detector; las salidas son tensiones de 0 a 3,3 V sobre un osciloscopio;
- la correspondencia fase → código es lineal e ilustrativa (2π sobre todo el rango del DAC, la misma de `hw_sweep.dac12`); un desfasador real necesita su curva de calibración, y uno termo-óptico responde a la potencia, no a la tensión;
- dos canales reproducen un Mach–Zehnder a la vez; un chip real fija todas las fases en paralelo, con un canal por desfasador;
- los tiempos (100 µs por tic, pasos de 5,6 ms) están elegidos para verse en el osciloscopio, no para una tasa de repetición óptica;
- la compresión, los desplazamientos y la puerta de Kerr de cada capa no se representan;
- la lectura por TDC no se construye: el pulso de compuerta es la señal que un TDC usaría como referencia de la ventana de medida. Ese enlace queda como texto en el libro.

## 2. Diagrama de bloques

```
                         clk100 (100 MHz, único reloj)
                                     │
 sw[3:0] ──► sincronizador ──┐       │
 btn0 ─────► sincronizador ──┤ rst   │
                             ▼       ▼
 phase_table.mem      ┌─────────────────────┐  start, code_a, code_b  ┌──────────┐  SYNC  ─► JA1
        │             │   phase_sequencer   ├────────────────────────►│ da2_spi  │  DINA  ─► JA2
        ▼    dir      │  búsqueda → rampa → │                         │ 16 bits  │  DINB  ─► JA3
 ┌───────────┐◄───────┤  asentamiento →     │                         │ 12,5 MHz │  SCLK  ─► JA4
 │ phase_rom │ palabra│  compuerta → espera │                         └──────────┘
 │ 256 × 32  ├───────►│                     ├─► gate ─► JD1                 Pmod DA2
 └───────────┘        │                     ├─► trig ─► JD2            (2 × DAC121S101)
                      │                     ├─► led[3:0] (capa)          OUT A, OUT B
                      └─────────────────────┘                          ─► osciloscopio
```

| archivo | función |
|---|---|
| `rtl/da2_spi.v` | controlador serie del Pmod DA2, los dos canales en la misma trama |
| `rtl/phase_rom.v` | memoria de la tabla, cargada con `$readmemh` |
| `rtl/phase_sequencer.v` | secuenciador: paso, rampa, compuerta, disparo, selección de capa |
| `rtl/arty_phase_demo.v` | módulo superior para la Arty A7 |
| `scripts/phases_to_mem.py` | tabla de fases (CSV) → archivo `.mem` |
| `scripts/build.tcl` | síntesis por lotes en Vivado |
| `constr/arty_a7.xdc` | pines y reloj |
| `data/` | tabla de ejemplo y su `.mem` |
| `tb/` | bancos de prueba |

### Formato de la tabla

Una palabra de 32 bits por paso:

| bits | campo | significado |
|---|---|---|
| 31 | `last` | último paso; el secuenciador vuelve a la dirección 0 |
| 30 | `ramp` | 1: rampa lineal desde el código anterior; 0: escalón |
| 29 | `gate` | 1: emitir el pulso de compuerta en este paso |
| 28 | `mark` | primer paso de una capa (activa la salida de disparo) |
| 27:24 | `layer` | número de capa, 0 a 14 |
| 23:12 | código B | fase externa del Mach–Zehnder |
| 11:0 | código A | fase interna del Mach–Zehnder |

Cada Mach–Zehnder de las mallas U1 y U2 de una capa es un paso. Con 3 modos son 6 pasos por capa. Caben 256 pasos.

### De los ángulos a los códigos

`código = round((fase mod 2π) / 2π · 4095)`. La tensión de salida es `3,3 V · código / 4096` (*por verificar* en la hoja de datos; la referencia del DAC es la alimentación del Pmod). Como la fase se toma módulo 2π, una fase muy próxima a 2π da un código cercano a 4095 y una próxima a 0 da un código cercano a 0: en la traza son extremos opuestos.

### Tiempos de un paso

Un tic son 100 µs (`TICK_DIV` = 10000 ciclos de 100 MHz).

| tramo | duración | qué ocurre |
|---|---|---|
| rampa | 32 tics, 3,2 ms | escalón: una escritura al inicio. Rampa: 32 escrituras, una por tic |
| asentamiento | 8 tics, 0,8 ms | códigos quietos |
| compuerta | 8 tics, 0,8 ms | `gate` en alto; el DAC no recibe ninguna trama |
| espera | 8 tics, 0,8 ms | `gate` en bajo |

Paso completo: 5,6 ms. Capa de 6 pasos: 33,6 ms. La compuerta sube 4,0 ms después del inicio del paso. Todo es parámetro del módulo superior.

### Protocolo del DAC

| dato | valor usado | hoja de datos | estado |
|---|---|---|---|
| trama | 16 bits, MSB primero: 2 sin uso, 2 de modo (00 = normal), 12 de dato | igual | verificado (TI SNAS265J, vía buscador) |
| flanco de muestreo | el DAC lee en el flanco de bajada de SCLK; la FPGA cambia el dato en el de subida | bajada | verificado |
| actualización | en la 16.ª bajada; SYNC se mantiene bajo hasta después | 16.ª bajada; si SYNC sube antes, la trama se descarta | verificado |
| frecuencia de SCLK | 12,5 MHz | máx. 30 MHz | verificado |
| SYNC en alto entre tramas | 80 ns como mínimo | mín. 20 ns (2,7–3,6 V, todo el rango de temperatura) | verificado |
| preparación y retención del dato | 40 ns y 40 ns | unos pocos ns | *por verificar* |
| SYNC a primera bajada de SCLK | 40 ns | unos pocos ns | *por verificar* |
| tiempo de asentamiento de la salida | se esperan 0,8 ms antes de la compuerta | del orden de 10 µs | *por verificar* |
| pines del Pmod DA2 | 1 SYNC, 2 DINA, 3 DINB, 4 SCLK, 5 GND, 6 VCC | pines 1 y 2 confirmados en el manual de Digilent | pines 3 y 4 *por verificar* |
| conector de las salidas analógicas | — | — | *por verificar* en la serigrafía del módulo |

No se pudo abrir ni ti.com ni digilent.com desde el entorno donde se escribió esto; lo verificado procede de los extractos del buscador. Antes de conectar: abrir el manual de referencia del Pmod DA2 y comprobar los pines 3 y 4 y el conector de salida.

## 3. Generar la tabla

Con los pesos entrenados (en el PC donde están los `logs/`):

```
python scripts/hw_sweep.py logs/<corrida> --quick
python fpga/scripts/phases_to_mem.py analysis/hw_sweep/gates_<corrida>.csv -o fpga/data/phase_table.mem
```

`--ramp none|all|alternate` elige escalón o rampa por capa (por omisión, escalón en las capas pares y rampa en las impares). `--elements U1` limita la tabla a una malla. El script también acepta un CSV sencillo `layer,phase_a,phase_b[,ramp,gate]` en radianes (`data/example_simple.csv`).

El `phase_table.mem` incluido sale de `data/gates_example.csv`, que es **sintético** (fases al azar con semilla fija, 3 capas), no un circuito entrenado. Hay que sustituirlo por el de una corrida real antes de tomar las capturas del libro.

## 4. Simulación

```
pip install cocotb            # 2.x
# Icarus Verilog: apt install iverilog (Linux), o el instalador de bleyer.org (Windows)
python fpga/tb/run_tests.py
```

Resultado del 18 nov 2026: 5 comprobaciones del script, 6 pruebas del controlador y 7 del módulo superior, todas correctas.

| prueba | qué comprueba |
|---|---|
| `test_frames_bit_exact` | 72 tramas (extremos, patrones alternos y 64 al azar, semilla 20261118) llegan bit a bit a los dos canales |
| `test_timing_exact` | semiperíodo, preparación y retención de 40 ns; actualización en la 16.ª bajada; duración de la trama |
| `test_back_to_back`, `test_start_ignored_while_busy`, `test_reset_mid_frame`, `test_idle_after_reset` | tramas seguidas (SYNC 90 ns en alto), orden ignorada durante una trama, reinicio a media trama, reposo |
| `test_table_file` | la memoria contiene exactamente el archivo `.mem` |
| `test_play_all` | tabla completa dos veces: cada código, cada punto de rampa, cada flanco de compuerta y de disparo, en el ciclo de reloj exacto |
| `test_ramp_is_linear` | rampas monótonas, uniformes, sin sobrepaso y que acaban en el objetivo |
| `test_single_layer`, `test_missing_layer`, `test_layer_change_on_the_fly` | selección de capa con los interruptores |
| `test_reset_while_gate_open` | el reinicio baja la compuerta y la tabla vuelve a empezar |

El modelo del DAC (`tb/da2_model.py`) rechaza cualquier trama que no tenga 16 flancos de bajada, bits de modo distintos de 00, SCLK por encima de 30 MHz, SYNC en alto menos de 20 ns o datos que cambien a menos de 20 ns del flanco de muestreo. En simulación los tics se acortan (200 ciclos) para que la tabla dure milisegundos; la lógica es la misma.

Además, yosys 0.33 (`synth_xilinx`) sintetiza el diseño sin errores: unos 194 biestables y 240 LUT. Es una comprobación de que el código es sintetizable, no sustituye a Vivado.

## 5. Síntesis en Vivado

```
cd fpga
vivado -mode batch -source scripts/build.tcl -tclargs 100     # Arty A7-100T (xc7a100tcsg324-1)
vivado -mode batch -source scripts/build.tcl -tclargs 35      # Arty A7-35T  (xc7a35ticsg324-1L)
```

La placa de la tesis es la **A7-100T**, que es el valor por omisión del script. Sale `build/arty_phase_demo_<placa>.bit` y los informes de tiempos y de uso. Los pines son los mismos en las dos placas; solo cambia el dispositivo. El código de dispositivo de la 35T está *por verificar* en el selector de placas de Vivado.

Con la interfaz gráfica: proyecto RTL nuevo con la placa Arty A7 que corresponda, añadir `rtl/*.v` y `data/phase_table.mem` como fuentes de diseño y `constr/arty_a7.xdc` como restricciones, módulo superior `arty_phase_demo`, generar el *bitstream*.

Qué revisar después de implementar:

1. `timing_<placa>.rpt`: holgura positiva (WNS ≥ 0) y ninguna ruta sin restringir inesperada en `check_timing_<placa>.rpt`.
2. En el registro de síntesis, que `$readmemh` haya leído `phase_table.mem` (si no la encuentra, la tabla queda vacía y las salidas no se mueven).
3. `io_<placa>.rpt`: los 6 pines de salida con el biestable en la celda de E/S (propiedad IOB).

## 6. Cableado

| señal | conector de la Arty | pin FPGA |
|---|---|---|
| Pmod DA2 completo | JA, fila superior (pines 1 a 6) | G13, B11, A11, D12 |
| `gate` | JD pin 1 | D4 |
| `trig` | JD pin 2 | D3 |
| masa de las sondas | JD pin 5 (GND) | — |

- El Pmod DA2 se enchufa directamente en la fila superior de JA; se alimenta a 3,3 V desde el propio conector.
- Canal 1 del osciloscopio a la salida A del Pmod, canal 2 a la salida B, disparo externo (o un tercer canal) a `trig`.
- Interruptores SW3..SW0: número de capa en binario. Los cuatro arriba: tabla completa.
- BTN0: reinicio. Los LED LD4..LD7 muestran la capa en curso.

## 7. Qué medir

1. Niveles: con la tabla de ejemplo, comparar cada meseta con la tensión que el script escribe en los comentarios del `.mem` (por ejemplo, código 3691 → 2,974 V). Tolerancia: la de la alimentación de 3,3 V.
2. Duración del paso: 5,6 ms entre escalones.
3. Rampa: 3,2 ms, 32 peldaños de 100 µs.
4. Compuerta: sube 4,0 ms después del inicio del paso, dura 0,8 ms, y las dos salidas analógicas están quietas mientras está en alto.
5. Trama serie: SCLK a 12,5 MHz, 16 pulsos por trama, SYNC en bajo 1,32 µs.

## 8. Capturas

| n.º | interruptores | canales | base de tiempos | qué muestra |
|---|---|---|---|---|
| C0 | 1111 | A, B; disparo en `trig` | 20 ms/div | tabla completa, todas las capas seguidas |
| C1..Cn | número de capa | A, B; disparo en `trig` | 5 ms/div | **una por capa**: los 6 pares de fases de esa capa |
| Cn+1 | una capa en escalón | A y `gate` | 1 ms/div | un paso: escalón, asentamiento y compuerta |
| Cn+2 | una capa en rampa | A y `gate` | 1 ms/div | un paso en rampa y su compuerta |
| Cn+3 | cualquiera | SCLK y DINA; disparo en SYNC | 200 ns/div | una trama de 16 bits (opcional; mejor con analizador lógico) |

Guardar cada captura con el nombre de la corrida, la capa y la fecha, y anotar junto a C1..Cn la fila correspondiente del `.mem`, para que el lector pueda contrastar tensiones y códigos.

## 9. Pendiente

- Sintetizar en Vivado y revisar los tres informes del apartado 5.
- Verificar en el manual del Pmod DA2 y en la hoja de datos del DAC121S101 las filas marcadas *por verificar*.
- Generar la tabla con una corrida entrenada y tomar las capturas.
