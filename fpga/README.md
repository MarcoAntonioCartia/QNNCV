# Demostración en FPGA: capa digital de control y lectura

Reproducción de una tabla de fases en los dos canales de un Pmod DA2, con un pulso de compuerta por paso, sobre una Arty A7. Base del Apéndice C de la tesis (sección 5.5).

**Estado (18 nov 2026):** dos diseños escritos y probados en simulación (Icarus Verilog 12.0 + cocotb 2.1.0): la tabla de fases (secciones 1 a 9) y la extensión de lazo cerrado con filtros IIR y FIR (sección 10). 35 pruebas: 34 pasan y 1 se omite a propósito. **Sintetizado e implementado con Vivado 2025.2 el 20 nov 2026 para la A7-100T (sección 5); no probado en la placa.** Los datos marcados *por verificar* no se pudieron contrastar con la hoja de datos original.

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

Cada Mach–Zehnder de las mallas U1 y U2 de una capa es un paso. Con 3 modos son 6 pasos por capa. Caben 256 pasos; el script escribe las 256 palabras y rellena las que sobran con pasos vacíos.

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
| SYNC en alto entre tramas | 90 ns como mínimo | mín. 20 ns (2,7–3,6 V, todo el rango de temperatura) | verificado |
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

El `phase_table.mem` incluido sale de `data/gates_trained.csv`: un circuito **entrenado** (familia de anillos, generador de VC de 3 modos y 6 capas, crítico denso, semilla 5, la mediana de las 20 semillas por W1 de fidelidad). Son 36 pasos en las capas 0 a 5. `data/gates_example.csv` es **sintético** (fases al azar con semilla fija, 3 capas) y queda solo como ejemplo de formato y para las pruebas. En la tabla entrenada las fases internas quedan todas cerca de π (códigos 1663 a 2044, de 1,34 a 1,65 V), así que el canal A apenas se mueve; 15 de las 36 fases externas caen junto al salto 0/2π, de modo que fases casi iguales aparecen en extremos opuestos del canal B.

## 4. Simulación

```
pip install cocotb            # 2.x
# Icarus Verilog: apt install iverilog (Linux), o el instalador de bleyer.org (Windows)
python fpga/tb/run_tests.py
```

Resultado del 18 nov 2026 para este diseño: 5 comprobaciones del script, 6 pruebas del controlador y 7 del módulo superior, todas correctas. `python fpga/tb/run_tests.py spi demo` ejecuta solo las 13 de cocotb; sin argumentos se ejecuta todo, incluida la extensión (unos 4 minutos).

Repetido el 21 nov 2026 en Windows, después del cambio de `trig` y con la tabla entrenada. Se usó Icarus Verilog 12.0 (instalador de bleyer.org, `winget install Icarus.Verilog`), cocotb 2.1.0 y Python 3.11. Pasaron las comprobaciones de los dos scripts y las 35 pruebas: 34 correctas y 1 omitida a propósito (spi 6, demo 7, adc 3, iir 4, pi 4, fir 4, lock 7). Tardó 149 s.

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

El modelo del DAC (`tb/da2_model.py`) rechaza cualquier trama que no tenga 16 flancos de bajada, bits de modo distintos de 00, SCLK por encima de 30 MHz, SYNC en alto menos de 20 ns o datos que cambien a menos de 20 ns del flanco de muestreo. En simulación se acortan los tics (200 ciclos), la rampa (8 puntos), el asentamiento, la compuerta y la espera (2, 3 y 2 tics) y la memoria (16 palabras) para que la tabla dure milisegundos; la lógica es la misma.

Además, yosys 0.33 (`synth_xilinx`) sintetiza el diseño sin errores: unos 194 biestables y 234 LUT. Es una comprobación de que el código es sintetizable, no sustituye a Vivado.

## 5. Síntesis en Vivado

```
cd fpga
vivado -mode batch -source scripts/build.tcl -tclargs 100 phase    # Arty A7-100T (xc7a100tcsg324-1)
vivado -mode batch -source scripts/build.tcl -tclargs 35 phase     # Arty A7-35T  (xc7a35ticsg324-1L)
```

La placa de la tesis es la **A7-100T**, que es el valor por omisión del script. Sale `build/phase/arty_phase_demo_<placa>.bit` y los informes de tiempos y de uso. Los pines son los mismos en las dos placas; solo cambia el dispositivo. El código de dispositivo de la 35T está *por verificar* en el selector de placas de Vivado.

Con la interfaz gráfica: proyecto RTL nuevo con la placa Arty A7 que corresponda, añadir `rtl/*.v` y `data/phase_table.mem` como fuentes de diseño y `constr/arty_a7.xdc` como restricciones, módulo superior `arty_phase_demo`, generar el *bitstream*.

Qué revisar después de implementar:

1. `timing_<placa>.rpt`: holgura positiva (WNS ≥ 0) y ninguna ruta sin restringir inesperada en `check_timing_<placa>.rpt`.
2. En el registro de síntesis, que `$readmemh` haya leído `phase_table.mem` (si no la encuentra, la tabla queda vacía y las salidas no se mueven).
3. `io_<placa>.rpt`: los 6 pines de salida con el biestable en la celda de E/S (propiedad IOB).

Resultado del 21 nov 2026 para `phase` (con la tabla entrenada) y del 20 nov 2026 para `lock` (Vivado 2025.2, xc7a100tcsg324-1), sin avisos ni avisos críticos en ninguno de los dos diseños:

| diseño | WNS | WHS | LUT | biestables | BRAM | DSP |
|---|---|---|---|---|---|---|
| `phase` | 2,409 ns | 0,122 ns | 225 | 209 | 0 | 0 |
| `lock` | 2,115 ns | 0,121 ns | 425 (18 como RAM distribuida) | 432 | 1 RAMB36 + 1 RAMB18 (tabla de franja 4096 × 12) | 2 DSP48E1 (uno en `iir_biquad`, uno en `fir_serial`) |

`$readmemh` leyó `phase_table.mem`, `fringe_lut.mem` y `fir_coeffs.mem`. Todos los puertos quedaron en los pines de la sección 6 y del apartado de cableado de la sección 10, y los biestables de las salidas serie, `gate` y `trig` (y de las entradas `ad1_d0`/`ad1_d1`) quedaron en las celdas de E/S. Único cambio necesario: en `arty_lock_demo.v`, `trig` se escribe en cada ciclo como copia de `sq` (10 ns de retraso), porque con la escritura condicional Vivado realimentaba el biestable a través de una LUT y no podía meterlo en la celda de E/S.

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

1. Niveles: comparar cada meseta con la tensión que el script escribe en los comentarios del `.mem` (por ejemplo, código 1978 → 1,594 V en el primer paso). Tolerancia: la de la alimentación de 3,3 V.
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

- Verificar en el manual del Pmod DA2 y en la hoja de datos del DAC121S101 las filas marcadas *por verificar*.
- Tomar las capturas (la tabla ya es la de una corrida entrenada).

## 10. Extensión: enganche de fase (IIR) y lectura con compuerta (FIR)

Segundo diseño, independiente del anterior (`rtl/arty_lock_demo.v`, otro *bitstream*). Añade un Pmod AD1 (dos AD7476A, 12 bits) y cierra un lazo con una señal analógica real.

### Qué demuestra y qué no

- **Lazo cerrado.** El canal A del DAC excita una red RC (10 kΩ, 100 nF, τ = 1 ms), que hace de desfasador termo-óptico lento. El canal 0 del ADC lee su salida. Dentro de la FPGA esa tensión, más una deriva lenta opcional, se toma como fase de un interferómetro y pasa por una tabla de seno de 4096 entradas: es la señal del «detector». Un controlador PI mantiene el detector a media franja moviendo el DAC.
- **Lectura.** El canal 1 del ADC pasa por un FIR de 64 coeficientes adaptado a la forma de un pulso. Una compuerta se abre cada 10 ms; la salida del FIR para la última muestra dentro de la compuerta es el valor de ese disparo, como la cuadratura de un disparo en una detección homodina.
- **Es real:** el DAC, el ADC, el retardo de la red RC, el ruido y la cuantización de ambos, y toda la aritmética en punto fijo.
- **Es emulado:** la franja (tabla de seno), la deriva (rampa triangular de ±1/4 de vuelta) y el pulso de lectura (hay que inyectarlo con el generador). No hay interferómetro ni luz.

### Bloques

```
            ┌────────── FPGA ───────────────────────────────────────────────┐
 ADC ch0 ──►│ ad1_spi ─► fase = 16·(código − 2048) + deriva ─► tabla seno ──┼─► «detector»
 (salida RC)│                                                    │          │
            │   consigna (2048 ± 300 con SW3) ──► error = consigna − detector│
            │                                       │                       │
            │                              iir_biquad (PI, anti-saturación) │
            │                                       │                       │
 DAC A ◄────┤ da2_spi ◄── 2048 + control ◄──────────┘                       │
   │        │                                                               │
   └─ R ─┬──┘ (red RC externa: 10 kΩ en serie, 100 nF a masa)               │
         C  │                                                               │
            │ ADC ch1 ─► fir_serial (64 coef.) ─► valor del disparo ────────┼─► DAC B (SW2)
            │              compuerta: 80 muestras de cada 1000 ─────────────┼─► gate (JB1)
            └───────────────────────────────────────────────────────────────┘
```

| archivo | función |
|---|---|
| `rtl/ad1_spi.v` | lector serie del Pmod AD1, los dos canales en la misma conversión |
| `rtl/iir_biquad.v` | sección IIR de segundo orden, forma directa I, salida con saturación |
| `rtl/fir_serial.v` | FIR de 64 coeficientes con un solo multiplicador |
| `rtl/arty_lock_demo.v` | módulo superior del lazo y de la lectura |
| `scripts/design_filters.py` | diseño del PI y del FIR; escribe `data/fringe_lut.mem` y `data/fir_coeffs.mem` |
| `constr/arty_a7_lock.xdc` | pines |

### El IIR: controlador PI

Una sola sección de segundo orden, `y[n] = B0·x[n] + B1·x[n−1] + B2·x[n−2] + NA1·y[n−1] + NA2·y[n−2]`. Con `NA1 = 1` y `B2 = NA2 = 0` es un PI en forma incremental; cambiando los cinco coeficientes es un PID, una muesca o un pasa-bajos.

- **Formatos:** coeficientes Q3.15 (18 bits), señales Q13.12 (25 bits). Un producto cabe en un bloque DSP48E1. Un multiplicador se usa cinco veces; el resultado queda registrado 8 ciclos después del inicio.
- **Bits fraccionarios en el estado:** la ganancia integral por muestra es 0,016. Sin fracción, un error de 1 código no movería nunca la salida y el lazo quedaría con error permanente.
- **Anti-saturación:** la salida se recorta a ±2047 y el valor recortado es el que se realimenta, así que el integrador no sigue contando mientras el DAC está en el tope.

Diseño (`scripts/design_filters.py`): planta `G(s) = K/(τs + 1)` con τ = 1 ms y K = 3,14 (pendiente de la franja a media altura, en códigos por código). El cero del PI se pone sobre el polo de la planta (Ti = τ) y Kp = 1,59 fija la constante de tiempo de lazo cerrado en 0,2 ms, cinco veces más rápido que la planta sola. Discretización bilineal a 100 kS/s: `B0 = 52438`, `B1 = −51916`.

### El FIR: lectura adaptada

`y[n] = Σ h[k]·x[n−k]`, 64 coeficientes Q1.17. Los coeficientes son la forma del pulso (gaussiana de σ = 9 muestras, centrada en la muestra 36 de 64) con la media restada, invertida en el tiempo y normalizada: para una entrada `a·pulso + constante`, la salida al final de la ventana es `a`. La suma de los coeficientes es exactamente cero, así que cualquier nivel constante se rechaza por completo.

### Arranque y captura

La franja es periódica. La pendiente solo es positiva a menos de un cuarto de vuelta del punto de trabajo, pero el error conserva el signo correcto hasta media vuelta, así que el lazo engancha desde ahí; más allá engancha en la franja siguiente. Falla solo cuando el punto estable más cercano queda fuera del recorrido del DAC: entonces el controlador acaba en el tope.

Tras un reinicio el condensador está descargado, a media vuelta exacta (equilibrio inestable). Sin espera, el lazo satura unas 20 muestras y engancha en unos 2 ms (simulado con `WARM_SAMPLES = 1`). Para arrancar sin ese transitorio, el DAC espera 10 ms a media escala (1024 muestras, diez constantes de tiempo de la red RC) antes de dejar actuar al controlador, y la deriva emulada no arranca hasta entonces.

Una versión anterior de este apartado decía que sin la espera el controlador se quedaba en el tope y que el margen de captura era de un cuarto de vuelta. Las dos afirmaciones eran erróneas; se corrigieron tras la verificación independiente (véase `GUIDE_digital_control.md`).

### Pruebas de la extensión

| banco | pruebas | qué comprueba |
|---|---|---|
| `adc` | 3 | 72 conversiones bit a bit en los dos canales; el modelo pone la línea en estado indeterminado fuera de la ventana válida de cada bit; período de SCLK y duración |
| `iir`, `pi` | 4 + 4 (1 omitida) | 1120 muestras (900 al azar y tramos constantes en los extremos) idénticas bit a bit a la referencia en Python, con y sin recorte; impulso, escalón, borrado; anti-saturación del PI |
| `fir` | 4 | 550 muestras al azar idénticas a la referencia; coeficientes cargados; un nivel constante da exactamente 0; amplitud de un pulso recuperada con error de 2 códigos como máximo |
| `lock` | 7 | lazo cerrado con modelo de la red RC |

Sin argumentos, `run_tests.py` ejecuta además 4 comprobaciones de `design_filters.py` (archivos al día, coeficientes del módulo superior iguales a los del diseño, constante de tiempo del lazo, FIR adaptado).

Resultados del banco `lock` (18 nov 2026):

- Cada valor interno de cada muestra (tabla, error, salida del PI, códigos del DAC) coincide con la referencia calculada a partir de los códigos del ADC.
- Respuesta al escalón: 63 % a las 20 muestras en los cuatro escalones medidos (diseño: 20), sobrepaso inferior al 5 %, error final de 4 códigos como máximo. La planta sola tarda 100.
- Deriva de ±1/4 de vuelta, 32 veces más rápida que en la placa: el detector se desvía 2047 códigos con el lazo abierto y 129 con el lazo cerrado.
- Realimentación rota (ADC fijo): el controlador va al tope y enciende el aviso; al restaurarla, el error vuelve a la ventana de enganche en 167 muestras.
- Lectura: seis pulsos de amplitudes entre −1500 y 1500 sobre un nivel de 300 códigos, con señal ajena fuera de la ventana: el valor del disparo es la amplitud exacta en los seis.

En la simulación la muestra dura 400 ciclos de reloj en vez de 1000 y la constante de tiempo de la red se escala igual, así que la dinámica del lazo en muestras es la de la placa. También se acortan la compuerta (cada 200 muestras), el escalón (cada 150) y la espera de arranque (500). yosys sintetiza el diseño: 2 bloques DSP48E1 (uno por filtro), la tabla en memoria de bloque, unos 470 biestables y 590 LUT.

### Tiempos

| magnitud | valor |
|---|---|
| muestreo | 100 kS/s (un ciclo de control cada 10 µs) |
| retardo del muestreo a la actualización del DAC | unos 3,4 µs (conversión 1,41 µs, cálculo 0,77 µs, trama hasta la 16.ª bajada 1,24 µs) |
| constante de tiempo de la planta | 1 ms (100 muestras) |
| constante de tiempo en lazo cerrado | 0,2 ms (20 muestras) |
| escalón de consigna (SW3) | ±300 códigos (±0,24 V en el DAC B), cada 5 ms |
| deriva (SW1) | triangular, ±1/4 de vuelta, período 0,66 s |
| compuerta | 0,8 ms cada 10 ms; ventana del FIR: los últimos 0,64 ms |

### Protocolo del ADC

| dato | valor usado | hoja de datos | estado |
|---|---|---|---|
| conversión | CS baja y muestrea; 16 flancos de bajada de SCLK; 4 ceros y 12 bits, MSB primero | igual | verificado (extractos de la hoja del AD7476A y del manual del Pmod AD1, vía buscador) |
| frecuencia de SCLK | 12,5 MHz | máx. 20 MHz | verificado |
| pines del Pmod AD1 | 1 CS, 2 D0, 3 D1, 4 CLK, 5 GND, 6 VCC | igual | verificado en copias del manual; confirmar en el original |
| lectura del bit | 70 ns después del flanco que lo saca y 10 ns antes del que lo cambia | tiempo de acceso de decenas de ns | *por verificar* |
| CS en alto entre conversiones | 8,7 µs | decenas de ns | *por verificar* |
| filtro de entrada del Pmod AD1 | — | paso bajo de dos polos; frecuencia de corte no confirmada | *por verificar*: si está cerca de 1 kHz altera la planta |

### Cableado y uso

| señal | conector |
|---|---|
| Pmod DA2 | JA, fila superior |
| Pmod AD1 | JD, fila superior |
| `gate` | JB pin 1 (E15) |
| `trig` | JB pin 2 (E16): cuadrada de la consigna, siempre activa |
| red RC | salida A del DA2 → 10 kΩ → nodo; nodo → 100 nF → masa; nodo → entrada A0 del AD1 |
| pulso de lectura | JDS6600 → entrada A1 del AD1, entre 0 y 3,3 V, en ráfaga disparada por `gate` si el generador lo permite (*por verificar*) |

JB no tiene resistencias en serie: solo sondas, nunca una señal que empuje contra la FPGA. La entrada A1 no debe salir del rango 0–3,3 V.

| interruptor | abajo | arriba |
|---|---|---|
| SW0 | lazo abierto, DAC A a media escala | lazo cerrado |
| SW1 | sin deriva | deriva activa |
| SW2 | DAC B = señal del detector | DAC B = valor del último disparo |
| SW3 | consigna fija | consigna en onda cuadrada |

LED: LD4 lazo cerrado, LD5 deriva, LD6 enganchado (error menor de 64 códigos durante 256 muestras), LD7 controlador en el tope.

Síntesis: `vivado -mode batch -source scripts/build.tcl -tclargs 100 lock` (sale en `build/lock/`).

### Capturas

| n.º | interruptores (SW3..SW0) | canales | base de tiempos | qué muestra |
|---|---|---|---|---|
| L1 | 0010 | DAC B, DAC A | 100 ms/div | lazo abierto con deriva: el detector recorre la franja, el control no se mueve |
| L2 | 0011 | DAC B, DAC A | 100 ms/div | lazo cerrado: el detector quieto a 1,65 V, el control dibuja la deriva invertida |
| L3 | 1001 | DAC B, nodo RC; disparo en `trig` | 200 µs/div | respuesta al escalón: medir el 63 % (0,2 ms previstos) y el sobrepaso |
| L4 | 1001 | DAC A, nodo RC; disparo en `trig` | 200 µs/div | esfuerzo de control: el pico proporcional que adelanta a la red RC |
| L5 | 0001, desconectando y reconectando la entrada A0 del AD1 | DAC A, DAC B | 5 ms/div | saturación (LD7) y reenganche sin cola de integrador |
| L6 | 0100 | entrada A1, `gate`, DAC B | 2 ms/div | lectura: pulso dentro de la compuerta y valor retenido por disparo |

Para L3, comparar con la planta sola (SW0 abajo y un escalón del generador sobre la red RC): 1 ms frente a 0,2 ms.

### Qué esperar distinto de la simulación

- La tolerancia de R y C mueve el polo de la planta; el cero del PI ya no lo cancela exactamente y la respuesta tendrá una cola lenta pequeña o algo de sobrepaso. Medir τ con el osciloscopio y, si hace falta, volver a ejecutar `design_filters.py` con ese valor.
- El filtro de entrada del Pmod AD1 añade retardo que la simulación no incluye.
- El ruido del ADC aparece en el DAC A multiplicado por Kp (1,6).
