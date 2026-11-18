# Guía: el control digital de la carpeta `fpga/`

Escrita el 18 nov 2026 para leer junto al código de `fpga/rtl/` y a `fpga/README.md`. El README dice qué hay y cómo usarlo; esta guía explica por qué funciona. Los circuitos están probados en simulación con parámetros acortados. Los márgenes de estabilidad, la banda muerta y las cifras de ruido son cálculos, no pruebas. Nada se ha sintetizado con Vivado ni medido en la placa.

---

## Parte A. Lazo abierto: de los ángulos a las tensiones

### A.1 Dos diseños

| | diseño 1: tabla de fases | diseño 2: enganche y lectura |
|---|---|---|
| módulo superior | `arty_phase_demo.v` | `arty_lock_demo.v` |
| lazo | abierto | cerrado |
| filtros | ninguno | PI (IIR) dentro del lazo, FIR fuera |
| qué es físico | DAC | DAC, ADC, red RC |
| qué es emulado | — | franja del interferómetro y deriva |

No están conectados entre sí. El lazo mantiene un punto de trabajo a media franja; no fija los ángulos entrenados del diseño 1.

### A.2 El enlace serie con el DAC

Cada escritura es una trama de 16 bits: 2 sin uso, 2 de modo (00) y 12 de dato, el más significativo primero. El DAC lee la línea de datos en cada **bajada** del reloj serie; la FPGA la cambia en la **subida**, medio período antes. A 12,5 MHz eso deja 40 ns de preparación y 40 ns de retención. La salida cambia en la bajada número 16; si la selección sube antes, la trama se descarta.

El lector del ADC funciona al revés: al bajar la selección el ADC muestrea y pone el primer bit; cada bajada del reloj saca el siguiente (cuatro ceros y doce bits). La FPGA lee cada bit 70 ns después del flanco que lo saca y 10 ns antes del que lo cambia.

### A.3 De los ángulos a los códigos

Una malla rectangular de M(M−1)/2 Mach–Zehnder más M fases de salida realiza cualquier interferómetro de M modos (Clements y otros 2016). Cada Mach–Zehnder tiene una fase interna (qué fracción de la luz cruza) y una externa (fase relativa entre los dos modos). `scripts/hw_sweep.py` las calcula; la demo pone la interna en el canal A y la externa en el B, un Mach–Zehnder por paso.

```
código  = round((φ mod 2π) / 2π · 4095)
tensión = 3,3 V · código / 4096          (fórmula por verificar en la hoja de datos)
```

La correspondencia es ilustrativa: un desfasador termo-óptico responde a la potencia disipada, no a la tensión, y necesita calibración.

### A.4 El secuenciador y la rampa

Máquina de seis estados: buscar palabra, decodificar, rampa (32 tics), asentamiento (8), compuerta (8), espera (8). Un tic son 100 µs y un paso dura siempre 56 tics.

La rampa lineal no usa divisor. Con el código anterior `p` y el nuevo `q`:

```
acumulador ← 32·p ;  en cada tic: acumulador ← acumulador + (q − p)
código tras el tic i = p + floor(i·(q − p)/32) ;  tras el tic 32 = q exacto
```

La llegada exacta se debe a acumular la diferencia sin redondear, no a que 32 sea potencia de dos; la potencia de dos solo hace gratis la división. Es un analizador diferencial digital (DDA) en punto fijo.

La compuerta sube como mínimo 0,8 ms después de la última escritura al DAC y mientras está en alto no se envía ninguna trama.

---

## Parte B. Lazo cerrado: el controlador PI

### B.1 Planta

Red RC de 10 kΩ y 100 nF: `G(s) = 1/(τs + 1)`, τ = 1 ms. El controlador muestrea cada T = 10 µs y el DAC retiene el valor entre escrituras. Si se ignoran el retardo de cálculo (3,4 µs), la cuantización y el filtro de entrada del Pmod AD1, el equivalente discreto es exacto en los instantes de muestreo:

```
G(z) = (1 − a)/(z − a),   a = e^(−T/τ) = 0,99005
```

### B.2 Franja

```
y = 2048 + 2047·sen φ,   φ = 2π·(v − 2048)/4096 + deriva
K = dy/dv a media franja = 2047·2π/4096 = 3,14 códigos de detector por código de ADC
```

- La pendiente vale K·cos φ. Es positiva solo a menos de un cuarto de vuelta del punto de trabajo.
- El controlador se mueve según el signo del error, que es correcto hasta media vuelta. El lazo engancha desde ahí; a media vuelta exacta hay un equilibrio inestable y más allá engancha en la franja siguiente.
- Falla solo cuando el punto estable más cercano queda fuera del recorrido del DAC.

### B.3 PI discreto

```
C(s) = Kp·(1 + 1/(Ti·s))
bilineal:  s → (2/T)·(z − 1)/(z + 1)
C(z) = (b0·z + b1)/(z − 1),  b0 = Kp·(1 + T/(2Ti)),  b1 = −Kp·(1 − T/(2Ti))
u[n] = u[n−1] + b0·e[n] + b1·e[n−1]
```

### B.4 Diseño por cancelación

Con Ti = τ el cero del PI, (2Ti − T)/(2Ti + T) = 199/201 = 0,990050, cae sobre el polo de la planta. El lazo abierto queda en un integrador:

```
L(z) ≈ g/(z − 1),   g = Kp·K·(1 + T/(2Ti))·(1 − a)
lazo cerrado: polo en 1 − g, respuesta 1 − (1 − g)^n
```

Se pide una constante de tiempo de 0,2 ms (20 muestras), cinco veces más rápida que la planta.

| magnitud | valor |
|---|---|
| Kp = τ/(τ_lc·K) | 1,5923 |
| b0, b1 | 1,6003 y −1,5844; en Q3.15: 52438 y −51916 |
| g | 0,0500 |
| polo de lazo cerrado | 0,9500 |
| constante de tiempo | 19,5 muestras |

En la simulación del circuito completo el 63 % se alcanza en la muestra 20.

### B.5 Estabilidad y precisión

- **Margen de fase:** cruce en g/T = 5000 rad/s (unos 800 Hz). El integrador continuo tendría 90°; el discreto pierde ω·T/2 = 1,4° y el retardo de 3,4 µs otro 1,0°. Quedan 87,6° (cálculo).
- **Escalón:** error cero en el modelo lineal. En el circuito queda una oscilación de ±1 código de ADC, porque un código de ADC son 3,14 de detector.
- **Rampa:** error = pendiente/g, con la pendiente en códigos de detector por muestra. Simulación: 6,28/0,05 = 126 previstos, 129 de máximo medido. Placa: 4 previstos, no simulado.
- **Cancelación imperfecta:** si la τ real es mayor que Ti hay sobrepaso; si es menor, una cola lenta. Entre 0,3 y 5 ms el lazo sigue estable con margen de 60° o más (cálculo); con 1,2 ms el sobrepaso es del 1,7 %.

### B.6 Punto fijo

- Coeficientes del PI en Q3.15 (18 bits), señales en Q13.12 (25 bits): un producto cabe en un bloque DSP48E1.
- **Fracción en el estado:** la ganancia integral por muestra es 522/32768 = 0,0159. Con estado entero y redondeo al más próximo, errores de hasta 31 códigos no moverían la salida. Con 12 bits de fracción, un error de un código ya la mueve.
- **Anti-saturación:** la salida se recorta a ±2047 y el valor recortado es el que se realimenta. En forma incremental, el integrador no puede pasar del tope y sale de él en la primera muestra en que b0·e[n] + b1·e[n−1] es negativo.
- **Arranque:** con el condensador descargado el lazo parte del equilibrio inestable; sin espera satura unas 20 muestras y engancha en unos 2 ms. El diseño espera 1024 muestras con el DAC a media escala para evitar ese transitorio.

### B.7 PID, IIR y FIR

Un PID discreto es un IIR: el PI es de primer orden y el PID de segundo. `iir_biquad.v` es la sección general de segundo orden; con otros cinco coeficientes sería un PID, una muesca o un paso bajo.

| | PID | IIR general | FIR |
|---|---|---|---|
| puede integrar | sí | sí | no |
| estabilidad del filtro solo | marginal (polo en z = 1) | depende de los polos | siempre estable |
| fase lineal exacta | no | no | con coeficientes simétricos o antisimétricos |
| retardo | pequeño | depende de los polos | (L−1)/2 muestras si es de fase lineal |
| uso habitual | dentro del lazo | dentro del lazo | fuera: medida y lectura |

---

## Parte C. El FIR de lectura

### C.1 Qué calcula

Un número por compuerta: la suma ponderada de las últimas 64 muestras, `valor = Σ w[k]·x[k]`. Es la versión digital de integrar una fotocorriente sobre la duración de un pulso, que es como una detección homodina pulsada obtiene un valor de cuadratura por pulso (Lvovsky y Raymer 2009). La escala no es la de un experimento: aquí la ventana dura 0,64 ms.

### C.2 Los pesos

Modelo: `x[k] = A·m[k] + c + ruido`, con m[k] la forma conocida del pulso (gaussiana de anchura 9 muestras centrada en la 36) y c un nivel constante desconocido. El estimador de mínimos cuadrados de A es

```
w[k] = (m[k] − m̄) / Σ(m[j] − m̄)²
Σ w[k] = 0  →  un nivel constante da exactamente 0
Σ w[k]·m[k] = 1  →  un pulso de amplitud A da A (salvo redondeo: 2 códigos como máximo)
```

En el archivo los coeficientes van invertidos, h[k] = w[63−k], porque un FIR multiplica h[0] por la muestra más reciente.

### C.3 Coste e hipótesis

Con ruido blanco de desviación s por muestra, el valor tiene desviación s·√(Σw²) = 0,353·s; el filtro adaptado sin resta de media daría 0,250·s. El factor 1,41 es propio de este pulso y esta ventana. Hipótesis: forma y sincronismo conocidos, ruido blanco, nivel constante dentro de la ventana. Una pendiente no se rechaza y un pulso desplazado 4 muestras pierde un 10 %.

### C.4 Un solo multiplicador

64 productos con un multiplicador usado 64 veces y un acumulador: 67 ciclos de reloj, 0,67 µs, de los 1000 que hay entre muestras.

---

## Parte D. Verificación

### D.1 Del circuito

- Modelos de los dos Pmod que rechazan cualquier trama fuera de protocolo.
- Referencias en Python exactas al bit para el IIR y el FIR.
- Modelo de la red RC con su exponencial exacta.
- Dos pruebas por mutación hechas a mano (compuerta siempre activa, trama de 15 bits).
- Síntesis con yosys como segunda herramienta.
- Tres revisores independientes: puertos y pines, Verilog, README contra código.

35 pruebas: 34 pasan y 1 se omite a propósito (la de anti-saturación no aplica a la sección genérica).

### D.2 De esta guía

Cinco verificadores independientes revisaron el texto (36 afirmaciones): derivaciones rehechas, cada número recalculado con código, fuentes, intento de refutación y pedagogía. Correcciones resultantes:

- El margen de captura es media vuelta, no un cuarto (tres cálculos independientes y ejecución del circuito).
- Sin la espera de arranque el lazo no se bloquea: satura unas 20 muestras y engancha (circuito ejecutado con `WARM_SAMPLES = 1`).
- El integrador discreto tiene 88,6° de margen, no 90°; los 87,6° finales eran correctos.
- El error ante escalón es cero solo en el modelo lineal.
- La rampa es un DDA, y su exactitud no depende de la potencia de dos.
- En la homodina el modo temporal lo fija el oscilador local.

Sin verificar: los tiempos de las hojas de datos, la fórmula de tensión del DAC, el orden de pines de los Pmod contra el manual original y la frecuencia de corte del filtro del Pmod AD1. Las fuentes solo se pudieron leer en extractos de buscador.

---

## Referencias

| referencia | respalda |
|---|---|
| Clements, W. R., Humphreys, P. C., Metcalf, B. J., Kolthammer, W. S. y Walmsley, I. A. (2016). Optimal design for universal multiport interferometers. *Optica* 3(12), 1460–1465. | malla rectangular de Mach–Zehnder |
| Franklin, G. F., Powell, J. D. y Workman, M. L. (1998). *Digital Control of Dynamic Systems*, 3.ª ed. Addison-Wesley. | retenedor de orden cero, transformación bilineal |
| Åström, K. J. y Hägglund, T. (2006). *Advanced PID Control*. ISA. | saturación del integrador, forma incremental |
| Oppenheim, A. V. y Schafer, R. W. (2010). *Discrete-Time Signal Processing*, 3.ª ed. Pearson. | FIR e IIR, fase lineal, precisión finita |
| Turin, G. L. (1960). An introduction to matched filters. *IRE Trans. Inf. Theory* 6(3), 311–329. | filtro adaptado |
| Lvovsky, A. I. y Raymer, M. G. (2009). Continuous-variable optical quantum-state tomography. *Rev. Mod. Phys.* 81, 299–332. | homodina pulsada |
| Xilinx UG479, *7 Series DSP48E1 Slice User Guide*. | multiplicador de 25 × 18 bits |
| TI DAC121S101 (SNAS265); Analog Devices AD7476A; manuales de Digilent del Pmod DA2 y del Pmod AD1. | protocolos serie |

**Orden de lectura sugerido:** A.2 con `da2_spi.v` abierto → A.4 con `phase_sequencer.v` → B.1 a B.4 con `design_filters.py` → B.6 con `iir_biquad.v` → C con `fir_serial.v`.
