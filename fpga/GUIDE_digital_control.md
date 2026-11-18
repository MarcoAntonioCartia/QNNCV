# Guide: the digital control in the `fpga/` folder

Written 18 Nov 2026 for reading alongside the code in `fpga/rtl/` and `fpga/README.md`. The README says what is there and how to use it; this guide explains why it works. The circuits are tested in simulation with shortened parameters. The stability margins, the dead band and the noise figures are calculations, not tests. Nothing has been synthesised with Vivado or measured on the board.

---

## Part A. Open loop: from angles to voltages

### A.1 Two designs

| | design 1: phase table | design 2: lock and readout |
|---|---|---|
| top level | `arty_phase_demo.v` | `arty_lock_demo.v` |
| loop | open | closed |
| filters | none | PI (IIR) inside the loop, FIR outside |
| physical | DAC | DAC, ADC, RC network |
| emulated | — | interferometer fringe and drift |

The two are not connected. The loop holds a working point at mid-fringe; it does not set the trained angles of design 1.

### A.2 The serial link to the DAC

Each write is a 16-bit frame: 2 unused bits, 2 mode bits (00) and 12 data bits, most significant first. The DAC reads the data line on every **falling** edge of the serial clock; the FPGA changes it on the **rising** edge, half a period earlier. At 12.5 MHz that leaves 40 ns of setup and 40 ns of hold. The output changes on the 16th falling edge; if the select line rises earlier, the frame is discarded.

The ADC reader works the other way round: when the select line falls the ADC samples and puts out the first bit; every falling clock edge brings out the next one (four zeros and twelve bits). The FPGA reads each bit 70 ns after the edge that brings it out and 10 ns before the one that replaces it.

### A.3 From angles to codes

A rectangular mesh of M(M−1)/2 Mach–Zehnder interferometers plus M output phases realises any M-mode interferometer (Clements et al. 2016). Each Mach–Zehnder has an internal phase (what fraction of the light crosses) and an external one (relative phase between the two modes). `scripts/hw_sweep.py` computes them; the demo puts the internal phase on channel A and the external one on channel B, one Mach–Zehnder per step.

```
code    = round((phi mod 2 pi) / 2 pi * 4095)
voltage = 3.3 V * code / 4096          (formula still to be checked against the data sheet)
```

The map is illustrative: a thermo-optic phase shifter responds to dissipated power, not to voltage, and needs calibration.

### A.4 The sequencer and the ramp

A six-state machine: fetch word, decode, ramp (32 ticks), settle (8), gate (8), hold (8). One tick is 100 µs and a step always lasts 56 ticks.

The linear ramp uses no divider. With the previous code `p` and the new one `q`:

```
accumulator <- 32 p ;  on every tick: accumulator <- accumulator + (q - p)
code after tick i = p + floor(i (q - p) / 32) ;  after tick 32 = q exactly
```

The exact arrival comes from accumulating the difference without rounding, not from 32 being a power of two; the power of two only makes the division free. This is a fixed-point digital differential analyser (DDA).

The gate rises at least 0.8 ms after the last DAC write, and no frame is sent while it is high.

---

## Part B. Closed loop: the PI controller

### B.1 Plant

An RC network of 10 kΩ and 100 nF: `G(s) = 1/(tau s + 1)`, tau = 1 ms. The controller samples every T = 10 µs and the DAC holds its value between writes. Ignoring the computation delay (3.4 µs), quantisation and the input filter of the Pmod AD1, the discrete equivalent is exact at the sampling instants:

```
G(z) = (1 - a)/(z - a),   a = e^(-T/tau) = 0.99005
```

### B.2 Fringe

```
y = 2048 + 2047 sin(phi),   phi = 2 pi (v - 2048)/4096 + drift
K = dy/dv at mid-fringe = 2047 * 2 pi / 4096 = 3.14 detector codes per ADC code
```

- The slope is K cos(phi). It is positive only within a quarter turn of the working point.
- The controller moves according to the sign of the error, which is correct up to half a turn. The loop pulls in from there; at exactly half a turn there is an unstable equilibrium, and beyond it the loop locks on the next fringe.
- It fails only when the nearest stable point lies outside the range of the DAC.

### B.3 Discrete PI

```
C(s) = Kp (1 + 1/(Ti s))
bilinear:  s -> (2/T) (z - 1)/(z + 1)
C(z) = (b0 z + b1)/(z - 1),  b0 = Kp (1 + T/(2 Ti)),  b1 = -Kp (1 - T/(2 Ti))
u[n] = u[n-1] + b0 e[n] + b1 e[n-1]
```

### B.4 Design by cancellation

With Ti = tau the zero of the PI, (2 Ti − T)/(2 Ti + T) = 199/201 = 0.990050, falls on the plant pole. The open loop reduces to an integrator:

```
L(z) ~ g/(z - 1),   g = Kp K (1 + T/(2 Ti)) (1 - a)
closed loop: pole at 1 - g, response 1 - (1 - g)^n
```

The target is a time constant of 0.2 ms (20 samples), five times faster than the plant.

| quantity | value |
|---|---|
| Kp = tau/(tau_cl K) | 1.5923 |
| b0, b1 | 1.6003 and −1.5844; in Q3.15: 52438 and −51916 |
| g | 0.0500 |
| closed-loop pole | 0.9500 |
| time constant | 19.5 samples |

In the simulation of the full circuit, 63 % is reached at sample 20.

### B.5 Stability and accuracy

- **Phase margin:** crossover at g/T = 5000 rad/s (about 800 Hz). A continuous integrator would have 90°; the discrete one loses omega T/2 = 1.4° and the 3.4 µs delay another 1.0°. That leaves 87.6° (calculation).
- **Step:** zero error in the linear model. In the circuit a ripple of ±1 ADC code remains, because one ADC code is 3.14 detector codes.
- **Ramp:** error = slope/g, with the slope in detector codes per sample. Simulation: 6.28/0.05 = 126 predicted, 129 maximum measured. Board: 4 predicted, not simulated.
- **Imperfect cancellation:** if the real tau is larger than Ti there is overshoot; if smaller, a slow tail. Between 0.3 and 5 ms the loop stays stable with a margin of 60° or more (calculation); at 1.2 ms the overshoot is 1.7 %.

### B.6 Fixed point

- PI coefficients in Q3.15 (18 bits), signals in Q13.12 (25 bits): one product fits a DSP48E1 block.
- **Fraction in the state:** the integral gain per sample is 522/32768 = 0.0159. With an integer state and round-to-nearest, errors of up to 31 codes would not move the output. With 12 fractional bits, an error of one code already does.
- **Anti-windup:** the output is clamped to ±2047 and the clamped value is what is fed back. In incremental form the integrator cannot go past the rail, and it leaves the rail on the first sample where b0 e[n] + b1 e[n−1] is negative.
- **Start-up:** with the capacitor discharged the loop starts from the unstable equilibrium; without a wait it saturates for about 20 samples and locks in about 2 ms. The design waits 1024 samples with the DAC at mid-scale to avoid that transient.

### B.7 PID, IIR and FIR

A discrete PID is an IIR: the PI is first order and the PID second order. `iir_biquad.v` is the general second-order section; with five other coefficients it would be a PID, a notch or a low-pass.

| | PID | general IIR | FIR |
|---|---|---|---|
| can integrate | yes | yes | no |
| stability of the filter alone | marginal (pole at z = 1) | depends on the poles | always stable |
| exact linear phase | no | no | with symmetric or antisymmetric coefficients |
| delay | small | depends on the poles | (L−1)/2 samples if linear phase |
| usual place | inside the loop | inside the loop | outside: measurement and readout |

---

## Part C. The readout FIR

### C.1 What it computes

One number per gate: the weighted sum of the last 64 samples, `value = sum_k w[k] x[k]`. It is the digital version of integrating a photocurrent over the duration of a pulse, which is how pulsed homodyne detection obtains one quadrature value per pulse (Lvovsky and Raymer 2009). The scale is not that of an experiment: here the window lasts 0.64 ms.

### C.2 The weights

Model: `x[k] = A m[k] + c + noise`, with m[k] the known pulse shape (a Gaussian of width 9 samples centred on sample 36) and c an unknown constant level. The least-squares estimator of A is

```
w[k] = (m[k] - mean(m)) / sum_j (m[j] - mean(m))^2
sum_k w[k] = 0       ->  a constant level gives exactly 0
sum_k w[k] m[k] = 1  ->  a pulse of amplitude A gives A (up to rounding: 2 codes at most)
```

In the file the coefficients are reversed, h[k] = w[63−k], because an FIR multiplies h[0] by the newest sample.

### C.3 Cost and hypotheses

With white noise of standard deviation s per sample, the value has standard deviation s sqrt(sum w²) = 0.353 s; the matched filter without mean removal would give 0.250 s. The factor 1.41 is specific to this pulse and this window. Hypotheses: known shape and timing, white noise, a level that is constant within the window. A slope is not rejected, and a pulse shifted by 4 samples loses 10 %.

### C.4 A single multiplier

64 products with one multiplier used 64 times and an accumulator: 67 clock periods, 0.67 µs, out of the 1000 between samples.

---

## Part D. Verification

### D.1 Of the circuit

- Models of both Pmods that reject any frame outside the protocol.
- Bit-exact Python references for the IIR and the FIR.
- A model of the RC network with its exact exponential.
- Two mutation tests done by hand (gate always on, 15-bit frame).
- Synthesis with yosys as a second tool.
- Three independent reviewers: ports and pins, Verilog, README against code.

35 tests: 34 pass and 1 is skipped on purpose (the anti-windup test does not apply to the general section).

### D.2 Of this guide

Five independent verifiers reviewed the text (36 claims): derivations redone, every number recomputed with code, sources, attempted refutation and pedagogy. Resulting corrections:

- The capture range is half a turn, not a quarter (three independent calculations and a run of the circuit).
- Without the start-up wait the loop does not get stuck: it saturates for about 20 samples and locks (circuit run with `WARM_SAMPLES = 1`).
- The discrete integrator has 88.6° of margin, not 90°; the final 87.6° was correct.
- The step error is zero only in the linear model.
- The ramp is a DDA, and its exactness does not depend on the power of two.
- In homodyne detection the temporal mode is set by the local oscillator.

Not verified: the data-sheet timings, the DAC voltage formula, the Pmod pin order against the original manual and the cutoff frequency of the Pmod AD1 input filter. Sources could only be read as search-result excerpts.

---

## References

| reference | supports |
|---|---|
| Clements, W. R., Humphreys, P. C., Metcalf, B. J., Kolthammer, W. S. and Walmsley, I. A. (2016). Optimal design for universal multiport interferometers. *Optica* 3(12), 1460–1465. | rectangular Mach–Zehnder mesh |
| Franklin, G. F., Powell, J. D. and Workman, M. L. (1998). *Digital Control of Dynamic Systems*, 3rd ed. Addison-Wesley. | zero-order hold, bilinear transform |
| Åström, K. J. and Hägglund, T. (2006). *Advanced PID Control*. ISA. | integrator windup, incremental form |
| Oppenheim, A. V. and Schafer, R. W. (2010). *Discrete-Time Signal Processing*, 3rd ed. Pearson. | FIR and IIR, linear phase, finite precision |
| Turin, G. L. (1960). An introduction to matched filters. *IRE Trans. Inf. Theory* 6(3), 311–329. | matched filter |
| Lvovsky, A. I. and Raymer, M. G. (2009). Continuous-variable optical quantum-state tomography. *Rev. Mod. Phys.* 81, 299–332. | pulsed homodyne detection |
| Xilinx UG479, *7 Series DSP48E1 Slice User Guide*. | 25 × 18 bit multiplier |
| TI DAC121S101 (SNAS265); Analog Devices AD7476A; Digilent manuals for the Pmod DA2 and the Pmod AD1. | serial protocols |

**Suggested reading order:** A.2 with `da2_spi.v` open → A.4 with `phase_sequencer.v` → B.1 to B.4 with `design_filters.py` → B.6 with `iir_biquad.v` → C with `fir_serial.v`.
