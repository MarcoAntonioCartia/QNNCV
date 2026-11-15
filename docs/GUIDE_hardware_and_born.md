# Guide: the hardware-realizability sweep and the Born-machine objective

> **Corrected 15 Nov 2026** after a fact check against the PDFs: Liu & Wang's kernel description, Benedetti's loss, Rudolph's role, the Clements variant, the evaluation-kernel wording, the propagation-loss arithmetic and the Lvovsky & Raymer attribution. **Second pass, same day** (from `research_notes/born_machines_study_guide.md`): Liu & Wang's bandwidth convention, the kernel sum, the shift-rule restriction, the finite-shot wording and the density-kernel weaknesses (B.1, B.3, B.4).

Written 15 Nov 2026 for reading alongside `scripts/hw_sweep.py` and the `MMDTrainer` in `src/training/qgan_2d.py`. Every reference at the end was checked against Crossref, arXiv or the local `Tesis/sources` index; the ones marked *local* have a PDF there.

---

## Part A. What a photonic chip would do to the trained generator

### A.1 The question

The benchmark trains an ideal circuit: no loss, perfect detectors, any squeezing, and a Kerr gate in every layer. A real photonic processor has none of these luxuries. The sweep takes the trained circuits as they are, adds one imperfection at a time, and measures how far the learned family degrades with the same scores as the benchmark. No retraining happens, so the result says how fragile the *trained solution* is, not how well a lossy device could learn.

Every condition uses the same 200 latent vectors as `scripts/rescore.py`. The ideal row therefore reproduces the benchmark numbers exactly (checked to five decimals), and each condition is a paired comparison against the ideal circuit of the same checkpoint.

### A.2 Detection efficiency (conditions `eta_raw`, `eta_cal`)

A homodyne detector with efficiency η behaves like a perfect detector behind a beam splitter of transmissivity η whose other input is vacuum. In operator form the measured quadrature is

```
x_out = sqrt(eta) * x + sqrt(1 - eta) * x_vac
```

with `x_vac` a vacuum quadrature, variance 1/2 in the ħ = 1 units of our readout. The measured distribution is therefore the ideal one, shrunk by √η and blurred by a Gaussian of variance (1 − η)/2, independently on each output mode. This is exact, not an approximation, so the sweep computes it analytically from the ideal density on a fine grid instead of simulating it. This is the standard treatment of detector inefficiency in homodyne tomography (see the review by Lvovsky and Raymer 2009; checked on the abstract only).

Two variants are reported:

- **raw**: what the detector outputs.
- **calibrated**: outcomes divided by √η, which an experimenter does routinely. It leaves only the blur, with variance (1 − η)/(2η).

Read the calibrated row as the realistic one; the raw row adds a trivial shrinkage on top.

### A.3 Propagation loss (condition `loss_db`)

Light is lost in waveguides and couplers along the whole circuit, not only at the detector. The sweep places a pure-loss channel of the given strength on every mode after every layer. Loss turns the pure state into a mixed one, so this condition needs the density-matrix simulation of Strawberry Fields, which is about 17 times slower than the pure-state run; it therefore uses 80 of the 200 samples, with an ideal reference on the same 80 (`ideal_nloss`). The level is given in dB per layer; 0.1 dB per layer over six layers is 0.6 dB in total, a transmission of 10^(−0.06) = 0.871, so about 13 percent of the light per mode is lost. Weedbrook et al. (2012), section on Gaussian channels, give the loss channel and its action on states.

### A.4 Finite squeezing (condition `rcap`)

Squeezing is the expensive resource of photonic hardware. The sweep clips every trained squeeze magnitude to |r| ≤ r_cap. In decibels the squeezing is 8.686 · r. For scale: the record in a dedicated optical setup is 15 dB (Vahlbruch et al. 2016); integrated and programmable processors work with far less (see Arrazola et al. 2021 for the X8 chip and Madsen et al. 2022 for Borealis). The gate table written per checkpoint reports the squeezing the trained circuits actually use, so you can compare it directly with what a platform offers.

### A.5 The Kerr gate (condition `kerr`)

The Kerr gate exp(iκ n²) is the only non-Gaussian element of the Killoran layer, and CV universality needs at least one non-Gaussian gate (Lloyd and Braunstein 1999, *local*). Without it the circuit is Gaussian and can only output Gaussian position densities, which cannot form a ring. No deployed photonic platform has a deterministic Kerr gate at the single-photon level, and there are physical arguments that large single-photon Kerr phases cannot be obtained this way (Shapiro 2006; Gea-Banacloche 2010). The sweep scales the trained Kerr strengths by 0.5 and by 0. The 0 row is the **Gaussian-only circuit**: what you get if you run the trained weights on hardware that skips the Kerr gates. Proposed routes around the blocker replace the Kerr gate by measurement-induced non-Gaussian operations, for example a weak cubic phase gate prepared with ancillary states and feed-forward (Marek, Filip and Furusawa 2011), or the GKP-based architecture of Bourassa et al. (2021).

### A.6 From trained weights to phase-shifter settings

Each trained interferometer is a mesh of beam splitters. The sweep computes its unitary with the Gaussian compiler of Strawberry Fields (Killoran et al. 2019) and decomposes it into a Clements-type rectangular mesh (Clements et al. 2016; the earlier triangular scheme is Reck et al. 1994), in Strawberry Fields' variant with symmetric Mach–Zehnder interferometers and the local phases moved to the end (Clements used asymmetric ones). Every Mach–Zehnder has an internal and an external phase; these are the voltages a chip controller would apply. The decomposition is checked by rebuilding the unitary (error below 10⁻¹⁴). `gates_<run>.csv` lists, per layer, the Mach–Zehnder phases, the output phases, the squeezing in dB, the displacements and the Kerr strengths, plus an *illustrative* 12-bit DAC code per phase for the FPGA demo on the Pmod DA2. That code assumes a phase shifter with a 2π swing across 0–3.3 V; a real device needs its own calibration curve.

### A.7 How to read the results

- **Fidelity W1 and coverage W1 are distances: lower is better.** Fidelity asks whether each generated density looks like some family member; coverage asks whether every family member is matched by some generated density.
- Compare each condition with the ideal row of the same checkpoint, never across checkpoints.
- A condition that barely moves the scores is *not* a hardware hurdle for this model; one that sends them to the level of an untrained circuit is.

### A.8 Why homodyne detection, and not heterodyne or photon counting

The readout measures the x quadrature of each of the two output modes by homodyne detection, so one shot is one point (x₁, x₂) and the model's distribution for a latent z is the Born distribution P(x₁, x₂) = |⟨x₁, x₂|ψ⟩|² with the ancilla summed out. This follows the architecture the thesis builds on: Killoran et al. (2019) and both continuous-variable Born machines (Čepaitė et al. 2022; Kolarovszki et al. 2024) read out by homodyne detection. Three reasons make it the right fit here, beyond convention:

- **No noise floor.** A homodyne outcome has the width of the state's x quadrature, which squeezing can push below the vacuum width. The ring is thin (width about 0.25 against a radius of about 1.15), so the readout must be able to resolve fine structure.
- **Hardware.** Homodyne detectors are mature and highly efficient, and their inefficiency is an exact Gaussian blur (A.2), so the sweep can model it without approximation.
- **Continuous outputs per shot.** The target is a continuous 2D density, and each shot delivers one sample of it.

**Heterodyne detection** measures x and p of the same mode at once: the mode is split on a 50:50 beam splitter, whose other port admits vacuum, and each half is homodyned. The outcome follows the Husimi Q function, which is the Wigner function smoothed by a vacuum-sized Gaussian (standard deviation 1/√2 ≈ 0.71 per axis in our ħ = 1 units). No state, squeezed or not, gives a narrower heterodyne distribution, so a feature of width 0.25 would be smeared to at least about 0.7. Rescaling the data so the ring is much larger than the blur would require large displacements, i.e. many photons, beyond what cutoff 12 holds. In x, the heterodyne distribution equals that of a calibrated homodyne detector at η = 0.5 (A.2: blur variance (1 − η)/(2η) = 1/2). The sweep's `eta_cal` condition at η = 0.5 therefore shows what heterodyne readout would do to the trained models: median fidelity W1 ×1.40, coverage W1 ×1.36 and MMD² ×1.98 relative to the ideal (20 dense-critic runs). Those models were trained for homodyne; one trained under heterodyne could compensate in part, but never below the blur. Heterodyne's one real advantage is two outputs per mode: a single output mode plus the ancilla would give a 2D sample from a 2-mode circuit (144 Fock amplitudes at cutoff 12 instead of 1728), at the price of that noise floor.

**Photon counting** returns integers. It suits discrete targets, as in Gaussian-boson-sampling Born machines, not a continuous density. **Expectation-value readout** (the ⟨x⟩ of each mode, as in Killoran's fraud-detection model and Chang et al. 2021) makes the circuit a deterministic function of z, with all randomness from the latent and many shots per generated value; our readout uses the circuit's own Born randomness for each latent.

The choice is unrelated to the input state being vacuum. A thermal input would only widen every quadrature distribution (variance (2n̄ + 1)/2 instead of 1/2), a classical noise floor of the same kind heterodyne adds, so it is one more thing to avoid, not a reason for another measurement. Heterodyne's extra noise comes from the vacuum entering the beam splitter, not from thermal light.

---

## Part B. The Born-machine objective

### B.1 What a Born machine is

A quantum circuit defines a probability distribution through the Born rule: the probability of an outcome is the squared amplitude. A *quantum circuit Born machine* trains the circuit so that this distribution matches the data (Benedetti et al. 2019, *local*). Instead of a critic, it uses a fixed statistical distance. Liu and Wang (2018, *local*) introduced the standard choice: the maximum mean discrepancy (MMD) with a *mixture* of Gaussian kernels with fixed absolute bandwidths, differentiable through the circuit. Their kernel divides by 2σ, not 2σ², so σ is a squared length: their σ = 0.25, 10, 1000 are length scales 0.5, 3.2 and 31.6 on a support of 1024 points, and even the widest, which they describe as the scale of the support, covers about 3 % of it in length terms (exponent read from extracted text; check p. 3 of the PDF by eye before quoting). Never compare bandwidth numbers between papers without checking the formula. Benedetti et al. themselves trained with a clipped negative log-likelihood, not an MMD. Čepaitė, Coyle and Kashefi (2022, *local*) built the continuous-variable version: homodyne samples of a CV circuit, trained by MMD. Coyle et al. (2020, *local*) discuss when such models could be hard to simulate classically.

Our generator is already a Born machine in this sense: its output is the Born distribution of the position quadratures. What changes in this arm is only the training objective.

### B.2 The maximum mean discrepancy

For two distributions P and Q and a kernel k, Gretton et al. (2012) define

```
MMD^2(P, Q) = E[k(a, a')] + E[k(b, b')] - 2 E[k(a, b)]       a, a' ~ P,  b, b' ~ Q
```

It is zero exactly when P = Q for a characteristic kernel such as the Gaussian, and it has an unbiased estimator from samples that drops the a = a' terms. Training a generator by minimising it is called generative moment matching (Li, Swersky and Zemel 2015; Dziugaite, Roy and Ghahramani 2015). It needs no critic and no adversarial game, which removes the main source of GAN instability; the price is that a fixed kernel may be blind to differences that a trained critic would find.

### B.3 Two choices of what a "sample" is

Our data is a family of densities, and each latent vector gives one density. That allows two different MMDs, both implemented (`--mmd-kernel`):

- **`density` (the main arm).** Each density is one point. The kernel compares whole densities: k(p, q) = Σ_b exp(−‖p − q‖² / (b² m)), where m is the median *squared* distance between training densities and b ∈ {0.5, 1, 2}, fixed before training. At b = 1 it has the same form as the kernel of the evaluation MMD² in `rescore.py`, but that one takes its median over the pooled generated and validation sample of each run; in terms of a Gaussian length scale h (exp(−d²/2h²)), h = b·√(m/2). The loss compares a batch of 8 generated densities with 32 training members. This learns the *family*: a generator that always outputs the same density cannot match the spread of the data, so collapse is penalised.
- **`sample` (the literal Liu–Wang baseline).** The points are grid positions, and the kernel compares positions (length scales 0.25, 0.5 and 1 in x units). Because the densities are known exactly, the loss is computed exactly (the unbiased U-statistic over the batch), with no sampling. These length scales do not reach the ring's diameter (about 2.3). That is not a departure from Liu and Wang, whose widest length scale is also far below their support (B.1), but a kernel near the ring diameter would help the loss see the hole from across the ring. It compares only the *mixture* of the generated densities with the mixture of the data. A generator that outputs the average ring blob every time can satisfy it, so it cannot see collapse. Keep it as the textbook Born machine, not as the main comparison.

### B.4 Fairness notes for the thesis

- The `density` kernel is close to the kernel of the evaluation MMD² in `rescore.py` (same distance, median bandwidth). Training on almost the test statistic flatters the arm on MMD²; judge it mainly on fidelity and coverage, and say so in the text.
- The kernel and its bandwidths are fixed from the training data before training. That is our own fairness choice; Rudolph et al. (2024, *local*) show that the bandwidth governs trainability (wide kernels train, narrow ones resolve higher-order correlations), which is why it has to be stated with every MMD number.
- Both kernels are a plain sum over the bandwidths, not a mean. Changing the number of bandwidths rescales the loss and so the effective learning rate; runs with different bandwidth counts are not comparable at the same learning rate.
- The `density` kernel is our own construction, an MMD on a space whose points are densities (nearest prior idea: conditional MMD, Ren et al. 2016). State its weaknesses: it needs the full density of each latent, which no hardware Born machine has, so it is a simulation-only objective; ‖p − q‖² compares cell by cell and ignores geometry, so once densities stop overlapping the kernel saturates and the gradient vanishes; 8 generated densities per batch give a noisy (unbiased) estimate; and its bandwidths span only a factor 4 in length (0.35–1.41 √m), against 63 for Liu and Wang.
- `--mmd-shots` is a sensitivity test of the loss to measurement noise with exact gradients, not hardware-like training. With the `sample` kernel the noisy loss is unbiased; with `density` it is biased (the histogram sits inside an exponential). On hardware the gradient also costs shots, two shifted circuits per parameter: 186 × 2 × 8 × 1000 ≈ 3.0 million shots per step at 1000 shots, against 8000 for the loss value.
- Only part of the circuit has a hardware gradient rule. Kerr gates have no known exact shift rule (Čepaitė et al. give only small-shift approximations), and a non-Gaussian gate also breaks the rule for every Gaussian gate *before* it, so Kolarovszki et al. (2024) train only gates placed after all non-Gaussian ones. With a Kerr gate ending every Killoran layer, only the last layer's Gaussian gates would qualify, while the Kerr strengths carry most of the gradient in our runs. Do not claim that the VC shift rules are exact for the MMD loss: they are presented for Gaussian gates, and exactness for a kernel loss has not been verified.
- Everything else is identical to the GAN arm: same generators, data, validation, checkpoint rule, ket-norm penalty and 300 epochs, so the comparison isolates the objective.

### B.5 Status of the code

Committed 15 Nov 2026 (`e35636c`). `--objective mmd` selects the `MMDTrainer`; the default `--objective wgan_gp` leaves the GAN path unchanged (the generator step now calls a hook that performs exactly the same operations). A three-epoch smoke check with toy classical and CV generators gives finite losses and non-zero gradients for both kernels; `tests/golden/test_mmd_objective.py` checks both kernels against independent references, their unbiasedness, the gradient and the shot path. The trainer signature and the recorded configuration gained five entries (`objective`, `mmd_kernel`, `mmd_bandwidths`, `mmd_real_batch`, `mmd_shots`); the end-to-end golden was regenerated in its own audited commit (`5b9168e`), with history and weights unchanged. No MMD training run has been launched yet.

---

## References

| reference | supports |
|---|---|
| Benedetti, M. et al. (2019). A generative modeling approach for benchmarking and training shallow quantum circuits. *npj Quantum Information* 5, 45. doi:10.1038/s41534-019-0157-8 (*local*) | quantum circuit Born machines |
| Bourassa, J. E. et al. (2021). Blueprint for a scalable photonic fault-tolerant quantum computer. *Quantum* 5, 392. doi:10.22331/q-2021-02-04-392 | non-Kerr route to universality |
| Chang, S. Y., Vallecorsa, S., Combarro, E. F. and Carminati, F. (2021). Quantum generative adversarial networks in a continuous-variable architecture to simulate high energy physics detectors. NeurIPS ML4PS 2020 workshop, arXiv:2101.11132 (*local*) | expectation-value readout of a CV generator (A.8) |
| Čepaitė, I., Coyle, B. and Kashefi, E. (2022). A continuous variable Born machine. *Quantum Machine Intelligence* 4, 6. doi:10.1007/s42484-022-00063-3 (*local*) | CV Born machine trained by MMD |
| Clements, W. R. et al. (2016). Optimal design for universal multiport interferometers. *Optica* 3, 1460. doi:10.1364/OPTICA.3.001460 | rectangular Mach–Zehnder mesh |
| Coyle, B. et al. (2020). The Born supremacy: quantum advantage and training of an Ising Born machine. *npj Quantum Information* 6, 60. doi:10.1038/s41534-020-00288-9 (*local*) | Born machines, hardness, training |
| Dziugaite, G. K., Roy, D. M. and Ghahramani, Z. (2015). Training generative neural networks via maximum mean discrepancy optimization. *UAI 2015*, arXiv:1505.03906 | MMD-trained generators |
| Gea-Banacloche, J. (2010). Impossibility of large phase shifts via the giant Kerr effect with single-photon wave packets. *Phys. Rev. A* 81, 043823. doi:10.1103/PhysRevA.81.043823 | Kerr blocker |
| Gretton, A. et al. (2012). A kernel two-sample test. *J. Mach. Learn. Res.* 13, 723–773; extended report: Gretton, Borgwardt, Rasch, Schölkopf, Smola (2008), A kernel method for the two-sample problem, arXiv:0805.2368 | MMD definition and estimator |
| Killoran, N. et al. (2019). Strawberry Fields: a software platform for photonic quantum computing. *Quantum* 3, 129. doi:10.22331/q-2019-03-11-129 | simulator, Gaussian compiler, decompositions |
| Killoran, N., Bromley, T. R., Arrazola, J. M., Schuld, M., Quesada, N. and Lloyd, S. (2019). Continuous-variable quantum neural networks. *Phys. Rev. Research* 1, 033063. doi:10.1103/PhysRevResearch.1.033063 (*local*) | the generator architecture and its homodyne readout (A.8) |
| Kolarovszki, Z., Nagy, D. T. and Zimborás, Z. (2024). Photonic CV Born machines trained from homodyne samples. *2024 IEEE QCE*, pp. 750–756. doi:10.1109/QCE60285.2024.00094, arXiv:2410.11785 (*local*; title paraphrased, take the exact one from the PDF) | homodyne-sampled CV Born machine, finite shots, shift-rule restriction |
| Li, Y., Swersky, K. and Zemel, R. (2015). Generative moment matching networks. *ICML 2015*, arXiv:1502.02761 | MMD-trained generators |
| Liu, J.-G. and Wang, L. (2018). Differentiable learning of quantum circuit Born machines. *Phys. Rev. A* 98, 062324. doi:10.1103/PhysRevA.98.062324 (*local*) | MMD loss for Born machines |
| Lloyd, S. and Braunstein, S. L. (1999). Quantum computation over continuous variables. *Phys. Rev. Lett.* 82, 1784. doi:10.1103/PhysRevLett.82.1784 (*local*) | non-Gaussian gate needed for universality |
| Lvovsky, A. I. and Raymer, M. G. (2009). Continuous-variable optical quantum-state tomography. *Rev. Mod. Phys.* 81, 299. doi:10.1103/RevModPhys.81.299 | homodyne efficiency model |
| Madsen, L. S. et al. (2022). Quantum computational advantage with a programmable photonic processor. *Nature* 606, 75. doi:10.1038/s41586-022-04725-x | Borealis, realistic squeezing and loss |
| Marek, P., Filip, R. and Furusawa, A. (2011). Deterministic implementation of weak quantum cubic nonlinearity. *Phys. Rev. A* 84, 053802. doi:10.1103/PhysRevA.84.053802 | measurement-induced non-Gaussian gate |
| Reck, M. et al. (1994) and Arrazola, J. M. et al. (2021) | already in the thesis bibliography (`reck:1994`, `arrazola:2021`) |
| Rudolph, M. S. et al. (2024). Trainability barriers and opportunities in quantum generative modeling. (*local*, `2305.02881`) | MMD bandwidth vs trainability |
| Shapiro, J. H. (2006). Single-photon Kerr nonlinearities do not help quantum computation. *Phys. Rev. A* 73, 062305. doi:10.1103/PhysRevA.73.062305 | Kerr blocker |
| Vahlbruch, H. et al. (2016). Detection of 15 dB squeezed states of light and their application for the absolute calibration of photoelectric quantum efficiency. *Phys. Rev. Lett.* 117, 110801. doi:10.1103/PhysRevLett.117.110801 | squeezing record |
| Weedbrook, C. et al. (2012). Gaussian quantum information. *Rev. Mod. Phys.* 84, 621. doi:10.1103/RevModPhys.84.621 | loss channel, Gaussian states and gates |

**Suggested reading order:** Lvovsky and Raymer, sections on homodyne detection and losses → Weedbrook et al., Gaussian channels → Clements et al. (short) → Liu and Wang → Čepaitė et al. → Gretton et al., sections 1–3.
