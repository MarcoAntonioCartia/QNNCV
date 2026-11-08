# References for the CV / DV / classical benchmark

Compiled 8 Nov 2026 by web search; every entry was confirmed against its
arXiv/publisher page unless marked UNVERIFIED. One line each on why it matters
here. Thesis citations should quote the journal version where one exists.

## Quantum GANs — foundational
- Lloyd & Weedbrook (2018), *Quantum generative adversarial learning*, PRL 121, 040502; arXiv:1804.09139 — the original QGAN proposal and the quantum-generator / classical-critic framing this work inherits.
- Dallaire-Demers & Killoran (2018), *Quantum generative adversarial networks*, PRA 98, 012324; arXiv:1804.08641 — first parametrised-circuit QGAN with circuit gradients.
- Zoufal, Lucchi & Woerner (2019), *Quantum GANs for learning and loading random distributions*, npj QI 5, 103; arXiv:1904.00043 — qubit generator + classical discriminator on a gridded 1-D/2-D distribution: the closest template for the DV baseline.
- Killoran, Bromley, Arrazola, Schuld, Quesada & Lloyd (2019), *Continuous-variable quantum neural networks*, PRR 1, 033063; arXiv:1806.06871 — the CV layer (interferometer–squeeze–interferometer–displacement–Kerr) the CV generator and the CV critic are built from.
- Huang et al. (2021), *Experimental quantum GANs for image generation*, PRApplied 16, 024051; arXiv:2010.06201 — superconducting "patch" QGAN benchmarked against MLP/CNN GANs: a precedent for matched classical comparison.
- Software: Killoran et al. (2019), *Strawberry Fields*, Quantum 3, 129; Bergholm et al., *PennyLane*, arXiv:1811.04968 (cite the version used: 0.45.1 here).

## Discrete-variable baselines
- Sedrakyan & Salavrakos (2024), *Photonic quantum GANs for classical data*, Optica Quantum 2(6), 458; arXiv:2405.06023 — Fock-encoded linear-optics generator on Quandela's Ascella; the inspiration for the data re-uploading (interleaved) encoding and the natural discrete-photonic counterpart.
- PennyLane demos: *Quantum GANs with Cirq + TensorFlow* (3-qubit toy; its TF interface is deprecated in PennyLane ≥ 0.42) and *Quantum GANs* (patch method, PyTorch; author UNVERIFIED) — the standard qubit-QGAN reference implementations.

## CV generative models and Born machines
- Čepaitė, Coyle & Kashefi (2022), *A continuous variable Born machine*, QMI 4, 6; arXiv:2011.00904 — non-adversarial CV competitor trained with MMD-type losses on homodyne samples.
- Kolarovszki, Nagy & Zimborás (2024), *On the learning abilities of photonic CV Born machines*, arXiv:2410.11785 (preprint) — CV Born machines at 2–4 modes, stochastic MMD gradients; sample-budget and trainability arguments.
- Rudolph et al. (2024), *Trainability barriers and opportunities in quantum generative modeling*, npj QI 10, 116; arXiv:2305.02881 — MMD trainability depends on kernel bandwidth; explicit losses hit their own barren plateaus; frames any MMD-vs-adversarial comparison.
- Benedetti et al. (2019), *A generative modeling approach for benchmarking and training shallow quantum circuits*, npj QI 5, 45; arXiv:1801.07686 — QCBM and the Bars-and-Stripes score: gridded toy datasets as quantum-generator benchmarks.

## Fair comparison of generative models
- Sajjadi et al. (2018), *Assessing generative models via precision and recall*, NeurIPS; arXiv:1806.00035 — quality vs coverage split (our fidelity / coverage pair).
- Kynkäänniemi et al. (2019), *Improved precision and recall metric*, NeurIPS; arXiv:1904.06991 — k-NN manifold estimates, computable on 2-D samples.
- Naeem et al. (2020), *Reliable fidelity and diversity metrics*, ICML; arXiv:2002.09797 — density & coverage, robust to outliers.
- Gretton et al. (2012), *A kernel two-sample test*, JMLR 13, 723 — MMD, its unbiased estimator and the permutation test used by `rescore.py`.
- Székely & Rizzo (2013), *Energy statistics*, JSPI 143, 1249 — the energy distance used as the full-2D metric.
- Arjovsky, Chintala & Bottou (2017), *Wasserstein GAN*, ICML; arXiv:1701.07875 — and Gulrajani et al. (2017), *Improved training of Wasserstein GANs*, NeurIPS; arXiv:1704.00028 — the WGAN-GP objective every arm shares.
- Schuld & Killoran (2022), *Is quantum advantage the right goal for QML?*, PRX Quantum 3, 030101 — argues for characterisation over "beat classical" claims; the register this benchmark should be written in.
- Bowles, Ahmed & Schuld (2024), *Better than classical? The subtle art of benchmarking QML models*, arXiv:2403.07059 — matched, simple classical baselines; classical often wins; the methodology the parameter-matched comparison follows.

## Coordinate networks (the classical generator family)
- Sitzmann et al. (2020), *Implicit neural representations with periodic activation functions* (SIREN), NeurIPS; arXiv:2006.09661 — sine-activated coordinate MLPs fit high-frequency 2-D fields with few parameters: the natural upgrade of `ClassicalCoordGenerator` if tanh units prove too smooth.
- Tancik et al. (2020), *Fourier features let networks learn high frequency functions in low dimensional domains*, NeurIPS; arXiv:2006.10739 — random Fourier features cure the spectral bias of coordinate MLPs; the other standard upgrade.
