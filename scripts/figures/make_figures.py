"""
Rebuild every thesis figure from the current analysis CSVs and run folders
(eval-only, CPU, ~1 min). Re-run after new runs are re-scored: each figure
reads whatever data exists, so cells still running are simply left out and
the n shown in the legends says how many seeds went in.

    python scripts/figures/make_figures.py
Outputs analysis/figures/*.pdf (for Overleaf) and *.png (previews).

Figures
  fig_forest_classical_vs_cv          A1 paired forest plot, 4 critics x 4 metrics
  fig_paired_seeds_classical_vs_cv    A2 one line per seed, fidelity and coverage
  fig_bands_val_<critic>              B1 validation metrics during training
  fig_bands_critic_<critic>           B2 critic Wasserstein estimate, one panel per arm
  fig_params_kl16_sampler_vs_explicit E  KL16 vs parameters (Riofrío-style, H1)
  fig_wigner_cv_best_median_worst     D1 density + single-mode Wigner, best/median/worst CV
  fig_wigner_negativity               D2 Wigner negative volume per CV cell
  fig_hw_degradation_cv               F  photonic imperfections (hw_sweep.csv)

D reads analysis/wigner/, written by scripts/wigner_eval.py (needs TF + SF;
re-run it for new runs before rebuilding).
"""

import style as S
import fig_forest
import fig_bands
import fig_params
import fig_wigner
import fig_hw

if __name__ == '__main__':
    S.apply_style()
    fig_forest.forest()
    fig_forest.paired_seeds()
    for crit in S.CRITICS:
        fig_bands.val_figure(crit)
        fig_bands.critic_figure(crit)
    fig_params.main()
    fig_wigner.main()
    fig_hw.main()
