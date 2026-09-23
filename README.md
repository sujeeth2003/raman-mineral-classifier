# Raman Mineral Classifier

Identify minerals from Raman spectra (RRUFF public database) with baseline correction, PCA-LDA,
PLS-DA and a 1D CNN. Cross-validation is grouped by specimen, and every model is stress-tested
with simulated instrument noise, baseline drift and wavenumber miscalibration.

## Data
RRUFF Raman library, raw (uncorrected) 532 nm spectra, 180-1280 cm^-1 resampled to a 2 cm^-1 grid.
Kept minerals with at least 5 distinct specimens: **70 minerals, 538 spectra, 536 specimens**.

Grouping caveat: RRUFF mostly measures each specimen once per laser, so only 2 specimens have two spectra.
The grouped split is enforced (asserted in code) but it prevents little leakage here. It would matter
more with the oriented/repeat-measurement data or a larger multi-laser set.

## Pipeline
1. Savitzky-Golay smoothing, then **AsLS baseline correction** (lam=1e5, p=0.01), clip at 0, scale to max 1.
2. Models: **PCA(40)+shrinkage LDA**, **PLS-DA** (latent variables chosen by inner CV), **1D CNN**
   (3 conv blocks, 60 epochs, AdamW + one-cycle, label smoothing).
3. 5-fold `StratifiedGroupKFold` (groups = RRUFF specimen ID). Mean +- sd over folds.
4. Ablation: same models with no baseline correction. CNN variant trained with augmented copies
   (noise <= 5 %, drift <= 1.0, shift <= 4 cm^-1; 8 copies per spectrum).
5. Robustness: perturb the *raw* held-out spectra, then rerun the full preprocessing + model.
   Noise sigma and drift amplitude are relative to the peak height; shift is a random-sign
   global wavenumber offset. Models are trained on clean data only (except the aug CNN).

