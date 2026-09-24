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

## Results (held-out, 5-fold)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| PCA-LDA (baseline) | 0.885 +- 0.045 | 0.862 |
| PLS-DA (baseline) | 0.844 +- 0.015 | 0.799 |
| CNN (baseline) | 0.907 +- 0.025 | 0.894 |
| CNN (baseline + aug) | **0.915 +- 0.007** | 0.901 |
| PCA-LDA (no baseline) | 0.836 +- 0.032 | 0.825 |
| PLS-DA (no baseline) | 0.816 +- 0.014 | 0.773 |
| CNN (no baseline) | 0.829 +- 0.030 | 0.800 |

Baseline correction adds 3-8 accuracy points to every model. Chance is about 1.4 %.

![robustness](results/robustness.png)

| Accuracy at... | PCA-LDA | PLS-DA | CNN | CNN + aug |
|---|---|---|---|---|
| drift 1.0 x peak height | 0.874 | 0.846 | 0.784 | 0.894 |
| drift 2.0 | 0.864 | 0.823 | 0.625 | 0.872 |
| shift 6 cm^-1 | 0.697 | 0.589 | 0.883 | 0.876 |
| shift 10 cm^-1 | 0.422 | 0.266 | 0.861 | 0.844 |
| noise 0.10 | 0.875 | 0.831 | 0.041 | 0.629 |
| noise 0.20 | 0.833 | 0.771 | 0.026 | 0.065 |

Full tables: `results/robustness_tables.md`, `results/clean_summary.csv`.

## What the robustness test shows
- **Baseline drift:** without baseline correction accuracy collapses (PCA-LDA 0.32 at drift 2.0 vs 0.86 with it). Augmentation makes the CNN as drift-proof as the linear models.
- **Wavenumber shift:** convolutional features are much more tolerant than PCA/PLS, which compare fixed
  channels (at 10 cm^-1: CNN 0.86 vs PLS-DA 0.27).
- **Noise: the CNN is the weak point.** It falls apart at noise 0.10 and, even with augmentation, at 0.20
  (augmentation only covered up to 0.05, so 0.10 is partial extrapolation). PCA-LDA and PLS-DA degrade gently.
  My untested guess is that clipping at zero and min-max scaling bias noisy spectra in a way the network never saw; I did not isolate the cause.
- So no model wins everywhere. The CNN is best on clean data, drift and shift; the linear models are best under heavy noise.

## Limitations
- 70 classes with 5-12 specimens each; accuracy has about +-3 points of fold-to-fold spread.
- Perturbations are synthetic. Real spectrometer drift was not measured.
- Single laser (532 nm). Hyperparameters were fixed, not tuned per model; the CNN was not tuned.
- Test-time folds contain classes seen in training only (closed-set), no unknown-mineral rejection.

