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

