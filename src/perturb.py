"""Instrument-style perturbations applied to RAW spectra (before preprocessing).

All amplitudes are relative to `scale`, the peak height of the clean baseline-corrected
spectrum, so "noise 0.05" means Gaussian noise with sigma = 5 % of the tallest peak.
"""
import numpy as np
from numpy.polynomial import legendre

from preprocess import asls_baseline


def signal_scale(X):
    """Peak height above baseline for each raw spectrum."""
    return np.array([np.clip(y - asls_baseline(y.astype(np.float64)), 0, None).max() for y in X])


def add_noise(X, scale, level, rng):
    return X + rng.normal(size=X.shape) * (level * scale)[:, None]


def add_drift(X, scale, amp, rng, order=4):
    """Add a smooth random baseline (fluorescence-like drift) with peak-to-peak `amp` * scale."""
    t = np.linspace(-1, 1, X.shape[1])
    out = X.astype(np.float64).copy()
    for i in range(len(X)):
        b = legendre.legval(t, rng.normal(size=order + 1) / (1 + np.arange(order + 1)))
        b = (b - b.min()) / max(np.ptp(b), 1e-12)
        out[i] += amp * scale[i] * b
    return out


