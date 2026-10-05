"""Baseline correction and normalisation for Raman spectra."""
import numpy as np
from scipy.linalg import solveh_banded
from scipy.signal import savgol_filter


def _second_diff_penalty_banded(n):
    """Upper banded (3 diagonals) form of D'D for the second-difference operator D."""
    d = np.zeros(n)
    d1 = np.zeros(n)
    d2 = np.zeros(n)
    d[:] = 6.0
    d[0] = d[-1] = 1.0
    d[1] = d[-2] = 5.0
    d1[1:] = -4.0
    d1[1] = d1[-1] = -2.0
    d2[2:] = 1.0
    return d, d1, d2


def asls_baseline(y, lam=1e5, p=0.01, n_iter=10):
    """Asymmetric least squares baseline (Eilers & Boelens, 2005).

    Points above the current fit get weight p, points below get 1 - p, so the
    fit hugs the bottom of the spectrum and ignores Raman peaks.
    """
    n = len(y)
    d, d1, d2 = _second_diff_penalty_banded(n)
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        ab = np.zeros((3, n))
        ab[0, 2:] = lam * d2[2:]
        ab[1, 1:] = lam * d1[1:]
        ab[2, :] = lam * d + w
        z = solveh_banded(ab, w * y)
        w = np.where(y > z, p, 1.0 - p)
    return z


def normalise(s):
    """Shift to zero minimum, scale to unit maximum."""
    s = s - s.min()
    m = s.max()
    return s / m if m > 0 else s


def preprocess(X, baseline=True, smooth=True, lam=1e5, p=0.01):
    """Return an (n, m) float32 array of baseline-corrected, smoothed, max-normalised spectra."""
    out = np.empty(X.shape, dtype=np.float32)
    for i, y in enumerate(X.astype(np.float64)):
        if smooth:
            y = savgol_filter(y, window_length=9, polyorder=3)
        if baseline:
            y = y - asls_baseline(y, lam=lam, p=p)
            y = np.clip(y, 0, None)
        out[i] = normalise(y)
    return out
