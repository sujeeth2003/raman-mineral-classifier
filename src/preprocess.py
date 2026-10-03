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


