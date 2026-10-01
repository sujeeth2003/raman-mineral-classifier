"""Instrument-style perturbations applied to RAW spectra (before preprocessing).

All amplitudes are relative to `scale`, the peak height of the clean baseline-corrected
spectrum, so "noise 0.05" means Gaussian noise with sigma = 5 % of the tallest peak.
"""
import numpy as np
from numpy.polynomial import legendre

from preprocess import asls_baseline


