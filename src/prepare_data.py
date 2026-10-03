"""Parse RRUFF raw Raman files onto a common wavenumber grid and build the class set.

Output: data/processed/dataset.npz with
    X        (n_spectra, n_points) raw intensity, interpolated onto GRID
    y        mineral name
    groups   RRUFF sample ID (one physical specimen; may have several spectra)
    laser    laser wavelength string

Only 532 nm spectra are used so the classifier is not confounded by excitation
wavelength. Minerals are kept only if they have >= MIN_SAMPLES distinct specimens,
which is what makes a split grouped by specimen possible.
"""
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "unz"
OUT = ROOT / "data" / "processed"

LASER = "532"
GRID = np.arange(180.0, 1280.0, 2.0)  # cm^-1, 550 points
MIN_SAMPLES = 5          # distinct specimens per mineral
MIN_SPECTRA_COVER = 0.99  # spectrum must cover this fraction of the grid


