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


def read_spectrum(path):
    xs, ys = [], []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("##") or not line.strip():
            continue
        parts = re.split(r"[,\s]+", line.strip())
        try:
            xv, yv = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        xs.append(xv)
        ys.append(yv)
    x, y = np.asarray(xs), np.asarray(ys)
    order = np.argsort(x)
    return x[order], y[order]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(RAW.glob("*Raman_Data_RAW*")):
        name, rid, _, laser = p.name.split("__")[:4]
        if laser != LASER:
            continue
        x, y = read_spectrum(p)
        if len(x) < 50:
            continue
        cover = ((GRID >= x.min()) & (GRID <= x.max())).mean()
        if cover < MIN_SPECTRA_COVER:
            continue
        yi = np.interp(GRID, x, y)
        if not np.isfinite(yi).all() or np.ptp(yi) == 0:
            continue
        rows.append((name, rid, yi))
    print(f"spectra parsed at {LASER} nm with full coverage: {len(rows)}")

