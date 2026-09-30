"""Figures and summary tables from results/results.json and the processed dataset."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from preprocess import asls_baseline, preprocess

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

COLORS = {
    "PCA-LDA": "#4c78a8", "PLS-DA": "#f58518", "CNN": "#54a24b",
}
XLABEL = {"noise": "Gaussian noise σ (fraction of peak height)",
          "drift": "Baseline drift p-p (fraction of peak height)",
          "shift": "Wavenumber shift (cm⁻¹)"}


def style(name):
    fam = name.split(" (")[0]
    ls = "--" if "no baseline" in name else "-"
    marker = "s" if "aug" in name else "o"
    return COLORS[fam], ls, marker


def fig_baseline():
    d = np.load(ROOT / "data" / "processed" / "dataset.npz")
    X, grid, y = d["X"].astype(np.float64), d["grid"], d["y"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    rng = np.random.default_rng(3)
    # spectra with a strong baseline relative to peak height make the point best
    ratio = np.array([asls_baseline(s).mean() / max(np.ptp(s), 1) for s in X])
    for ax, i in zip(axes, np.argsort(ratio)[-3:]):
        z = asls_baseline(X[i])
        ax.plot(grid, X[i], color="#999", lw=1, label="raw")
        ax.plot(grid, z, color="#e45756", lw=1.5, label="AsLS baseline")
        ax.plot(grid, X[i] - z, color="#4c78a8", lw=1, label="corrected")
        ax.set_title(y[i], fontsize=10)
        ax.set_xlabel("Raman shift (cm⁻¹)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("intensity (a.u.)")
    fig.tight_layout()
    fig.savefig(RES / "baseline_correction.png", dpi=160)
    plt.close(fig)

