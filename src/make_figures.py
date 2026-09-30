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


