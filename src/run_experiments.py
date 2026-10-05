"""Cross-validated comparison and robustness test.

Protocol
--------
* 5-fold StratifiedGroupKFold: every spectrum of a given RRUFF specimen lands in the same fold,
  and each mineral is spread across folds.
* Models are trained on clean training spectra only (plus augmented copies for one CNN variant).
* Held-out spectra are evaluated clean and after perturbing the RAW signal with noise,
  baseline drift or wavenumber shift, then re-running the whole preprocessing chain.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

import perturb
from models import CNN1D, PCALDA, PLSDA
from preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED = 0
N_AUG = 8  # augmented copies per training spectrum for the "CNN + aug" variant

LEVELS = {
    "noise": [0.0, 0.02, 0.05, 0.1, 0.2, 0.4],     # sigma / peak height
    "drift": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],      # baseline peak-to-peak / peak height
    "shift": [0.0, 1.0, 2.0, 4.0, 6.0, 10.0],      # cm^-1 (random sign per spectrum)
}
# the augmented CNN sees noise <= 0.05, drift <= 1.0, |shift| <= 4 cm^-1 in training;
# the larger levels above are deliberately outside that range.
AUG_RANGE = {"noise": 0.05, "drift": 1.0, "shift": 4.0}

# name -> (model family, baseline correction on?, augmented training?)
VARIANTS = {
    "PCA-LDA (baseline)":     ("pca", True, False),
    "PCA-LDA (no baseline)":  ("pca", False, False),
    "PLS-DA (baseline)":      ("pls", True, False),
    "PLS-DA (no baseline)":   ("pls", False, False),
    "CNN (baseline)":         ("cnn", True, False),
    "CNN (no baseline)":      ("cnn", False, False),
    "CNN (baseline + aug)":   ("cnn", True, True),
}


def make_model(family, n_classes):
    if family == "pca":
        return PCALDA()
    if family == "pls":
        return PLSDA(n_classes)
    return CNN1D(n_classes, seed=SEED)


