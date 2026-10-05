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

