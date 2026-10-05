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

