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


def augment(Xraw, grid, scale, rng, n_copies):
    """Random combination of noise, drift and axis shift on raw spectra."""
    out = []
    for _ in range(n_copies):
        X = perturb.add_noise(Xraw, scale, rng.uniform(0, AUG_RANGE["noise"]), rng)
        X = perturb.add_drift(X, scale, rng.uniform(0, AUG_RANGE["drift"]), rng)
        X = perturb.shift_axis(X, grid, rng.uniform(0, AUG_RANGE["shift"]), rng)
        out.append(X)
    return np.concatenate(out)


def score(pred, y):
    return float((pred == y).mean()), float(f1_score(y, pred, average="macro", zero_division=0))


def main(n_folds=5):
    d = np.load(ROOT / "data" / "processed" / "dataset.npz")
    Xraw, names, groups, grid = d["X"].astype(np.float64), d["y"], d["groups"], d["grid"]
    le = LabelEncoder()
    y = le.fit_transform(names)
    n_classes = len(le.classes_)
    print(f"{len(y)} spectra | {n_classes} minerals | {len(set(groups))} specimens", flush=True)

    scale_all = perturb.signal_scale(Xraw)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    clean, robust = [], []
    t0 = time.time()
    for fold, (tr, te) in enumerate(cv.split(Xraw, y, groups)):
        assert not set(groups[tr]) & set(groups[te]), "specimen leaked across split"
        rng = np.random.default_rng(SEED + fold)

        # held-out spectra at every perturbation level (raw domain), built once per fold
        tests = {("clean", 0.0): Xraw[te]}
        for kind, levels in LEVELS.items():
            for lv in levels:
                if lv == 0:
                    continue
                if kind == "noise":
                    Xp = perturb.add_noise(Xraw[te], scale_all[te], lv, rng)
                elif kind == "drift":
                    Xp = perturb.add_drift(Xraw[te], scale_all[te], lv, rng)
                else:
                    Xp = perturb.shift_axis(Xraw[te], grid, lv, rng)
                tests[(kind, lv)] = Xp

        Xaug_raw = augment(Xraw[tr], grid, scale_all[tr], rng, N_AUG)
        yaug = np.tile(y[tr], N_AUG)

        for name, (family, use_bl, use_aug) in VARIANTS.items():
            Xtr = preprocess(Xraw[tr], baseline=use_bl)
            ytr = y[tr]
            if use_aug:
                Xtr = np.concatenate([Xtr, preprocess(Xaug_raw, baseline=use_bl)])
                ytr = np.concatenate([ytr, yaug])
            model = make_model(family, n_classes).fit(Xtr, ytr)
            for (kind, lv), Xt in tests.items():
                acc, f1 = score(model.predict(preprocess(Xt, baseline=use_bl)), y[te])
                if kind == "clean":
                    clean.append(dict(fold=fold, model=name, acc=acc, f1=f1))
                    # the clean point is also the zero level of every robustness curve
                    for k in LEVELS:
                        robust.append(dict(fold=fold, model=name, kind=k, level=0.0, acc=acc, f1=f1))
                else:
                    robust.append(dict(fold=fold, model=name, kind=kind, level=lv, acc=acc, f1=f1))
            print(f"fold {fold} | {name:24s} clean acc {clean[-1]['acc']:.3f} | {time.time() - t0:5.0f}s", flush=True)

    RES.mkdir(exist_ok=True)
    json.dump(dict(clean=clean, robust=robust, classes=le.classes_.tolist(),
                   n_spectra=int(len(y)), n_specimens=int(len(set(groups)))),
              open(RES / "results.json", "w"), indent=1)
    print("saved results.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
