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


def fig_pca():
    d = np.load(ROOT / "data" / "processed" / "dataset.npz")
    X, y = d["X"].astype(np.float64), d["y"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    top = pd.Series(y).value_counts().index[:8]
    cmap = plt.get_cmap("tab10")
    for ax, bl, title in zip(axes, [False, True], ["no baseline correction", "AsLS baseline correction"]):
        P = preprocess(X, baseline=bl)
        pca = PCA(2, random_state=0).fit(P)
        S = pca.transform(P)
        ax.scatter(S[:, 0], S[:, 1], s=6, color="#ddd")
        for k, m in enumerate(top):
            sel = y == m
            ax.scatter(S[sel, 0], S[sel, 1], s=18, color=cmap(k), label=m)
        ev = pca.explained_variance_ratio_
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(RES / "pca_scores.png", dpi=160)
    plt.close(fig)


def main():
    r = json.load(open(RES / "results.json"))
    clean = pd.DataFrame(r["clean"])
    robust = pd.DataFrame(r["robust"])

    summ = clean.groupby("model", sort=False).agg(
        acc_mean=("acc", "mean"), acc_sd=("acc", "std"), f1_mean=("f1", "mean"), f1_sd=("f1", "std")).round(3)
    summ.to_csv(RES / "clean_summary.csv")
    print(summ.to_string(), "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    tables = {}
    for ax, kind in zip(axes, ["noise", "drift", "shift"]):
        sub = robust[robust.kind == kind]
        g = sub.groupby(["model", "level"], sort=False).acc.agg(["mean", "std"]).reset_index()
        tables[kind] = g.pivot(index="model", columns="level", values="mean").round(3)
        for name, gg in g.groupby("model", sort=False):
            c, ls, mk = style(name)
            ax.errorbar(gg.level, gg["mean"], yerr=gg["std"], color=c, ls=ls, marker=mk, ms=4,
                        lw=2.2 if "aug" in name else 1.4, capsize=2, label=name)
        ax.set_xlabel(XLABEL[kind])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("held-out accuracy (5-fold, mean ± sd)")
    axes[2].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(RES / "robustness.png", dpi=160)
    plt.close(fig)
    with open(RES / "robustness_tables.md", "w") as f:
        for kind, t in tables.items():
            f.write(f"### {kind}\n\n```\n{t.to_string()}\n```\n\n")
            print(kind, "\n", t.to_string(), "\n")
