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

