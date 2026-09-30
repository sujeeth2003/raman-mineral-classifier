"""PCA-LDA, PLS-DA and 1D CNN classifiers behind a common fit / predict interface."""
import numpy as np
import torch
from torch import nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold


