"""PCA-LDA, PLS-DA and 1D CNN classifiers behind a common fit / predict interface."""
import numpy as np
import torch
from torch import nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold


class PCALDA:
    def __init__(self, n_components=40):
        self.pca = PCA(n_components=n_components, random_state=0)
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    def fit(self, X, y):
        self.lda.fit(self.pca.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.lda.predict(self.pca.transform(X))


class PLSDA:
    """PLS regression onto one-hot class indicators; prediction is the arg-max column.

    The number of latent variables is chosen by inner cross-validation on the training data.
    """

    def __init__(self, n_classes, grid=(10, 20, 30, 40, 55)):
        self.n_classes, self.grid = n_classes, grid

    def _onehot(self, y):
        Y = np.zeros((len(y), self.n_classes))
        Y[np.arange(len(y)), y] = 1.0
        return Y

