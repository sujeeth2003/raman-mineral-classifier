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

    def fit(self, X, y):
        Y = self._onehot(y)
        counts = np.bincount(y, minlength=self.n_classes)
        k = int(min(3, counts[counts > 0].min()))
        best, best_acc = self.grid[0], -1.0
        if k >= 2:
            skf = StratifiedKFold(k, shuffle=True, random_state=0)
            for nc in self.grid:
                accs = []
                for tr, va in skf.split(X, y):
                    m = PLSRegression(n_components=nc, scale=False).fit(X[tr], Y[tr])
                    accs.append((m.predict(X[va]).argmax(1) == y[va]).mean())
                if np.mean(accs) > best_acc:
                    best, best_acc = nc, float(np.mean(accs))
        self.n_components = best
        self.model = PLSRegression(n_components=best, scale=False).fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X).argmax(1)


class _Net(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))

        self.features = nn.Sequential(block(1, 16, 9), block(16, 32, 7), block(32, 64, 5), nn.AdaptiveAvgPool1d(8))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 8, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class CNN1D:
    def __init__(self, n_classes, epochs=60, lr=2e-3, batch=64, seed=0):
        self.n_classes, self.epochs, self.lr, self.batch, self.seed = n_classes, epochs, lr, batch, seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        self.net = _Net(self.n_classes)
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(y, dtype=torch.long)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        steps = self.epochs * int(np.ceil(len(X) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(perm), self.batch):
                idx = perm[i:i + self.batch]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss_fn(self.net(Xt[idx]), yt[idx]).backward()
                opt.step()
                sched.step()
        return self

