# src/models.py
import numpy as np
from .sat_infer import SATInfer
from .cnf_encoder import CNFEncoder

class StructPerceptron:
    def __init__(self, feat_dim=128, L=26, lr=1.0):
        self.W = np.zeros((L, feat_dim))   # label weights
        self.encoder = CNFEncoder()
        self.decoder = SATInfer(self.encoder)
        self.lr = lr

    def phi(self, X, label_idx):
        return X @ self.W[label_idx].T   # same as dot(W[label], x)

    def predict(self, X):
        scores = np.stack([X @ w for w in self.W])  # (L, n)
        scores = scores.T  # (n, L)
        return self.decoder.decode(scores)

    def update(self, X, y_true):
        y_pred = self.predict(X)
        if y_pred != y_true.tolist():
            for pos,(yt,yp) in enumerate(zip(y_true, y_pred)):
                self.W[yt] += self.lr * X[pos]
                self.W[yp] -= self.lr * X[pos]
            return 1  # mistake
        return 0
