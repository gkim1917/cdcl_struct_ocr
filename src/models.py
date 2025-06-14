# src/models.py
import numpy as np
from typing import Sequence, List
from .sat_infer import SATInfer

class StructPerceptron:
    """
    Unary structured perceptron with optional SAT decoding.

    sat_train = False  → use greedy arg-max during training  (fast)
    sat_train = True   → use SAT decoding for every predict  (slow)
    """
    def __init__(self, feat_dim=128, L=26, lr=0.2, sat_infer=False):
        self.lr = lr
        self.L = L
        
        # unary weights
        self.W = np.random.normal(0, 0.01, (L, feat_dim + 1))
        self.T = np.zeros((L, L), dtype=np.float32)

        self.W_cumulative = np.zeros_like(self.W, dtype=np.float64)
        self.T_cumulative = np.zeros_like(self.T, dtype=np.float64)
        self.step = 0

        self.sat_infer = sat_infer
        self.sat_engine = SATInfer() if sat_infer else None

    def _add_bias(self,X):
        ones = np.ones((len(X), 1), dtype=X.dtype)
        return np.hstack([X, ones])

    def _viterbi(self, scores):
        """scores = (n,L) array of unary dot products"""
        n, L = scores.shape
        bp  = np.zeros((n, L), dtype=np.int16)
        delta   = np.zeros((n, L), dtype=np.float32) 

        delta[0] = scores[0]                # first position: unary only
        for i in range(1, n):
            tmp = delta[i-1][:, None] + self.T        # L×L
            bp[i]   = tmp.argmax(0)
            delta[i] = tmp.max(0) + scores[i]

        y = [int(delta[-1].argmax())]
        for i in range(n - 1, 0, -1):
            y.append(int(bp[i, y[-1]]))
        y.reverse()
        return y

    def _greedy_argmax(self, scores):
        return scores.argmax(axis=1).tolist()

    def predict(self, X):
        scores = self._add_bias(X) @ self.W.T      # (n × L)
        if self.sat_infer:
            # SATInfer.argmax expects a numpy array
            return self.sat_engine.argmax(scores, self.T)
        else:
            return self._viterbi(scores)

    def update(self, X, y_true):
        """Structured perceptron update rule"""
        y_pred = self.predict(X)
        if y_pred == y_true:
            self._avg_step()
            return 0

        Xb = self._add_bias(X)

        # unary update
        for t, (yt, yp) in enumerate(zip(y_true, y_pred)):
            if yt != yp:
                self.W[yt] += self.lr * Xb[t]
                self.W[yp] -= self.lr * Xb[t]
        
        # bigram update
        for (p, q) in zip(y_true[:-1], y_true[1:]):
            self.T[p, q] += self.lr
        for (p, q) in zip(y_pred[:-1], y_pred[1:]):
            self.T[p, q] -= self.lr

        self._avg_step()
        return 1

    def _avg_step(self):
        self.step += 1
        self.W_cumulative += self.W
        self.T_cumulative += self.T

    def averaged_params(self):
        """call after training → replace W,T by their running averages"""
        denom = max(1, self.step)
        self.W = (self.W_cumulative / denom).astype(np.float32)
        self.T = (self.T_cumulative / denom).astype(np.float32)