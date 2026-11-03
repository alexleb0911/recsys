import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, reg=100.0):
        self.reg = reg

    def fit(self, history):
        if self.reg <= 0:
            raise ValueError('reg must be positive')
        self.history = sparse.csr_matrix(history, dtype=float)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        factor = cho_factor(gram, lower=True, check_finite=False)
        precision = cho_solve(factor, np.eye(gram.shape[0]), check_finite=False)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.history[users] @ self.weights
