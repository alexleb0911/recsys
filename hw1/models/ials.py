import numpy as np
from scipy import sparse
from scipy.linalg import solve


class IALS:
    def __init__(self, factors=32, reg=10.0, alpha=20.0, iterations=8, seed=42):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def make_target(self, history, counts=None, last_days=None, cutoff=None):
        return history.copy()

    def _update(self, target, observed, fixed):
        result = np.zeros((observed.shape[0], self.factors))
        base = fixed.T @ fixed + self.reg * np.eye(self.factors)
        for row in range(observed.shape[0]):
            start, end = observed.indptr[row:row + 2]
            ids = observed.indices[start:end]
            if len(ids) == 0:
                continue
            vectors = fixed[ids]
            values = target.data[start:end]
            system = base + self.alpha * (vectors.T @ vectors)
            rhs = (1 + self.alpha) * (vectors.T @ values)
            result[row] = solve(system, rhs, assume_a='pos', check_finite=False)
        return result

    def fit(self, history, counts=None, last_days=None, cutoff=None):
        history = sparse.csr_matrix(history, dtype=float)
        history.sort_indices()
        target = self.make_target(history, counts, last_days, cutoff)
        assert np.array_equal(history.indices, target.indices)
        assert np.array_equal(history.indptr, target.indptr)
        transposed = history.T.tocsr()
        transposed_target = target.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.05, (history.shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._update(target, history, self.item_factors)
            self.item_factors = self._update(transposed_target, transposed, self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
