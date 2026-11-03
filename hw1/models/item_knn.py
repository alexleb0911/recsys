import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors=50, shrink=20.0):
        self.neighbors = neighbors
        self.shrink = shrink

    def fit(self, history):
        self.history = sparse.csr_matrix(history, dtype=float)
        gram = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.diag(gram))
        denominator = norms[:, None] * norms[None, :]
        similarity = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
        similarity *= gram / (gram + self.shrink)
        np.fill_diagonal(similarity, 0)
        rows, cols, values = [], [], []
        for target in range(gram.shape[0]):
            ids = np.flatnonzero(similarity[:, target] > 0)
            order = np.lexsort((ids, -similarity[ids, target]))[:self.neighbors]
            ids = ids[order]
            rows.extend(ids)
            cols.extend([target] * len(ids))
            values.extend(similarity[ids, target])
        self.weights = sparse.csc_matrix((values, (rows, cols)), shape=gram.shape)
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
