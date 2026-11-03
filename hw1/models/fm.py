import numpy as np
from scipy import sparse
from scipy.special import expit


class FM:
    def __init__(self, factors=16, epochs=12, lr=0.01, reg=1e-4,
                 batch_size=1024, seed=42):
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed

    def initialize(self, n_users, item_features):
        self.n_users = n_users
        self.item_features = sparse.csr_matrix(item_features, dtype=float)
        dim = n_users + item_features.shape[1]
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(dim)
        self.V = rng.normal(0, 0.05, (dim, self.factors))
        self.mw, self.sw = np.zeros_like(self.w), np.zeros_like(self.w)
        self.mv, self.sv = np.zeros_like(self.V), np.zeros_like(self.V)
        self.step = 0

    def pairs(self, users, items):
        user_features = sparse.csr_matrix(
            (np.ones(len(users)), (np.arange(len(users)), users)),
            shape=(len(users), self.n_users))
        return sparse.hstack([user_features, self.item_features[items]], format='csr')

    def forward(self, features):
        projected = features @ self.V
        squared = features.multiply(features)
        score = features @ self.w + 0.5 * np.sum(
            projected ** 2 - squared @ (self.V ** 2), axis=1)
        return np.asarray(score).ravel(), projected, squared

    def gradients(self, features, projected, squared, errors):
        gw = np.asarray(features.T @ errors).ravel()
        gv = features.T @ (errors[:, None] * projected)
        gv -= np.asarray(squared.T @ errors).ravel()[:, None] * self.V
        return gw, np.asarray(gv)

    def loss_gradients(self, positive, negative):
        pos, vp, xp2 = self.forward(positive)
        neg, vn, xn2 = self.forward(negative)
        weight = expit(neg - pos)
        gwp, gvp = self.gradients(positive, vp, xp2, -weight)
        gwn, gvn = self.gradients(negative, vn, xn2, weight)
        size = positive.shape[0]
        loss = np.logaddexp(0, neg - pos).mean()
        loss += self.reg / 2 * (np.sum(self.w ** 2) + np.sum(self.V ** 2))
        return float(loss), (gwp + gwn) / size + self.reg * self.w, \
            (gvp + gvn) / size + self.reg * self.V

    def update(self, gw, gv):
        self.step += 1
        for parameter, first, second, gradient in [
            (self.w, self.mw, self.sw, gw), (self.V, self.mv, self.sv, gv)]:
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient ** 2
            parameter -= self.lr * (first / (1 - 0.9 ** self.step)) / (
                np.sqrt(second / (1 - 0.999 ** self.step)) + 1e-8)

    def fit(self, history, item_features):
        history = sparse.csr_matrix(history)
        self.initialize(history.shape[0], item_features)
        seen = history.toarray().astype(bool)
        candidates = np.flatnonzero(np.asarray(history.sum(axis=0)).ravel() > 0)
        pairs = history.tocoo()
        eligible = seen[:, candidates].sum(axis=1) < len(candidates)
        users, items = pairs.row[eligible[pairs.row]], pairs.col[eligible[pairs.row]]
        if len(users) == 0:
            raise ValueError('No positive pairs with eligible negatives')
        rng = np.random.default_rng(self.seed)
        self.losses = []
        for epoch in range(self.epochs):
            order = rng.permutation(len(users))
            losses = []
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                u, i = users[batch], items[batch]
                j = rng.choice(candidates, len(u))
                collision = seen[u, j]
                while collision.any():
                    j[collision] = rng.choice(candidates, int(collision.sum()))
                    collision = seen[u, j]
                positive, negative = self.pairs(u, i), self.pairs(u, j)
                loss, gw, gv = self.loss_gradients(positive, negative)
                self.update(gw, gv)
                losses.append((loss, len(batch)))
            self.losses.append(sum(loss * n for loss, n in losses) / len(users))
        self._prepare_scoring()
        return self

    def _prepare_scoring(self):
        item_v = self.V[self.n_users:]
        self.projected_items = self.item_features @ item_v
        self.item_offsets = self.item_features @ self.w[self.n_users:] + 0.5 * np.sum(
            self.projected_items ** 2 - self.item_features.multiply(self.item_features) @ (item_v ** 2), axis=1)

    def score(self, users):
        return self.w[users, None] + self.item_offsets[None, :] + self.V[users] @ self.projected_items.T
