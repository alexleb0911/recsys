import numpy as np
from scipy import sparse
from models import EASE, IALS, CustomIALS, FM
from evaluation import user_metrics, top_items


def check_implementations():
    rng = np.random.default_rng(7)
    history = sparse.csr_matrix((rng.random((12, 7)) < 0.35).astype(float))
    ease = EASE(reg=2).fit(history)
    gram = (history.T @ history).toarray()
    residual = (gram + 2 * np.eye(7)) @ ease.weights - gram
    mask = ~np.eye(7, dtype=bool)
    assert np.max(np.abs(residual[mask])) < 1e-10
    assert np.all(ease.weights.diagonal() == 0)

    als = IALS(factors=3, reg=2, alpha=4, iterations=1)
    fixed = rng.normal(size=(7, 3))
    values = als._update(history, history, fixed)
    for u in range(12):
        p = history[u].toarray().ravel()
        c = 1 + 4 * p
        expected = np.linalg.solve(fixed.T @ (c[:, None] * fixed) + 2 * np.eye(3), fixed.T @ (c * p))
        assert np.allclose(values[u], expected)
    days, counts = history.copy(), history.copy()
    days.data[:] = 10
    counts.data[:] = 2
    custom = CustomIALS(tau=30, frequency_scale=2)
    target = custom.make_target(history, counts, days, cutoff=20)
    assert np.allclose(target.data, (1 - np.exp(-1)) * np.exp(-10 / 30))
    assert not np.allclose(target.data, history.data)

    features = sparse.hstack([sparse.eye(7), sparse.csr_matrix(np.eye(2)[np.arange(7) % 2])], format='csr')
    fm = FM(factors=3, reg=0.02)
    fm.initialize(12, features)
    u, i, j = np.array([0, 2]), np.array([1, 4]), np.array([3, 5])
    xp, xn = fm.pairs(u, i), fm.pairs(u, j)
    score, _, _ = fm.forward(xp)
    brute = []
    for row in xp:
        ids, weights = row.indices, row.data
        result = float(row @ fm.w)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                result += weights[a] * weights[b] * (fm.V[ids[a]] @ fm.V[ids[b]])
        brute.append(result)
    assert np.allclose(score, brute)
    _, gw, gv = fm.loss_gradients(xp, xn)
    for parameter, grad, index in [(fm.w, gw, 13), (fm.V, gv, (0, 1)), (fm.V, gv, (13, 2))]:
        old = parameter[index]
        parameter[index] = old + 1e-6
        above = fm.loss_gradients(xp, xn)[0]
        parameter[index] = old - 1e-6
        below = fm.loss_gradients(xp, xn)[0]
        parameter[index] = old
        assert np.isclose((above - below) / 2e-6, grad[index], atol=1e-7)
    fm._prepare_scoring()
    assert np.allclose(score, fm.score(u)[np.arange(len(u)), i])
    metric = user_metrics(np.array([1, 7, 2]), {1, 2}, k=3)
    assert metric['Recall@10'] == 1
    assert np.isclose(metric['NDCG@10'], 1.5 / (1 + 1 / np.log2(3)))
    assert np.array_equal(top_items(np.ones(5), [0, 2], np.ones(5, bool), k=2), [1, 3])
    print('Проверки EASE, iALS, FM и метрик пройдены.')
