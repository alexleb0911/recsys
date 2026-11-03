import numpy as np
import pandas as pd

K = 10
PRIMARY_METRIC = 'NDCG@10'


def top_items(scores, seen, candidates, k=K):
    eligible = candidates.copy()
    eligible[seen] = False
    ids = np.flatnonzero(eligible)
    return ids[np.lexsort((ids, -scores[ids]))[:k]]


def user_metrics(top, relevant, k=K):
    hits = np.isin(top, list(relevant)).astype(float)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    return {'NDCG@10': float(hits @ discounts[:len(hits)] / discounts[:min(k, len(relevant))].sum()),
            'Recall@10': float(hits.sum() / len(relevant)),
            'HR@10': float(hits.sum() > 0)}


def evaluation_context(history, future):
    candidates = np.asarray(history.sum(axis=0)).ravel() > 0
    truth = {}
    pairs = future[['u', 'i']].drop_duplicates()
    repeated = 0
    cold_items = 0
    for u, group in pairs.groupby('u', sort=True):
        u = int(u)
        seen = set(history.indices[history.indptr[u]:history.indptr[u + 1]])
        relevant = set()
        for item in group.i:
            if not candidates[item]:
                cold_items += 1
            elif item in seen:
                repeated += 1
            elif seen:
                relevant.add(int(item))
        if relevant:
            truth[u] = relevant
    if not truth:
        raise ValueError('No eligible evaluation users')
    report = {'users': len(truth), 'candidate_items': int(candidates.sum()),
              'future_distinct_pairs': len(pairs), 'repeated_pairs': repeated,
              'cold_item_pairs': cold_items, 'relevant_pairs': sum(map(len, truth.values()))}
    return {'history': history, 'candidates': candidates, 'truth': truth, 'report': report}


def evaluate(model, context, batch_size=128):
    history, candidates, truth = (context[key] for key in ['history', 'candidates', 'truth'])
    users = np.array(sorted(truth), dtype=int)
    rows = []
    recommendations = set()
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.asarray(model.score(batch))
        assert scores.shape == (len(batch), history.shape[1])
        assert np.isfinite(scores).all()
        for u, values in zip(batch, scores):
            seen = history.indices[history.indptr[u]:history.indptr[u + 1]]
            top = top_items(values, seen, candidates)
            assert len(top) == K and len(set(top)) == K
            assert set(top).isdisjoint(seen)
            recommendations.update(top)
            rows.append({'u': int(u), **user_metrics(top, truth[u])})
    details = pd.DataFrame(rows)
    metrics = details[['NDCG@10', 'Recall@10', 'HR@10']].mean().to_dict()
    metrics['Coverage@10'] = len(recommendations) / candidates.sum()
    metrics['users'] = len(details)
    return metrics, details


class Popularity:
    def fit(self, history):
        self.popularity = np.asarray(history.sum(axis=0)).ravel()
        return self

    def score(self, users):
        return np.broadcast_to(self.popularity, (len(users), len(self.popularity)))
