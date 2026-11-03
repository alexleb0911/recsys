import numpy as np
from .ials import IALS


class CustomIALS(IALS):
    def __init__(self, tau=90.0, frequency_scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.frequency_scale = frequency_scale

    def make_target(self, history, counts=None, last_days=None, cutoff=None):
        if counts is None or last_days is None or cutoff is None:
            raise ValueError('CustomIALS requires purchase counts, last days and cutoff')
        counts = counts.tocsr(copy=True)
        last_days = last_days.tocsr(copy=True)
        counts.sort_indices()
        last_days.sort_indices()
        for matrix in (counts, last_days):
            assert np.array_equal(history.indptr, matrix.indptr)
            assert np.array_equal(history.indices, matrix.indices)
        target = history.copy()
        age = cutoff - last_days.data
        if np.any(age < 0):
            raise ValueError('Future events in training history')
        target.data = (1 - np.exp(-counts.data / self.frequency_scale)) * np.exp(-age / self.tau)
        return target
