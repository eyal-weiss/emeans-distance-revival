"""Bit-identity and behaviour tests for DistanceAwareRevivalEmeans (run: python -m pytest tests)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from faiss.contrib.e_means import NumpyEmeans  # noqa: E402
from distance_revival import DistanceAwareRevivalEmeans  # noqa: E402
import synthetic  # noqa: E402


def blobs(seed=0, n=40_000, d=16, g=64):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, (g, d))
    X = mu[rng.integers(0, g, n)] + 0.3 * rng.normal(size=(n, d))
    return np.ascontiguousarray(X, dtype=np.float32)


def fit(cls, X, k, T=3, seed=0, **kw):
    init = X[np.sort(np.random.RandomState(seed).permutation(len(X))[:k])]
    m = cls(X.shape[1], k, nepochs=T, seed=seed, **kw)
    m.train(X, init_centroids=init)
    return m


@pytest.mark.parametrize("record", [False, True])
def test_count_is_library(record):
    X = blobs()
    a = fit(NumpyEmeans, X, 128, unused_threshold=0.1)
    b = fit(DistanceAwareRevivalEmeans, X, 128, revival="count", record_ratios=record)
    assert np.array_equal(a.centroids, b.centroids)
    assert a.n_splits == b.n_splits > 0


def test_off_is_library_off():
    X = blobs()
    a = fit(NumpyEmeans, X, 128, unused_threshold=0.0)
    b = fit(DistanceAwareRevivalEmeans, X, 128, revival="off")
    assert np.array_equal(a.centroids, b.centroids)


@pytest.mark.parametrize("rule", ["near", "exact"])
def test_infinite_tau_is_library(rule):
    X = blobs()
    a = fit(NumpyEmeans, X, 128, unused_threshold=0.1)
    b = fit(DistanceAwareRevivalEmeans, X, 128, revival=rule, tau_near=np.inf, abandon_passes=None)
    assert np.array_equal(a.centroids, b.centroids)


def test_near_identical_on_plain_blobs():
    X = blobs(1)
    a = fit(NumpyEmeans, X, 128, unused_threshold=0.1, seed=1)
    b = fit(DistanceAwareRevivalEmeans, X, 128, revival="near", seed=1)
    assert b.n_protected == 0
    assert np.array_equal(a.centroids, b.centroids)


def test_near_keeps_far_groups():
    X, _, info = synthetic.make("synth_far:50", 0)
    from sklearn.cluster import kmeans_plusplus
    init = kmeans_plusplus(X, 105, random_state=0)[0].astype(np.float32)
    res = {}
    for rule in ("count", "near"):
        m = DistanceAwareRevivalEmeans(X.shape[1], 105, revival=rule, nepochs=5, seed=0)
        m.train(X, init_centroids=init)
        res[rule] = synthetic.far_covered(m.centroids, info["far_centres"], info["D"])
    assert res["near"] > res["count"]


def tie_heavy(seed=0, n=40_000, d=8):
    """integer-valued data with many duplicate rows: many exactly tied distances."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.integers(0, 4, (n, d)), dtype=np.float32)


@pytest.mark.parametrize("rule", ["near", "exact"])
def test_infinite_tau_is_library_with_ties(rule):
    X = tie_heavy()
    a = fit(NumpyEmeans, X, 128, unused_threshold=0.1)
    b = fit(DistanceAwareRevivalEmeans, X, 128, revival=rule, tau_near=np.inf, abandon_passes=None)
    assert np.array_equal(a.centroids, b.centroids)
    assert a.n_splits == b.n_splits
