"""Synthetic datasets (regenerated per seed; scored in-sample).

synth_far:<m>     d=32. 100 Gaussian blobs (2,000 points each; e-means paper's mixture recipe,
                  sigma_mu^2 = 2/3, sigma_w^2 = 1/3) plus 5 far groups of m points each, centred at
                  distance D=20 from the origin, spread sigma_w. k = 105, so an optimal solution gives
                  every far group its own centre. Metric: far groups that end with a centre within D/4.
synth_hetero:<d>  100 blobs of 2,000 points with very different scales: 90 tight (sigma = 0.1) and
                  10 broad (sigma = 3), means drawn N(0, 10^2 I). k = 100. Probes the global
                  yardstick m: a redundant small cluster inside a broad blob can look "far" relative
                  to the global per-point error, especially in low dimension.
"""
import numpy as np


def _far(seed, m, D=20.0, d=32, G=100, per=2000, R=5):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, np.sqrt(2 / 3), (G, d))
    X = mu[rng.integers(0, G, G * per)] + rng.normal(0, np.sqrt(1 / 3), (G * per, d))
    dirs = rng.normal(size=(R, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    far_c = D * dirs
    far = np.repeat(far_c, m, axis=0) + rng.normal(0, np.sqrt(1 / 3), (R * m, d))
    X = np.vstack([X, far]).astype(np.float32)
    X = np.ascontiguousarray(X[rng.permutation(len(X))])
    return X, dict(k_default=G + R, far_centres=far_c.astype(np.float32), D=D)


def _hetero(seed, d, G=100, per=2000, n_broad=10, tight=0.1, broad=3.0, spread=10.0):
    rng = np.random.default_rng(10_000 + seed)
    mu = rng.normal(0, spread, (G, d))
    scale = np.full(G, tight)
    scale[:n_broad] = broad
    lab = np.repeat(np.arange(G), per)
    X = mu[lab] + rng.normal(size=(G * per, d)) * scale[lab, None]
    X = np.ascontiguousarray(X[rng.permutation(len(X))].astype(np.float32))
    return X, dict(k_default=G)


def make(name, seed):
    kind, _, par = name.partition(":")
    if kind == "synth_far":
        X, info = _far(seed, int(par))
    elif kind == "synth_hetero":
        X, info = _hetero(seed, int(par))
    else:
        raise KeyError(name)
    info.update(name=name, n_train=len(X), d=X.shape[1], in_sample=True)
    return X, X, info


def far_covered(centroids, far_centres, D):
    d2 = ((far_centres[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return int((d2.min(1) < (D / 4) ** 2).sum())
