"""Amendment 1, A2(a) (post hoc): the bound versus the utility estimate for the candidates.

python diag.py DATASET K T SEED OUT.json      (data via DISTANCE_REVIVAL_DATA)

Runs the proposed rule ('near', unchanged decisions) while also collecting the online estimate of
Fritzke's utility per point of every cluster (an EMA of sum(d2^2 - d1^2) divided by the matching
EMA of counts, Ng_j). At every revival check it logs, per candidate: delta_j/MSE_hat (the bound),
(G_j/Ng_j)/MSE_hat (the utility estimate; logged under the key "exact"), and N_j / mean N.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main():
    ds, k, T, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    import faiss
    from faiss.contrib.e_means import NumpyEmeans
    from faiss.contrib.e_means.e_means import _numpy_unused_threshold
    from distance_revival import DistanceAwareRevivalEmeans
    import datasets

    class Diag(DistanceAwareRevivalEmeans):
        def _assign_rows(self, x, centroids):
            # library assignment (via the near arm), utility gap measured on the side (Amendment 2)
            distance, labels = super()._assign_rows(x, centroids)
            if self._collect and len(centroids) >= 2:
                d2, i2 = faiss.knn(x, centroids, 2)
                second = np.where(i2[:, 0] == labels, d2[:, 1], d2[:, 0]).astype(np.float64)
                self._batch_gap += np.bincount(labels, weights=np.maximum(second - distance, 0.0),
                                               minlength=self.k)
            return distance, labels

        def _choose_splits(self, rng):
            usage = self._usage
            cand = np.flatnonzero(usage < _numpy_unused_threshold(usage, self.k, self._unused_threshold))
            if len(cand):
                C = self._centroids
                delta = faiss.knn(np.ascontiguousarray(C[cand]), C, 2)[0][:, 1].astype(np.float64)
                m = self._dist.sum() / max(float(usage.sum(dtype=np.float64)), 1e-300)
                n_j = usage[cand].astype(np.float64)
                n_g = self._gapn[cand]
                seen = self._seen[cand] & (n_g > 0)
                ex = np.full(len(cand), np.inf)
                ex[seen] = self._gap[cand][seen] / n_g[seen]
                self.diag.append(dict(step=int(self._step), bound=(delta / m).tolist(), exact=(ex / m).tolist(),
                                      rel_count=(n_j / usage.mean()).tolist()))
            return super()._choose_splits(rng)

    X, Xte, _ = datasets.load(ds, seed)
    X = np.ascontiguousarray(X)
    init = X[np.sort(np.random.RandomState(seed).permutation(len(X))[:k])]
    m = Diag(X.shape[1], k, revival="near", nepochs=T, seed=seed)
    m.diag = []
    t0 = time.time()
    m.train(X, init_centroids=init)
    b = np.concatenate([np.asarray(e["bound"]) for e in m.diag]) if m.diag else np.zeros(0)
    e = np.concatenate([np.asarray(e["exact"]) for e in m.diag]) if m.diag else np.zeros(0)
    prot = b > 4
    summary = dict(dataset=ds, k=k, T=T, seed=seed, seconds=time.time() - t0, candidates=int(len(b)),
                   protected_by_bound=int(prot.sum()),
                   protected_but_exact_le4=int((prot & (e <= 4)).sum()),
                   evicted_by_bound_but_exact_gt4=int((~prot & (e > 4)).sum()),
                   unknown_utility=int(np.isinf(e).sum()),
                   median_bound_over_exact=float(np.median(b[np.isfinite(e)] / np.maximum(e[np.isfinite(e)], 1e-12))) if np.isfinite(e).any() else None,
                   median_bound_over_exact_protected=float(np.median(b[prot & np.isfinite(e)] / np.maximum(e[prot & np.isfinite(e)], 1e-12))) if (prot & np.isfinite(e)).any() else None,
                   median_exact_protected=float(np.median(e[prot & np.isfinite(e)])) if (prot & np.isfinite(e)).any() else None,
                   quantiles_bound=np.quantile(b, [.1, .5, .9, .99]).tolist() if len(b) else None,
                   quantiles_exact=np.quantile(e[np.isfinite(e)], [.1, .5, .9, .99]).tolist() if np.isfinite(e).any() else None,
                   n_splits=int(m.n_splits), n_protected=int(m.n_protected))
    json.dump(dict(summary=summary, checks=m.diag), open(out, "w"))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
