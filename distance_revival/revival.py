"""Distance-aware revival for e-means.

A drop-in subclass of ``faiss.contrib.e_means.NumpyEmeans`` (FAISS 1.16). The library's size
test is unchanged: every 5 batches, clusters whose running count N_j is below
``unused_threshold`` times the average count become *candidates*. This class adds a second
condition that decides which candidates are actually removed (each removed centre is reused to
split a large donor cluster, exactly as in the library):

  revival="near"   (proposed; the distance test) remove candidate j only if
                       delta_j <= tau_near * MSE_hat      or      j received no point for W epochs,
                   where delta_j is the squared distance from c_j to the nearest other centre and
                   MSE_hat = sum_i D_i / sum_i N_i is a running estimate of the mean squared error
                   (D_i is an EMA of cluster i's batch distortion, updated with the same step as N_i).
                   If c_j is the mean of its points, moving all of them to the nearest other centre
                   increases their squared error by delta_j per point on average; letting each point
                   pick its own nearest remaining centre can only cost less. So delta_j bounds the
                   average removal cost per point (Proposition 1 of the paper). The second clause,
                   the abandonment clause, is a safeguard for a centre that has lost all its points.
  revival="exact"  (the utility estimate) the same rule with an online estimate of Fritzke's utility
                   per point, G_j / Ng_j, in place of delta_j. G_j is an EMA of the sum over j's
                   points of (d2^2 - d1^2), d1 and d2 being each point's distances to its assigned
                   centre and to its nearest other centre, and Ng_j is the matching EMA of counts
                   (both restart when a split creates the centre). Needs an extra search for the two
                   nearest centres of every point in every batch; the library's assignment is
                   unchanged. A centre that has not received a point since its creation has unknown
                   utility and is not removed by the utility test (the abandonment clause still
                   applies to it).
  revival="count"  the library's rule (size test only).
  revival="off"    no revival.

``unused_threshold`` defaults to 0.1 in every dimension (the library turns revival off for d <= 2
by default since FAISS 1.16; pass unused_threshold=None to use the library's resolution).
``abandon_passes`` is the waiting period W, in epochs (default 2); ``abandon_passes=None``
disables the abandonment clause.
"""
import numpy as np
import faiss
from faiss.contrib.e_means import NumpyEmeans
from faiss.contrib.e_means.e_means import _numpy_unused_threshold

# log-spaced bins for the candidate ratio delta_j / MSE_hat (recorded when record_ratios=True)
RATIO_BINS = np.logspace(-3, 6, 91)


class DistanceAwareRevivalEmeans(NumpyEmeans):
    def __init__(self, d, k, revival="near", unused_threshold=0.1, tau_near=4.0,
                 abandon_passes=2.0, record_ratios=False, **kw):
        if revival not in ("near", "exact", "count", "off"):
            raise ValueError(revival)
        if revival == "off":
            unused_threshold = 0.0
        super().__init__(d, k, unused_threshold=unused_threshold, **kw)
        if self.spherical and (revival in ("near", "exact") or record_ratios):
            raise NotImplementedError("the distance test is implemented for L2 (non-spherical) e-means only")
        self.revival = revival
        self.tau_near = float(tau_near)
        self.abandon_passes = None if abandon_passes is None else float(abandon_passes)
        self.record_ratios = bool(record_ratios)
        self.n_protected = 0          # candidates kept alive by the distance test
        self.n_candidates = 0         # candidates nominated by the size test (all checks)
        self.n_abandoned = 0          # candidates removed only because of the abandonment clause
        self.ratio_hist = np.zeros(len(RATIO_BINS) + 1, dtype=np.int64)
        self._collect = False

    # ------------------------------------------------------------------ bookkeeping
    @property
    def _needs_stats(self):
        return self.revival in ("near", "exact") or self.record_ratios

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        self._dist = np.zeros(self.k, dtype=np.float64)      # EMA of per-cluster distortion D_j
        self._gap = np.zeros(self.k, dtype=np.float64)       # EMA of sum (d2^2 - d1^2), exact only
        self._seen = np.zeros(self.k, dtype=bool)            # received >= 1 point since birth (exact)
        self._gapn = np.zeros(self.k, dtype=np.float64)      # EMA of the counts behind _gap (exact)
        self._last_hit = np.zeros(self.k, dtype=np.int64)    # last step with >= 1 point
        self._batch_dist = np.zeros(self.k, dtype=np.float64)
        self._batch_gap = np.zeros(self.k, dtype=np.float64)
        self._collect = self._needs_stats

    def _assign_rows(self, x, centroids):
        # the library's assignment is never changed; extra statistics are measured on the side
        distance, labels = super()._assign_rows(x, centroids)
        if not self._collect:
            return distance, labels
        self._batch_dist += np.bincount(labels, weights=np.maximum(distance, 0),
                                        minlength=self.k)
        if self.revival == "exact" and len(centroids) >= 2:
            d2, i2 = faiss.knn(x, centroids, 2)
            # squared distance to the nearest centre other than the assigned one
            second = np.where(i2[:, 0] == labels, d2[:, 1], d2[:, 0]).astype(np.float64)
            gap = np.maximum(second - distance, 0.0)
            self._batch_gap += np.bincount(labels, weights=gap, minlength=self.k)
        return distance, labels

    def _accumulate_core(self, x, index=None, labels_out=None):
        if self._collect and self._pending_stats is None:   # first chunk of a logical batch
            self._batch_dist[:] = 0.0
            self._batch_gap[:] = 0.0
        return super()._accumulate_core(x, index, labels_out=labels_out)

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        if not self._collect:
            return
        touched = counts > 0
        self._last_hit[touched] = self._step
        self._seen[touched] = True
        batch_n = np.asarray(counts, dtype=np.float64)
        for ema, batch in ((self._dist, self._batch_dist), (self._gap, self._batch_gap), (self._gapn, batch_n)):
            if replace_usage:
                ema[touched] = alpha * batch[touched]
            else:
                ema *= decay
                ema[touched] += alpha * batch[touched]

    def _apply_splits(self, source, target):
        if self._collect:
            half = self._dist[source] / 2.0               # distortion is shared like the count
            self._dist[source] = half
            self._dist[target] = half
            self._gap[source] = self._gap[source] / 2.0   # the donor keeps its per-point estimate
            self._gapn[source] = self._gapn[source] / 2.0
            self._gap[target] = 0.0                       # a newborn centre's utility is unknown:
            self._gapn[target] = 0.0                      # numerator and denominator restart together
            self._seen[target] = False
            self._last_hit[target] = self._step           # birth starts the abandonment clock
        return super()._apply_splits(source, target)

    def finalize(self):
        self._collect = False
        return super().finalize()

    # ------------------------------------------------------------------ revival policy
    def _choose_splits(self, rng):
        if self.revival in ("count", "off") and not self.record_ratios:
            return super()._choose_splits(rng)
        usage = self._usage
        threshold = _numpy_unused_threshold(usage, self.k, self._unused_threshold)
        unused = usage < threshold
        donors = (~unused) & (usage > 0)
        evict = np.flatnonzero(unused)
        if len(evict):
            self.n_candidates += len(evict)
            C = self._centroids
            delta = faiss.knn(np.ascontiguousarray(C[evict]), C, 2)[0][:, 1].astype(np.float64)
            m = self._dist.sum() / max(float(usage.sum(dtype=np.float64)), 1e-300)
            if self.record_ratios:
                self.ratio_hist += np.bincount(np.searchsorted(RATIO_BINS, delta / max(m, 1e-300)),
                                               minlength=len(RATIO_BINS) + 1)
            if self.revival in ("near", "exact"):
                if self.revival == "near":
                    cost = delta
                else:   # EMA estimate of the per-point utility; unknown (never observed) -> +inf
                    n_g = self._gapn[evict]
                    seen = self._seen[evict] & (n_g > 0)
                    cost = np.full(len(evict), np.inf)
                    cost[seen] = self._gap[evict][seen] / n_g[seen]
                keep = cost <= self.tau_near * m           # True = remove
                if self.abandon_passes is not None:
                    window = self.abandon_passes * self.recipe["steps_per_pass"]
                    abandoned = (self._step - self._last_hit[evict]) >= window
                    self.n_abandoned += int((abandoned & ~keep).sum())
                    keep |= abandoned
                self.n_protected += int((~keep).sum())
                evict = evict[keep]
        nsplit = min(len(evict), int(donors.sum()))
        if nsplit == 0:
            return None, None
        # identical donor draw to the library (same RNG stream when the same clusters are removed)
        uniform = np.maximum(rng.random_sample(self.k), np.finfo("float64").tiny)
        keys = np.full(self.k, -np.inf)
        keys[donors] = np.log(usage[donors]) - np.log(-np.log(uniform[donors]))
        source = np.argsort(keys)[-nsplit:][::-1]
        target = evict[:nsplit]
        return source, target
