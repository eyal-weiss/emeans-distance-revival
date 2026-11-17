"""REFERENCE (first, plain-PyTorch version; superseded by distance_revival/torch_emeans.py, which
makes the same decisions faster). Kept as a readable statement of the GPU rule and to reproduce
results/gpu/runs_reference.jsonl.gz.

Distance-aware revival for the PyTorch (GPU) e-means of FAISS 1.16.

A drop-in subclass of ``faiss.contrib.e_means.Emeans``, the GPU counterpart of
``distance_revival.DistanceAwareRevivalEmeans`` (NumpyEmeans). The rule is the paper's:

  revival="near"  remove a size candidate j (N_j < unused_threshold * mean N) only if
                      delta_j <= tau_near * MSE_hat     or     j received no point for W epochs,
                  delta_j = squared distance from c_j to the nearest other centre,
                  MSE_hat = sum_i D_i / sum_i N_i, D_i an EMA of cluster i's batch distortion.
  revival="count" the library's rule; "off" no revival.

GPU design
- Per batch (inside the library's CUDA graph when it replays): the distortion of every row is
  added to its cluster (index_add_ on the distances the assignment already returns), D and the
  last-hit step are updated with fixed-shape tensor operations, and nothing is read back to the
  host. ``_capture_operations_match`` is overridden so that these overrides keep graph replay.
- Every 5 batches (outside the graph, where the library already reads two counts back):
  the candidates' nearest-other-centre distances are computed with one matrix product per chunk,
  and the rule picks which candidates are removed. Donors are drawn exactly as the library does
  (same generator, same Gumbel keys), so when no candidate is protected the run is the library's.

Only L2 (non-spherical) e-means is supported, as in the CPU version.
"""
import torch

from faiss.contrib.e_means import Emeans

__all__ = ["DistanceAwareRevivalTorchEmeans"]


def nearest_other_sqdist(C, cand, chunk=4096):
    """Squared distance from C[cand] to the nearest other row of C.

    FP32 matrix product to rank, then the distance to the winner recomputed in FP64.
    """
    out = torch.empty(len(cand), dtype=torch.float64, device=C.device)
    cn = C.square().sum(1)
    rows = torch.arange(min(chunk, len(cand)), device=C.device)
    for a in range(0, len(cand), chunk):
        ids = cand[a:a + chunk]
        q = C[ids]
        d2 = torch.addmm(cn[None, :], q, C.t(), alpha=-2.0)
        d2[rows[:len(ids)], ids] = float("inf")
        j = d2.argmin(1)
        out[a:a + len(ids)] = (q.double() - C[j].double()).square().sum(1)
    return out


class DistanceAwareRevivalTorchEmeans(Emeans):
    def __init__(self, d, k, revival="near", unused_threshold=0.1, tau_near=4.0,
                 abandon_passes=2.0, record_stats=None, **kw):
        if revival not in ("near", "count", "off"):
            raise ValueError(revival)
        if revival == "off":
            unused_threshold = 0.0
        super().__init__(d, k, unused_threshold=unused_threshold, **kw)
        if self.spherical:
            raise NotImplementedError("the distance test is implemented for L2 e-means only")
        self.revival = revival
        self.tau_near = float(tau_near)
        self.abandon_passes = None if abandon_passes is None else float(abandon_passes)
        # statistics are needed by "near"; record_stats=True keeps them on for other rules (tests)
        self._needs_stats = revival == "near" if record_stats is None else bool(record_stats)
        self._collect = False
        self.n_candidates = self.n_protected = self.n_abandoned = 0

    # ------------------------------------------------------------------ graph replay
    def _capture_operations_match(self, owner=None):
        # Our overrides of _assign_rows and _apply_statistics are capture-safe: fixed shapes,
        # device-side step counter, no host reads. Check against this class instead of Emeans.
        return super()._capture_operations_match(
            DistanceAwareRevivalTorchEmeans if owner is None else owner)

    # ------------------------------------------------------------------ bookkeeping
    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        dev = self._usage.device
        self._dist = torch.zeros(self.k, dtype=torch.float64, device=dev)       # D_j
        self._batch_dist = torch.zeros(self.k, dtype=torch.float32, device=dev)
        self._last_hit = torch.zeros(self.k, dtype=torch.int64, device=dev)
        self._step_dev = torch.zeros((), dtype=torch.int64, device=dev)          # == self._step
        self._collect = self._needs_stats

    def _assign_rows(self, x, centroids, centroid_norm=None, distance_out=None, labels_out=None):
        distance, labels = super()._assign_rows(x, centroids, centroid_norm, distance_out, labels_out)
        if self._collect:
            self._batch_dist.index_add_(0, labels, distance.float())
        return distance, labels

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        if not self._collect:
            return
        if ids is not None:
            raise NotImplementedError("sparse statistics")
        touched = counts > 0
        torch.where(touched, self._step_dev, self._last_hit, out=self._last_hit)
        batch = self._batch_dist.double()
        if replace_usage:
            torch.where(touched, batch * alpha, self._dist, out=self._dist)
        else:
            self._dist.mul_(decay).add_(batch * alpha)
        self._batch_dist.zero_()
        self._step_dev.add_(1)

    def _apply_splits(self, source, target):
        if self._collect:
            half = self._dist[source] / 2.0          # distortion is shared like the count
            self._dist[source] = half
            self._dist[target] = half
            self._last_hit[target] = self._step       # birth starts the abandonment clock
        return super()._apply_splits(source, target)

    def finalize(self):
        self._collect = False
        return super().finalize()

    # ------------------------------------------------------------------ revival policy
    def _choose_splits(self, rng):
        if self.revival != "near":
            return super()._choose_splits(rng)
        usage = self._usage
        threshold = usage.sum() * self._unused_threshold / self.k
        unused = usage < threshold
        donors = (~unused) & (usage > 0)
        nunused, ndonors = torch.stack((unused.sum(), donors.sum())).tolist()   # the library's read
        if nunused == 0 or ndonors == 0:
            return None, None
        # candidates in ascending index order, as the library takes its targets
        cand = torch.sort(unused.byte(), descending=True, stable=True).indices[:nunused]
        delta = nearest_other_sqdist(self._centroids, cand)
        mhat = self._dist.sum() / usage.double().sum().clamp_min(1e-300)
        remove = delta <= self.tau_near * mhat
        abandoned = torch.zeros_like(remove)
        if self.abandon_passes is not None:
            window = self.abandon_passes * self.recipe["steps_per_pass"]
            abandoned = (self._step - self._last_hit[cand]) >= window
        nremove, nab_only = torch.stack(((remove | abandoned).sum(), (abandoned & ~remove).sum())).tolist()
        self.n_candidates += nunused
        self.n_protected += nunused - nremove
        self.n_abandoned += nab_only
        nsplit = min(nremove, ndonors)
        if nsplit == 0:
            return None, None
        evict = cand[torch.sort((remove | abandoned).byte(), descending=True, stable=True).indices[:nsplit]]
        # identical donor draw to the library
        uniform = torch.rand(self.k, dtype=usage.dtype, device=usage.device,
                             generator=rng).clamp_min(torch.finfo(usage.dtype).tiny)
        keys = torch.where(donors, usage.log() - (-uniform.log()).log(), float("-inf"))
        source = torch.topk(keys, nsplit).indices
        return source, evict
