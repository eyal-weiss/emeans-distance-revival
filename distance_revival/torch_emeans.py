"""Distance-aware revival for the PyTorch (GPU) e-means of FAISS 1.16.

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
- Every 5 batches (outside the graph): kernels.distance_rule ranks each candidate's nearest other
  centres on a scaled FP16 copy of the table with tensor cores, re-scores the best two in FP64 and
  applies the rule; then one read of two counts, as the library does at this point. Donors are
  drawn exactly as the library draws them (same generator, same Gumbel keys), so when no candidate
  is protected the run is the library's.

A first implementation, not tuned. Only single-GPU, L2 (non-spherical) e-means is supported, as in
the CPU version. Tested with FAISS 1.16.0, PyTorch 2.15 and Triton 3.9 on an RTX 3090 Ti.
"""
import torch

from faiss.contrib.e_means import Emeans

from .kernels import BLOCK_D, BLOCK_K, distance_rule

__all__ = ["DistanceAwareRevivalTorchEmeans"]


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
        # the rule needs the statistics; record_stats=True also keeps them for the other rules (tests)
        self._needs_stats = revival == "near" or bool(record_stats)
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
        self._counters = torch.zeros(3, dtype=torch.int64, device=dev)  # candidates, protected, abandoned-only
        # padded FP16 copy of the centroids, scaled by a power of two to use FP16's range, for the
        # candidates' neighbour search (refreshed at each check)
        kp = -(-self.k // BLOCK_K) * BLOCK_K
        dp = -(-self.d // BLOCK_D) * BLOCK_D
        self._table = torch.zeros((kp, dp), dtype=torch.float16, device=dev)
        self._table_norms = torch.full((kp,), float("inf"), dtype=torch.float32, device=dev)
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
        self._read_counters()
        return super().finalize()

    # ------------------------------------------------------------------ revival policy
    def _choose_splits(self, rng):
        if self.revival != "near":
            return super()._choose_splits(rng)
        usage = self._usage
        threshold = usage.sum() * self._unused_threshold / self.k
        unused = usage < threshold
        donors = (~unused) & (usage > 0)
        table = self._table[:self.k]
        # power of two putting max|c| in [64, 128) (o-means' rule for its FP16 table); on the device
        scale = torch.exp2(6.0 - torch.floor(torch.log2(self._centroids.abs().amax().clamp_min(1e-30))))
        torch.mul(self._centroids, scale, out=table[:, :self.d])
        torch.sum(table.float().square(), 1, out=self._table_norms[:self.k])
        mhat = self._dist.sum() / usage.double().sum().clamp_min(1e-300)
        window = (float("inf") if self.abandon_passes is None
                  else self.abandon_passes * self.recipe["steps_per_pass"])
        gone = distance_rule(self._table, self._table_norms, self._centroids, unused, self._last_hit,
                             self._counters, mhat, self._step, window, self.tau_near, self.k, self.d)
        # the one read per check (the library reads two counts here)
        nremove, ndonors = torch.stack((gone.sum(), donors.sum())).tolist()
        nsplit = min(nremove, ndonors)
        if nsplit == 0:
            return None, None
        # removed candidates in ascending index order (the library's target order)
        evict = torch.sort(gone.byte(), descending=True, stable=True).indices[:nsplit]
        # identical donor draw to the library
        uniform = torch.rand(self.k, dtype=usage.dtype, device=usage.device,
                             generator=rng).clamp_min(torch.finfo(usage.dtype).tiny)
        keys = torch.where(donors, usage.log() - (-uniform.log()).log(), float("-inf"))
        source = torch.topk(keys, nsplit).indices
        return source, evict

    def _read_counters(self):
        if getattr(self, "_counters", None) is not None:
            self.n_candidates, self.n_protected, self.n_abandoned = self._counters.tolist()
