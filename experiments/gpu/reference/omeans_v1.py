# Portions adapted from o-means (github.com/owlgebra-ai/faiss, contrib/o_means, commit 9ecbf0c):
#   Copyright (c) Meta Platforms, Inc. and affiliates. MIT license.
"""REFERENCE (first, plain-PyTorch version; superseded by distance_revival/omeans.py, which makes
the same decisions much faster). Kept as a readable statement of the GPU rule and to reproduce
results/gpu/runs_reference.jsonl.gz.

Distance-aware revival for o-means (owlgebra-ai/faiss contrib/o_means, commit 9ecbf0c).

o-means keeps e-means' revival rule; this swaps in the paper's distance test, everything else
(two-level assignment, cells, split pointers, schedule, closing mean, random streams) unchanged.

GPU design, in o-means' style
- Per batch: each row's squared distance to its assigned centroid (centroids before the update,
  as the library's assignment sees them) is added to its cluster with index_add_; D (an EMA of
  cluster distortion) and the last step each cluster received a row are updated with fixed-shape
  tensor operations. Nothing is read back to the host.
- Every revival check (every 5 batches): one value is read back, the number of size candidates,
  to size the candidates' nearest-other-centre search (one FP32 matrix product per chunk). The
  rule, the donor draw and the bookkeeping of the splits stay on the device, with fixed shapes as
  in o-means (pairs past m write to a spare slot).

Rules: "near" (the paper's), "count" (o-means' own rule; with the statistics on, for testing),
"off" (no revival).
"""
import torch

import faiss.contrib.o_means.o_means as om_mod
from faiss.contrib.o_means.o_means import Omeans, _Training
from faiss.contrib.o_means.triton_kernels import accumulate, apply_splits, ema_update
from faiss.contrib.o_means.triton_two_level import build_cells, permute_rows, record_splits, two_level_assign
from faiss.contrib.e_means.base import _SPLIT_EPSILON, _CHECK_UNUSED_EVERY, _epoch_seed

__all__ = ["OmeansDistanceRevival"]


def nearest_other_sqdist(C, cand, chunk=4096):
    """Squared distance from C[cand] to the nearest other row of C (FP32 rank, FP64 value)."""
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


class _DistanceTraining(_Training):
    rule = "near"
    tau_near = 4.0
    abandon_passes = 2.0

    def __init__(self, x, init, sched, seed):
        super().__init__(x, init, sched, seed)
        kp = self.C.shape[0]
        dev = x.device
        self.spare = kp                                   # write slot for unused split pairs
        self.D = torch.zeros(kp + 1, dtype=torch.float64, device=dev)
        self.bd = torch.zeros(kp + 1, dtype=torch.float32, device=dev)
        self.last_hit = torch.zeros(kp + 1, dtype=torch.int64, device=dev)
        self.counters = torch.zeros(4, dtype=torch.int64, device=dev)   # candidates, protected, abandoned, splits
        self.step_i = 0
        self.steps_per_pass = -(-sched.n // sched.batch_size)

    def stats(self):
        c, p, a, s = self.counters.tolist()
        return dict(candidates=c, protected=p, abandoned=a, splits=s)

    def rebuild_cells(self, first_pass):
        if self.cells is None:
            iters = om_mod._FIRST_BUILD_ITERS
        else:
            iters = om_mod._FIRST_PASS_REBUILD_ITERS if first_pass else om_mod._REBUILD_ITERS
        order, self.cells = build_cells(self.C, self.sched.k, self.sched.nlist, iters,
                                        prev=self.cells, generator=self.cell_rng)
        permute_rows(order, self.sched.k, [self.C, self.c16, self.cn, self.usage,
                                           self.close_sums, self.close_cnt, self.D, self.last_hit])
        self.ids = self.ids[order]

    def revive(self):
        if self.rule == "off":
            return
        k, d = self.sched.k, self.sched.d
        u = self.usage[:k]
        thr = u.sum() * (self.sched.unused_threshold / k)
        dead = u < thr
        donors = (~dead) & (u > 0)
        if self.rule == "near":
            nd = int(dead.sum())                          # the one value read back per check
            if nd:
                cand = torch.sort(dead.to(torch.uint8), descending=True, stable=True).indices[:nd]
                delta = nearest_other_sqdist(self.C[:k, :d], cand)
                mhat = self.D[:k].sum() / u.double().sum().clamp_min(1e-300)
                remove = delta <= self.tau_near * mhat
                window = self.abandon_passes * self.steps_per_pass
                abandoned = (self.step_i - self.last_hit[cand]) >= window
                self.counters[0] += nd
                self.counters[1] += (~(remove | abandoned)).sum()
                self.counters[2] += (abandoned & ~remove).sum()
                dead = torch.zeros_like(dead)
                dead[cand] = remove | abandoned
        # o-means' revive() from here on, with `dead` possibly filtered (same random stream)
        torch.rand(k, generator=self.split_rng, out=self.uniform)
        r = self.uniform.clamp_min(torch.finfo(torch.float32).tiny)
        keys = torch.where(donors, u.log() - (-r.log()).log(), float("-inf"))
        src = torch.argsort(keys, descending=True)
        tgt = torch.sort(dead.to(torch.uint8), descending=True, stable=True).indices
        m = torch.minimum(dead.sum(), donors.sum())
        pairs = k // 2 + 1
        live = torch.arange(pairs, device=m.device) < m
        s = torch.where(live, src[:pairs], self.spare)
        t = torch.where(live, tgt[:pairs], self.spare)
        half = self.D[s] / 2.0                            # distortion is shared like the mass
        self.D[s] = half
        self.D[t] = half
        self.last_hit[t] = self.step_i                    # birth starts the abandonment clock
        self.D[self.spare] = 0.0
        self.counters[3] += m
        recency = self.sched.recency_close
        apply_splits(self.C, self.c16, self.cn, self.usage, src, tgt, m, self.sched.d,
                     _SPLIT_EPSILON, max_pairs=pairs,
                     wsum=self.close_sums if recency else None,
                     wcnt=self.close_cnt if recency else None)
        record_splits(self.cells.child, src, tgt, m, pairs)

    def train(self, labels_out=None):
        # o-means' loop (o_means.py:348-382) plus the statistics lines marked STATS
        sched = self.sched
        n, k, d = sched.n, sched.k, sched.d
        last_epoch = len(sched.epoch_rows) - 1
        perm, epoch = None, -1
        rows_done, since_rebuild = 0, 0
        for step in sched.steps:
            if step.epoch != epoch:
                epoch = step.epoch
                generator = torch.Generator(device=self.x.device)
                generator.manual_seed(_epoch_seed(self.seed, epoch))
                perm = torch.randperm(n, generator=generator, device=self.x.device)
            src, idx, R = self.batch_rows(perm, step.b0, step.b1)
            first_pass = rows_done < n
            every = _CHECK_UNUSED_EVERY if first_pass else sched.rebuild_every
            if self.cells is None or since_rebuild >= every:
                self.rebuild_cells(first_pass)
                since_rebuild = 0
            since_rebuild += 1
            labels = two_level_assign(src, self.c16, self.cn, self.cells, self.ws,
                                      sched.nprobe, idx=idx, rows=R)
            if labels_out is not None and epoch == last_epoch:
                labels_out[perm[step.b0:step.b1]] = self.ids[labels]
            rows_done += R
            if self.rule != "off":                                         # STATS
                lab = labels[:R].long()
                rows = (src[idx] if idx is not None else src[:R])[:, :d]
                self.bd.index_add_(0, lab, (rows.float() - self.C[lab, :d]).square().sum(1))
            accumulate(src, labels, self.sums, self.cnt, d, idx=idx)
            if self.rule != "off":                                         # STATS (cnt is cleared below)
                self.last_hit[:k].masked_fill_(self.cnt[:k] > 0, self.step_i)
            close = step.omega is not None
            ema_update(self.C, self.c16, self.cn, self.usage, self.sums, self.cnt, k, d,
                       step.alpha, self.close_sums if close else None,
                       self.close_cnt if close else None,
                       omega=step.omega if close else 1.0)
            if self.rule != "off":                                         # STATS
                self.D.mul_(1.0 - step.alpha).add_(self.bd.double(), alpha=step.alpha)
                self.bd.zero_()
            if step.check:
                self.revive()
            self.step_i += 1
        return self.closing_mean()


class OmeansDistanceRevival(Omeans):
    """o-means with the distance test in its revival. ``revival`` is "near", "count" or "off"."""

    def __init__(self, d, k, nepochs=5, revival="near", tau_near=4.0, abandon_passes=2.0, **kw):
        if revival not in ("near", "count", "off"):
            raise ValueError(revival)
        super().__init__(d, k, nepochs=nepochs, **kw)
        self.revival = revival
        self.tau_near = float(tau_near)
        self.abandon_passes = float(abandon_passes)
        self.revival_stats = None

    def train(self, x, init_centroids=None):
        cls = type("_RuleTraining", (_DistanceTraining,),
                   dict(rule=self.revival, tau_near=self.tau_near, abandon_passes=self.abandon_passes))
        holder = {}
        init = cls.__init__

        def capture(run, *a, **kw):
            init(run, *a, **kw)
            holder["run"] = run
        cls.__init__ = capture
        original = om_mod._Training
        om_mod._Training = cls          # Omeans.train builds its run from the module's _Training
        try:
            out = super().train(x, init_centroids)
        finally:
            om_mod._Training = original
        self.revival_stats = holder["run"].stats()
        return out
