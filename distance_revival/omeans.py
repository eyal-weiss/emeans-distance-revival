# Portions of this file (the kernels and loop marked as copies of o-means) are adapted from o-means,
# github.com/owlgebra-ai/faiss, contrib/o_means (commit 9ecbf0c), whose files carry this notice:
#   Copyright (c) Meta Platforms, Inc. and affiliates.
#   This source code is licensed under the MIT license found in the LICENSE file in the root
#   directory of this source tree.
"""Distance-aware revival for o-means (Bajaj, Nupur and Garg, 2026; owlgebra-ai/faiss contrib/o_means).

o-means keeps e-means' training loop and revival rule and replaces the assignment step. This class
runs o-means with the paper's distance test in its revival; everything else (two-level
assignment, cells, split pointers, schedule, closing mean, random streams) is o-means'. Tested
with o-means at commit 9ecbf0c.

GPU design, in o-means' style (a first implementation, not tuned)
- Per batch: our copy of o-means' accumulate kernel also writes each row's squared distance to its
  assigned centroid (the centroids before the update, as the assignment saw them) and records the
  step; the distances are summed per cluster in FP64 with index_add_ (deterministic under
  torch.use_deterministic_algorithms, so a run stays bitwise reproducible as o-means' runs are), and
  our copy of the EMA kernel also updates D, the EMA of cluster distortion. A batch takes o-means'
  three launches plus the index_add_.
- Every revival check, nothing read back to the host: kernels.distance_rule finds each candidate's
  nearest other centroids on o-means' FP16 table with tensor cores, re-scores the best two in FP64,
  confirms every protection by an exact search and applies the rule. A copy of o-means' split
  kernel also shares the donor's distortion with the child and starts the child's clock.

Rules: "near" (the paper's), "count" (o-means' own rule, with the statistics on: for tests), "off".
"""
import dataclasses

import torch
import triton
import triton.language as tl

try:
    import faiss.contrib.o_means.o_means as om_mod
    from faiss.contrib.o_means.o_means import Omeans, _Training, _check_rows
    from faiss.contrib.o_means.schedule import make_schedule
    from faiss.contrib.o_means.triton_two_level import (build_cells, permute_rows, record_splits,
                                                        two_level_assign)
except ImportError as error:
    raise ImportError(
        "distance_revival.omeans needs o-means: copy contrib/o_means of github.com/owlgebra-ai/faiss "
        "(commit 9ecbf0c) into the contrib folder of your faiss installation (see the README)") from error
from faiss.contrib.e_means.base import _SPLIT_EPSILON, _CHECK_UNUSED_EVERY, _epoch_seed

from .kernels import distance_rule

__all__ = ["OmeansDistanceRevival"]

# ---------------------------------------------------------------------------- fused kernels
@triton.jit(do_not_specialize=["R", "step"])
def _accumulate_dist_kernel(x_ptr, idx_ptr, lab_ptr, sum_ptr, cnt_ptr, C_ptr, rowd_ptr, last_ptr,
                            R, d, sx, ss, sC, step,
                            HAS_IDX: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    # o-means' _accumulate_kernel (triton_kernels.py) plus the lines marked DIST
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < R
    if HAS_IDX:
        rows = tl.load(idx_ptr + r, mask=rm, other=0).to(tl.int64)
    else:
        rows = r.to(tl.int64)
    lab = tl.load(lab_ptr + r, mask=rm, other=0).to(tl.int64)
    acc = tl.zeros((BLOCK_R,), tl.float32)                                       # DIST
    for d0 in range(0, d, BLOCK_D):
        col = d0 + tl.arange(0, BLOCK_D)
        m = rm[:, None] & (col < d)[None, :]
        vf = tl.load(x_ptr + rows[:, None] * sx + col[None, :], mask=m, other=0.0).to(tl.float32)
        v = (vf * 16777216.0).to(tl.int64)
        tl.atomic_add(sum_ptr + lab[:, None] * ss + col[None, :], v, mask=m, sem="relaxed")
        c = tl.load(C_ptr + lab[:, None] * sC + col[None, :], mask=m, other=0.0)  # DIST
        diff = vf - c                                                             # DIST
        acc += tl.sum(diff * diff, axis=1)                                        # DIST
    tl.atomic_add(cnt_ptr + lab, tl.full((BLOCK_R,), 1, tl.int32), mask=rm, sem="relaxed")
    tl.store(rowd_ptr + r, acc.to(tl.float64), mask=rm)                           # DIST
    tl.store(last_ptr + lab, tl.zeros((BLOCK_R,), tl.int64) + step, mask=rm)      # DIST


@triton.jit
def _ema_update_dist_kernel(C_ptr, c16_ptr, cn_ptr, usage_ptr, sum_ptr, cnt_ptr, psum_ptr, pcnt_ptr,
                            D_ptr, bd_ptr, k, d, sC, s16, sS, sP, alpha, decay, omega,
                            CLOSE: tl.constexpr, REPLACE: tl.constexpr, WEIGHTED: tl.constexpr,
                            BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
    # o-means' _ema_update_kernel plus the lines marked DIST
    rows = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    valid = rows < k
    cnt = tl.load(cnt_ptr + rows, mask=valid, other=0)
    touched = cnt > 0
    tm = valid & touched
    count = cnt.to(tl.float32)
    old = tl.load(usage_ptr + rows, mask=valid, other=0.0)
    if REPLACE:
        retained = tl.zeros((BLOCK_K,), tl.float32)
    else:
        retained = old * decay
    step = alpha * count
    mass = retained + step
    w = step / tl.where(touched, mass, 1.0)
    div = tl.where(touched, count, 1.0)
    r64 = rows.to(tl.int64)
    norm = tl.zeros((BLOCK_K,), tl.float32)
    for d0 in range(0, d, BLOCK_D):
        col = d0 + tl.arange(0, BLOCK_D)
        m = tm[:, None] & (col < d)[None, :]
        s = tl.load(sum_ptr + r64[:, None] * sS + col[None, :], mask=m, other=0)
        sf = s.to(tl.float32) * (1.0 / 16777216.0)
        mean = sf / div[:, None]
        coff = r64[:, None] * sC + col[None, :]
        c = tl.load(C_ptr + coff, mask=m, other=0.0)
        newc = tl.where((w == 1.0)[:, None], mean, c + w[:, None] * (mean - c))
        tl.store(C_ptr + coff, newc, mask=m)
        h = newc.to(c16_ptr.dtype.element_ty)
        tl.store(c16_ptr + r64[:, None] * s16 + col[None, :], h, mask=m)
        hf = h.to(tl.float32)
        norm += tl.sum(hf * hf, axis=1)
        if CLOSE:
            poff = r64[:, None] * sP + col[None, :]
            ps = tl.load(psum_ptr + poff, mask=m, other=0)
            if WEIGHTED:
                tl.store(psum_ptr + poff, ps + omega * sf, mask=m)
            else:
                tl.store(psum_ptr + poff, ps + s, mask=m)
        tl.store(sum_ptr + r64[:, None] * sS + col[None, :], tl.zeros_like(s), mask=m)
    tl.store(cn_ptr + rows, norm, mask=tm)
    if REPLACE:
        newu = tl.where(touched, mass, old)
    else:
        newu = tl.where(touched, mass, retained)
    tl.store(usage_ptr + rows, newu, mask=valid)
    if CLOSE:
        pc = tl.load(pcnt_ptr + rows, mask=tm, other=0)
        if WEIGHTED:
            tl.store(pcnt_ptr + rows, pc + omega * count, mask=tm)
        else:
            tl.store(pcnt_ptr + rows, pc + cnt.to(pc.dtype), mask=tm)
    tl.store(cnt_ptr + rows, tl.zeros_like(cnt), mask=tm)
    bd = tl.load(bd_ptr + rows, mask=valid, other=0)                              # DIST
    bdf = bd                                                                      # DIST
    dold = tl.load(D_ptr + rows, mask=valid, other=0.0)                           # DIST
    a64 = alpha.to(tl.float64)                                                    # DIST
    if REPLACE:                                                                   # DIST
        dnew = tl.where(touched, a64 * bdf, dold)
    else:
        dnew = dold * decay.to(tl.float64) + a64 * bdf
    tl.store(D_ptr + rows, dnew, mask=valid)                                      # DIST
    tl.store(bd_ptr + rows, tl.zeros_like(bd), mask=valid)                        # DIST


def accumulate_dist(x16, labels, sums, cnt, d, C, rowd, bd, last_hit, step, idx=None):
    """o-means' accumulate, plus: each row's squared distance to its centroid (before the update)
    written to rowd, summed per cluster into bd (FP64) with index_add_ (deterministic under
    torch.use_deterministic_algorithms, like the rest of a run), and the step recorded per cluster."""
    R = int(labels.shape[0])
    if R == 0:
        return
    block_d = min(128, triton.next_power_of_2(d))
    _accumulate_dist_kernel[(triton.cdiv(R, 16),)](
        x16, idx if idx is not None else labels, labels, sums, cnt, C, rowd, last_hit, R, d,
        x16.stride(0), sums.stride(0), C.stride(0), int(step),
        HAS_IDX=idx is not None, BLOCK_R=16, BLOCK_D=block_d, num_warps=4)
    bd.index_add_(0, labels, rowd[:R])


def ema_update_dist(C, c16, cn, usage, sums, cnt, k, d, alpha, D, bd,
                    close_sums=None, close_cnt=None, omega=1.0):
    close = close_sums is not None
    weighted = close and close_sums.dtype != sums.dtype
    block_d = min(128, triton.next_power_of_2(d))
    block_k = max(1, 1024 // block_d)
    _ema_update_dist_kernel[(triton.cdiv(k, block_k),)](
        C, c16, cn, usage, sums, cnt,
        close_sums if close else sums, close_cnt if close else cnt, D, bd,
        k, d, C.stride(0), c16.stride(0), sums.stride(0),
        close_sums.stride(0) if close else 0, float(alpha), float(1.0 - alpha), float(omega),
        CLOSE=close, REPLACE=alpha == 1.0, WEIGHTED=weighted,
        BLOCK_K=block_k, BLOCK_D=block_d, num_warps=4)


# o-means' _split_kernel (triton_kernels.py) plus the lines marked DIST
@triton.jit(do_not_specialize=["step"])
def _split_dist_kernel(C_ptr, c16_ptr, cn_ptr, usage_ptr, src_ptr, tgt_ptr, m_ptr, d, sC, s16, eps,
                       wsum_ptr, wcnt_ptr, sW, D_ptr, last_ptr, step,
                  SPLIT_CLOSE: tl.constexpr, BLOCK_D: tl.constexpr):
    i = tl.program_id(0)
    m = tl.load(m_ptr)
    if i < m:
        s = tl.load(src_ptr + i).to(tl.int64)
        t = tl.load(tgt_ptr + i).to(tl.int64)
        nlo = tl.zeros((BLOCK_D,), tl.float32)
        nhi = tl.zeros((BLOCK_D,), tl.float32)
        ndiff = tl.zeros((BLOCK_D,), tl.int32)
        for d0 in range(0, d, BLOCK_D):
            col = d0 + tl.arange(0, BLOCK_D)
            cm = col < d
            v = tl.load(C_ptr + s * sC + col, mask=cm, other=0.0)
            off = eps * tl.where(col % 2 == 0, 1.0, -1.0)
            lo = v * (1.0 - off)
            hi = v * (1.0 + off)
            hlo = lo.to(c16_ptr.dtype.element_ty)
            hhi = hi.to(c16_ptr.dtype.element_ty)
            ndiff += tl.where(cm & (hlo != hhi), 1, 0)
            tl.store(C_ptr + s * sC + col, lo, mask=cm)
            tl.store(C_ptr + t * sC + col, hi, mask=cm)
            tl.store(c16_ptr + s * s16 + col, hlo, mask=cm)
            tl.store(c16_ptr + t * s16 + col, hhi, mask=cm)
            flo = hlo.to(tl.float32)
            fhi = hhi.to(tl.float32)
            nlo += flo * flo
            nhi += fhi * fhi
            if SPLIT_CLOSE:
                hw = tl.load(wsum_ptr + s * sW + col, mask=cm, other=0.0) * 0.5
                tl.store(wsum_ptr + s * sW + col, hw, mask=cm)
                tl.store(wsum_ptr + t * sW + col, hw, mask=cm)
        norm_lo = tl.sum(nlo, axis=0)
        norm_hi = tl.sum(nhi, axis=0)
        if tl.sum(ndiff, axis=0) == 0:
            # Donor and child round to the same FP16 row: move the child's
            # first coordinate one FP16 step (outwards, or inwards if that
            # would overflow) so that the two can separate.
            h0 = tl.load(c16_ptr + t * s16)
            b = h0.to(tl.uint16, bitcast=True).to(tl.int32)
            mag = b & 0x7FFF
            neg = (b & 0x8000) != 0
            up = tl.where(mag == 0, 1, tl.where(neg, b - 1, b + 1))
            inward = tl.where(mag == 0, 0x8001, tl.where(neg, b + 1, b - 1))
            moved = tl.where((up & 0x7FFF) < 0x7C00, up, inward)
            moved = tl.where(mag > 0x7C00, b, moved)          # NaN stays NaN
            nb = moved.to(tl.uint16).to(c16_ptr.dtype.element_ty, bitcast=True)
            f_old = h0.to(tl.float32)
            f_new = nb.to(tl.float32)
            norm_hi = norm_hi + f_new * f_new - f_old * f_old
            tl.store(c16_ptr + t * s16, nb)
            tl.store(C_ptr + t * sC, f_new)
        tl.store(cn_ptr + s, norm_lo)
        tl.store(cn_ptr + t, norm_hi)
        half = tl.load(usage_ptr + s) * 0.5
        tl.store(usage_ptr + s, half)
        tl.store(usage_ptr + t, half)
        if SPLIT_CLOSE:
            hc = tl.load(wcnt_ptr + s) * 0.5
            tl.store(wcnt_ptr + s, hc)
            tl.store(wcnt_ptr + t, hc)
        hd = tl.load(D_ptr + s) * 0.5                                             # DIST
        tl.store(D_ptr + s, hd)                                                   # DIST
        tl.store(D_ptr + t, hd)                                                   # DIST
        tl.store(last_ptr + t, step.to(tl.int64))                                 # DIST


def apply_splits_dist(C, c16, cn, usage, src, tgt, m, d, eps, max_pairs, D, last_hit, step,
                      wsum=None, wcnt=None):
    """o-means' apply_splits; donor and child also share the donor's distortion D, and the
    child's abandonment clock starts at this step."""
    block_d = min(256, triton.next_power_of_2(d))
    split_close = wsum is not None
    _split_dist_kernel[(max(1, max_pairs),)](C, c16, cn, usage, src, tgt, m, d, C.stride(0),
                                             c16.stride(0), float(eps),
                                             wsum if split_close else C,
                                             wcnt if split_close else usage,
                                             wsum.stride(0) if split_close else 0,
                                             D, last_hit, int(step),
                                             SPLIT_CLOSE=split_close, BLOCK_D=block_d, num_warps=4)


# ---------------------------------------------------------------------------- training loop
class _DistanceTraining(_Training):
    def __init__(self, x, init, sched, seed, rule="near", tau_near=4.0, abandon_passes=2.0):
        super().__init__(x, init, sched, seed)
        self.rule = rule
        self.tau_near = float(tau_near)
        kp = self.C.shape[0]
        dev = x.device
        self.D = torch.zeros(kp, dtype=torch.float64, device=dev)
        self.bd = torch.zeros(kp, dtype=torch.float64, device=dev)
        self.rowd = torch.empty(sched.batch_size, dtype=torch.float64, device=dev)
        self.last_hit = torch.zeros(kp, dtype=torch.int64, device=dev)
        self.counters = torch.zeros(4, dtype=torch.int64, device=dev)   # candidates, protected, abandoned, splits
        self.step_i = 0
        steps_per_pass = -(-sched.n // sched.batch_size)
        self.window = float("inf") if abandon_passes is None else float(abandon_passes) * steps_per_pass

    def stats(self):
        k = self.sched.k
        mse_hat = self.D[:k].sum() / self.usage[:k].double().sum().clamp_min(1e-300)
        c, p, a, s = self.counters.tolist()
        return dict(candidates=c, protected=p, abandoned=a, splits=s,
                    mse_hat=float(mse_hat) * 2.0 ** (-2 * self.e))

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
        if self.rule == "near":                           # nothing is read back to the host
            mhat = self.D[:k].sum() / u.double().sum().clamp_min(1e-300)
            dead = distance_rule(self.c16, self.cn, self.C, dead, self.last_hit, self.counters, mhat,
                                 self.step_i, self.window, self.tau_near, k, d)
        # o-means' revive() from here on, with `dead` possibly filtered (same random stream)
        torch.rand(k, generator=self.split_rng, out=self.uniform)
        r = self.uniform.clamp_min(torch.finfo(torch.float32).tiny)
        keys = torch.where(donors, u.log() - (-r.log()).log(), float("-inf"))
        src = torch.argsort(keys, descending=True)
        tgt = torch.sort(dead.to(torch.uint8), descending=True, stable=True).indices
        m = torch.minimum(dead.sum(), donors.sum())
        self.counters[3] += m
        recency = self.sched.recency_close
        apply_splits_dist(self.C, self.c16, self.cn, self.usage, src, tgt, m, self.sched.d,
                          _SPLIT_EPSILON, k // 2 + 1, self.D, self.last_hit, self.step_i,
                          wsum=self.close_sums if recency else None,
                          wcnt=self.close_cnt if recency else None)
        record_splits(self.cells.child, src, tgt, m, k // 2 + 1)

    def train(self, labels_out=None):
        # o-means' loop (o_means.py:348-382) with accumulate/ema_update replaced by the fused copies
        if self.rule == "off":
            return super().train(labels_out)
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
            accumulate_dist(src, labels, self.sums, self.cnt, d, self.C, self.rowd, self.bd,
                            self.last_hit, self.step_i, idx=idx)
            close = step.omega is not None
            ema_update_dist(self.C, self.c16, self.cn, self.usage, self.sums, self.cnt, k, d,
                            step.alpha, self.D, self.bd,
                            self.close_sums if close else None,
                            self.close_cnt if close else None,
                            omega=step.omega if close else 1.0)
            if step.check:
                self.revive()
            self.step_i += 1
        return self.closing_mean()


class OmeansDistanceRevival(Omeans):
    """o-means with the distance test in its revival.

    revival: "near" (the proposed rule), "count" (o-means' rule) or "off".
    unused_threshold: the size threshold of the candidates, 0.1 in every dimension like the CPU and
    PyTorch e-means classes; None keeps o-means' own choice (0.1, and no revival for d <= 2).
    tau_near, abandon_passes: the rule's parameters (None disables the abandonment clause).
    All other options are o-means'.
    """

    def __init__(self, d, k, nepochs=5, revival="near", tau_near=4.0, abandon_passes=2.0,
                 unused_threshold=0.1, **kw):
        if revival not in ("near", "count", "off"):
            raise ValueError(revival)
        super().__init__(d, k, nepochs=nepochs, **kw)
        self.revival = revival
        self.tau_near = float(tau_near)
        self.abandon_passes = None if abandon_passes is None else float(abandon_passes)
        self.unused_threshold = None if unused_threshold is None else float(unused_threshold)
        self.revival_stats = None

    def train(self, x, init_centroids=None):
        """o-means' Omeans.train (o_means.py:103-152), with this class's training run and threshold."""
        if self.centroids is not None:
            raise RuntimeError("create a new Omeans object for another run")
        _check_rows(x, self.d)
        n = x.shape[0]
        sched = make_schedule(n, self.d, self.k, self.nepochs, self.batch_law,
                              self.nlist, self.nprobe)
        if self.unused_threshold is not None and self.unused_threshold != sched.unused_threshold:
            tau = self.unused_threshold
            sched = dataclasses.replace(sched, unused_threshold=tau, steps=tuple(
                dataclasses.replace(step, check=tau > 0 and (t + 1) % _CHECK_UNUSED_EVERY == 0)
                for t, step in enumerate(sched.steps)))
        if init_centroids is None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(self.seed)
            init_centroids = x[torch.randperm(n, generator=generator, device=x.device)[:self.k]]
        elif tuple(init_centroids.shape) != (self.k, self.d):
            raise ValueError("init_centroids must have shape (k, d)")
        elif init_centroids.device != x.device:
            raise ValueError("init_centroids must be on the device of x")
        labels = None
        if self.capture_assignments:
            labels = torch.empty(n, dtype=torch.int64, device=x.device)
            if sched.epoch_rows[-1] < n:
                labels.fill_(-1)
        run = _DistanceTraining(x, init_centroids, sched, self.seed, rule=self.revival,
                                tau_near=self.tau_near, abandon_passes=self.abandon_passes)
        self.centroids = run.train(labels)
        self.assignments = labels
        self.recipe = dict(
            batch_law=sched.batch_law,
            batch_size=sched.batch_size,
            alpha0=sched.alpha0,
            nepochs=sched.nepochs,
            epoch_rows=list(sched.epoch_rows),
            total_steps=len(sched.steps),
            unused_threshold=sched.unused_threshold,
            nlist=sched.nlist,
            nprobe=sched.nprobe,
            rebuild_every=sched.rebuild_every,
            closing_mean="recency_weighted" if sched.recency_close else "last_pass",
            resident_fp16_copy=run.x16 is not None,
            scale_exponent=run.e,
            revival=self.revival,
            tau_near=self.tau_near,
            abandon_passes=self.abandon_passes,
        )
        self.revival_stats = run.stats()
        return self.centroids, self.assignments
