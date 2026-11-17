"""Triton kernels of the distance test, shared by the GPU e-means and o-means classes.

distance_rule(...) decides, for every size candidate (dead[j] true), whether it is removed:

    removed_j = dead_j and ( delta_j <= tau * mhat  or  step - last_hit_j >= window )

delta_j is the squared distance from centroid j to its nearest other centroid. Candidates are first
sorted to the front of a fixed-length index list (a stable device sort); kernels read the count from
device memory and tiles past it return at once, so the work grows with the number of candidates
while that number is never read back to the host.

1. Scan: (candidate tiles of BLOCK_M) x (up to 8 slices of the table) rank the nearest other
   centroids on an FP16 copy of the table with tensor cores (FP32 accumulation) and keep the best two.
2. Apply: merge the slices, re-score both exactly from the FP32 centroids (FP64 sums) and apply the
   rule. A removal by distance is exact: some other centre is within the threshold.
3. Verify: a candidate not removed by distance may hide a nearer centre that the 16-bit ranking
   missed, so each one is confirmed by an exact search (FP32 differences, no norms, all k centres,
   split over up to 64 slices); if some centre is within the threshold, it is removed after all.
   The decisions are therefore those of an exact FP32 search.

Counters (candidates, kept, removed only by the abandonment clause) are added on the device.

The scan kernel follows the tiling of o-means' dense assignment kernel (owlgebra-ai/faiss,
contrib/o_means/triton_kernels.py, _assign_ksplit_kernel). A first implementation, not tuned: the
search could stop as soon as some centre within tau * mhat is found (enough to remove a
candidate), and in o-means it could search only the cells near each candidate.
"""
import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_K = 256
BLOCK_D = 32


MAX_PARTS = 8
VERIFY_PARTS = 64          # slices of the centres in the exact search, of about VERIFY_SLICE each
VERIFY_SLICE = 256
VERIFY_PROGRAMS = 64       # programs per slice, each looping over candidates
VERIFY_BLOCK_K, VERIFY_BLOCK_D = 128, 32


@triton.jit
def _merge_two(b1, l1, b2, l2, c1, m1, c2, m2):
    """Best two of the sorted pairs (b1 <= b2) and (c1 <= c2); ties keep the first pair."""
    first = c1 < b1
    n2 = tl.where(first, tl.minimum(b1, c2), tl.minimum(b2, c1))
    k2 = tl.where(first, tl.where(b1 <= c2, l1, m2), tl.where(b2 <= c1, l2, m1))
    n1 = tl.where(first, c1, b1)
    k1 = tl.where(first, m1, l1)
    return n1, k1, n2, k2


@triton.jit
def _rule_scan_kernel(t_ptr, tn_ptr, cand_ptr, ncand_ptr, ws_s_ptr, ws_l_ptr, KPART, DP, st, KCAP,
                      BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    part = tl.program_id(1)
    i = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ncand = tl.load(ncand_ptr)
    if pid * BLOCK_M < ncand:
        dead = i < ncand
        rows = tl.load(cand_ptr + i, mask=dead, other=0).to(tl.int64)
        qbase = t_ptr + rows[:, None] * st
        best = tl.full((BLOCK_M,), float("inf"), tl.float32)
        lab = tl.zeros((BLOCK_M,), tl.int64)
        best2 = tl.full((BLOCK_M,), float("inf"), tl.float32)
        lab2 = tl.zeros((BLOCK_M,), tl.int64)
        kbeg = part * KPART
        for k0 in range(kbeg, kbeg + KPART, BLOCK_K):
            cid = k0 + tl.arange(0, BLOCK_K)
            cbase = t_ptr + cid[:, None].to(tl.int64) * st
            acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
            for d0 in range(0, DP, BLOCK_D):
                col = d0 + tl.arange(0, BLOCK_D)
                q = tl.load(qbase + col[None, :], mask=dead[:, None], other=0.0)
                c = tl.load(cbase + col[None, :])
                acc = tl.dot(q, tl.trans(c), acc=acc, out_dtype=tl.float32)
            score = tl.load(tn_ptr + cid)[None, :] - 2.0 * acc
            score = tl.where(cid[None, :] == rows[:, None], float("inf"), score)   # not itself
            bs, bl = tl.min(score, axis=1, return_indices=True, return_indices_tie_break_left=True)
            score2 = tl.where(tl.arange(0, BLOCK_K)[None, :] == bl[:, None], float("inf"), score)
            bs2, bl2 = tl.min(score2, axis=1, return_indices=True, return_indices_tie_break_left=True)
            best, lab, best2, lab2 = _merge_two(best, lab, best2, lab2,
                                                bs, bl.to(tl.int64) + k0, bs2, bl2.to(tl.int64) + k0)
        base = part * 2 * KCAP
        tl.store(ws_s_ptr + base + i, best, mask=dead)
        tl.store(ws_l_ptr + base + i, lab, mask=dead)
        tl.store(ws_s_ptr + base + KCAP + i, best2, mask=dead)
        tl.store(ws_l_ptr + base + KCAP + i, lab2, mask=dead)


@triton.jit(do_not_specialize=["step"])
def _rule_apply_kernel(C_ptr, cand_ptr, ncand_ptr, ws_s_ptr, ws_l_ptr, out_ptr, unsure_ptr, last_ptr,
                       cnt_ptr, mhat_ptr, d, sC, step, window, tau, KCAP,
                       PARTS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ncand = tl.load(ncand_ptr)
    if pid * BLOCK_M < ncand:
        dead = i < ncand
        rows = tl.load(cand_ptr + i, mask=dead, other=0).to(tl.int64)
        best = tl.load(ws_s_ptr + i, mask=dead, other=float("inf"))
        lab = tl.load(ws_l_ptr + i, mask=dead, other=0)
        best2 = tl.load(ws_s_ptr + KCAP + i, mask=dead, other=float("inf"))
        lab2 = tl.load(ws_l_ptr + KCAP + i, mask=dead, other=0)
        for p in tl.static_range(1, PARTS):
            base = p * 2 * KCAP
            c1 = tl.load(ws_s_ptr + base + i, mask=dead, other=float("inf"))
            m1 = tl.load(ws_l_ptr + base + i, mask=dead, other=0)
            c2 = tl.load(ws_s_ptr + base + KCAP + i, mask=dead, other=float("inf"))
            m2 = tl.load(ws_l_ptr + base + KCAP + i, mask=dead, other=0)
            best, lab, best2, lab2 = _merge_two(best, lab, best2, lab2, c1, m1, c2, m2)
        # exact FP64 distances to the best two, keep the smaller (guards against 16-bit misranking)
        delta = tl.zeros((BLOCK_M,), tl.float64)
        delta2 = tl.zeros((BLOCK_M,), tl.float64)
        m2ok = dead & (best2 < float("inf"))
        for d0 in range(0, d, BLOCK_DE):
            col = d0 + tl.arange(0, BLOCK_DE)
            cm = (col < d)[None, :]
            a = tl.load(C_ptr + rows[:, None] * sC + col[None, :], mask=dead[:, None] & cm, other=0.0).to(tl.float64)
            b = tl.load(C_ptr + lab[:, None] * sC + col[None, :], mask=dead[:, None] & cm, other=0.0).to(tl.float64)
            b2 = tl.load(C_ptr + lab2[:, None] * sC + col[None, :], mask=m2ok[:, None] & cm, other=0.0).to(tl.float64)
            delta += tl.sum((a - b) * (a - b), axis=1)
            delta2 += tl.sum((a - b2) * (a - b2), axis=1)
        delta = tl.where(m2ok, tl.minimum(delta, delta2), delta)
        mhat = tl.load(mhat_ptr)
        near = delta <= tau * mhat
        last = tl.load(last_ptr + rows, mask=dead, other=0)
        abandoned = (step - last).to(tl.float32) >= window
        removed = dead & (near | abandoned)
        tl.store(out_ptr + rows, removed.to(tl.int8), mask=dead)
        # not near by these two: confirmed by an exact search (1 = kept, 2 = removed only as abandoned)
        tl.store(unsure_ptr + rows, tl.where(near, 0, tl.where(abandoned, 2, 1)).to(tl.int8), mask=dead)
        tl.atomic_add(cnt_ptr + 0, tl.sum(dead.to(tl.int64), axis=0), sem="relaxed")
        tl.atomic_add(cnt_ptr + 1, tl.sum((dead & ~removed).to(tl.int64), axis=0), sem="relaxed")
        tl.atomic_add(cnt_ptr + 2, tl.sum((dead & abandoned & ~near).to(tl.int64), axis=0), sem="relaxed")


@triton.jit
def _rule_verify_scan_kernel(C_ptr, list_ptr, nlist_ptr, vbest_ptr, k, KSLICE, d, sC,
                             BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
    # exact search: min over other centres of sum_d (c_j - c_i)^2 in FP32 (differences, no norms).
    # Grid: (programs looping over the candidates, one at a time) x (thin slices of the centres).
    n = tl.load(nlist_ptr)
    kbeg = tl.program_id(1) * KSLICE
    for j in range(tl.program_id(0), n, tl.num_programs(0)):
        row = tl.load(list_ptr + j).to(tl.int64)
        best = tl.full((BLOCK_K,), float("inf"), tl.float32)
        for k0 in range(kbeg, kbeg + KSLICE, BLOCK_K):
            cid = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            cok = cid < k
            acc = tl.zeros((BLOCK_K,), tl.float32)
            for d0 in range(0, d, BLOCK_D):
                col = d0 + tl.arange(0, BLOCK_D)
                cm = col < d
                q = tl.load(C_ptr + row * sC + col, mask=cm, other=0.0)
                c = tl.load(C_ptr + cid[:, None] * sC + col[None, :], mask=cok[:, None] & cm[None, :], other=0.0)
                diff = c - q[None, :]
                acc += tl.sum(diff * diff, axis=1)
            best = tl.minimum(best, tl.where(cok & (cid != row), acc, float("inf")))
        # non-negative FP32 values order like their bit patterns
        tl.atomic_min(vbest_ptr + j, tl.min(best, axis=0).to(tl.int32, bitcast=True), sem="relaxed")


@triton.jit
def _rule_verify_apply_kernel(list_ptr, nlist_ptr, vbest_ptr, unsure_ptr, out_ptr, cnt_ptr, mhat_ptr,
                              tau, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(nlist_ptr)
    if pid * BLOCK < n:
        live = i < n
        rows = tl.load(list_ptr + i, mask=live, other=0).to(tl.int64)
        delta = tl.load(vbest_ptr + i, mask=live, other=0x7F800000).to(tl.float32, bitcast=True)
        near = live & (delta.to(tl.float64) <= tau * tl.load(mhat_ptr))
        flag = tl.load(unsure_ptr + rows, mask=live, other=0)
        kept_now_removed = near & (flag == 1)
        tl.store(out_ptr + rows, tl.full((BLOCK,), 1, tl.int8), mask=kept_now_removed)
        tl.atomic_add(cnt_ptr + 1, -tl.sum(kept_now_removed.to(tl.int64), axis=0), sem="relaxed")
        tl.atomic_add(cnt_ptr + 2, -tl.sum((near & (flag == 2)).to(tl.int64), axis=0), sem="relaxed")


def distance_rule(table, table_norms, C, dead, last_hit, counters, mhat, step, window, tau, k, d):
    """Return a bool tensor of length k: the candidates removed by the distance test.

    table: [KP, DP] FP16 copy of the centroids (KP a multiple of BLOCK_K, DP of BLOCK_D, zero
    padding); table_norms: [KP] FP32 squared norms of the table rows, +inf for rows >= k;
    C: FP32 centroids, row stride C.stride(0), first d columns used; dead: bool [>= k];
    last_hit: int64 [>= k]; counters: int64 [>= 3]; mhat: FP64 0-dim tensor.
    """
    KP, DP = table.shape
    assert KP % BLOCK_K == 0 and DP % BLOCK_D == 0 and table.stride(1) == 1 and C.stride(1) == 1
    tiles = KP // BLOCK_K
    parts = MAX_PARTS
    while tiles % parts:
        parts //= 2
    flag = dead[:k].view(torch.uint8)
    cand = torch.sort(flag, descending=True, stable=True).indices     # candidates first, no host read
    ncand = flag.sum(dtype=torch.int64)
    ws_s = torch.empty((parts * 2, k), dtype=torch.float32, device=C.device)
    ws_l = torch.empty((parts * 2, k), dtype=torch.int64, device=C.device)
    out = torch.zeros(k, dtype=torch.int8, device=C.device)
    unsure = torch.zeros(k, dtype=torch.int8, device=C.device)
    grid_m = triton.cdiv(k, BLOCK_M)
    _rule_scan_kernel[(grid_m, parts)](
        table, table_norms, cand, ncand, ws_s, ws_l, KP // parts, DP, table.stride(0), k,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D, num_warps=4, num_stages=2)
    _rule_apply_kernel[(grid_m,)](
        C, cand, ncand, ws_s, ws_l, out, unsure, last_hit, counters, mhat, d, C.stride(0), int(step),
        float(window), float(tau), k, PARTS=parts, BLOCK_M=BLOCK_M,
        BLOCK_DE=min(32, triton.next_power_of_2(max(d, 1))), num_warps=4)
    # Removals above are exact (some centre is within the threshold); a candidate not removed by
    # distance may hide a nearer centre that the 16-bit ranking missed, so confirm it exactly.
    flag = unsure.ne(0).view(torch.uint8)
    pending = torch.sort(flag, descending=True, stable=True).indices
    npending = flag.sum(dtype=torch.int64)
    vbest = torch.full((k,), 0x7F800000, dtype=torch.int32, device=C.device)   # +inf
    vparts = max(1, min(VERIFY_PARTS, triton.cdiv(k, VERIFY_SLICE)))
    kslice = triton.cdiv(triton.cdiv(k, vparts), VERIFY_BLOCK_K) * VERIFY_BLOCK_K
    _rule_verify_scan_kernel[(VERIFY_PROGRAMS, vparts)](
        C, pending, npending, vbest, k, kslice, d, C.stride(0),
        BLOCK_K=VERIFY_BLOCK_K, BLOCK_D=VERIFY_BLOCK_D, num_warps=4)
    _rule_verify_apply_kernel[(triton.cdiv(k, 256),)](
        pending, npending, vbest, unsure, out, counters, mhat, float(tau), BLOCK=256, num_warps=4)
    return out.view(torch.bool)
