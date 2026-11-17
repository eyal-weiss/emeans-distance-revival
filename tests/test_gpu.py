"""GPU tests for the PyTorch e-means and o-means classes (skipped without CUDA, PyTorch and Triton;
the o-means tests also need o-means). Run: python -m pytest tests/test_gpu.py

They mirror tests/test_revival.py: with the library's own rule the classes reproduce the library
bit for bit, on plain blobs the distance test removes the same clusters as the library (identical
runs), and on data with far groups it keeps the far groups' centres that the size rule removes.
They also compare the classes with the first, plain-PyTorch versions (experiments/gpu/reference),
which compute the statistics in FP64 with ordinary PyTorch operations: the same decisions give the
same centres, so this checks the bookkeeping (distortion, last-hit steps, splits, cell rebuilds).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS, before CUDA starts

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "gpu", "reference"))
from faiss.contrib.e_means import Emeans  # noqa: E402
from distance_revival.torch_emeans import DistanceAwareRevivalTorchEmeans  # noqa: E402
import synthetic  # noqa: E402


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def cuda(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).cuda()


def blobs(seed=0, n=40_000, d=16, g=64):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, (g, d))
    return cuda(mu[rng.integers(0, g, n)] + 0.3 * rng.normal(size=(n, d)))


def far_groups(seed=0):
    """synth_far:50 (shuffled) and 105 starting centres: 100 random rows plus the 5 far-group centres."""
    X, _, info = synthetic.make("synth_far:50", seed)
    rng = np.random.default_rng(seed)
    init = np.vstack([X[rng.choice(len(X), 100, replace=False)], info["far_centres"]])
    x, init = cuda(X), cuda(init)
    assert covered(init, info) == 5
    return x, init, info


def covered(centroids, info):
    return synthetic.far_covered(centroids.cpu().numpy(), info["far_centres"], info["D"])


# ------------------------------------------------------------------------------- e-means (PyTorch)
def emeans(cls, x, k, T=2, seed=0, init=None, **kw):
    m = cls(x.shape[1], k, nepochs=T, seed=seed, capture_assignments=False, **kw)
    captures = [0]
    original = m._capture_step

    def counted(*a, **kw2):
        captures[0] += 1
        return original(*a, **kw2)
    m._capture_step = counted
    m.train(x, init)
    m.graph_captures = captures[0]
    return m


def test_emeans_count_is_library():
    x = blobs()
    a = emeans(Emeans, x, 256, unused_threshold=0.1)
    b = emeans(DistanceAwareRevivalTorchEmeans, x, 256, revival="count", record_stats=True)
    assert torch.equal(a.centroids, b.centroids)
    assert a.n_splits == b.n_splits > 0
    assert a.graph_captures == b.graph_captures == 1


def test_emeans_off_is_library_off():
    x = blobs()
    a = emeans(Emeans, x, 256, unused_threshold=0.0)
    b = emeans(DistanceAwareRevivalTorchEmeans, x, 256, revival="off")
    assert torch.equal(a.centroids, b.centroids)


def test_emeans_near_identical_on_plain_blobs():
    x = blobs(1)
    a = emeans(Emeans, x, 256, seed=1, unused_threshold=0.1)
    b = emeans(DistanceAwareRevivalTorchEmeans, x, 256, seed=1, revival="near")
    assert b.n_candidates > 0 and b.n_protected == 0
    assert b.graph_captures == 1                      # the statistics run inside the CUDA graph
    assert torch.equal(a.centroids, b.centroids)


def test_emeans_near_keeps_far_groups():
    x, init, info = far_groups()
    res = {rule: covered(emeans(DistanceAwareRevivalTorchEmeans, x, 105, T=5, init=init,
                                revival=rule).centroids, info) for rule in ("count", "near")}
    assert res["near"] == 5 and res["count"] < 5


def test_emeans_near_always_records_statistics():
    x, init, _ = far_groups()
    a = emeans(DistanceAwareRevivalTorchEmeans, x, 105, T=3, init=init, revival="near")
    b = emeans(DistanceAwareRevivalTorchEmeans, x, 105, T=3, init=init, revival="near",
               record_stats=False)
    assert torch.equal(a.centroids, b.centroids) and a.n_protected == b.n_protected > 0


def test_emeans_matches_reference():
    from torch_emeans_v1 import DistanceAwareRevivalTorchEmeans as Reference
    x, init, _ = far_groups(1)
    for T in (1, 3):
        a = emeans(DistanceAwareRevivalTorchEmeans, x, 105, T=T, init=init, revival="near")
        b = emeans(Reference, x, 105, T=T, init=init, revival="near")
        assert a.n_protected == b.n_protected > 0 and a.n_splits == b.n_splits
        assert torch.equal(a.centroids, b.centroids)


# ------------------------------------------------------------------------------- o-means
def omeans_classes():
    pytest.importorskip("faiss.contrib.o_means")
    from faiss.contrib.o_means import Omeans
    from distance_revival.omeans import OmeansDistanceRevival
    return Omeans, OmeansDistanceRevival


def omeans(cls, x, k, T=2, seed=0, init=None, **kw):
    m = cls(x.shape[1], k, nepochs=T, seed=seed, capture_assignments=False, **kw)
    return m, m.train(x, init)[0]


def test_omeans_count_is_library():
    Omeans, Ours = omeans_classes()
    x = blobs()
    _, a = omeans(Omeans, x, 256)
    m, b = omeans(Ours, x, 256, revival="count")
    assert torch.equal(a, b)
    assert m.revival_stats["splits"] > 0


def test_omeans_near_identical_on_plain_blobs():
    Omeans, Ours = omeans_classes()
    x = blobs(1)
    _, a = omeans(Omeans, x, 256, seed=1)
    m, b = omeans(Ours, x, 256, seed=1, revival="near")
    assert m.revival_stats["candidates"] > 0 and m.revival_stats["protected"] == 0
    assert torch.equal(a, b)


def test_omeans_near_keeps_far_groups():
    _, Ours = omeans_classes()
    x, init, info = far_groups()
    res = {rule: covered(omeans(Ours, x, 105, T=5, init=init, revival=rule)[1], info)
           for rule in ("count", "near")}
    assert res["near"] == 5 and res["count"] < 5


def test_omeans_matches_reference():
    _, Ours = omeans_classes()
    from omeans_v1 import OmeansDistanceRevival as Reference
    x, init, _ = far_groups(1)
    for T in (1, 3):                                   # T = 1 uses the recency-weighted closing mean
        m, a = omeans(Ours, x, 105, T=T, init=init, revival="near")
        r, b = omeans(Reference, x, 105, T=T, init=init, revival="near")
        assert m.revival_stats["protected"] == r.revival_stats["protected"] > 0
        assert m.revival_stats["splits"] == r.revival_stats["splits"]
        assert torch.equal(a, b)


def test_omeans_threshold_in_two_dimensions():
    """o-means turns revival off for d <= 2; this class uses 0.1 unless unused_threshold=None."""
    _, Ours = omeans_classes()
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40_000, 2)), [50.0, 50.0] + 0.1 * rng.normal(size=(30, 2))])
    x = cuda(X)
    init = torch.cat([x[torch.randperm(40_000, generator=torch.Generator().manual_seed(0))[:127].cuda()],
                      x[-1:]])
    m, c = omeans(Ours, x, 128, T=3, init=init, revival="near")
    assert m.revival_stats["candidates"] > 0 and m.revival_stats["protected"] > 0   # checks ran
    assert ((c - torch.tensor([50.0, 50.0], device="cuda")) ** 2).sum(1).min() < 1.0
    m, _ = omeans(Ours, x, 128, T=3, init=init, revival="near", unused_threshold=None)
    assert m.revival_stats["candidates"] == 0 and m.recipe["unused_threshold"] == 0.0   # no checks


def test_omeans_abandonment_off():
    _, Ours = omeans_classes()
    x, init, info = far_groups()
    m, c = omeans(Ours, x, 105, T=3, init=init, revival="near", abandon_passes=None)
    assert m.revival_stats["abandoned"] == 0 and covered(c, info) == 5


def test_omeans_distortion_estimate_on_tight_high_dimensional_data():
    """The fixed point of the batch distortion must not round small distances away."""
    _, Ours = omeans_classes()
    rng = np.random.default_rng(0)
    d, g = 1024, 64
    mu = rng.normal(size=(g, d))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    x = cuda(mu[rng.integers(0, g, 60_000)] + 0.002 * rng.normal(size=(60_000, d)))
    m, c = omeans(Ours, x, 128, T=2, revival="near")
    nearest = torch.cdist(x, c).min(1).values
    train_mse = float((nearest ** 2).mean())
    assert 0.5 < m.revival_stats["mse_hat"] / train_mse < 2.0


# ------------------------------------------------------------------------------- direct state tests
def fp16_table(C):
    """The e-means class's scaled FP16 table for distance_rule."""
    from distance_revival.kernels import BLOCK_D, BLOCK_K
    k, d = C.shape
    kp, dp = -(-k // BLOCK_K) * BLOCK_K, -(-d // BLOCK_D) * BLOCK_D
    table = torch.zeros((kp, dp), dtype=torch.float16, device="cuda")
    scale = 2.0 ** (6 - np.floor(np.log2(float(C.abs().max()))))
    table[:k, :d] = (C * scale).half()
    norms = torch.full((kp,), float("inf"), device="cuda")
    norms[:k] = table[:k].float().square().sum(1)
    return table, norms


def exact_rule(C, dead, last_hit, step, window, thr):
    """FP64 oracle: nearest other centre by exact differences."""
    c = C.double().cpu().numpy()
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    near = d2.min(1) <= thr
    abandoned = (step - last_hit) >= window
    removed = dead & (near | abandoned)
    return removed, [int(dead.sum()), int((dead & ~removed).sum()), int((dead & abandoned & ~near).sum())]


def run_rule(C, dead, last_hit, step, window, thr, tau=4.0):
    from distance_revival.kernels import distance_rule
    table, norms = fp16_table(C)
    counters = torch.zeros(3, dtype=torch.int64, device="cuda")
    mhat = torch.tensor(thr / tau, dtype=torch.float64, device="cuda")
    out = distance_rule(table, norms, C, torch.from_numpy(dead).cuda(), torch.from_numpy(last_hit).cuda(),
                        counters, mhat, step, window, tau, C.shape[0], C.shape[1])
    return out.cpu().numpy(), counters.tolist()


@pytest.mark.parametrize("k,d", [(5, 1), (300, 2), (1000, 40), (300, 40)])
def test_distance_rule_matches_exact_oracle(k, d):
    rng = np.random.default_rng(k + d)
    C = cuda(rng.normal(size=(k, d)) * rng.uniform(0.1, 10, size=(k, 1)))
    dead = rng.random(k) < 0.3
    last_hit = rng.integers(0, 40, k).astype(np.int64)
    c = C.double().cpu().numpy()
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    delta = np.sort(d2.min(1))
    gap = np.argmax(np.diff(delta[k // 4: 3 * k // 4])) + k // 4       # threshold far from every delta
    thr = 0.5 * (delta[gap] + delta[gap + 1])
    got, counters = run_rule(C, dead, last_hit, 40, 30.0, thr)
    want, want_counters = exact_rule(C, dead, last_hit, 40, 30.0, thr)
    assert np.array_equal(got, want) and counters == want_counters


def test_distance_rule_exact_check_overrides_fp16_ties():
    """Four centres that round to the same FP16 value: the 16-bit ranking cannot tell them apart."""
    C = cuda([[64.02], [64.025], [64.0], [64.001]])
    dead = np.array([False, False, True, False])
    last_hit = np.zeros(4, dtype=np.int64)
    got, counters = run_rule(C, dead, last_hit, 1, 1e9, 1e-5)
    assert got.tolist() == [False, False, True, False] and counters == [1, 0, 0]
    got, counters = run_rule(C, dead, last_hit, 1, 1e9, 1e-7)          # now nothing is that close
    assert got.tolist() == [False] * 4 and counters == [1, 1, 0]
    # abandoned as well: removed either way, but not counted as removed *only* by abandonment
    got, counters = run_rule(C, dead, last_hit, 100, 10.0, 1e-5)
    assert got.tolist() == [False, False, True, False] and counters == [1, 0, 0]


def test_distance_rule_no_candidates():
    C = cuda(np.random.default_rng(0).normal(size=(300, 8)))
    got, counters = run_rule(C, np.zeros(300, dtype=bool), np.zeros(300, dtype=np.int64), 5, 3.0, 1.0)
    assert not got.any() and counters == [0, 0, 0]


def test_emeans_graph_replay_matches_eager():
    x, init, _ = far_groups(1)
    a = emeans(DistanceAwareRevivalTorchEmeans, x, 105, T=3, init=init, revival="near")
    b = DistanceAwareRevivalTorchEmeans(x.shape[1], 105, nepochs=3, seed=0, capture_assignments=False,
                                        revival="near")
    b._replay_eligible = lambda x: False                       # every batch on the eager path
    b.train(x, init)
    assert a.graph_captures == 1
    assert (a.n_candidates, a.n_protected, a.n_splits) == (b.n_candidates, b.n_protected, b.n_splits)
    assert torch.allclose(a.centroids, b.centroids, rtol=1e-5, atol=1e-5)


def test_omeans_batch_statistics_and_splits():
    """accumulate_dist, ema_update_dist and apply_splits_dist against plain PyTorch."""
    omeans_classes()
    from distance_revival.omeans import accumulate_dist, apply_splits_dist, ema_update_dist
    rng = np.random.default_rng(0)
    k, d, dp, kp, R = 37, 20, 32, 256, 1000
    x16 = torch.zeros((R, dp), dtype=torch.float16, device="cuda")
    x16[:, :d] = cuda(rng.normal(size=(R, d))).half()
    C = torch.zeros((kp, dp), device="cuda")
    C[:k, :d] = cuda(rng.normal(size=(k, d)))
    c16 = C.half()
    cn = torch.full((kp,), float("inf"), device="cuda")
    cn[:k] = c16[:k].float().square().sum(1)
    labels = torch.from_numpy(rng.integers(0, k, R).astype(np.int32)).cuda()
    sums = torch.zeros((kp, dp), dtype=torch.int64, device="cuda")
    cnt = torch.zeros(kp, dtype=torch.int32, device="cuda")
    rowd = torch.empty(R, dtype=torch.float64, device="cuda")
    bd = torch.zeros(kp, dtype=torch.float64, device="cuda")
    last_hit = torch.full((kp,), -1, dtype=torch.int64, device="cuda")
    C0 = C.clone()
    accumulate_dist(x16, labels, sums, cnt, d, C, rowd, bd, last_hit, 7)
    lab = labels.long()
    want = torch.zeros(kp, dtype=torch.float64, device="cuda").index_add_(
        0, lab, (x16[:, :d].float() - C0[lab, :d]).square().sum(1).double())
    assert torch.allclose(bd, want, rtol=1e-5)
    hit = torch.bincount(lab, minlength=kp) > 0
    assert torch.equal(last_hit[hit], torch.full_like(last_hit[hit], 7)) and (last_hit[~hit] == -1).all()
    assert torch.equal(cnt.long(), torch.bincount(lab, minlength=kp))
    usage = torch.ones(kp, device="cuda")
    D = torch.full((kp,), 2.0, dtype=torch.float64, device="cuda")
    ema_update_dist(C, c16, cn, usage, sums, cnt, k, d, 0.25, D, bd)
    assert torch.allclose(D[:k], 0.75 * 2.0 + 0.25 * want[:k]) and (bd == 0).all()
    # a second, shorter batch read through an index, into the same (larger) buffer
    idx = torch.from_numpy(rng.choice(R, 300, replace=False)).cuda()
    labels2 = torch.from_numpy(rng.integers(0, k - 1, 300).astype(np.int32)).cuda()   # cluster k-1 untouched
    C1 = C.clone()
    accumulate_dist(x16, labels2, sums, cnt, d, C, rowd, bd, last_hit, 8, idx=idx)
    lab2 = labels2.long()
    want2 = torch.zeros(kp, dtype=torch.float64, device="cuda").index_add_(
        0, lab2, (x16[idx, :d].float() - C1[lab2, :d]).square().sum(1).double())
    assert torch.allclose(bd, want2, rtol=1e-5)
    # alpha = 1 replaces D for clusters that received rows and keeps it for the others
    D1 = D.clone()
    ema_update_dist(C, c16, cn, usage, sums, cnt, k, d, 1.0, D, bd)
    touched = torch.bincount(lab2, minlength=kp) > 0
    assert not touched[k - 1] and float(D[k - 1]) == float(D1[k - 1])
    assert torch.allclose(D[:k][touched[:k]], want2[:k][touched[:k]])
    D0, src, tgt = D.clone(), torch.arange(k, device="cuda"), torch.arange(k, device="cuda").flip(0)
    m = torch.tensor(3, device="cuda")
    apply_splits_dist(C, c16, cn, usage, src, tgt, m, d, 1e-4, k // 2 + 1, D, last_hit, 11)
    for i in range(3):
        s, t = int(src[i]), int(tgt[i])
        assert float(D[s]) == float(D[t]) == float(D0[s]) / 2 and int(last_hit[t]) == 11
    assert torch.equal(D[3:k - 3], D0[3:k - 3])
