"""Validation runs of the GPU classes (PyTorch e-means and o-means): python experiments/gpu/validate.py
--jobs experiments/gpu/jobs_main.json --out runs/gpu.jsonl  (GPU_DETERMINISTIC=1 for bitwise runs).

Job fields: impl (emeans|omeans), ds, k, T, seed, arm (count|count_h|near|off), tau (size threshold).
  count   : the library as shipped (Emeans / Omeans), revival at tau
  count_h : our class with the library's rule and the statistics on (must equal count bit for bit)
  near    : our class, the paper's rule
  off     : no revival
Output: one JSON line per run with the exact test MSE (FP32 ranking, FP64 distance), wall time,
splits, protections and, for e-means, how many CUDA graphs were captured (graph replay in use).
"""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from faiss.contrib.e_means import Emeans
from faiss.contrib.o_means import Omeans
import faiss.contrib.o_means.schedule as om_schedule
REFERENCE = os.environ.get("GPU_REFERENCE") == "1"
if REFERENCE:   # the first, plain-PyTorch versions (results/gpu/runs_reference)
    sys.path.insert(0, os.path.join(HERE, "reference"))
    from torch_emeans_v1 import DistanceAwareRevivalTorchEmeans
    from omeans_v1 import OmeansDistanceRevival
else:
    from distance_revival.torch_emeans import DistanceAwareRevivalTorchEmeans
    from distance_revival.omeans import OmeansDistanceRevival

# prepared datasets (experiments/download.sh): $DISTANCE_REVIVAL_DATA/<name>/{train,test}.npy
DATA = os.environ.get("DISTANCE_REVIVAL_DATA", os.path.join(HERE, "..", "..", "data"))
_resolve = om_schedule._resolve_unused_threshold


def test_mse(test, cent):
    tot = 0.0
    cn = (cent * cent).sum(1)
    for a in range(0, len(test), 4096):
        q = test[a:a + 4096]
        j = torch.addmm(cn[None, :], q, cent.t(), alpha=-2.0).argmin(1)
        tot += float((q.double() - cent[j].double()).square().sum())
    return tot / len(test)


def run_emeans(x, k, T, seed, arm, tau):
    d = x.shape[1]
    kw = dict(nepochs=T, seed=seed, capture_assignments=False)
    if arm == "count":
        model = Emeans(d, k, unused_threshold=tau, **kw)
    elif arm == "off":
        model = Emeans(d, k, unused_threshold=0.0, **kw)
    elif arm == "count_h":
        model = DistanceAwareRevivalTorchEmeans(d, k, revival="count", record_stats=True,
                                                unused_threshold=tau, **kw)
    else:
        model = DistanceAwareRevivalTorchEmeans(d, k, revival="near", unused_threshold=tau, **kw)
    captures = [0]
    original = model._capture_step

    def counted(*a, **kw2):
        captures[0] += 1
        return original(*a, **kw2)
    model._capture_step = counted
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cent, _ = model.train(x)
    torch.cuda.synchronize(); el = time.perf_counter() - t0
    extra = {}
    if isinstance(model, DistanceAwareRevivalTorchEmeans):
        extra = dict(candidates=model.n_candidates, protected=model.n_protected,
                     abandoned=model.n_abandoned)
    return cent, el, dict(splits=model.n_splits, graph_captures=captures[0],
                          backend=model.recipe.get("assignment_backend"),
                          batch=model.recipe["batch_size"], **extra)


def run_omeans(x, k, T, seed, arm, tau):
    d = x.shape[1]
    om_schedule._resolve_unused_threshold = (lambda t, dd: tau)   # stock Omeans has no option for it
    try:
        if arm == "count":
            model = Omeans(d, k, nepochs=T, seed=seed, capture_assignments=False)
        else:
            rule = {"count_h": "count", "near": "near", "off": "off"}[arm]
            model = OmeansDistanceRevival(d, k, nepochs=T, revival=rule, seed=seed,
                                          capture_assignments=False,
                                          **({} if REFERENCE else dict(unused_threshold=tau)))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        cent, _ = model.train(x)
        torch.cuda.synchronize(); el = time.perf_counter() - t0
    finally:
        om_schedule._resolve_unused_threshold = _resolve
    extra = dict(model.revival_stats) if getattr(model, "revival_stats", None) else {}
    return cent, el, dict(nlist=model.recipe["nlist"], nprobe=model.recipe["nprobe"],
                          batch=model.recipe["batch_size"], **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.environ.get("GPU_DETERMINISTIC") == "1":
        torch.use_deterministic_algorithms(True)
    jobs = json.load(open(a.jobs))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["key"] for l in open(a.out)}
    cache = {}
    with open(a.out, "a") as f:
        for j in jobs:
            key = "|".join(str(j[c]) for c in ("impl", "ds", "k", "T", "seed", "arm", "tau"))
            if key in done:
                continue
            if j["ds"] not in cache:
                cache.clear(); torch.cuda.empty_cache()
                load = lambda p: torch.from_numpy(np.array(np.load(p, mmap_mode="r"), dtype=np.float32)).cuda()
                cache[j["ds"]] = (load(os.path.join(DATA, j["ds"], "train.npy")),
                                  load(os.path.join(DATA, j["ds"], "test.npy")))
            x, test = cache[j["ds"]]
            try:
                fn = run_emeans if j["impl"] == "emeans" else run_omeans
                cent, el, info = fn(x, j["k"], j["T"], j["seed"], j["arm"], j["tau"])
                r = dict(j, key=key, test_mse=test_mse(test, cent), time_s=el,
                         cent_sha=str(hash(cent.cpu().numpy().tobytes())), **info)
            except Exception as e:
                import traceback
                r = dict(j, key=key, error=repr(e)[:300], tb=traceback.format_exc()[-1500:])
            f.write(json.dumps(r) + "\n"); f.flush()
            print(key, r.get("test_mse"), r.get("time_s"), r.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
