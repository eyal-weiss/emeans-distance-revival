"""Experiment grid for the paper (protocol frozen 2026-11-11; see ../../protocol.md).

  python grid.py count [--host H]                       job counts per experiment
  python grid.py run --host H --workers 8 --threads 4 --out RUNDIR [--only E1,E2,...]

Every dataset is assigned to one host, so all arms of a (dataset, k, seed, init) group run on the
same machine from the same initial centres. k-means++ centres are computed first and cached.
Results: RUNDIR/results.jsonl (one record per job, resumable), errors.jsonl, progress.json, DONE.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

TAB = ["covtype_z", "covtype_mm", "kdd_z", "census_z", "msd_z", "geonames"]
DEV = {"covtype_z", "covtype_mm"}                       # used to design the rule
EMB = ["deep1m", "sift1m", "spacev1m", "deep10m"]
HOST = {"kdd_z": "pikachu", "msd_z": "pikachu", "deep1m": "pikachu",
        "census_z": "eevee", "covtype_z": "eevee", "covtype_mm": "eevee", "geonames": "eevee",
        "sift1m": "eevee",
        # Amendment 1 (2026-11-11 20:55)
        "power_z": "pikachu", "shuttle_z": "eevee", "spacev1m": "eevee", "deep10m": "pikachu"}
# rough sizes for scheduling (n_train, d); synthetic: 200k x 32 or 200k x d
SIZE = {"covtype_z": (561e3, 54), "covtype_mm": (561e3, 54), "kdd_z": (1.05e6, 122),
        "census_z": (2.0e6, 68), "msd_z": (495e3, 90), "geonames": (1e6, 2),
        "deep1m": (1e6, 96), "sift1m": (1e6, 128),
        "power_z": (2.0e6, 7), "shuttle_z": (5.8e4, 9), "spacev1m": (1e6, 100), "deep10m": (1e7, 96)}

ARMS = {
    "count": dict(revival="count"),
    "off": dict(revival="off"),
    "near": dict(revival="near"),
    "near_noab": dict(revival="near", abandon_passes=None),
    "exact": dict(revival="exact"),
}
for _t in (1, 2, 3, 6, 8, 16, 32):
    ARMS[f"near_tau{_t}"] = dict(revival="near", tau_near=float(_t))
for _w in (1, 3):
    ARMS[f"near_w{_w}"] = dict(revival="near", abandon_passes=float(_w))
MAIN_ARMS = ["count", "off", "near", "near_noab", "exact"]
ABL_ARMS = [f"near_tau{t}" for t in (1, 2, 3, 6, 8, 16, 32)] + ["near_w1", "near_w3"]


def host_of(ds):
    return "pikachu" if ds.startswith("synth_") else HOST[ds]


def _job(exp, kind, ds, k, seed, init, T=None, arm=None):
    key = f"{kind}|{arm}|{ds}|k{k}|s{seed}|{init}|T{T}"
    return dict(key=key, exp=exp, kind=kind, dataset=ds, k=k, seed=seed, init=init, T=T, arm=arm)


def all_jobs():
    J = []
    # E1: non-embedding suite (held-out + development), random and k-means++ initialization
    for ds in TAB:
        for k in (256, 1024, 4096):
            for seed in range(10):
                for init in ("random", "kmpp"):
                    J.append(_job("E1", "lloyd", ds, k, seed, init))
                    for T in (1, 2, 3, 5, 10):
                        for arm in MAIN_ARMS:
                            J.append(_job("E1", "emeans", ds, k, seed, init, T, arm))
    # E2: embeddings at the paper's 1M / k=16384 configuration (the original two datasets)
    for ds in ("deep1m", "sift1m"):
        for seed in range(10):
            if seed < 3:
                J.append(_job("E2", "lloyd", ds, 16384, seed, "random"))
            for T in (1, 2, 3, 5):
                for arm in ("count", "off", "near"):
                    J.append(_job("E2", "emeans", ds, 16384, seed, "random", T, arm))
                if seed < 5 and T in (2, 5):
                    J.append(_job("E2", "emeans", ds, 16384, seed, "random", T, "exact"))
    # E3: ablations / sensitivity (random initialization, 5 seeds)
    for ds in ("covtype_z", "kdd_z", "geonames"):
        for seed in range(5):
            for T in (2, 5, 10):
                for arm in ABL_ARMS:
                    J.append(_job("E3", "emeans", ds, 1024, seed, "random", T, arm))
    for seed in range(5):
        for T in (2, 5):
            for arm in ("near_tau1", "near_tau2", "near_tau8", "near_w1"):
                J.append(_job("E3", "emeans", "deep1m", 16384, seed, "random", T, arm))
    # S: synthetic (development far groups; heterogeneous-scale probe)
    synth = [(f"synth_far:{m}", 105) for m in (50, 100, 150, 250, 400)] + \
            [(f"synth_hetero:{d}", 100) for d in (2, 16)]
    for ds, k in synth:
        for seed in range(10):
            for init in ("random", "kmpp"):
                J.append(_job("S", "lloyd", ds, k, seed, init))
                for arm in MAIN_ARMS:
                    J.append(_job("S", "emeans", ds, k, seed, init, 5, arm))
    # ---------------- Amendment 1 (2026-11-11 20:55 EST): appended, original keys unchanged
    # A1: seeds 10-29 for the headline configuration (T=2, random initialization)
    for ds in TAB:
        for k in (256, 1024, 4096):
            for seed in range(10, 30):
                for arm in ("count", "off", "near", "exact"):
                    J.append(_job("A1", "emeans", ds, k, seed, "random", 2, arm))
    # A2b: seeds 5-9 for every ablation arm
    for ds in ("covtype_z", "kdd_z", "geonames"):
        for seed in range(5, 10):
            for T in (2, 5, 10):
                for arm in ABL_ARMS:
                    J.append(_job("A2b", "emeans", ds, 1024, seed, "random", T, arm))
    for seed in range(5, 10):
        for T in (2, 5):
            for arm in ("near_tau1", "near_tau2", "near_tau8", "near_w1"):
                J.append(_job("A2b", "emeans", "deep1m", 16384, seed, "random", T, arm))
    # A3: SPACEV 1M (paper suite) and a DEEP 10M bit-identity check (run with 16 threads per job)
    for seed in range(10):
        if seed < 3:
            J.append(_job("A3", "lloyd", "spacev1m", 16384, seed, "random"))
        for T in (2, 5):
            for arm in ("count", "off", "near"):
                J.append(_job("A3", "emeans", "spacev1m", 16384, seed, "random", T, arm))
    for seed in range(3):
        for arm in ("count", "near"):
            J.append(_job("A3big", "emeans", "deep10m", 65536, seed, "random", 2, arm))
    # A4: two more held-out tabular datasets
    for ds in ("power_z", "shuttle_z"):
        for k in (256, 1024):
            for seed in range(10):
                J.append(_job("A4", "lloyd", ds, k, seed, "random"))
                for T in (1, 2, 5, 10):
                    for arm in MAIN_ARMS:
                        J.append(_job("A4", "emeans", ds, k, seed, "random", T, arm))
    return J


def cost(job):
    if job["dataset"].startswith("synth_"):
        n, d = 2e5, 32
    else:
        n, d = SIZE[job["dataset"]]
    passes = 25 if job["kind"] == "lloyd" else job["T"] * (1.4 if job["arm"] == "exact" else 1.0)
    return passes * n * max(d, 16) * job["k"]


# ------------------------------------------------------------------------------ worker side
_THREADS = None


def _init(threads):
    global _THREADS
    _THREADS = threads
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)
    import faiss
    faiss.omp_set_num_threads(threads)


def _kmpp_path(out, ds, k, seed):
    return os.path.join(out, "kmpp", f"{ds.replace(':', '_')}_k{k}_s{seed}.npy")


def kmpp_job(args):
    out, ds, k, seed = args
    import numpy as np
    import datasets
    path = _kmpp_path(out, ds, k, seed)
    if os.path.exists(path):
        return path, 0.0
    X, _, _ = datasets.load(ds, seed)
    from sklearn.cluster import kmeans_plusplus
    t0 = time.perf_counter()
    C = kmeans_plusplus(np.asarray(X), k, random_state=seed)[0].astype(np.float32)
    secs = time.perf_counter() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path + ".tmp.npy", C)
    os.replace(path + ".tmp.npy", path)
    return path, secs


def _score(C, X, Xte, info, ds, k):
    import numpy as np
    import faiss

    def errors(x, chunk=65536):
        c = np.ascontiguousarray(C, dtype=np.float32)
        err = np.empty(len(x), dtype=np.float64)
        lab = np.empty(len(x), dtype=np.int64)
        for s in range(0, len(x), chunk):
            xb = np.ascontiguousarray(x[s:s + chunk], dtype=np.float32)
            _, l = faiss.knn(xb, c, 1)
            l = l.ravel()
            diff = xb.astype(np.float64) - c[l].astype(np.float64)
            err[s:s + len(xb)] = (diff * diff).sum(1)
            lab[s:s + len(xb)] = l
        return err, lab

    te_err, _ = errors(Xte)
    if ds in EMB:      # paper convention: train error on a fixed 200k subset
        idx = np.sort(np.random.RandomState(4321).choice(len(X), 200_000, replace=False))
        Xtr = X[idx]
    else:
        Xtr = X
    tr_err, tr_lab = errors(Xtr)
    cnt = np.bincount(tr_lab, minlength=k).astype(np.float64)
    ntail = max(1, len(tr_err) // 1000)
    tail = np.partition(tr_err, len(tr_err) - ntail)[-ntail:]
    rec = dict(test_mse=float(te_err.mean()), train_mse=float(tr_err.mean()),
               imbalance=float(k * (cnt ** 2).sum() / cnt.sum() ** 2),
               empty=int((cnt == 0).sum()), tail_mse=float(tail.mean()), max_err=float(tr_err.max()))
    if "far_centres" in info:
        import synthetic
        rec["far_covered"] = synthetic.far_covered(C, info["far_centres"], info["D"])
    return rec


def run_job(args):
    out, job = args
    import numpy as np
    import faiss
    import datasets
    t_start = time.time()
    try:
        ds, k, seed = job["dataset"], job["k"], job["seed"]
        X, Xte, info = datasets.load(ds, seed)
        n, d = X.shape
        if job["init"] == "random":
            rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])
            init = np.ascontiguousarray(X[rows], dtype=np.float32)
        else:
            init = np.load(_kmpp_path(out, ds, k, seed))
        rec = dict(job)
        if job["arm"] == "exact":
            rec["impl"] = "exact_v3"      # Amendments 2-3: library assignment kept, unknown utility protected,
                                          # numerator and denominator restart together at birth
        if job["kind"] == "lloyd":
            km = faiss.Kmeans(d, k, niter=25, seed=seed, max_points_per_centroid=10**9,
                              min_points_per_centroid=1, verbose=False)
            t0 = time.perf_counter()
            km.train(np.ascontiguousarray(X, dtype=np.float32), init_centroids=init)
            rec["seconds"] = time.perf_counter() - t0
            C = km.centroids
        else:
            from distance_revival import DistanceAwareRevivalEmeans
            m = DistanceAwareRevivalEmeans(d, k, nepochs=job["T"], seed=seed, record_ratios=True,
                                    **ARMS[job["arm"]])
            t0 = time.perf_counter()
            m.train(X, init_centroids=init)
            rec["seconds"] = time.perf_counter() - t0
            C = m.centroids
            r = m.recipe
            rec.update(n_splits=int(m.n_splits), n_protected=int(m.n_protected),
                       n_candidates=int(m.n_candidates), n_abandoned=int(m.n_abandoned),
                       batch=int(r["batch_size"]), alpha0=float(r["alpha0"]),
                       total_steps=int(r["total_steps"]), steps_per_pass=int(r["steps_per_pass"]),
                       batch_law=str(r["batch_law"]), unused_threshold=float(m._unused_threshold))
            if m.ratio_hist.any():
                rec["ratio_hist"] = m.ratio_hist.tolist()
        rec.update(_score(C, X, Xte, info, ds, k))
        rec.update(n=int(n), d=int(d), threads=_THREADS, wall=time.time() - t_start,
                   faiss=faiss.__version__, host=os.uname().nodename)
        return rec, None
    except Exception:
        return None, dict(job, error=traceback.format_exc())


# ------------------------------------------------------------------------------ driver
def run(a):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    jobs = [j for j in all_jobs() if host_of(j["dataset"]) == a.host and (only is None or j["exp"] in only)]
    if a.arm:
        jobs = [j for j in jobs if j["arm"] == a.arm]
    if a.pilot:   # one cheap job per (experiment, dataset, kind, arm), seed 0, plus one k-means++ initialization
        kmin = {}
        for j in jobs:
            kmin[j["dataset"]] = min(kmin.get(j["dataset"], 10**9), j["k"])
        pick, seen = [], set()
        for j in jobs:
            tag = (j["exp"], j["dataset"], j["kind"], j["arm"], j["init"])
            if (j["seed"] == 0 and j["k"] == kmin[j["dataset"]] and tag not in seen
                    and (j["T"] is None or j["T"] == min(t for t in (1, 2, 5) if any(
                        x["T"] == t for x in jobs if x["dataset"] == j["dataset"])))
                    and (j["init"] == "random" or j["arm"] in (None, "near"))):
                seen.add(tag)
                pick.append(j)
        jobs = pick
    res_path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    todo = sorted([j for j in jobs if j["key"] not in done], key=cost, reverse=True)
    started = time.time()

    def progress(n_done, n_err, phase):
        el = time.time() - started
        json.dump(dict(host=a.host, phase=phase, total=len(jobs), done=len(done) + n_done, errors=n_err,
                       remaining=len(todo) - n_done, elapsed_min=round(el / 60, 1),
                       updated=time.strftime("%Y-%m-%d %H:%M:%S")),
                  open(os.path.join(a.out, "progress.json"), "w"))

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx, initializer=_init,
                             initargs=(a.threads,)) as pool:
        # k-means++ centres are queued first (largest k first); runs that need them are queued as
        # soon as their centres exist; everything else is queued right away (largest first).
        from concurrent.futures import wait, FIRST_COMPLETED
        groups = {}
        for j in todo:
            if j["init"] == "kmpp":
                groups.setdefault((j["dataset"], j["k"], j["seed"]), []).append(j)
        need = sorted([t for t in groups if not os.path.exists(_kmpp_path(a.out, *t))], key=lambda t: -t[1])
        kfuts = {pool.submit(kmpp_job, (a.out, *t)): t for t in need}
        ready = [j for j in todo if j["init"] != "kmpp" or (j["dataset"], j["k"], j["seed"]) not in need]
        rfuts = {pool.submit(run_job, (a.out, j)) for j in sorted(ready, key=cost, reverse=True)}
        n_done = n_err = kdone = 0
        with open(res_path, "a") as rf, open(os.path.join(a.out, "errors.jsonl"), "a") as ef, \
                open(os.path.join(a.out, "kmpp_times.jsonl"), "a") as kt:
            pending = set(kfuts) | rfuts
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    if fut in kfuts:
                        t = kfuts[fut]
                        try:
                            path, secs = fut.result()
                            kt.write(json.dumps(dict(path=os.path.basename(path), seconds=secs)) + "\n")
                            kt.flush()
                            kdone += 1
                            pending |= {pool.submit(run_job, (a.out, j))
                                        for j in sorted(groups[t], key=cost, reverse=True)}
                        except Exception:
                            for j in groups[t]:
                                ef.write(json.dumps(dict(j, error="kmpp failed: " + traceback.format_exc())) + "\n")
                                n_err += 1
                            ef.flush()
                        continue
                    rec, err = fut.result()
                    if rec is not None:
                        rf.write(json.dumps(rec) + "\n")
                        rf.flush()
                        n_done += 1
                    else:
                        ef.write(json.dumps(err) + "\n")
                        ef.flush()
                        n_err += 1
                progress(n_done, n_err, f"runs (kmpp {kdone}/{len(need)})")
        progress(n_done, n_err, "finished")
    open(os.path.join(a.out, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "run"])
    ap.add_argument("--host")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--only")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--arm")
    a = ap.parse_args()
    if a.cmd == "count":
        from collections import Counter
        J = [j for j in all_jobs() if a.host is None or host_of(j["dataset"]) == a.host]
        c = Counter((host_of(j["dataset"]), j["exp"]) for j in J)
        tot = Counter()
        for j in J:
            tot[host_of(j["dataset"])] += cost(j)
        for key in sorted(c):
            print(key, c[key])
        for h, v in tot.items():
            print(h, f"~{2 * v / 2e11 / 3600:.1f} lane-hours at ~200 GFLOP/s per job")
        return
    run(a)


if __name__ == "__main__":
    main()
