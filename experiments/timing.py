"""E6: running-time overhead of the distance test. Run with no other jobs of ours on the host.

python timing.py --out timing.jsonl --threads 16 --repeats 6
Arms: library (faiss.contrib.e_means.NumpyEmeans, unused_threshold=0.1), near, exact (our subclass,
no ratio recording). Same data, same random initialization, interleaved repeats with the order of
arms rotated.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(a.threads)
    import numpy as np
    import faiss
    faiss.omp_set_num_threads(a.threads)
    from faiss.contrib.e_means import NumpyEmeans
    from distance_revival import DistanceAwareRevivalEmeans
    import datasets
    configs = [("deep1m", 16384, 2), ("kdd_z", 4096, 2)]
    with open(a.out, "a") as f:
        for ds, k, T in configs:
            X, _, _ = datasets.load(ds)
            X = np.ascontiguousarray(X)
            init = X[np.sort(np.random.RandomState(0).permutation(len(X))[:k])]
            arms = ("library", "near", "exact")
            for rep in range(a.repeats):
                order = arms[rep % 3:] + arms[:rep % 3]          # rotate the order across repeats
                for arm in order:
                    if arm == "library":
                        m = NumpyEmeans(X.shape[1], k, nepochs=T, seed=0, unused_threshold=0.1)
                    else:
                        m = DistanceAwareRevivalEmeans(X.shape[1], k, nepochs=T, seed=0, revival=arm)
                    t0 = time.perf_counter()
                    m.train(X, init_centroids=init)
                    secs = time.perf_counter() - t0
                    rec = dict(dataset=ds, k=k, T=T, arm=arm, rep=rep, position=order.index(arm), seconds=secs, threads=a.threads, impl="exact_v3",
                               n_splits=int(m.n_splits), host=os.uname().nodename)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
