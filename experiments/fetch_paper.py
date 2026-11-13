"""Fetch one of the e-means paper's datasets with its own downloader, check it against the paper's
fingerprints, and save it in our layout.

python fetch_paper.py BENCH_DIR PAPER_NAME OUT_ROOT OUT_NAME
  BENCH_DIR: faiss/benchs/e_means (paper_data.py, paper_download.py, paper_splits/)
  env EMEANS_DATA / EMEANS_CACHE: where the paper's raw files and its cache go
"""
import hashlib
import json
import os
import sys

import numpy as np


def main():
    bench, name, out_root, out_name = sys.argv[1:5]
    sys.path.insert(0, bench)
    import paper_data as data
    import paper_download as dl
    todo = dl.needs(name)
    held = {}
    h = dl.heldout_needs(name)
    if h is not None:
        held[h[0]] = h[1]
    dl.download(data.ROOT, todo, 16)
    if held:
        dl.download_heldout(data.ROOT, held, 16)
    train, test = data.load(name)                      # raises on a fingerprint mismatch
    d = os.path.join(out_root, out_name)
    os.makedirs(d, exist_ok=True)
    n, dim = train.shape
    tr = np.lib.format.open_memmap(os.path.join(d, "train.npy"), mode="w+", dtype=np.float32, shape=(n, dim))
    for s in range(0, n, 1 << 20):
        tr[s:s + (1 << 20)] = data.rows(train, name, s, min(n, s + (1 << 20)))
    tr.flush()
    del tr
    np.save(os.path.join(d, "test.npy"), np.ascontiguousarray(test, dtype=np.float32))

    def sha(p):
        hh = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 22), b""):
                hh.update(c)
        return hh.hexdigest()
    meta = dict(name=out_name, source=f"e-means paper split '{name}' (benchs/e_means/paper_data.py, "
                "fingerprint-checked)", scaling="paper", n_train=int(n), n_test=int(len(test)), d=int(dim),
                train_sha256=sha(os.path.join(d, "train.npy")), test_sha256=sha(os.path.join(d, "test.npy")))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
