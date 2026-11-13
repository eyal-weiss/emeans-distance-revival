# Distance-aware revival for e-means

Code for *Small Is Not Dead: Distance-Aware Cluster Revival for e-means* (Eyal Weiss, 2026; arXiv link to come).

[e-means](https://arxiv.org/abs/2611.09501) (FAISS 1.16, `faiss.contrib.e_means`) periodically relocates the centres of small clusters (dead-cluster revival): every 5 batches, every cluster whose running count is below a tenth of the average count is removed, and its centre is reused to split a large cluster. The rule considers only size, so on data with small, isolated groups it repeatedly removes centres whose removal is costly.

This package adds a second condition. A cluster *j* selected by the size test (a *candidate*) is removed only if

    delta_j <= tau_near * MSE_hat      or      j received no point for W epochs,

where `delta_j` is the squared distance from its centre to the nearest other centre, and `MSE_hat` is a running estimate of the mean squared error (tau_near = 4, W = 2).
- When a centre is the mean of its points, `delta_j` is an upper bound on the average increase in squared error, per point of the cluster, caused by removing it while the other centres are kept fixed (Fritzke's utility divided by the cluster size).
- The second clause (the abandonment clause) is a safeguard for a centre that has lost all its points far from every other centre; in the paper's experiments it rarely acts and has no measurable effect.
- On the e-means paper's embedding benchmarks, the distance test never prevents a removal, and runs are bit-identical to the library.

## Usage

```python
import numpy as np
from distance_revival import DistanceAwareRevivalEmeans

x = np.random.rand(100_000, 32).astype("float32")
km = DistanceAwareRevivalEmeans(x.shape[1], 1024, revival="near", nepochs=2)   # tau_near=4, abandon_passes=2
km.train(x)
centroids = km.centroids
print(km.n_splits, km.n_protected)
```

`revival` can be:
- `"near"`: the proposed rule;
- `"exact"`: the same rule with an online estimate of Fritzke's utility per point in place of `delta_j` (an extra search for the two nearest centres of every point in every batch);
- `"count"`: the library's rule;
- `"off"`.

`DistanceAwareRevivalEmeans` is a subclass of `faiss.contrib.e_means.NumpyEmeans` (CPU, L2 only; spherical mode is rejected). All other options are the library's.

## Requirements

Python 3.12, `faiss-cpu==1.16.0`, `numpy`, `scikit-learn` (experiments), `scipy` and `matplotlib` (analysis). Exact versions are in `requirements.txt`.

## Reproducing the paper

- **Protocol.** `protocol.md` is the experimental protocol, frozen before the runs, with its three dated amendments.
- **Data.** Run `bash experiments/download.sh` with `BENCH` pointing to FAISS's `benchs/e_means` at commit `81e083c8ac6ee1024337a843748b08a5071bd228` (2026-11-10), which provides the e-means paper's own splits and fingerprints. It downloads and prepares every dataset into `$DISTANCE_REVIVAL_DATA/<name>/{train,test}.npy`, each with a `meta.json` that records sources, row counts and SHA-256 checksums:
  - KDD Cup 1999: figshare file 5976045;
  - US Census 1990, YearPredictionMSD, household power and Shuttle: UCI;
  - GeoNames: `allCountries.zip`;
  - Covertype: scikit-learn;
  - DEEP 1M, DEEP 10M and SPACEV 1M: via the paper's downloader;
  - SIFT 1M: a range request on BIGANN.
- **Runs.** `experiments/grid.py count` lists the grid. `experiments/grid.py run --host H --workers 8 --threads 4 --out RUNDIR` runs every job assigned to host group `H`. We split the grid over two machines, `pikachu` and `eevee`; run both groups on one machine to cover everything. `--only E1,E2,...` selects experiments (including the amendment's `A1`–`A4`), and `--arm exact` selects the re-runs of the utility-estimate arm (Amendments 2 and 3). Results are appended to `RUNDIR/results.jsonl` and runs resume automatically.
- **Analysis.** Put every `results.jsonl` under one folder, one subfolder each (for example by symbolic links), then run `experiments/analyze.py THAT_FOLDER PAPERDIR`. It writes all tables, figures and in-text numbers, and the latest corrected exact-arm records (`impl=exact_v3`, Amendment 3) supersede the earlier ones automatically. The released records are in `results/runs.jsonl.gz` (see `results/README.md`).
- **Other scripts.**
  - `experiments/timing.py`: overhead measurements.
  - `experiments/diag.py`: the bound-versus-utility diagnostic.
  - `experiments/illustration.py`: Figure 1.
- **Tests.** `tests/` contains bit-identity tests against the library, including tie-heavy data (`python -m pytest tests`).

## License

MIT (see `LICENSE`). FAISS is MIT-licensed.
