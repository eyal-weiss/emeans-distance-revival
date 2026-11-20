# Distance-aware revival for e-means

Code for *Small Is Not Dead: Distance-Aware Cluster Revival for e-means* (Eyal Weiss, 2026; [arXiv:2611.18238](https://arxiv.org/abs/2611.18238)).

[e-means](https://arxiv.org/abs/2611.09501) (FAISS 1.16, `faiss.contrib.e_means`) periodically relocates the centres of small clusters (dead-cluster revival): every 5 batches, every cluster whose running count is below a tenth of the average count is removed, and its centre is reused to split a large cluster. The rule considers only size, so on data with small, isolated groups it repeatedly removes centres whose removal is costly.

This package adds a second condition. A cluster *j* selected by the size test (a *candidate*) is removed only if

    delta_j <= tau_near * MSE_hat      or      j received no point for W epochs,

where `delta_j` is the squared distance from its centre to the nearest other centre, and `MSE_hat` is a running estimate of the mean squared error (tau_near = 4, W = 2).
- When a centre is the mean of its points, `delta_j` is an upper bound on the average increase in squared error, per point of the cluster, caused by removing it while the other centres are kept fixed (Fritzke's utility divided by the cluster size).
- The second clause (the abandonment clause) is a safeguard for a centre that has lost all its points far from every other centre; in the paper's experiments it rarely acts and has no measurable effect.
- On the e-means paper's embedding benchmarks, the distance test never prevents a removal, and runs are bit-identical to the library.

## Follow-up: net-gain revival

A cost-based revival rule proposed for FAISS on the day before our paper
([pull request #5915](https://github.com/facebookresearch/faiss/pull/5915)) is considerably stronger than distance-aware
revival on our benchmarks. Our follow-up paper, *Revive What Pays: Evidence-Weighted Net-Gain Revival for Mini-Batch
k-Means*, builds on it: a move is made only if the predicted gain of the split is more likely than not to exceed the
predicted cost of the removal, with every estimate weighted by its evidence. Its code, protocols and records are in
[eyal-weiss/emeans-net-gain-revival](https://github.com/eyal-weiss/emeans-net-gain-revival).

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

## GPU: FAISS's PyTorch e-means and o-means

> The arXiv paper (v1) evaluates the CPU class above only; the code of that version is tag [`arxiv-v1`](https://github.com/eyal-weiss/emeans-distance-revival/tree/arxiv-v1). The GPU classes below were added after the paper. They implement the same rule with the same parameters (tau_near = 4, W = 2 epochs) and were validated in a smaller study (450 runs, below). They are a first implementation, not tuned for speed.

- `DistanceAwareRevivalTorchEmeans` (`distance_revival.torch_emeans`): a subclass of FAISS 1.16's PyTorch `Emeans` (`faiss.contrib.e_means.Emeans`) for one CUDA GPU. It keeps the library's CUDA-graph replay of the training step.
- `OmeansDistanceRevival` (`distance_revival.omeans`): [o-means](https://owlgebra.ai/blog/o-means/) (Bajaj, Nupur and Garg, 2026; code in the FAISS fork [owlgebra-ai/faiss](https://github.com/owlgebra-ai/faiss), `contrib/o_means`) is a GPU k-means for large k that keeps e-means' training loop and revival rule and replaces its assignment step with a two-level search. This class runs o-means with the distance-aware rule in place of its revival rule; everything else is o-means'. Tested with o-means at commit `9ecbf0c`.

```python
import torch
from distance_revival import DistanceAwareRevivalTorchEmeans, OmeansDistanceRevival

x = torch.rand(1_000_000, 96, device="cuda")
km = DistanceAwareRevivalTorchEmeans(x.shape[1], 16384, revival="near", nepochs=2)
centroids, labels = km.train(x)
print(km.n_splits, km.n_protected)

om = OmeansDistanceRevival(x.shape[1], 16384, revival="near", nepochs=2)
centroids, labels = om.train(x)
print(om.revival_stats)
```

`revival` is `"near"` (the proposed rule), `"count"` (the library's rule) or `"off"`; all other options are the library's. Both classes use `unused_threshold=0.1` in every dimension, like the CPU class (both libraries turn revival off for d <= 2 by default); pass `unused_threshold=None` to keep the library's own choice.

**Installation.** PyTorch with Triton (tested: PyTorch 2.15.0 and Triton 3.9.0, `requirements-gpu.txt`) on an NVIDIA GPU (tested on an RTX 3090 Ti). o-means is not on PyPI; to use it, copy its package into your FAISS installation:

    git clone https://github.com/owlgebra-ai/faiss owlgebra-faiss && git -C owlgebra-faiss checkout 9ecbf0c
    cp -r owlgebra-faiss/contrib/o_means "$(python -c 'import faiss, os; print(os.path.dirname(faiss.__file__))')/contrib/"

**Validation** (one RTX 3090 Ti; 5 seeds; 2 epochs, and also 1 epoch on the embeddings; `results/gpu/summary.txt`):
- With the library's own rule, both classes give bit-identical centres to the library (`tests/test_gpu.py`).
- On SPACEV 1M and DEEP 1M (k = 16,384, the e-means paper's configuration), the distance test never prevents a removal, and every run is bit-identical to the library, as on the CPU.
- On data with small, isolated groups, the gains match the CPU results. Test error relative to the library's rule:
  - e-means: KDD -75% (k = 1,024) and -44% (k = 4,096); Shuttle -72% (k = 256) and -39% (k = 1,024); GeoNames -30% and -16% (k = 1,024 and 4,096).
  - o-means: KDD -75%, -41% and -21% (k = 1,024, 4,096 and 16,384); Shuttle -71% and -26%; GeoNames -28%, -14% and -3.5%.
  - MSD: +0.2% to +0.4%, the same small cost as on the CPU.
- Wall time per fit, relative to the library's rule: e-means +3.5% to +5% on the embeddings and +15% on KDD; o-means +11% to +12% on the embeddings and +6% on KDD; on GeoNames (d = 2, revival forced on, about 100 revival checks per fit) +27% and +32%. For comparison, revival itself costs the libraries 0.5% to 7% relative to no revival (6% to 14% on GeoNames). The GPU was shared with another user's job, so these timings are rough (the libraries' own revival cost varied by up to 5 points between repeated measurements).

**Differences from the CPU class** (the rule is the same):
- The nearest other centre of each candidate is ranked on an FP16 copy of the centres with tensor cores. A removal is decided from the exact distance to the centre found, and every candidate that the test would keep is confirmed by an exact FP32 search over all centres, so the decisions are those of an exact search (the CPU class uses FAISS's FP32 search).
- FAISS's GPU e-means assigns points in BF16 by default, and o-means assigns them approximately (its two-level search), so GPU and CPU runs are not bit-comparable; their gains are.
- In o-means, each point's error is computed inside a copy of o-means' accumulate kernel and summed per cluster in FP64 with `index_add_`, which is deterministic under `torch.use_deterministic_algorithms`, so runs stay bitwise reproducible, as o-means' are.
- Not covered: multi-GPU e-means, spherical e-means and the utility-estimate rule (`"exact"`).

**Not tuned.** Two clear next steps are noted in `distance_revival/kernels.py`: stop the neighbour search as soon as some centre within tau_near * MSE_hat is found (enough to remove a candidate), and, in o-means, search only the cells near each candidate. `experiments/gpu/reference/` keeps the first, plain-PyTorch versions: they are easier to read and make the same decisions (all 450 validation runs), but they are slower (o-means +45% to +71%).

**Experiments.** `experiments/gpu/validate.py` runs a job list (`jobs_sanity.json`, `jobs_main.json`, `jobs_timing.json`) on the prepared datasets (`GPU_DETERMINISTIC=1` for bitwise-reproducible runs; `GPU_REFERENCE=1` for the reference versions). `experiments/gpu/analyze.py` writes the summary, and `experiments/gpu/profile_kernels.py` lists the CUDA time per kernel. The records are in `results/gpu/` (see `results/README.md`).

## Requirements

Python 3.12, `faiss-cpu==1.16.0`, `numpy`, `scikit-learn` (experiments), `scipy` and `matplotlib` (analysis). Exact versions are in `requirements.txt`. The GPU classes also need PyTorch and Triton (`requirements-gpu.txt`, see above).

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
- **Tests.** `tests/` contains bit-identity tests against the library, including tie-heavy data (`python -m pytest tests`). The GPU tests (`tests/test_gpu.py`) are skipped on machines without CUDA, and the o-means tests without o-means.

## Citation

If you use this code or its results, please cite the paper ([arXiv:2611.18238](https://arxiv.org/abs/2611.18238)):

```bibtex
@misc{weiss2026small,
  title         = {Small Is Not Dead: Distance-Aware Cluster Revival for e-means},
  author        = {Weiss, Eyal},
  year          = {2026},
  eprint        = {2611.18238},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.18238}
}
```

## License

MIT (see `LICENSE`). FAISS is MIT-licensed. `distance_revival/omeans.py` and `experiments/gpu/reference/omeans_v1.py` contain code adapted from o-means (owlgebra-ai/faiss, MIT-licensed); its copyright notice is kept in those files.
