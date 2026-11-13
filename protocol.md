# Experimental protocol — frozen 2026-11-11 15:05 EST, before any run of the grid below

> **Note added 2026-11-13, before the code was released.** The package `emeans_rr` and the class `RobustRevivalEmeans` named below were renamed `distance_revival` and `DistanceAwareRevivalEmeans` (and the repository `emeans-rr` became `emeans-distance-revival`), to match the paper's term, distance-aware revival. The rename changed no behaviour. The protocol text below is otherwise unchanged.

Code: `arxiv/code` (`emeans_rr.RobustRevivalEmeans`, `experiments/grid.py`, `experiments/datasets.py`). FAISS 1.16.0 (`faiss-cpu`, PyPI), NumPy 2.5.4, scikit-learn 1.9.1, Python 3.12. CPU only (ARCS servers Pikachu: Threadripper 3990X; Eevee: Threadripper 7980X), 8 workers × 4 threads per host.

## Rule and parameters (frozen)
- Proposed rule `near`: library size test (N_j < 0.1 · mean N, every 5 batches) + evict only if δ_j ≤ τ · m or no hit for W passes. **τ = 4, W = 2 passes**, chosen during development on Covertype, synthetic far groups and 200k slices of DEEP/SIFT (revival_fix_proposal.md). Not retuned on any held-out dataset.
- Arms: `count` (library rule; the library default for d > 2), `off` (no revival; the library default for d ≤ 2 since FAISS 1.16), `near` (proposed), `near_noab` (near without the abandonment clause), `exact` (same test with the exact per-point deletion cost from second-nearest distances), FAISS Lloyd with 25 iterations (`lloyd`, `max_points_per_centroid=1e9`, i.e. no subsampling). The size threshold is 0.1 at every dimension for count/near/near_noab/exact.
- All other e-means settings are the library defaults (scaling laws, `batch_law="auto"`, closing mean).

## Datasets
Development (used to design the rule; reported, labelled as development): Covertype z-scored and min-max; synthetic far groups; (earlier) 200k slices of DEEP/SIFT.
Held-out (never used during development):
- KDD Cup 1999 full (4.9M rows; one-hot protocol/service/flag; z-scored), US Census 1990 (68 attributes, z-scored), YearPredictionMSD (90 features, year dropped, z-scored), GeoNames (latitude, longitude in degrees, 1M-row subsample).
- Embeddings at the e-means paper's configuration: DEEP* 1M (the paper's exact split) and SIFT 1M (first 1M BIGANN rows), k = 16,384.
- Synthetic heterogeneous-scale probe (`synth_hetero:2`, `synth_hetero:16`), designed to expose a possible weakness of the global yardstick m.
Tabular preprocessing (all): remove exact duplicate rows; hold out 20,000 rows (seed 1234); test = one half of the held-out rows (seed 1234), as in the e-means paper; scale with training statistics.

## Grid
- E1 (tabular): k ∈ {256, 1024, 4096}; T ∈ {1, 2, 3, 5, 10} passes; seeds 0–9; starts: random data rows and greedy k-means++ (scikit-learn `kmeans_plusplus`, same centres for all arms of a seed); arms count/off/near/near_noab/exact + Lloyd@25 per start.
- E2 (embeddings): k = 16,384; T ∈ {1, 2, 3, 5}; seeds 0–9; random rows; arms count/off/near; exact at T ∈ {2, 5} seeds 0–4; Lloyd@25 seeds 0–2 (sanity check against the paper's Table 5).
- E3 (sensitivity): τ ∈ {1, 2, 3, 6, 8, 16, 32}, W ∈ {1, 3} on Covertype-z, KDD, GeoNames (k = 1024, T ∈ {2, 5, 10}, seeds 0–4) and DEEP 1M (τ ∈ {1, 2, 8}, W = 1; T ∈ {2, 5}).
- S (synthetic): far groups m ∈ {50, 100, 150, 250, 400}, hetero d ∈ {2, 16}; T = 5; seeds 0–9; both starts; all main arms + Lloyd@25.
- E6 (overhead): separate timing runs after the grid, on an otherwise idle host, 5 interleaved repeats.

## Metrics and statistics
- Test MSE on held-out rows (FP32 ranking, FP64 distance); train MSE on all training rows (tabular, synthetic) or a fixed 200k subset (embeddings, as in the paper); cluster-size imbalance k·Σn_j²/n²; tail error (mean of the worst 0.1% per-point errors); far groups covered (synthetic); splits, candidates, protections; δ/m histogram of every size candidate.
- Reported: medians over seeds; per-seed paired differences; win/tie/loss counts (tie: |rel. diff| < 1e-4); two-sided Wilcoxon signed-rank tests (paired by seed) of near vs count and near vs off, Holm-corrected within each experiment; passes needed to reach Lloyd@25 (linear interpolation between budgets).
- Every configuration in the grid is reported (appendix tables), including unfavourable ones.

## Amendment 1 — 2026-11-11 20:55 EST (approved by the author; written before any of these runs)
Added to increase statistical power and breadth. τ = 4 and W = 2 are unchanged. Every outcome is reported.
- **A1 (power):** seeds 10–29 for the headline configuration. Covers every tabular dataset of E1 and k ∈ {256, 1024, 4096}, at T = 2 passes from random starts, for the arms count/off/near/exact (30 seeds in total there). Lloyd@25 stays at 10 seeds.
- **A2 (diagnostics, post hoc for the first item):**
  - **(a)** Why near over-protects on YearPredictionMSD at k = 4096. Seed-0 runs log, for every size candidate at every check, the bound ratio δ_j/m next to the exact per-point utility ratio (G_j/N_j)/m. Also run on KDD and DEEP for comparison.
  - **(b)** Seeds 5–9 for every E3 ablation arm (10 seeds in total).
- **A3 (embedding breadth):**
  - SPACEV 1M from the e-means paper's suite (its split, via `benchs/e_means/paper_data.py`), k = 16,384, T ∈ {2, 5}, seeds 0–9, arms count/off/near; Lloyd@25 for seeds 0–2.
  - DEEP 10M (the paper's split) at k = 65,536, T = 2, seeds 0–2, arms count/near only. This is a bit-identity check at 10× scale; no Lloyd.
- **A4 (two more held-out datasets, chosen before any result on them):**
  - UCI Individual Household Electric Power Consumption: 7 numeric columns; rows with missing values dropped; deduplicated; z-scored.
  - UCI Statlog Shuttle: training and test files combined; 9 numeric attributes; deduplicated; z-scored.
  - Grid: random starts; k ∈ {256, 1024}; T ∈ {1, 2, 5, 10}; seeds 0–9; arms count/off/near/near_noab/exact plus Lloyd@25.
- **Codex review:** an independent review runs after the experiments, at 05:00 EST at the latest.

## Amendment 2 — 2026-11-12 02:12 EST (after independent review round 1; written before these reruns)
The review found that the `exact` arm took its assignment from a 2-NN search instead of the library's 1-NN search. Under ties and FP32 rounding these can disagree (on KDD, k=1024, seed 0, the threshold-off exact arm made 331 splits versus the library's 329), so "only revival differs" did not hold for that arm.

- **Change 1.** The corrected arm keeps the library's assignment unchanged and measures the distance to the nearest other centre with a separate 2-NN search.
- **Change 2.** A centre that has received no point since its birth (or since a split created it) now has unknown utility. It is not evicted on utility grounds, but it can still be abandoned. Before, its utility counted as zero.

**Re-run:** every `exact` run of E1, E2, S, A1 and A4, with records tagged `impl=exact_v2`. Also re-run: the A2(a) diagnostic, built on the same estimate, and the E6 timings (now counterbalanced, with the order of arms rotated across repeats).

**Unchanged:** τ = 4 and W = 2. All other arms are unaffected: the near arm was verified bit-identical before and after the change.

## Amendment 3 — 2026-11-12 04:57 EST (after independent review round 2; written before these reruns)
Review round 2 found a bias in the corrected estimate. When a split created a centre, Amendment 2 reset its utility numerator G_j but kept its share of the donor's count N_j as the denominator. After the first observation, its estimate was therefore attenuated, toward eviction.

- **Change.** The estimate now divides by a separate EMA of the counts behind G_j, and the two restart together at birth.
- **Re-run:** every utility-estimate run (records tagged `impl=exact_v3`), the diagnostic and the timings.
- **Unchanged:** all other arms, which do not read these statistics.
