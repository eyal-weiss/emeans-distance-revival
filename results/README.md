# Raw results

- **`runs.jsonl.gz`:** one JSON record per run, covering every experiment of the paper (the main grid and Amendments 1–3).
  - Fields include: dataset, k, seed, initialization (`init`), epochs (`T`), arm, test/train MSE, imbalance, tail error, splits, protections, candidates, the histogram of δ_j/MSE-hat (`ratio_hist`, bins `distance_revival.RATIO_BINS`), wall time and host.
  - The utility-estimate arm (`arm=exact`) appears in three versions: untagged (original), `impl=exact_v2` (Amendment 2) and `impl=exact_v3` (Amendment 3, used in the paper). `analyze.py` keeps only the latest automatically.
- **`timing.jsonl`:** the controlled, counterbalanced overhead timings (Section 5.6).
- **`diag/*.json.gz`:** the bound-versus-estimate diagnostic logs (Section 5.7).

To regenerate the paper's tables and numbers:

    mkdir -p all/runs && gunzip -c results/runs.jsonl.gz > all/runs/results.jsonl
    DISTANCE_REVIVAL_TIMING=results/timing.jsonl DISTANCE_REVIVAL_DIAG=results/diag/msd_z_k4096_T10_s0.json.gz \
        python experiments/analyze.py all paper_out
