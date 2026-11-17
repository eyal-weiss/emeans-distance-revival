"""Summary of the GPU validation: per implementation and config, our rule and 'off' relative to the
library's rule (paired by seed), next to the paper's CPU numbers; then wall-time medians.
python experiments/gpu/analyze.py RUNS.jsonl[.gz] [TIMING.jsonl[.gz]] [CPU_REF.json]"""
import gzip, json, sys, collections, statistics as st
read = lambda p: [json.loads(l) for l in (gzip.open(p, "rt") if p.endswith(".gz") else open(p))]
main = read(sys.argv[1])
timing = read(sys.argv[2]) if len(sys.argv) > 2 else []
cpu = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
for r in main + timing:
    if "error" in r: print("ERROR", r["key"], r["error"])
R = collections.defaultdict(dict)
for r in main:
    if "error" not in r: R[(r["impl"], r["ds"], r["k"], r["T"])][(r["seed"], r["arm"])] = r
for impl in ("emeans", "omeans"):
    print(f"\n== {impl} GPU (deterministic mode; 5 seeds; T = epochs) ==")
    print(f"{'dataset':9s} {'k':>6s} {'T':>1s} | {'ours vs lib':>11s} {'W/T/L':>6s} | {'off vs lib':>10s} | {'kept/run':>8s} {'identical':>9s} | {'CPU paper: ours':>15s} {'off':>8s}")
    for key in sorted(k for k in R if k[0] == impl):
        d = R[key]; seeds = sorted(s for s, a in d if a == "count")
        pr = [(d[(s, "count")], d[(s, "near")]) for s in seeds if (s, "near") in d]
        rel = st.mean(n["test_mse"] / c["test_mse"] - 1 for c, n in pr) * 100
        w = sum(n["test_mse"] < c["test_mse"] for c, n in pr); t = sum(n["test_mse"] == c["test_mse"] for c, n in pr)
        same = sum(n["cent_sha"] == c["cent_sha"] for c, n in pr)
        off = st.mean(d[(s, "off")]["test_mse"] / d[(s, "count")]["test_mse"] - 1 for s in seeds if (s, "off") in d) * 100
        kept = st.mean(n.get("protected", 0) for c, n in pr)
        ref = cpu.get(f"{key[1]}|{key[2]}|{key[3]}")
        refs = f"{ref['near_pct']:+14.2f}% {ref['off_pct']:+7.2f}%" if ref else f"{'-':>15s} {'-':>8s}"
        print(f"{key[1]:9s} {key[2]:6d} {key[3]:1d} | {rel:+10.2f}% {w}/{t}/{len(pr)-w-t:<2d} | {off:+9.2f}% | {kept:8.0f} {same:>5d}/{len(pr)} | {refs}")
    gc = [r.get("graph_captures") for r in main if r["impl"] == "emeans" and "error" not in r]
    if impl == "emeans": print(f"CUDA graph replay used in {sum(1 for g in gc if g)}/{len(gc)} e-means runs")
if timing:
    T = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in timing:
        if "error" not in r and r["seed"] > 0: T[(r["impl"], r["ds"], r["k"], r["T"])][r["arm"]].append(r["time_s"] * 1000)
    print("\n== wall time per fit, median of seeds 1-5 (ms; non-deterministic mode; GPU shared with another user) ==")
    for key in sorted(T):
        m = {a: st.median(v) for a, v in T[key].items()}
        print(f"{key[0]:6s} {key[1]:9s} k={key[2]:5d} T={key[3]} | lib {m['count']:7.1f} | ours {m['near']:7.1f} ({(m['near']/m['count']-1)*100:+5.1f}%) | off {m['off']:7.1f}")
