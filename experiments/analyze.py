"""Aggregate grid results into the paper's tables, figures and numbers.

python analyze.py RESULTS_DIR PAPER_DIR
  RESULTS_DIR: arxiv/results/<run> (contains <host>/results.jsonl)
  writes PAPER_DIR/{numbers.tex, tables/*.tex, figures/*.pdf} and RESULTS_DIR/summary.md
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

HELD = ["kdd_z", "census_z", "msd_z", "geonames", "power_z", "shuttle_z"]
DEV = ["covtype_z", "covtype_mm"]
EMB = ["deep1m", "sift1m", "spacev1m", "deep10m"]
NAME = {"kdd_z": "KDD Cup 99", "census_z": "Census 1990", "msd_z": "YearPredictionMSD",
        "geonames": "GeoNames (2-D)", "covtype_z": "Covertype (z-score)",
        "covtype_mm": "Covertype (min-max)", "deep1m": "DEEP 1M", "sift1m": "SIFT 1M",
        "power_z": "Household power", "shuttle_z": "Shuttle", "spacev1m": "SPACEV 1M", "deep10m": "DEEP 10M"}
ARMNAME = {"count": "Count (library)", "off": "Off (no revival)", "near": "Distance (ours)",
           "near_noab": "Distance, no abandonment", "exact": "Estimate"}
# method names in the LaTeX tables (the code keys count/off/near/exact are unchanged)
TEXARM = {"count": "\\textsc{count}", "off": "\\textsc{off}", "near": "\\textsc{distance}",
          "near_noab": "\\textsc{distance}, no abandonment", "exact": "\\textsc{estimate}", "lloyd": "Lloyd@25"}
TS = (1, 2, 3, 5, 10)
TIE = 1e-4


def load(results_dir):
    R = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*", "results.jsonl"))):
        for line in open(f):
            try:
                R.append(json.loads(line))
            except Exception:
                pass
    # Amendments 2-3: records of the latest corrected exact arm supersede earlier ones
    for impl in ("exact_v3", "exact_v2"):
        if any(r.get("impl") == impl for r in R):
            R = [r for r in R if r.get("arm") != "exact" or r.get("impl") == impl]
            break
    return R


class Index:
    def __init__(self, R):
        self.em = defaultdict(dict)       # (arm, ds, k, init, T) -> seed -> rec
        self.ll = defaultdict(dict)       # (ds, k, init) -> seed -> rec
        for r in R:
            if r["kind"] == "lloyd":
                self.ll[(r["dataset"], r["k"], r["init"])][r["seed"]] = r
            else:
                self.em[(r["arm"], r["dataset"], r["k"], r["init"], r["T"])][r["seed"]] = r

    def get(self, arm, ds, k, init, T):
        return self.em.get((arm, ds, k, init, T), {})

    def lloyd(self, ds, k, init):
        return self.ll.get((ds, k, init), {})


def med(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(x)) if x else float("nan")


def paired(a, b, metric="test_mse"):
    seeds = sorted(set(a) & set(b))
    return np.array([a[s][metric] for s in seeds]), np.array([b[s][metric] for s in seeds])


def wtl(x, y):
    """wins/ties/losses of x vs y (lower is better)."""
    rel = (x - y) / np.maximum(np.abs(y), 1e-300)
    return int((rel < -TIE).sum()), int((np.abs(rel) <= TIE).sum()), int((rel > TIE).sum())


def wilcoxon_p(x, y):
    from scipy.stats import wilcoxon
    d = x - y
    if len(d) == 0 or np.all(np.abs(d) <= TIE * np.abs(y)):
        return 1.0
    try:
        return float(wilcoxon(x, y, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    m = len(ps)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def rel_change(x, y):
    """median over seeds of per-seed % change of x vs y."""
    if len(x) == 0:
        return float("nan")
    return float(np.median(100 * (x / y - 1)))


def passes_to(curve, target):
    """curve: {T: median metric}; the smallest TESTED number of epochs T whose median is <= target
    (separately trained runs per number of epochs; no interpolation). inf if none."""
    pts = [(T, curve[T]) for T in TS if T in curve and not math.isnan(curve[T])]
    if not pts:
        return float("nan")
    for t, v in pts:
        if v <= target:
            return float(t)
    return float("inf")


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if v == float("inf"):
        return "$>10$"
    return f"{v:.{nd}f}"


def pct(v):
    if math.isnan(v):
        return "--"
    s = f"{v:+.1f}\\%"
    return s.replace("-", "$-$") if v < 0 else s


# ------------------------------------------------------------------------------ analyses
def tabular_table(I, datasets, init, T, out):
    """rows: dataset x k. cols: count/off/near/exact relative to Lloyd@25 (same initialization);
    near vs count and near vs off: median change, W/T/L, Holm-adjusted p (computed by caller)."""
    rows = []
    for ds in datasets:
        for k in (256, 1024, 4096):
            L = I.lloyd(ds, k, init)
            if not L:
                continue
            ref = med([r["test_mse"] for r in L.values()])
            arms = {a: I.get(a, ds, k, init, T) for a in ("count", "off", "near", "exact")}
            if not arms["near"]:
                continue
            ratio = {a: med([r["test_mse"] for r in v.values()]) / ref if v else float("nan")
                     for a, v in arms.items()}
            x, y = paired(arms["near"], arms["count"])
            xo, yo = paired(arms["near"], arms["off"])
            rows.append(dict(ds=ds, k=k, ratio=ratio, n_near=len(x), d_cnt=rel_change(x, y), wtl_cnt=wtl(x, y),
                             p_cnt=wilcoxon_p(x, y), d_off=rel_change(xo, yo), wtl_off=wtl(xo, yo),
                             p_off=wilcoxon_p(xo, yo), lloyd=ref,
                             prot=med([r.get("n_protected", 0) for r in arms["near"].values()]),
                             splits_cnt=med([r.get("n_splits", 0) for r in arms["count"].values()]),
                             splits_near=med([r.get("n_splits", 0) for r in arms["near"].values()])))
    return rows


def pfmt(p):
    if p is None or math.isnan(p):
        return "--"
    if p < 1e-3:
        e = int(math.floor(math.log10(p)))
        return f"$<10^{{{e + 1}}}$"
    return f"{p:.3f}"


def write_tabular_tex(rows_held, rows_dev, T, init, path):
    def line(r):
        w, t, l = r["wtl_cnt"]
        wo, to, lo = r["wtl_off"]
        return (f"{NAME[r['ds']]} & {r['k']} & {r['n_near']} & {fmt(r['ratio']['count'])} & {fmt(r['ratio']['off'])} & "
                f"\\textbf{{{fmt(r['ratio']['near'])}}} & {fmt(r['ratio']['exact'])} & "
                f"{pct(r['d_cnt'])} & {w}/{t}/{l} & {pfmt(r.get('p_cnt_holm'))} & "
                f"{pct(r['d_off'])} & {wo}/{to}/{lo} & {pfmt(r.get('p_off_holm'))} \\\\")
    s = ["\\begin{tabular}{lllcccccccccc}", "\\toprule",
         " & & & \\multicolumn{4}{c}{Median test MSE / median test MSE of Lloyd@25} & "
         "\\multicolumn{3}{c}{\\textsc{distance} vs.\\ \\textsc{count}} & \\multicolumn{3}{c}{\\textsc{distance} vs.\\ \\textsc{off}}\\\\",
         "\\cmidrule(lr){4-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}",
         "Dataset & $k$ & Seeds & \\textsc{count} & \\textsc{off} & \\textsc{distance} & \\textsc{estimate} & $\\Delta$ & W/T/L & $p$ "
         "& $\\Delta$ & W/T/L & $p$\\\\", "\\midrule",
         "\\multicolumn{13}{l}{\\emph{Held-out datasets}}\\\\"]
    s += [line(r) for r in rows_held]
    if rows_dev:
        s += ["\\midrule", "\\multicolumn{13}{l}{\\emph{Development datasets}}\\\\"] + [line(r) for r in rows_dev]
    s += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(s) + "\n")


def passes_table(I, datasets, init, path):
    rows = []
    for ds in datasets:
        for k in (256, 1024, 4096):
            L = I.lloyd(ds, k, init)
            if not L:
                continue
            ref = med([r["test_mse"] for r in L.values()])
            row = dict(ds=ds, k=k)
            for a in ("count", "off", "near", "exact"):
                curve = {T: med([r["test_mse"] for r in I.get(a, ds, k, init, T).values()]) for T in TS}
                row[a] = passes_to(curve, ref)
            rows.append(row)
    s = ["\\begin{tabular}{llcccc}", "\\toprule",
         "Dataset & $k$ & \\textsc{count} & \\textsc{off} & \\textsc{distance} & \\textsc{estimate}\\\\", "\\midrule"]
    for r in rows:
        cells = []
        for a in ("count", "off", "near", "exact"):
            v = r[a]
            cells.append("--" if math.isnan(v) else ("$>10$" if v == float("inf") else f"{v:.0f}"))
        s.append(f"{NAME[r['ds']]} & {r['k']} & " + " & ".join(cells) + "\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(s) + "\n")
    return rows


def embedding_rows(I):
    rows = []
    for ds in EMB:
        kk = 65536 if ds == "deep10m" else 16384
        L = I.lloyd(ds, kk, "random")
        ref = med([r["test_mse"] for r in L.values()]) if L else float("nan")
        for T in (1, 2, 3, 5):
            c, n, o = (I.get(a, ds, kk, "random", T) for a in ("count", "near", "off"))
            if not n:
                continue
            seeds = sorted(set(c) & set(n))
            ident = sum(1 for s in seeds if c[s]["test_mse"] == n[s]["test_mse"]
                        and c[s]["train_mse"] == n[s]["train_mse"])
            rows.append(dict(ds=ds, T=T, runs=len(seeds), identical=ident,
                             prot=int(sum(n[s].get("n_protected", 0) for s in seeds)),
                             splits=med([n[s].get("n_splits", 0) for s in seeds]),
                             count=med([r["test_mse"] for r in c.values()]) / ref,
                             near=med([r["test_mse"] for r in n.values()]) / ref,
                             off=med([r["test_mse"] for r in o.values()]) / ref if o else float("nan"),
                             lloyd=ref, k=kk))
    return rows


def ratio_hist(I, ds, k, arm="count", init="random", T=2):
    from distance_revival import RATIO_BINS
    h = np.zeros(len(RATIO_BINS) + 1)
    for r in I.get(arm, ds, k, init, T).values():
        if "ratio_hist" in r:
            h += np.asarray(r["ratio_hist"])
    return RATIO_BINS, h


def fig_ratio_cdf(I, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    styles = [("deep1m", 16384, "C0", "-"), ("sift1m", 16384, "C0", "--"), ("kdd_z", 1024, "C3", "-"),
              ("census_z", 1024, "C2", "-"), ("msd_z", 1024, "C4", "-"), ("geonames", 1024, "C1", "-"),
              ("covtype_z", 1024, "C3", ":")]
    out = {}
    for ds, k, col, ls in styles:
        bins, h = ratio_hist(I, ds, k)
        if h.sum() == 0:
            continue
        # bucket i holds values in [bins[i-1], bins[i]); plot CDF at right edges
        cdf = np.cumsum(h) / h.sum()
        edges = np.concatenate([bins, [bins[-1] * 10]])
        ax.step(edges, cdf, where="post", color=col, ls=ls, lw=1.6, label=f"{NAME[ds]} ({int(h.sum()):,})")
        out[ds] = dict(n=int(h.sum()), frac_above4=float(h[np.searchsorted(bins, 4.0, side="right"):].sum() / h.sum()))
    ax.axvline(4, color="k", lw=0.8)
    ax.text(4.4, 0.04, r"$\tau_{\mathrm{near}}=4$", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1e-1, 1e5)
    ax.set_xlabel(r"$\delta_j / \widehat{\mathrm{MSE}}$ of candidates (count-based revival, 2 epochs)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return out


def fig_curves(I, path, panels=(("kdd_z", 1024), ("census_z", 1024), ("geonames", 1024), ("msd_z", 1024))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.7))
    for ax, (ds, k) in zip(np.atleast_1d(axes), panels):
        L = I.lloyd(ds, k, "random")
        if not L:
            continue
        ref = med([r["test_mse"] for r in L.values()])
        # hollow markers and a dashed line for Estimate keep coinciding curves visible
        for arm, col, mk, ls, ms in (("count", "C3", "s", "-", 5), ("off", "C1", "^", "-", 5),
                                     ("near", "C0", "o", "-", 6), ("exact", "C2", "x", "--", 4)):
            Ts = [T for T in TS if I.get(arm, ds, k, "random", T)]
            if not Ts:
                continue
            y = [med([r["test_mse"] for r in I.get(arm, ds, k, "random", T).values()]) / ref for T in Ts]
            ax.plot(Ts, y, marker=mk, color=col, ls=ls, lw=1.4, ms=ms, mfc="none", label=ARMNAME[arm])
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(list(TS))
        ax.set_xticklabels([str(t) for t in TS])
        ax.set_title(f"{NAME[ds]}, $k={k}$", fontsize=9)
        ax.set_xlabel("Epochs")
    np.atleast_1d(axes)[0].set_ylabel("Test MSE / Lloyd@25")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sensitivity(I):
    out = {}
    for ds, k in (("covtype_z", 1024), ("kdd_z", 1024), ("geonames", 1024), ("deep1m", 16384)):
        for T in (2, 5, 10):
            # ablation arms have seeds 0-4 only: compare every arm on exactly those seeds
            base = {s: r for s, r in I.get("count", ds, k, "random", T).items() if s < 10}
            if not base:
                continue
            row = {}
            for arm in ["near_tau1", "near_tau2", "near_tau3", "near", "near_tau6", "near_tau8", "near_tau16",
                        "near_tau32", "near_w1", "near_w3", "near_noab", "off", "exact"]:
                a = {s: r for s, r in I.get(arm, ds, k, "random", T).items() if s < 10}
                if not a:
                    continue
                x, y = paired(a, base)
                xt, yt = paired(a, base, "train_mse")
                row[arm] = dict(test=rel_change(x, y), train=rel_change(xt, yt), n=len(x))
            out[(ds, T)] = row
    return out


def _w(path, lines):
    open(path, "w").write("\n".join(lines) + "\n")


def tex_tables(I, paper, emb, sens, N):
    T_ = os.path.join(paper, "tables")
    # embeddings
    s = ["\\begin{tabular}{lcccccc}", "\\toprule",
         " & & & \\multicolumn{3}{c}{Test MSE / Lloyd@25} & \\\\ \\cmidrule(lr){4-6}",
         "Dataset & Epochs & Identical runs & \\textsc{count} & \\textsc{off} & \\textsc{distance} & Splits per run\\\\", "\\midrule"]
    for r in emb:
        f4 = lambda v: "--" if math.isnan(v) else f"{v:.4f}"
        s.append(f"{NAME[r['ds']]} & {r['T']} & {r['identical']}/{r['runs']} & {f4(r['count'])} & {f4(r['off'])} & "
                 f"{f4(r['near'])} & {r['splits']:.0f}\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    _w(os.path.join(T_, "emb.tex"), s)
    # near vs exact
    s = ["\\begin{tabular}{llcccc}", "\\toprule",
         "Dataset & $k$ & $\\Delta$ test ($T=2$) & $\\Delta$ test ($T=5$) & W/T/L ($T=2$) & Time ratio\\\\", "\\midrule"]
    gaps, tratios = [], []
    for ds in HELD + DEV + EMB:
        for k in ((16384,) if ds in EMB else (256, 1024, 4096)):
            cells, w2, tr = [], "--", []
            for T in (2, 5):
                a, b = I.get("near", ds, k, "random", T), I.get("exact", ds, k, "random", T)
                if not a or not b:
                    cells.append("--")
                    continue
                x, y = paired(a, b)
                d = rel_change(x, y)
                gaps.append(d)
                cells.append(pct(d))
                tn, te = paired(a, b, "seconds")
                tr += list(tn / te)
                if T == 2:
                    w2 = "/".join(map(str, wtl(x, y)))
            if all(c == "--" for c in cells):
                continue
            tratios += tr
            s.append(f"{NAME[ds]} & {k} & {cells[0]} & {cells[1]} & {w2} & {np.median(tr):.2f}\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    _w(os.path.join(T_, "exact.tex"), s)
    if gaps:
        N["ExactGapMedian"] = float(np.median(gaps))
        N["ExactGapMax"] = float(np.max(gaps))
        N["ExactTimeRatioMedian"] = float(np.median(tratios))
    # sensitivity
    arms = ["near_tau1", "near_tau2", "near_tau3", "near", "near_tau6", "near_tau8", "near_tau16", "near_tau32",
            "near_w1", "near_w3", "near_noab", "off"]
    head = ["$\\taunear{=}1$", "2", "3", "\\textbf{4}", "6", "8", "16", "32", "$W{=}1$", "$W{=}3$", "No clause",
            "\\textsc{off}"]
    s = ["\\begin{tabular}{ll" + "c" * len(arms) + "}", "\\toprule",
         "Dataset & $T$ & " + " & ".join(head) + "\\\\", "\\midrule"]
    for (ds, T), row in sens.items():
        s.append(f"{NAME[ds]} & {T} & " + " & ".join(
            (f"{row[a]['test']:+.1f}".replace("-", "$-$") if a in row else "--") for a in arms) + "\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    _w(os.path.join(T_, "sens.tex"), s)
    # imbalance / tail (random start, T=2)
    s = ["\\begin{tabular}{llcccc}", "\\toprule",
         " & & \\multicolumn{2}{c}{Imbalance $k\\sum_j |P_j|^2/n^2$} & \\multicolumn{2}{c}{Mean of the worst 0.1\\% errors} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
         "Dataset & $k$ & \\textsc{count} & \\textsc{distance} & \\textsc{count} & \\textsc{distance}\\\\", "\\midrule"]
    imb_up = []
    for ds in HELD + DEV:
        for k in (256, 1024, 4096):
            c, n = I.get("count", ds, k, "random", 2), I.get("near", ds, k, "random", 2)
            if not c or not n:
                continue
            ic, inn = med([r["imbalance"] for r in c.values()]), med([r["imbalance"] for r in n.values()])
            imb_up.append(100 * (inn / ic - 1))
            s.append(f"{NAME[ds]} & {k} & {ic:.2f} & {inn:.2f} & {med([r['tail_mse'] for r in c.values()]):.3g} & "
                     f"{med([r['tail_mse'] for r in n.values()]):.3g}\\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    _w(os.path.join(T_, "imbalance.tex"), s)
    if imb_up:
        N["ImbalanceIncreaseMedian"] = float(np.median(imb_up))
        N["ImbalanceIncreaseMax"] = float(np.max(imb_up))
    # synthetic far groups + hetero
    s = ["\\begin{tabular}{ll" + "c" * 7 + "}", "\\toprule",
         " & & \\multicolumn{5}{c}{Far groups kept (of 5), by group size $n_{\\mathrm{far}}$} & \\multicolumn{2}{c}{Heterogeneous, MSE} \\\\",
         "\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}",
         "Initialization & Method & 50 & 100 & 150 & 250 & 400 & $d{=}2$ & $d{=}16$\\\\", "\\midrule"]
    for init in ("random", "kmpp"):
        for arm in ("lloyd", "count", "off", "near", "near_noab", "exact"):
            cells = []
            for m in (50, 100, 150, 250, 400):
                recs = list((I.lloyd(f"synth_far:{m}", 105, init) if arm == "lloyd"
                             else I.get(arm, f"synth_far:{m}", 105, init, 5)).values())
                cells.append(f"{np.mean([r['far_covered'] for r in recs]):.1f}" if recs else "--")
            for d in (2, 16):
                recs = list((I.lloyd(f"synth_hetero:{d}", 100, init) if arm == "lloyd"
                             else I.get(arm, f"synth_hetero:{d}", 100, init, 5)).values())
                cells.append(f"{med([r['test_mse'] for r in recs]):.3g}" if recs else "--")
            label = TEXARM.get(arm, arm)
            s.append(f"{'Random' if init == 'random' else 'k-means++'} & {label} & " + " & ".join(cells) + "\\\\")
        if init == "random":
            s.append("\\midrule")
    s += ["\\bottomrule", "\\end{tabular}"]
    _w(os.path.join(T_, "synth.tex"), s)


def text_numbers(I, paper):
    """Every number the prose cites, as LaTeX macros (paper/numbers.tex)."""
    M = {}

    def d(arm_a, arm_b, ds, k, init="random", T=2, metric="test_mse", seeds=None):
        a, b = I.get(arm_a, ds, k, init, T), I.get(arm_b, ds, k, init, T)
        if seeds is not None:
            a = {s: r for s, r in a.items() if s in seeds}
            b = {s: r for s, r in b.items() if s in seeds}
        x, y = paired(a, b, metric)
        return rel_change(x, y)

    def ratio(arm, ds, k, init="random", T=2):
        L = I.lloyd(ds, k, init)
        v = I.get(arm, ds, k, init, T)
        if not L or not v:
            return float("nan")
        return med([r["test_mse"] for r in v.values()]) / med([r["test_mse"] for r in L.values()])

    ks = (256, 1024, 4096)
    def gains(ds, T=2):
        v = [-d("near", "count", ds, k, T=T) for k in ks if I.get("near", ds, k, "random", T)]
        return v
    for ds, tag in (("kdd_z", "Kdd"), ("geonames", "Geo"), ("covtype_z", "Cov"), ("census_z", "Census"),
                    ("msd_z", "Msd"), ("power_z", "Power"), ("shuttle_z", "Shuttle")):
        g = gains(ds)
        if g:
            M[f"{tag}GainMin"] = f"{min(g):.0f}"
            M[f"{tag}GainMax"] = f"{max(g):.0f}"
            M[f"{tag}DeltaMin"] = f"{-max(g):+.1f}"
            M[f"{tag}DeltaMax"] = f"{-min(g):+.1f}"
            M[f"{tag}AbsMax"] = f"{max(abs(v) for v in g):.1f}"
        g10 = gains(ds, T=10)
        if g10:
            M[f"{tag}DeltaMaxTen"] = f"{-min(g10):+.1f}"
            M[f"{tag}GainMaxTen"] = f"{max(g10):.0f}"
    held = [ds for ds in HELD if any(I.get("near", ds, k, "random", 2) for k in ks)]
    vs_off = [d("near", "off", ds, k) for ds in held for k in ks if I.get("near", ds, k, "random", 2)]
    if vs_off:
        M["WorstVsOffTwo"] = f"{max(vs_off):+.2f}"
    # epochs to reach Lloyd@25 (random initialization): configurations where count never gets there
    never, near_p = 0, []
    for ds in held:
        for k in ks:
            L = I.lloyd(ds, k, "random")
            if not L:
                continue
            ref = med([r["test_mse"] for r in L.values()])
            cc = {T: med([r["test_mse"] for r in I.get("count", ds, k, "random", T).values()]) for T in TS}
            nn = {T: med([r["test_mse"] for r in I.get("near", ds, k, "random", T).values()]) for T in TS}
            if passes_to(cc, ref) == float("inf"):
                never += 1
                pn = passes_to(nn, ref)
                if pn != float("inf"):
                    near_p.append(pn)
    M["NeverReachCount"] = str(never)
    M["NearReachMax"] = f"{max(near_p):.0f}" if near_p else "--"
    M["NearReachCount"] = str(len(near_p))
    M["NearReachWithinTwo"] = str(sum(1 for v in near_p if v <= 2.0))
    M["CovCountRatio"] = f"{ratio('count', 'covtype_z', 256):.2f}"
    M["CovOffBelowLloyd"] = f"{100 * (1 - ratio('off', 'covtype_z', 256)):.0f}"
    M["CovCountRatioTen"] = f"{ratio('count', 'covtype_z', 256, T=10):.2f}"
    # the same two ratios as percentages above Lloyd@25 (the text uses percentages throughout)
    M["CovCountAboveLloyd"] = f"{100 * (ratio('count', 'covtype_z', 256) - 1):.0f}"
    M["CovCountAboveLloydTen"] = f"{100 * (ratio('count', 'covtype_z', 256, T=10) - 1):.0f}"
    # k-means++ initialization
    for ds, tag in (("kdd_z", "Kdd"), ("geonames", "Geo")):
        M[f"{tag}KmppCount"] = f"{ratio('count', ds, 1024, 'kmpp'):.2f}"
        M[f"{tag}KmppNear"] = f"{ratio('near', ds, 1024, 'kmpp'):.2f}"
        M[f"{tag}KmppOff"] = f"{ratio('off', ds, 1024, 'kmpp'):.2f}"
        M[f"{tag}KmppExact"] = f"{ratio('exact', ds, 1024, 'kmpp'):.2f}"
    # imbalance and tail (random, T=2)
    def mv(arm, ds, k, key):
        return med([r[key] for r in I.get(arm, ds, k, "random", 2).values()])
    M["MsdImbCount"] = f"{mv('count', 'msd_z', 256, 'imbalance'):.2f}"
    M["MsdImbNear"] = f"{mv('near', 'msd_z', 256, 'imbalance'):.2f}"
    M["KddTailFactor"] = f"{mv('count', 'kdd_z', 1024, 'tail_mse') / mv('near', 'kdd_z', 1024, 'tail_mse'):.1f}"
    M["CovTailFactor"] = f"{mv('count', 'covtype_z', 256, 'tail_mse') / mv('near', 'covtype_z', 256, 'tail_mse'):.0f}"
    # synthetic heterogeneous scales, d=2 (random initialization)
    M["HeteroNearVsCount"] = f"{d('near', 'count', 'synth_hetero:2', 100, T=5):+.1f}"
    M["HeteroNearVsCountSixteen"] = f"{d('near', 'count', 'synth_hetero:16', 100, T=5):+.1f}"
    M["HeteroNearVsCountKmpp"] = f"{d('near', 'count', 'synth_hetero:2', 100, init='kmpp', T=5):+.1f}"
    # sensitivity (seeds 0-9 when available)
    sd = set(range(10))
    M["DeepTauOne"] = f"{d('near_tau1', 'count', 'deep1m', 16384, T=2, seeds=sd):+.1f}"
    M["KddWOneTrain"] = f"{-d('near_w1', 'count', 'kdd_z', 1024, T=5, metric='train_mse', seeds=sd):.0f}"
    M["KddNearTrainFive"] = f"{-d('near', 'count', 'kdd_z', 1024, T=5, metric='train_mse', seeds=sd):.0f}"
    # --- Review round 1 additions
    tr = [-d("near", "exact", "kdd_z", k, metric="train_mse") for k in (1024, 4096)]
    M["ExactTrainKddLow"], M["ExactTrainKddHigh"] = f"{min(tr):.0f}", f"{max(tr):.0f}"
    te = [d("near", "exact", "kdd_z", k) for k in (1024, 4096)]
    M["ExactTestKddLow"], M["ExactTestKddHigh"] = f"{-max(te):.1f}", f"{-min(te):.1f}"
    # how much lower the estimate's test MSE is than near's (relative to near), over the full grid
    msd_ex = [(-d("exact", "near", "msd_z", k, init=init, T=T), init, k, T) for init in ("random", "kmpp")
              for k in ks for T in TS if I.get("exact", "msd_z", k, init, T) and I.get("near", "msd_z", k, init, T)]
    v, ii, kk, TT = max(msd_ex)
    M["ExactBetterMsdMax"] = f"{v:.1f}"
    M["ExactBetterMsdWhere"] = f"{'k-means++' if ii == 'kmpp' else 'random'} initialization, $k={kk}$, {TT} epochs"
    worst = []
    for init in ("random", "kmpp"):
        for k in ks:
            for T in TS:
                if I.get("near", "msd_z", k, init, T) and I.get("count", "msd_z", k, init, T):
                    a_, b_ = paired(I.get("near", "msd_z", k, init, T), I.get("count", "msd_z", k, init, T))
                    worst.append((rel_change(a_, b_), wtl(a_, b_)[2], len(a_), init, k, T))
    v, losses, n, init, kk, TT = max(worst)
    M["MsdWorstAll"] = f"{v:+.1f}"
    M["MsdWorstLosses"] = f"{losses}/{n}"
    M["MsdWorstWhere"] = f"{'k-means++' if init == 'kmpp' else 'random'} initialization, $k={kk}$, {TT} epochs"
    noab = []
    for ds in held:
        for init in ("random", "kmpp"):
            for k in ks:
                for T in TS:
                    if I.get("near_noab", ds, k, init, T) and I.get("near", ds, k, init, T):
                        a_, b_ = paired(I.get("near_noab", ds, k, init, T), I.get("near", ds, k, init, T))
                        noab.append(abs(rel_change(a_, b_)))
    M["NoabMaxAbs"] = f"{max(noab):.2f}" if noab else "--"
    M["NoabMedianAbs"] = f"{float(np.median(noab)):.3f}" if noab else "--"
    # how often the abandonment clause acts: runs of Distance longer than W = 2 epochs on the held-out datasets
    ab_runs = ab_hit = ab_rem = rem = 0
    for (arm, ds, k, init, T), seeds in I.em.items():
        if arm == "near" and ds in HELD and T > 2:
            for r in seeds.values():
                ab_runs += 1
                ab_hit += int(r.get("n_abandoned", 0) > 0)
                ab_rem += int(r.get("n_abandoned", 0))
                rem += int(r.get("n_splits", 0))
    M["AbandonRuns"] = f"{ab_runs:,}".replace(",", "{,}")
    M["AbandonRunsHit"] = f"{ab_hit:,}".replace(",", "{,}")
    M["AbandonRemovals"] = f"{ab_rem:,}".replace(",", "{,}")
    M["AbandonAllRemovals"] = f"{rem:,}".replace(",", "{,}")
    for ds, tag in (("kdd_z", "Kdd"), ("census_z", "Census"), ("msd_z", "Msd"), ("geonames", "Geo"),
                    ("covtype_z", "Cov"), ("deep1m", "Deep")):
        kk = 16384 if ds == "deep1m" else 1024
        bins, h = ratio_hist(I, ds, kk)
        if h.sum():
            # bucket i holds [bins[i-1], bins[i]); the bucket containing 4 straddles it, so report an interval
            b4 = int(np.searchsorted(bins, 4.0, side="right"))
            hi = h[b4:].sum() / h.sum()
            lo = h[b4 + 1:].sum() / h.sum()
            M[f"FracAboveFour{tag}"] = f"{100 * lo:.0f}--{100 * hi:.0f}" if round(100 * lo) != round(100 * hi) else f"{100 * hi:.0f}"
    offs = []
    for ds in ("deep1m", "sift1m", "spacev1m"):
        for T in (1, 2, 3, 5):
            if I.get("off", ds, 16384, "random", T):
                offs.append(d("off", "count", ds, 16384, T=T))
    M["EmbOffMin"], M["EmbOffMax"] = f"{min(offs):.1f}", f"{max(offs):.1f}"
    t2 = {s_: r for s_, r in I.get("near_tau2", "deep1m", 16384, "random", 2).items() if s_ < 10}
    M["DeepTauTwoProt"] = str(sum(r.get("n_protected", 0) for r in t2.values()))
    M["DeepTauTwoRuns"] = str(len(t2))
    M["DeepTauTwoDelta"] = f"{d('near_tau2', 'count', 'deep1m', 16384, T=2, seeds=sd):+.2f}"
    tdir = os.path.join(paper, "..", "results", "timing")
    tpath = os.environ.get("DISTANCE_REVIVAL_TIMING") or next(
        (os.path.join(tdir, f) for f in ("timing_v3.jsonl", "timing_v2.jsonl") if os.path.exists(os.path.join(tdir, f))), "")
    if tpath and os.path.exists(tpath):
        TR = [json.loads(l) for l in open(tpath)]
        tm = defaultdict(list)
        byrep = defaultdict(dict)
        for r in TR:
            tm[(r["dataset"], r["arm"])].append(r["seconds"])
            byrep[(r["dataset"], r["rep"])][r["arm"]] = r["seconds"]
        for ds, tag in (("deep1m", "Deep"), ("kdd_z", "Kdd")):
            for arm, at in (("library", "Lib"), ("near", "Near"), ("exact", "Exact")):
                if tm[(ds, arm)]:
                    M[f"Time{tag}{at}"] = f"{float(np.median(tm[(ds, arm)])):.1f}"
        nr = [d_["near"] / d_["library"] for d_ in byrep.values() if "near" in d_ and "library" in d_]
        er = [100 * (float(np.median(tm[(ds, "exact")])) / float(np.median(tm[(ds, "library")])) - 1)
              for ds in ("deep1m", "kdd_z") if tm[(ds, "exact")] and tm[(ds, "library")]]
        if nr:
            M["NearRatioMin"], M["NearRatioMax"] = f"{min(nr):.2f}", f"{max(nr):.2f}"
            M["TimingReps"] = str(max(r["rep"] for r in TR) + 1)
        if er:
            M["ExactOverMin"], M["ExactOverMax"] = f"{min(er):.0f}", f"{max(er):.0f}"
    dpath = os.environ.get("DISTANCE_REVIVAL_DIAG", os.path.join(paper, "..", "results", "diag", "msd_z_k4096_T10_s0.json"))
    if not os.path.exists(dpath) and os.path.exists(dpath + ".gz"):
        dpath += ".gz"
    if os.path.exists(dpath):
        import gzip
        dg = json.load(gzip.open(dpath, "rt") if dpath.endswith(".gz") else open(dpath))["summary"]
        M["DiagMsdCand"] = f"{dg['candidates']:,}".replace(",", "{,}")
        M["DiagMsdProt"] = f"{dg['protected_by_bound']:,}".replace(",", "{,}")
        M["DiagMsdAgree"] = f"{100 * (1 - dg['protected_but_exact_le4'] / dg['protected_by_bound']):.0f}"
        M["DiagMsdRatio"] = f"{dg['median_bound_over_exact_protected']:.2f}"
        M["DiagMsdMedianUtil"] = f"{dg.get('median_exact_protected', dg['quantiles_exact'][1]):.0f}"
    lines = ["% generated by code/experiments/analyze.py -- do not edit"]
    for key, v in M.items():
        if "nan" in v:
            v = "\\tbd{" + key + "}"
        lines.append(f"\\newcommand{{\\{key}}}{{{v}}}")
    return M, lines


def main():
    res, paper = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(paper, "tables"), exist_ok=True)
    os.makedirs(os.path.join(paper, "figures"), exist_ok=True)
    R = load(res)
    I = Index(R)
    md = [f"# Summary of {res} ({len(R)} records)\n"]
    N = {}

    # E1 main tables. Holm families as in the protocol ("within each experiment"): all held-out
    # comparisons (near vs count and near vs off, both initializations, every number of epochs and k) form one family; the
    # development datasets form a second family.
    tabs = {}
    for init in ("random", "kmpp"):
        for T in (1, 2, 3, 5, 10):
            tabs[(init, T)] = (tabular_table(I, HELD, init, T, None), tabular_table(I, DEV, init, T, None))
    for fam in (0, 1):
        rows = [r for v in tabs.values() for r in v[fam]]
        ps = holm([r["p_cnt"] for r in rows] + [r["p_off"] for r in rows]) if rows else []
        for i, r in enumerate(rows):
            r["p_cnt_holm"], r["p_off_holm"] = ps[i], ps[len(rows) + i]
        N["HolmFamilySize" + ("Held" if fam == 0 else "Dev")] = 2 * len(rows)
    for (init, T), (held, dev) in tabs.items():
            if not held and not dev:
                continue
            write_tabular_tex(held, dev, T, init, os.path.join(paper, "tables", f"tab_{init}_T{T}.tex"))
            md.append(f"\n## E1 {init} start, T={T}: test MSE / Lloyd@25 (median), near vs count / off\n")
            md.append("| dataset | k | count | off | near | exact | near vs count | W/T/L | p(Holm) | near vs off | W/T/L | p(Holm) | prot | splits cnt->near |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in held + dev:
                md.append(f"| {NAME[r['ds']]}{' (dev)' if r['ds'] in DEV else ''} | {r['k']} | "
                          + " | ".join(f"{r['ratio'][a]:.3f}" for a in ("count", "off", "near", "exact"))
                          + f" | {r['d_cnt']:+.2f}% | {'/'.join(map(str, r['wtl_cnt']))} | {r.get('p_cnt_holm', r['p_cnt']):.3g}"
                          + f" | {r['d_off']:+.2f}% | {'/'.join(map(str, r['wtl_off']))} | {r.get('p_off_holm', r['p_off']):.3g}"
                          + f" | {r['prot']:.0f} | {r['splits_cnt']:.0f}->{r['splits_near']:.0f} |")
            if init == "random" and T == 2 and held:
                N["HeldMedianGainVsCount"] = float(np.median([-r["d_cnt"] for r in held]))
                N["HeldMaxGainVsCount"] = float(np.max([-r["d_cnt"] for r in held]))
                N["HeldWorstVsOff"] = float(np.max([r["d_off"] for r in held]))
                N["HeldBeatLloydNear"] = sum(r["ratio"]["near"] < 1 for r in held)
                N["HeldBeatLloydCount"] = sum(r["ratio"]["count"] < 1 for r in held)
                N["HeldConfigs"] = len(held)
    for init in ("random", "kmpp"):
        rows = passes_table(I, HELD + DEV, init, os.path.join(paper, "tables", f"passes_{init}.tex"))
        md.append(f"\n## Passes to reach Lloyd@25 test MSE ({init} start)\n")
        md.append("| dataset | k | count | off | near | exact |")
        md.append("|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {NAME[r['ds']]} | {r['k']} | " + " | ".join(fmt(r[a], 1) for a in ("count", "off", "near", "exact")) + " |")

    # E2 embeddings
    emb = embedding_rows(I)
    md.append("\n## E2 embeddings (k=16384): identical runs near vs count, ratios to Lloyd@25\n")
    md.append("| dataset | T | identical/runs | protections (sum) | splits | count | off | near |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in emb:
        md.append(f"| {NAME[r['ds']]} | {r['T']} | {r['identical']}/{r['runs']} | {r['prot']} | {r['splits']:.0f} | "
                  f"{r['count']:.4f} | {r['off']:.4f} | {r['near']:.4f} |")
    if emb:
        N["EmbIdentical"] = sum(r["identical"] for r in emb)
        N["EmbRuns"] = sum(r["runs"] for r in emb)
        N["EmbProtections"] = sum(r["prot"] for r in emb)

    # exact vs near
    md.append("\n## near vs exact utility (random start): median % change of near vs exact, time ratio\n")
    md.append("| dataset | k | T | test | train | W/T/L | time near/exact |")
    md.append("|---|---|---|---|---|---|---|")
    for ds in HELD + DEV + EMB:
        for k in ((16384,) if ds in EMB else (256, 1024, 4096)):
            for T in (2, 5):
                a, b = I.get("near", ds, k, "random", T), I.get("exact", ds, k, "random", T)
                if not a or not b:
                    continue
                x, y = paired(a, b)
                xt, yt = paired(a, b, "train_mse")
                tn, te = paired(a, b, "seconds")
                md.append(f"| {NAME[ds]} | {k} | {T} | {rel_change(x, y):+.2f}% | {rel_change(xt, yt):+.2f}% | "
                          f"{'/'.join(map(str, wtl(x, y)))} | {np.median(tn / te):.2f} |")

    # imbalance and tail
    md.append("\n## Imbalance and tail error (random start, T=2): near vs count\n")
    md.append("| dataset | k | imbalance count | imbalance near | tail count | tail near | max err count | max err near |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ds in HELD + DEV:
        for k in (256, 1024, 4096):
            c, n = I.get("count", ds, k, "random", 2), I.get("near", ds, k, "random", 2)
            if not c or not n:
                continue
            md.append(f"| {NAME[ds]} | {k} | {med([r['imbalance'] for r in c.values()]):.2f} | "
                      f"{med([r['imbalance'] for r in n.values()]):.2f} | {med([r['tail_mse'] for r in c.values()]):.4g} | "
                      f"{med([r['tail_mse'] for r in n.values()]):.4g} | {med([r['max_err'] for r in c.values()]):.4g} | "
                      f"{med([r['max_err'] for r in n.values()]):.4g} |")

    # sensitivity
    sens = sensitivity(I)
    md.append("\n## Sensitivity (random start, 5 seeds): median % change in test (train) vs count\n")
    for (ds, T), row in sens.items():
        md.append(f"- {NAME[ds]} k, T={T}: " + "; ".join(f"{a}: {v['test']:+.2f}% ({v['train']:+.2f}%)" for a, v in row.items()))

    # synthetic
    md.append("\n## Synthetic (T=5): far groups covered (mean) / median MSE\n")
    for init in ("random", "kmpp"):
        md.append(f"\n**{init}**\n")
        md.append("| arm | " + " | ".join(f"m={m}" for m in (50, 100, 150, 250, 400)) + " | hetero d=2 | hetero d=16 |")
        md.append("|---|" + "---|" * 7)
        for arm in ("lloyd", "count", "off", "near", "near_noab", "exact"):
            cells = []
            for m in (50, 100, 150, 250, 400):
                ds = f"synth_far:{m}"
                recs = (I.lloyd(ds, 105, init) if arm == "lloyd" else I.get(arm, ds, 105, init, 5)).values()
                recs = list(recs)
                cells.append(f"{np.mean([r['far_covered'] for r in recs]):.1f} / {med([r['test_mse'] for r in recs]):.3f}"
                             if recs else "-")
            for d in (2, 16):
                ds = f"synth_hetero:{d}"
                recs = list((I.lloyd(ds, 100, init) if arm == "lloyd" else I.get(arm, ds, 100, init, 5)).values())
                cells.append(f"{med([r['test_mse'] for r in recs]):.4g}" if recs else "-")
            md.append(f"| {arm} | " + " | ".join(cells) + " |")

    try:
        tex_tables(I, paper, emb, sens, N)
    except Exception as e:
        md.append(f"\n(tex table error: {e!r})")

    # figures
    try:
        N["RatioStats"] = fig_ratio_cdf(I, os.path.join(paper, "figures", "ratio_cdf.pdf"))
        fig_curves(I, os.path.join(paper, "figures", "curves.pdf"))
    except Exception as e:      # figures are best effort while results are partial
        md.append(f"\n(figure error: {e})")

    try:
        M, lines = text_numbers(I, paper)
        emb_lines = [f"\\newcommand{{\\EmbIdentical}}{{{N.get('EmbIdentical', '--')}}}",
                     f"\\newcommand{{\\EmbRuns}}{{{N.get('EmbRuns', '--')}}}",
                     f"\\newcommand{{\\HolmFamilySizeHeld}}{{{N.get('HolmFamilySizeHeld', '--')}}}",
                     f"\\newcommand{{\\HolmFamilySizeDev}}{{{N.get('HolmFamilySizeDev', '--')}}}"]
        open(os.path.join(paper, "numbers.tex"), "w").write("\n".join(lines + emb_lines) + "\n")
        md.append("\n## Text macros\n")
        md += [f"- {k}: {v}" for k, v in M.items()]
    except Exception as e:
        import traceback
        md.append(f"\n(text number error: {traceback.format_exc()})")

    open(os.path.join(res, "summary.md"), "w").write("\n".join(md) + "\n")
    json.dump(N, open(os.path.join(res, "numbers.json"), "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
