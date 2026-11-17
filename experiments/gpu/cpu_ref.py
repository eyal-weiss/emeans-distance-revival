"""The paper's CPU (NumpyEmeans) results for the GPU validation configs (random starts, paired by
seed): python experiments/gpu/cpu_ref.py results/runs.jsonl.gz results/gpu/cpu_ref.json"""
import gzip, json, collections, statistics as st, sys
R = collections.defaultdict(dict)
for l in gzip.open(sys.argv[1], "rt"):
    r = json.loads(l)
    if r.get("init") != "random" or r.get("arm") not in ("count", "near", "off") or r.get("test_mse") is None:
        continue
    R[(r["dataset"], r["k"], r["T"])][(r["seed"], r["arm"])] = r["test_mse"]
want = [("kdd_z",1024,2),("kdd_z",4096,2),("shuttle_z",256,2),("shuttle_z",1024,2),("geonames",1024,2),("geonames",4096,2),
        ("power_z",1024,2),("power_z",4096,2),("msd_z",1024,2),("spacev1m",16384,1),("spacev1m",16384,2),
        ("deep1m",16384,1),("deep1m",16384,2)]
out = {}
for key in want:
    d = R.get(key, {})
    seeds = sorted(s for s, a in d if a == "count" and (s, "near") in d)
    if not seeds:
        print(key, "no CPU records"); continue
    near = st.mean(d[(s, "near")] / d[(s, "count")] - 1 for s in seeds) * 100
    offs = [s for s in seeds if (s, "off") in d]
    off = st.mean(d[(s, "off")] / d[(s, "count")] - 1 for s in offs) * 100 if offs else None
    w = sum(d[(s, "near")] < d[(s, "count")] for s in seeds); t = sum(d[(s, "near")] == d[(s, "count")] for s in seeds)
    out["|".join(map(str, key))] = dict(n=len(seeds), near_pct=near, off_pct=off, wtl=f"{w}/{t}/{len(seeds)-w-t}")
    print(key, f"n={len(seeds)} near {near:+.2f}% ({w}/{t}/{len(seeds)-w-t})  off " + (f"{off:+.2f}%" if off is not None else "-"))
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "cpu_ref.json", "w"), indent=1)
