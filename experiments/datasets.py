"""Dataset preparation and loading.

Prepared layout: $DISTANCE_REVIVAL_DATA/<name>/{train.npy, test.npy, meta.json[, labels_train.npy]}

Protocol for every tabular dataset (fixed before any run, 2026-11-11):
  1. parse; one-hot encode categorical text columns (KDD only);
  2. remove exact duplicate rows (KDD Cup 1999 is ~75% duplicates; held-out rows must not be
     copies of training rows);
  3. hold out 20,000 rows (RandomState(1234) permutation); the held-out set is split in halves
     with RandomState(1234) (validation, test) as in the e-means paper; we report the test half;
  4. scale with training statistics: z-score ("z"), min-max ("mm"), or none ("raw").
Raw downloads are untrusted data: they are only parsed (never executed); zip members are streamed.

CLI (run with python -I):
  python -I datasets.py prepare <name> --raw RAW_DIR --out DATA_ROOT
  names: covtype_z covtype_mm kdd_z census_z msd_z geonames power_z shuttle_z sift1m
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import zipfile

import numpy as np

HOLDOUT = 20_000
SEED = 1234


def data_root():
    return os.environ.get("DISTANCE_REVIVAL_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


# ------------------------------------------------------------------------------ loading
_CACHE = {}


def load(name, seed=0):
    """(train, test, info). Arrays are float32, C-contiguous, memory-mapped when prepared on disk."""
    if name.startswith("synth_far:") or name.startswith("synth_hetero:"):
        import synthetic
        return synthetic.make(name, seed)
    if name not in _CACHE:
        d = os.path.join(data_root(), name)
        train = np.load(os.path.join(d, "train.npy"), mmap_mode="r")
        test = np.load(os.path.join(d, "test.npy"))
        info = json.load(open(os.path.join(d, "meta.json")))
        _CACHE[name] = (train, test, info)
    return _CACHE[name]


# ------------------------------------------------------------------------------ helpers
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _zip_member(zpath, suffix):
    z = zipfile.ZipFile(zpath)
    names = [n for n in z.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"{zpath}: expected one member ending in {suffix}, got {names}")
    return z.open(names[0])


def _halves(evaluation):
    perm = np.random.RandomState(SEED).permutation(evaluation.shape[0])
    n_val = int(round(0.5 * evaluation.shape[0]))
    return evaluation[perm[:n_val]], evaluation[perm[n_val:]]


def _finish(X, out, name, scaling, meta, labels=None, n_train=None):
    """dedup -> split -> scale -> save. X float64 (n, d)."""
    n_raw = len(X)
    X, first = np.unique(X, axis=0, return_index=True)
    order = np.sort(first)                      # keep original row order among unique rows
    pos = np.argsort(first)
    X = X[pos]
    if labels is not None:
        labels = labels[order]
    n_unique = len(X)
    perm = np.random.RandomState(SEED).permutation(n_unique)
    if n_train is not None:                     # subsample (GeoNames)
        perm = perm[:n_train + HOLDOUT]
    held_idx, train_idx = perm[:HOLDOUT], perm[HOLDOUT:]
    train, held = X[train_idx], X[held_idx]
    if scaling == "z":
        mu, sd = train.mean(0), train.std(0)
    elif scaling == "mm":
        mu, sd = train.min(0), train.max(0) - train.min(0)
    else:
        mu, sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
    sd = np.where(sd == 0, 1.0, sd)
    train = np.ascontiguousarray(((train - mu) / sd).astype(np.float32))
    held = ((held - mu) / sd).astype(np.float32)
    _, test = _halves(held)
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "train.npy"), train)
    np.save(os.path.join(d, "test.npy"), np.ascontiguousarray(test))
    if labels is not None:
        np.save(os.path.join(d, "labels_train.npy"), labels[train_idx])
    meta.update(name=name, scaling=scaling, n_raw=int(n_raw), n_unique=int(n_unique),
                n_train=int(len(train)), n_test=int(len(test)), d=int(train.shape[1]),
                holdout=HOLDOUT, seed=SEED,
                train_sha256=_sha256(os.path.join(d, "train.npy")),
                test_sha256=_sha256(os.path.join(d, "test.npy")))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if not k.endswith("sha256")}))


# ------------------------------------------------------------------------------ parsers
def prep_covtype(raw, out, scaling):
    X = np.load(os.path.join(raw, "covtype_X.npy")).astype(np.float64)
    y = np.load(os.path.join(raw, "covtype_y.npy"))
    meta = dict(source="UCI Covertype via sklearn.datasets.fetch_covtype", license="CC BY 4.0",
                raw_sha256=_sha256(os.path.join(raw, "covtype_X.npy")))
    _finish(X, out, f"covtype_{scaling}", scaling, meta, labels=y)


KDD_CATEGORICAL = (1, 2, 3)        # protocol_type, service, flag


def prep_kdd(raw, out):
    zpath = os.path.join(raw, "kddcup.data.gz")    # figshare 5976045 (scikit-learn's source)
    with gzip.open(zpath) as f:
        text = f.read().decode("ascii")
    rows = [r.split(",") for r in text.strip().split("\n")]
    del text
    vocab = {c: sorted({r[c] for r in rows}) for c in KDD_CATEGORICAL}
    index = {c: {v: i for i, v in enumerate(vocab[c])} for c in KDD_CATEGORICAL}
    numeric_cols = [c for c in range(41) if c not in KDD_CATEGORICAL]
    width = len(numeric_cols) + sum(len(vocab[c]) for c in KDD_CATEGORICAL)
    X = np.zeros((len(rows), width), dtype=np.float64)
    labels = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        X[i, :len(numeric_cols)] = [float(r[c]) for c in numeric_cols]
        off = len(numeric_cols)
        for c in KDD_CATEGORICAL:
            X[i, off + index[c][r[c]]] = 1.0
            off += len(vocab[c])
        labels[i] = r[41].rstrip(".")
    labels = labels.astype("U32")
    meta = dict(source="KDD Cup 1999 full data (kddcup.data.corrected, 4.9M rows), figshare file 5976045", license="CC BY 4.0",
                raw_sha256=_sha256(zpath), one_hot={str(c): vocab[c] for c in KDD_CATEGORICAL})
    _finish(X, out, "kdd_z", "z", meta, labels=labels)


def prep_census(raw, out):
    zpath = os.path.join(raw, "us+census+data+1990.zip")
    with _zip_member(zpath, "USCensus1990.data.txt") as f:
        tf = io.TextIOWrapper(f, encoding="ascii")
        header = tf.readline().strip().split(",")
        X = np.loadtxt(tf, delimiter=",", dtype=np.float64)
    assert header[0] == "caseid", header[:3]
    X = X[:, 1:]                                 # drop caseid
    meta = dict(source="UCI US Census Data (1990), USCensus1990.data.txt", license="CC BY 4.0",
                raw_sha256=_sha256(zpath), columns=header[1:])
    _finish(X, out, "census_z", "z", meta)


def prep_msd(raw, out):
    zpath = os.path.join(raw, "yearpredictionmsd.zip")
    with _zip_member(zpath, "YearPredictionMSD.txt") as f:
        A = np.loadtxt(io.TextIOWrapper(f, encoding="ascii"), delimiter=",", dtype=np.float64)
    meta = dict(source="UCI YearPredictionMSD (year column dropped)", license="CC BY 4.0",
                raw_sha256=_sha256(zpath))
    _finish(A[:, 1:], out, "msd_z", "z", meta, labels=A[:, 0].astype(np.int32))


def prep_geonames(raw, out, n_train=1_000_000):
    zpath = os.path.join(raw, "allCountries.zip")
    lat, lon, cc = [], [], []
    with _zip_member(zpath, "allCountries.txt") as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            p = line.split("\t")
            lat.append(float(p[4]))
            lon.append(float(p[5]))
            cc.append(p[8])
    X = np.column_stack([np.asarray(lat), np.asarray(lon)])
    meta = dict(source="GeoNames allCountries.txt (latitude, longitude in degrees)", license="CC BY 4.0",
                raw_sha256=_sha256(zpath), subsample_train=n_train)
    _finish(X, out, "geonames", "raw", meta, labels=np.asarray(cc, dtype="U2"), n_train=n_train)


def prep_power(raw, out):
    zpath = os.path.join(raw, "individual+household+electric+power+consumption.zip")
    rows = []
    with _zip_member(zpath, "household_power_consumption.txt") as f:
        tf = io.TextIOWrapper(f, encoding="ascii")
        header = tf.readline().strip().split(";")
        for line in tf:
            p = line.strip().split(";")
            if len(p) != 9 or "?" in p or "" in p:
                continue                          # rows with missing values are dropped
            rows.append([float(v) for v in p[2:]])
    X = np.asarray(rows, dtype=np.float64)
    meta = dict(source="UCI Individual Household Electric Power Consumption (7 numeric columns)",
                license="CC BY 4.0", raw_sha256=_sha256(zpath), columns=header[2:])
    _finish(X, out, "power_z", "z", meta)


def prep_shuttle(raw, out):
    import subprocess
    zpath = os.path.join(raw, "statlog+shuttle.zip")
    parts = []
    z = zipfile.ZipFile(zpath)
    for stem in ("shuttle.trn", "shuttle.tst"):
        names = [n for n in z.namelist() if n.endswith(stem) or n.endswith(stem + ".Z")]
        assert len(names) == 1, (stem, z.namelist())
        if names[0].endswith(".Z"):                  # Unix compress format: decompress with gzip
            tmp = os.path.join(raw, stem + ".Z")     # our own file name, not the archive's
            with z.open(names[0]) as src, open(tmp, "wb") as dst:
                dst.write(src.read())
            text = subprocess.run(["gzip", "-dc", tmp], check=True, capture_output=True).stdout.decode("ascii")
        else:
            text = z.open(names[0]).read().decode("ascii")
        parts.append(np.array([[float(v) for v in r.split()] for r in text.strip().split("\n")]))
    A = np.vstack(parts)
    meta = dict(source="UCI Statlog (Shuttle), shuttle.trn + shuttle.tst (class column dropped)",
                license="CC BY 4.0", raw_sha256=_sha256(zpath))
    _finish(A[:, :9], out, "shuttle_z", "z", meta, labels=A[:, 9].astype(np.int32))


def prep_sift(raw, out):
    """SIFT 1M: the first 1,000,000 BIGANN base vectors; test = second half (RandomState(1234)) of the
    next 10,000 rows. raw/base.1B.u8bin may be a prefix of the file (>= 8 + 1,010,000 * 128 bytes)."""
    path = os.path.join(raw, "base.1B.u8bin")
    d = int(np.fromfile(path, dtype=np.int32, count=2)[1])
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(1_010_000, d))
    train = np.ascontiguousarray(mm[:1_000_000], dtype=np.float32)
    _, test = _halves(np.asarray(mm[1_000_000:1_010_000], dtype=np.float32))
    dd = os.path.join(out, "sift1m")
    os.makedirs(dd, exist_ok=True)
    np.save(os.path.join(dd, "train.npy"), train)
    np.save(os.path.join(dd, "test.npy"), np.ascontiguousarray(test))
    meta = dict(name="sift1m", source="first 1M rows of BIGANN base.1B.u8bin; test = half of the next 10k rows",
                scaling="raw", n_train=len(train), n_test=len(test), d=d,
                train_sha256=_sha256(os.path.join(dd, "train.npy")), test_sha256=_sha256(os.path.join(dd, "test.npy")))
    json.dump(meta, open(os.path.join(dd, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare"])
    ap.add_argument("name")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.name in ("covtype_z", "covtype_mm"):
        prep_covtype(a.raw, a.out, a.name.split("_")[1])
    elif a.name == "kdd_z":
        prep_kdd(a.raw, a.out)
    elif a.name == "census_z":
        prep_census(a.raw, a.out)
    elif a.name == "msd_z":
        prep_msd(a.raw, a.out)
    elif a.name == "geonames":
        prep_geonames(a.raw, a.out)
    elif a.name == "power_z":
        prep_power(a.raw, a.out)
    elif a.name == "shuttle_z":
        prep_shuttle(a.raw, a.out)
    elif a.name == "sift1m":
        prep_sift(a.raw, a.out)
    else:
        raise KeyError(a.name)


if __name__ == "__main__":
    main()
