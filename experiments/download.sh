#!/bin/bash
# Download and prepare every dataset of the paper into $DISTANCE_REVIVAL_DATA (default ./data).
# Raw downloads go to $RAW (default ./raw), one folder per dataset; they are only parsed, never executed.
#   bash experiments/download.sh            (from the repository root)
set -e
here=$(cd "$(dirname "$0")" && pwd)
DATA=${DISTANCE_REVIVAL_DATA:-$PWD/data}
RAW=${RAW:-$PWD/raw}
PY=${PYTHON:-python}
mkdir -p "$DATA" "$RAW"/{kdd,census,msd,geonames,power,shuttle,covtype}

fetch() { [ -s "$2" ] || curl -sSfL -o "$2" "$1"; }
fetch https://ndownloader.figshare.com/files/5976045 "$RAW/kdd/kddcup.data.gz"        # KDD Cup 1999 (scikit-learn's source)
fetch https://archive.ics.uci.edu/static/public/116/us+census+data+1990.zip "$RAW/census/us+census+data+1990.zip"
fetch https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip "$RAW/msd/yearpredictionmsd.zip"
fetch https://download.geonames.org/export/dump/allCountries.zip "$RAW/geonames/allCountries.zip"
fetch https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip "$RAW/power/individual+household+electric+power+consumption.zip"
fetch https://archive.ics.uci.edu/static/public/148/statlog+shuttle.zip "$RAW/shuttle/statlog+shuttle.zip"
# Covertype: scikit-learn's copy, saved as two .npy files
[ -s "$RAW/covtype/covtype_X.npy" ] || $PY -c "
import numpy as np, sys
from sklearn.datasets import fetch_covtype
d = fetch_covtype()
np.save(sys.argv[1] + '/covtype_X.npy', d.data); np.save(sys.argv[1] + '/covtype_y.npy', d.target)" "$RAW/covtype"

cd "$here"
for spec in "kdd_z kdd" "census_z census" "msd_z msd" "geonames geonames" "power_z power" "shuttle_z shuttle" \
            "covtype_z covtype" "covtype_mm covtype"; do
  set -- $spec
  [ -s "$DATA/$1/train.npy" ] || $PY -I datasets.py prepare "$1" --raw "$RAW/$2" --out "$DATA"
done

# Embeddings: the e-means paper's own splits, fetched with its downloader and checked against its
# fingerprints. Needs FAISS's benchs/e_means at the revision we used (2026-11-10):
#   git clone https://github.com/facebookresearch/faiss && git -C faiss checkout 81e083c8ac6ee1024337a843748b08a5071bd228
BENCH=${BENCH:-$PWD/faiss/benchs/e_means}
if [ -d "$BENCH" ]; then
  export EMEANS_DATA=${EMEANS_DATA:-$RAW/paperdata} EMEANS_CACHE=${EMEANS_CACHE:-$RAW/papercache}
  for spec in "deep1b-1M deep1m" "spacev-1M spacev1m" "deep1b-10M deep10m"; do
    set -- $spec
    [ -s "$DATA/$2/train.npy" ] || $PY fetch_paper.py "$BENCH" "$1" "$DATA" "$2"
  done
  # SIFT 1M: the first 1,010,000 rows of BIGANN (range request), prepared by datasets.py
  mkdir -p "$RAW/bigann"
  [ -s "$RAW/bigann/base.1B.u8bin" ] || curl -sSfL -r 0-129280007 -o "$RAW/bigann/base.1B.u8bin" \
      https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
  [ -s "$DATA/sift1m/train.npy" ] || $PY -I datasets.py prepare sift1m --raw "$RAW/bigann" --out "$DATA"
else
  echo "set BENCH to faiss/benchs/e_means (commit 81e083c) to fetch DEEP/SPACEV"
fi
echo "datasets in $DATA"
