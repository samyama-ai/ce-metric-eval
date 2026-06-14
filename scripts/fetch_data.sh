#!/usr/bin/env bash
# Fetch the STATS-CEB + job-light benchmark (true cardinalities, sub-plan SQL, 4 released
# estimator outputs) into ./data/. ~50MB shallow clone. Run from the repo root.
set -euo pipefail
mkdir -p data && cd data
BENCH_COMMIT=670cb8d4bf4cbfa32f94fdf17f33973d3fd67d1b   # pinned for reproducibility (the version we used)
if [ ! -d End-to-End-CardEst-Benchmark ]; then
  git clone https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark.git
  git -C End-to-End-CardEst-Benchmark checkout "$BENCH_COMMIT"
fi
echo "data ready. Next: python -m ce_metric_eval.workload 4   (computes STATS-CEB true cardinalities, ~10-20 min, cached)"
