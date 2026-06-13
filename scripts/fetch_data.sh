#!/usr/bin/env bash
# Fetch the STATS-CEB + job-light benchmark (true cardinalities, sub-plan SQL, 4 released
# estimator outputs) into ./data/. ~50MB shallow clone. Run from the repo root.
set -euo pipefail
mkdir -p data && cd data
if [ ! -d End-to-End-CardEst-Benchmark ]; then
  git clone --depth 1 https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark.git
fi
echo "data ready. Next: python -m ce_metric_eval.workload 4   (computes STATS-CEB true cardinalities, ~10-20 min, cached)"
