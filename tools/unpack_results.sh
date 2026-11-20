#!/bin/bash
# Unpack the gzipped per-problem records so the analyzers can read them (plain copies are git-ignored).
#   bash tools/unpack_results.sh
#   python3 experiments/diag_5k/analyze5k.py --dir results/relational/test5000          # -> report.md there
#   python3 experiments/diag_overlap/analyze.py --dir results/relational/overlap_val1000 # -> report.md there
#   python3 experiments/diag_ablate/analyze_ablate.py --dir results/relational/ablation_test5000  # -> report.md there
set -e
cd "$(dirname "$0")/.."
for d in results/relational/test5000 results/relational/overlap_val1000 results/relational/branchoff_val1000 results/relational/ablation_test5000; do
  for f in "$d"/*.json.gz; do gunzip -kf "$f"; done
  echo "unpacked $d"
done
