#!/bin/bash
# case_001 全配置自测: Q1/2/3 × N5..2 × V1-V4, 输出定向 test/
set -e
cd "$(dirname "$0")/.."
export A2026_RESULTS="$(pwd)/test"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$(pwd)/test/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"
for Q in 3 2 1; do
  echo "=== Q$Q ==="
  python3 -u pipeline.py case_001 $Q --nlist 5,4,3,2 --versions v1,v2,v3,v4
done
echo "TEST_RUN_DONE"
