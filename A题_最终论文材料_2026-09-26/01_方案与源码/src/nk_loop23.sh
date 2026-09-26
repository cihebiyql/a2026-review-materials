#!/bin/bash
cd "C:/shumo_live/02_求解/A题_2026/n5_push"
for K in 3 2; do
while read c q; do
  [ -z "$c" ] && continue
  grep -q "\"case\": \"$c\", .*\"K\": $K" mechA_nk_q${q}_N${K}.jsonl 2>/dev/null && continue
  NK_Q=$q NK_K=$K NK_ONE=$c SHARD=0 NSHARD=1 py -3.11 mechA_nk.py >> nk_run.log 2>&1 || echo "{\"case\": \"$c\", \"q\": $q, \"status\": \"CRASHED\", \"K\": $K}" >> mechA_nk_q${q}_N${K}.jsonl
done < nk_targets.txt
done
echo "NK32_LOOP_DONE" >> nk_run.log
