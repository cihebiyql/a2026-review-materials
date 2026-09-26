#!/bin/bash
cd "C:/shumo_live/02_求解/A题_2026/n5_push"
while read c q; do
  [ -z "$c" ] && continue
  grep -q "\"case\": \"$c\", .*\"K\": 4" mechA_nk_q${q}_N4.jsonl 2>/dev/null && continue
  NK_Q=$q NK_K=4 NK_ONE=$c SHARD=0 NSHARD=1 py -3.11 mechA_nk.py >> nk_run.log 2>&1 || echo "{\"case\": \"$c\", \"q\": $q, \"status\": \"CRASHED\"}" >> mechA_nk_q${q}_N4.jsonl
done < nk_targets.txt
echo "NK4_LOOP_DONE" >> nk_run.log
