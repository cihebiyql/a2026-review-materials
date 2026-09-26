#!/bin/bash
cd "C:/shumo_live/02_求解/A题_2026/n5_push"
while read c; do
  [ -z "$c" ] && continue
  for name in shared_input pipe_balance merge_greedy; do
    for q in 2 3; do
      grep -q "\"$c|q$q|$name\"" baseline_hinted.jsonl 2>/dev/null && continue
      py -3.11 - << PYE >> hinted_loop.log 2>&1 || echo "{\"key\": \"$c|q$q|$name\", \"status\": \"CRASHED\"}" >> baseline_hinted.jsonl
import sys, os
os.environ['HINT_ONE'] = '$c'; os.environ['HINT_Q'] = '$q'; os.environ['HINT_NAME'] = '$name'
exec(open('baseline_hinted.py', encoding='utf-8').read().replace("for i in range(1, 101):", "for i in ([int('$c'.split('_')[1])]):").replace("for q, ev in ((2, ev_p2), (3, ev_p3)):", "for q, ev in ((int('$q'), ev_p2 if int('$q')==2 else ev_p3),):").replace("for name in ('shared_input', 'pipe_balance', 'merge_greedy'):", "for name in ('$name',):"))
PYE
    done
  done
done < <(printf 'case_%03d\n' $(seq 1 100))
echo HINTED_LOOP_DONE >> hinted_loop.log
