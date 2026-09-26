#!/bin/bash
cd "C:/shumo_live/02_求解/A题_2026/n5_push"
while read key; do
  [ -z "$key" ] && continue
  grep -q "\"key\": \"$key\"" baseline_full_metrics.jsonl 2>/dev/null && continue
  ENRICH_ONE="$key" py -3.11 enrich_all.py >> enrich_loop.log 2>&1 || echo "{\"key\": \"$key\", \"status\": \"CRASHED\"}" >> baseline_full_metrics.jsonl
done < <(py -3.11 -c "
import json, os
targets=set()
for fn, algos in (('baseline_results.jsonl',('topo_equal','greedy_balance','heft_like','kl_partition')),('baseline_hinted.jsonl',('shared_input','pipe_balance','merge_greedy')),('baseline_extra.jsonl',('random_search','core_resident'))):
    if not os.path.exists(fn): continue
    for l in open(fn,encoding='utf-8'):
        if not l.strip().startswith('{'): continue
        try: r=json.loads(l)
        except: continue
        if 'sp' in r and r.get('algo') in algos:
            targets.add(f\"{r['case']}|{r['q']}|{r['algo']}\")
for t in sorted(targets): print(t)
")
echo ENRICH_LOOP_DONE >> enrich_loop.log
