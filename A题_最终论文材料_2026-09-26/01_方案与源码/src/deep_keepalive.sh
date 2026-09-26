#!/bin/bash
for q in 3 2; do
  for i in $(seq 1 30); do
    grep -q "SEQ_DONE" mechA_deep_seq_q${q}b.log 2>/dev/null && break
    py -3.11 "C:/shumo_live/02_求解/A题_2026/n5_push/mechA_deep_seq.py" $q >> mechA_deep_seq_q${q}b.log 2>&1
    sleep 2
  done
done
echo "KEEPALIVE_DONE" >> mechA_deep_seq_q3b.log
