#!/bin/bash
cd "C:/shumo_live/02_求解/A题_2026/n5_push"
while read c; do
  [ -z "$c" ] && continue
  grep -q "^$c:" mechA_p3_r2.log 2>/dev/null && continue
  py -3.11 "C:/shumo_live/02_求解/A题_2026/n5_push/mechA_p3_scan.py" one "$c" >> mechA_p3_r2.log 2>&1 || echo "$c: CRASHED(skipped)" >> mechA_p3_r2.log
done < mechA_p3_remaining.txt
echo "LOOP_DONE" >> mechA_p3_r2.log
