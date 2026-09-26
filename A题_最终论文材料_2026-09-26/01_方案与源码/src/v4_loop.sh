#!/bin/bash
while read c; do
  [ -z "$c" ] && continue
  grep -q "^$c:" v4_local_q2c.log 2>/dev/null && continue
  py -3.11 "C:/shumo_live/02_求解/A题_2026/n5_push/mechA_v4.py" 2 "$c" >> v4_local_q2c.log 2>&1 || echo "$c: CRASHED" >> v4_local_q2c.log
done < "C:/shumo_live/02_求解/A题_2026/n5_push/v4_targets3.txt"
echo "LOOP_DONE" >> v4_local_q2c.log
