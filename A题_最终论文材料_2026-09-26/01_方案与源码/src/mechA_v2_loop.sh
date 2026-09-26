#!/bin/bash
cd "C:/shumo_live/02_求解/A题_2026/n5_push"
LOG=mechA_v2_q$1.log
while read c; do
  [ -z "$c" ] && continue
  grep -q "^$c:" $LOG 2>/dev/null && continue
  py -3.11 "C:/shumo_live/02_求解/A题_2026/n5_push/mechA_v2.py" one $1 "$c" >> $LOG 2>&1 || echo "$c: CRASHED" >> $LOG
done < $2
echo "LOOP_DONE" >> $LOG
