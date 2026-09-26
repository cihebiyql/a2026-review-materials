#!/bin/bash
cd "C:/shumo_live/02_求解/A题_2026/n5_push"
for q in 3; do
  for i in $(seq 1 100); do
    c=$(printf "case_%03d" $i)
    grep -q "\"case\": \"$c\", \"q\": $q" bytes_audit_one.log 2>/dev/null && continue
    AUD_ONE=$q AUD_CASE=$c py -3.11 audit_bytes_all.py >> bytes_audit_one.log 2>&1 || echo "{\"case\": \"$c\", \"q\": $q, \"status\": \"CRASHED\"}" >> bytes_audit_one.log
  done
done
echo "AUDIT_LOOP_DONE" >> bytes_audit_one.log
