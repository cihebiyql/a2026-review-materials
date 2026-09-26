#!/bin/bash
# node23 缺口求解 v3: 常规优先, 崩溃例自动降级 --official-eval
# 本脚本及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
cd /data/qlyu/tmp_shumo/a2026_reinforce
export A2026_ROOT=/data/qlyu/tmp_shumo/a2026_mechA
export A2026_ATT=/data/qlyu/tmp_shumo/a2026/att
export A2026_SC=/data/qlyu/tmp_shumo/a2026/singlecore
export A2026_LOGITS=/data/qlyu/tmp_shumo/a2026_reinforce/n5_push/reinforce_v5
export A2026_COUT=/data/qlyu/tmp_shumo/a2026_reinforce/n5_push/compliant_out
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
P=/data/qlyu/anaconda3/bin/python; export P
one() {
  c=$0; q=$1; N=$2
  f=n5_push/compliant_out/${c}_q${q}_N${N}_compliant.json
  [ -f "$f" ] && return 0
  timeout 540 $P n5_push/solve_compliant.py $c $q $N --budget_s 420 >/dev/null 2>&1
  [ -f "$f" ] && return 0
  timeout 560 $P n5_push/solve_compliant.py $c $q $N --budget_s 420 --samples 24 --official-eval >/dev/null 2>&1
  [ -f "$f" ] && echo "OK $c $q $N" || echo "FAIL $c $q $N"
}
export -f one
export P
for round in 1 2 3 4 5 6; do
  : > fill_todo.txt
  for q in 1 2 3; do
    for f in n5_push/reinforce_v5/*_q${q}_logits.json; do
      c=$(basename $f | sed "s/_q${q}_logits.json//")
      for N in 2 3 4 5; do
        [ -f n5_push/compliant_out/${c}_q${q}_N${N}_compliant.json ] || echo "$c $q $N"
      done
    done
  done | sort -u > fill_todo.txt
  n=$(wc -l < fill_todo.txt)
  echo "ROUND $round todo=$n $(date +%H:%M)"
  [ "$n" -eq 0 ] && break
  cat fill_todo.txt | xargs -P 6 -L 1 bash -c 'one "$@"'
  sleep 30
done
echo N23_FILL3_DONE $(date)
