#!/bin/bash
# node23 缺口并行求解: 与训练并进, 循环扫描已有 logits 的 (case,q,N) 缺口
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
for round in 1 2 3 4 5 6 7 8; do
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
  cat fill_todo.txt | xargs -P 6 -L 1 bash -c 'c=$0; q=$1; N=$2; $P n5_push/solve_compliant.py $c $q $N --budget_s 480 2>>n5_push/compliant_out/fill_err.log | tail -1' 
  sleep 60
done
echo N23_FILL_DONE $(date)
