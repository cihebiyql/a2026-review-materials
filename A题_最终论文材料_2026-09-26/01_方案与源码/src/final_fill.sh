#!/bin/bash
# 终扫补漏: 对所有缺输出的 (case,Q,N) 跑合规求解(logits 自动回退兜底)
# 本脚本及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
cd $HOME/shumo/a2026_reinforce/n5_push
export A2026_ROOT=$HOME/shumo/a2026_reinforce
export A2026_ATT=$HOME/shumo/att
export A2026_SC=$HOME/shumo/a2026/results/singlecore
export A2026_LOGITS=$PWD/reinforce_v5
export A2026_COUT=$PWD/compliant_out
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
P=$HOME/.conda/envs/py311/bin/python
: > fill_todo.txt
for q in 1 2 3; do
  for f in reinforce_v5/*_q${q}_logits.json reinforce_v5/*_q$((3-q%3))_logits.json; do :; done
  # 该问有本问或可回退 logits 的案例(去重)
  for f in reinforce_v5/*_q${q}_logits.json; do
    c=$(basename $f | sed "s/_q${q}_logits.json//")
    for N in 2 3 4 5; do
      [ -f compliant_out/${c}_q${q}_N${N}_compliant.json ] || echo "$c $q $N"
    done
  done
done | sort -u > fill_todo.txt
wc -l < fill_todo.txt
cat fill_todo.txt | xargs -P 24 -L 1 bash -c 'c=$0; q=$1; N=$2; $P solve_compliant.py $c $q $N --budget_s 480 2>>fill_err.log | tail -1'
echo FILL_DONE $(date)
