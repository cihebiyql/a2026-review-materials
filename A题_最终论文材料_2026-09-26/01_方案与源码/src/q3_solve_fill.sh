#!/bin/bash
# q3 求解补漏: 对所有已有 q3 logits 且未产出的 (case,N) 跑合规求解
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
cat reinforce_v5/*_q3_logits.json | $P - << 'PYEOF' > q3_todo.txt
import sys, json, re, os
buf = sys.stdin.read()
for m in re.finditer(r'\{"case": "(case_\d+)"[^}]*', buf):
    pass
# 逐文件解析更稳: 改为扫目录
PYEOF
rm -f q3_todo.txt
for f in reinforce_v5/*_q3_logits.json; do
  c=$(basename $f | sed 's/_q3_logits.json//')
  for N in 2 3 4 5; do
    [ -f compliant_out/${c}_q3_N${N}_compliant.json ] || echo "$c $N"
  done
done > q3_todo.txt
wc -l < q3_todo.txt
cat q3_todo.txt | xargs -P 24 -L 1 bash -c 'c=$0; N=$1; $P solve_compliant.py $c 3 $N --budget_s 480 2>/dev/null | tail -1'
echo Q3_SOLVE_FILL_DONE $(date)
