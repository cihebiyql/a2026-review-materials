#!/bin/bash
# 豁免批2: node23 双模式失败的 q3 例, 官方评估器通道求解
# 本脚本及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
cd $HOME/shumo/a2026_reinforce/n5_push
export A2026_ROOT=$HOME/shumo/a2026_reinforce
export A2026_ATT=$HOME/shumo/att
export A2026_SC=$HOME/shumo/a2026/results/singlecore
export A2026_LOGITS=$PWD/reinforce_v5
export A2026_COUT=$PWD/compliant_out
P=$HOME/.conda/envs/py311/bin/python; export P
for c in case_014 case_018 case_021 case_030 case_031 case_033 case_038 case_039 case_040 case_041 case_042 case_043 case_053 case_054 case_058 case_059 case_060 ; do
  for N in 2 3 4 5; do
    [ -f compliant_out/${c}_q3_N${N}_compliant.json ] || $P solve_compliant.py $c 3 $N --budget_s 480 --samples 24 --official-eval 2>>exempt2_err.log | tail -1
  done
done
echo EXEMPT2_DONE $(date)
