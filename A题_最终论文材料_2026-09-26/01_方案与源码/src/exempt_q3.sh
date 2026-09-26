#!/bin/bash
# 豁免例(014/040/062) q3 合规求解: 官方评估器通道 + q2 logits 回退
cd /root/shumo/a2026_reinforce/n5_push 2>/dev/null || cd $HOME/shumo/a2026_reinforce/n5_push
export A2026_ROOT=$HOME/shumo/a2026_reinforce
export A2026_ATT=$HOME/shumo/att
export A2026_SC=$HOME/shumo/a2026/results/singlecore
export A2026_LOGITS=$PWD/reinforce_v5
export A2026_COUT=$PWD/compliant_out
P=$HOME/.conda/envs/py311/bin/python
for c in case_014 case_040 case_062; do
  for N in 2 3 4 5; do
    $P solve_compliant.py $c 3 $N --budget_s 480 --samples 32 --official-eval
  done
done
echo EXEMPT_ALL_DONE
