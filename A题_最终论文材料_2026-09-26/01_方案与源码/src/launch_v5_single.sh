#!/bin/bash
# v5 单案例单进程发射（每案例独立进程, 规避 FastEval 多实例堆损坏）
# 用法: bash launch_v5_single.sh <case> [workers]
CASE=$1
W=${2:-16}
cd /data/qlyu/tmp_shumo/a2026_reinforce
export A2026_ROOT=/data/qlyu/tmp_shumo/a2026_mechA
export A2026_SC=/data/qlyu/tmp_shumo/a2026/singlecore
export A2026_ATT=/data/qlyu/tmp_shumo/a2026/att
export A2026_ROUT=/data/qlyu/tmp_shumo/a2026_reinforce/out_v5
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
P=/data/qlyu/anaconda3/bin/python
nohup $P reinforce_assign.py 3 $CASE --iters 50 --N 128 --workers $W --seed 0 > v5_$CASE.log 2>&1 < /dev/null &
echo "LAUNCHED $CASE workers=$W pid=$!"
