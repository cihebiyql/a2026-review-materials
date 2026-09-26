#!/bin/bash
# v5 指派学习 · 3 分片并行（每分片 16 worker，共 48 核，叠加在跑战役 ~16 核 = 64 核满载）
cd /data/qlyu/tmp_shumo/a2026_reinforce
export A2026_ROOT=/data/qlyu/tmp_shumo/a2026_mechA
export A2026_SC=/data/qlyu/tmp_shumo/a2026/singlecore
export A2026_ATT=/data/qlyu/tmp_shumo/a2026/att
export A2026_ROUT=/data/qlyu/tmp_shumo/a2026_reinforce/out_v5
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
P=/data/qlyu/anaconda3/bin/python
nohup $P reinforce_assign.py 3 case_062,case_029 --iters 50 --N 128 --workers 16 --seed 0 > v5_shard1.log 2>&1 < /dev/null &
nohup $P reinforce_assign.py 3 case_016,case_045 --iters 50 --N 128 --workers 16 --seed 0 > v5_shard2.log 2>&1 < /dev/null &
nohup $P reinforce_assign.py 3 case_084,case_095,case_064 --iters 50 --N 128 --workers 16 --seed 0 > v5_shard3.log 2>&1 < /dev/null &
echo "SHARDS_LAUNCHED $(date)"
