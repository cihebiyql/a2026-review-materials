#!/bin/bash
# submit_v5cmp.sh — v5 合规产线(训练+求解) SLURM array, 10 分片
#SBATCH -J v5cmp
#SBATCH -p all
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH -o logs/v5cmp_%a.out
#SBATCH -e logs/v5cmp_%a.err
#SBATCH --array=0-9%8

export A2026_ROOT=$HOME/shumo/a2026_reinforce
export A2026_ATT=$HOME/shumo/att
export A2026_SC=$HOME/shumo/a2026/results/singlecore
export A2026_LOGITS=$A2026_ROOT/n5_push/reinforce_v5
export A2026_COUT=$A2026_ROOT/n5_push/compliant_out
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export V5PY=$HOME/.conda/envs/py311/bin/python
mkdir -p $A2026_COUT logs
cd $A2026_ROOT/n5_push
$V5PY run_shard.py $SLURM_ARRAY_TASK_ID
