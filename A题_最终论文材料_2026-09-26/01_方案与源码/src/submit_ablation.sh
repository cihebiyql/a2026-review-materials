#!/bin/bash
#SBATCH -J v5abl
#SBATCH -p amd_256q
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=230G
#SBATCH -t 300
#SBATCH -o logs/abl_%a.out
#SBATCH -e logs/abl_%a.err
export A2026_ROOT=$HOME/shumo/a2026_reinforce
export A2026_ATT=$HOME/shumo/att
export A2026_SC=$HOME/shumo/a2026/results/singlecore
export A2026_LOGITS=$A2026_ROOT/n5_push/reinforce_v5
export A2026_COUT=$A2026_ROOT/n5_push/compliant_out
export A2026_ABL=$A2026_ROOT/n5_push/ablation_out
export V5PY=$HOME/.conda/envs/py311/bin/python
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd $A2026_ROOT/n5_push
$V5PY v5_ablation.py $1 --workers 24
