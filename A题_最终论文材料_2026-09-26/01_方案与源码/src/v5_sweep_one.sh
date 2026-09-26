#!/bin/bash
# v5_sweep_one.sh — 单案例扫, 崩溃自动降级隔离模式
# 用法: bash v5_sweep_one.sh <case>
# 判定: 正常模式跑完(RESULT 行出现)即成功; 进程死亡/堆损坏 -> --isolated 重试一次
CASE=$1
cd /data/qlyu/tmp_shumo/a2026_reinforce
export A2026_ROOT=/data/qlyu/tmp_shumo/a2026_mechA
export A2026_SC=/data/qlyu/tmp_shumo/a2026/singlecore
export A2026_ATT=/data/qlyu/tmp_shumo/a2026/att
export A2026_ROUT=/data/qlyu/tmp_shumo/a2026_reinforce/out_v5
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
P=/data/qlyu/anaconda3/bin/python
LOG=v5_${CASE}.log

$P -u reinforce_assign.py 3 $CASE --iters 50 --N 128 --workers 12 --seed 0 > $LOG 2>&1
RC=$?
if grep -q "RESULT" $LOG 2>/dev/null; then
    echo "$CASE OK rc=$RC"
    exit 0
fi
echo "$CASE CRASHED rc=$RC -> isolated retry"
$P -u reinforce_assign.py 3 $CASE --iters 50 --N 128 --workers 12 --seed 0 \
   --isolated --pyexe $P > ${LOG%.log}_iso.log 2>&1
RC2=$?
if grep -q "RESULT" ${LOG%.log}_iso.log 2>/dev/null; then
    echo "$CASE ISO_OK rc=$RC2"
    exit 0
fi
echo "$CASE ISO_FAILED rc=$RC2"
exit 1
