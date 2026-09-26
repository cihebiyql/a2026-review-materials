#!/bin/bash
# v5_sweep_launch.sh — 剩余案例全库扫队列(4 并行 lane, 单案例单进程纪律)
# 跳过 jsonl 已有 RESULT 的案例 + 正在隔离测试的 045
cd /data/qlyu/tmp_shumo/a2026_reinforce
mkdir -p out_v5
DONE=$(grep -o '"case": "case_[0-9]*"' out_v5/reinforce_assign_q3.jsonl 2>/dev/null | sort -u | sed 's/.*case_/case_/' | tr -d '"')
echo "already done: $DONE"
: > sweep_queue.txt
for i in $(seq 1 100); do
    C=$(printf 'case_%03d' $i)
    case " $DONE " in *" $C "*) continue;; esac
    [ "$C" = "case_045" ] && continue        # 隔离测试进行中
    echo $C >> sweep_queue.txt
done
echo "queue size: $(wc -l < sweep_queue.txt)"
cat sweep_queue.txt | xargs -P 4 -I {} bash v5_sweep_one.sh {}
echo SWEEP_DONE $(date)
