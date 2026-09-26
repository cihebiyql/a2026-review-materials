#!/bin/bash
# ============ 一键全量入口（零配置：官方附件已内置在 official/） ============
# 产物: results/tables/*.csv + results/figs/*.png (赛题格式)
set -e
cd "$(dirname "$0")"
# 包内自带附件与单核基准——未显式覆盖时自动使用
export A2026_OFFICIAL="${A2026_OFFICIAL:-$PWD/official}"
export A2026_SC="\${A2026_SC:-$PWD/sc}"
[ ! -f "$A2026_OFFICIAL/data/config.txt" ] && { echo "缺官方附件: $A2026_OFFICIAL"; exit 1; }
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
[ -z "$PIPELINE_PY" ] && [ -x /root/venv_n60/bin/python ] && \
    export PIPELINE_PY=/root/venv_n60/bin/python
echo "=== [1/5] 全库计算: 100例 × Q1-3 × N2-5 × 版本阶梯 V1→V4 ==="
python3 run_batch.py --qs 3,2,1 --stop-at v4
echo "=== [2/5] 逐例表 + 汇总 ==="
PYTHONPATH=. python3 s9_evidence/make_tables.py
echo "=== [3/5] 正文图(1~5核折线/Q3两配置/版本消融) ==="
PYTHONPATH=. python3 s9_evidence/make_curves.py
echo "=== [4/5] 失败分析 + 增益分布 ==="
PYTHONPATH=. python3 s9_evidence/failure_report.py
echo "=== [5/5] 权威证据记录(全部方案官方单遍重评) ==="
PYTHONPATH=. python3 s9_evidence/requalify.py
echo "ALL_DONE $(date)  结果在 results/tables 与 results/figs"
