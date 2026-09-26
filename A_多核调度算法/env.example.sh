# env.example.sh — 可选覆盖模板（默认零配置已可直接 bash run_all.sh；
# 仅当想用外部路径时拷贝为 env.sh 修改）
# 官方附件根(默认: 包内 official/)
# export A2026_OFFICIAL=/path/to/official
# 单核基准缓存(默认: 包内 sc/)
# export A2026_SC=/root/a2026/sc
# numba 0.60 解释器(默认: 存在则自动用)
# export PIPELINE_PY=/root/venv_n60/bin/python
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
