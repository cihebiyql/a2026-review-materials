# paths.py — 全项目统一路径与 sys.path 装配（唯一入口，所有模块 import 它）
import os
import sys

PIPE_ROOT = os.path.dirname(os.path.abspath(__file__))
DEPS = os.path.join(PIPE_ROOT, 'fasteval')

OFFICIAL = os.environ.get('A2026_OFFICIAL') or os.path.join(PIPE_ROOT, 'official')  # 官方附件根(包内自带)
SC_DIR = os.environ.get('A2026_SC') or os.path.join(PIPE_ROOT, 'sc')  # 单核基准缓存
RESULTS = os.environ.get('A2026_RESULTS') or os.path.join(PIPE_ROOT, 'results')

for p in (DEPS, os.path.join(OFFICIAL, 'code') if OFFICIAL else None):
    if p and p not in sys.path:
        sys.path.insert(0, p)

CASES = [f'case_{i:03d}' for i in range(1, 101)]


def sc_makespan(case):
    import json
    with open(os.path.join(SC_DIR, f'{case}_sc.json'), encoding='utf-8') as f:
        return json.load(f)['makespan']
