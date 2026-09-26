# -*- coding: utf-8 -*-
"""run_batch.py — 全库批量调度（run_all.sh 的计算入口）。

执行模型：每 (case,Q) 一个**完全独立**的子进程（python pipeline.py），
车道数并发。单 case 进程崩溃（个别图×方案形状组合会触发加速评估
内核崩溃）只损失该 case，救援梯度：
  尝试1: 正常跑 v1-v3（FastEval 反馈）
  尝试2: 失败则用官方评估器（纯 Python，不崩）重跑 v1-v3（慢但稳）
  v4 车道: 正常跑；失败重试一次；仍失败则跳过（表内该例 v4 缺行）
断点续跑：runs/*.jsonl 已含 (case,Q) 全部四版本记录的自动跳过。
用法：python3 run_batch.py [--qs 3,2,1] [--lanes-a 32] [--lanes-b 8]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import paths
from pipeline import N_LIST

PY = os.environ.get('PIPELINE_PY', sys.executable)


def _done_keys(chain=('v1', 'v2', 'v3', 'v4')):
    done = set()
    runs = os.path.join(paths.RESULTS, 'runs')
    need = {(v, n) for v in chain for n in N_LIST}
    for fn in os.listdir(runs) if os.path.isdir(runs) else []:
        if not fn.endswith('.jsonl'):
            continue
        key = fn[:-6]
        got = set()
        try:
            for line in open(os.path.join(runs, fn), encoding='utf-8'):
                d = json.loads(line)
                got.add((d['version'], d['N']))
        except Exception:
            pass
        if need <= got:
            done.add(key)                    # 'case_001_q3'
    return done


def _spawn(c, q, versions, eval_mode, logdir):
    env = dict(os.environ)
    if eval_mode == 'official':
        env['PIPELINE_EVAL'] = 'official'
    lg = open(os.path.join(logdir, f'{c}_q{q}_{eval_mode}.log'), 'w')
    p = subprocess.Popen(
        [PY, '-u', os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'pipeline.py'),
         c, str(q), '--versions', ','.join(versions)],
        stdout=lg, stderr=lg, env=env, cwd=os.path.dirname(
            os.path.abspath(__file__)))
    return p, lg


def _run_lane(tasks, versions, lanes, tag, logdir):
    """独立子进程车道调度；返回 {task: 'ok'|'crash'|'official_fallback'}。"""
    queue = list(tasks)
    procs = {}
    status = {}
    n_launch = 0
    while queue or procs:
        while queue and len(procs) < lanes:
            (c, q, mode) = queue.pop(0)
            p, lg = _spawn(c, q, versions, mode, logdir)
            procs[p.pid] = (p, (c, q, mode), lg)
            n_launch += 1
        time.sleep(2)
        for pid in [pid for pid, (_, _, _) in procs.items()
                    if procs[pid][0].poll() is not None]:
            p, task, lg = procs.pop(pid)
            lg.close()
            c, q, mode = task
            rc = p.returncode
            if rc == 0:
                status[(c, q)] = status.get((c, q), '') + '+ok'
            else:
                status[(c, q)] = status.get((c, q), '') + f'+crash{rc}'
            print(f'[{tag} {len(status)}/{len(tasks) + 0}] {c}_q{q} '
                  f'({mode}) rc={rc}', flush=True)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qs', default='3,2,1')
    ap.add_argument('--stop-at', default='v4', choices=['v1', 'v2', 'v3', 'v4'],
                    help='版本阶梯终点(该版本及之前的全部版本)')
    ap.add_argument('--lanes-a', type=int, default=32)
    ap.add_argument('--lanes-b', type=int, default=8)
    ap.add_argument('--lanes-rescue', type=int, default=32,
                    help='官方评估救援车道数(纯Python,可高并发)')
    ap.add_argument('--v4-workers', type=int, default=16)
    ap.add_argument('--v4-iters', type=int, default=50)
    ap.add_argument('--only', default=None)
    a = ap.parse_args()

    chain = {'v1': ('v1',), 'v2': ('v1', 'v2'),
             'v3': ('v1', 'v2', 'v3'),
             'v4': ('v1', 'v2', 'v3', 'v4')}[a.stop_at]
    chain_a = tuple(v for v in chain if v != 'v4')

    logdir = os.path.join(paths.RESULTS, 'driver_logs')
    os.makedirs(logdir, exist_ok=True)
    qs = [int(x) for x in a.qs.split(',')]
    cases = a.only.split(',') if a.only else paths.CASES
    tasks = [(c, q) for q in qs for c in cases]
    done = _done_keys(chain)
    todo = [(c, q) for (c, q) in tasks if f'{c}_q{q}' not in done]
    print(f'total={len(tasks)} todo={len(todo)} done={len(done)} '
          f'stop_at={a.stop_at}', flush=True)

    # ---- 阶段 A: v1..stop_at(-v4), FastEval ----
    st = _run_lane([(c, q, 'fast') for (c, q) in todo],
                   chain_a, a.lanes_a, 'A', logdir)
    crashed = [(c, q) for (c, q) in todo if 'crash' in st.get((c, q), '')]
    # ---- 阶段 A 救援: 官方评估器反馈(纯 Python 不崩) ----
    if crashed:
        print(f'A-rescue with official-eval feedback: {len(crashed)} cases',
              flush=True)
        _run_lane([(c, q, 'official') for (c, q) in crashed],
                  chain_a, a.lanes_rescue, 'A2', logdir)

    # ---- 阶段 B: v4 (16-worker fork 池 × 8 车道; 仅 fast 模式完成的案例,
    #      崩溃案例 v4 跳过——make_tables 以 v3 结果继承并标注) ----
    todo_v4 = [(c, q) for (c, q) in todo if 'ok' in st.get((c, q), '')]
    env_extra = {'PIPELINE_V4_WORKERS': str(a.v4_workers),
                 'PIPELINE_V4_ITERS': str(a.v4_iters)}
    os.environ.update(env_extra)
    stb = _run_lane([(c, q, 'v4') for (c, q) in todo_v4],
                    ('v4',), a.lanes_b, 'B', logdir)
    v4crash = [(c, q) for (c, q) in todo_v4 if 'crash' in stb.get((c, q), '')]
    if v4crash:
        print(f'B-retry: {len(v4crash)} cases', flush=True)
        _run_lane([(c, q, 'v4') for (c, q) in v4crash],
                  ('v4',), a.lanes_b, 'B2', logdir)
    print('BATCH_DONE', flush=True)


if __name__ == '__main__':
    main()
