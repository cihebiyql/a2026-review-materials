# -*- coding: utf-8 -*-
"""s1_preprocess.dag_view — DAG 数据预处理。

职责：加载官方用例 JSON，构建算法侧的图视图：
  - Op-Op 压缩依赖图（COPY_IN/OUT 收缩掉，官方 step1 同构）
  - 周期/管道映射、张量视图（生产/消费/字节/位置）
全部语义与官方评估器 stub_multicore_cut_and_schedule 一致（直接复用其
_build_op_adjacency / _contract_excluded_copy_nodes，保证合法性判定同源）。
"""
from paths import OFFICIAL  # noqa: F401  (触发 sys.path 装配)

import json
import os
from collections import defaultdict

from stub_multicore_cut_and_schedule import (  # noqa: E402
    _build_op_adjacency, _contract_excluded_copy_nodes, EXCLUDED_COPY_TYPES)


def load_case(case):
    data = os.path.join(OFFICIAL, 'data')
    with open(os.path.join(data, f'{case}.json'), encoding='utf-8') as f:
        return json.load(f)


def op_dag(graph):
    """返回 (ids, preds, succs)：非 COPY 算子的压缩 Op-Op 依赖图。"""
    ids = sorted(o['id'] for o in graph['ops']
                 if o.get('op') not in EXCLUDED_COPY_TYPES)
    _, succs = _build_op_adjacency(graph)
    cp, cs = _contract_excluded_copy_nodes(ids, succs)
    return ids, cp, cs


def pipe_map(graph):
    return {o['id']: o.get('pipe', 'PIPE_V') for o in graph['ops']
            if o.get('op') not in EXCLUDED_COPY_TYPES}


def cycles_map(graph):
    return {o['id']: max(1, o.get('cycles', 1)) for o in graph['ops']
            if o.get('op') not in EXCLUDED_COPY_TYPES}


def tensor_views(graph):
    """op->生产张量 / op->消费张量 / 张量大小 / 张量位置。"""
    op_ids = {o['id'] for o in graph['ops']}
    prod, cons = defaultdict(set), defaultdict(set)
    tsize = {t['id']: t['size'] for t in graph['tensors']}
    tpos = {t['id']: t.get('pos', 'UB') for t in graph['tensors']}
    for e in graph['edges']:
        s, t = e['source'], e['target']
        if s in op_ids and t not in op_ids:
            prod[s].add(t)
        elif t in op_ids and s not in op_ids:
            cons[t].add(s)
    return prod, cons, tsize, tpos
