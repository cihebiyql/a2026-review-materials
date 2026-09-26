# -*- coding: utf-8 -*-
"""s8_evaluate.fasteval_adapter — FastEval 统一入口。

FastEval = 官方评估器的 numba bit-exact CPU 复刻
（三问内核均已通过 fast == official 逐位一致性验证）。搜索阶段全部用它做
真值反馈；只有终选方案再走官方评估器复核（s8_evaluate.official）。
"""
import os

import paths  # noqa: F401


def make_fe(Q, graph):
    if os.environ.get('PIPELINE_EVAL') == 'official':
        return OfficialFE(Q, graph)      # 稳定性回退: 官方评估器纯 Python 反馈
    if Q == 1:
        from fast_eval_p1 import FastEvalP1
        return FastEvalP1(graph)
    if Q == 2:
        from fast_eval_p2 import FastEvalP2
        return FastEvalP2(graph)
    if Q == 3:
        from fast_eval_p2 import FastEvalP3
        return FastEvalP3(graph)
    raise ValueError(f'Q must be 1/2/3, got {Q}')


class OfficialFE:
    """稳定性回退：直接用官方评估器当反馈（慢 ~10-50×，但进程稳定）。"""

    def __init__(self, Q, graph):
        self.Q, self.graph = Q, graph
        from official_eval import ev_p1, ev_p2, ev_p3
        self._ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[Q]

    def evaluate(self, plan):
        r, _ = self._ev(self.graph, plan)
        return r['makespan'], None
