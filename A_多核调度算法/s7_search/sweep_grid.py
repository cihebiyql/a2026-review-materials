# -*- coding: utf-8 -*-
"""s7_search.sweep_grid — 候选搜索（V2 / V3：确定性网格）。

  - sweep_bl (V2)：对种子方案的 op->核 做 B×L 网格重标深扫。
    网格 = L∈{2,3,4,6,8} × B∈{4,6,...,238}（590 候选，全例统一）。
  - sweep_v4 (V3)：三结构扫描。cfg = (L∈{3,4,DP}) × (均衡装箱) × (相位)
    共 12 组合 × B∈{8..320} 12 档（144 候选，全例统一）。
搜索反馈 = FastEval 真值；种子本身参与比较（best-so-far 单调不劣）。
返回 (best_mk, best_plan, best_cfg, n_evals)。
"""
import time
from s6_order.relabel import relabel_bl, relabel_v4


V2_LS = (2, 3, 4, 6, 8)
V2_BS = tuple(range(4, 240, 2))
V3_B_LIST = (8, 16, 24, 32, 48, 64, 96, 128, 160, 200, 240, 320)


def sweep_bl(fe, seed_plan, n_cores, op_core, comp_of, depth):
    # 先评种子：best 初值 = 种子 mk（版本阶梯单调不劣的保证）
    seed_mk, _ = fe.evaluate(seed_plan)
    best_mk, best_plan, best_cfg = seed_mk, seed_plan, None
    n_done = 1
    for L in V2_LS:
        for B in V2_BS:
            try:
                pl = relabel_bl(n_cores, op_core, comp_of, depth, B, L)
                mk, _ = fe.evaluate(pl)
                n_done += 1
                if mk < best_mk:
                    best_mk, best_plan, best_cfg = mk, pl, (B, L)
            except Exception:
                continue
    return best_mk, best_plan, best_cfg, n_done


def sweep_v4(fe, seed_plan, n_cores, op_core, comp_of, depth, pipe_of):
    seed_mk, _ = fe.evaluate(seed_plan)
    best_mk, best_plan, best_cfg = seed_mk, seed_plan, None
    n_done = 1
    cfgs = []
    for L in (3, 4, None):
        for bal in (False, True):
            for ph in (False, True):
                cfgs.append((L, bal, ph))
    for (L, bal, ph) in cfgs:
        for B in V3_B_LIST:
            try:
                pl = relabel_v4(n_cores, op_core, comp_of, depth, B, L,
                                pipe_of, balance=bal, phase=ph,
                                dp=(L is None))
                mk, _ = fe.evaluate(pl)
                n_done += 1
                if mk < best_mk:
                    best_mk, best_plan, best_cfg = mk, pl, (B, L, bal, ph)
            except Exception:
                continue
    return best_mk, best_plan, best_cfg, n_done


def pipe01_map(graph):
    return {o['id']: 1 if o.get('pipe') == 'PIPE_M' else 0
            for o in graph['ops'] if o.get('op') not in
            ('COPY_IN', 'COPY_OUT')}
