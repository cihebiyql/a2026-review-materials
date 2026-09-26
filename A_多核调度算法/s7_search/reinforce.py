# -*- coding: utf-8 -*-
"""s7_search.reinforce — V4 候选搜索：学习式核指派（REINFORCE）。

方法骨架（ICLR23 one-shot priority sampling 的 A 题移植）：
  每链/段一个 logit；采样 = logits + i.i.d. Gumbel → argsort 得单元全局序
  → 确定性解码器(双管负载均衡)生成 op->核 → B×L 重标 → FastEval 真值
  → REINFORCE（批内代价标准化 + logit 范数正则）+ Adam。
统一协议（无逐例超参）：B_CAND = [16,48,96,160,240]，L=4，
iters=50 × N=128 采样，seed=0，logit 初始化围绕种子方案诱导序（热启动）。
种子方案由调用方显式传入（上一版本输出），支持 Q∈{1,2,3}。
"""
import time
from collections import defaultdict

import numpy as np

from s3_units.topo_units import build_units
from s5_assign.core_assign import decoder_assign
from s6_order.relabel import relabel_perm, relabel_perm_seg

B_CAND = [16, 48, 96, 160, 240]
L_FIXED = 4
C_LOGITS = 0.001
EPS = 0.1
LR = 0.05

_G = {}


def _eval_one(args):
    order_chains, B = args
    try:
        op_core_r = decoder_assign(_G['graph'], _G['comp_of'],
                                   list(order_chains), _G['n_cores'],
                                   _G['q'], _G.get('params'))
        if _G['seg_mode']:
            plan = relabel_perm_seg(_G['n_cores'], op_core_r, _G['comp_of'],
                                    _G['depth'], B, L_FIXED)
        else:
            chain_pos = {int(k): float(p) for p, k in enumerate(order_chains)}
            plan = relabel_perm(_G['n_cores'], op_core_r, _G['comp_of'],
                                _G['depth'], chain_pos, B, L_FIXED)
        mk, _ = _G['fe'].evaluate(plan)
        return mk
    except Exception:
        return None


def _reinforce_grad(logits, order, costs):
    """E[∇logπ(序列)·C̄]，无偏策略梯度（批内标准化）。"""
    N, n = order.shape
    mask = np.ones((N, n), dtype=bool)
    grad = np.zeros((N, n))
    rows = np.arange(N)
    for i in range(n):
        z = np.where(mask, logits[None, :], -np.inf)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        grad -= p
        picked = order[:, i]
        grad[rows, picked] += 1.0
        mask[rows, picked] = False
    return (grad * costs[:, None]).mean(axis=0)


def train(fe, graph, seed_plan, Q, iters=50, N=128, workers=16, seed=0,
          params=None, log_every=10):
    """返回 (best_mk, best_plan, best_B, rec)。种子恒参与比较（单调不劣）。"""
    import multiprocessing as mp
    t0 = time.time()
    from s2_features.graph_features import comp_depth
    from s3_units.topo_units import op_core_of
    from s1_preprocess.dag_view import op_dag

    plan0 = seed_plan
    mk0, _ = fe.evaluate(plan0)
    ids, preds, succs = op_dag(graph)
    comp_raw, depth = comp_depth(ids, preds, succs)
    comp_of = build_units(comp_raw, depth)
    seg_mode = len(set(comp_of.values())) != len(set(comp_raw.values()))
    op_core = op_core_of(plan0)
    n_cores = len(plan0['core_schedules'])

    chain_ops = defaultdict(list)
    for o, k in comp_of.items():
        chain_ops[k].append(o)
    chains = sorted(chain_ops, key=lambda k: min(chain_ops[k]))
    n = len(chains)
    rank = np.arange(n, dtype=np.float64)
    logits = (1.0 - 2.0 * rank / max(1, n - 1)).astype(np.float64)

    _G.update(fe=fe, graph=graph, comp_of=comp_of, depth=depth,
              n_cores=n_cores, q=Q, seg_mode=seg_mode,
              params=params or {})

    def build_plan(order_row, B):
        op_core_r = decoder_assign(graph, comp_of, list(order_row),
                                   n_cores, Q, params or {})
        if seg_mode:
            return relabel_perm_seg(n_cores, op_core_r, comp_of, depth,
                                    B, L_FIXED)
        chain_pos = {int(k): float(p) for p, k in enumerate(order_row)}
        return relabel_perm(n_cores, op_core_r, comp_of, depth, chain_pos,
                            B, L_FIXED)

    ctx = mp.get_context('fork')
    pool = ctx.Pool(workers)
    rs = np.random.RandomState(seed)
    m = np.zeros_like(logits)
    v = np.zeros_like(logits)
    best = (mk0, plan0, None)
    n_eval = 0
    try:
        for it in range(iters):
            G = rs.gumbel(size=(N, n))
            order_mat = np.argsort(-(logits[None, :] + G), axis=1)
            order_chains = np.array(chains)[order_mat]
            B_arr = np.array(B_CAND)[rs.randint(len(B_CAND), size=N)]
            mks = pool.map(_eval_one,
                           [(order_chains[j], int(B_arr[j]))
                            for j in range(N)])
            n_eval += N
            arr = np.array([mk if mk is not None and mk < 1e15 else np.nan
                            for mk in mks], dtype=np.float64)
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                continue
            arr[~np.isfinite(arr)] = valid.max() * 2
            C_bar = (arr - arr.mean()) / max(arr.std(), EPS)

            j = int(np.nanargmin(arr))
            if arr[j] < best[0]:
                best = (float(arr[j]),
                        build_plan(order_chains[j], int(B_arr[j])),
                        int(B_arr[j]))

            g = _reinforce_grad(logits, order_mat, C_bar)
            g += C_LOGITS * 2.0 * logits / n
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** (it + 1))
            vh = v / (1 - 0.999 ** (it + 1))
            logits -= LR * mh / (np.sqrt(vh) + 1e-8)

            greedy_order = np.array(chains)[np.argsort(-logits)]
            g_mks = pool.map(_eval_one,
                             [(greedy_order, B) for B in B_CAND])
            n_eval += len(B_CAND)
            for B, gmk in zip(B_CAND, g_mks):
                if gmk is not None and gmk < best[0]:
                    best = (gmk, build_plan(greedy_order, B), B)
    finally:
        pool.close()
        pool.join()

    rec = {'mk0': mk0, 'best': best[0], 'best_B': best[2], 'n_chains': n,
           'seg_mode': seg_mode, 'iters': iters, 'N': N, 'seed': seed,
           'n_eval': n_eval, 'wall_s': round(time.time() - t0, 1)}
    return best[0], best[1], best[2], rec
