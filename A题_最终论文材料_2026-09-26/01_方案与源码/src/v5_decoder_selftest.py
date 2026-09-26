# -*- coding: utf-8 -*-
"""v5_decoder_selftest.py — v5 指派解码器 Q=3 自测 (case_064 + case_029)。

运行:
    cd C:/shumo_live/02_求解/A题_2026/n5_push
    py -3.11 v5_decoder_selftest.py
  (A2026_ATT 默认 C:/shumo_live/a_data; FastEval 首次调用有 numba 编译开销,
   全程约 2-4 分钟)

内容 (每例, FastEvalP3, B=48 L=4, n_cores=冠军核数):
  1) min_id 序 (feats['chains'] 原序) → 解码器指派 → relabel_perm → 评估:
     必须无异常且 mk < 3×冠军 mk
  2) 32 个随机链序 (np.random.RandomState(0) 洗牌) 过解码器: mk 均值
  3) 对照组: 32 个完全随机指派 (链均匀随机到核, 序用 min_id) 过 relabel: mk 均值
  4) 打印冠军 mk; 判据: 解码器随机序均值 应明显好于 随机指派均值
     (逐例 + 两例合计; 单链退化例 case_064 预期两组持平, 见打印说明)
"""
import os
import sys
import time

import numpy as np

ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
N5 = ROOT + '/n5_push'
for p in (N5, N5 + '/superlinear_analysis', ROOT + '/fast_eval',
          ROOT + '/v3_solver'):
    if p not in sys.path:
        sys.path.insert(0, p)

from reinforce_order import relabel_perm, champion_seed   # noqa: E402
from phase3_mechA2 import comp_depth                       # noqa: E402
from common import load_case                               # noqa: E402
from fast_eval_p2 import FastEvalP3                        # noqa: E402
from v5_assign_decoder import chain_features, assign_chain_cores  # noqa: E402

CASES = ('case_064', 'case_029')
Q = 3
B, L = 48, 4
N_RND = 32


def run_case(case):
    t0 = time.time()
    plan0, sp0_seed = champion_seed(case, Q)
    assert plan0 is not None, f'{case}: no champion seed'
    graph = load_case(case)
    fe = FastEvalP3(graph)
    mk0, _ = fe.evaluate(plan0)
    comp_of, depth = comp_depth(graph)
    feats = chain_features(graph, comp_of)
    chains = feats['chains']
    n = len(chains)
    n_cores = len(plan0['core_schedules'])

    def eval_assign(op_core, order):
        chain_pos = {int(k): float(i) for i, k in enumerate(order)}
        plan = relabel_perm(n_cores, op_core, comp_of, depth, chain_pos, B, L)
        mk, info = fe.evaluate(plan)
        return mk, info

    # ---- 1) min_id 序过解码器 ----
    op_core1 = assign_chain_cores(chains, feats, comp_of, n_cores, q=Q)
    assert len(op_core1) == len(comp_of), \
        f'op_core 覆盖 {len(op_core1)} != comp_of {len(comp_of)}'
    assert set(op_core1.values()) <= set(range(n_cores))
    mk1, info1 = eval_assign(op_core1, chains)
    per_core = np.bincount(np.fromiter(op_core1.values(), dtype=int),
                           minlength=n_cores)
    ok1 = mk1 < 3 * mk0

    # ---- 2) 32 个随机链序过解码器 ----
    rs = np.random.RandomState(0)
    mks_dec = []
    for t in range(N_RND):
        order = [chains[i] for i in rs.permutation(n)]
        oc = assign_chain_cores(order, feats, comp_of, n_cores, q=Q)
        mk, _ = eval_assign(oc, order)
        mks_dec.append(mk)
    mks_dec = np.array(mks_dec, dtype=float)

    # ---- 3) 对照: 32 个完全随机指派 (链均匀随机到核, 序 min_id) ----
    rs2 = np.random.RandomState(0)
    mks_rnd = []
    for t in range(N_RND):
        rc = {k: int(rs2.randint(n_cores)) for k in chains}
        oc = {op: rc[kk] for op, kk in comp_of.items()}
        mk, _ = eval_assign(oc, chains)
        mks_rnd.append(mk)
    mks_rnd = np.array(mks_rnd, dtype=float)

    n_shared = len(feats['pair_bytes'])
    n_multi = len(feats['shared'])
    print(f'\n===== {case} (Q={Q}, B={B}, L={L}, n_cores={n_cores}) =====')
    print(f'链数 n={n} (op 数 {len(comp_of)}), 跨链 pair 数={n_shared}, '
          f'多消费链张量数={n_multi}')
    print(f'冠军 mk = {mk0} (seed sp={sp0_seed:.4f})')
    print(f'[1] min_id 序过解码器: mk={mk1}  (3×冠军={3 * mk0}, '
          f'判据 mk<3×冠军: {"PASS" if ok1 else "FAIL"}; '
          f'每核op数={per_core.tolist()}; '
          f'cross={info1["cross_task_traffic"] / 1e6:.2f}MB, '
          f'spill={info1["spill_added_copy_bytes"] / 1e6:.2f}MB, '
          f'hit={info1["cache_stats"]["hit_rate"]:.3f})')
    print(f'[2] 解码器×32随机链序: mean={mks_dec.mean():.1f} '
          f'std={mks_dec.std():.1f} min={mks_dec.min():.0f} '
          f'max={mks_dec.max():.0f}')
    print(f'[3] 完全随机指派×32:   mean={mks_rnd.mean():.1f} '
          f'std={mks_rnd.std():.1f} min={mks_rnd.min():.0f} '
          f'max={mks_rnd.max():.0f}')
    gain = 100 * (1 - mks_dec.mean() / mks_rnd.mean())
    single = (n == 1)
    if single:
        print(f'[4] 判据(解码器均值 明显好于 随机指派均值): '
              f'单链退化例 —— 链级指派只能是"全图一核", 两组按构造相同 '
              f'(增益 {gain:+.1f}% ≈ 0)。单核解 vs 冠军: '
              f'{100 * (mk1 / mk0 - 1):+.1f}%')
    else:
        ok4 = mks_dec.mean() < mks_rnd.mean() - 2 * max(
            mks_dec.std(), 1e-9)
        print(f'[4] 判据(解码器均值 明显好于 随机指派均值): '
              f'{"PASS" if ok4 else "FAIL"}  '
              f'(解码器均值低 {gain:.1f}%)')
    print(f'耗时 {time.time() - t0:.0f}s')
    return {'case': case, 'n': n, 'mk0': mk0, 'mk1': mk1, 'ok1': ok1,
            'dec_mean': float(mks_dec.mean()), 'dec_std': float(mks_dec.std()),
            'dec_min': float(mks_dec.min()), 'dec_max': float(mks_dec.max()),
            'rnd_mean': float(mks_rnd.mean()), 'rnd_std': float(mks_rnd.std()),
            'single': single}


def main():
    print('v5 指派解码器 Q=3 自测 (FastEvalP3, RandomState(0), 32 trials)')
    recs = [run_case(c) for c in CASES]
    print('\n===== 汇总 =====')
    for r in recs:
        tag = '单链退化' if r['single'] else \
              ('PASS' if r['dec_mean'] < r['rnd_mean'] else 'FAIL')
        print(f"{r['case']}: 冠军 {r['mk0']} | min_id {r['mk1']} | "
              f"解码器随机序均值 {r['dec_mean']:.1f} | 随机指派均值 "
              f"{r['rnd_mean']:.1f} | 判据4[{tag}] | 判据1 "
              f"[{'PASS' if r['ok1'] else 'FAIL'}]")
    tot_dec = sum(r['dec_mean'] for r in recs)
    tot_rnd = sum(r['rnd_mean'] for r in recs)
    print(f'两例合计: 解码器 {tot_dec:.1f} vs 随机指派 {tot_rnd:.1f} '
          f'(低 {100 * (1 - tot_dec / tot_rnd):.1f}%)')


if __name__ == '__main__':
    main()
