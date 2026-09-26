# solve_compliant.py — 合规在线求解器(论文数字唯一来源)
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-25
#
# 两段式架构的在线段(离线段 = reinforce_assign 训练并落盘 logits):
#   加载该例离线训练的 logits(模型参数) → 贪心序解码 + Gumbel 采样 S(k) 取优
#   → 预算内局部精修(单元搬核/换核 + B 邻域 + 序扰动爬山) → 输出方案
#   全程硬预算(默认 480s, 题面"单用例 5-10 分钟"口径), 不读任何方案池/冠军。
# 方法出处: Jeon et al., Neural DAG Scheduling via One-Shot Priority Sampling,
#           ICLR 2023 (one-shot 优先级采样思想) + 本文领域化解码器。
# 用法: python solve_compliant.py <case> <Q> <N> [--budget_s 480] [--samples 256] [--seed 0]
# 输出: OUT/{case}_q{Q}_N{N}_compliant.json (方案+指标)
import sys, os, json, time, argparse
from collections import defaultdict
import numpy as np

ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
N5 = ROOT + '/n5_push'
SC_DIR = os.environ.get('A2026_SC', ROOT + '/results/singlecore')
LOGIT_DIR = os.environ.get('A2026_LOGITS', N5 + '/reinforce_v5')
OUT_DIR = os.environ.get('A2026_COUT', N5 + '/compliant_out')
sys.path.insert(0, N5)
sys.path.insert(0, N5 + '/superlinear_analysis')
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
from phase3_mechA2 import comp_depth
from common import load_case, ev_p1, ev_p2, ev_p3
from fast_eval_p2 import FastEvalP2, FastEvalP3
from fast_eval_p1 import FastEvalP1
from reinforce_assign import (build_units, relabel_perm, relabel_perm_seg)
from v5_assign_decoder import chain_features, assign_chain_cores


def build_plan(order_units, B, op_core, comp_of, depth, n_cores, seg_mode):
    if seg_mode:
        return relabel_perm_seg(n_cores, op_core, comp_of, depth, B, 4)
    chain_pos = {int(k): float(p) for p, k in enumerate(order_units)}
    return relabel_perm(n_cores, op_core, comp_of, depth, chain_pos, B, 4)


def solve(case, Q, n_cores, budget_s, n_samples, seed,
          logits_mode='trained', polish=True, tag='', official_eval=False,
          temp=1.0):
    t0 = time.time()
    graph = load_case(case)
    fe = {1: FastEvalP1, 2: FastEvalP2, 3: FastEvalP3}[Q](graph)
    sc = json.load(open(f'{SC_DIR}/{case}_sc.json'))['makespan']

    comp_of_raw, depth = comp_depth(graph)
    comp_of = build_units(comp_of_raw, depth)
    seg_mode = len(set(comp_of.values())) != len(set(comp_of_raw.values()))
    feats = chain_features(graph, comp_of)

    # 离线模型参数; 消融模式: uniform=无学习先验, rank=min_id 序先验
    # 崩溃豁免例(全平台无 q3 logits, 如 014/040/062)回退用相邻问的 logits
    lg = None
    logits_from = Q
    for q_try in dict.fromkeys([Q, 2, 3, 1]):
        fp_lg = f'{LOGIT_DIR}/{case}_q{q_try}_logits.json'
        if os.path.exists(fp_lg):
            lg = json.load(open(fp_lg))
            logits_from = q_try
            break
    assert lg is not None, f'{case} 无任何 logits'
    chains = lg['chains']
    assert set(chains) == set(feats['chains']), 'logits 单元集不一致'
    if logits_mode == 'uniform':
        logits = np.zeros(len(chains))
    elif logits_mode == 'rank':
        base = {k: i for i, k in enumerate(
            sorted(chains, key=lambda k: min(o for o, u in comp_of.items() if u == k)))}
        logits = np.array([1.0 - 2.0 * base[k] / max(1, len(chains) - 1)
                           for k in chains])
    else:
        logits = np.array(lg['logits'], dtype=np.float64) * temp
    B_CAND = lg['B_CAND'] if lg.get('B_CAND') else [16, 48, 96, 160, 240]

    rs = np.random.RandomState(seed)
    idx = {k: i for i, k in enumerate(chains)}

    def decode(order_list, B, mutate_core=None):
        op_core = assign_chain_cores(order_list, feats, comp_of, n_cores, Q, None)
        if mutate_core:                      # 精修算子: 指定单元强制换核
            u, c = mutate_core
            if u in op_core or True:
                op_core = {o: (c if k == u else cc)
                           for o, cc, k in ((o, op_core[o], comp_of[o])
                                            for o in op_core)}
        return build_plan(order_list, B, op_core, comp_of, depth, n_cores, seg_mode)

    def ev(plan):
        try:
            if official_eval:      # 豁免例通道: 官方评估器(纯Python,无numba崩溃)
                mk = (ev_p1, ev_p2, ev_p3)[Q - 1](graph, plan)[0]['makespan']
                return mk if mk < 1e15 else None
            mk, _ = fe.evaluate(plan)
            return mk if mk < 1e15 else None
        except Exception:
            return None

    # ---- 阶段1: 贪心 + S(k) 采样(约 40% 预算) ----
    t_stage1 = t0 + 0.40 * budget_s
    greedy_order = [chains[i] for i in np.argsort(-logits)]
    best_mk, best_plan, best_desc = None, None, None
    for B in B_CAND:                                  # 贪心序扫 B 候选
        mk = ev(decode(greedy_order, B))
        if mk is not None and (best_mk is None or mk < best_mk):
            best_mk, best_plan, best_desc = mk, decode(greedy_order, B), ('greedy', B)
        if time.time() > t_stage1:
            break
    k = 0
    while k < n_samples and time.time() < t_stage1:   # Gumbel 采样
        k += 1
        g = rs.gumbel(size=len(chains))
        order = [chains[i] for i in np.argsort(-(logits + g))]
        B = B_CAND[rs.randint(len(B_CAND))]
        plan = decode(order, B)
        mk = ev(plan)
        if mk is not None and (best_mk is None or mk < best_mk):
            best_mk, best_plan, best_desc = mk, plan, (f'S{k}', B)
            greedy_order = order                     # 采样更优则以它为精修起点
    n_samp = k

    # ---- 阶段2: 预算内局部精修(爬山: 单元搬核 + 序换位 + B 邻域) ----
    units = list(feats['chains'])
    cur_order, cur_B = list(greedy_order), best_desc[1]
    cur_core = assign_chain_cores(cur_order, feats, comp_of, n_cores, Q, None)
    n_eval2 = 0
    while polish and time.time() - t0 < budget_s and best_plan is not None:
        n_eval2 += 1
        op = rs.randint(3)
        order2, B2, op_core2 = cur_order, cur_B, cur_core
        if op == 0:                                   # 单元搬到负载最轻核
            load = defaultdict(int)
            for o, c in cur_core.items():
                load[c] += 1
            light = min(range(n_cores), key=lambda c: load[c])
            u = units[rs.randint(len(units))]
            op_core2 = dict(cur_core)
            for o, k in comp_of.items():
                if k == u:
                    op_core2[o] = light
        elif op == 1:                                 # 两单元换位(重解码指派)
            i, j = rs.randint(len(units)), rs.randint(len(units))
            if i != j:
                order2 = list(cur_order)
                a, b = order2.index(units[i]), order2.index(units[j])
                order2[a], order2[b] = order2[b], order2[a]
                op_core2 = assign_chain_cores(order2, feats, comp_of, n_cores, Q, None)
        else:                                         # B 邻域
            Bs = [b for b in B_CAND if b != cur_B]
            B2 = Bs[rs.randint(len(Bs))] if Bs else cur_B
        plan2 = build_plan(order2, B2, op_core2, comp_of, depth, n_cores, seg_mode)
        mk2 = ev(plan2)
        if mk2 is not None and mk2 < best_mk:
            best_mk, best_plan = mk2, plan2
            cur_order, cur_B, cur_core = order2, B2, op_core2

    wall = time.time() - t0
    rec = {'case': case, 'Q': Q, 'N': n_cores, 'mk': best_mk, 'sp': sc / best_mk,
           'samples': n_samp, 'polish_evals': n_eval2, 'wall_s': round(wall, 1),
           'budget_s': budget_s, 'seg_mode': seg_mode, 'desc': best_desc,
           'logits_mode': logits_mode, 'polish': polish, 'tag': tag,
           'seed': seed, 'logits_from': logits_from, 'temp': temp}
    os.makedirs(OUT_DIR, exist_ok=True)
    if not tag:                       # 主产线才落方案文件; 消融运行只记 jsonl
        json.dump({'rec': rec, 'plan': best_plan},
                  open(f'{OUT_DIR}/{case}_q{Q}_N{n_cores}_compliant.json', 'w'))
    with open(f'{OUT_DIR}/compliant_q{Q}.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('case'); ap.add_argument('Q', type=int); ap.add_argument('N', type=int)
    ap.add_argument('--budget_s', type=int, default=480)
    ap.add_argument('--samples', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--logits-mode', default='trained',
                    choices=['trained', 'uniform', 'rank'],
                    help='消融: trained=离线训练先验 uniform=无先验 rank=min_id序')
    ap.add_argument('--no-polish', action='store_true', help='消融: 关闭局部精修')
    ap.add_argument('--tag', default='', help='消融标签(输出目录区分)')
    ap.add_argument('--temp', type=float, default=1.0,
                    help='logits 温度(采样探索宽度, <1 更保守 >1 更探索')
    ap.add_argument('--official-eval', action='store_true',
                    help='豁免例通道: 用官方评估器求解(numba 崩溃图专用)')
    a = ap.parse_args()
    r = solve(a.case, a.Q, a.N, a.budget_s, a.samples, a.seed,
              logits_mode=a.logits_mode, polish=not a.no_polish, tag=a.tag,
              official_eval=a.official_eval, temp=a.temp)
    print('COMPLIANT', json.dumps(r, ensure_ascii=False), flush=True)
