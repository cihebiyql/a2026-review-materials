# mechA v4: 三个正交结构升级,每个独立开关,真值(FastEval/官方)验收
#   --balance: 链按 (M功,V功) 配对装箱(重的M链配重的V链)替代大小贪心
#   --phase:   跨核相位错开(batch 序按核索引旋转)
#   --dp:      逐层自适应切分(代理代价 DP:管道均衡+边界字节)替代固定 L
import sys, json, os, glob
from collections import defaultdict

ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
N5 = ROOT + '/n5_push'
sys.path.insert(0, N5 + '/superlinear_analysis')
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
sys.path.insert(0, os.environ.get('A2026_ATT', r'C:/shumo_live/a_data/code'))
from phase3_mechA2 import comp_depth
from common import load_case, ev_p2, ev_p3
from fast_eval_p2 import FastEvalP2, FastEvalP3

Q = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def op_pipe_of(graph):
    return {o['id']: 1 if o.get('pipe') == 'PIPE_M' else 0 for o in graph['ops']}


def build_batches(ops, comps, comp_list, B, balance, pipe_of):
    """ comps: {comp_id: [ops]}. 返回 batch 列表(每批为 comp_id 列表). """
    if not balance:
        order = sorted(comp_list, key=lambda k: min(comps[k]))
        batch, size, out = [], 0, []
        for k in order:
            sz = len(comps[k])
            if batch and size + sz > B:
                out.append(batch); batch, size = [], 0
            batch.append(k); size += sz
        if batch:
            out.append(batch)
        return out
    # 均衡装箱: M 功降序链与 V 功降序链交替配对,批内 M/V 比例均衡
    mload = {k: sum(pipe_of.get(o, 0) * 1 for o in comps[k]) for k in comp_list}
    # 用 cycles 细化: M 链按 M功 降序,V 功作为次键
    order = sorted(comp_list, key=lambda k: (-mload[k], min(comps[k])))
    batch, size, out = [], 0, []
    # 贪心: 每次选与当前批 M/V 失衡互补的最重链
    remaining = set(comp_list)
    while remaining:
        if not batch:
            k = order[0] if order[0] in remaining else next(iter(remaining))
        else:
            cur_m = sum(mload[k2] for k2 in batch)
            cur_v = sum(len(comps[k2]) - mload[k2] for k2 in batch)
            want_m = cur_v > cur_m  # 缺什么补什么
            best_k, best_key = None, None
            for k2 in remaining:
                if (mload[k2] > 0) == want_m or len(batch) == 0:
                    key = (-len(comps[k2]), min(comps[k2]))
                    if best_key is None or key < best_key:
                        best_key, best_k = key, k2
            if best_k is None:
                best_k = next(iter(sorted(remaining, key=lambda k: min(comps[k]))))
            k = best_k
        batch.append(k); size += len(comps[k]); remaining.discard(k)
        if size >= B and remaining:
            out.append(batch); batch, size = [], 0
    if batch:
        out.append(batch)
    return out


def split_levels(bops, depth, L, dp, pipe_of):
    """ bops: 该 batch 的全部 op. 返回层切分(子图列表). L=None 且 dp=True 时用代理 DP. """
    maxd = max(depth[o] for o in bops)
    if not dp or L is not None:
        out, lev = [], 0
        while lev <= maxd:
            lops = [o for o in bops if lev <= depth[o] < lev + L]
            if lops:
                out.append(lops)
            lev += L
        return out
    # 代理 DP: 逐层决定切点. 代价 = sum over 段 (管道失衡惩罚 + 边界字节代理)
    # 段 [i,j) 的代理: |M功-V功|/总功 * (j-i) + 切点数惩罚*λ
    lev_ops = defaultdict(list)
    for o in bops:
        lev_ops[depth[o]].append(o)
    levels = [lev_ops[d] for d in sorted(lev_ops)]
    n = len(levels)
    W = [sum(1 for o in lv) for lv in levels]
    M = [sum(pipe_of.get(o, 0) for o in lv) for lv in levels]
    INF = float('inf')
    lam = 0.35  # 切点惩罚系数(每段的固定开销代理)
    f = [INF] * (n + 1); back = [0] * (n + 1)
    f[0] = 0.0
    for j in range(1, n + 1):
        for i in range(max(0, j - 8), j):  # 段长上限 8 层
            m = sum(M[i:j]); w = sum(W[i:j])
            imb = abs(m - (w - m)) / max(1, w)
            cost = imb * (j - i) + lam
            if f[i] + cost < f[j]:
                f[j] = f[i] + cost
                back[j] = i
    segs, j = [], n
    while j > 0:
        i = back[j]
        segs.append((i, j))
        j = i
    segs.reverse()
    return [[o for lv in levels[i:j] for o in lv] for i, j in segs]


def relabel_v4(plan, op_core, comp_of, depth, B, L, om, pipe_of,
               balance=False, phase=False, dp=False):
    core_ops = defaultdict(list)
    for op, c in op_core.items():
        core_ops[c].append(op)
    n_cores = len(plan['core_schedules'])
    new_n2s, new_cs = {}, [[] for _ in range(n_cores)]
    sgid = 0
    for c in range(n_cores):
        ops = core_ops.get(c, [])
        comps = defaultdict(list)
        for o in ops:
            comps[comp_of[o]].append(o)
        comp_list = list(comps)
        if om == 'min_id':
            comp_list.sort(key=lambda k: min(comps[k]))
        elif om == 'size_desc':
            comp_list.sort(key=lambda k: -len(comps[k]))
        batches = build_batches(ops, comps, comp_list, B, balance, pipe_of)
        if phase and c > 0:
            rot = c % max(1, len(batches))
            batches = batches[rot:] + batches[:rot]
        for batch in batches:
            bset = set(batch)
            bops = [o for o in ops if comp_of[o] in bset]
            for lops in split_levels(bops, depth, L, dp, pipe_of):
                sgid += 1
                for op in lops:
                    new_n2s[str(op)] = sgid
                new_cs[c].append(sgid)
    return {'node_to_subgraph': new_n2s, 'core_schedules': new_cs}


def champion_seed(case):
    best = (0.0, None)
    pats = [f'{N5}/superlinear_analysis/{case}_q{Q}_mechA.json']
    for sub in ('refined2', 'refined', 'strand_n5'):
        pats.append(f'{N5}/{sub}/{case}_q{Q}_*.json')
    pats.append(f'{N5}/pull_plans/{case}_q{Q}_pull.json')
    for pat in pats:
        for fp in glob.glob(pat):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            if d.get('sp') and d['sp'] > best[0] and 'plan' in d:
                best = (d['sp'], d['plan'])
    return best[1]


def sweep(case, fe, ev, graph, pipe_of):
    plan0 = champion_seed(case)
    if plan0 is None:
        return None, None
    sc = json.load(open(f'{ROOT}/results/singlecore/{case}_sc.json'))['makespan']
    mk0, _ = fe.evaluate(plan0)
    comp_of, depth = comp_depth(graph)
    core_of_sg = {sg: c for c, sgl in enumerate(plan0['core_schedules']) for sg in sgl}
    op_core = {op: core_of_sg[sg] for op, sg in {int(k): v for k, v in plan0['node_to_subgraph'].items()}.items()}
    best = (mk0, None, None)
    cfgs = []
    for om in ('min_id',):
        for L in (3, 4, None):          # None = DP 切分
            for bal in (False, True):
                for ph in (False, True):
                    cfgs.append((om, L, bal, ph))
    for om, L, bal, ph in cfgs:
        for B in (8, 16, 24, 32, 48, 64, 96, 128, 160, 200, 240, 320):
            try:
                plb = relabel_v4(plan0, op_core, comp_of, depth, B, L, om,
                                  pipe_of, balance=bal, phase=ph,
                                  dp=(L is None))
                mk1, _ = fe.evaluate(plb)
                if mk1 < best[0]:
                    best = (mk1, plb, (om, B, L, bal, ph))
            except Exception:
                continue
    return best, sc


if __name__ == '__main__':
    cases = sys.argv[2].split(',') if len(sys.argv) > 2 else \
        ['case_008', 'case_084', 'case_095', 'case_041', 'case_043', 'case_012']
    for case in cases:
        try:
            graph = load_case(case)
            pipe_of = op_pipe_of(graph)
            fe = FastEvalP2(graph) if Q == 2 else FastEvalP3(graph)
            ev = ev_p2 if Q == 2 else ev_p3
            best, sc = sweep(case, fe, ev, graph, pipe_of)
            if best is None:
                print(case, 'no_seed', flush=True); continue
            mk_best, pl_best, P = best
            line = f'{case}: mk0={best[0] if pl_best is None else ""} mk={mk_best} cfg={P}'
            if pl_best is not None and mk_best < 1e17:
                mk_off = ev(graph, pl_best)[0]['makespan']
                line += f' official={mk_off} match={mk_off==mk_best} sp={sc/mk_off:.4f}'
                if mk_off == mk_best:
                    old = json.load(open(f'{N5}/superlinear_analysis/{case}_q{Q}_mechA.json'))['sp'] \
                        if os.path.exists(f'{N5}/superlinear_analysis/{case}_q{Q}_mechA.json') else 0
                    if sc / mk_off > old:
                        json.dump({'plan': pl_best, 'mk': mk_off, 'sp': sc / mk_off,
                                   'source': f'mechA_v4 {P}'},
                                  open(f'{N5}/superlinear_analysis/{case}_q{Q}_mechA.json', 'w'))
                        line += ' SAVED(优于v3)'
            print(line, flush=True)
        except Exception as e:
            print(case, 'ERR', str(e)[:80], flush=True)
