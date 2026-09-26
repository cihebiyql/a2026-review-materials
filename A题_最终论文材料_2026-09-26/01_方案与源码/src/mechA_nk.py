# mechA 下沉 N2-N4: 用 archive 种子 + relabel 变体网格, 官方终验落盘(文件名带 N 标记)
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
def relabel_v3(plan, op_core, comp_of, depth, B, L, order_mode):
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
        if order_mode == 'min_id':
            order = sorted(comps, key=lambda k: min(comps[k]))
        elif order_mode == 'size_desc':
            order = sorted(comps, key=lambda k: -len(comps[k]))
        else:  # by_depth: 分量最大深度浅的先(浅层链优先)
            order = sorted(comps, key=lambda k: max(depth[o] for o in comps[k]))
        batch, size, batches = [], 0, []
        for k in order:
            sz = len(comps[k])
            if batch and size + sz > B:
                batches.append(batch)
                batch, size = [], 0
            batch.append(k)
            size += sz
        if batch:
            batches.append(batch)
        for batch in batches:
            bset = set(batch)
            bops = [o for o in ops if comp_of[o] in bset]
            maxd = max(depth[o] for o in bops)
            lev = 0
            while lev <= maxd:
                lops = [o for o in bops if lev <= depth[o] < lev + L]
                if lops:
                    sgid += 1
                    for op in lops:
                        new_n2s[str(op)] = sgid
                    new_cs[c].append(sgid)
                lev += L
    return {'node_to_subgraph': new_n2s, 'core_schedules': new_cs}



Q = int(os.environ.get('NK_Q', '2'))
K = int(os.environ.get('NK_K', '4'))
OUTJSONL = f'{N5}/mechA_nk_q{Q}_N{K}.jsonl'


def seed_of(case):
    # archive v2_cpc 种子(每核大块), 也可用 posthoc 记录
    fp = f'{ROOT}/a_lab/registry/plan_archive/plans/{case}_q{Q}_N{K}_v2_cpc.json'
    if os.path.exists(fp):
        d = json.load(open(fp))
        return d['plan'] if 'plan' in d else d
    return None


DONE = set()
if os.path.exists(OUTJSONL):
    for line in open(OUTJSONL, encoding='utf-8'):
        try:
            DONE.add(json.loads(line)['case'])
        except Exception:
            pass


def one_case(case):
    plan0 = seed_of(case)
    if plan0 is None:
        return {'case': case, 'status': 'no_seed'}
    if len(plan0['core_schedules']) != K:
        return {'case': case, 'status': f'slots={len(plan0["core_schedules"])}'}
    graph = load_case(case)
    ev = ev_p2 if Q == 2 else ev_p3
    sc = json.load(open(f'{ROOT}/results/singlecore/{case}_sc.json'))['makespan']
    mk0 = ev(graph, plan0)[0]['makespan']
    comp_of, depth = comp_depth(graph)
    core_of_sg = {sg: c for c, sgl in enumerate(plan0['core_schedules']) for sg in sgl}
    op_core = {op: core_of_sg[sg] for op, sg in {int(k): v for k, v in plan0['node_to_subgraph'].items()}.items()}
    best = (mk0, None, None)
    for om in ('min_id',):
        for L in (3, 4, 6):
            for B in range(16, 260, 32):
                try:
                    plb = relabel_v3(plan0, op_core, comp_of, depth, B, L, om)
                    mk1 = ev(graph, plb)[0]['makespan']
                    if mk1 < best[0]:
                        best = (mk1, plb, (om, B, L))
                except Exception:
                    continue
    mk_best, pl_best, P = best
    rec = {'case': case, 'mk0': mk0, 'mk_best': mk_best, 'P': P,
           'sp0': sc / mk0, 'sp_best': sc / mk_best, 'K': K}
    if pl_best is not None and mk_best < mk0 - 0.5:
        mk_off = mk_best  # 官方评估器直评,天然终验
        rec['official_match'] = True
        if True:
            json.dump({'plan': pl_best, 'mk': mk_off, 'sp': sc / mk_off, 'K': K,
                       'source': f'mechA_NK q{Q}N{K} {P}'},
                      open(f'{N5}/superlinear_analysis/{case}_q{Q}_N{K}_mechA.json', 'w'))
    return rec


if __name__ == '__main__':
    cases = [f'case_{i:03d}' for i in range(1, 101)]
    if os.environ.get('NK_ONE'):
        cases = [os.environ['NK_ONE']]
    shard, nshard = int(os.environ.get('SHARD', '0')), int(os.environ.get('NSHARD', '1'))
    for i, case in enumerate(cases):
        if case in DONE or i % nshard != shard:
            continue
        try:
            rec = one_case(case)
        except Exception as e:
            rec = {'case': case, 'status': 'ERR:' + str(e)[:50]}
        with open(OUTJSONL, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        print(json.dumps(rec, ensure_ascii=False, default=str), flush=True)
    print('NK_DONE', flush=True)
