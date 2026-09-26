# mechA deep 顺序版:内联全部逻辑,逐例 append jsonl,崩溃重启续跑
# 用法: py -3.11 mechA_deep_seq.py 3   (q=2/3)
import sys, json, os, glob
from collections import defaultdict

Q = int(sys.argv[1])
BASE = r'C:/shumo_live/02_求解/A题_2026/n5_push'
AUD = r'C:/shumo_live/02_求解/A题_2026/audit_20260924_latest'
sys.path.insert(0, BASE + '/superlinear_analysis')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from phase3_mechA2 import comp_depth
from common import load_case, ev_p2, ev_p3
from fast_eval_p2 import FastEvalP2, FastEvalP3

OUTJSONL = f'{BASE}/mechA_deep_q{Q}.jsonl'
DONE = set()
if os.path.exists(OUTJSONL):
    for line in open(OUTJSONL, encoding='utf-8'):
        try:
            DONE.add(json.loads(line)['case'])
        except Exception:
            pass


def relabel_L(plan, op_core, comp_of, depth, B, L):
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
        order = sorted(comps, key=lambda k: min(comps[k]))
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


def champion_seed(case):
    best = (0.0, None)
    pats = [f'{BASE}/superlinear_analysis/{case}_q{Q}_mechA.json',
            f'{BASE}/split_probe_out/{case}_q{Q}_best.json',
            f'{BASE}/gpu_search_out/{case}_q{Q}_gpu.json']
    for sub in ('refined2', 'refined', 'strand_n5'):
        pats.append(f'{BASE}/{sub}/{case}_q{Q}_*.json')
    pats.append(f'{BASE}/pull_plans/{case}_q{Q}_pull.json')
    for pat in pats:
        for fp in glob.glob(pat):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            if d.get('sp') and d['sp'] > best[0] and 'plan' in d:
                best = (d['sp'], d['plan'])
    return best[1]


def one_case(case):
    plan0 = champion_seed(case)
    if plan0 is None:
        return {'case': case, 'status': 'no_seed'}
    graph = load_case(case)
    fe = FastEvalP2(graph) if Q == 2 else FastEvalP3(graph)
    sc = json.load(open(rf'C:/shumo_live/02_求解/A题_2026/results/singlecore/{case}_sc.json'))['makespan']
    mk0, _ = fe.evaluate(plan0)
    comp_of, depth = comp_depth(graph)
    core_of_sg = {sg: c for c, sgl in enumerate(plan0['core_schedules']) for sg in sgl}
    op_core = {op: core_of_sg[sg] for op, sg in {int(k): v for k, v in plan0['node_to_subgraph'].items()}.items()}
    best = (mk0, None, None)
    for L in (2, 3, 4, 6, 8):
        for B in range(4, 240, 2):
            try:
                plb = relabel_L(plan0, op_core, comp_of, depth, B, L)
                mk1, _ = fe.evaluate(plb)
                if mk1 < best[0]:
                    best = (mk1, plb, (B, L))
            except Exception:
                continue
    mk_best, pl_best, BL = best
    rec = {'case': case, 'mk0': mk0, 'mk_best': mk_best, 'BL': BL,
           'sp0': sc / mk0, 'sp_best': sc / mk_best}
    if pl_best is not None and mk_best < mk0 - 0.5:
        ev = ev_p2 if Q == 2 else ev_p3
        mk_off = ev(graph, pl_best)[0]['makespan']
        rec['official_match'] = (mk_off == mk_best)
        if mk_off == mk_best:
            json.dump({'plan': pl_best, 'mk': mk_off, 'sp': sc / mk_off,
                       'source': f'mechA_deep_seq B={BL[0]} L={BL[1]}'},
                      open(f'{BASE}/superlinear_analysis/{case}_q{Q}_mechA.json', 'w'))
    return rec


SKIP = {'case_014', 'case_040'}  # 已知触发 numba 原生崩溃的大图,跳过(后续用 node1 补)
for i in range(1, 101):
    case = f'case_{i:03d}'
    if case in DONE or case in SKIP:
        continue
    try:
        rec = one_case(case)
    except Exception as e:
        rec = {'case': case, 'status': 'ERR:' + str(e)[:60]}
    with open(OUTJSONL, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
    print(json.dumps(rec, ensure_ascii=False, default=str), flush=True)
print('SEQ_DONE', flush=True)
