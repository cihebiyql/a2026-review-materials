# 顺序维度探针:固定划分/分核,只重排核内子图顺序(合法邻域=相邻独立对交换,
# 传递闭包判合法性),死锁(-2)或变差即回退。量化"顺序作为未搜索变量"的真实增益。
import sys, json, random, time
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case
from fast_eval_p2 import FastEvalP3
from collections import defaultdict, deque


def sg_dag(g, n2s):
    ins = defaultdict(list); outs = defaultdict(list)
    for e in g['edges']:
        outs[e['source']].append(e['target']); ins[e['target']].append(e['source'])
    E = set(); opids = set(o['id'] for o in g['ops'])
    for t, preds in ins.items():
        if t in opids: continue
        for u in preds:
            for v in outs.get(t, []):
                if u in opids and v in opids:
                    su, sv = n2s.get(u), n2s.get(v)
                    if su is not None and sv is not None and su != sv:
                        E.add((su, sv))
    return E


def reach(sgs, E):
    """子图集合内的可达关系(传递闭包)。"""
    succ = defaultdict(set)
    sset = set(sgs)
    for a, b in E:
        if a in sset and b in sset:
            succ[a].add(b)
    R = {}
    for s in sgs:
        seen = set(); dq = deque([s])
        while dq:
            x = dq.popleft()
            for y in succ[x]:
                if y not in seen:
                    seen.add(y); dq.append(y)
        R[s] = seen
    return R


def probe(case, budget=25, seed=13):
    d = json.load(open(f'refined/{case}_q3_N5.json'))
    plan = d['plan']; g = load_case(case)
    n2s = {int(k): v for k, v in plan['node_to_subgraph'].items()}
    E = sg_dag(g, n2s)
    fe = FastEvalP3(g)
    mk0 = fe.evaluate(plan)[0]
    assert mk0 == d['mk'], (case, mk0, d['mk'])
    rng = random.Random(seed)
    cur = json.loads(json.dumps(plan))
    cur_mk = mk0
    # 每核预计算可达关系(结构不变,复用)
    R = {ci: reach(c, E) for ci, c in enumerate(cur['core_schedules'])}
    n_ok = n_dead = n_bad = 0
    best_mk = mk0
    t0 = time.perf_counter()
    for it in range(budget):
        if time.perf_counter() - t0 > 55: break
        ci = rng.randrange(len(cur['core_schedules']))
        c = cur['core_schedules'][ci]
        if len(c) < 2: continue
        i = rng.randrange(len(c) - 1)
        a, b = c[i], c[i + 1]
        if b in R[ci].get(a, ()) or a in R[ci].get(b, ()):
            continue  # 有传递依赖,交换非法
        cand = json.loads(json.dumps(cur))
        cc = cand['core_schedules'][ci]
        cc[i], cc[i + 1] = cc[i + 1], cc[i]
        try:
            mk = fe.evaluate(cand)[0]
        except RuntimeError as ex:
            if 'error -2' in str(ex):
                n_dead += 1
            else:
                n_bad += 1
            continue
        n_ok += 1
        if mk < cur_mk - 0.5:
            cur, cur_mk = cand, mk
            best_mk = min(best_mk, mk)
            R[ci] = reach(c, E)
    return case, mk0, best_mk, n_ok, n_dead, n_bad


if __name__ == '__main__':
    for case in ['case_005', 'case_047', 'case_086', 'case_035', 'case_056', 'case_048']:
        case, mk0, best, n_ok, n_dead, n_bad = probe(case)
        d_sp = (mk0 - best) / mk0 * 100
        print(f'{case}: mk {mk0} -> {best} ({d_sp:+.2f}%, sp {57225/mk0:.0f}基线相对) '
              f'| evals ok={n_ok} dead={n_dead} err={n_bad}', flush=True)
