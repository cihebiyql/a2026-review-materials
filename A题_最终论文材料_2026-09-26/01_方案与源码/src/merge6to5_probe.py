# P1 六核降五核第一层:固定切分,15 种核对合并 + 全局 H0 拓扑投影交错
# (GPT 三轮审阅 §1:方向加权只排序不筛选;交错用全核拓扑序投影,不能只看两核)
import sys, json, os, time
from collections import defaultdict, deque
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p1

ROOT = r'C:/shumo_live/02_求解/A题_2026'


def sg_dep_edges(g, n2s):
    """子图级依赖(原图 op->tensor->op 跨子图边)。"""
    ins = defaultdict(list); outs = defaultdict(list)
    for e in g['edges']:
        outs[e['source']].append(e['target']); ins[e['target']].append(e['source'])
    E = set(); opids = set(o['id'] for o in g['ops'])
    for t, preds in ins.items():
        if t in opids: continue
        for u in preds:
            for v in outs.get(t, []):
                if u in opids and v in opids:
                    a, b = n2s.get(u), n2s.get(v)
                    if a is not None and b is not None and a != b:
                        E.add((a, b))
    return E


def topo_of_H0(E_dep, scheds):
    """H0 = 依赖边 ∪ 各核顺序边 的拓扑序(原方案合法故无环)。"""
    succ = defaultdict(set); indeg = defaultdict(int)
    nodes = []
    for lst in scheds:
        for s in lst:
            nodes.append(s)
    for lst in scheds:
        for i in range(len(lst) - 1):
            a, b = lst[i], lst[i + 1]
            if b not in succ[a]:
                succ[a].add(b); indeg[b] += 1
    for a, b in E_dep:
        if b not in succ[a]:
            succ[a].add(b); indeg[b] += 1
    dq = deque([s for s in nodes if indeg[s] == 0])
    out = []
    while dq:
        s = dq.popleft(); out.append(s)
        for v in sorted(succ[s]):
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    return out if len(out) == len(nodes) else None


def probe_merge(case, wall_s=240):
    d = json.load(open(f'{ROOT}/n5_push/pull_plans_q1/{case}_q1_pull.json'))
    plan = d['plan'] if 'plan' in d else d
    g = load_case(case)
    n2s = {int(k): v for k, v in plan['node_to_subgraph'].items()}
    scheds = [list(c) for c in plan['core_schedules']]
    K6 = len(scheds)
    assert K6 == 6, f'{case}: slots={K6}'
    E = sg_dep_edges(g, n2s)
    sc = json.load(open(f'{ROOT}/results/singlecore/{case}_sc.json'))['makespan']
    ver = json.load(open(f'{ROOT}/audit_20260924_latest/verified_merge_q1.json'))
    cur_best_sp = ver[case]
    t0 = time.perf_counter()
    results = []
    seen = set()
    for a in range(6):
        for b in range(a + 1, 6):
            # 两种交错偏好: 全局拓扑序投影 / 双核列表轮替(Kahn 就绪即取)
            for pref in ('topo', 'alt'):
                sigma = topo_of_H0(E, scheds)
                if sigma is None:
                    return {'case': case, 'status': 'H0_CYCLE'}
                merged_set = set(scheds[a]) | set(scheds[b])
                if pref == 'topo':
                    merged = [s for s in sigma if s in merged_set]
                else:
                    la, lb = list(scheds[a]), list(scheds[b])
                    merged = []
                    while la or lb:
                        if la: merged.append(la.pop(0))
                        if lb: merged.append(lb.pop(0))
                new_scheds = [scheds[i] for i in range(6) if i not in (a, b)] + [merged]
                key = json.dumps(new_scheds, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                cand = {'node_to_subgraph': plan['node_to_subgraph'],
                        'core_schedules': new_scheds}
                try:
                    mk = ev_p1(g, cand)[0]['makespan']
                    results.append((sc / mk, f'merge c{a}+c{b}/{pref}', mk))
                except Exception as ex:
                    results.append((0.0, f'merge c{a}+c{b}/{pref}', 'ERR:' + str(ex)[:40]))
                if time.perf_counter() - t0 > wall_s:
                    break
    results.sort(reverse=True)
    best_sp, best_desc, best_mk = results[0]
    return {'case': case, 'cur_sp': cur_best_sp, 'best_merge_sp': best_sp,
            'best_desc': best_desc, 'best_mk': best_mk,
            'gain': best_sp - cur_best_sp, 'n_cand': len(results),
            'top3': results[:3]}


if __name__ == '__main__':
    cases = sys.argv[1:] or ['case_080', 'case_062', 'case_055', 'case_079',
                             'case_013', 'case_082', 'case_008', 'case_025',
                             'case_095', 'case_020', 'case_030', 'case_036']
    out = []
    for c in cases:
        try:
            r = probe_merge(c)
        except Exception as ex:
            r = {'case': c, 'status': 'ERR:' + repr(ex)[:80]}
        print(json.dumps(r, ensure_ascii=False, default=str), flush=True)
        out.append(r)
    gains = [r for r in out if r.get('gain', 0) > 0.02]
    print(f'== ΔS>=0.02 的例数: {len(gains)}/{len(out)}; 总增益 {sum(r.get("gain",0) for r in out):.3f} ==')
    json.dump(out, open(f'{ROOT}/n5_push/merge6to5_results.json', 'w'),
              ensure_ascii=False, indent=1, default=str)
