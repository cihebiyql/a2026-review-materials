# 题面建议基准(重写干净版): shared_input/pipe_balance/merge_greedy × 三问, 官方评估器, 环回退topo
import sys, json, os, time
from collections import defaultdict

ROOT = r'C:/shumo_live/02_求解/A题_2026'
sys.path.insert(0, ROOT + '/n5_push')
sys.path.insert(0, ROOT + '/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p1, ev_p2, ev_p3
from ops_standalone import Ops
from baselines import b_topo_equal

K = 5
OUTJ = ROOT + '/n5_push/baseline_hinted.jsonl'


def consumers_map(g):
    """tensor -> 消费它的 op 列表; producer: tensor -> 生产 op"""
    ins = defaultdict(list); outs = defaultdict(list)
    opids = set(o['id'] for o in g['ops'])
    for e in g['edges']:
        outs[e['source']].append(e['target']); ins[e['target']].append(e['source'])
    prod, cons = {}, defaultdict(list)
    for t, ps in ins.items():
        if t in opids:
            continue
        for u in ps:
            if u in opids:
                prod[t] = u
        for v in outs.get(t, []):
            if v in opids:
                cons[t].append(v)
    return prod, cons



def emit(n2s, core_of_sg, order_hint):
    sched = [[] for _ in range(K)]
    for sg in order_hint:
        c = core_of_sg.get(sg)
        if c is not None and sg not in sched[c]:
            sched[c].append(sg)
    return {'node_to_subgraph': {str(o): s for o, s in n2s.items()}, 'core_schedules': sched}



def b1_shared_input(g, ops, real):
    """题面方向: 共享输入的节点聚合同子图; 子图按总功均衡分核."""
    prod, cons = consumers_map(g)
    # union-find 聚合共享同一输入的消费者
    parent = {o: o for o in real}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for t, cs in cons.items():
        if len(cs) > 1:
            for c in cs[1:]:
                union(cs[0], c)
    groups = defaultdict(list)
    for o in real:
        groups[find(o)].append(o)
    # 超大组按拓扑位再切(容量保护)
    cyc = ops.op_cycle
    sg_id = 0
    n2s, core_of_sg, order = {}, {}, []
    gs = sorted(groups.values(), key=lambda grp: min(ops.pos.get(o, 0) for o in grp))
    loads = [0] * K
    for grp in gs:
        w = sum(cyc.get(o, 0) for o in grp)
        MAXSG = 400
        chunks = [grp[i:i+MAXSG] for i in range(0, len(grp), MAXSG)] or [[]]
        for ch in chunks:
            sg_id += 1
            for o in ch:
                n2s[o] = sg_id
            c = min(range(K), key=lambda c2: loads[c2])
            core_of_sg[sg_id] = c
            loads[c] += sum(cyc.get(o, 0) for o in ch)
            order.append(sg_id)
    return emit(n2s, core_of_sg, order)



def b2_pipe_balance(g, ops, real):
    """题面方向: 子图内 M/V 负载均衡——按拓扑层切, 每子图吸纳使 M/V 失衡最小的层段."""
    pipe = {o['id']: (1 if p.get('pipe') == 'PIPE_M' else 0) for o in g['ops'] for p in [o]}
    depth = defaultdict(list)
    for o in real:
        depth[ops.pos.get(o, 0) * 20 // (max(ops.pos.get(x, 0) for x in real) + 1)].append(o)
    keys = sorted(depth)
    n2s, core_of_sg, order = {}, {}, []
    loads = [0] * K
    sg = 0
    m_cur = v_cur = 0
    cur = []
    for kk in keys:
        m_add = sum(pipe.get(o, 0) for o in depth[kk])
        v_add = len(depth[kk]) - m_add
        # 若加入该层使段内失衡恶化且段已足够大, 则切段
        imb_now = abs(m_cur - v_cur); imb_new = abs(m_cur + m_add - (v_cur + v_add))
        if cur and len(cur) >= 32 and imb_new > imb_now:
            sg += 1
            c = min(range(K), key=lambda c2: loads[c2])
            for o in cur:
                n2s[o] = sg
            core_of_sg[sg] = c
            loads[c] += m_cur + v_cur
            order.append(sg)
            cur, m_cur, v_cur = [], 0, 0
        cur += depth[kk]; m_cur += m_add; v_cur += v_add
    if cur:
        sg += 1
        c = min(range(K), key=lambda c2: loads[c2])
        for o in cur:
            n2s[o] = sg
        core_of_sg[sg] = c
        order.append(sg)
    return emit(n2s, core_of_sg, order)



def b3_merge_greedy(g, ops, real, q, fe):
    """题面方向: 从细切分出发贪心合并相邻子图, FastEval 显示负收益即停."""
    plan = b2_pipe_balance(g, ops, real)
    try:
        mk = fe.evaluate(plan)[0]
    except Exception:
        return plan
    improved = True
    guard = 0
    while improved and guard < 8:
        guard += 1
        improved = False
        best_plan, best_mk = plan, mk
        import random as _r
        _r2 = _r.Random(guard)
        for c in ([max(range(K), key=lambda c3: sum(1 for _ in plan['core_schedules'][c3]))] if guard % 2 == 0 else range(K)):
            lst = plan['core_schedules'][c]
            idxs = list(range(len(lst) - 1))
            _r2.shuffle(idxs)
            for i in idxs[:10]:
                cand = json.loads(json.dumps(plan))
                a, b2 = cand['core_schedules'][c][i], cand['core_schedules'][c][i+1]
                # 合并 b 入 a
                cand['core_schedules'][c][i] = a
                cand['core_schedules'][c].pop(i+1)
                for k2, v in cand['node_to_subgraph'].items():
                    if v == b2:
                        cand['node_to_subgraph'][k2] = a
                try:
                    mk2 = fe.evaluate(cand)[0]
                except Exception:
                    continue
                if mk2 < best_mk - 0.5:
                    best_plan, best_mk = cand, mk2
        if best_mk < mk - 0.5:
            plan, mk = best_plan, best_mk
            improved = True
    return plan


done = set()
if os.path.exists(OUTJ):
    for line in open(OUTJ, encoding='utf-8'):
        try:
            done.add(json.loads(line)['key'])
        except Exception:
            pass


if __name__ == '__main__':
    done = set()
    if os.path.exists(OUTJ):
        for line in open(OUTJ, encoding='utf-8'):
            try:
                d = json.loads(line)
                if 'sp' in d:
                    done.add(d['key'])
            except Exception:
                pass
    for i in range(1, 101):
        case = 'case_%03d' % i
        g = load_case(case)
        ops = Ops(g)
        real = [o['id'] for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT')]
        for q, ev in ((1, ev_p1), (2, ev_p2), (3, ev_p3)):

            class _OH:
                def __init__(s2, gg):
                    s2.g = gg

                def evaluate(s2, p):
                    try:
                        return ({1: ev_p1, 2: ev_p2, 3: ev_p3}[q](s2.g, p)[0]['makespan'], {})
                    except Exception:
                        return (1e12, {})

            for name in ('shared_input', 'pipe_balance', 'merge_greedy'):
                key = case + '|q' + str(q) + '|' + name
                if key in done:
                    continue
                t0 = time.perf_counter()
                try:
                    if name == 'shared_input':
                        plan = b1_shared_input(g, ops, real)
                    elif name == 'pipe_balance':
                        plan = b2_pipe_balance(g, ops, real)
                    else:
                        plan = b3_merge_greedy(g, ops, real, q, _OH(g))
                    fb = False
                    try:
                        mk = ev(g, plan)[0]['makespan']
                    except Exception:
                        fb = True
                        plan = b_topo_equal(g, ops, real)
                        mk = ev(g, plan)[0]['makespan']
                    sc = json.load(open(ROOT + '/results/singlecore/' + case + '_sc.json'))['makespan']
                    rec = {'key': key, 'case': case, 'q': q, 'algo': name, 'mk': mk,
                           'sp': round(sc / mk, 4), 'fallback': fb,
                           'solve_s': round(time.perf_counter() - t0, 1)}
                except Exception as e:
                    rec = {'key': key, 'case': case, 'q': q, 'algo': name,
                           'status': 'ERR:' + str(e)[:50]}
                with open(OUTJ, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
                print(json.dumps(rec, ensure_ascii=False, default=str)[:120], flush=True)
    print('HINTED_DONE', flush=True)
