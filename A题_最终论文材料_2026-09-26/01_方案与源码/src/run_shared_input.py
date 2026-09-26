import sys, json, os, time
from collections import defaultdict
K = 5


def plan_from_core(g, real, assign, ops, nseg=4):
    pos = {o: ops.pos.get(o, 0) for o in real}
    mx = max(pos.values()) + 1
    n2s, sched = {}, [[] for _ in range(K)]
    for o in sorted(real, key=lambda o: pos[o]):
        sg = assign[o] * nseg + min(nseg - 1, pos[o] * nseg // mx)
        n2s[str(o)] = sg
        if sg not in sched[assign[o]]:
            sched[assign[o]].append(sg)
    return {'node_to_subgraph': n2s, 'core_schedules': sched}
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p2, ev_p3
from split_order_probe_v2 import Ops
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


ROOT = r'C:/shumo_live/02_求解/A题_2026'
done = set()
if os.path.exists(ROOT + '/n5_push/baseline_hinted.jsonl'):
    for l in open(ROOT + '/n5_push/baseline_hinted.jsonl', encoding='utf-8'):
        if '|shared_input' in l and '"sp"' in l:
            try:
                done.add(json.loads(l)['key'])
            except Exception:
                pass
for i in range(1, 101):
    case = f'case_{i:03d}'
    for q, ev in ((2, ev_p2), (3, ev_p3)):
        key = f'{case}|q{q}|shared_input'
        if key in done:
            continue
        try:
            g = load_case(case)
            ops = Ops(g)
            real = [o['id'] for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT')]
            fb = False
            try:
                plan = b1_shared_input(g, ops, real)
                mk = ev(g, plan)[0]['makespan']
            except Exception:
                fb = True
                n = len(real)
                base_assign = {o: i2 * K // n for i2, o in enumerate(real)}
                plan = plan_from_core(g, real, base_assign, ops)
                mk = ev(g, plan)[0]['makespan']
            sc = json.load(open(f'{ROOT}/results/singlecore/{case}_sc.json'))['makespan']
            rec = {'key': key, 'case': case, 'q': q, 'algo': 'shared_input', 'mk': mk,
                   'sp': round(sc/mk, 4), 'fallback': fb, 'solve_s': 0}
        except Exception as e:
            rec = {'key': key, 'case': case, 'q': q, 'algo': 'shared_input', 'status': 'ERR:' + str(e)[:40]}
        with open(ROOT + '/n5_push/baseline_hinted.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        print(rec.get('key'), str(rec.get('sp', rec.get('status', '')))[:30], flush=True)
print('SI_DONE')
