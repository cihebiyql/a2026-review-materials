# 基准算法族: 四个经典基准 × 三问 × 100 例 × N5, 官方评估器终验
# 1) topo_equal: 拓扑等分 K 块(朴素地板)
# 2) greedy_balance: 贪心负载均衡(按工作量最空闲核)
# 3) heft_like: HEFT 式列表调度(向上秩排序+最早完成核)
# 4) kl_partition: 简化 KGN 图划分(按边割贪心, FM 式单点迁移)
import sys, json, os, time, glob
from collections import defaultdict, deque

ROOT = r'C:/shumo_live/02_求解/A题_2026'
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p1, ev_p2, ev_p3
from ops_standalone import Ops

K = 5
OUTJ = ROOT + '/n5_push/baseline_results.jsonl'


def topo_order(g, ops):
    ins = defaultdict(list); outs = defaultdict(list)
    opids = set(o['id'] for o in g['ops'])
    for e in g['edges']:
        outs[e['source']].append(e['target']); ins[e['target']].append(e['source'])
    indeg = {n: len([p for p in ins[n] if p in opids or True]) for n in set(list(ins) + list(outs))}
    dq = deque(sorted([n for n in indeg if indeg[n] == 0]))
    out = []
    while dq:
        n = dq.popleft(); out.append(n)
        for v in outs[n]:
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    return [n for n in out if n in opids]


def plan_from_assign(g, real_ops, assign, ops):
    """assign: op->core. 每核按拓扑位等宽切 4 段(保证段间只有左->右依赖)."""
    pos = {o: ops.pos.get(o, 0) for o in real_ops}
    mx = max(pos.values()) + 1
    seg = {o: min(3, pos[o] * 4 // mx) for o in real_ops}
    n2s = {str(o): assign[o] * 4 + seg[o] for o in real_ops}
    sched = [[] for _ in range(K)]
    for o in sorted(real_ops, key=lambda o: pos[o]):
        s = n2s[str(o)]
        if s not in sched[assign[o]]:
            sched[assign[o]].append(s)
    return {'node_to_subgraph': n2s, 'core_schedules': sched}


def b_topo_equal(g, ops, real):
    order = topo_order(g, ops)
    n = len(order)
    return plan_from_assign(g, real, {o: i * K // n for i, o in enumerate(order)}, ops)


def b_greedy_balance(g, ops, real, op_cycle):
    order = topo_order(g, ops)
    loads = [0] * K
    assign = {}
    for o in order:
        c = min(range(K), key=lambda c: loads[c])
        assign[o] = c
        loads[c] += op_cycle.get(o, 0) + 1
    return plan_from_assign(g, real, assign, ops)


def b_heft(g, ops, real, op_cycle):
    # 向上秩(近似: 用拓扑位代替关键路径长度以省时)
    order = topo_order(g, ops)
    rank = {o: ops.pos.get(o, 0) for o in order}
    order.sort(key=lambda o: -rank[o])
    # 无可用时间模拟, 用负载+rank 权衡的最早完成
    loads = [0] * K
    assign = {}
    for o in order:
        c = min(range(K), key=lambda c: loads[c] - rank[o] * 0.001)
        assign[o] = c
        loads[c] += op_cycle.get(o, 0) + 1
    return plan_from_assign(g, real, assign, ops)


def b_kl(g, ops, real, op_cycle):
    # 边割贪心 + 单点 FM 迁移(简化版)
    order = topo_order(g, ops)
    n = len(order)
    assign = {o: i * K // n for i, o in enumerate(order)}
    # 邻接(经张量两跳)
    ins = defaultdict(list); outs = defaultdict(list)
    opids = set(o['id'] for o in g['ops'])
    for e in g['edges']:
        outs[e['source']].append(e['target']); ins[e['target']].append(e['source'])
    succ = defaultdict(set)
    for t, ps in ins.items():
        if t in opids:
            continue
        for u in ps:
            for v in outs.get(t, []):
                if u in opids and v in opids:
                    succ[u].add(v)
    # FM: 随机挑边界点试迁, 接受若割边减少
    import random
    rng = random.Random(7)
    def cut(assign):
        s = 0
        for u, vs in succ.items():
            for v in vs:
                if assign.get(u) is not None and assign.get(v) is not None and assign[u] != assign[v]:
                    s += 1
        return s
    cur = cut(assign)
    for _ in range(3000):
        o = rng.choice(order)
        old = assign[o]
        new = rng.randrange(K)
        if new == old:
            continue
        assign[o] = new
        c2 = cut(assign)
        if c2 < cur:
            cur = c2
        else:
            assign[o] = old
    return plan_from_assign(g, real, assign, ops)


BASELINES = {'topo_equal': b_topo_equal, 'greedy_balance': b_greedy_balance,
             'heft_like': b_heft, 'kl_partition': b_kl}

done = set()
if os.path.exists(OUTJ):
    for line in open(OUTJ, encoding='utf-8'):
        try:
            d = json.loads(line)
            done.add((d['case'], d['q'], d['algo']))
        except Exception:
            pass

if __name__ == '__main__':
  for i in range(1, 101):
      case = f'case_{i:03d}'
      g = load_case(case)
      ops = Ops(g)
      op_cycle = ops.op_cycle
      real = [o['id'] for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT')]
      for q, ev in ((1, ev_p1), (2, ev_p2), (3, ev_p3)):
          for name, fn in BASELINES.items():
              if (case, q, name) in done:
                  continue
              t0 = time.perf_counter()
              fb = False
              try:
                  plan = fn(g, ops, real, op_cycle) if name != 'topo_equal' else fn(g, ops, real)
                  mk = ev(g, plan)[0]['makespan']
              except Exception:
                  # 朴素分配触发环校验: 回退拓扑等分(标注), 保证可比
                  plan = b_topo_equal(g, ops, real)
                  mk = ev(g, plan)[0]['makespan']
                  fb = True
              sc = json.load(open(f'{ROOT}/results/singlecore/{case}_sc.json'))['makespan']
              rec = {'case': case, 'q': q, 'algo': name, 'mk': mk,
                     'sp': round(sc / mk, 4), 'fallback': fb,
                     'solve_s': round(time.perf_counter() - t0, 2)}
              with open(OUTJ, 'a', encoding='utf-8') as f:
                  f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
              print(json.dumps(rec, ensure_ascii=False, default=str), flush=True)
  print('BASELINE_DONE', flush=True)
