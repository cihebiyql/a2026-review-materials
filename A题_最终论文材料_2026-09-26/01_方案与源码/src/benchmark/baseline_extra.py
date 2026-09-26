# 新基准: B5 随机搜索(合法随机分配×N取优) / B6 同核驻留复用(题面方向六: 最大化同核生产-消费者对)
import sys, json, os, time, random
from collections import defaultdict

ROOT = r'C:/shumo_live/02_求解/A题_2026'
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p1, ev_p2, ev_p3
from ops_standalone import Ops

K = 5
OUTJ = ROOT + '/n5_push/baseline_extra.jsonl'


def consumers_map(g):
    ins = defaultdict(list); outs = defaultdict(list)
    opids = set(o['id'] for o in g['ops'])
    for e in g['edges']:
        outs[e['source']].append(e['target']); ins[e['target']].append(e['source'])
    prod, cons = {}, defaultdict(list)
    for t, ps in ins.items():
        if t in opids: continue
        for u in ps:
            if u in opids: prod[t] = u
        for v in outs.get(t, []):
            if v in opids: cons[t].append(v)
    return prod, cons


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


def b5_random(g, ops, real, q, fe, seed=11, budget=60):
    rng = random.Random(seed)
    best_plan, best_mk = None, float('inf')
    n = len(real)
    base = {o: i * K // n for i, o in enumerate(real)}
    cands = [dict(base)]
    for _ in range(budget - 1):
        # 随机扰动: 从基线出发随机重分配 10% 节点
        c = dict(base)
        for o in rng.sample(real, max(1, n // 10)):
            c[o] = rng.randrange(K)
        cands.append(c)
    for c in cands:
        try:
            plan = plan_from_core(g, real, c, ops)
            mk = fe.evaluate(plan)[0]
            if mk and mk < best_mk:
                best_plan, best_mk = plan, mk
        except Exception:
            continue
    return best_plan


def b6_core_resident(g, ops, real, q, fe):
    """题面方向六: 同核子图可通过缓存复用驻留数据——把生产者-消费者对尽量同核."""
    prod, cons = consumers_map(g)
    # 按张量字节加权生产-消费亲和
    tsize = {t['id']: t.get('size_bytes', 0) for t in g['tensors']}
    aff = defaultdict(lambda: defaultdict(int))  # op -> op -> bytes
    for t, cs in cons.items():
        u = prod.get(t)
        if u is None: continue
        sz = tsize.get(t, 1)
        for v in cs:
            if v != u:
                aff[u][v] += sz
                aff[v][u] += sz
    # 贪心: 按拓扑序处理, 每个op分配到与它亲和最强的已分配邻居的核; 平衡兜底
    order = sorted(real, key=lambda o: ops.pos.get(o, 0))
    assign = {}
    loads = [0] * K
    cyc = ops.op_cycle
    for o in order:
        best_c, best_aff = None, -1
        for u, a in aff.get(o, {}).items():
            if u in assign:
                c = assign[u]
                if a > best_aff and loads[c] < 3.0 * (sum(cyc.get(x, 0) for x in real) / K):
                    best_c, best_aff = c, a
        if best_c is None:
            best_c = min(range(K), key=lambda c: loads[c])
        assign[o] = best_c
        loads[best_c] += cyc.get(o, 0)
    return plan_from_core(g, real, assign, ops)


done = set()
if os.path.exists(OUTJ):
    for line in open(OUTJ, encoding='utf-8'):
        try:
            d2 = json.loads(line)
            if 'sp' in d2:
                done.add(d2['key'])
        except Exception:
            pass

if __name__ == '__main__':
  shard, nshard = int(os.environ.get('SHARD', '0')), int(os.environ.get('NSHARD', '1'))
  ONE = os.environ.get('EX_ONE')
  for i in (range(1, 101) if not ONE else [int(ONE.split('_')[1])]):
      if not ONE and i % nshard != shard:
          continue
      case = f'case_{i:03d}'
      for q, ev in ((1, ev_p1), (2, ev_p2), (3, ev_p3)):
          for name in ('random_search', 'core_resident'):
              key = f'{case}|q{q}|{name}'
              if key in done:
                  continue
              t0 = time.perf_counter()
              try:
                  g = load_case(case)
                  ops = Ops(g)
                  real = [o['id'] for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT')]
                  fe = None
                  if name == 'random_search':
                      plan = b5_random(g, ops, real, q, fe)
                      if plan is None:
                          plan = plan_from_core(g, real, {o: i2 * K // len(real) for i2, o in enumerate(real)}, ops)
                  else:
                      plan = b6_core_resident(g, ops, real, q, fe)
                  mk = ev(g, plan)[0]['makespan']
                  sc = json.load(open(f'{ROOT}/results/singlecore/{case}_sc.json'))['makespan']
                  rec = {'key': key, 'case': case, 'q': q, 'algo': name, 'mk': mk,
                         'sp': round(sc / mk, 4), 'solve_s': round(time.perf_counter() - t0, 1)}
              except Exception as e:
                  rec = {'key': key, 'case': case, 'q': q, 'algo': name, 'status': 'ERR:' + str(e)[:50]}
              with open(OUTJ, 'a', encoding='utf-8') as f:
                  f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
              print(json.dumps(rec, ensure_ascii=False, default=str), flush=True)
  print('EXTRA_DONE', flush=True)
