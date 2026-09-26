# GA 基准(官方评估器版): q2/q3, POP12×GENS6, 大图豁免
import sys, json, os, time, random

ROOT = r'C:/shumo_live/02_求解/A题_2026'
sys.path.insert(0, ROOT + '/v3_solver')
sys.path.insert(0, ROOT + '/n5_push')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p1, ev_p2, ev_p3
from ops_standalone import Ops

K = 5
POP, GENS, ELITE = 12, 6, 3
BIG_SKIP = {14, 16, 21, 28, 47, 67, 79, 91}
OUTJ = ROOT + '/n5_push/baseline_ga.jsonl'


def plan_from_core(real, assign, ops):
    pos = {o: ops.pos.get(o, 0) for o in real}
    mx = max(pos.values()) + 1
    n2s, sched = {}, [[] for _ in range(K)]
    for o in sorted(real, key=lambda o: pos[o]):
        sg = assign[o] * 4 + min(3, pos[o] * 4 // mx)
        n2s[str(o)] = sg
        if sg not in sched[assign[o]]:
            sched[assign[o]].append(sg)
    return {'node_to_subgraph': n2s, 'core_schedules': sched}


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
        if i in BIG_SKIP:
            continue
        case = 'case_%03d' % i
        for q, ev in ((1, ev_p1), (2, ev_p2), (3, ev_p3)):
            key = case + '|q' + str(q)
            if key in done:
                continue
            t0 = time.perf_counter()
            try:
                g = load_case(case)
                ops = Ops(g)
                real = [o['id'] for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT')]

                class _O:
                    def __init__(s2, gg):
                        s2.g = gg

                    def evaluate(s2, p):
                        try:
                            return (ev(s2.g, p)[0]['makespan'], {})
                        except Exception:
                            return (1e12, {})

                fe = _O(g)
                rng = random.Random(7)
                n = len(real)
                base = {}
                for i2, o in enumerate(real):
                    base[o] = i2 * K // n
                pop = [dict(base)]
                for _ in range(POP - 1):
                    c = {}
                    for o in real:
                        c[o] = rng.randrange(K)
                    pop.append(c)
                scores = [fe.evaluate(plan_from_core(real, c, ops))[0] for c in pop]
                for gen in range(GENS):
                    order = sorted(range(POP), key=lambda i2: scores[i2])
                    newpop = []
                    for i3 in order[:ELITE]:
                        newpop.append(dict(pop[i3]))
                    while len(newpop) < POP:
                        pa = pop[order[rng.randrange(POP)]]
                        pb = pop[order[rng.randrange(POP)]]
                        cut = rng.random()
                        child = {}
                        for o in real:
                            child[o] = pa[o] if rng.random() < cut else pb[o]
                        for o in real:
                            if rng.random() < 0.05:
                                child[o] = rng.randrange(K)
                        newpop.append(child)
                    pop = newpop
                    scores = [fe.evaluate(plan_from_core(real, c, ops))[0] for c in pop]
                best_i = min(range(POP), key=lambda i2: scores[i2])
                plan = plan_from_core(real, pop[best_i], ops)
                mk = ev(g, plan)[0]['makespan']
                sc = json.load(open(ROOT + '/results/singlecore/' + case + '_sc.json'))['makespan']
                rec = {'key': key, 'case': case, 'q': q, 'algo': 'genetic', 'mk': mk,
                       'sp': round(sc / mk, 4), 'solve_s': round(time.perf_counter() - t0, 1)}
            except Exception as e:
                rec = {'key': key, 'case': case, 'q': q, 'algo': 'genetic',
                       'status': 'ERR:' + str(e)[:50]}
            with open(OUTJ, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
            print(json.dumps(rec, ensure_ascii=False, default=str)[:120], flush=True)
    print('GA_DONE', flush=True)
