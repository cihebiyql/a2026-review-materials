# P1 真值精修:搬移+顺序邻域(FastEvalP1 bit-exact 目标, 官方 ev_p1 终验)
# P1 场景A:每子图独立 Task;邻域只做 move/order(分裂会加 Task 边界开销,首轮不用)
import os, sys, json, random, time, warnings
warnings.filterwarnings('ignore')
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')

from common import load_case, ev_p1
from fast_eval_p1 import FastEvalP1
from split_order_probe_v2 import Ops, State, kahn_pref, global_reach

ROOT = Path(r'C:/shumo_live/02_求解/A题_2026')
OUT = HERE / 'p1_refine_out'; OUT.mkdir(exist_ok=True)
VER = json.load(open(ROOT / 'audit_20260924_latest/verified_merge_q1.json'))


def refine_p1(case, evals_cap=500, wall_s=280, seed=59):
    g = load_case(case)
    ops = Ops(g)
    fe = FastEvalP1(g)
    seed_fp = None
    for cand_fp in (OUT / f'{case}_q1_refined.json', HERE / f'p1_seeds/{case}_seed.json'):
        if cand_fp.exists():
            seed_fp = cand_fp; break
    d = json.load(open(seed_fp))
    plan = d['plan'] if 'plan' in d else d
    K = 5
    st = State(plan, K)
    mk0 = fe.evaluate(plan)[0]
    cur_mk = mk0
    E = ops.sg_edges(st.n2s)
    Rg = global_reach(E)
    rng = random.Random(seed)
    stats = {'move': [0, 0], 'order': [0, 0], 'split': [0, 0], 'evals': 0, 'dedup': 0}
    pos_cache = {}
    seen = set()
    t0 = time.perf_counter()
    stall = 0
    while stats['evals'] < evals_cap and time.perf_counter() - t0 < wall_s:
        nb = st.nodes_by_sg()
        work = {sg: sum(ops.op_cycle.get(n, 0) for n in ns) for sg, ns in nb.items()}
        kind = rng.choices(['move', 'order', 'split'], [5, 3, 4])[0]
        cand = None
        if kind == 'split':
            big = sorted(nb, key=lambda s: -sum(ops.op_cycle.get(n, 0) for n in nb[s]))[:3]
            big = [s for s in big if len(nb[s]) >= 4]
            if big:
                src_sg = rng.choice(big)
                nodes = sorted(nb[src_sg], key=lambda n: ops.pos.get(n, 0))
                frac = rng.choice([0.3, 0.5, 0.7])
                cut = max(1, min(len(nodes) - 1, int(len(nodes) * frac)))
                loads = defaultdict(float)
                for s, c in st.sg_core.items():
                    loads[c] += work.get(s, 0)
                dsts = sorted(range(K), key=lambda c: loads[c])
                dsts = [c for c in dsts if c != st.sg_core[src_sg]][:2]
                if dsts:
                    dst = rng.choice(dsts)
                    cs = State(st.emit(), K)
                    new_sg = cs.next_id
                    for n in nodes[cut:]:
                        cs.n2s[n] = new_sg
                    cs.sg_core[new_sg] = dst
                    cs.next_id = new_sg + 1
                    E2 = ops.sg_edges(cs.n2s)
                    cs.sched[dst].append(new_sg)
                    pos0 = {s: i for i, s in enumerate(cs.sched[dst])}
                    new_l = kahn_pref(cs.sched[dst], pos0, E2)
                    pos0b = {s: i for i, s in enumerate(cs.sched[st.sg_core[src_sg]])}
                    new_s = kahn_pref(cs.sched[st.sg_core[src_sg]], pos0b, E2)
                    if new_l and new_s:
                        cs.sched[dst], cs.sched[st.sg_core[src_sg]] = new_l, new_s
                        cand = cs
        elif kind == 'move':
            wk = sorted(st.sg_core, key=lambda s: -work.get(s, 0))
            src_sg = rng.choice(wk[:6])
            loads = defaultdict(float)
            for s, c in st.sg_core.items():
                loads[c] += work.get(s, 0)
            dsts = sorted(range(K), key=lambda c: loads[c])
            dsts = [c for c in dsts if c != st.sg_core[src_sg]][:2]
            if dsts:
                dst = rng.choice(dsts)
                cs = State(st.emit(), K)
                src_c = cs.sg_core[src_sg]
                cs.sg_core[src_sg] = dst
                E2 = ops.sg_edges(cs.n2s)
                cs.sched[src_c] = [s for s in cs.sched[src_c] if s != src_sg]
                cs.sched[dst].append(src_sg)
                pos0a = {s: i for i, s in enumerate(cs.sched[dst])}
                new_l = kahn_pref(cs.sched[dst], pos0a, E2)
                pos0b = {s: i for i, s in enumerate(cs.sched[src_c])}
                new_s = kahn_pref(cs.sched[src_c], pos0b, E2)
                if new_l and new_s:
                    cs.sched[dst], cs.sched[src_c] = new_l, new_s
                    cand = cs
        else:
            cores = [c for c in range(K) if len(st.sched[c]) >= 2]
            if cores:
                ci = rng.choice(cores)
                i = rng.randrange(len(st.sched[ci]) - 1)
                a, b = st.sched[ci][i], st.sched[ci][i + 1]
                if b not in Rg.get(a, ()) and a not in Rg.get(b, ()):
                    cs = State(st.emit(), K)
                    cs.sched[ci][i], cs.sched[ci][i + 1] = cs.sched[ci][i + 1], cs.sched[ci][i]
                    cand = cs
        if cand is None:
            continue
        key = json.dumps([cand.n2s.get(n) for n in sorted(cand.n2s)]) + json.dumps(cand.sched)
        if key in seen:
            stats['dedup'] += 1
            stall += 1
            if stall > 8000:
                break
            continue
        stall = 0
        seen.add(key)
        stats['evals'] += 1
        try:
            mk = fe.evaluate(cand.emit())[0]
        except Exception:
            stats[kind][1] += 1
            continue
        if mk < cur_mk - 0.5:
            st, cur_mk = cand, mk
            E = ops.sg_edges(st.n2s)
            Rg = global_reach(E)
            stats[kind][0] += 1
        else:
            stats[kind][1] += 1
    # 官方终验(仅当有改进)
    res = {'case': case, 'mk0': mk0, 'best': cur_mk, 'stats': stats}
    if cur_mk < mk0 - 0.5:
        try:
            mk_off = ev_p1(g, st.emit())[0]['makespan']
            sc = json.load(open(ROOT / f'results/singlecore/{case}_sc.json'))['makespan']
            res.update(official_mk=mk_off, official_match=(mk_off == cur_mk),
                       sp_old=VER.get(case), sp_new=sc / mk_off)
            if mk_off == cur_mk:
                json.dump({'plan': st.emit(), 'mk': mk_off, 'sp': sc / mk_off,
                           'seed_mk0': mk0, 'stats': stats},
                          open(OUT / f'{case}_q1_refined.json', 'w'))
        except Exception as ex:
            res['official_err'] = str(ex)[:80]
    return res


def _worker(args):
    case, cap, wall = args
    try:
        return refine_p1(case, cap, wall)
    except Exception as ex:
        return {'case': case, 'status': 'ERR:' + repr(ex)[:100]}


if __name__ == '__main__':
    import multiprocessing as mp
    cases = [f'case_{i:03d}' for i in range(1, 101)]
    if len(sys.argv) > 1 and sys.argv[1] == 'pilot':
        cases = cases[:10]
    with open(OUT / 'run.log', 'a', encoding='utf-8') as logf:
        with mp.Pool(10) as pool:
            for r in pool.imap_unordered(_worker, [(c, 500, 280) for c in cases]):
                line = json.dumps(r, ensure_ascii=False, default=str)
                print(line, flush=True)
                logf.write(line + '\n'); logf.flush()
