# P2 GPU 深搜索:批量贪心(每批 512 候选 → node1 GPU 评估 → 取最优为新冠军)
# 邻域: move/order/split 混合单步变异(自当前冠军)
import os, sys, json, random, time, warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')

from common import load_case, ev_p2
from split_order_probe_v2 import Ops, State, kahn_pref, global_reach
import subprocess, tempfile

ROOT = Path(r'C:/shumo_live/02_求解/A题_2026')
OUT = HERE / 'gpu_search_out'; OUT.mkdir(exist_ok=True)
POOL = json.load(open(ROOT / 'audit_20260924_latest/final_preview_q2.json'))


def seed_plan_of(case):
    for sub in ('refined2', 'refined', 'strand_n5'):
        fp = HERE / sub / f'{case}_q2_N5.json'
        if fp.exists():
            d = json.load(open(fp))
            if 'plan' in d:
                return d['plan']
    fp = HERE / 'pull_plans' / f'{case}_q2_pull.json'
    if fp.exists():
        d = json.load(open(fp))
        return d['plan'] if 'plan' in d else d
    return None


def mutate(st, ops, E, Rg, K, rng, work):
    """从状态 st 单步变异,返回新 State(或 None)。"""
    nb = st.nodes_by_sg()
    kind = rng.choices(['move', 'order', 'split'], [5, 4, 4])[0]
    cand = None
    if kind == 'move':
        wk = sorted(st.sg_core, key=lambda s: -work.get(s, 0))
        src_sg = rng.choice(wk[:8])
        loads = defaultdict(float)
        for s, c in st.sg_core.items():
            loads[c] += work.get(s, 0)
        dsts = [c for c in sorted(range(K), key=lambda c: loads[c]) if c != st.sg_core[src_sg]][:2]
        if dsts:
            dst = rng.choice(dsts)
            cs = State(st.emit(), K)
            src_c = cs.sg_core[src_sg]
            cs.sg_core[src_sg] = dst
            E2 = ops.sg_edges(cs.n2s)
            cs.sched[src_c] = [s for s in cs.sched[src_c] if s != src_sg]
            cs.sched[dst].append(src_sg)
            nl = kahn_pref(cs.sched[dst], {s: i for i, s in enumerate(cs.sched[dst])}, E2)
            ns = kahn_pref(cs.sched[src_c], {s: i for i, s in enumerate(cs.sched[src_c])}, E2)
            if nl and ns:
                cs.sched[dst], cs.sched[src_c] = nl, ns
                cand = cs
    elif kind == 'order':
        cores = [c for c in range(K) if len(st.sched[c]) >= 2]
        if cores:
            ci = rng.choice(cores)
            i = rng.randrange(len(st.sched[ci]) - 1)
            a, b = st.sched[ci][i], st.sched[ci][i + 1]
            if b not in Rg.get(a, ()) and a not in Rg.get(b, ()):
                cs = State(st.emit(), K)
                cs.sched[ci][i], cs.sched[ci][i + 1] = cs.sched[ci][i + 1], cs.sched[ci][i]
                cand = cs
    else:
        big = sorted(nb, key=lambda s: -work.get(s, 0))[:4]
        big = [s for s in big if len(nb[s]) >= 4]
        if big:
            src_sg = rng.choice(big)
            nodes = sorted(nb[src_sg], key=lambda n: ops.pos.get(n, 0))
            frac = rng.choice([0.3, 0.5, 0.7])
            cut = max(1, min(len(nodes) - 1, int(len(nodes) * frac)))
            loads = defaultdict(float)
            for s, c in st.sg_core.items():
                loads[c] += work.get(s, 0)
            dsts = [c for c in sorted(range(K), key=lambda c: loads[c]) if c != st.sg_core[src_sg]][:2]
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
                nl = kahn_pref(cs.sched[dst], {s: i for i, s in enumerate(cs.sched[dst])}, E2)
                ns = kahn_pref(cs.sched[st.sg_core[src_sg]], {s: i for i, s in enumerate(cs.sched[st.sg_core[src_sg]])}, E2)
                if nl and ns:
                    cs.sched[dst], cs.sched[st.sg_core[src_sg]] = nl, ns
                    cand = cs
    return cand


def gpu_eval_batch(case, plans, tag):
    """调用 gpu_link/batch_eval.py 批量评估,返回 {idx: mk or None}。"""
    inj = OUT / f'{case}_{tag}_in.jsonl'
    with open(inj, 'w') as f:
        for i, p in enumerate(plans):
            f.write(json.dumps({'case': case, 'idx': i, 'plan': p}) + '\n')
    outj = OUT / f'{case}_{tag}_out.jsonl'
    r = subprocess.run(['py', '-3.11', str(HERE / 'gpu_link/batch_eval.py'),
                        '--in', str(inj), '--out', str(outj), '--gpus', '8', '--pack', 'node1'],
                       capture_output=True, text=True, timeout=1200)
    res = {}
    if outj.exists():
        for line in open(outj, encoding='utf-8'):
            try:
                d = json.loads(line)
            except Exception:
                continue
            res[d.get('idx')] = d.get('mk') if d.get('status') == 'ok' else None
    return res, r.returncode, (r.stderr or '')[-300:]


def search(case, n_batches=14, batch=384, seed=101):
    g = load_case(case)
    ops = Ops(g)
    plan = seed_plan_of(case)
    if plan is None:
        return {'case': case, 'status': 'no_seed'}
    K = 5
    st = State(plan, K)
    E = ops.sg_edges(st.n2s)
    Rg = global_reach(E)
    rng = random.Random(seed)
    sc = json.load(open(ROOT / f'results/singlecore/{case}_sc.json'))['makespan']
    t0 = time.perf_counter()
    best_plan, best_sp = plan, POOL.get(case, 0)
    cur_sp = best_sp
    hist = []
    for bi in range(n_batches):
        work = {sg: sum(ops.op_cycle.get(n, 0) for n in ns)
                for sg, ns in st.nodes_by_sg().items()}
        cands, seen = [], set()
        tries = 0
        while len(cands) < batch and tries < batch * 8:
            tries += 1
            cs = mutate(st, ops, E, Rg, K, rng, work)
            if cs is None:
                continue
            key = json.dumps([cs.n2s.get(n) for n in sorted(cs.n2s)]) + json.dumps(cs.sched)
            if key in seen:
                continue
            seen.add(key)
            cands.append(cs.emit())
        res, rc, err = gpu_eval_batch(case, cands, f'b{bi}')
        if not res:
            return {'case': case, 'status': f'gpu_fail rc={rc} {err}'}
        best_mk, best_cand = None, None
        for i, mk in res.items():
            if mk is not None and (best_mk is None or mk < best_mk):
                best_mk, best_cand = mk, cands[i]
        # 与当前比较:需要当前冠军的 mk(用池值换算)
        cur_mk = sc / cur_sp if cur_sp else None
        if best_mk is not None and (cur_mk is None or best_mk < cur_mk - 0.5):
            st = State(best_cand, K)
            E = ops.sg_edges(st.n2s)
            Rg = global_reach(E)
            cur_sp = sc / best_mk
            best_plan, best_sp = best_cand, cur_sp
            hist.append((bi, best_mk))
        print(f'  {case} batch{bi}: {len(res)} evald, best_mk={best_mk} cur_sp={cur_sp:.4f} '
              f'({time.perf_counter()-t0:.0f}s)', flush=True)
    out = {'case': case, 'sp0': POOL.get(case), 'sp_new': cur_sp,
           'gain': cur_sp - POOL.get(case, 0), 'hist': hist,
           'wall': round(time.perf_counter() - t0, 1)}
    if cur_sp > POOL.get(case, 0) + 1e-9:
        mk_off = ev_p2(g, best_plan)[0]['makespan']
        out.update(official_mk=mk_off, official_sp=sc / mk_off,
                   official_match=(abs(sc / mk_off - cur_sp) < 1e-6))
        if abs(sc / mk_off - cur_sp) < 1e-6:
            json.dump({'plan': best_plan, 'mk': mk_off, 'sp': sc / mk_off},
                      open(OUT / f'{case}_q2_gpu.json', 'w'))
    return out


if __name__ == '__main__':
    cases = sys.argv[1:]
    for c in cases:
        r = search(c)
        print(json.dumps(r, ensure_ascii=False, default=str), flush=True)
