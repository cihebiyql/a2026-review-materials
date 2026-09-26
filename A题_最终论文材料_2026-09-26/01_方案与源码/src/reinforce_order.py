# reinforce_order.py — 链级优先级学习: ICLR23 one-shot priority sampling 的 A题移植
#   方法(Jeon et al., Neural DAG Scheduling via One-Shot Priority Sampling):
#     每链一个 logit; 采样 = logits + i.i.d. Gumbel -> argsort 得全局链序 (Gumbel-Top-k)
#     解码 = mechA relabel(batch B × level L, 复刻 mechA_deep_seq.relabel_L) + FastEval bit-exact
#     训练 = REINFORCE, 批内代价标准化(论文 Eq.13) + logit 范数正则 c=0.001(论文 §3.3)
#   与 B×L 网格的唯一差异: 链序从固定 min_id 变为可学习分布。
# 用法: python reinforce_order.py <Q> <case,...> [--iters 50] [--N 128] [--workers 16] [--seed 0]
# 环境: A2026_ROOT(默认 C:/shumo_live/02_求解/A题_2026) A2026_SC(单核基准目录) A2026_ATT(附件)
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-25
import sys, os, json, time, glob, argparse
from collections import defaultdict
import numpy as np

ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
N5 = ROOT + '/n5_push'
SC_DIR = os.environ.get('A2026_SC', ROOT + '/results/singlecore')
OUT_DIR = os.environ.get('A2026_ROUT', N5 + '/reinforce_out')
sys.path.insert(0, N5 + '/superlinear_analysis')
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
from phase3_mechA2 import comp_depth
from common import load_case, ev_p2, ev_p3
from fast_eval_p2 import FastEvalP2, FastEvalP3

# 逐例配置: B 候选(围绕冠军 B* 或中段代表值), L 固定 4(全部已知增益均为 L=4)
CFG = {
    'case_084': [110, 165, 220, 240],
    'case_095': [24, 48, 72, 96],
    'case_008': [24, 48, 72, 96],
    'case_016': [32, 64, 96, 160, 240],
    'case_002': [4, 8, 16, 32],
}
B_DEFAULT = [16, 48, 96, 160, 240]
L_FIXED = 4
C_LOGITS = 0.001   # 论文 logit 范数正则系数
EPS = 0.1          # 论文 Eq.13 分母截断
LR = 0.05

_G = {}  # fork 前填充, worker 继承


def champion_seed(case, Q):
    best = (0.0, None)
    pats = [f'{N5}/superlinear_analysis/{case}_q{Q}_mechA.json',
            f'{N5}/split_probe_out/{case}_q{Q}_best.json',
            f'{N5}/gpu_search_out/{case}_q{Q}_gpu.json']
    for sub in ('refined2', 'refined', 'strand_n5'):
        pats.append(f'{N5}/{sub}/{case}_q{Q}_*.json')
    pats.append(f'{N5}/pull_plans/{case}_q{Q}_pull.json')
    for pat in pats:
        for fp in glob.glob(pat):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            if d.get('sp') and d['sp'] > best[0] and 'plan' in d:
                best = (d['sp'], d['plan'])
    return best[1], best[0]


def relabel_perm(n_cores, op_core, comp_of, depth, chain_pos, B, L):
    """mechA_deep_seq.relabel_L 逐行复刻, 仅链序从 min_id 换成 chain_pos(学习序)。"""
    core_ops = defaultdict(list)
    for op, c in op_core.items():
        core_ops[c].append(op)
    new_n2s, new_cs = {}, [[] for _ in range(n_cores)]
    sgid = 0
    for c in range(n_cores):
        ops = core_ops.get(c, [])
        comps = defaultdict(list)
        for o in ops:
            comps[comp_of[o]].append(o)
        comp_list = sorted(comps, key=lambda k: chain_pos[k])
        batch, size, batches = [], 0, []
        for k in comp_list:
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


def _eval_one(args):
    """worker: (链全局序的 position 数组, B) -> FastEval makespan。失败返回 None。"""
    order_pos, B = args
    chain_pos = {int(k): float(p) for p, k in enumerate(order_pos)}
    try:
        plan = relabel_perm(_G['n_cores'], _G['op_core'], _G['comp_of'],
                            _G['depth'], chain_pos, B, L_FIXED)
        mk, _ = _G['fe'].evaluate(plan)
        return mk
    except Exception:
        return None


def reinforce_logits(logits, order, costs):
    """REINFORCE 梯度: E[∇logπ(序列)·C̄]。order: (N,n) 采样序; 返回 (n,) 梯度。"""
    N, n = order.shape
    mask = np.ones((N, n), dtype=bool)
    grad = np.zeros((N, n))
    rows = np.arange(N)
    for i in range(n):
        z = np.where(mask, logits[None, :], -np.inf)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        grad -= p
        picked = order[:, i]
        grad[rows, picked] += 1.0
        mask[rows, picked] = False
    return (grad * costs[:, None]).mean(axis=0)


def run_case(case, Q, iters, N, workers, seed):
    t_start = time.time()
    plan0, sp0_seed = champion_seed(case, Q)
    if plan0 is None:
        return {'case': case, 'status': 'no_seed'}
    graph = load_case(case)
    fe = FastEvalP2(graph) if Q == 2 else FastEvalP3(graph)
    sc = json.load(open(f'{SC_DIR}/{case}_sc.json'))['makespan']
    mk0, _ = fe.evaluate(plan0)
    comp_of, depth = comp_depth(graph)
    core_of_sg = {sg: c for c, sgl in enumerate(plan0['core_schedules']) for sg in sgl}
    op_core = {op: core_of_sg[sg] for op, sg in
               {int(k): v for k, v in plan0['node_to_subgraph'].items()}.items()}
    n_cores = len(plan0['core_schedules'])

    # 链集合与基础序(min_id), logit 初始化围绕基础序
    chain_ops = defaultdict(list)
    for o, k in comp_of.items():
        chain_ops[k].append(o)
    chains = sorted(chain_ops, key=lambda k: min(chain_ops[k]))
    n = len(chains)
    base_pos = {k: i for i, k in enumerate(chains)}
    rank = np.array([base_pos[k] for k in chains], dtype=np.float64)
    logits = (1.0 - 2.0 * rank / max(1, n - 1)).astype(np.float64)

    B_CAND = CFG.get(case, B_DEFAULT)
    _G.update(fe=fe, n_cores=n_cores, op_core=op_core, comp_of=comp_of, depth=depth)

    # 起点自检: 基础序(min_id) + 冠军B 应复现网格冠军 mk
    base_order_arr = np.array(chains)
    mk_sanity = _eval_one((base_order_arr, B_CAND[len(B_CAND) // 2]))

    import multiprocessing as mp
    ctx = mp.get_context('fork')
    pool = ctx.Pool(workers)

    rs = np.random.RandomState(seed)
    m = np.zeros_like(logits)
    v = np.zeros_like(logits)
    best = (mk0, None, None)  # (mk, plan, B)
    log = []

    def eval_batch(order_mat, B_arr):
        tasks = [(order_mat[j], int(B_arr[j])) for j in range(order_mat.shape[0])]
        return pool.map(_eval_one, tasks)

    for it in range(iters):
        t0 = time.time()
        G = rs.gumbel(size=(N, n))
        scores = logits[None, :] + G
        order_mat = np.argsort(-scores, axis=1)          # (N,n) 链索引序列
        order_chains = np.array(chains)[order_mat]       # 映射回链 id
        B_arr = np.array(B_CAND)[rs.randint(len(B_CAND), size=N)]
        mks = eval_batch(order_chains, B_arr)
        arr = np.array([mk if mk is not None and mk < 1e15 else np.nan
                        for mk in mks], dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            log.append({'iter': it, 'status': 'all_invalid'})
            continue
        pen = valid.max() * 2 if valid.size else 1e17
        arr[~np.isfinite(arr)] = pen
        C_bar = (arr - arr.mean()) / max(arr.std(), EPS)

        # 采样最优追踪
        j = int(np.nanargmin(arr))
        if arr[j] < best[0]:
            chain_pos = {int(k): float(p) for p, k in enumerate(order_chains[j])}
            plan = relabel_perm(n_cores, op_core, comp_of, depth, chain_pos,
                                int(B_arr[j]), L_FIXED)
            best = (float(arr[j]), plan, int(B_arr[j]))

        # REINFORCE + logit 范数正则 + Adam
        g = reinforce_logits(logits, order_mat, C_bar)
        g += C_LOGITS * 2.0 * logits / n
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** (it + 1))
        vh = v / (1 - 0.999 ** (it + 1))
        logits -= LR * mh / (np.sqrt(vh) + 1e-8)

        # greedy(学习到的确定性序)在每个 B 上评估
        greedy_order = np.array(chains)[np.argsort(-logits)]
        g_mks = eval_batch(np.tile(greedy_order, (len(B_CAND), 1)), np.array(B_CAND))
        for B, gmk in zip(B_CAND, g_mks):
            if gmk is not None and gmk < best[0]:
                chain_pos = {int(k): float(p) for p, k in enumerate(greedy_order)}
                plan = relabel_perm(n_cores, op_core, comp_of, depth, chain_pos, B, L_FIXED)
                best = (gmk, plan, B)

        log.append({'iter': it, 'mean': float(arr.mean()), 'std': float(arr.std()),
                    'best': float(best[0]), 't': round(time.time() - t0, 1)})
        print(f'{case} it{it} mean={arr.mean():.0f} best={best[0]:.0f} '
              f'({time.time()-t0:.1f}s)', flush=True)

    pool.close()
    pool.join()

    rec = {'case': case, 'Q': Q, 'mk0': mk0, 'sp0': sc / mk0, 'seed_sp': sp0_seed,
           'mk_sanity': mk_sanity, 'n_chains': n, 'B_CAND': B_CAND,
           'best': best[0], 'best_B': best[2], 'log': log,
           'wall_s': round(time.time() - t_start, 1)}
    if best[1] is not None and best[0] < mk0 - 0.5:
        ev = ev_p2 if Q == 2 else ev_p3
        mk_off = ev(graph, best[1])[0]['makespan']
        rec['official'] = mk_off
        rec['official_match'] = (mk_off == best[0])
        rec['sp_best'] = sc / mk_off
        os.makedirs(OUT_DIR, exist_ok=True)
        if mk_off == best[0]:
            json.dump({'plan': best[1], 'mk': mk_off, 'sp': sc / mk_off,
                       'source': f'reinforce_order B={best[2]} L={L_FIXED} '
                                 f'iters={iters} N={N} seed={seed}'},
                      open(f'{OUT_DIR}/{case}_q{Q}_reinforce.json', 'w'))
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Q', type=int)
    ap.add_argument('cases')
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    outl = f'{OUT_DIR}/reinforce_q{a.Q}.jsonl'
    for case in a.cases.split(','):
        try:
            rec = run_case(case, a.Q, a.iters, a.N, a.workers, a.seed)
        except Exception as e:
            rec = {'case': case, 'status': 'ERR:' + str(e)[:120]}
        with open(outl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        brief = {k: rec.get(k) for k in ('case', 'status', 'mk0', 'best', 'sp0',
                                         'sp_best', 'official_match', 'wall_s')}
        print('RESULT', json.dumps(brief, ensure_ascii=False, default=str), flush=True)
    print('ALL_DONE', flush=True)
