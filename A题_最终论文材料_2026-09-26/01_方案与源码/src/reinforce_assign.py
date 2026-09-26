# reinforce_assign.py — v5 指派层 REINFORCE 训练器 (reinforce_order.py 的指派层上移版)
#   方法(同 ICLR23 one-shot priority sampling 骨架, 解码层换掉):
#     每链一个 logit; 采样 = logits + i.i.d. Gumbel -> argsort 得全局链序 (Gumbel-Top-k)
#     解码 = 贪心核指派解码器(v5_assign_decoder, 领域知识内嵌) -> op->core
#            + mechA relabel(batch B × level L, 复刻 relabel_L; 指派序=重标序)
#            + FastEval bit-exact
#     训练 = REINFORCE, 批内代价标准化(论文 Eq.13) + logit 范数正则 c=0.001(论文 §3.3)
#   与 reinforce_order.py 的唯一结构差异: op->core 不再固定为冠军方案, 每个 rollout
#   由解码器按采样链序现场生成; relabel_perm 的 chain_pos 用采样序本身。
#   --stub 模式: 解码器恒返回冠军 op_core(不依赖 v5_assign_decoder 模块), 且评估直接
#   返回冠军 plan 的 mk(不走 relabel —— B 变化时 relabel(冠军op_core,B) 不再复现冠军),
#   用于训练器管线 bit-exact 自测: 所有 rollout == mk0, best == mk0。
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-25
# 用法: py -3.11 reinforce_assign.py <Q> <case,...> [--iters 50] [--N 128] [--workers 48] [--seed 0] [--stub]
# 环境: A2026_ROOT(默认 C:/shumo_live/02_求解/A题_2026) A2026_SC(单核基准目录)
#       A2026_ATT(附件) A2026_ROUT(默认 N5/reinforce_v5) A2026_DECO_PARAMS(解码器参数 json)
# 纪律: 随机数只用 np.random.RandomState(节点机 numba 0.60 兼容); 本文件不用 numba。
import sys, os, json, time, glob, argparse
from collections import defaultdict
import numpy as np

ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
N5 = ROOT + '/n5_push'
SC_DIR = os.environ.get('A2026_SC', ROOT + '/results/singlecore')
OUT_DIR = os.environ.get('A2026_ROUT', N5 + '/reinforce_v5')
sys.path.insert(0, N5)                              # v5_assign_decoder 并行开发中
sys.path.insert(0, N5 + '/superlinear_analysis')
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
from phase3_mechA2 import comp_depth
from common import load_case, ev_p1, ev_p2, ev_p3
from fast_eval_p2 import FastEvalP2, FastEvalP3
from fast_eval_p1 import FastEvalP1

# 解码器接口(并行开发模块, import 失败兜底到 --stub):
#   chain_features(graph, comp_of) -> feats dict
#   assign_chain_cores(chains_order, feats, comp_of, n_cores, q, params) -> {op_id: core}
try:
    from v5_assign_decoder import chain_features, assign_chain_cores
    HAVE_DECODER = True
except Exception:
    chain_features = None
    assign_chain_cores = None
    HAVE_DECODER = False

# 解码器超参(代价项权重等), 环境变量可覆盖, 不改训练器即可调解码器
DECODER_PARAMS = json.loads(os.environ.get('A2026_DECO_PARAMS', '{}'))

# 逐例配置: B 候选, L 固定 4(全部已知增益均为 L=4)
CFG = {
    'case_084': [110, 165, 220, 240],
    'case_095': [24, 48, 72, 96],
}
B_DEFAULT = [16, 48, 96, 160, 240]
L_FIXED = 4
C_LOGITS = 0.001   # 论文 logit 范数正则系数
EPS = 0.1          # 论文 Eq.13 分母截断
LR = 0.05

_G = {}  # fork 前填充, worker 继承; spawn 时经 initializer 重建


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
    """mechA_deep_seq.relabel_L 逐行复刻, 仅链序从 min_id 换成 chain_pos(采样序)。"""
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


def assign_chain_cores_stub(chains_order, feats, comp_of, n_cores, q, params):
    """stub 解码器: 恒返回冠军方案的 op_core(不依赖 v5_assign_decoder 模块)。"""
    return dict(_G['op_core'])


def _assign_cores(chains_order, feats, comp_of, n_cores, q, params):
    """指派分发: --stub 走冠军 op_core, 否则走 v5_assign_decoder.assign_chain_cores。"""
    if _G['stub']:
        return assign_chain_cores_stub(chains_order, feats, comp_of,
                                       n_cores, q, params)
    return assign_chain_cores(chains_order, feats, comp_of, n_cores, q, params)


def _eval_one(args):
    """worker: (链全局序(链id数组), B) -> FastEval makespan。失败返回 None。
    stub 模式: 解码器恒返回冠军 op_core, 但 relabel(冠军op_core, 采样序, 非冠军B)
    不再复现冠军(只有冠军 op_core+冠军序+冠军B 才有 mk0), 故按约定直接返回冠军
    plan 的 mk, 不走 relabel —— 自测断言: 所有 rollout == mk0。"""
    order_chains, B = args
    try:
        if _G['stub']:
            # 接口自测: 仍走一次解码器调用路径(结果弃用), 但不 relabel
            _assign_cores(list(order_chains), _G['feats'], _G['comp_of'],
                          _G['n_cores'], _G['q'], DECODER_PARAMS)
            return _G['stub_mk']
        op_core_r = _assign_cores(list(order_chains), _G['feats'], _G['comp_of'],
                                  _G['n_cores'], _G['q'], DECODER_PARAMS)
        if _G['topo_pos'] is not None:      # 段模式: 排名相邻同核批(合法性由构造保证)
            plan = relabel_perm_seg(_G['n_cores'], op_core_r, _G['comp_of'],
                                    _G['depth'], B, L_FIXED)
        else:
            chain_pos = {int(k): float(p) for p, k in enumerate(order_chains)}
            plan = relabel_perm(_G['n_cores'], op_core_r, _G['comp_of'],
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


def _init_worker(payload):
    """spawn 模式 initializer: 用主进程传来的 _G 快照重建 worker 全局态。"""
    _G.update(payload)


class _SerialPool:
    """multiprocessing 全不可用时的兜底: 串行 map。"""

    def map(self, fn, tasks):
        return [fn(t) for t in tasks]

    def close(self):
        pass

    def join(self):
        pass


def _make_pool(workers):
    """fork 优先(集群 Linux, _G 由 fork 继承); Windows 本机无 fork -> spawn +
    initializer 重建 _G(FastEval 已实测可 pickle); 再失败 -> 串行兜底。"""
    import multiprocessing as mp
    try:
        return mp.get_context('fork').Pool(workers)
    except ValueError:
        pass
    try:
        return mp.get_context('spawn').Pool(
            workers, initializer=_init_worker, initargs=(dict(_G),))
    except Exception:
        return _SerialPool()


def build_units(comp_of, depth, min_units=96, max_units=192):
    """辫状图自适应: 链(弱连通分量)数不足时, 按 (depth, op_id) 拓扑序切均衡段
    作为指派单元。同 depth 内无依赖边 => 该序为合法拓扑序, 段在拓扑序上连续
    => 凸性安全与 relabel 合法性不变。链数足够时原样返回链参数化。"""
    n_chains = len(set(comp_of.values()))
    if n_chains >= min_units:
        return comp_of
    ops = sorted(comp_of, key=lambda o: (depth[o], o))
    n_seg = min(max_units, max(min_units, len(ops) // 150 or 1))
    size = max(1, -(-len(ops) // n_seg))          # ceil(ops/段数)
    return {o: i // size for i, o in enumerate(ops)}


def relabel_perm_seg(n_cores, op_core, comp_of, depth, B, L):
    """段模式合法重标: 段id=拓扑排名。批 = 全局排名相邻且同核的段连续串
    (超 B 断批), 批内按 L 层窗切子图。批的排名区间不含异核段 => 商图无环
    (所有子图依赖沿排名单调)。段可能被 op_core 切到多核(冠军热启动情形),
    按 (段,核) 碎片处理, 碎片排名=段排名, 等排名碎片间无依赖 => 仍无环。"""
    unit_ops = defaultdict(list)
    for o, u in comp_of.items():
        unit_ops[u].append(o)
    # 全局碎片序列: (段排名, 核) 升序
    pieces = defaultdict(list)          # (u, c) -> ops
    for o, u in comp_of.items():
        pieces[(u, op_core[o])].append(o)
    seq = sorted(pieces, key=lambda uc: (uc[0], uc[1]))
    batches = defaultdict(list)         # core -> [batch(碎片列表)]
    cur_core, cur_batch, cur_size = None, [], 0
    for (u, c) in seq:
        sz = len(pieces[(u, c)])
        if c != cur_core or (cur_batch and cur_size + sz > B):
            if cur_batch:
                batches[cur_core].append(cur_batch)
            cur_batch, cur_size, cur_core = [], 0, c
        cur_batch.append((u, c))
        cur_size += sz
    if cur_batch:
        batches[cur_core].append(cur_batch)

    new_n2s, new_cs = {}, [[] for _ in range(n_cores)]
    sgid = 0
    for c in range(n_cores):
        for batch in batches.get(c, []):
            bset = set(batch)
            bops = [o for (u, cc) in bset for o in pieces[(u, cc)]]
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


class IsolatedPool:
    """平台补丁: 每方案一次子进程评估(v5_isolated_eval.py), 崩溃=None, 主进程免疫。
    解码/重标在主进程完成(纯 Python 无 numba), 子进程只做 load+FastEval+evaluate。"""

    def __init__(self, case, q, workers, pyexe, timeout=300):
        from concurrent.futures import ThreadPoolExecutor
        self.case, self.q, self.pyexe, self.timeout = case, q, pyexe, timeout
        self.n_ok = 0
        self.n_crash = 0
        self.ex = ThreadPoolExecutor(max_workers=workers)
        self.script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'v5_isolated_eval.py')

    def eval_plan(self, plan):
        import subprocess
        try:
            p = subprocess.run(
                [self.pyexe, self.script, self.case, str(self.q)],
                input=json.dumps(plan), capture_output=True, text=True,
                timeout=self.timeout)
            if p.returncode == 0:
                mk = int(json.loads(p.stdout.strip().splitlines()[-1])['mk'])
                self.n_ok += 1
                return mk
        except Exception:
            pass
        self.n_crash += 1
        return None

    def eval_many(self, plans):
        return list(self.ex.map(self.eval_plan, plans))

    def close(self):
        self.ex.shutdown()

    def join(self):
        pass


def run_case(case, Q, iters, N, workers, seed, stub=False,
             isolated=False, pyexe=None):
    t_start = time.time()
    plan0, sp0_seed = champion_seed(case, Q)
    if plan0 is None:
        return {'case': case, 'status': 'no_seed'}
    if not stub and not HAVE_DECODER:
        return {'case': case, 'status': 'ERR:v5_assign_decoder 模块不可用(或加 --stub)'}
    graph = load_case(case)
    fe = {1: FastEvalP1, 2: FastEvalP2, 3: FastEvalP3}[Q](graph)
    sc = json.load(open(f'{SC_DIR}/{case}_sc.json'))['makespan']
    mk0, _ = fe.evaluate(plan0)
    comp_of, depth = comp_depth(graph)
    n_raw = len(set(comp_of.values()))
    comp_of = build_units(comp_of, depth)   # 辫状图(链过少)切换为拓扑段单元
    n_units_in = len(set(comp_of.values()))
    seg_mode = n_units_in != n_raw  # 段模式: 段间有依赖,重标序必须用拓扑序(段id即拓扑排名)
    if seg_mode:
        topo_pos = {k: float(k) for k in set(comp_of.values())}
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
    # 解码器特征预计算一次(stub 模式不需要解码器模块)
    feats = chain_features(graph, comp_of) if (HAVE_DECODER and not stub) else {}
    _G.update(fe=fe, n_cores=n_cores, op_core=op_core, comp_of=comp_of,
              depth=depth, feats=feats, q=Q, stub=stub, stub_mk=mk0,
              topo_pos=topo_pos if seg_mode else None)

    # 起点自检: 基础序(min_id) + 中位 B 经 解码器+重标 评估(stub 恒为 mk0);
    # 同时探测 pool 可用性(spawn 失败自动降级串行)
    base_order_arr = np.array(chains)

    def build_plan(order_chains_row, B):
        """主进程重建 plan: 解码器确定性(纯 Python, 平台安全) -> 与子进程结果一致。"""
        op_core_r = _assign_cores(list(order_chains_row), feats, comp_of,
                                  n_cores, Q, DECODER_PARAMS)
        if _G['topo_pos'] is not None:
            return relabel_perm_seg(n_cores, op_core_r, comp_of, depth,
                                    B, L_FIXED)
        chain_pos = {int(k): float(p) for p, k in enumerate(order_chains_row)}
        return relabel_perm(n_cores, op_core_r, comp_of, depth, chain_pos,
                            B, L_FIXED)

    iso = None
    if isolated and not stub:
        iso = IsolatedPool(case, Q, workers, pyexe or sys.executable)

        def eval_batch(order_mat, B_arr):
            plans = [build_plan(order_mat[j], int(B_arr[j]))
                     for j in range(order_mat.shape[0])]
            return iso.eval_many(plans)

        mk_sanity = iso.eval_many(
            [build_plan(base_order_arr, B_CAND[len(B_CAND) // 2])])[0]
    else:
        pool = _make_pool(workers)
        try:
            mk_sanity = pool.map(
                _eval_one, [(base_order_arr, B_CAND[len(B_CAND) // 2])])[0]
        except Exception:
            pool.close()
            pool.join()
            pool = _SerialPool()
            mk_sanity = _eval_one((base_order_arr, B_CAND[len(B_CAND) // 2]))

        def eval_batch(order_mat, B_arr):
            tasks = [(order_mat[j], int(B_arr[j])) for j in range(order_mat.shape[0])]
            return pool.map(_eval_one, tasks)

    rs = np.random.RandomState(seed)
    m = np.zeros_like(logits)
    v = np.zeros_like(logits)
    best = (mk0, None, None)  # (mk, plan, B)
    log = []

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
            plan = build_plan(order_chains[j], int(B_arr[j]))
            best = (float(arr[j]), plan, int(B_arr[j]))

        # REINFORCE + logit 范数正则 + Adam
        g = reinforce_logits(logits, order_mat, C_bar)
        g += C_LOGITS * 2.0 * logits / n
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** (it + 1))
        vh = v / (1 - 0.999 ** (it + 1))
        logits -= LR * mh / (np.sqrt(vh) + 1e-8)

        # greedy(学习到的确定性序)经 解码器+重标, 在每个 B 上评估
        greedy_order = np.array(chains)[np.argsort(-logits)]
        g_mks = eval_batch(np.tile(greedy_order, (len(B_CAND), 1)), np.array(B_CAND))
        for B, gmk in zip(B_CAND, g_mks):
            if gmk is not None and gmk < best[0]:
                best = (gmk, build_plan(greedy_order, B), B)

        log.append({'iter': it, 'mean': float(arr.mean()), 'std': float(arr.std()),
                    'best': float(best[0]), 't': round(time.time() - t0, 1)})
        print(f'{case} it{it} mean={arr.mean():.0f} best={best[0]:.0f} '
              f'({time.time()-t0:.1f}s)', flush=True)

    if iso is not None:
        iso.close()
    else:
        pool.close()
        pool.join()

    rec = {'case': case, 'Q': Q, 'mk0': mk0, 'sp0': sc / mk0, 'seed_sp': sp0_seed,
           'mk_sanity': mk_sanity, 'n_chains': n, 'n_units_in': n_units_in,
           'B_CAND': B_CAND, 'isolated': isolated,
           'iso_ok': getattr(iso, 'n_ok', None), 'iso_crash': getattr(iso, 'n_crash', None),
           'best': best[0], 'best_B': best[2], 'log': log, 'stub': stub,
           'have_decoder': HAVE_DECODER, 'wall_s': round(time.time() - t_start, 1)}
    if best[1] is not None and best[0] < mk0 - 0.5:
        ev = (ev_p1, ev_p2, ev_p3)[Q - 1]
        mk_off = ev(graph, best[1])[0]['makespan']
        rec['official'] = mk_off
        rec['official_match'] = (mk_off == best[0])
        rec['sp_best'] = sc / mk_off
        os.makedirs(OUT_DIR, exist_ok=True)
        if mk_off == best[0]:
            json.dump({'plan': best[1], 'mk': mk_off, 'sp': sc / mk_off,
                       'source': f'reinforce_assign B={best[2]} L={L_FIXED} '
                                 f'iters={iters} N={N} seed={seed} stub={stub}'},
                      open(f'{OUT_DIR}/{case}_q{Q}_reinforce.json', 'w'))

    # logits 落盘(合规两段式的"离线模型参数"): 供 solve_compliant 在线秒级复用
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({'case': case, 'Q': Q, 'logits': [float(x) for x in logits],
               'chains': [int(k) for k in chains], 'seg_mode': seg_mode,
               'B_CAND': B_CAND, 'best_B': best[2], 'L': L_FIXED,
               'n_cores_trained': n_cores, 'iters': iters, 'N_samples': N,
               'seed': seed, 'mk0': mk0},
              open(f'{OUT_DIR}/{case}_q{Q}_logits.json', 'w'))
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Q', type=int)
    ap.add_argument('cases')
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stub', action='store_true')
    ap.add_argument('--isolated', action='store_true',
                    help='每方案一次子进程评估(平台补丁, 崩溃例专用)')
    ap.add_argument('--pyexe', default=sys.executable)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    outl = f'{OUT_DIR}/reinforce_assign_q{a.Q}.jsonl'
    for case in a.cases.split(','):
        try:
            rec = run_case(case, a.Q, a.iters, a.N, a.workers, a.seed, a.stub,
                           isolated=a.isolated, pyexe=a.pyexe)
        except Exception as e:
            rec = {'case': case, 'status': 'ERR:' + str(e)[:120]}
        with open(outl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        brief = {k: rec.get(k) for k in ('case', 'status', 'mk0', 'best', 'sp0',
                                         'sp_best', 'official_match', 'best_B',
                                         'wall_s', 'n_chains')}
        print('RESULT', json.dumps(brief, ensure_ascii=False, default=str), flush=True)
    print('ALL_DONE', flush=True)
