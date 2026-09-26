# -*- coding: utf-8 -*-
"""decoder.py — 学习式核指派的确定性贪心指派解码器。

架构位置（框架 V4 学习层）:
    链 logits + Gumbel-Top-k → 链全局序 chains_order
      → assign_chain_cores()                ← 本模块(确定性, 领域知识内嵌)
      → reinforce_order.relabel_perm(冠军 B×L) → FastEval 真值 → REINFORCE

===== WM / WV 实际使用的 JSON 字段 =====
- 负载: op["cycles"] (int), 逐 op 取 max(1, cycles) 累加; op 缺 "cycles"
  字段时该 op 退化为计 1 (即 op 计数)。仅统计 comp_of 内的非 COPY 算子。
- 管线归属: op["pipe"] == "PIPE_M" 计入 WM, 其余 (PIPE_V / PIPE_MTE* / 缺省)
  计入 WV。实测数据里非 COPY 算子只有 PIPE_M/PIPE_V 两种 (MTE 只出现在被
  排除的 COPY_IN/OUT 上), MTE 归 V 是保守兜底。
- 张量字节: tensors[].size; 生产/消费关系由 edges[] 的 op↔tensor 邻接推出
  (source 为 op & target 为 tensor ⇒ 生产; 反之 ⇒ 消费), 与
  v3_solver/common.tensor_views 同一语义, 但本模块零第三方依赖自行解析。

===== 链 (chain) 定义 =====
链 = 弱连通分量 id（s2_features.comp_depth）, 与重标器
reinforce_order.relabel_perm 的 comp_of 完全同一套 id。
feats["chains"] 按链内最小 op id 升序 (= 重标器 min_id 序)。

===== 设计注意 1: 不做巨链切分 =====
一条链整体映射到一个核 (op 的核 = 其链的核)。case_064 这类全图单弱连通
分量的例子会退化为"全图单核"—— 这是实测正确的（case_064 型全图单分量例:
按层深等分摊核反而显著差于单核, 跨核 500cyc 延迟主导）。巨链切分若要引入应作为显式参数, 默认关闭。

===== 设计注意 2: comp_depth 分解下 pair_bytes/shared 恒为空 =====
弱连通分量以张量为边: 同一张量的生产 op 与全部消费 op 必然连通、落进同一
条链, 故 pair_bytes (跨链生产-消费) 与 shared (≥2 条消费链) 在全 100 例
实测均为空集 —— w_traffic / w_spread 两项在此分解下是结构性死代码,
解码器退化为纯双管负载均衡 (自测 case_029: 9383 vs 随机指派 13339)。
两项保留的意义: 若训练器换用更细的链分解 (如 strand 切, 真的会把张量
切到链间), 这两项立即生效, 接口不变。
"""
from collections import defaultdict

# 兜底 params 默认值 (与接口文档一致)
DEFAULT_PARAMS = {'w_traffic': 1.0, 'w_spread': 0.3}


def chain_features(graph, comp_of):
    """预计算链特征。

    返回 dict:
      chains : list[int]  全部链 id, 按链内最小 op id 升序 (与重标器 min_id 序一致)
      WM, WV : dict chain->float  链的 M管/V管 负载 (字段见模块 docstring)
      pair_bytes : dict (a,b)->float  链 a 与链 b 之间的共享张量总字节
                   (a<b 无向; 张量一侧生产 op、另一侧消费 op, 跨链才计;
                    同一张量对同一无序链对只计一次 —— 与 FastEval 的
                    跨核拷贝按 (生产核, 消费核) 对计费语义一致)
      shared : list of (tensor_bytes, [consumer_chains], producer_chain)
               多消费者张量 (≥2 条不同消费链), P3 摊开奖励用;
               图输入张量无生产 op, 不在 tprod 中, 天然被跳过
      --- 以下为附加缓存键 (纯派生, 供解码器 O(1) 查表) ---
      adj : dict chain -> {邻居链: pair_bytes}    pair_bytes 的邻接表
      shared_by_consumer : dict chain -> [shared 下标]  消费链索引
      n_ops : int  comp_of 覆盖的 op 数 (自检用)
    """
    op_ids = set()
    pipe_of, cyc_of = {}, {}
    for o in graph.get('ops', []):
        oid = o['id']
        op_ids.add(oid)
        pipe_of[oid] = o.get('pipe', 'PIPE_V')
        cyc_of[oid] = max(1, o['cycles']) if 'cycles' in o else 1

    tsize = {t['id']: t.get('size', 0) for t in graph.get('tensors', [])}

    # ---- 链集合与双管负载 ----
    chain_ops = defaultdict(list)
    for op, k in comp_of.items():
        chain_ops[k].append(op)
    chains = sorted(chain_ops, key=lambda k: min(chain_ops[k]))
    WM, WV = {}, {}
    for k, ops in chain_ops.items():
        wm = wv = 0.0
        for o in ops:
            if pipe_of.get(o, 'PIPE_V') == 'PIPE_M':
                wm += cyc_of.get(o, 1)
            else:
                wv += cyc_of.get(o, 1)
        WM[k], WV[k] = float(wm), float(wv)

    # ---- tensor -> 生产 op / 消费 ops (经 edges 的 op↔tensor 邻接) ----
    prod, cons = defaultdict(set), defaultdict(set)
    for e in graph.get('edges', []):
        s, t = e['source'], e['target']
        if s in op_ids and t not in op_ids:
            prod[s].add(t)
        elif t in op_ids and s not in op_ids:
            cons[t].add(s)
    tprod = {}
    for o, ts in prod.items():
        for t in ts:
            tprod[t] = o          # 实测数据恒为单生产者; 多生产者取遍历最后一个
    tcons = defaultdict(set)
    for o, ts in cons.items():
        for t in ts:
            tcons[t].add(o)

    # ---- 跨链 pair_bytes 与多消费者张量 ----
    pair_bytes = defaultdict(float)
    shared = []
    for t, p in tprod.items():
        pk = comp_of.get(p)
        if pk is None:
            continue
        cks = {comp_of[c] for c in tcons.get(t, ()) if c in comp_of}
        if not cks:
            continue
        sz = float(tsize.get(t, 0))
        if sz > 0.0:
            for ck in cks:
                if ck != pk:
                    ab = (pk, ck) if pk < ck else (ck, pk)
                    pair_bytes[ab] += sz
        if len(cks) >= 2:                       # ≥2 条不同消费链 ⇒ 可摊开
            shared.append((sz, sorted(cks), pk))

    adj = defaultdict(dict)
    for (a, b), v in pair_bytes.items():
        adj[a][b] = v
        adj[b][a] = v
    shared_by_consumer = defaultdict(list)
    for i, (_, cks, _pk) in enumerate(shared):
        for ck in cks:
            shared_by_consumer[ck].append(i)

    return {
        'chains': chains,
        'WM': WM,
        'WV': WV,
        'pair_bytes': dict(pair_bytes),
        'shared': shared,
        'adj': {k: dict(v) for k, v in adj.items()},
        'shared_by_consumer': dict(shared_by_consumer),
        'n_ops': len(comp_of),
    }


def assign_chain_cores(chains_order, feats, comp_of, n_cores, q=3, params=None):
    """按 chains_order (链 id 序列, 先=优先) 贪心指派。

    对每条链 k 选核 c 最小化:
      cost(c) = 负载项 + w_traffic × Σ_{已指派到其他核的链 j} pair_bytes(k,j)
                (q==3 再减去多消费者张量的摊开奖励, 见下)
      负载项 = max(WM[c]+WM[k], WV[c]+WV[k])   (双管线, 取 max 而非和)
    负载项说明: 字面增量 max(WM[c]+WM[k],WV[c]+WV[k]) − max(WM[c],WV[c])
      对单管链 (如全 PIPE_V 的 case_029, 每链 inc ≡ WV[k] 与核无关) 是平坦的,
      叠加"与前驱同核"平手规则会把全部链堆到一核 (实测 mk 37555 = 全图单核,
      反而差于随机指派 13339)。故负载项取指派后的核负载 max(...) 本身
      (= 增量 + 当前负载), 既保住 "取 max 而非和" 的双管线语义, 又恢复
      list-scheduling 的均衡性 (LPT 式贪心)。
    q==3 额外: 对 shared 中多消费者张量 T, 若 T 的已指派消费链覆盖 >=2 个核
      且 c 是未被覆盖的新核, 奖励 w_spread × bytes(T) × 已覆盖核数 / 总消费链数
      (目标: P3 的 L2 命中率 —— 消费方跨核摊开 + FIFO 读序对齐)。
    平手规则: 优先选与前一条已指派链同核 (省跨核延迟); 无前驱取最小核号。
    params 默认 {'w_traffic': 1.0, 'w_spread': 0.3}。
    返回 op_core: {op_id: core_int} (op 的核 = 其链的核; 链不切分)。
    纯 python, 复杂度 O(链数×核数 + 邻接查表增量维护)。
    """
    prm = dict(DEFAULT_PARAMS)
    if params:
        prm.update(params)
    w_tr = float(prm['w_traffic'])
    w_sp = float(prm['w_spread'])
    n_cores = max(1, int(n_cores))

    chains = feats['chains']
    WM, WV = feats['WM'], feats['WV']

    adj = feats.get('adj')
    if adj is None:                            # 兜底: 从 pair_bytes 重建
        adj = defaultdict(dict)
        for (a, b), v in feats['pair_bytes'].items():
            adj[a][b] = v
            adj[b][a] = v
    shared = feats['shared']
    sbc = feats.get('shared_by_consumer')
    if sbc is None:
        sbc = defaultdict(list)
        for i, (_sz, cks, _pk) in enumerate(shared):
            for ck in cks:
                sbc[ck].append(i)

    # 防御: chains_order 里不在链集合的 id 丢弃; 缺的链按 min_id 序补在最后
    chain_set = set(chains)
    in_order = set(chains_order)
    todo = [k for k in chains_order if k in chain_set]
    todo += [k for k in chains if k not in in_order]

    coreWM = [0.0] * n_cores
    coreWV = [0.0] * n_cores
    core_of = {}
    apt = defaultdict(float)                    # 链 -> 已指派邻居的 pair 字节总量
    apc = defaultdict(lambda: [0.0] * n_cores)  # 链 -> 各核上已指派邻居的字节
    covered = [set() for _ in shared]           # 多消费者张量已覆盖的核
    prev_core = None

    for k in todo:
        wmk, wvk = WM.get(k, 0.0), WV.get(k, 0.0)
        t_assigned = apt.get(k, 0.0)
        on_core = apc[k]
        sh_idx = sbc.get(k, ()) if (q == 3 and w_sp != 0.0) else ()
        costs = []
        for c in range(n_cores):
            aM, aV = coreWM[c] + wmk, coreWV[c] + wvk
            load = aM if aM > aV else aV          # 指派后核负载, 取 max 而非和
            cost = load + w_tr * (t_assigned - on_core[c])
            for i in sh_idx:
                S = covered[i]
                if len(S) >= 2 and c not in S:
                    cost -= w_sp * shared[i][0] * len(S) / len(shared[i][1])
            costs.append(cost)
        cmin = min(costs)
        c = costs.index(cmin)
        if prev_core is not None and costs[prev_core] <= cmin + 1e-9:
            c = prev_core                       # 平手: 与前驱链同核
        # 提交
        coreWM[c] += wmk
        coreWV[c] += wvk
        core_of[k] = c
        for nb, b in adj.get(k, {}).items():
            apt[nb] += b
            apc[nb][c] += b
        for i in sbc.get(k, ()):
            covered[i].add(c)
        prev_core = c

    return {op: core_of[kk] for op, kk in comp_of.items()}
