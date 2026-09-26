# -*- coding: utf-8 -*-
"""A题2026 官方 P2（场景B）/ P3（缓存）评估器的 numba 复刻。

复刻范围：
  _build_scene_b_tasks（每核合并 Task + 跨核 COPY 对，与 P3 共享）
  → step1 + _prioritize_task_seq（子图优先级稳定桶排）
  → step2 / step3（复用 P1 内核）
  → 场景B事件循环（外部释放堆）
  → P3：+ FIFO Cache（命中走 CACHE_READ 池，未命中走 DDR 池）

与官方对齐的语义要点：
  - 单一 id 计数器 max(op∪tensor)+1 无跳号（与 P1 不同！）
  - 跨核对：先 ddr，再 copy_out，再 copy_in
  - copy 挂靠子图：input/cross 取消费侧最早子图，output/cross 取生产侧最晚子图
  - tensors 字典插入序 = 原图 tensor 升序逐个 touched，copy 的 ddr 紧随其后，
    direct 跨核的 local tensor 在最后（step2 lifecycle 仅非 DDR，天然分段）
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fast_eval_p1 import (  # noqa: E402
    GraphCodec, stage_step1, stage_step2, build_ext_all, stage_step3,
    _transpose_csr, BW, CAP_L1, CAP_UB, T_COPY_IN, T_COPY_OUT)
from stub_multicore_cut_and_schedule import derive_multicore_plan  # noqa: E402

DELAY_B = 500
CACHE_CAP = 1048576
CACHE_BW = 250


# =====================================================================
# 一、场景B任务构造（每核一 Task）
# =====================================================================

class TaskBuildB:
    """P2/P3 的每核任务图，接口与 TaskBuild 兼容（供 step1/2/3 复用）。"""

    def __init__(self, codec, plan_view, bandwidth=BW):
        gc = codec
        n_ops_all = len(gc.op_ids)
        n_ten_all = len(gc.t_ids)
        mapping = plan_view["mapping"]
        num_cores = plan_view["num_cores"]
        core_by_subgraph = plan_view["core_by_subgraph"]
        self.n_tasks = num_cores
        self.num_cores = num_cores
        self.subgraph_ids = list(range(num_cores))  # task=core

        # op 归核
        op_core = np.full(n_ops_all, -1, dtype=np.int64)
        op_sgsub = np.full(n_ops_all, -1, dtype=np.int64)
        for oidx in range(n_ops_all):
            sg = mapping.get(int(gc.op_ids[oidx]))
            if sg is not None:
                op_core[oidx] = core_by_subgraph[sg]
                op_sgsub[oidx] = sg

        # 每核子图序（rank：子图 → 核内序号）
        core_orders = plan_view["core_orders"]
        sg_rank_of_core = [{} for _ in range(num_cores)]
        for c in range(num_cores):
            for r, sg in enumerate(core_orders.get(c, [])):
                sg_rank_of_core[c][sg] = r

        # ---- 逐 tensor（升序）建边与 COPY ----
        prod_o = gc.prod_ops
        cons_o = gc.cons_ops
        prod_t = np.repeat(np.arange(n_ten_all, dtype=np.int64),
                           np.diff(gc.prod_ptr))
        cons_t = np.repeat(np.arange(n_ten_all, dtype=np.int64),
                           np.diff(gc.cons_ptr))
        cons_is_copout = gc.op_type[cons_o] == 2

        next_id = max(gc.max_op_id, gc.max_tid) + 1

        in_pairs = []    # (core, local op slot or None, tensor slot placeholder)
        out_pairs = []
        ci_backings = []  # (core, local tensor gid, ddr gid)：COPY_IN backing
        # 先按核聚合原始 op/tensor，边界信息第二遍填
        # 每 core 的 tensors 用 dict 模拟（保持插入序）
        core_tensors = [dict() for _ in range(num_cores)]  # gid -> (size,pos)
        core_ops = [[] for _ in range(num_cores)]          # dicts
        op_sub = [dict() for _ in range(num_cores)]        # op gid -> sg
        edges_by_core = [[] for _ in range(num_cores)]     # (kind, a, b)
        cross_links = []   # (src_core, dst_core, out_gid, in_gid, size)
        cross_task_traffic = 0
        task_graph_copy_traffic = 0

        def alloc():
            nonlocal next_id
            v = next_id
            next_id += 1
            return v

        def add_copy(core, local_gid, size, sg, ddr_gid=None, kind="in"):
            nonlocal next_id, task_graph_copy_traffic
            if ddr_gid is None:
                ddr_gid = alloc()
            copy_gid = alloc()
            task_graph_copy_traffic += size
            core_tensors[core][ddr_gid] = (size, 0)
            core_ops[core].append({
                "gid": copy_gid, "type": T_COPY_IN if kind == "in"
                else T_COPY_OUT,
                "pipe": 0 if kind == "in" else 1,
                "cycles": max(1, math.ceil(size / bandwidth))})
            op_sub[core][copy_gid] = sg
            if kind == "in":
                edges_by_core[core].append(("t2o", copy_gid, ddr_gid))
                edges_by_core[core].append(("o2t", copy_gid, local_gid))
                ci_backings.append((core, local_gid, ddr_gid))
            else:
                edges_by_core[core].append(("t2o", copy_gid, local_gid))
                edges_by_core[core].append(("o2t", copy_gid, ddr_gid))
            return copy_gid, ddr_gid

        for tidx in range(n_ten_all):
            tid = int(gc.t_ids[tidx])
            size = int(gc.t_size[tidx])
            ps = prod_o[gc.prod_ptr[tidx]:gc.prod_ptr[tidx + 1]]
            cs = cons_o[gc.cons_ptr[tidx]:gc.cons_ptr[tidx + 1]]
            ep = sorted(int(gc.op_ids[p]) for p in ps if op_core[p] >= 0)
            ec = sorted(int(gc.op_ids[c]) for c in cs if op_core[c] >= 0)
            epc = {int(gc.op_ids[p]): int(op_core[p]) for p in ps
                   if op_core[p] >= 0}
            ecc = {int(gc.op_ids[c]): int(op_core[c]) for c in cs
                   if op_core[c] >= 0}
            prod_cores = sorted(set(epc.values()))
            cons_cores = sorted(set(ecc.values()))
            touched = sorted(set(prod_cores) | set(cons_cores))
            if not touched:
                continue
            pos = gc.t_onchip[tidx] if gc.t_onchip[tidx] else 2
            has_copout = bool(np.any(cons_is_copout[
                gc.cons_ptr[tidx]:gc.cons_ptr[tidx + 1]]))
            for core in touched:
                core_tensors[core][tid] = (size, pos)
                for p in ep:
                    if epc[p] == core:
                        edges_by_core[core].append(("o2t", p, tid))
                for c in ec:
                    if ecc[c] == core:
                        edges_by_core[core].append(("t2o", c, tid))
            # 图输入
            if ec and not ep:
                for dst in cons_cores:
                    dst_sgs = {op_sgsub[int(np.searchsorted(gc.op_ids, o))]
                               for o in ec if ecc[o] == dst}
                    dst_sg = min(dst_sgs, key=lambda s:
                                 sg_rank_of_core[dst].get(s, 1 << 30))
                    add_copy(dst, tid, size, dst_sg, kind="in")
            # 图输出
            if ep and (has_copout or not ec):
                for src in prod_cores:
                    src_sgs = {op_sgsub[int(np.searchsorted(gc.op_ids, o))]
                               for o in ep if epc[o] == src}
                    src_sg = max(src_sgs, key=lambda s:
                                 sg_rank_of_core[src].get(s, -1))
                    add_copy(src, tid, size, src_sg, kind="out")
            # 跨核对
            for src in prod_cores:
                src_sgs = {op_sgsub[int(np.searchsorted(gc.op_ids, o))]
                           for o in ep if epc[o] == src}
                src_sg = max(src_sgs, key=lambda s:
                             sg_rank_of_core[src].get(s, -1))
                for dst in cons_cores:
                    if src == dst:
                        continue
                    dst_sgs = {op_sgsub[int(np.searchsorted(gc.op_ids, o))]
                               for o in ec if ecc[o] == dst}
                    dst_sg = min(dst_sgs, key=lambda s:
                                 sg_rank_of_core[dst].get(s, 1 << 30))
                    ddr_gid = alloc()
                    out_gid, _ = add_copy(src, tid, size, src_sg,
                                          ddr_gid=ddr_gid, kind="out")
                    in_gid, _ = add_copy(dst, tid, size, dst_sg,
                                         ddr_gid=ddr_gid, kind="in")
                    cross_links.append((src, dst, out_gid, in_gid, size))
                    cross_task_traffic += size

        # 直接 op-op 边
        for si, di in zip(gc.direct_src, gc.direct_dst):
            sc, dc = int(op_core[si]), int(op_core[di])
            sg_s, sg_d = int(op_sgsub[si]), int(op_sgsub[di])
            sgid_s, sgid_d = int(gc.op_ids[si]), int(gc.op_ids[di])
            if sc < 0 or dc < 0:
                continue
            if sc == dc:
                edges_by_core[sc].append(("oo", sgid_s, sgid_d))
                continue
            size = 0  # 官方 data_size 默认 0
            local_gid, ddr_gid = alloc(), alloc()
            for c in (sc, dc):
                core_tensors[c][local_gid] = (size, 2)
            edges_by_core[sc].append(("o2t", sgid_s, local_gid))
            edges_by_core[dc].append(("t2o", sgid_d, local_gid))
            out_gid, _ = add_copy(sc, local_gid, size, sg_s,
                                  ddr_gid=ddr_gid, kind="out")
            in_gid, _ = add_copy(dc, local_gid, size, sg_d,
                                 ddr_gid=ddr_gid, kind="in")
            cross_links.append((sc, dc, out_gid, in_gid, size))
            cross_task_traffic += size

        # 原始 ops 按核填（official 先 originals 后 copies；平铺时按 gid 排序
        # 保持"槽序 == id 序"不变量）
        for oidx in range(n_ops_all):
            c = int(op_core[oidx])
            if c >= 0:
                core_ops[c].append({
                    "gid": int(gc.op_ids[oidx]), "type": 0,
                    "pipe": int(gc.op_pipe[oidx]),
                    "cycles": int(gc.op_cycles[oidx])})
                op_sub[c][int(gc.op_ids[oidx])] = int(op_sgsub[oidx])

        # ---- 平铺数组（接口兼容 TaskBuild）----
        total_ops = sum(len(core_ops[c]) for c in range(num_cores))
        total_tensors = sum(len(core_tensors[c]) for c in range(num_cores))
        self.t_op_ptr = np.zeros(num_cores + 1, dtype=np.int64)
        self.t_ten_ptr = np.zeros(num_cores + 1, dtype=np.int64)
        op_type = np.zeros(total_ops, dtype=np.int64)
        op_pipe = np.zeros(total_ops, dtype=np.int64)
        op_cycles = np.zeros(total_ops, dtype=np.int64)
        op_gid = np.zeros(total_ops, dtype=np.int64)
        op_rank = np.zeros(total_ops, dtype=np.int64)
        ten_size = np.zeros(total_tensors, dtype=np.int64)
        ten_pos = np.zeros(total_tensors, dtype=np.int64)
        ten_gid = np.zeros(total_tensors, dtype=np.int64)
        ten_backing = np.full(total_tensors, -1, dtype=np.int64)
        in_pairs = []
        out_pairs = []
        direct_pairs = []
        oc = 0
        tc = 0
        op_slot_by_core = []
        for c in range(num_cores):
            self.t_op_ptr[c] = oc
            self.t_ten_ptr[c] = tc
            slot_of_tgid = {}
            for t_gid, (sz, ps_) in core_tensors[c].items():
                ten_gid[tc] = t_gid
                ten_size[tc] = sz
                ten_pos[tc] = ps_
                slot_of_tgid[t_gid] = tc
                tc += 1
            # ops 按 gid 升序平铺（originals 在前，copies 按创建序）
            ordered_ops = sorted(core_ops[c], key=lambda o: o["gid"])
            op_slot = {}
            for o in ordered_ops:
                op_gid[oc] = o["gid"]
                op_type[oc] = o["type"]
                op_pipe[oc] = o["pipe"]
                op_cycles[oc] = o["cycles"]
                sg = op_sub[c][o["gid"]]
                op_rank[oc] = sg_rank_of_core[c].get(sg, 1 << 30)
                op_slot[o["gid"]] = oc
                oc += 1
            for kind, a, b in edges_by_core[c]:
                if kind == "o2t":
                    out_pairs.append((op_slot[a], slot_of_tgid[b]))
                elif kind == "t2o":
                    in_pairs.append((op_slot[a], slot_of_tgid[b]))
                else:
                    direct_pairs.append((c, op_slot[a], op_slot[b]))
            # COPY_IN backing：该核上 input/cross 的 local tensor 以其 ddr 为
            # backing（官方 _find_copy_in_backings 语义，spill 时复用不拷出）
            for (bc, bl, bd) in ci_backings:
                if bc == c and bl in slot_of_tgid:
                    ten_backing[slot_of_tgid[bl]] = slot_of_tgid[bd]
            op_slot_by_core.append(op_slot)
        self.t_op_ptr[num_cores] = oc
        self.t_ten_ptr[num_cores] = tc
        self.op_type = op_type
        self.op_pipe = op_pipe
        self.op_cycles = op_cycles
        self.op_gid = op_gid
        self.op_rank = op_rank
        self.ten_size = ten_size
        self.ten_pos = ten_pos
        self.ten_gid = ten_gid
        self.ten_backing = ten_backing
        self.task_graph_copy_traffic = task_graph_copy_traffic
        self.cross_task_traffic = cross_task_traffic
        self.partition_added = task_graph_copy_traffic -             gc.original_copy_traffic
        # raw 序（官方 set 迭代序复刻需要；edges_by_core 的追加点序）
        self.in_pairs_raw = in_pairs
        self.out_pairs_raw = out_pairs
        in_pairs = sorted(in_pairs)
        out_pairs = sorted(out_pairs)
        self.out_ptr = np.zeros(total_ops + 1, dtype=np.int64)
        for o, _ in out_pairs:
            self.out_ptr[o + 1] += 1
        self.out_ptr = np.cumsum(self.out_ptr)
        self.out_ten = np.array([t for _, t in out_pairs], dtype=np.int64)
        self.in_ptr = np.zeros(total_ops + 1, dtype=np.int64)
        for o, _ in in_pairs:
            self.in_ptr[o + 1] += 1
        self.in_ptr = np.cumsum(self.in_ptr)
        self.in_ten = np.array([t for _, t in in_pairs], dtype=np.int64)
        if direct_pairs:
            dt = np.array([p[0] for p in direct_pairs], dtype=np.int64)
            ds = np.array([p[1] for p in direct_pairs], dtype=np.int64)
            dd = np.array([p[2] for p in direct_pairs], dtype=np.int64)
            order = np.lexsort((dd, ds, dt))
        else:
            dt = ds = dd = np.zeros(0, dtype=np.int64)
            order = np.zeros(0, dtype=np.int64)
        self.direct_src = ds[order] if len(order) else ds
        self.direct_dst = dd[order] if len(order) else dd
        self.direct_task = dt[order] if len(order) else dt
        base_of_core = [int(self.t_op_ptr[c]) for c in range(num_cores)]
        # 跨核链接：存 (src_core, src_local) / (dst_core, dst_local)；
        # 全局 gop = task_op_base[core] + local（场景阶段加基址）
        src_cl = []
        dst_cl = []
        for (sc_, dc_, out_gid, in_gid, sz) in cross_links:
            src_cl.append((sc_, op_slot_by_core[sc_][out_gid]
                           - base_of_core[sc_]))
            dst_cl.append((dc_, op_slot_by_core[dc_][in_gid]
                           - base_of_core[dc_]))
        self.cross_src_cl = src_cl
        self.cross_dst_cl = dst_cl


# =====================================================================
# 二、场景B事件循环宿主编排 + 对外入口
# =====================================================================

def stage_scene_b(tb, s3_list, exts, delay=DELAY_B, use_cache=False,
                  cache_cap=CACHE_CAP, cache_bw=CACHE_BW):
    from scene_b_kernel import scene_b_sim
    n_tasks = tb.num_cores
    task_op_base = np.zeros(n_tasks + 1, dtype=np.int64)
    for k in range(n_tasks):
        task_op_base[k + 1] = task_op_base[k] + s3_list[k]["n_ops"]
    n_gops = int(task_op_base[-1])
    link_src = np.array([task_op_base[c] + s for c, s in tb.cross_src_cl],
                        dtype=np.int64)
    link_dst = np.array([task_op_base[c] + s for c, s in tb.cross_dst_cl],
                        dtype=np.int64)
    n_links = len(link_src)

    gop_pipe = np.zeros(n_gops, dtype=np.int64)
    gop_dur = np.zeros(n_gops, dtype=np.int64)
    gop_is_ddr = np.zeros(n_gops, dtype=np.bool_)
    gop_core = np.zeros(n_gops, dtype=np.int64)
    task_seq_parts = []
    succ_parts = []
    tsp_parts = []
    tsp_ptr_per_task = []
    for k in range(n_tasks):
        s3 = s3_list[k]
        b = task_op_base[k]
        for i in range(s3["n_ops"]):
            gop_pipe[b + i] = int(exts[k]["op_pipe"][i])
            gop_dur[b + i] = int(s3["op_dur"][i])
            gop_is_ddr[b + i] = bool(s3["op_is_ddr"][i])
            gop_core[b + i] = k
        task_seq_parts.append(s3["seq_ext"] + b)
        edges = set()
        for g in range(s3["n_ops"]):
            for e in range(s3["succ_ptr"][g], s3["succ_ptr"][g + 1]):
                edges.add((g, int(s3["succ_arr"][e])))
        for s, d in zip(s3["deps"], s3["depd"]):
            edges.add((int(s), int(d)))
        succ_parts.extend((b + s, b + d) for s, d in edges)
        tsp_parts.append(s3["pipe_seq"])
        tsp_ptr_per_task.append(s3["pipe_seq_ptr"])

    task_seq_arr = np.concatenate(task_seq_parts)
    task_seq_ptr = np.zeros(n_tasks + 1, dtype=np.int64)
    for k in range(n_tasks):
        task_seq_ptr[k + 1] = task_seq_ptr[k] + len(task_seq_parts[k])
    tsp_arr = np.concatenate(tsp_parts)
    tsp_ptr = np.zeros(n_tasks * 5, dtype=np.int64)
    tsp_base = np.zeros(n_tasks, dtype=np.int64)
    acc = 0
    for k in range(n_tasks):
        p = tsp_ptr_per_task[k]
        for q in range(5):
            tsp_ptr[k * 5 + q] = p[q]
        tsp_base[k] = acc
        acc += len(tsp_parts[k])

    if succ_parts:
        comp = np.unique(np.array([s * n_gops + d for s, d in succ_parts],
                                  dtype=np.int64))
        esrc = (comp // n_gops).astype(np.int64)
        edst = (comp % n_gops).astype(np.int64)
    else:
        esrc = np.zeros(0, dtype=np.int64)
        edst = np.zeros(0, dtype=np.int64)
    succ_ptr = np.zeros(n_gops + 1, dtype=np.int64)
    np.add.at(succ_ptr, esrc + 1, 1)
    succ_ptr = np.cumsum(succ_ptr)
    succ_arr = edst[np.argsort(esrc * n_gops + edst, kind="stable")]
    pred_rem = np.zeros(n_gops, dtype=np.int64)
    np.add.at(pred_rem, edst, 1)

    # 外部前驱/后继（跨核链接）
    ext_pred = np.full(n_gops, -1, dtype=np.int64)
    ext_succ_ptr = np.zeros(n_gops + 1, dtype=np.int64)
    np.add.at(ext_succ_ptr, link_src + 1, 1)
    ext_succ_ptr = np.cumsum(ext_succ_ptr)
    ext_succ_arr = link_dst[
        np.argsort(link_src * n_gops + link_dst, kind="stable")]

    # 缓存键（P3）：COPY_IN/spill-in 的非 DDR 输出 tensor 的 logical gid
    cache_idx_of_gop = np.full(n_gops, -1, dtype=np.int64)
    keys = set()
    per_gop_key = {}
    for k in range(n_tasks):
        ext = exts[k]
        b = task_op_base[k]
        op_type = ext["op_type"]
        out_ptr = ext_local_out_ptr(ext)
        out_ten = ext_local_out_ten(ext)
        ten_logical = ext["ten_logical"]
        ten_pos = ext["ten_pos"]
        for i in range(ext["n_ext_ops"]):
            if op_type[i] in (1, 3):
                for e in range(out_ptr[i], out_ptr[i + 1]):
                    t = int(out_ten[e])
                    if ten_pos[t] != 0:
                        lg = int(ten_logical[t])
                        if int(ext["ten_size"][t]) > 0:
                            per_gop_key[b + i] = lg
                            keys.add(lg)
                        break
    key_sorted = np.array(sorted(keys), dtype=np.int64)
    cache_size_of_key = np.zeros(max(1, len(key_sorted)), dtype=np.int64)
    for k in range(n_tasks):
        ext = exts[k]
        for j in range(ext["n_ext_ten"]):
            if ext["ten_pos"][j] != 0:
                lg = int(ext["ten_logical"][j])
                idx = int(np.searchsorted(key_sorted, lg))
                if idx < len(key_sorted) and key_sorted[idx] == lg:
                    cache_size_of_key[idx] = int(ext["ten_size"][j])
    for g, lg in per_gop_key.items():
        cache_idx_of_gop[g] = int(np.searchsorted(key_sorted, lg))

    # scratch
    op_status = np.zeros(n_gops, dtype=np.int64)
    op_end = np.zeros(n_gops, dtype=np.float64)
    cursor_k = np.zeros(n_tasks * 4, dtype=np.int64)
    ready_slot = np.full(n_tasks * 4, -1, dtype=np.int64)
    exec_op = np.full(n_tasks * 4, -1, dtype=np.int64)
    exec_end = np.zeros(n_tasks * 4, dtype=np.float64)
    ddr_pool_op = np.zeros(n_gops + 8, dtype=np.int64)
    ddr_pool_w = np.zeros(n_gops + 8, dtype=np.float64)
    ddr_pool_alive = np.zeros(n_gops + 8, dtype=np.bool_)
    ddr_pool_n = np.zeros(1, dtype=np.int64)
    ddr_slot = np.full(n_gops, -1, dtype=np.int64)
    ddr_last_upd = np.zeros(1, dtype=np.float64)
    sa = np.zeros(max(n_gops + 8, n_links + 8), dtype=np.int64)
    sw = np.zeros(n_gops + 8, dtype=np.float64)
    rel_sched = np.zeros(n_gops, dtype=np.int64)
    heap = np.zeros(n_links + 8, dtype=np.int64)
    heap_n = np.zeros(1, dtype=np.int64)
    n_keys = max(1, len(key_sorted))
    in_cache = np.zeros(n_keys, dtype=np.int64)
    fifo_q = np.zeros(n_keys + 8, dtype=np.int64)
    fifo_sz = np.zeros(n_keys, dtype=np.int64)
    fifo_head = np.zeros(1, dtype=np.int64)
    fifo_tail = np.zeros(1, dtype=np.int64)
    fifo_n = np.zeros(1, dtype=np.int64)
    cache_used = np.zeros(1, dtype=np.int64)
    cache_stats = np.zeros(4, dtype=np.int64)
    cac_pool_op = np.zeros(n_gops + 8, dtype=np.int64)
    cac_pool_w = np.zeros(n_gops + 8, dtype=np.float64)
    cac_pool_alive = np.zeros(n_gops + 8, dtype=np.bool_)
    cac_pool_n = np.zeros(1, dtype=np.int64)
    cac_slot = np.full(n_gops, -1, dtype=np.int64)
    cac_last_upd = np.zeros(1, dtype=np.float64)
    ret_buf = np.zeros(n_tasks * 4 + 8, dtype=np.int64)
    out_mk = np.zeros(1, dtype=np.float64)

    r = scene_b_sim(
        n_tasks, n_gops, n_links,
        gop_pipe, gop_dur, gop_is_ddr,
        link_src, link_dst,
        succ_ptr, succ_arr,
        gop_core, task_op_base,
        task_seq_ptr, task_seq_arr,
        tsp_ptr, tsp_arr, tsp_base,
        op_status, op_end, pred_rem,
        cursor_k, ready_slot,
        exec_op, exec_end,
        delay,
        ddr_pool_op, ddr_pool_w, ddr_pool_alive, ddr_pool_n,
        ddr_slot, ddr_last_upd, sa, sw,
        ext_pred, ext_succ_ptr, ext_succ_arr,
        rel_sched, heap, heap_n,
        1 if use_cache else 0, cache_idx_of_gop, cache_size_of_key,
        cache_cap, cache_bw,
        cac_pool_op, cac_pool_w, cac_pool_alive, cac_pool_n,
        cac_slot, cac_last_upd,
        in_cache, fifo_q, fifo_sz, fifo_head, fifo_tail, fifo_n,
        cache_used, cache_stats, ret_buf,
        out_mk)
    if r < 0:
        raise RuntimeError(f"scene_b_sim error {r} now={out_mk[0]}")
    stats = None
    if use_cache:
        hits, misses, hb, mb = (int(x) for x in cache_stats)
        total_bytes = hb + mb
        stats = {"hits": hits, "misses": misses, "accesses": hits + misses,
                 "hit_bytes": hb, "miss_bytes": mb,
                 "hit_rate": (hb / total_bytes) if total_bytes else 0.0}
    return int(round(out_mk[0])), stats


def ext_local_out_ptr(ext):
    """ext 的 op→tensor out CSR（stage_step3_one 同构，轻量重建）。"""
    n_ops = ext["n_ext_ops"]
    out_ptr = np.zeros(n_ops + 1, dtype=np.int64)
    for o, _ in ext["out_pairs"]:
        out_ptr[o + 1] += 1
    return np.cumsum(out_ptr)


def ext_local_out_ten(ext):
    pairs = sorted(ext["out_pairs"])
    return np.array([t for _, t in pairs], dtype=np.int64)


def _prioritize(tb, seqs):
    """_prioritize_task_seq 复刻：按子图 rank 稳定桶排（保持 step1 序）。"""
    out = []
    for k in range(tb.n_tasks):
        out.append(seqs[k][np.argsort(tb.op_rank[seqs[k]], kind="stable")])
    return out


def _run_p23(graph, plan, gc, use_cache):
    pv = derive_multicore_plan(graph, plan)
    tb = TaskBuildB(gc, pv)
    seqs = _prioritize(tb, stage_step1(tb))
    pss = stage_step2(tb, seqs)
    exts = build_ext_all(tb, seqs, pss)
    s3 = stage_step3(exts)
    mk, cstats = stage_scene_b(tb, s3, exts, use_cache=use_cache)
    spill = sum(e["spill_traffic"] for e in exts)
    task_copy = tb.task_graph_copy_traffic
    original = gc.original_copy_traffic
    info = {
        "makespan": mk,
        "num_cores": tb.num_cores,
        "spill_added_copy_bytes": int(spill),
        "partition_added_copy_bytes": int(task_copy - original),
        "scheduled_copy_bytes": int(task_copy + spill),
        "original_copy_bytes": int(original),
        "added_copy_bytes": int(task_copy - original + spill),
        "cross_task_traffic": int(tb.cross_task_traffic),
        "n_tasks": tb.n_tasks,
    }
    if cstats is not None:
        info["cache_stats"] = cstats
    return mk, info


class FastEvalP2:
    """官方 evaluate_scene_b 的 numba 复刻。"""

    def __init__(self, graph_json):
        self.graph = graph_json
        self.gc = GraphCodec(graph_json)

    def evaluate(self, plan, delay=DELAY_B):
        pv = derive_multicore_plan(self.graph, plan)
        tb = TaskBuildB(self.gc, pv)
        seqs = _prioritize(tb, stage_step1(tb))
        pss = stage_step2(tb, seqs)
        exts = build_ext_all(tb, seqs, pss)
        s3 = stage_step3(exts)
        mk, _ = stage_scene_b(tb, s3, exts, delay=delay, use_cache=False)
        spill = sum(e["spill_traffic"] for e in exts)
        task_copy = tb.task_graph_copy_traffic
        original = self.gc.original_copy_traffic
        info = {
            "makespan": mk,
            "num_cores": tb.num_cores,
            "spill_added_copy_bytes": int(spill),
            "partition_added_copy_bytes": int(task_copy - original),
            "scheduled_copy_bytes": int(task_copy + spill),
            "original_copy_bytes": int(original),
            "added_copy_bytes": int(task_copy - original + spill),
            "cross_task_traffic": int(tb.cross_task_traffic),
        }
        return mk, info


class FastEvalP3(FastEvalP2):
    """官方 evaluate_problem_3 的 numba 复刻（FIFO Cache）。"""

    def evaluate(self, plan, delay=DELAY_B, cache_cap=CACHE_CAP,
                 cache_bw=CACHE_BW):
        pv = derive_multicore_plan(self.graph, plan)
        tb = TaskBuildB(self.gc, pv)
        seqs = _prioritize(tb, stage_step1(tb))
        pss = stage_step2(tb, seqs)
        exts = build_ext_all(tb, seqs, pss)
        s3 = stage_step3(exts)
        mk, cstats = stage_scene_b(tb, s3, exts, delay=delay,
                                   use_cache=True, cache_cap=cache_cap,
                                   cache_bw=cache_bw)
        spill = sum(e["spill_traffic"] for e in exts)
        task_copy = tb.task_graph_copy_traffic
        original = self.gc.original_copy_traffic
        info = {
            "makespan": mk,
            "num_cores": tb.num_cores,
            "spill_added_copy_bytes": int(spill),
            "partition_added_copy_bytes": int(task_copy - original),
            "scheduled_copy_bytes": int(task_copy + spill),
            "original_copy_bytes": int(original),
            "added_copy_bytes": int(task_copy - original + spill),
            "cross_task_traffic": int(tb.cross_task_traffic),
            "cache_stats": cstats,
        }
        return mk, info
