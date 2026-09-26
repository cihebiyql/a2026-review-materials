# -*- coding: utf-8 -*-
"""A题2026 官方 P1 评估器的 numba 复刻（bit-exact 目标）。

复刻范围：evaluate_scene_a 全路径
  derive_multicore_plan（官方原函数，host 调用）
  → 任务构造（边界 COPY 插入，host numpy）
  → step1 反向 DFS 拓扑（njit）
  → step2 Belady spill 插入（njit）
  → step3 管道调度 + 内存额度 + DDR 争用（njit）
  → 场景A多核事件循环（njit）

设计纪律：
- 官方 code/ 一字不改；本模块只读官方源码语义，逐段对拍验收；
- 所有数值/顺序语义与官方逐点对齐（排序键、插入序、浮点运算次序）；
- 返回 summarize() 所需字段（makespan / data_movement_bytes /
  cross_task_traffic / num_cores / 每任务 local_makespan / memory_peak），
  不复刻 per_core_timeline / ddr_contention_log 等纯报告字段。

用法：
  from fast_eval_p1 import FastEvalP1
  fe = FastEvalP1(graph_json)                # 每图一次（编码缓存）
  mk, info = fe.evaluate(plan)               # 每方案一次（numba 全程）
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

ATT = Path(os.environ.get("A2026_ATT", r"C:/shumo_live/a_data"))
CODE = ATT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from stub_multicore_cut_and_schedule import (  # noqa: E402
    derive_multicore_plan, EXCLUDED_COPY_TYPES)

# 与官方 config.txt 对齐（common.py 第二轮核验值）
BW = 60
CAP_L1 = 524288
CAP_UB = 131072
CROSS_WAIT_A = 1000
SAME_WAIT_A = 100
PIPES = ("PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_V")  # 官方固定顺序
PIPE_MTE2, PIPE_MTE3, PIPE_M, PIPE_V = 0, 1, 2, 3

# op 类型编码（任务图内只会出现这几种）
T_ELE = 0      # 普通计算 op（原 eligible op）
T_COPY_IN = 1
T_COPY_OUT = 2
T_SPILL_IN = 3   # step2 生成（op 字段 COPY_IN）
T_SPILL_OUT = 4  # step2 生成（op 字段 COPY_OUT）


# =====================================================================
# 一、图编码（每图一次，host numpy）
# =====================================================================

class GraphCodec:
    """把 graph_json 压成 numpy 视图 + 预计算的关联结构。

    约定：所有 op/tensor 用其全局 id 的升序位次作为内部下标
    （oidx/tidx），排序语义与官方按 id 的 sorted() 完全一致。
    """

    def __init__(self, graph_json):
        g = graph_json
        ops = g["ops"]
        tensors = g["tensors"]
        edges = g["edges"]

        # ---- ops（含 COPY 类型；按 id 升序）----
        self.op_ids = np.array(sorted(o["id"] for o in ops), dtype=np.int64)
        n_ops = len(self.op_ids)
        pos_of_op = {int(i): k for k, i in enumerate(self.op_ids)}
        self.op_type = np.zeros(n_ops, dtype=np.int64)  # 0普通 1COPY_IN 2COPY_OUT
        self.op_pipe = np.full(n_ops, -1, dtype=np.int64)
        self.op_cycles = np.ones(n_ops, dtype=np.int64)
        for o in ops:
            k = pos_of_op[o["id"]]
            t = o.get("op")
            if t == "COPY_IN":
                self.op_type[k] = 1
            elif t == "COPY_OUT":
                self.op_type[k] = 2
            self.op_pipe[k] = PIPES.index(o.get("pipe", "PIPE_V"))
            self.op_cycles[k] = max(1, o.get("cycles", 1))
        self.op_eligible = self.op_type == 0

        # ---- tensors（按 id 升序）----
        self.t_ids = np.array(sorted(t["id"] for t in tensors), dtype=np.int64)
        n_ten = len(self.t_ids)
        pos_of_ten = {int(i): k for k, i in enumerate(self.t_ids)}
        self.t_size = np.zeros(n_ten, dtype=np.int64)
        self.t_onchip = np.zeros(n_ten, dtype=np.int64)  # 1=L1, 2=UB, 0=DDR
        for t in tensors:
            k = pos_of_ten[t["id"]]
            self.t_size[k] = int(t["size"])
            self.t_onchip[k] = {"L1": 1, "UB": 2}.get(t.get("pos", "UB"), 0)

        # ---- 边：op↔tensor 关联（producers/consumers CSR，按 tensor 升序）----
        # producers[t] = {op: op→tensor 边}；consumers[t] = {op: tensor→op 边}
        prod_pairs = []   # (tidx, oidx)
        cons_pairs = []
        direct_pairs = []  # (src_oidx, dst_oidx)
        op_id_set = set(int(x) for x in self.op_ids)
        for e in edges:
            s, d = e["source"], e["target"]
            s_op, d_op = s in op_id_set, d in op_id_set
            if s_op and not d_op:
                prod_pairs.append((pos_of_ten[d], pos_of_op[s]))
            elif (not s_op) and d_op:
                cons_pairs.append((pos_of_ten[s], pos_of_op[d]))
            elif s_op and d_op and s != d:
                direct_pairs.append((pos_of_op[s], pos_of_op[d],
                                     int(e.get("data_size", 0) or 0)))
        prod_pairs.sort()
        cons_pairs.sort()
        self._build_csr(n_ten, prod_pairs, cons_pairs)
        # producers/consumers 中每个 tensor 的 op 列表已按 oidx 升序
        # （= 按 op 全局 id 升序，与官方 sorted(local_producers) 对齐）
        self.direct_src = np.array([p[0] for p in direct_pairs], dtype=np.int64)
        self.direct_dst = np.array([p[1] for p in direct_pairs], dtype=np.int64)
        self.direct_size = np.array([p[2] for p in direct_pairs],
                                    dtype=np.int64)

        # ---- 官方 _copy_traffic_bytes(原图)（常数，预计算）----
        total = 0
        for e in edges:
            s, d = e["source"], e["target"]
            if s in op_id_set and d not in op_id_set:
                o = pos_of_op[s]
                if self.op_type[o] == 1:  # COPY_IN
                    total += int(self.t_size[pos_of_ten[d]])
            elif (s not in op_id_set) and d in op_id_set:
                o = pos_of_op[d]
                if self.op_type[o] == 2:  # COPY_OUT
                    total += int(self.t_size[pos_of_ten[s]])
        self.original_copy_traffic = total

        # ---- id 计数器（任务构造的边界 id 分配基准）----
        self.max_op_id = int(self.op_ids[-1]) if n_ops else 0
        self.max_tid = int(self.t_ids[-1]) if n_ten else 0

    def _build_csr(self, n_ten, prod_pairs, cons_pairs):
        self.prod_ptr = np.zeros(n_ten + 1, dtype=np.int64)
        for t, _ in prod_pairs:
            self.prod_ptr[t + 1] += 1
        self.prod_ptr = np.cumsum(self.prod_ptr)
        self.prod_ops = np.array([o for _, o in prod_pairs], dtype=np.int64)
        self.cons_ptr = np.zeros(n_ten + 1, dtype=np.int64)
        for t, _ in cons_pairs:
            self.cons_ptr[t + 1] += 1
        self.cons_ptr = np.cumsum(self.cons_ptr)
        self.cons_ops = np.array([o for _, o in cons_pairs], dtype=np.int64)

    # -- 便捷查询 --
    def tensor_state(self, tidx):
        """返回 (tidx, is_l1) —— 官方 pos∈capacity 判定。"""
        return self.t_onchip[tidx] != 0


# =====================================================================
# 二、任务构造（每方案一次，host numpy；对齐 _build_scene_a_tasks）
# =====================================================================

class TaskBuild:
    """一次方案评估的全部任务图，打包为平铺数组。

    本地编号规则（与全局 id 升序完全同序，供官方 sorted() 语义复刻）：
      ops:    [任务内原 eligible op 按 id 升序] + [边界 COPY 按创建序]
      tensors:[touched 原 tensor 按 id 升序]     + [边界 DDR 按创建序]
    """

    def __init__(self, codec, plan_view, bandwidth=BW):
        gc = codec
        n_ops_all = len(gc.op_ids)
        n_ten_all = len(gc.t_ids)
        mapping = plan_view["mapping"]           # 全局 op id -> subgraph id
        subgraph_ids = plan_view["subgraph_ids"]  # 已升序
        n_tasks = len(subgraph_ids)
        sg_slot = {sg: k for k, sg in enumerate(subgraph_ids)}

        # op 归属（oidx -> task slot），eligible 才有
        op_task = np.full(n_ops_all, -1, dtype=np.int64)
        op_ids = gc.op_ids
        for oidx in range(n_ops_all):
            sg = mapping.get(int(op_ids[oidx]))
            if sg is not None:
                op_task[oidx] = sg_slot[sg]

        # 每个 task 的 op 列表（oidx 升序 = 全局 id 升序）
        task_op_count = np.zeros(n_tasks, dtype=np.int64)
        for oidx in range(n_ops_all):
            t = op_task[oidx]
            if t >= 0:
                task_op_count[t] += 1
        task_op_ptr = np.zeros(n_tasks + 1, dtype=np.int64)
        np.cumsum(task_op_count, out=task_op_ptr[1:])
        task_ops = np.zeros(int(task_op_ptr[-1]), dtype=np.int64)
        fill = task_op_ptr[:-1].copy()
        for oidx in range(n_ops_all):
            t = op_task[oidx]
            if t >= 0:
                task_ops[fill[t]] = oidx
                fill[t] += 1

        # ---- 逐 tensor 判定 touched / 边界（向量化）----
        # producers/consumers CSR 已按 (tidx, oidx) 排序
        prod_t = np.repeat(np.arange(n_ten_all, dtype=np.int64),
                           np.diff(gc.prod_ptr))
        prod_o = gc.prod_ops
        cons_t = np.repeat(np.arange(n_ten_all, dtype=np.int64),
                           np.diff(gc.cons_ptr))
        cons_o = gc.cons_ops
        prod_task = np.where(op_task[prod_o] >= 0, op_task[prod_o], -1)
        cons_task = np.where(op_task[cons_o] >= 0, op_task[cons_o], -1)
        prod_elig = gc.op_eligible[prod_o]
        cons_elig = gc.op_eligible[cons_o]
        cons_is_copout = gc.op_type[cons_o] == 2

        # 每 (tensor, task) 的 local producer/consumer 计数
        key_pt = prod_task[prod_task >= 0] * n_ten_all + prod_t[prod_task >= 0]
        key_ct = cons_task[cons_task >= 0] * n_ten_all + cons_t[cons_task >= 0]
        n_pairs = n_tasks * n_ten_all
        locp = np.zeros(n_pairs, dtype=np.int64)
        np.add.at(locp, key_pt, 1)
        locc = np.zeros(n_pairs, dtype=np.int64)
        np.add.at(locc, key_ct, 1)
        # eligible consumers 总数（不含 COPY，全图，按 tensor 聚集）
        elig_total = np.zeros(n_ten_all, dtype=np.int64)
        np.add.at(elig_total, cons_t[cons_elig], 1)
        # has_original_copy_out（任意 COPY_OUT consumer，不论归属）
        copout_t = cons_t[cons_is_copout]
        copout_any = np.zeros(n_ten_all, dtype=np.int64)
        np.add.at(copout_any, copout_t, 1)

        touched = (locp.reshape(n_tasks, n_ten_all) > 0) | (
            locc.reshape(n_tasks, n_ten_all) > 0)

        # ---- 边界判定（对齐官方 input/output_boundary 定义）----
        # input_boundary: locc>0 & locp==0
        # output_boundary: locp>0 & (has_copy_out | elig_total==0
        #                            | elig_total > locc（有 task 外消费者）)
        in_b = (locc.reshape(n_tasks, n_ten_all) > 0) & (
            locp.reshape(n_tasks, n_ten_all) == 0)
        out_b = (locp.reshape(n_tasks, n_ten_all) > 0) & (
            (copout_any[None, :] > 0)
            | (elig_total[None, :] == 0)
            | ((elig_total[None, :]
                - locc.reshape(n_tasks, n_ten_all)) > 0))
        n_in_b = int(in_b.sum())
        n_out_b = int(out_b.sum())

        # ---- 全局平铺数组 ----
        total_ops = int(task_op_ptr[-1]) + n_in_b + n_out_b
        total_touched = int(touched.sum())
        total_tensors = total_touched + n_in_b + n_out_b
        # direct op-op 边按任务分桶（一次线性扫描）
        direct_by_task = [[] for _ in range(n_tasks)]
        for si, di in zip(gc.direct_src, gc.direct_dst):
            ts, td = int(op_task[si]), int(op_task[di])
            if ts >= 0 and ts == td:
                direct_by_task[ts].append((int(si), int(di)))
        # 全局 id（对拍用；分配序与官方 new_boundary_ids 一致）
        # 官方 used_ids = 原图 op∪tensor id，且随分配动态增长；
        # 两个计数器单调递增，都要跳过原图 id 与对方计数器新分配的 id
        used_ids_sorted = np.union1d(gc.op_ids, gc.t_ids)
        new_allocs = set()
        next_op_id = gc.max_op_id + 1
        next_tid = max(gc.max_tid, 10000) + 1

        def _alloc(kind):
            nonlocal next_op_id, next_tid
            if kind == "op":
                cur = next_op_id
            else:
                cur = next_tid
            while True:
                i = int(np.searchsorted(used_ids_sorted, cur))
                if i < len(used_ids_sorted) and used_ids_sorted[i] == cur:
                    cur += 1
                    continue
                if cur in new_allocs:
                    cur += 1
                    continue
                break
            new_allocs.add(cur)
            if kind == "op":
                next_op_id = cur + 1
            else:
                next_tid = cur + 1
            return cur

        self.n_tasks = n_tasks
        self.subgraph_ids = subgraph_ids
        self.core_of = np.array(
            [plan_view["core_by_subgraph"][sg] for sg in subgraph_ids],
            dtype=np.int64)
        self.num_cores = plan_view["num_cores"]
        # core_orders: 每核的 subgraph 顺序（转成 task slot 序）
        self.core_orders = [
            [sg_slot[sg] for sg in plan_view["core_orders"].get(c, [])]
            for c in range(self.num_cores)]
        # 任务前驱（slot 序）
        sg_preds = [set() for _ in range(n_tasks)]
        for s, d in plan_view["dependency_pairs"]:
            sg_preds[sg_slot[d]].add(sg_slot[s])
        self.task_preds = [sorted(x) for x in sg_preds]

        # op 描述（平铺，逐任务连续）
        self.t_op_ptr = np.zeros(n_tasks + 1, dtype=np.int64)
        self.t_ten_ptr = np.zeros(n_tasks + 1, dtype=np.int64)
        op_type = np.zeros(total_ops, dtype=np.int64)
        op_pipe = np.zeros(total_ops, dtype=np.int64)
        op_cycles = np.zeros(total_ops, dtype=np.int64)
        op_gid = np.zeros(total_ops, dtype=np.int64)
        # tensor 描述（平铺，逐任务连续）
        ten_size = np.zeros(total_tensors, dtype=np.int64)
        ten_pos = np.zeros(total_tensors, dtype=np.int64)  # 0=DDR,1=L1,2=UB
        ten_gid = np.zeros(total_tensors, dtype=np.int64)
        # COPY_IN backing（_find_copy_in_backings 复刻）：
        # input 边界 tensor 的 ddr 边界槽即其 backing
        ten_backing = np.full(total_tensors, -1, dtype=np.int64)
        op_out_pairs = []   # (flat op, flat tensor)
        op_in_pairs = []
        direct_pairs_t = []  # (task, src flat op, dst flat op)
        cross_task_traffic = 0
        task_graph_copy_traffic = 0

        # 第一遍：填原 op / touched tensor，并建 oidx→flat 槽位查询表
        # （基址用含边界增量的完整 ptr：先算 ptr 再回填，避免槽位重叠）
        self.t_op_ptr[0] = 0
        self.t_op_ptr[1:] = np.cumsum(
            task_op_count + in_b.sum(axis=1) + out_b.sum(axis=1))
        self.t_ten_ptr[0] = 0
        self.t_ten_ptr[1:] = np.cumsum(
            touched.sum(axis=1) + in_b.sum(axis=1) + out_b.sum(axis=1))
        t_op_base = self.t_op_ptr[:-1].copy()
        t_ten_base = self.t_ten_ptr[:-1].copy()
        flat_op_of = np.full(n_ops_all, -1, dtype=np.int64)
        op_cursor = 0
        ten_cursor = 0
        touched_lists = []
        for k in range(n_tasks):
            tlist = np.nonzero(touched[k])[0]  # tidx 升序
            touched_lists.append(tlist)
            for oidx in task_ops[task_op_ptr[k]:task_op_ptr[k + 1]]:
                op_type[op_cursor] = T_ELE
                op_pipe[op_cursor] = gc.op_pipe[oidx]
                op_cycles[op_cursor] = gc.op_cycles[oidx]
                op_gid[op_cursor] = gc.op_ids[oidx]
                flat_op_of[oidx] = op_cursor
                op_cursor += 1
            for tidx in tlist:
                ten_size[ten_cursor] = gc.t_size[tidx]
                ten_pos[ten_cursor] = gc.t_onchip[tidx] if gc.t_onchip[tidx] else 2
                ten_gid[ten_cursor] = gc.t_ids[tidx]
                ten_cursor += 1
            # 跳过本任务预留的边界槽位（第二遍才填充）
            op_cursor = int(self.t_op_ptr[k + 1])
            ten_cursor = int(self.t_ten_ptr[k + 1])
        # 第二遍：逐任务逐 touched tensor 建 边/边界COPY/流量
        # （保持官方遍历序：task 升序 × tensor 升序 × input 先于 output）
        op_offset_acc = t_op_base + task_op_count   # 每任务边界 op 追加点
        ten_offset_acc = t_ten_base + np.array(
            [len(x) for x in touched_lists])        # 边界 tensor 追加点
        for k in range(n_tasks):
            for j, tidx in enumerate(touched_lists[k]):
                tidx = int(tidx)
                size = int(gc.t_size[tidx])
                lten = t_ten_base[k] + j
                # local producers/consumers（oidx 列表，升序）
                lp = prod_o[gc.prod_ptr[tidx]:gc.prod_ptr[tidx + 1]]
                lp = lp[op_task[lp] == k]
                lc = cons_o[gc.cons_ptr[tidx]:gc.cons_ptr[tidx + 1]]
                lc = lc[op_task[lc] == k]
                for oidx in lp:   # op → tensor
                    op_out_pairs.append((int(flat_op_of[oidx]), lten))
                for oidx in lc:   # tensor → op
                    op_in_pairs.append((int(flat_op_of[oidx]), lten))
                if in_b[k, tidx]:
                    ddr_lten = int(ten_offset_acc[k])
                    ten_size[ddr_lten] = size
                    ten_pos[ddr_lten] = 0
                    ten_gid[ddr_lten] = _alloc("ten")
                    ten_backing[lten] = ddr_lten
                    ten_offset_acc[k] += 1
                    cop = int(op_offset_acc[k])
                    op_type[cop] = T_COPY_IN
                    op_pipe[cop] = PIPE_MTE2
                    op_cycles[cop] = max(1, math.ceil(size / bandwidth))
                    op_gid[cop] = _alloc("op")
                    op_offset_acc[k] += 1
                    op_out_pairs.append((cop, lten))
                    op_in_pairs.append((cop, ddr_lten))
                    task_graph_copy_traffic += size
                if out_b[k, tidx]:
                    ddr_lten = int(ten_offset_acc[k])
                    ten_size[ddr_lten] = size
                    ten_pos[ddr_lten] = 0
                    ten_gid[ddr_lten] = _alloc("ten")
                    ten_offset_acc[k] += 1
                    cop = int(op_offset_acc[k])
                    op_type[cop] = T_COPY_OUT
                    op_pipe[cop] = PIPE_MTE3
                    op_cycles[cop] = max(1, math.ceil(size / bandwidth))
                    op_gid[cop] = _alloc("op")
                    op_offset_acc[k] += 1
                    op_out_pairs.append((cop, ddr_lten))
                    op_in_pairs.append((cop, lten))
                    task_graph_copy_traffic += size
                    # cross_task_traffic += size × 远端消费任务数
                    ec = cons_o[gc.cons_ptr[tidx]:gc.cons_ptr[tidx + 1]]
                    ec = ec[gc.op_eligible[ec]]
                    remote = set()
                    for oidx in ec:
                        tk = int(op_task[oidx])
                        if tk != k:
                            remote.add(tk)
                    cross_task_traffic += size * len(remote)
            # direct op-op 边（task 内，已分桶）
            for si, di in direct_by_task[k]:
                direct_pairs_t.append(
                    (k, int(flat_op_of[si]), int(flat_op_of[di])))

        self.op_type = op_type
        self.op_pipe = op_pipe
        self.op_cycles = op_cycles
        self.op_gid = op_gid
        self.ten_size = ten_size
        self.ten_pos = ten_pos
        self.ten_gid = ten_gid
        self.ten_backing = ten_backing
        self.task_graph_copy_traffic = task_graph_copy_traffic
        self.cross_task_traffic = cross_task_traffic
        self.partition_added = (task_graph_copy_traffic
                                - gc.original_copy_traffic)
        # op↔tensor CSR（raw 构造序另存：官方 set 迭代序复刻需要）
        self.in_pairs_raw = list(op_in_pairs)
        self.out_pairs_raw = list(op_out_pairs)
        op_out_pairs.sort()
        op_in_pairs.sort()
        n_flat_ops = int(self.t_op_ptr[-1])
        self.out_ptr = np.zeros(n_flat_ops + 1, dtype=np.int64)
        for o, _ in op_out_pairs:
            self.out_ptr[o + 1] += 1
        self.out_ptr = np.cumsum(self.out_ptr)
        self.out_ten = np.array([t for _, t in op_out_pairs], dtype=np.int64)
        self.in_ptr = np.zeros(n_flat_ops + 1, dtype=np.int64)
        for o, _ in op_in_pairs:
            self.in_ptr[o + 1] += 1
        self.in_ptr = np.cumsum(self.in_ptr)
        self.in_ten = np.array([t for _, t in op_in_pairs], dtype=np.int64)
        # direct 边（task, src, dst）→ 平铺
        dt = np.array([p[0] for p in direct_pairs_t], dtype=np.int64)
        ds = np.array([p[1] for p in direct_pairs_t], dtype=np.int64)
        dd = np.array([p[2] for p in direct_pairs_t], dtype=np.int64)
        order = np.lexsort((dd, ds, dt))
        self.direct_src = ds[order] if len(order) else ds
        self.direct_dst = dd[order] if len(order) else dd
        self.direct_task = dt[order] if len(order) else dt

        # step2 起始 next_id（每任务独立）：max(op ids, tensor ids)+1
        # 需要全局 id 序来分配 spill id —— 我们用局部计数器等价复刻：
        # 官方 next_id = max(任务图 op ids, tensor ids)+1 后单调递增，
        # 与"局部下标继续往上编号"一一对应（全局顺序=局部顺序）。
        # 因此 spill 分配的新 op/tensor 直接用平铺数组尾部的连续新槽，
        # 仅需保证其"排序等价 id"递增 —— numba 内核用槽号即可。

    # ---- 后续内核所需的输入打包 ----

    def kernel_inputs(self):
        """返回 (t_op_ptr, t_ten_ptr, op_type, op_pipe, op_cycles,
        ten_size, ten_pos, out_ptr, out_ten, in_ptr, in_ten,
        direct_task/src/dst) 供 numba 内核使用。"""
        return (self.t_op_ptr, self.t_ten_ptr, self.op_type, self.op_pipe,
                self.op_cycles, self.ten_size, self.ten_pos,
                self.out_ptr, self.out_ten, self.in_ptr, self.in_ten,
                self.direct_task, self.direct_src, self.direct_dst)


# =====================================================================
# 四、阶段驱动：step2（Belady spill）
# =====================================================================

def stage_step2_one(tb, k, seq, cap_l1=CAP_L1, cap_ub=CAP_UB):
    """单任务 step2。返回 dict（含 ps 数组与 uses CSR）。"""
    from fast_kernels import step2_sim
    lo, hi = int(tb.t_op_ptr[k]), int(tb.t_op_ptr[k + 1])
    tlo, thi = int(tb.t_ten_ptr[k]), int(tb.t_ten_ptr[k + 1])
    n = hi - lo
    n_ten = thi - tlo

    # ---- tensor uses CSR：iterate seq ops；in/out tens -> (tensor, step, op)
    pairs = []
    for p, op in enumerate(seq):
        u = int(op)  # 平铺 op 下标
        for e in range(tb.in_ptr[u], tb.in_ptr[u + 1]):
            pairs.append((int(tb.in_ten[e]) - tlo, p, u))
        for e in range(tb.out_ptr[u], tb.out_ptr[u + 1]):
            pairs.append((int(tb.out_ten[e]) - tlo, p, u))
    pairs.sort()
    uses_tid = []
    uses_step = []
    uses_op = []
    prev_key = (-1, -1)
    for (tid, st, op) in pairs:
        if (tid, st) != prev_key:
            prev_key = (tid, st)
            uses_tid.append(tid)
            uses_step.append(st)
            uses_op.append(op)
    uses_ptr = np.zeros(n_ten + 1, dtype=np.int64)
    for tid in uses_tid:
        uses_ptr[tid + 1] += 1
    uses_ptr = np.cumsum(uses_ptr)
    uses_step_a = np.array(uses_step, dtype=np.int64)
    uses_op_a = np.array(uses_op, dtype=np.int64)
    has_use = np.diff(uses_ptr) > 0

    # ---- uses_at_step（仅 lifecycle：on-chip 且有 use；平铺 tensor 序）
    lifecycle = (tb.ten_pos[tlo:thi] != 0) & has_use
    entries = []  # (step, lifecycle_rank, use_idx, tid)
    rank = 0
    for tid in range(n_ten):
        if lifecycle[tid]:
            up0, up1 = int(uses_ptr[tid]), int(uses_ptr[tid + 1])
            for u in range(up0, up1):
                entries.append((int(uses_step_a[u]), rank, u - up0, tid))
            rank += 1
    entries.sort()
    uat_ptr = np.zeros(n + 1, dtype=np.int64)
    for ent in entries:
        uat_ptr[ent[0] + 1] += 1
    uat_ptr = np.cumsum(uat_ptr)
    uat_tid = np.array([e[3] for e in entries], dtype=np.int64)
    uat_idx = np.array([e[2] for e in entries], dtype=np.int64)

    # ---- scratch
    cap = 4 * n_ten + 64
    act_alive = np.zeros(n_ten, dtype=np.bool_)
    act_nu = np.zeros(n_ten, dtype=np.int64)
    act_ui = np.zeros(n_ten, dtype=np.int64)
    act_rank = np.zeros(n_ten, dtype=np.int64)
    act_order = np.empty(cap, dtype=np.int64)
    resid = np.zeros(2, dtype=np.int64)
    cur_gen = np.zeros(n_ten, dtype=np.int64)
    rel_buf = np.empty(max(1, len(entries)), dtype=np.int64)
    si_cap = cap
    si_head = np.full(n + 1, -1, dtype=np.int64)
    si_tail = np.full(n + 1, -1, dtype=np.int64)
    si_next = np.full(si_cap, -1, dtype=np.int64)
    si_tid_a = np.zeros(si_cap, dtype=np.int64)
    si_ui_a = np.zeros(si_cap, dtype=np.int64)
    ps_cap = cap
    ps_victim = np.zeros(ps_cap, dtype=np.int64)
    ps_prev_step = np.zeros(ps_cap, dtype=np.int64)
    ps_next_step = np.zeros(ps_cap, dtype=np.int64)
    ps_prev_op = np.zeros(ps_cap, dtype=np.int64)
    ps_next_op = np.zeros(ps_cap, dtype=np.int64)
    ps_ui = np.zeros(ps_cap, dtype=np.int64)
    ps_type = np.zeros(ps_cap, dtype=np.int64)
    ps_size = np.zeros(ps_cap, dtype=np.int64)
    ps_nu = np.zeros(ps_cap, dtype=np.int64)

    seq_local = np.asarray(seq, dtype=np.int64) - lo
    ten_is_l1 = (tb.ten_pos[tlo:thi] == 1)
    ten_size = tb.ten_size[tlo:thi].copy()
    n_ps = step2_sim(
        n, seq_local, uses_ptr, uses_step_a, uses_op_a,
        uat_ptr, uat_tid, uat_idx, ten_is_l1, ten_size,
        cap_l1, cap_ub,
        act_alive, act_nu, act_ui, act_rank, act_order,
        resid, cur_gen, rel_buf,
        si_head, si_tail, si_next, si_tid_a, si_ui_a,
        ps_victim, ps_prev_step, ps_next_step, ps_prev_op,
        ps_next_op, ps_ui, ps_type, ps_size, ps_nu)
    if n_ps < 0:
        raise RuntimeError(f"step2_sim error {n_ps} in task {k}")
    return {
        "n": n, "n_ten": n_ten, "tlo": tlo, "lo": lo, "n_ps": n_ps,
        "victim": ps_victim[:n_ps], "prev_step": ps_prev_step[:n_ps],
        "next_step": ps_next_step[:n_ps], "prev_op": ps_prev_op[:n_ps],
        "next_op": ps_next_op[:n_ps], "ui": ps_ui[:n_ps],
        "type": ps_type[:n_ps], "size": ps_size[:n_ps],
        "uses_ptr": uses_ptr, "uses_step": uses_step_a,
        "uses_op": uses_op_a,
    }


def stage_step2(tb, seqs):
    return [stage_step2_one(tb, k, seqs[k]) for k in range(tb.n_tasks)]



# =====================================================================
# 五、扩展图组装（step2 输出 → step3 输入）
# =====================================================================

def build_ext_task(tb, k, seq, ps):
    """组装单任务扩展图（对齐官方 _build_extended_graph 的输出语义）。

    返回 dict：
      ops: 局部 op 数组（原 + spill），type/pipe/cycles/gid
      tensors: 局部 tensor 数组（原 + backing/renamed），size/pos/gid
      in_pairs/out_pairs: (op局部, tensor局部) 边（含改接与新增）
      direct: (src局部, dst局部) 直接边
      seq_ext: 扩展序列（局部 op）
      spill_traffic: Σ size*(1+copies)
    """
    lo, hi = int(tb.t_op_ptr[k]), int(tb.t_op_ptr[k + 1])
    tlo, thi = int(tb.t_ten_ptr[k]), int(tb.t_ten_ptr[k + 1])
    n = hi - lo
    n_ten = thi - tlo

    op_type = list(tb.op_type[lo:hi])
    op_pipe = list(tb.op_pipe[lo:hi])
    op_cycles = list(tb.op_cycles[lo:hi])
    op_gid = list(tb.op_gid[lo:hi])
    ten_size = list(tb.ten_size[tlo:thi])
    ten_pos = list(tb.ten_pos[tlo:thi])
    ten_gid = list(tb.ten_gid[tlo:thi])
    ten_logical = list(ten_gid)   # renamed 槽位将被改写为 victim 的 gid
    ten_backing = [int(b) - tlo if b >= 0 else -1
                   for b in tb.ten_backing[tlo:thi]]

    in_pairs = [(o - lo, t - tlo) for o, t in tb.in_pairs_raw
                if lo <= o < hi]
    out_pairs = [(o - lo, t - tlo) for o, t in tb.out_pairs_raw
                 if lo <= o < hi]
    direct = [(int(tb.direct_src[i]) - lo, int(tb.direct_dst[i]) - lo)
              for i in range(len(tb.direct_src))
              if int(tb.direct_task[i]) == k]
    n_orig_in = len(in_pairs)

    # ---- spill id 分配（官方：单一计数器 = max(op ids, tensor ids)+1）----
    next_id = max(max(op_gid) if op_gid else 0,
                  max(ten_gid) if ten_gid else 0) + 1

    def alloc():
        nonlocal next_id
        v = next_id
        next_id += 1
        return v

    n_ps = ps["n_ps"]
    # 记录信息（trigger 序）
    recs = []  # dict: victim(局部tensor), size, prev_step, next_step, copies
    for i in range(n_ps):
        victim = int(ps["victim"][i])
        size = int(ps["size"][i])
        copies = ten_backing[victim] == -1
        spill_out_id = None
        so_slot = None
        if copies:
            spill_out_id = alloc()
        spill_in_id = alloc()
        if copies:
            backing_tid = alloc()
            ten_backing[victim] = len(ten_size)  # 新槽
            ten_size.append(size)
            ten_pos.append(0)
            ten_gid.append(backing_tid)
            ten_logical.append(backing_tid)
        else:
            backing_tid = ten_gid[ten_backing[victim]]
        to_gid = alloc()
        to_slot = len(ten_size)
        ten_size.append(size)
        ten_pos.append(ten_pos[victim])
        ten_gid.append(to_gid)
        ten_logical.append(ten_gid[victim])
        # 新 op
        if copies:
            so_slot = len(op_type)
            op_type.append(T_SPILL_OUT)
            op_pipe.append(PIPE_MTE3)
            op_cycles.append(max(1, size // 64))
            op_gid.append(spill_out_id)
            out_pairs.append((so_slot, ten_backing[victim]))  # spill_out → backing
            in_pairs.append((so_slot, victim))                # victim → spill_out
        si_slot = len(op_type)
        op_type.append(T_SPILL_IN)
        op_pipe.append(PIPE_MTE2)
        op_cycles.append(max(1, size // 64))
        op_gid.append(spill_in_id)
        backing_slot = None
        # 找 backing 槽（ten_backing[victim] 或按 gid）
        for s in range(len(ten_gid)):
            if ten_gid[s] == backing_tid:
                backing_slot = s
                break
        out_pairs.append((si_slot, to_slot))      # spill_in → renamed
        in_pairs.append((si_slot, backing_slot))  # backing → spill_in
        recs.append({
            "victim": victim, "size": size, "copies": copies,
            "prev_step": int(ps["prev_step"][i]),
            "next_step": int(ps["next_step"][i]),
            "to_slot": to_slot, "so_slot": so_slot if copies else None,
            "si_slot": si_slot,
        })

    # ---- 消费边改接（incarnation）----
    # 每个有记录的 tensor：按 trigger 序的记录（next_step 递增），
    # use step >= record.next_step 的消费边改接 to_slot
    recs_by_tid = {}
    for r in recs:
        recs_by_tid.setdefault(r["victim"], []).append(r)
    seq_step = {int(op) - lo: p for p, op in enumerate(seq)}
    for ei in range(n_orig_in):   # 官方只改接原始边；新 spill 边不参与
        o, t = in_pairs[ei]
        rl = recs_by_tid.get(t)
        if not rl:
            continue
        step = seq_step[o]
        phys = t
        for r in rl:
            if step >= r["next_step"]:
                phys = r["to_slot"]
        # 官方按 record_index 单调推进（等价于取最后一个满足条件的记录）
        if phys != t:
            in_pairs[ei] = (o, phys)

    # ---- seq_ext ----
    insert_after = {}
    insert_before = {}
    for r in recs:
        if r["so_slot"] is not None:
            insert_after.setdefault(r["prev_step"], []).append(r["so_slot"])
        insert_before.setdefault(r["next_step"], []).append(r["si_slot"])
    seq_local = [int(op) - lo for op in seq]
    seq_ext = []
    for i, op in enumerate(seq_local):
        if i > 0:
            seq_ext.extend(insert_after.get(i - 1, []))
        seq_ext.extend(insert_before.get(i, []))
        seq_ext.append(op)
    seq_ext.extend(insert_after.get(n - 1, []))

    spill_traffic = sum(r["size"] * (1 + int(r["copies"])) for r in recs)
    return {
        "n": n, "n_ext_ops": len(op_type), "n_ext_ten": len(ten_size),
        "n_ten0": n_ten, "n_ops0": n,
        "ten_logical": np.array(ten_logical, dtype=np.int64),
        "op_type": np.array(op_type, dtype=np.int64),
        "op_pipe": np.array(op_pipe, dtype=np.int64),
        "op_cycles": np.array(op_cycles, dtype=np.int64),
        "op_gid": np.array(op_gid, dtype=np.int64),
        "ten_size": np.array(ten_size, dtype=np.int64),
        "ten_pos": np.array(ten_pos, dtype=np.int64),
        "ten_gid": np.array(ten_gid, dtype=np.int64),
        "in_pairs": in_pairs, "out_pairs": out_pairs, "direct": direct,
        "seq_ext": np.array(seq_ext, dtype=np.int64),
        "spill_traffic": spill_traffic, "n_ps": n_ps,
        "ten_backing": ten_backing,
    }


def build_ext_all(tb, seqs, ps_list):
    return [build_ext_task(tb, k, seqs[k], ps_list[k])
            for k in range(tb.n_tasks)]


# =====================================================================
# 六、阶段驱动：step3（管道调度模拟）
# =====================================================================

def stage_step3_one(ext, bandwidth=BW, cap_l1=CAP_L1, cap_ub=CAP_UB):
    from step3_kernel import step3_sim
    n_ops = ext["n_ext_ops"]
    if n_ops == 0:
        z = np.zeros(0, dtype=np.int64)
        return {"makespan": 0, "peak": (0, 0), "deps": z, "depd": z,
                "pipe_seq": z, "pipe_seq_ptr": np.zeros(5, dtype=np.int64),
                "pred_ptr": np.ones(1, dtype=np.int64), "pred_arr": z,
                "succ_ptr": np.ones(1, dtype=np.int64), "succ_arr": z,
                "op_dur": z, "op_is_ddr": np.zeros(0, dtype=np.bool_),
                "seq_ext": z, "n_ops": 0, "op_end": np.zeros(0, dtype=np.float64)}
    n_ten = ext["n_ext_ten"]
    op_type = ext["op_type"]
    op_pipe = ext["op_pipe"]
    ten_size = ext["ten_size"]
    ten_pos = ext["ten_pos"]

    # ---- in/out CSR ----
    in_pairs = sorted(ext["in_pairs"])
    out_pairs = sorted(ext["out_pairs"])
    # 官方 consume_inputs/release_dead_outputs 按 set(in/out_tids) 迭代序
    # （CPython 集序，非排序）→ 宿主用真 set 复刻（元素=全局 tensor id）
    ten_gid = ext["ten_gid"]
    raw_in = ext["in_pairs"]
    raw_out = ext["out_pairs"]
    by_op_in = [[] for _ in range(n_ops)]
    for o, t in raw_in:
        by_op_in[o].append(int(ten_gid[t]))
    by_op_out = [[] for _ in range(n_ops)]
    for o, t in raw_out:
        by_op_out[o].append(int(ten_gid[t]))
    gid2slot = {}
    for t in range(n_ten):
        gid2slot[int(ten_gid[t])] = t
    ord_in_parts = []
    ord_out_parts = []
    for o in range(n_ops):
        ord_in_parts.extend(gid2slot[g] for g in set(by_op_in[o]))
        ord_out_parts.extend(gid2slot[g] for g in set(by_op_out[o]))
    ord_in_ptr = np.zeros(n_ops + 1, dtype=np.int64)
    for o in range(n_ops):
        ord_in_ptr[o + 1] = ord_in_ptr[o] + len(set(by_op_in[o]))
    ord_out_ptr = np.zeros(n_ops + 1, dtype=np.int64)
    for o in range(n_ops):
        ord_out_ptr[o + 1] = ord_out_ptr[o] + len(set(by_op_out[o]))
    ord_in = np.array(ord_in_parts, dtype=np.int64)
    ord_out = np.array(ord_out_parts, dtype=np.int64)
    in_ptr = np.zeros(n_ops + 1, dtype=np.int64)
    for o, _ in in_pairs:
        in_ptr[o + 1] += 1
    in_ptr = np.cumsum(in_ptr)
    in_ten = np.array([t for _, t in in_pairs], dtype=np.int64)
    out_ptr = np.zeros(n_ops + 1, dtype=np.int64)
    for o, _ in out_pairs:
        out_ptr[o + 1] += 1
    out_ptr = np.cumsum(out_ptr)
    out_ten = np.array([t for _, t in out_pairs], dtype=np.int64)

    # ---- 时长 / DDR 标记 ----
    op_dur = np.zeros(n_ops, dtype=np.int64)
    op_is_ddr = np.zeros(n_ops, dtype=np.bool_)
    for o in range(n_ops):
        t = op_type[o]
        if t == 1:  # COPY_IN: out tensors
            total = sum(int(ten_size[int(out_ten[e])])
                        for e in range(out_ptr[o], out_ptr[o + 1]))
            op_dur[o] = max(1, -(-total // bandwidth))
        elif t in (2, 3, 4):  # COPY_OUT / SPILL: in tensors
            total = sum(int(ten_size[int(in_ten[e])])
                        for e in range(in_ptr[o], in_ptr[o + 1]))
            op_dur[o] = max(1, -(-total // bandwidth))
        else:
            op_dur[o] = int(ext["op_cycles"][o])
        if t in (1, 2, 3, 4):
            for e in range(in_ptr[o], in_ptr[o + 1]):
                if ten_pos[int(in_ten[e])] == 0:
                    op_is_ddr[o] = True
            for e in range(out_ptr[o], out_ptr[o + 1]):
                if ten_pos[int(out_ten[e])] == 0:
                    op_is_ddr[o] = True

    # ---- 依赖边（收缩 + direct）→ pred/succ CSR ----
    d_arr = ext["direct"]
    ten_cons_ptr, ten_cons_arr = _transpose_csr(in_ptr, in_ten, n_ops, n_ten)
    prod_ptr, prod_arr = _transpose_csr(out_ptr, out_ten, n_ops, n_ten)
    es = []
    for o, t in out_pairs:
        for c in range(ten_cons_ptr[t], ten_cons_ptr[t + 1]):
            es.append((o, int(ten_cons_arr[c])))
    for s, d in d_arr:
        es.append((s, d))
    if es:
        comp = np.unique(np.array([s * n_ops + d for s, d in es],
                                  dtype=np.int64))
        esrc = (comp // n_ops).astype(np.int64)
        edst = (comp % n_ops).astype(np.int64)
    else:
        esrc = np.zeros(0, dtype=np.int64)
        edst = np.zeros(0, dtype=np.int64)
    pred_ptr = np.zeros(n_ops + 1, dtype=np.int64)
    succ_ptr = np.zeros(n_ops + 1, dtype=np.int64)
    np.add.at(pred_ptr, edst + 1, 1)
    np.add.at(succ_ptr, esrc + 1, 1)
    pred_ptr = np.cumsum(pred_ptr)
    succ_ptr = np.cumsum(succ_ptr)
    o2 = np.argsort(edst * n_ops + esrc, kind="stable")
    pred_arr = esrc[o2]
    succ_arr = edst[np.argsort(esrc * n_ops + edst, kind="stable")]

    cons_ptr, cons_arr = ten_cons_ptr, ten_cons_arr

    # ---- pipe 序 / 分配序 ----
    seq_ext = ext["seq_ext"]
    pipe_seq = seq_ext.copy()
    pipe_seq_ptr = np.zeros(5, dtype=np.int64)
    for op in seq_ext:
        pipe_seq_ptr[op_pipe[op] + 1] += 1
    pipe_seq_ptr = np.cumsum(pipe_seq_ptr)
    fillp = pipe_seq_ptr[:4].copy()
    for op in seq_ext:
        p = op_pipe[op]
        pipe_seq[fillp[p]] = op
        fillp[p] += 1
    ten_pos_l1 = np.where(ten_pos == 1, 1,
                          np.where(ten_pos == 2, 2, 0)).astype(np.int64)
    alloc_list = [int(op) for op in seq_ext
                  if any(ten_pos_l1[int(out_ten[e])] != 0
                         for e in range(out_ptr[op], out_ptr[op + 1]))]
    alloc_order = np.array(alloc_list, dtype=np.int64)
    alloc_rank = np.full(n_ops, -1, dtype=np.int64)
    for r, op in enumerate(alloc_list):
        alloc_rank[op] = r

    # ---- scratch ----
    op_status = np.zeros(n_ops, dtype=np.int64)
    op_start = np.zeros(n_ops, dtype=np.float64)
    op_end = np.zeros(n_ops, dtype=np.float64)
    pred_rem = np.diff(pred_ptr).astype(np.int64).copy()
    pipe_cursor = np.zeros(4, dtype=np.int64)
    ready_op = np.full(4, -1, dtype=np.int64)
    alloc_ready = np.full(4, -1, dtype=np.int64)
    exec_op = np.full(4, -1, dtype=np.int64)
    exec_end = np.zeros(4, dtype=np.float64)
    mem_used = np.zeros(2, dtype=np.int64)
    mem_peak = np.zeros(2, dtype=np.int64)
    remaining_cons = np.zeros(n_ten, dtype=np.int64)
    resident = np.zeros(n_ten, dtype=np.bool_)
    n_managed = int(np.sum(ten_pos_l1 != 0))
    cr_cap = n_managed + 8
    cr_base = np.zeros(2, dtype=np.int64)
    cr_lim = np.zeros(2, dtype=np.int64)
    cr_lim[0] = cr_cap
    cr_lim[1] = 2 * cr_cap
    cr_bytes = np.zeros(2 * cr_cap, dtype=np.int64)
    cr_tensor = np.full(2 * cr_cap, -1, dtype=np.int64)
    cr_kind = np.zeros(2 * cr_cap, dtype=np.int64)
    cr_head = np.zeros(2, dtype=np.int64)
    cr_tail = np.zeros(2, dtype=np.int64)
    cr_base[1] = cr_cap
    dep_cap = 8 * n_ops + 256
    dep_src = np.zeros(dep_cap, dtype=np.int64)
    dep_tgt = np.zeros(dep_cap, dtype=np.int64)
    dep_cnt = np.zeros(1, dtype=np.int64)
    pool_ops = np.zeros(n_ops + 8, dtype=np.int64)
    pool_work = np.zeros(n_ops + 8, dtype=np.float64)
    pool_alive = np.zeros(n_ops + 8, dtype=np.bool_)
    pool_count = np.zeros(1, dtype=np.int64)
    pool_slot = np.full(n_ops, -1, dtype=np.int64)
    ddr_last_update = np.zeros(1, dtype=np.float64)
    sa = np.zeros(n_ops + 8, dtype=np.int64)
    sw = np.zeros(n_ops + 8, dtype=np.float64)
    max_deg = int(max(1,
                      np.max(np.diff(in_ptr)) if n_ops else 1,
                      np.max(np.diff(out_ptr)) if n_ops else 1))
    out_set_buf = np.zeros(max_deg + 1, dtype=np.int64)
    in_set_buf = np.zeros(max_deg + 1, dtype=np.int64)
    req_buf = np.zeros(2, dtype=np.int64)
    out_mk = np.zeros(4, dtype=np.float64)
    dbg = np.zeros(8, dtype=np.int64)

    r = step3_sim(
        n_ops, n_ten, op_pipe, op_is_ddr, op_dur,
        in_ptr, in_ten, out_ptr, out_ten,
        seq_ext, cap_l1, cap_ub,
        ten_size, ten_pos_l1,
        prod_ptr, prod_arr, cons_ptr, cons_arr,
        succ_ptr, succ_arr,
        ord_in_ptr, ord_in, ord_out_ptr, ord_out,
        pipe_seq_ptr, pipe_seq,
        alloc_order,
        op_status, op_start, op_end, pred_rem,
        pipe_cursor, ready_op, alloc_ready, alloc_rank,
        exec_op, exec_end,
        mem_used, mem_peak, remaining_cons, resident,
        cr_bytes, cr_tensor, cr_kind, cr_head, cr_tail, cr_base, cr_lim,
        dep_src, dep_tgt, dep_cap, dep_cnt,
        pool_ops, pool_work, pool_alive, pool_count, pool_slot,
        ddr_last_update, sa, sw,
        out_set_buf, in_set_buf, req_buf, out_mk, dbg)
    if r < 0:
        raise RuntimeError(
            f"step3_sim error {r} diag: now={out_mk[0]} done={out_mk[1]} "
            f"next_rank={out_mk[2]}/{out_mk[3]} ready={out_mk[4:8]} "
            f"alloc_ready={out_mk[8:12]} dbg={dbg.tolist()}")
    return {
        "makespan": int(round(out_mk[0])),
        "peak": (int(out_mk[1]), int(out_mk[2])),
        "deps": dep_src[:dep_cnt[0]], "depd": dep_tgt[:dep_cnt[0]],
        "pipe_seq": pipe_seq, "pipe_seq_ptr": pipe_seq_ptr,
        "pred_ptr": pred_ptr, "pred_arr": pred_arr,
        "succ_ptr": succ_ptr, "succ_arr": succ_arr,
        "op_dur": op_dur, "op_is_ddr": op_is_ddr,
        "seq_ext": seq_ext, "n_ops": n_ops,
        "op_end": op_end,
    }


def stage_step3(exts):
    return [stage_step3_one(e) for e in exts]


# =====================================================================
# 三、阶段驱动：step1（含邻接构造）
# =====================================================================

def _transpose_csr(op_ptr, op_ten, n_flat_ops, n_flat_ten):
    """op 主序 (op→tensor) CSR → tensor 主序 (tensor→op) CSR，op 升序。"""
    op_of_edge = np.repeat(np.arange(n_flat_ops, dtype=np.int64),
                           np.diff(op_ptr))
    key = op_ten * n_flat_ops + op_of_edge
    order = np.argsort(key)
    ptr = np.zeros(n_flat_ten + 1, dtype=np.int64)
    np.add.at(ptr, op_ten[order] + 1, 1)
    ptr = np.cumsum(ptr)
    arr = op_of_edge[order]
    return ptr, arr


def stage_step1(tb):
    """逐任务：依赖边展开（njit）→ 邻接 CSR（host）→ step1 序（njit）。

    返回 seqs: list[np.ndarray]（每任务的平铺 op 下标序列）。
    """
    from fast_kernels import fill_task_edges, step1_seq
    n_tasks = tb.n_tasks
    n_flat_ops = int(tb.t_op_ptr[-1])
    n_flat_ten = int(tb.t_ten_ptr[-1])

    ten_cons_ptr, ten_cons_arr = _transpose_csr(
        tb.in_ptr, tb.in_ten, n_flat_ops, n_flat_ten)
    ten_prod_ptr, _ = _transpose_csr(tb.out_ptr, tb.out_ten,
                                     n_flat_ops, n_flat_ten)
    prod_cnt = np.diff(ten_prod_ptr)
    cons_cnt = np.diff(ten_cons_ptr)

    d_lo = np.searchsorted(tb.direct_task, np.arange(n_tasks))
    d_hi = np.searchsorted(tb.direct_task, np.arange(1, n_tasks + 1))

    seqs = []
    for k in range(n_tasks):
        lo, hi = int(tb.t_op_ptr[k]), int(tb.t_op_ptr[k + 1])
        tlo, thi = int(tb.t_ten_ptr[k]), int(tb.t_ten_ptr[k + 1])
        n = hi - lo
        if n == 0:
            seqs.append(np.zeros(0, dtype=np.int64))
            continue
        e_cap = int(np.sum(prod_cnt[tlo:thi] * cons_cnt[tlo:thi])) + int(
            d_hi[k] - d_lo[k]) + 1
        ebuf_src = np.empty(e_cap, dtype=np.int64)
        ebuf_dst = np.empty(e_cap, dtype=np.int64)
        cnt, err = fill_task_edges(
            lo, hi, tb.out_ptr, tb.out_ten, ten_cons_ptr, ten_cons_arr,
            int(d_lo[k]), int(d_hi[k]), tb.direct_src, tb.direct_dst,
            ebuf_src, ebuf_dst)
        if err:
            raise RuntimeError("edge buffer overflow in task %d" % k)
        # 排序去重 → pred/succ CSR（局部下标）
        if cnt:
            comp = ebuf_src[:cnt] * n + ebuf_dst[:cnt]
            comp_sorted = np.unique(comp)
            src = comp_sorted // n
            dst = comp_sorted % n
        else:
            src = np.zeros(0, dtype=np.int64)
            dst = np.zeros(0, dtype=np.int64)
        pred_ptr = np.zeros(n + 1, dtype=np.int64)
        succ_ptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(pred_ptr, dst + 1, 1)
        np.add.at(succ_ptr, src + 1, 1)
        pred_ptr = np.cumsum(pred_ptr)
        succ_ptr = np.cumsum(succ_ptr)
        pred_arr = src[np.argsort(dst * n + src, kind="stable")] \
            if len(src) else np.zeros(0, dtype=np.int64)
        # preds of d: 按 dst 聚集；用排序 (dst, src) 后切分
        order2 = np.argsort(dst * n + src, kind="stable")
        pred_arr = src[order2]
        succ_arr = dst[np.argsort(src * n + dst, kind="stable")]
        is_ci = tb.op_type[lo:hi] == T_COPY_IN
        is_co = tb.op_type[lo:hi] == T_COPY_OUT
        seq = np.empty(n, dtype=np.int64)
        m = step1_seq(n, pred_ptr, pred_arr, succ_ptr, succ_arr,
                      is_ci, is_co, seq)
        if m != n:
            raise RuntimeError("step1 incomplete in task %d" % k)
        seqs.append(seq + lo)
    return seqs


# =====================================================================
# 七、场景A多核事件循环 + 对外入口
# =====================================================================

def stage_scene_a(tb, s3_list, exts, same_wait=SAME_WAIT_A,
                  cross_wait=CROSS_WAIT_A):
    from scene_a_kernel import scene_a_sim
    n_tasks = tb.n_tasks
    n_cores = tb.num_cores
    task_op_base = np.zeros(n_tasks + 1, dtype=np.int64)
    for k in range(n_tasks):
        task_op_base[k + 1] = task_op_base[k] + s3_list[k]["n_ops"]
    n_gops = int(task_op_base[-1])

    gop_pipe = np.zeros(n_gops, dtype=np.int64)
    gop_dur = np.zeros(n_gops, dtype=np.int64)
    gop_is_ddr = np.zeros(n_gops, dtype=np.bool_)
    task_of_gop = np.zeros(n_gops, dtype=np.int64)
    task_seq_parts = []
    succ_parts = []   # 全局 (src, dst)
    tsp_parts = []    # 局部 op 序（每任务 4 段）
    tsp_ptr_per_task = []
    for k in range(n_tasks):
        s3 = s3_list[k]
        b = task_op_base[k]
        for i in range(s3["n_ops"]):
            gop_pipe[b + i] = int(exts[k]["op_pipe"][i])
            gop_dur[b + i] = int(s3["op_dur"][i])
            gop_is_ddr[b + i] = bool(s3["op_is_ddr"][i])
            task_of_gop[b + i] = k
        task_seq_parts.append(s3["seq_ext"] + b)
        # succ（数据 + 内存复用，去重）
        edges = set()
        for g in range(s3["n_ops"]):
            for e in range(s3["succ_ptr"][g], s3["succ_ptr"][g + 1]):
                edges.add((g, int(s3["succ_arr"][e])))
        for s, d in zip(s3["deps"], s3["depd"]):
            edges.add((int(s), int(d)))
        succ_parts.extend((b + s, b + d) for s, d in edges)
        # 管道序（局部）
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

    # 全局 succ CSR
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

    # 核序 / 任务前驱
    core_ptr = np.zeros(n_cores + 1, dtype=np.int64)
    for c in range(n_cores):
        core_ptr[c + 1] = core_ptr[c] + len(tb.core_orders[c])
    core_tasks = np.array([t for c in range(n_cores) for t in
                           tb.core_orders[c]], dtype=np.int64)
    task_core = np.zeros(n_tasks, dtype=np.int64)
    for k in range(n_tasks):
        task_core[k] = tb.core_of[k]
    tp_parts = tb.task_preds
    task_pred_ptr = np.zeros(n_tasks + 1, dtype=np.int64)
    for k in range(n_tasks):
        task_pred_ptr[k + 1] = task_pred_ptr[k] + len(tp_parts[k])
    task_pred_arr = np.array([p for l in tp_parts for p in l], dtype=np.int64)

    # scratch
    op_status = np.zeros(n_gops, dtype=np.int64)
    op_end = np.zeros(n_gops, dtype=np.float64)
    cursor_k = np.zeros(n_tasks * 4, dtype=np.int64)
    ready_slot = np.full(n_tasks * 4, -1, dtype=np.int64)
    exec_op = np.full(n_cores * 4, -1, dtype=np.int64)
    exec_end = np.zeros(n_cores * 4, dtype=np.float64)
    task_status = np.zeros(n_tasks, dtype=np.int64)
    task_end_arr = np.zeros(n_tasks, dtype=np.int64)
    core_index = np.zeros(n_cores, dtype=np.int64)
    core_active = np.full(n_cores, -1, dtype=np.int64)
    core_prev_end = np.full(n_cores, -1, dtype=np.int64)
    pool_op = np.zeros(n_gops + 8, dtype=np.int64)
    pool_w = np.zeros(n_gops + 8, dtype=np.float64)
    pool_alive = np.zeros(n_gops + 8, dtype=np.bool_)
    pool_n = np.zeros(1, dtype=np.int64)
    pool_slot = np.full(n_gops, -1, dtype=np.int64)
    last_upd = np.zeros(1, dtype=np.float64)
    sa = np.zeros(n_gops + 8, dtype=np.int64)
    sw = np.zeros(n_gops + 8, dtype=np.float64)
    out_mk = np.zeros(1, dtype=np.float64)

    r = scene_a_sim(
        n_cores, n_tasks, n_gops,
        core_ptr, core_tasks, task_core,
        task_pred_ptr, task_pred_arr,
        task_seq_ptr, task_seq_arr, task_op_base,
        tsp_base, tsp_ptr, tsp_arr,
        gop_pipe, gop_dur, gop_is_ddr,
        succ_ptr, succ_arr, task_of_gop,
        op_status, pred_rem, op_end,
        cursor_k, ready_slot, exec_op, exec_end,
        task_status, task_end_arr, core_index, core_active, core_prev_end,
        pool_op, pool_w, pool_alive, pool_n, pool_slot,
        last_upd, sa, sw,
        same_wait, cross_wait, out_mk)
    if r < 0:
        raise RuntimeError(f"scene_a_sim error {r} now={out_mk[0]}")
    return int(round(out_mk[0])), op_end, task_end_arr


class FastEvalP1:
    """官方 evaluate_scene_a 的 numba 复刻（构造+step1/2/3+多核事件循环）。

    用法：
        fe = FastEvalP1(graph_json)     # 每图一次
        mk, info = fe.evaluate(plan)    # 每方案一次
    info 含 data_movement_bytes 各分项 / cross_task_traffic / num_cores /
    每任务 local_makespan。
    """

    def __init__(self, graph_json):
        self.graph = graph_json
        self.gc = GraphCodec(graph_json)

    def evaluate(self, plan):
        gc = self.gc
        pv = derive_multicore_plan(self.graph, plan)
        tb = TaskBuild(gc, pv)
        seqs = stage_step1(tb)
        pss = stage_step2(tb, seqs)
        exts = build_ext_all(tb, seqs, pss)
        s3 = stage_step3(exts)
        mk, op_end_g, task_ends = stage_scene_a(tb, s3, exts)
        spill = sum(e["spill_traffic"] for e in exts)
        task_copy = tb.task_graph_copy_traffic
        original = gc.original_copy_traffic
        info = {
            "makespan": mk,
            "num_cores": tb.num_cores,
            "local_makespans": [x["makespan"] for x in s3],
            "spill_added_copy_bytes": int(spill),
            "partition_added_copy_bytes": int(task_copy - original),
            "scheduled_copy_bytes": int(task_copy + spill),
            "original_copy_bytes": int(original),
            "added_copy_bytes": int(task_copy - original + spill),
            "cross_task_traffic": int(tb.cross_task_traffic),
            "n_tasks": tb.n_tasks,
        }
        return mk, info
