# -*- coding: utf-8 -*-
"""A题2026 fast_eval 的 numba 内核（step1/step2/step3/场景A）。

纪律：每个内核的迭代顺序/排序键/浮点运算次序与官方逐点对齐；
排序键一律编码为唯一 int64 复合键（无 stability 依赖）。
numba 0.60（node1 base）兼容：不使用 np.random*；np.argsort/np.sort/
searchsorted/np.where 等基础算子可用。
"""
import numpy as np
from numba import njit

KEY_BITS = 21                 # 局部下标/depth < 2^21（任务规模 ≤ ~2M）
KEY_M = 1 << KEY_BITS         # 2097152
KEY_MASK = KEY_M - 1


@njit(cache=True)
def _encode_key(a, depth, idx):
    """官方 key=(¬is_copy_in, depth, -id) 升序的复合键。
    等价序：a 升序 → depth 升序 → idx 降序。"""
    return ((np.int64(a) * KEY_M + np.int64(depth)) * KEY_M
            + (KEY_MASK - np.int64(idx)))


@njit(cache=True)
def fill_task_edges(op_lo, op_hi, out_ptr, out_ten,
                    ten_cons_ptr, ten_cons_arr,
                    d_lo, d_hi, direct_src, direct_dst,
                    ebuf_src, ebuf_dst):
    """任务图的 op-op 依赖边展开（tensor 笛卡尔积 + direct 边）。

    填 ebuf（局部下标），返回 (cnt, err)；err=1 表示容量不足。
    去重/排序由宿主 numpy 完成。
    """
    cnt = 0
    for u in range(op_lo, op_hi):
        for e in range(out_ptr[u], out_ptr[u + 1]):
            t = out_ten[e]
            for c in range(ten_cons_ptr[t], ten_cons_ptr[t + 1]):
                if cnt >= ebuf_src.size:
                    return 0, 1
                ebuf_src[cnt] = u - op_lo
                ebuf_dst[cnt] = ten_cons_arr[c] - op_lo
                cnt += 1
    for e in range(d_lo, d_hi):
        if cnt >= ebuf_src.size:
            return 0, 1
        ebuf_src[cnt] = direct_src[e] - op_lo
        ebuf_dst[cnt] = direct_dst[e] - op_lo
        cnt += 1
    return cnt, 0


# ---------------- step2：Belady spill 插入（模拟内核）----------------

@njit(cache=True)
def step2_sim(n_steps, seq_local,
              uses_ptr, uses_step, uses_op,
              uat_ptr, uat_tid, uat_idx,
              ten_is_l1, ten_size,
              cap_l1, cap_ub,
              act_alive, act_nu, act_ui, act_rank, act_order,
              resid, cur_gen, rel_buf,
              si_head, si_tail, si_next, si_tid, si_ui,
              ps_victim, ps_prev_step, ps_next_step, ps_prev_op,
              ps_next_op, ps_ui, ps_type, ps_size, ps_nu):
    """官方 step2_spill_insertion 主模拟复刻（语义逐点对齐）。

    局部下标；cap: (L1, UB)。返回 spill 事件数；负数 = 错误码。
    - act_*: 插入序活跃表（act_order 记录 append 顺序）
    - si_*: spill_in 按步链表（append 顺序）
    - ps_*: pending_spills 按 trigger 顺序
    """
    rank_counter = 0
    act_len = 0
    resid[0] = 0
    resid[1] = 0
    n_ps = 0
    for t in range(n_steps):
        curg = t + 1
        for e in range(uat_ptr[t], uat_ptr[t + 1]):
            cur_gen[uat_tid[e]] = curg

        # 0. 计划内 COPY_IN（物理换回），append 顺序
        ent = si_head[t]
        while ent != -1:
            tid = si_tid[ent]
            ui = si_ui[ent]
            T = 0 if ten_is_l1[tid] else 1
            if not act_alive[tid]:
                up = uses_ptr[tid]
                n_use = uses_ptr[tid + 1] - up
                new_nu = uses_step[up + ui + 1] if ui + 1 < n_use else -1
                act_alive[tid] = True
                act_nu[tid] = new_nu
                act_ui[tid] = ui
                act_rank[tid] = rank_counter
                rank_counter += 1
                if act_len >= act_order.size:
                    return -3
                act_order[act_len] = tid
                act_len += 1
                resid[T] += ten_size[tid]
            ent = si_next[ent]

        # 1. 本步 use：first alloc / mid 更新；末次使用记入 rel_buf
        rel_cnt = 0
        for e in range(uat_ptr[t], uat_ptr[t + 1]):
            tid = uat_tid[e]
            idx = uat_idx[e]
            T = 0 if ten_is_l1[tid] else 1
            up = uses_ptr[tid]
            n_use = uses_ptr[tid + 1] - up
            if idx == 0:
                next_nu = uses_step[up + 1] if 1 < n_use else -1
                if not act_alive[tid]:
                    act_alive[tid] = True
                    act_rank[tid] = rank_counter
                    rank_counter += 1
                    if act_len >= act_order.size:
                        return -3
                    act_order[act_len] = tid
                    act_len += 1
                    resid[T] += ten_size[tid]
                act_nu[tid] = next_nu
                act_ui[tid] = 1
            elif idx < n_use - 1:
                if act_alive[tid]:
                    act_nu[tid] = uses_step[up + idx + 1]
                    act_ui[tid] += 1
            if idx == n_use - 1:
                rel_buf[rel_cnt] = tid
                rel_cnt += 1

        # 1.5 容量检查 + Belady SPILL（L1 先于 UB）
        for T in range(2):
            cap = cap_l1 if T == 0 else cap_ub
            while resid[T] > cap:
                best = -1
                best_nu = -1
                best_rank = 0
                for i in range(act_len):
                    tid = act_order[i]
                    if not act_alive[tid]:
                        continue
                    nu = act_nu[tid]
                    if nu < 0:
                        continue
                    if (0 if ten_is_l1[tid] else 1) != T:
                        continue
                    if cur_gen[tid] == curg:
                        continue
                    rk = act_rank[tid]
                    if best == -1 or nu > best_nu or (nu == best_nu
                                                      and rk < best_rank):
                        best = tid
                        best_nu = nu
                        best_rank = rk
                if best == -1:
                    return -1
                if n_ps >= ps_victim.size:
                    return -3
                ui = act_ui[best]
                up = uses_ptr[best]
                n_use = uses_ptr[best + 1] - up
                if ui > 0:
                    ps_prev_step[n_ps] = uses_step[up + ui - 1]
                    ps_prev_op[n_ps] = uses_op[up + ui - 1]
                else:
                    ps_prev_step[n_ps] = uses_step[up]
                    ps_prev_op[n_ps] = uses_op[up]
                ps_next_step[n_ps] = best_nu
                ps_next_op[n_ps] = uses_op[up + ui]
                ps_victim[n_ps] = best
                ps_ui[n_ps] = ui
                ps_type[n_ps] = T
                ps_size[n_ps] = ten_size[best]
                ps_nu[n_ps] = best_nu
                n_ps += 1
                act_alive[best] = False
                resid[T] -= ten_size[best]
                # spill_in 挂到 next_use 步（append 到该步链表尾；节点 id=ps 序）
                ent = n_ps - 1
                si_tid[ent] = best
                si_ui[ent] = ui
                si_next[ent] = -1
                stp = best_nu
                if si_head[stp] == -1:
                    si_head[stp] = ent
                    si_tail[stp] = ent
                else:
                    si_next[si_tail[stp]] = ent
                    si_tail[stp] = ent

        # 2. 释放本步末次使用的输入/输出
        for i in range(rel_cnt):
            tid = rel_buf[i]
            if act_alive[tid]:
                act_alive[tid] = False
                T = 0 if ten_is_l1[tid] else 1
                resid[T] -= ten_size[tid]
    return n_ps

@njit(cache=True)
def step1_seq(n, pred_ptr, pred_arr, succ_ptr, succ_arr,
              is_copy_in, is_copy_out, seq_out):
    """官方 step1_from_adj 复刻。返回写入长度（= n）。

    pred/succ 为去重升序 CSR（局部下标）。
    depth 用 Kahn 最长路（值与官方 BFS 收敛结果一致：均=最长路深度）。
    栈容量上界：每条 (u,p) 前驱对至多压栈一次 + starts/refill ≤ n+n。
    """
    # ---- depth（Kahn 最长路）----
    depth = np.empty(n, dtype=np.int64)
    indeg = np.empty(n, dtype=np.int64)
    for v in range(n):
        indeg[v] = pred_ptr[v + 1] - pred_ptr[v]
        depth[v] = 0
    topo = np.empty(n, dtype=np.int64)
    tail = 0
    for v in range(n):
        if indeg[v] == 0:
            topo[tail] = v
            tail += 1
    head = 0
    while head < tail:
        u = topo[head]
        head += 1
        for k in range(succ_ptr[u], succ_ptr[u + 1]):
            w = succ_arr[k]
            if depth[u] + 1 > depth[w]:
                depth[w] = depth[u] + 1
            indeg[w] -= 1
            if indeg[w] == 0:
                topo[tail] = w
                tail += 1

    # ---- 复合键（唯一 int64，避免稳定排序依赖）----
    key = np.empty(n, dtype=np.int64)
    skey = np.empty(n, dtype=np.int64)
    for v in range(n):
        key[v] = _encode_key(1 if is_copy_in[v] == 0 else 0, depth[v], v)
        skey[v] = _encode_key(1 if is_copy_out[v] == 0 else 0, depth[v], v)

    m = pred_arr.size
    stack = np.empty(n + m + 2, dtype=np.int64)

    # ---- 出发点（succ 为空；无则下标 0 = id 最小）----
    n_starts = 0
    for v in range(n):
        if succ_ptr[v + 1] - succ_ptr[v] == 0:
            n_starts += 1
    if n_starts == 0:
        stack[0] = 0
        sp = 1
    else:
        starts = np.empty(n_starts, dtype=np.int64)
        sk = np.empty(n_starts, dtype=np.int64)
        c = 0
        for v in range(n):
            if succ_ptr[v + 1] - succ_ptr[v] == 0:
                starts[c] = v
                sk[c] = skey[v]
                c += 1
        o = np.argsort(sk)
        sp = 0
        for j in range(n_starts):
            stack[sp] = starts[o[j]]
            sp += 1

    visited = np.zeros(n, dtype=np.bool_)
    seq_len = 0
    n_visited = 0
    tmp = np.empty(n, dtype=np.int64)
    pk = np.empty(n, dtype=np.int64)

    while sp > 0 or n_visited < n:
        if sp == 0:
            # 剩余节点按 key 升序压栈
            cnt = 0
            for v in range(n):
                if not visited[v]:
                    tmp[cnt] = v
                    cnt += 1
            for z in range(cnt):
                pk[z] = key[tmp[z]]
            kk = np.empty(cnt, dtype=np.int64)
            for z in range(cnt):
                kk[z] = pk[z]
            o = np.argsort(kk)
            for j in range(cnt):
                stack[sp] = tmp[o[j]]
                sp += 1
            continue
        u = stack[sp - 1]
        if visited[u]:
            sp -= 1
            continue
        pc = 0
        for k in range(pred_ptr[u], pred_ptr[u + 1]):
            p = pred_arr[k]
            if not visited[p]:
                tmp[pc] = p
                pc += 1
        if pc > 0:
            for z in range(pc):
                pk[z] = key[tmp[z]]
            kk = np.empty(pc, dtype=np.int64)
            for z in range(pc):
                kk[z] = pk[z]
            o = np.argsort(kk)
            for z in range(pc):
                stack[sp] = tmp[o[z]]
                sp += 1
        else:
            visited[u] = True
            n_visited += 1
            seq_out[seq_len] = u
            seq_len += 1
            sp -= 1
    return seq_len
