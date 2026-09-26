# -*- coding: utf-8 -*-
"""GPU step2：Belady spill 模拟的 CUDA 批量内核（纯索引模式）。

与 fast_kernels.step2_sim 逐位对齐。
"""
import numpy as np
from numba import cuda


@cuda.jit
def step2_batch_kernel(op_base, ten_packed_base, step_packed_base,
                       uses_ptr, uses_step, uses_op,
                       uat_ptr, uat_tid, uat_idx,
                       ten_is_l1_flat, ten_size_flat,
                       cap_l1, cap_ub,
                       # scratch（按线程分区，宿主保证容量）
                       act_alive, act_nu, act_ui, act_rank, act_order,
                       resid_flat, cur_gen_flat, rel_buf,
                       si_head, si_tail, si_next, si_tid_arr, si_ui_arr,
                       ps_victim, ps_prev_step, ps_next_step,
                       ps_prev_op, ps_next_op, ps_ui, ps_type, ps_size,
                       ps_n_flat,
                       max_ops, max_ten, ps_cap, act_cap):
    """一线程一 (plan,task)。返回写入 ps_n_flat[i]。"""
    i = cuda.grid(1)
    if i >= op_base.size - 1:
        return
    n = int(op_base[i + 1] - op_base[i])
    n_ten = int(ten_packed_base[i + 1] - ten_packed_base[i])
    kb = i * max_ten          # scratch 张量状态基址
    tb = int(ten_packed_base[i])  # 打包数据（uses_ptr/ten_is_l1/ten_size）基址
    spb = int(step_packed_base[i])  # uat_ptr 打包基址
    sb = i * ps_cap           # pending spill 基址
    aob = i * act_cap         # act_order 基址（独立分区，防重叠）
    rb = i * max_ops          # rel_buf 基址
    ub = i * (n + 1)          # si_head/si_tail 基址（按步）

    # 状态清零
    resid_flat[i * 2] = 0
    resid_flat[i * 2 + 1] = 0
    for t in range(n_ten):
        act_alive[kb + t] = False
        act_nu[kb + t] = 0
        act_ui[kb + t] = 0
        act_rank[kb + t] = 0
        cur_gen_flat[kb + t] = 0
    for s in range(n + 1):
        si_head[ub + s] = -1
        si_tail[ub + s] = -1
    ps_n_flat[i] = 0
    rank_counter = 0
    act_len = 0
    n_ps = 0

    # 主循环（步序 = 0..n-1）
    for t in range(n):
        curg = t + 1
        for e in range(uat_ptr[spb + t], uat_ptr[spb + t + 1]):
            cur_gen_flat[kb + uat_tid[e]] = curg

        # 0. 计划内 COPY_IN（物理换回）
        ent = si_head[ub + t]
        while ent != -1:
            tid = si_tid_arr[sb + ent]
            ui = si_ui_arr[sb + ent]
            T = 0 if ten_is_l1_flat[tb + tid] else 1
            if not act_alive[kb + tid]:
                up = uses_ptr[tb + tid]
                n_use = int(uses_ptr[tb + tid + 1] - up)
                new_nu = uses_step[up + ui + 1] if ui + 1 < n_use else -1
                act_alive[kb + tid] = True
                act_nu[kb + tid] = new_nu
                act_ui[kb + tid] = ui
                act_rank[kb + tid] = rank_counter
                rank_counter += 1
                if act_len < act_cap:
                    act_order[aob + act_len] = tid
                    act_len += 1
                if T == 0:
                    resid_flat[i * 2] += ten_size_flat[tb + tid]
                else:
                    resid_flat[i * 2 + 1] += ten_size_flat[tb + tid]
            ent = si_next[sb + ent]

        # 1. 本步 use：first alloc / mid 更新；末次记 rel_buf
        rel_cnt = 0
        for e in range(uat_ptr[spb + t], uat_ptr[spb + t + 1]):
            tid = uat_tid[e]
            idx = uat_idx[e]
            T = 0 if ten_is_l1_flat[tb + tid] else 1
            up = uses_ptr[tb + tid]
            n_use = int(uses_ptr[tb + tid + 1] - up)
            if idx == 0:
                next_nu = uses_step[up + 1] if 1 < n_use else -1
                if not act_alive[kb + tid]:
                    act_alive[kb + tid] = True
                    act_rank[kb + tid] = rank_counter
                    rank_counter += 1
                    if act_len < act_cap:
                        act_order[aob + act_len] = tid
                        act_len += 1
                    if T == 0:
                        resid_flat[i * 2] += ten_size_flat[tb + tid]
                    else:
                        resid_flat[i * 2 + 1] += ten_size_flat[tb + tid]
                act_nu[kb + tid] = next_nu
                act_ui[kb + tid] = 1
            elif idx < n_use - 1:
                if act_alive[kb + tid]:
                    act_nu[kb + tid] = uses_step[up + idx + 1]
                    act_ui[kb + tid] += 1
            if idx == n_use - 1:
                rel_buf[rb + rel_cnt] = tid
                rel_cnt += 1

        # 1.5 容量检查 + Belady SPILL（L1 先于 UB）
        for T in range(2):
            cap = cap_l1 if T == 0 else cap_ub
            if i == 0 and T == 1 and resid_flat[i * 2 + 1] > cap:
', t,
                       resid_flat[i * 2 + 1], cap)
            while (resid_flat[i * 2 + T] if T == 0
                   else resid_flat[i * 2 + 1]) > cap:
                best = -1
                best_nu = -1
                best_rank = 0
                for a in range(act_len):
                    tid = act_order[aob + a]
                    if not act_alive[kb + tid]:
                        continue
                    nu = act_nu[kb + tid]
                    if nu < 0:
                        continue
                    if (0 if ten_is_l1_flat[tb + tid] else 1) != T:
                        continue
                    if cur_gen_flat[kb + tid] == curg:
                        continue
                    rk = act_rank[kb + tid]
                    if best == -1 or nu > best_nu or (nu == best_nu
                                                      and rk < best_rank):
                        best = tid
                        best_nu = nu
                        best_rank = rk
                if best == -1:
                    return
                if n_ps >= ps_cap:
                    return
                ui = act_ui[kb + best]
                up = uses_ptr[tb + best]
                n_use = int(uses_ptr[tb + best + 1] - up)
                if ui > 0:
                    ps_prev_step[sb + n_ps] = uses_step[up + ui - 1]
                    ps_prev_op[sb + n_ps] = uses_op[up + ui - 1]
                else:
                    ps_prev_step[sb + n_ps] = uses_step[up]
                    ps_prev_op[sb + n_ps] = uses_op[up]
                ps_next_step[sb + n_ps] = best_nu
                ps_next_op[sb + n_ps] = uses_op[up + ui]
                ps_victim[sb + n_ps] = best
                ps_ui[sb + n_ps] = ui
                ps_type[sb + n_ps] = T
                ps_size[sb + n_ps] = ten_size_flat[tb + best]
                n_ps += 1
                act_alive[kb + best] = False
                if T == 0:
                    resid_flat[i * 2] -= ten_size_flat[tb + best]
                else:
                    resid_flat[i * 2 + 1] -= ten_size_flat[tb + best]
                # spill_in 挂到 next_use 步链表尾
                if n_ps <= ps_cap:
                    si_tid_arr[sb + n_ps - 1] = best
                    si_ui_arr[sb + n_ps - 1] = ui
                    si_next[sb + n_ps - 1] = -1
                    stp = best_nu
                    if 0 <= stp <= n:
                        if si_head[ub + stp] == -1:
                            si_head[ub + stp] = n_ps - 1
                            si_tail[ub + stp] = n_ps - 1
                        else:
                            si_next[sb + si_tail[ub + stp]] = n_ps - 1
                            si_tail[ub + stp] = n_ps - 1

        # 2. 释放本步末次使用的输入/输出
        for r in range(rel_cnt):
            tid = rel_buf[rb + r]
            if act_alive[kb + tid]:
                act_alive[kb + tid] = False
                T = 0 if ten_is_l1_flat[tb + tid] else 1
                if T == 0:
                    resid_flat[i * 2] -= ten_size_flat[tb + tid]
                else:
                    resid_flat[i * 2 + 1] -= ten_size_flat[tb + tid]

    ps_n_flat[i] = n_ps
