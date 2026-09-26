# -*- coding: utf-8 -*-
"""GPU 批量工作包：把 CPU 复刻内核搬上 4090，与 CPU 链路逐位对齐。

v2：纯索引运算（无切片视图），避免 numba cuda 视图兼容性问题。
"""
import numpy as np
from numba import cuda

KEY_M = 1 << 21
KEY_MASK = KEY_M - 1


@cuda.jit
def step1_batch_kernel(op_base, pred_ptr, pred_arr, succ_ptr, succ_arr,
                       is_ci_flat, is_co_flat,
                       keys_flat, skeys_flat, depth_flat, topo_flat,
                       stack_flat, visited_flat, seq_flat, tmp_flat,
                       indeg_flat, max_ops, stack_stride):
    """一线程一 (plan,task)：官方 step1_from_adj 复刻（纯索引版）。"""
    i = cuda.grid(1)
    if i >= op_base.size - 1:
        return
    lo = op_base[i]
    hi = op_base[i + 1]
    n = hi - lo
    kb = i * max_ops
    stb = i * stack_stride

    # ---- Kahn 最长路 ----
    for v in range(n):
        indeg_flat[kb + v] = pred_ptr[lo + v + 1] - pred_ptr[lo + v]
        depth_flat[kb + v] = 0
        visited_flat[kb + v] = False
        seq_flat[lo + v] = -1
    tail = 0
    for v in range(n):
        if indeg_flat[kb + v] == 0:
            topo_flat[kb + tail] = v
            tail += 1
    head = 0
    while head < tail:
        u = topo_flat[kb + head]
        head += 1
        for e in range(succ_ptr[lo + u], succ_ptr[lo + u + 1]):
            w = succ_arr[e]
            if depth_flat[kb + u] + 1 > depth_flat[kb + w]:
                depth_flat[kb + w] = depth_flat[kb + u] + 1
            indeg_flat[kb + w] -= 1
            if indeg_flat[kb + w] == 0:
                topo_flat[kb + tail] = w
                tail += 1

    # ---- 复合键 ----
    for v in range(n):
        a1 = 1 if is_ci_flat[lo + v] == 0 else 0
        a2 = 1 if is_co_flat[lo + v] == 0 else 0
        keys_flat[kb + v] = (a1 * KEY_M + depth_flat[kb + v]) * KEY_M \
            + (KEY_MASK - v)
        skeys_flat[kb + v] = (a2 * KEY_M + depth_flat[kb + v]) * KEY_M \
            + (KEY_MASK - v)

    # ---- 出发点（sink 节点按 skey 插入排序）----
    sp = 0
    n_starts = 0
    for v in range(n):
        if succ_ptr[lo + v + 1] - succ_ptr[lo + v] == 0:
            tmp_flat[kb + n_starts] = v
            n_starts += 1
    if n_starts == 0:
        stack_flat[stb] = 0
        sp = 1
    else:
        for z in range(1, n_starts):
            kv = skeys_flat[kb + tmp_flat[kb + z]]
            vv = tmp_flat[kb + z]
            j = z - 1
            while j >= 0 and skeys_flat[kb + tmp_flat[kb + j]] > kv:
                tmp_flat[kb + j + 1] = tmp_flat[kb + j]
                j -= 1
            tmp_flat[kb + j + 1] = vv
        for j in range(n_starts):
            stack_flat[stb + sp] = tmp_flat[kb + j]
            sp += 1

    # ---- DFS 主循环 ----
    seq_len = 0
    n_visited = 0
    while sp > 0 or n_visited < n:
        if sp == 0:
            cnt = 0
            for v in range(n):
                if not visited_flat[kb + v]:
                    tmp_flat[kb + cnt] = v
                    cnt += 1
            for z in range(1, cnt):
                kv = keys_flat[kb + tmp_flat[kb + z]]
                vv = tmp_flat[kb + z]
                j = z - 1
                while j >= 0 and keys_flat[kb + tmp_flat[kb + j]] > kv:
                    tmp_flat[kb + j + 1] = tmp_flat[kb + j]
                    j -= 1
                tmp_flat[kb + j + 1] = vv
            for j in range(cnt):
                stack_flat[stb + sp] = tmp_flat[kb + j]
                sp += 1
            continue
        u = stack_flat[stb + sp - 1]
        if visited_flat[kb + u]:
            sp -= 1
            continue
        pc = 0
        for e in range(pred_ptr[lo + u], pred_ptr[lo + u + 1]):
            p = pred_arr[e]
            if not visited_flat[kb + p]:
                tmp_flat[kb + pc] = p
                pc += 1
        if pc > 0:
            for z in range(1, pc):
                kv = keys_flat[kb + tmp_flat[kb + z]]
                vv = tmp_flat[kb + z]
                j = z - 1
                while j >= 0 and keys_flat[kb + tmp_flat[kb + j]] > kv:
                    tmp_flat[kb + j + 1] = tmp_flat[kb + j]
                    j -= 1
                tmp_flat[kb + j + 1] = vv
            for z in range(pc):
                stack_flat[stb + sp] = tmp_flat[kb + z]
                sp += 1
        else:
            visited_flat[kb + u] = True
            n_visited += 1
            seq_flat[lo + u] = seq_len
            seq_len += 1
            sp -= 1
