# -*- coding: utf-8 -*-
"""场景A多核事件循环的 numba 复刻内核（multicore_cut_evaluate_problem_1）。

状态空间：全局 op 下标（task_op_base[k] + 局部下标）。
  - 每核每管道单发射槽（PIPE_SLOTS=1）
  - cursor 规则：只有 (task, pipe) 的 cursor op 能进入就绪槽
  - 任务激活：核序 + 跨核/同核等待
  - DDR 公平带宽池（全核共享，按 (work, op) 投影，ceil 取整）
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _queue_ready(gop, op_status, pred_rem, gop_pipe, task_of_gop,
                 tsp_base, tsp_ptr, tsp_arr, cursor_k, ready_slot,
                 task_op_base):
    k = task_of_gop[gop]
    pipe = gop_pipe[gop]
    ck = k * 4 + pipe
    if op_status[gop] != 0 or pred_rem[gop] != 0:
        return 0
    # cursor 检查：该 (task,pipe) 的第 cursor 个 op（tsp_ptr 为 k×5 平铺）
    base = tsp_base[k] + tsp_ptr[k * 5 + pipe]
    cnt = tsp_ptr[k * 5 + pipe + 1] - tsp_ptr[k * 5 + pipe]
    cur = cursor_k[ck]
    if cur >= cnt:
        return 0
    if tsp_arr[base + cur] != gop - task_op_base[k]:
        return 0
    op_status[gop] = 1
    if ready_slot[ck] != -1:
        return -6
    ready_slot[ck] = gop
    return 0


@njit(cache=True)
def _advance_ddr(now, last_upd, pool_op, pool_w, pool_alive, pool_n, sa):
    elapsed = now - last_upd[0]
    while elapsed > 1e-9:
        n_active = 0
        for i in range(pool_n[0]):
            if pool_alive[i] and pool_w[i] > 1e-9:
                sa[n_active] = i
                n_active += 1
        if n_active == 0:
            break
        min_work = pool_w[sa[0]]
        for i in range(1, n_active):
            if pool_w[sa[i]] < min_work:
                min_work = pool_w[sa[i]]
        ttf = min_work * n_active
        if ttf >= elapsed - 1e-9:
            share = elapsed / n_active
            for i in range(n_active):
                j = sa[i]
                w = pool_w[j] - share
                pool_w[j] = 0.0 if w < 0.0 else w
            break
        for i in range(n_active):
            j = sa[i]
            w = pool_w[j] - min_work
            pool_w[j] = 0.0 if w < 0.0 else w
        elapsed -= ttf
    last_upd[0] = now


@njit(cache=True)
def _reschedule_ddr_a(now, pool_op, pool_w, pool_alive, pool_n,
                      sa, sw, op_end, n_cores, exec_op, exec_end):
    """problem_1.reschedule_ddr：cursor += (work-prev)*n_active，ceil 投影。"""
    n = 0
    for i in range(pool_n[0]):
        if pool_alive[i]:
            w = pool_w[i]
            if w < 0.0:
                w = 0.0
            sa[n] = pool_op[i]
            sw[n] = w
            n += 1
    if n == 0:
        return
    for i in range(1, n):
        ka = sa[i]
        kw = sw[i]
        j = i - 1
        while j >= 0 and (sw[j] > kw or (sw[j] == kw and sa[j] > ka)):
            sa[j + 1] = sa[j]
            sw[j + 1] = sw[j]
            j -= 1
        sa[j + 1] = ka
        sw[j + 1] = kw
    cursor = np.float64(now)
    previous = 0.0
    active = n
    i = 0
    while i < n:
        work = sw[i]
        cursor += (work - previous) * active
        j = i
        while j < n and abs(sw[j] - work) <= 1e-9:
            op_end[sa[j]] = np.float64(np.ceil(cursor - 1e-9))
            j += 1
        active -= j - i
        previous = work
        i = j
    for c in range(n_cores):
        for p in range(4):
            e = c * 4 + p
            if exec_op[e] != -1:
                for q in range(n):
                    if sa[q] == exec_op[e]:
                        exec_end[e] = op_end[exec_op[e]]
                        break


@njit(cache=True)
def scene_a_sim(n_cores, n_tasks, n_gops,
                core_ptr, core_tasks,
                task_core, task_pred_ptr, task_pred_arr,
                task_seq_ptr, task_seq_arr,   # 每任务 seq_ext（全局 op）
                task_op_base,
                tsp_base, tsp_ptr, tsp_arr,   # 每任务×4管道 序（局部 op）
                gop_pipe, gop_dur, gop_is_ddr,
                succ_ptr, succ_arr,
                task_of_gop,
                op_status, pred_rem, op_end,
                cursor_k, ready_slot,
                exec_op, exec_end,
                task_status, task_end_arr, core_index, core_active,
                core_prev_end,
                pool_op, pool_w, pool_alive, pool_n, pool_slot,
                last_upd, sa, sw,
                same_wait, cross_wait, out_mk):
    """返回 0 正常；负数错误。out_mk[0]=makespan。"""
    for g in range(n_gops):
        op_status[g] = 0
        op_end[g] = 0.0
    for k in range(n_tasks):
        task_status[k] = 0     # 0 waiting 1 active 2 done
        task_end_arr[k] = 0
    for c in range(n_cores):
        core_index[c] = 0
        core_active[c] = -1
        core_prev_end[c] = -1
    n_exec = n_cores * 4
    for e in range(n_exec):
        exec_op[e] = -1
        exec_end[e] = 0.0
    for k in range(n_tasks * 4):
        cursor_k[k] = 0
        ready_slot[k] = -1
    pool_n[0] = 0
    last_upd[0] = 0.0
    now = 0.0

    pool_dead = 0
    it = 0
    while True:
        it += 1
        if it > 2000000:
            return -7
        # ---- retire ----
        _advance_ddr(now, last_upd, pool_op, pool_w, pool_alive, pool_n, sa)
        retired_ddr = False
        for e in range(n_exec):
            if exec_op[e] != -1 and exec_end[e] <= now + 1e-9:
                g = exec_op[e]
                exec_op[e] = -1
                op_status[g] = 3
                k = task_of_gop[g]
                pipe = gop_pipe[g]
                ck = k * 4 + pipe
                cur = cursor_k[ck]
                cursor_k[ck] = cur + 1
                base = tsp_base[k] + tsp_ptr[k * 5 + pipe]
                cnt = tsp_ptr[k * 5 + pipe + 1] - tsp_ptr[k * 5 + pipe]
                if cur + 1 < cnt:
                    nxt = task_op_base[k] + tsp_arr[base + cur + 1]
                    r = _queue_ready(nxt, op_status, pred_rem, gop_pipe,
                                     task_of_gop, tsp_base, tsp_ptr, tsp_arr,
                                     cursor_k, ready_slot, task_op_base)
                    if r < 0:
                        return r
                if pool_slot[g] != -1:
                    pool_alive[pool_slot[g]] = False
                    pool_slot[g] = -1
                    retired_ddr = True
                    pool_dead += 1
                    if pool_dead > 64:
                        w = 0
                        for q in range(pool_n[0]):
                            if pool_alive[q]:
                                if w != q:
                                    pool_op[w] = pool_op[q]
                                    pool_w[w] = pool_w[q]
                                    pool_alive[w] = True
                                    pool_slot[pool_op[w]] = w
                                w += 1
                        pool_n[0] = w
                        pool_dead = 0
                for e2 in range(succ_ptr[g], succ_ptr[g + 1]):
                    s2 = succ_arr[e2]
                    pred_rem[s2] -= 1
                    r = _queue_ready(s2, op_status, pred_rem, gop_pipe,
                                     task_of_gop, tsp_base, tsp_ptr, tsp_arr,
                                     cursor_k, ready_slot, task_op_base)
                    if r < 0:
                        return r
        if retired_ddr:
            _reschedule_ddr_a(now, pool_op, pool_w, pool_alive, pool_n,
                              sa, sw, op_end, n_cores, exec_op, exec_end)
        # ---- 任务完成检查 ----
        for k in range(n_tasks):
            if task_status[k] != 1:
                continue
            b0 = task_seq_ptr[k]
            b1 = task_seq_ptr[k + 1]
            all_done = True
            for e in range(b0, b1):
                if op_status[task_seq_arr[e]] != 3:
                    all_done = False
                    break
            if all_done:
                task_status[k] = 2
                task_end_arr[k] = int(now)
                c = task_core[k]
                core_active[c] = -1
                core_prev_end[c] = int(now)
                core_index[c] += 1
        # ---- activate ----
        for c in range(n_cores):
            if core_active[c] != -1:
                continue
            if core_index[c] >= core_ptr[c + 1] - core_ptr[c]:
                continue
            k = core_tasks[core_ptr[c] + core_index[c]]
            ok_pred = True
            for e in range(task_pred_ptr[k], task_pred_ptr[k + 1]):
                if task_status[task_pred_arr[e]] != 2:
                    ok_pred = False
                    break
            if ok_pred:
                rel = 0
                if core_prev_end[c] >= 0:
                    rel = core_prev_end[c] + same_wait
                for e in range(task_pred_ptr[k], task_pred_ptr[k + 1]):
                    pk = task_pred_arr[e]
                    if task_core[pk] != c:
                        r2 = task_end_arr[pk] + cross_wait
                        if r2 > rel:
                            rel = r2
                if rel <= now:
                    task_status[k] = 1
                    core_active[c] = k
                    for e in range(task_seq_ptr[k], task_seq_ptr[k + 1]):
                        g = task_seq_arr[e]
                        r = _queue_ready(g, op_status, pred_rem, gop_pipe,
                                         task_of_gop, tsp_base, tsp_ptr,
                                         tsp_arr, cursor_k, ready_slot,
                                         task_op_base)
                        if r < 0:
                            return r
        all_tasks_done = True
        for k in range(n_tasks):
            if task_status[k] != 2:
                all_tasks_done = False
                break
        if all_tasks_done:
            break
        # ---- issue ----
        issued_pass = True
        while issued_pass:
            issued_pass = False
            for c in range(n_cores):
                k = core_active[c]
                if k == -1:
                    continue
                for p in range(4):
                    e = c * 4 + p
                    while exec_op[e] == -1:
                        g = ready_slot[k * 4 + p]
                        if g == -1:
                            break
                        ready_slot[k * 4 + p] = -1
                        issued_pass = True
                        op_status[g] = 2
                        dur = gop_dur[g]
                        op_end[g] = now + np.float64(dur)
                        exec_op[e] = g
                        exec_end[e] = op_end[g]
                        if gop_is_ddr[g]:
                            _advance_ddr(now, last_upd, pool_op, pool_w,
                                         pool_alive, pool_n, sa)
                            if pool_n[0] >= pool_op.size:
                                return -3
                            slot = pool_n[0]
                            pool_op[slot] = g
                            pool_w[slot] = np.float64(dur)
                            pool_alive[slot] = True
                            pool_slot[g] = slot
                            pool_n[0] += 1
                            _reschedule_ddr_a(now, pool_op, pool_w,
                                              pool_alive, pool_n, sa, sw,
                                              op_end, n_cores,
                                              exec_op, exec_end)

        # ---- 时间推进（executor 端点 + 待释放任务时刻，始终合并收集）----
        has = False
        t_next = 0.0
        for e in range(n_exec):
            if exec_op[e] != -1:
                if not has or exec_end[e] < t_next:
                    t_next = exec_end[e]
                    has = True
        for c in range(n_cores):
            if core_active[c] != -1:
                continue
            if core_index[c] >= core_ptr[c + 1] - core_ptr[c]:
                continue
            k = core_tasks[core_ptr[c] + core_index[c]]
            ok_pred = True
            for e in range(task_pred_ptr[k], task_pred_ptr[k + 1]):
                if task_status[task_pred_arr[e]] != 2:
                    ok_pred = False
                    break
            if ok_pred:
                rel = 0
                if core_prev_end[c] >= 0:
                    rel = core_prev_end[c] + same_wait
                for e in range(task_pred_ptr[k], task_pred_ptr[k + 1]):
                    pk = task_pred_arr[e]
                    if task_core[pk] != c:
                        r2 = task_end_arr[pk] + cross_wait
                        if r2 > rel:
                            rel = r2
                if rel > now:
                    if not has or np.float64(rel) < t_next:
                        t_next = np.float64(rel)
                        has = True
        if not has:
            out_mk[0] = now
            return -2
        if t_next <= now:
            out_mk[0] = now
            return -8
        now = t_next

    mk = 0
    for k in range(n_tasks):
        if task_end_arr[k] > mk:
            mk = task_end_arr[k]
    out_mk[0] = np.float64(mk)
    return 0
