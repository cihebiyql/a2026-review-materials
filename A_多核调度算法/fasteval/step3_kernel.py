# -*- coding: utf-8 -*-
"""step3 官方管道调度模拟的 numba 复刻内核（语义逐点对齐 schedule_step3）。

核心结构：
  - pipe 顺序 = seq_ext 每管道投影（固定发射序，cursor 推进）
  - allocation_order = seq_ext 中有 managed 输出的 op（严格按 seq 序分配内存）
  - 内存额度 FIFO（VIRGIN/WAR/WAW），消费时记录 (source, target) 复用依赖
  - DDR 公平带宽池：advance_ddr_work / reschedule_ddr_ends（浮点序对齐）
  - PIPE_SLOTS = 1（官方写死）
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _queue_if_ready(op, op_status, pred_rem, op_pipe, pipe_seq_ptr,
                    pipe_seq, pipe_cursor, ready_op, alloc_ready,
                    in_alloc_rank):
    if op_status[op] != 0 or pred_rem[op] != 0:
        return 0
    pipe = op_pipe[op]
    if pipe_cursor[pipe] >= pipe_seq_ptr[pipe + 1] - pipe_seq_ptr[pipe]:
        return 0
    order_head = pipe_seq_ptr[pipe]
    if pipe_seq[order_head + pipe_cursor[pipe]] != op:
        return 0
    op_status[op] = 1
    if in_alloc_rank[op] >= 0:
        alloc_ready[pipe] = op
    else:
        if ready_op[pipe] != -1:
            return -6
        ready_op[pipe] = op
    return 0


@njit(cache=True)
def _dedup_sorted(ptr, ten, op, buf):
    """op 的 tensor 列表去重升序写入 buf，返回个数。"""
    n = 0
    last = -1
    for e in range(ptr[op], ptr[op + 1]):
        t = ten[e]
        if t != last:
            last = t
            buf[n] = t
            n += 1
    return n


@njit(cache=True)
def _allocate_tensor(ti, now, producer_op, ten_size, ten_pos_l1,
                     resident, mem_used, mem_peak,
                     cr_bytes, cr_tensor, cr_kind, cr_head, cr_tail,
                     cr_base, cr_lim,
                     cons_ptr, cons_arr, prod_ptr, prod_arr,
                     dep_src, dep_tgt, dep_cnt, dep_cap):
    """allocate_tensor + consume_free_credit 复刻。返回 0 或负错误。"""
    T = 0 if ten_pos_l1[ti] == 1 else 1
    remaining = ten_size[ti]
    while remaining > 0 and cr_head[T] < cr_tail[T]:
        h = cr_base[T] + cr_head[T]
        taken = remaining if remaining < cr_bytes[h] else cr_bytes[h]
        ct = cr_tensor[h]
        if ct >= 0:
            # sources = consumers if any else producers（均已升序）
            if cons_ptr[ct + 1] - cons_ptr[ct] > 0:
                p0, p1 = cons_ptr[ct], cons_ptr[ct + 1]
            else:
                p0, p1 = prod_ptr[ct], prod_ptr[ct + 1]
            for si in range(p0, p1):
                s = cons_arr[si] if cons_ptr[ct + 1] - cons_ptr[ct] > 0 \
                    else prod_arr[si]
                if s == producer_op:
                    continue
                if dep_cnt[0] >= dep_cap:
                    return -3
                dep_src[dep_cnt[0]] = s
                dep_tgt[dep_cnt[0]] = producer_op
                dep_cnt[0] += 1
        cr_bytes[h] -= taken
        remaining -= taken
        if cr_bytes[h] == 0:
            cr_head[T] += 1
    if remaining > 0:
        return -5
    resident[ti] = True
    mem_used[T] += ten_size[ti]
    if mem_used[T] > mem_peak[T]:
        mem_peak[T] = mem_used[T]
    return 0


@njit(cache=True)
def _release_tensor(ti, ten_size, ten_pos_l1, resident, mem_used,
                    cr_bytes, cr_tensor, cr_kind, cr_head, cr_tail, cr_base,
                    cr_lim,
                    cons_ptr, cons_arr, prod_ptr, prod_arr):
    T = 0 if ten_pos_l1[ti] == 1 else 1
    resident[ti] = False
    mem_used[T] -= ten_size[ti]
    if cr_tail[T] >= cr_lim[T]:
        return -3
    h = cr_base[T] + cr_tail[T]
    cr_bytes[h] = ten_size[ti]
    cr_tensor[h] = ti
    cr_kind[h] = 1  # WAR if consumers else WAW（edges 集合等价）
    cr_tail[T] += 1
    return 0


@njit(cache=True)
def _advance_ddr_work(now, ddr_last_update, pool_ops, pool_work, pool_alive,
                      pool_count, sa, sb):
    """官方 advance_ddr_work（浮点运算序逐点对齐）。"""
    elapsed = now - ddr_last_update[0]
    while elapsed > 1e-9:
        n_active = 0
        for i in range(pool_count[0]):
            if pool_alive[i] and pool_work[i] > 1e-9:
                sa[n_active] = i
                n_active += 1
        if n_active == 0:
            break
        min_work = pool_work[sa[0]]
        for i in range(1, n_active):
            if pool_work[sa[i]] < min_work:
                min_work = pool_work[sa[i]]
        ttf = min_work * n_active
        if ttf >= elapsed - 1e-9:
            share = elapsed / n_active
            for i in range(n_active):
                j = sa[i]
                w = pool_work[j] - share
                pool_work[j] = 0.0 if w < 0.0 else w
            break
        for i in range(n_active):
            j = sa[i]
            w = pool_work[j] - min_work
            pool_work[j] = 0.0 if w < 0.0 else w
        elapsed -= ttf
    ddr_last_update[0] = now


@njit(cache=True)
def _reschedule_ddr(now, pool_ops, pool_work, pool_alive, pool_count,
                    sa, sw, op_end, exec_op, exec_end):
    """官方 reschedule_ddr_ends 的投影部分（stages 仅日志，跳过）。"""
    n = 0
    for i in range(pool_count[0]):
        if pool_alive[i]:
            w = pool_work[i]
            if w < 0.0:
                w = 0.0
            sa[n] = pool_ops[i]
            sw[n] = w
            n += 1
    if n == 0:
        return
    # 按 (work, op) 插入排序（n 很小，≤ 在飞 MTE op 数）
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
    base_cursor = np.float64(now)
    adj_cursor = np.float64(now)
    previous = 0.0
    active = n
    i = 0
    while i < n:
        work = sw[i]
        delta = work - previous
        if active > 1:
            base_delta = delta * (active - 1)
            adj_delta = base_delta * active / (active - 1)
        else:
            base_delta = delta
            adj_delta = delta
        base_cursor += base_delta
        adj_cursor += adj_delta
        j = i
        while j < n and abs(sw[j] - work) <= 1e-9:
            proj = np.ceil(adj_cursor - 1e-9)
            op_end[sa[j]] = np.float64(proj)
            j += 1
        active -= j - i
        previous = work
        i = j
    # executors 端点同步
    for p in range(4):
        if exec_op[p] != -1:
            # projected 不含该 op（已退休）时保留原值 —— 官方 get(op, end)
            for q in range(n):
                if sa[q] == exec_op[p]:
                    exec_end[p] = op_end[exec_op[p]]
                    break


@njit(cache=True)
def step3_sim(n_ops, n_ten, op_pipe, op_is_ddr, op_dur,
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
              out_set_buf, in_set_buf, req_buf, out_mk, dbg):
    """返回 0 正常；负数错误。out_mk: [makespan, peakL1, peakUB, n_dep]。"""
    # ---- 剩余消费者（全体 managed）+ 初始驻留（sorted(managed) = 槽序）----
    for ti in range(n_ten):
        if ten_pos_l1[ti] == 0:
            continue
        remaining_cons[ti] = cons_ptr[ti + 1] - cons_ptr[ti]
        if prod_ptr[ti + 1] - prod_ptr[ti] == 0 and \
                cons_ptr[ti + 1] - cons_ptr[ti] > 0:
            T = 0 if ten_pos_l1[ti] == 1 else 1
            resident[ti] = True
            mem_used[T] += ten_size[ti]
            if mem_used[T] > mem_peak[T]:
                mem_peak[T] = mem_used[T]
    for T in range(2):
        cap = cap_l1 if T == 0 else cap_ub
        unused = cap - mem_used[T]
        if unused > 0:
            if cr_tail[T] >= cr_lim[T]:
                return -3
            h = cr_base[T] + cr_tail[T]
            cr_bytes[h] = unused
            cr_tensor[h] = -1   # VIRGIN（无 sources）
            cr_kind[h] = 0
            cr_tail[T] += 1

    # pred_rem 初值由宿主预填（数据依赖计数），这里不覆盖
    dep_cnt[0] = 0
    pool_dead = 0

    next_alloc_rank = 0
    n_alloc = alloc_order.size
    done_count = 0
    now = 0.0
    for p in range(4):
        pipe_cursor[p] = 0
        ready_op[p] = -1
        alloc_ready[p] = -1
        exec_op[p] = -1
        exec_end[p] = 0.0
    for op in range(n_ops):
        op_status[op] = 0
        op_start[op] = 0.0
        op_end[op] = 0.0
    pool_count[0] = 0
    ddr_last_update[0] = 0.0

    for i in range(n_ops):
        r = _queue_if_ready(seq_ext[i], op_status, pred_rem, op_pipe,
                            pipe_seq_ptr, pipe_seq, pipe_cursor, ready_op,
                            alloc_ready, alloc_rank)
        if r < 0:
            return r

    it = 0
    while True:
        it += 1
        if it > 1000000:
            return -7
        # ---- retire ----
        _advance_ddr_work(now, ddr_last_update, pool_ops, pool_work,
                          pool_alive, pool_count, sa, sw)
        retired_ddr = False
        for p in range(4):
            if exec_op[p] != -1 and exec_end[p] <= now + 1e-9:
                op = exec_op[p]
                exec_op[p] = -1
                op_status[op] = 3
                done_count += 1
                # consume_inputs（官方 set 迭代序）
                for z in range(ord_in_ptr[op], ord_in_ptr[op + 1]):
                    ti = ord_in[z]
                    if ten_pos_l1[ti] == 0:
                        continue
                    if remaining_cons[ti] > 0:
                        remaining_cons[ti] -= 1
                        if remaining_cons[ti] == 0:
                            r = _release_tensor(
                                ti, ten_size, ten_pos_l1, resident, mem_used,
                                cr_bytes, cr_tensor, cr_kind, cr_head,
                                cr_tail, cr_base, cr_lim,
                                cons_ptr, cons_arr, prod_ptr, prod_arr)
                            if r < 0:
                                return r
                # release_dead_outputs（官方 set 迭代序）
                for z in range(ord_out_ptr[op], ord_out_ptr[op + 1]):
                    ti = ord_out[z]
                    if ten_pos_l1[ti] != 0 and remaining_cons[ti] == 0:
                        if resident[ti]:
                            r = _release_tensor(
                                ti, ten_size, ten_pos_l1, resident, mem_used,
                                cr_bytes, cr_tensor, cr_kind, cr_head,
                                cr_tail, cr_base, cr_lim,
                                cons_ptr, cons_arr, prod_ptr, prod_arr)
                            if r < 0:
                                return r
                # pipe cursor 前进 + 唤醒下一个
                pc = pipe_cursor[op_pipe[op]]
                pipe_cursor[op_pipe[op]] = pc + 1
                base = pipe_seq_ptr[op_pipe[op]]
                total = pipe_seq_ptr[op_pipe[op] + 1] - base
                if pc + 1 < total:
                    r = _queue_if_ready(pipe_seq[base + pc + 1], op_status,
                                        pred_rem, op_pipe, pipe_seq_ptr,
                                        pipe_seq, pipe_cursor, ready_op,
                                        alloc_ready, alloc_rank)
                    if r < 0:
                        return r
                # ddr 退休（含池压实）
                if pool_slot[op] != -1:
                    pool_alive[pool_slot[op]] = False
                    pool_slot[op] = -1
                    retired_ddr = True
                    pool_dead += 1
                    if pool_dead > 64:
                        w = 0
                        for q in range(pool_count[0]):
                            if pool_alive[q]:
                                if w != q:
                                    pool_ops[w] = pool_ops[q]
                                    pool_work[w] = pool_work[q]
                                    pool_alive[w] = True
                                    pool_slot[pool_ops[w]] = w
                                w += 1
                        pool_count[0] = w
                        pool_dead = 0
                # 后继前驱计数递减 + 唤醒（官方 retire 的 succ 循环）
                for si in range(succ_ptr[op], succ_ptr[op + 1]):
                    s2 = succ_arr[si]
                    pred_rem[s2] -= 1
                    r2 = _queue_if_ready(s2, op_status, pred_rem, op_pipe,
                                         pipe_seq_ptr, pipe_seq, pipe_cursor,
                                         ready_op, alloc_ready, alloc_rank)
                    if r2 < 0:
                        return r2
        if retired_ddr:
            _reschedule_ddr(now, pool_ops, pool_work, pool_alive,
                            pool_count, sa, sw, op_end, exec_op, exec_end)
        if done_count == n_ops:
            break
        # ---- issue ----
        issued_pass = True
        while issued_pass:
            issued_pass = False
            for p in range(4):
                while exec_op[p] == -1:
                    op = -1
                    if next_alloc_rank < n_alloc:
                        cand = alloc_order[next_alloc_rank]
                        if alloc_ready[p] == cand and op_pipe[cand] == p:
                            # can_allocate_outputs 检查
                            n_out = _dedup_sorted(out_ptr, out_ten, cand,
                                                  out_set_buf)
                            ok = True
                            for z in range(n_out):
                                ti = out_set_buf[z]
                                if ten_pos_l1[ti] == 0 or resident[ti]:
                                    continue
                                T = 0 if ten_pos_l1[ti] == 1 else 1
                                req_buf[T] += ten_size[ti]
                            for T in range(2):
                                cap = cap_l1 if T == 0 else cap_ub
                                if mem_used[T] + req_buf[T] > cap:
                                    ok = False
                                req_buf[T] = 0
                            if ok:
                                alloc_ready[p] = -1
                                op = cand
                    if op == -1 and ready_op[p] != -1:
                        op = ready_op[p]
                        ready_op[p] = -1
                    if op == -1:
                        break
                    issued_pass = True
                    if alloc_rank[op] >= 0:
                        next_alloc_rank += 1
                    op_status[op] = 2
                    # allocate_outputs
                    n_out = _dedup_sorted(out_ptr, out_ten, op, out_set_buf)
                    for z in range(n_out):
                        ti = out_set_buf[z]
                        if ten_pos_l1[ti] == 0 or resident[ti]:
                            continue
                        r = _allocate_tensor(
                            ti, now, op, ten_size, ten_pos_l1, resident,
                            mem_used, mem_peak, cr_bytes, cr_tensor, cr_kind,
                            cr_head, cr_tail, cr_base, cr_lim,
                            cons_ptr, cons_arr, prod_ptr,
                            prod_arr, dep_src, dep_tgt, dep_cnt, dep_cap)
                        if r < 0:
                            dbg[0] = op
                            dbg[1] = ti
                            dbg[2] = ten_size[ti]
                            dbg[3] = ten_pos_l1[ti]
                            dbg[4] = mem_used[0]
                            dbg[5] = mem_used[1]
                            dbg[6] = cr_head[0] * 1000000 + cr_tail[0]
                            dbg[7] = cr_head[1] * 1000000 + cr_tail[1]
                            return r
                    dur = op_dur[op]
                    op_start[op] = now
                    op_end[op] = now + np.float64(dur)
                    exec_op[p] = op
                    exec_end[p] = op_end[op]
                    if op_is_ddr[op]:
                        _advance_ddr_work(now, ddr_last_update, pool_ops,
                                          pool_work, pool_alive, pool_count,
                                          sa, sw)
                        if pool_count[0] >= pool_ops.size:
                            return -3
                        slot = pool_count[0]
                        pool_ops[slot] = op
                        pool_work[slot] = np.float64(dur)
                        pool_alive[slot] = True
                        pool_slot[op] = slot
                        pool_count[0] += 1
                        _reschedule_ddr(now, pool_ops, pool_work, pool_alive,
                                        pool_count, sa, sw, op_end, exec_op,
                                        exec_end)
        # ---- 时间推进 ----
        has = False
        t_next = 0.0
        for p in range(4):
            if exec_op[p] != -1:
                if not has or exec_end[p] < t_next:
                    t_next = exec_end[p]
                    has = True
        if not has:
            out_mk[0] = now
            out_mk[1] = np.float64(done_count)
            out_mk[2] = np.float64(next_alloc_rank)
            out_mk[3] = np.float64(n_alloc)
            for p in range(4):
                out_mk[4 + p] = np.float64(ready_op[p] + 1000000)
                out_mk[8 + p] = np.float64(alloc_ready[p] + 1000000)
            return -2
        if t_next <= now:
            return -8
        now = t_next

    mk = 0.0
    for op in range(n_ops):
        if op_end[op] > mk:
            mk = op_end[op]
    out_mk[0] = mk
    out_mk[1] = np.float64(mem_peak[0])
    out_mk[2] = np.float64(mem_peak[1])
    out_mk[3] = np.float64(dep_cnt[0])
    return 0
