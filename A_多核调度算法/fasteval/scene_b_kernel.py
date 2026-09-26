# -*- coding: utf-8 -*-
"""场景B（P2）与问题3（P3）多核事件循环的 numba 复刻内核。

与场景A的差异：
  - 每核任务恒活跃（无激活机制）
  - 跨核 COPY_IN 经 external release 堆等待源核 COPY_OUT + delay
  - queue_if_ready：cursor 检查在 release 检查之前
  - P3：FIFO Cache（COPY_IN 含 spill-in 查询；命中走 CACHE_READ 池）
"""
import numpy as np
from numba import njit

HEAP_STRIDE = 4194304  # 2^22：gop 上界（release*2^22 + gop 复合键）


@njit(cache=True)
def _heap_push(heap, heap_n, v):
    i = heap_n[0]
    heap[i] = v
    while i > 0:
        p = (i - 1) >> 1
        if heap[p] <= heap[i]:
            break
        t = heap[p]
        heap[p] = heap[i]
        heap[i] = t
        i = p
    heap_n[0] += 1


@njit(cache=True)
def _heap_pop(heap, heap_n):
    top = heap[0]
    n = heap_n[0] - 1
    heap_n[0] = n
    heap[0] = heap[n]
    i = 0
    while True:
        l = 2 * i + 1
        r = l + 1
        m = i
        if l < n and heap[l] < heap[m]:
            m = l
        if r < n and heap[r] < heap[m]:
            m = r
        if m == i:
            break
        t = heap[m]
        heap[m] = heap[i]
        heap[i] = t
        i = m
    return top


@njit(cache=True)
def _queue_ready_b(gop, now, op_status, pred_rem, gop_pipe, gop_core,
                   task_op_base, tsp_ptr, tsp_arr, tsp_base,
                   cursor_k, ready_slot, ext_pred, op_end, delay,
                   rel_sched, heap, heap_n):
    if op_status[gop] != 0 or pred_rem[gop] != 0:
        return 0
    k = gop_core[gop]
    pipe = gop_pipe[gop]
    ck = k * 4 + pipe
    base = tsp_base[k] + tsp_ptr[k * 5 + pipe]
    cnt = tsp_ptr[k * 5 + pipe + 1] - tsp_ptr[k * 5 + pipe]
    cur = cursor_k[ck]
    if cur >= cnt:
        return 0
    if tsp_arr[base + cur] != gop - task_op_base[k]:
        return 0
    # external release（单外部前驱：跨核链接的 COPY_OUT）
    ep = ext_pred[gop]
    if ep != -1:
        if op_status[ep] != 3:
            return 0
        release = int(op_end[ep]) + delay
        if release > now:
            if rel_sched[gop] == 0:
                rel_sched[gop] = 1
                _heap_push(heap, heap_n,
                           np.int64(release) * HEAP_STRIDE + gop)
            return 0
    rel_sched[gop] = 0
    op_status[gop] = 1
    if ready_slot[ck] != -1:
        return -6
    ready_slot[ck] = gop
    return 0


@njit(cache=True)
def _advance_pool(now, last_upd, pool_op, pool_w, pool_alive, pool_n, sa):
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
def _resched_pool(now, pool_op, pool_w, pool_alive, pool_n, sa, sw,
                  op_end, n_cores, exec_op, exec_end):
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
def scene_b_sim(n_cores, n_gops, n_links,
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
                use_cache, cache_idx_of_gop, cache_size_of_key,
                cache_cap, cache_bw,
                cac_pool_op, cac_pool_w, cac_pool_alive, cac_pool_n,
                cac_slot, cac_last_upd,
                in_cache, fifo_q, fifo_sz, fifo_head, fifo_tail, fifo_n,
                cache_used, cache_stats, ret_buf,
                out_mk):
    """返回 0 正常。out_mk[0]=makespan；cache_stats=[hits,misses,hb,mb]。"""
    for g in range(n_gops):
        op_status[g] = 0
        op_end[g] = 0.0
        rel_sched[g] = 0
        ext_pred[g] = -1
        ddr_slot[g] = -1
        cac_slot[g] = -1
    for e in range(n_cores * 4):
        exec_op[e] = -1
        exec_end[e] = 0.0
    for k in range(n_cores * 4):
        cursor_k[k] = 0
        ready_slot[k] = -1
    ddr_pool_n[0] = 0
    ddr_last_upd[0] = 0.0
    cac_pool_n[0] = 0
    cac_last_upd[0] = 0.0
    heap_n[0] = 0
    fifo_head[0] = 0
    fifo_tail[0] = 0
    fifo_n[0] = 0
    cache_used[0] = 0
    for i in range(4):
        cache_stats[i] = 0
    ddr_dead = 0
    cac_dead = 0
    remaining = n_gops
    for e in range(n_links):
        ext_pred[link_dst[e]] = link_src[e]
    now = 0.0

    # 初始 pass
    for k in range(n_cores):
        for e in range(task_seq_ptr[k], task_seq_ptr[k + 1]):
            r = _queue_ready_b(task_seq_arr[e], now, op_status, pred_rem,
                               gop_pipe, gop_core, task_op_base, tsp_ptr,
                               tsp_arr, tsp_base, cursor_k, ready_slot,
                               ext_pred, op_end, delay, rel_sched,
                               heap, heap_n)
            if r < 0:
                return r

    it = 0
    while True:
        it += 1
        if it > 3000000:
            return -7
        # ---- retire ----
        _advance_pool(now, ddr_last_upd, ddr_pool_op, ddr_pool_w,
                      ddr_pool_alive, ddr_pool_n, sa)
        if use_cache:
            _advance_pool(now, cac_last_upd, cac_pool_op, cac_pool_w,
                          cac_pool_alive, cac_pool_n, sa)
        retired_ddr = False
        retired_cac = False
        retired_items_n = 0
        for e in range(n_cores * 4):
            if exec_op[e] != -1 and exec_end[e] <= now:
                g = exec_op[e]
                exec_op[e] = -1
                op_status[g] = 3
                remaining -= 1
                ret_buf[retired_items_n] = g
                retired_items_n += 1
                if ddr_slot[g] != -1:
                    ddr_pool_alive[ddr_slot[g]] = False
                    ddr_slot[g] = -1
                    retired_ddr = True
                    ddr_dead += 1
                    if ddr_dead > 64:
                        _compact(ddr_pool_op, ddr_pool_w, ddr_pool_alive,
                                 ddr_pool_n, ddr_slot)
                        ddr_dead = 0
                if cac_slot[g] != -1:
                    cac_pool_alive[cac_slot[g]] = False
                    cac_slot[g] = -1
                    retired_cac = True
                    cac_dead += 1
                    if cac_dead > 64:
                        _compact(cac_pool_op, cac_pool_w, cac_pool_alive,
                                 cac_pool_n, cac_slot)
                        cac_dead = 0
                # cache insert（COPY_IN 完成后写缓存）
                if use_cache:
                    ci = cache_idx_of_gop[g]
                    if ci >= 0:
                        sz = cache_size_of_key[ci]
                        _insert_cache(ci, sz, cache_cap, in_cache, fifo_q,
                                      fifo_sz, fifo_head, fifo_tail, fifo_n,
                                      cache_used)
                # pipe cursor 前进 + 唤醒
                k = gop_core[g]
                pipe = gop_pipe[g]
                ck = k * 4 + pipe
                cur = cursor_k[ck]
                cursor_k[ck] = cur + 1
                base = tsp_base[k] + tsp_ptr[k * 5 + pipe]
                cnt = tsp_ptr[k * 5 + pipe + 1] - tsp_ptr[k * 5 + pipe]
                if cur + 1 < cnt:
                    nxt = task_op_base[k] + tsp_arr[base + cur + 1]
                    r = _queue_ready_b(nxt, now, op_status, pred_rem,
                                       gop_pipe, gop_core, task_op_base,
                                       tsp_ptr, tsp_arr, tsp_base, cursor_k,
                                       ready_slot, ext_pred, op_end, delay,
                                       rel_sched, heap, heap_n)
                    if r < 0:
                        return r
                for e2 in range(succ_ptr[g], succ_ptr[g + 1]):
                    s2 = succ_arr[e2]
                    pred_rem[s2] -= 1
                    r = _queue_ready_b(s2, now, op_status, pred_rem,
                                       gop_pipe, gop_core, task_op_base,
                                       tsp_ptr, tsp_arr, tsp_base, cursor_k,
                                       ready_slot, ext_pred, op_end, delay,
                                       rel_sched, heap, heap_n)
                    if r < 0:
                        return r
        if retired_ddr:
            _resched_pool(now, ddr_pool_op, ddr_pool_w, ddr_pool_alive,
                          ddr_pool_n, sa, sw, op_end, n_cores,
                          exec_op, exec_end)
        if retired_cac:
            _resched_pool(now, cac_pool_op, cac_pool_w, cac_pool_alive,
                          cac_pool_n, sa, sw, op_end, n_cores,
                          exec_op, exec_end)
        # 外部释放堆到期处理
        while heap_n[0] > 0 and heap[0] // HEAP_STRIDE <= now:
            v = _heap_pop(heap, heap_n)
            g = int(v % HEAP_STRIDE)
            rel_sched[g] = 0
            r = _queue_ready_b(g, now, op_status, pred_rem, gop_pipe,
                               gop_core, task_op_base, tsp_ptr, tsp_arr,
                               tsp_base, cursor_k, ready_slot, ext_pred,
                               op_end, delay, rel_sched, heap, heap_n)
            if r < 0:
                return r
        # retired_items 的外部后继唤醒（须在 resched 后）
        for z in range(retired_items_n):
            g = ret_buf[z]
            for e2 in range(ext_succ_ptr[g], ext_succ_ptr[g + 1]):
                t2 = ext_succ_arr[e2]
                r = _queue_ready_b(t2, now, op_status, pred_rem, gop_pipe,
                                   gop_core, task_op_base, tsp_ptr, tsp_arr,
                                   tsp_base, cursor_k, ready_slot, ext_pred,
                                   op_end, delay, rel_sched, heap, heap_n)
                if r < 0:
                    return r
        if remaining == 0:
            break
        # ---- issue ----
        issued_pass = True
        while issued_pass:
            issued_pass = False
            for c in range(n_cores):
                for p in range(4):
                    e = c * 4 + p
                    while exec_op[e] == -1:
                        g = ready_slot[c * 4 + p]
                        if g == -1:
                            break
                        ready_slot[c * 4 + p] = -1
                        issued_pass = True
                        op_status[g] = 2
                        dur = gop_dur[g]
                        if use_cache:
                            ci = cache_idx_of_gop[g]
                            if ci >= 0:
                                sz = cache_size_of_key[ci]
                                hit = in_cache[ci] == 1
                                if hit:
                                    dur = max(1, -(-sz // cache_bw))
                                    cache_stats[0] += 1
                                    cache_stats[2] += sz
                                else:
                                    cache_stats[1] += 1
                                    cache_stats[3] += sz
                        op_end[g] = now + np.float64(dur)
                        exec_op[e] = g
                        exec_end[e] = op_end[g]
                        ci_g = cache_idx_of_gop[g] if use_cache else -1
                        if ci_g >= 0 and in_cache[ci_g] == 1:
                            _advance_pool(now, cac_last_upd, cac_pool_op,
                                          cac_pool_w, cac_pool_alive,
                                          cac_pool_n, sa)
                            if cac_pool_n[0] >= cac_pool_op.size:
                                return -3
                            s_ = cac_pool_n[0]
                            cac_pool_op[s_] = g
                            cac_pool_w[s_] = np.float64(dur)
                            cac_pool_alive[s_] = True
                            cac_slot[g] = s_
                            cac_pool_n[0] += 1
                            _resched_pool(now, cac_pool_op, cac_pool_w,
                                          cac_pool_alive, cac_pool_n,
                                          sa, sw, op_end, n_cores,
                                          exec_op, exec_end)
                        elif gop_is_ddr[g]:
                            _advance_pool(now, ddr_last_upd, ddr_pool_op,
                                          ddr_pool_w, ddr_pool_alive,
                                          ddr_pool_n, sa)
                            if ddr_pool_n[0] >= ddr_pool_op.size:
                                return -3
                            s_ = ddr_pool_n[0]
                            ddr_pool_op[s_] = g
                            ddr_pool_w[s_] = np.float64(dur)
                            ddr_pool_alive[s_] = True
                            ddr_slot[g] = s_
                            ddr_pool_n[0] += 1
                            _resched_pool(now, ddr_pool_op, ddr_pool_w,
                                          ddr_pool_alive, ddr_pool_n,
                                          sa, sw, op_end, n_cores,
                                          exec_op, exec_end)
        # ---- 时间推进 ----
        has = False
        t_next = 0.0
        for e in range(n_cores * 4):
            if exec_op[e] != -1:
                if not has or exec_end[e] < t_next:
                    t_next = exec_end[e]
                    has = True
        if heap_n[0] > 0:
            rt = np.float64(heap[0] // HEAP_STRIDE)
            if not has or rt < t_next:
                t_next = rt
                has = True
        if not has:
            out_mk[0] = now
            return -2
            return -2
        if t_next <= now:
            out_mk[0] = now
            return -8
        now = t_next

    mk = 0.0
    for g in range(n_gops):
        if op_end[g] > mk:
            mk = op_end[g]
    out_mk[0] = mk
    return 0


@njit(cache=True)
def _compact(pool_op, pool_w, pool_alive, pool_n, pool_slot):
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


@njit(cache=True)
def _insert_cache(ci, sz, cache_cap, in_cache, fifo_q, fifo_sz,
                  fifo_head, fifo_tail, fifo_n, cache_used):
    if sz > cache_cap:
        return
    if in_cache[ci] == 1:
        return
    while fifo_n[0] > 0 and cache_used[0] + sz > cache_cap:
        old = fifo_q[fifo_head[0]]
        fifo_head[0] += 1
        fifo_n[0] -= 1
        in_cache[old] = 0
        cache_used[0] -= fifo_sz[old]
    fifo_q[fifo_tail[0]] = ci
    fifo_sz[ci] = sz
    fifo_tail[0] += 1
    fifo_n[0] += 1
    in_cache[ci] = 1
    cache_used[0] += sz
