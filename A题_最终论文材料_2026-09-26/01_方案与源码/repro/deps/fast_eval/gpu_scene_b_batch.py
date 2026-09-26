# -*- coding: utf-8 -*-
"""GPU 场景B批量内核：一线程一 plan（纯基址+偏移，[N,1] 发射）。

由 gpu_scene_b.scene_b_kernel（单 plan [1,1] 发射版）机械变换而来：
  - 每个 plan 的输入位于全 plan 拼接数组中该 plan 的偏移区段
    （gop_off/ptrN_off/core_off/core1_off/core5_off/ts_off/succ_off/
    link_off，与 pack_candidates.save_npz / run_batch 的 npz 布局一致）；
  - 每个 plan 的 scratch（模拟状态）位于按 plan 索引 × 统一步长的独立
    分区，内核内不分配、不共享、不依赖线程执行顺序；
  - plo（plan_lo）：线程 pid 处理输入 plan plo+pid，scratch 槽位为 pid
    （供宿主分块/分片复用同一块 scratch）。

失败码编号与 CPU scene_b_kernel.scene_b_sim 统一（旧单 plan GPU 内核
编号错位，语义相同）：
  无事件死锁：GPU 旧 -1 → 本内核 -2（= CPU -2）
  时间不推进：GPU 旧 -2 → 本内核 -8（= CPU -8）
  迭代上限：  GPU 旧 -3 → 本内核 -7（= CPU -7）
除编号外语义逐位不变。
"""
import numpy as np
from numba import cuda

HS = 4194304  # 2^22 heap stride


@cuda.jit(cache=True)
def scene_b_batch_kernel(n_plans, plo, plan_meta,
                         gop_off, ptrN_off, core_off, core1_off, core5_off,
                         ts_off, succ_off, link_off,
                         # 打包数据（全 plan 拼接，按 plan 偏移访问）
                         gop_pipe, gop_dur, gop_is_ddr, gop_core,
                         task_seq_ptr, task_seq_arr,
                         pipe_ptr, pipe_seq_arr, pipe_ptr_base, pipe_seq_base,
                         task_op_base,
                         succ_ptr, succ_arr, pred_cnt,
                         ext_pred_arr, ext_succ_ptr, ext_succ_arr,
                         # scratch（按 plan 分区，槽位 = pid）
                         st_status, st_end, st_pred,
                         st_pcur, st_rdy,
                         st_eop, st_eend,
                         st_po, st_pw, st_pa, st_pn, st_pslot, st_lu,
                         st_sa, st_sw,
                         rel_sched, heap, heap_n, ret_buf, out_mk,
                         # 分区步长
                         s_ng, s_c4, s_pool, s_sa, s_heap, s_ret):
    """一线程一 plan 的场景B模拟。out_mk[pid] = makespan / 负失败码。"""
    pid = cuda.grid(1)
    if pid >= n_plans:
        return
    pl = pid + plo
    # ---- 输入基址（plan 级偏移）----
    gb = gop_off[pl]        # gop 字段（pipe/dur/is_ddr/core/pred_cnt/
                            # ext_pred/pipe_seq_arr）
    pnb = ptrN_off[pl]      # n_gops+1 字段（succ_ptr/ext_succ_ptr）
    cb = core_off[pl]       # n_cores 字段（pipe_ptr_base/pipe_seq_base）
    c1b = core1_off[pl]     # n_cores+1 字段（task_seq_ptr/task_op_base）
    c5b = core5_off[pl]     # n_cores*5 字段（pipe_ptr）
    tsb = ts_off[pl]        # task_seq_arr
    scb = succ_off[pl]      # succ_arr
    lkb = link_off[pl]      # ext_succ_arr
    n_cores = int(core_off[pl + 1] - cb)
    n_gops = int(gop_off[pl + 1] - gb)
    delay_v = int(plan_meta[pl, 3])
    # ---- scratch 基址（plan 分区）----
    b_stat = pid * s_ng
    b_end = pid * s_ng
    b_pred = pid * s_ng
    b_slot = pid * s_ng
    b_rel = pid * s_ng
    b_c4 = pid * s_c4
    b_pool = pid * s_pool
    b_sw = pid * s_pool
    b_sa = pid * s_sa
    b_heap = pid * s_heap
    b_ret = pid * s_ret

    # ---- 初始化 ----
    for g in range(n_gops):
        st_status[b_stat + g] = 0
        st_end[b_end + g] = 0.0
        st_pred[b_pred + g] = pred_cnt[gb + g]
        st_pslot[b_slot + g] = -1
        rel_sched[b_rel + g] = 0
    for e in range(n_cores * 4):
        st_eop[b_c4 + e] = -1
        st_eend[b_c4 + e] = 0.0
    for k in range(n_cores * 4):
        st_pcur[b_c4 + k] = 0
        st_rdy[b_c4 + k] = -1
    st_pn[pid] = 0
    st_lu[pid] = 0.0
    heap_n[pid] = 0
    remaining = n_gops

    # ---- 初始 pass（所有核全活跃）----
    for c in range(n_cores):
        for e in range(task_seq_ptr[c1b + c], task_seq_ptr[c1b + c + 1]):
            g = task_seq_arr[tsb + e]
            # 内联 queue_if_ready
            if st_status[b_stat + g] != 0 or st_pred[b_pred + g] != 0:
                continue
            p = gop_pipe[gb + g]
            ck = c * 4 + p  # core*4+pipe（场景B task=core）
            cur = st_pcur[b_c4 + ck]
            pb = pipe_ptr_base[cb + c]
            cnt = pipe_ptr[c5b + pb + p + 1] - pipe_ptr[c5b + pb + p]
            if cur >= cnt:
                continue
            if pipe_seq_arr[gb + pipe_seq_base[cb + c] +
                            pipe_ptr[c5b + pb + p] + cur] \
                    != g - task_op_base[c1b + c]:
                continue
            # 外部释放检查
            ep = ext_pred_arr[gb + g]
            if ep != -1:
                if st_status[b_stat + ep] != 3:
                    continue
                release = int(st_end[b_end + ep]) + delay_v
                if release > 0:  # now=0
                    if rel_sched[b_rel + g] == 0:
                        rel_sched[b_rel + g] = 1
                        # push to heap（内联）
                        hv = np.int64(release) * HS + g
                        idx = heap_n[pid]
                        heap[b_heap + idx] = hv
                        heap_n[pid] += 1
                        while idx > 0:
                            par = (idx - 1) >> 1
                            if heap[b_heap + par] <= heap[b_heap + idx]:
                                break
                            tmp = heap[b_heap + par]
                            heap[b_heap + par] = heap[b_heap + idx]
                            heap[b_heap + idx] = tmp
                            idx = par
                    continue
            rel_sched[b_rel + g] = 0
            st_status[b_stat + g] = 1
            st_rdy[b_c4 + ck] = g

    now = 0.0
    it = 0
    while True:
        it += 1
        if it > 3000000:
            out_mk[pid] = -7.0
            return

        # ---- advance_ddr ----
        elapsed = now - st_lu[pid]
        while elapsed > 1e-9:
            na = 0
            for q in range(st_pn[pid]):
                if st_pa[b_pool + q] and st_pw[b_pool + q] > 1e-9:
                    st_sa[b_sa + na] = q
                    na += 1
            if na == 0:
                break
            mw = st_pw[b_pool + st_sa[b_sa + 0]]
            for q in range(1, na):
                if st_pw[b_pool + st_sa[b_sa + q]] < mw:
                    mw = st_pw[b_pool + st_sa[b_sa + q]]
            ttf = mw * na
            if ttf >= elapsed - 1e-9:
                sh = elapsed / na
                for q in range(na):
                    st_pw[b_pool + st_sa[b_sa + q]] -= sh
                    if st_pw[b_pool + st_sa[b_sa + q]] < 0.0:
                        st_pw[b_pool + st_sa[b_sa + q]] = 0.0
                break
            for q in range(na):
                st_pw[b_pool + st_sa[b_sa + q]] -= mw
                if st_pw[b_pool + st_sa[b_sa + q]] < 0.0:
                    st_pw[b_pool + st_sa[b_sa + q]] = 0.0
            elapsed -= ttf
        st_lu[pid] = now

        # ---- retire ----
        ret_ddr = False
        ret_n = 0
        for c in range(n_cores):
            for p in range(4):
                e = c * 4 + p
                if st_eop[b_c4 + e] != -1 and st_eend[b_c4 + e] <= now + 1e-9:
                    g = st_eop[b_c4 + e]
                    st_eop[b_c4 + e] = -1
                    st_status[b_stat + g] = 3
                    remaining -= 1
                    ret_buf[b_ret + ret_n] = g
                    ret_n += 1
                    # pipe cursor
                    pipe = gop_pipe[gb + g]
                    ck = c * 4 + pipe
                    cur = st_pcur[b_c4 + ck]
                    st_pcur[b_c4 + ck] = cur + 1
                    pb = pipe_ptr_base[cb + c]
                    cnt = pipe_ptr[c5b + pb + pipe + 1] - \
                        pipe_ptr[c5b + pb + pipe]
                    if cur + 1 < cnt:
                        nxt = task_op_base[c1b + c] + \
                            pipe_seq_arr[gb + pipe_seq_base[cb + c] +
                                         pipe_ptr[c5b + pb + pipe] + cur + 1]
                        # 内联 queue_if_ready
                        if st_status[b_stat + nxt] == 0 and \
                                st_pred[b_pred + nxt] == 0:
                            np2 = gop_pipe[gb + nxt]
                            nck = c * 4 + np2
                            ncur = st_pcur[b_c4 + nck]
                            npb = pipe_ptr_base[cb + c]
                            ncnt = pipe_ptr[c5b + npb + np2 + 1] - \
                                pipe_ptr[c5b + npb + np2]
                            if ncur < ncnt:
                                nseq = pipe_seq_arr[
                                    gb + pipe_seq_base[cb + c] +
                                    pipe_ptr[c5b + npb + np2] + ncur]
                                if nseq == nxt - task_op_base[c1b + c]:
                                    ep2 = ext_pred_arr[gb + nxt]
                                    if ep2 != -1:
                                        if st_status[b_stat + ep2] != 3:
                                            pass  # 等外部
                                        else:
                                            rel2 = int(st_end[b_end + ep2]) \
                                                + delay_v
                                            if rel2 <= now:
                                                st_status[b_stat + nxt] = 1
                                                st_rdy[b_c4 + nck] = nxt
                                            elif rel_sched[b_rel + nxt] == 0:
                                                rel_sched[b_rel + nxt] = 1
                                                hv2 = np.int64(rel2) * HS \
                                                    + nxt
                                                idx2 = heap_n[pid]
                                                heap[b_heap + idx2] = hv2
                                                heap_n[pid] += 1
                                                while idx2 > 0:
                                                    par2 = (idx2 - 1) >> 1
                                                    if heap[b_heap + par2] \
                                                            <= heap[
                                                                b_heap + idx2]:
                                                        break
                                                    tmp2 = heap[b_heap + par2]
                                                    heap[b_heap + par2] = \
                                                        heap[b_heap + idx2]
                                                    heap[b_heap + idx2] = tmp2
                                                    idx2 = par2
                                    else:
                                        st_status[b_stat + nxt] = 1
                                        st_rdy[b_c4 + nck] = nxt
                    # ddr 退休
                    if st_pslot[b_slot + g] != -1:
                        st_pa[b_pool + st_pslot[b_slot + g]] = False
                        st_pslot[b_slot + g] = -1
                        ret_ddr = True
                    # succ 唤醒
                    for e2 in range(succ_ptr[pnb + g], succ_ptr[pnb + g + 1]):
                        s2 = succ_arr[scb + e2]
                        st_pred[b_pred + s2] -= 1
                        if st_status[b_stat + s2] == 0 and \
                                st_pred[b_pred + s2] == 0:
                            sp = gop_pipe[gb + s2]
                            sc = gop_core[gb + s2]
                            sck = sc * 4 + sp
                            scur = st_pcur[b_c4 + sck]
                            spb = pipe_ptr_base[cb + sc]
                            scnt = pipe_ptr[c5b + spb + sp + 1] - \
                                pipe_ptr[c5b + spb + sp]
                            if scur < scnt:
                                sseq = pipe_seq_arr[
                                    gb + pipe_seq_base[cb + sc] +
                                    pipe_ptr[c5b + spb + sp] + scur]
                                if sseq == s2 - task_op_base[c1b + sc]:
                                    ep3 = ext_pred_arr[gb + s2]
                                    if ep3 != -1:
                                        if st_status[b_stat + ep3] == 3:
                                            rel3 = int(st_end[b_end + ep3]) \
                                                + delay_v
                                            if rel3 <= now:
                                                st_status[b_stat + s2] = 1
                                                st_rdy[b_c4 + sck] = s2
                                            elif rel_sched[b_rel + s2] == 0:
                                                rel_sched[b_rel + s2] = 1
                                                hv3 = np.int64(rel3) * HS \
                                                    + s2
                                                idx3 = heap_n[pid]
                                                heap[b_heap + idx3] = hv3
                                                heap_n[pid] += 1
                                                while idx3 > 0:
                                                    par3 = (idx3 - 1) >> 1
                                                    if heap[b_heap + par3] \
                                                            <= heap[
                                                                b_heap + idx3]:
                                                        break
                                                    tmp3 = heap[b_heap + par3]
                                                    heap[b_heap + par3] = \
                                                        heap[b_heap + idx3]
                                                    heap[b_heap + idx3] = tmp3
                                                    idx3 = par3
                                    else:
                                        st_status[b_stat + s2] = 1
                                        st_rdy[b_c4 + sck] = s2

        # reschedule_ddr（场景A版：简单投影）
        if ret_ddr:
            nn = 0
            for q in range(st_pn[pid]):
                if st_pa[b_pool + q]:
                    w2 = st_pw[b_pool + q]
                    if w2 < 0.0:
                        w2 = 0.0
                    st_sw[b_sw + nn] = w2
                    st_sa[b_sa + nn] = st_po[b_pool + q]  # 存 op（非池下标）
                    nn += 1
            if nn > 0:
                # 按 (work, op) 插入排序
                for qi in range(1, nn):
                    ka = st_sa[b_sa + qi]
                    kw = st_sw[b_sw + qi]
                    qj = qi - 1
                    while qj >= 0 and (st_sw[b_sw + qj] > kw or
                                       (st_sw[b_sw + qj] == kw and
                                        st_sa[b_sa + qj] > ka)):
                        st_sw[b_sw + qj + 1] = st_sw[b_sw + qj]
                        st_sa[b_sa + qj + 1] = st_sa[b_sa + qj]
                        qj -= 1
                    st_sw[b_sw + qj + 1] = kw
                    st_sa[b_sa + qj + 1] = ka
                cursor = np.float64(now)
                pv = 0.0
                act = nn
                qi = 0
                while qi < nn:
                    wk = st_sw[b_sw + qi]
                    cursor += (wk - pv) * act
                    qj = qi
                    while qj < nn and abs(st_sw[b_sw + qj] - wk) <= 1e-9:
                        val = cursor - 1e-9
                        iv = int(val)
                        if val > iv:
                            iv += 1
                        st_end[b_end + st_sa[b_sa + qj]] = np.float64(iv)
                        qj += 1
                    act -= qj - qi
                    pv = wk
                    qi = qj
                for c in range(n_cores):
                    for p in range(4):
                        e3 = c * 4 + p
                        if st_eop[b_c4 + e3] != -1:
                            for q in range(nn):
                                if st_sa[b_sa + q] == st_eop[b_c4 + e3]:
                                    st_eend[b_c4 + e3] = \
                                        st_end[b_end + st_eop[b_c4 + e3]]
                                    break

        # ---- 释放堆到期处理 ----
        while heap_n[pid] > 0 and heap[b_heap + 0] // HS <= now:
            # pop min
            top = heap[b_heap + 0]
            hn = heap_n[pid] - 1
            heap_n[pid] = hn
            heap[b_heap + 0] = heap[b_heap + hn]
            idx = 0
            while True:
                l = 2 * idx + 1
                r = l + 1
                m = idx
                if l < hn and heap[b_heap + l] < heap[b_heap + m]:
                    m = l
                if r < hn and heap[b_heap + r] < heap[b_heap + m]:
                    m = r
                if m == idx:
                    break
                tmp = heap[b_heap + m]
                heap[b_heap + m] = heap[b_heap + idx]
                heap[b_heap + idx] = tmp
                idx = m
            g = int(top % HS)
            rel_sched[b_rel + g] = 0
            # 重新 queue_if_ready
            if st_status[b_stat + g] == 0 and st_pred[b_pred + g] == 0:
                p4 = gop_pipe[gb + g]
                c4 = gop_core[gb + g]
                ck4 = c4 * 4 + p4
                cur4 = st_pcur[b_c4 + ck4]
                pb4 = pipe_ptr_base[cb + c4]
                cnt4 = pipe_ptr[c5b + pb4 + p4 + 1] - \
                    pipe_ptr[c5b + pb4 + p4]
                if cur4 < cnt4:
                    sq4 = pipe_seq_arr[gb + pipe_seq_base[cb + c4] +
                                       pipe_ptr[c5b + pb4 + p4] + cur4]
                    if sq4 == g - task_op_base[c1b + c4]:
                        st_status[b_stat + g] = 1
                        st_rdy[b_c4 + ck4] = g

        # ---- retired 的外部后继唤醒 ----
        for z in range(ret_n):
            gz = ret_buf[b_ret + z]
            for e6 in range(ext_succ_ptr[pnb + gz],
                            ext_succ_ptr[pnb + gz + 1]):
                t2 = ext_succ_arr[lkb + e6]
                if st_status[b_stat + t2] == 0 and st_pred[b_pred + t2] == 0:
                    tp = gop_pipe[gb + t2]
                    tc = gop_core[gb + t2]
                    tck = tc * 4 + tp
                    tcur = st_pcur[b_c4 + tck]
                    tpb_ = pipe_ptr_base[cb + tc]
                    tcnt = pipe_ptr[c5b + tpb_ + tp + 1] - \
                        pipe_ptr[c5b + tpb_ + tp]
                    if tcur < tcnt:
                        tseq = pipe_seq_arr[gb + pipe_seq_base[cb + tc] +
                                            pipe_ptr[c5b + tpb_ + tp] + tcur]
                        if tseq == t2 - task_op_base[c1b + tc]:
                            tep = ext_pred_arr[gb + t2]
                            if tep != -1:
                                if st_status[b_stat + tep] == 3:
                                    trel = int(st_end[b_end + tep]) + delay_v
                                    if trel <= now:
                                        st_status[b_stat + t2] = 1
                                        st_rdy[b_c4 + tck] = t2
                                    elif rel_sched[b_rel + t2] == 0:
                                        rel_sched[b_rel + t2] = 1
                                        thv = np.int64(trel) * HS + t2
                                        tidx = heap_n[pid]
                                        heap[b_heap + tidx] = thv
                                        heap_n[pid] += 1
                                        while tidx > 0:
                                            tpar = (tidx - 1) >> 1
                                            if heap[b_heap + tpar] <= \
                                                    heap[b_heap + tidx]:
                                                break
                                            ttmp = heap[b_heap + tpar]
                                            heap[b_heap + tpar] = \
                                                heap[b_heap + tidx]
                                            heap[b_heap + tidx] = ttmp
                                            tidx = tpar
                            else:
                                st_status[b_stat + t2] = 1
                                st_rdy[b_c4 + tck] = t2

        if remaining == 0:
            break

        # ---- issue ----
        ipass = True
        while ipass:
            ipass = False
            for c in range(n_cores):
                for p in range(4):
                    e = c * 4 + p
                    while st_eop[b_c4 + e] == -1:
                        g = st_rdy[b_c4 + c * 4 + p]
                        if g == -1:
                            break
                        st_rdy[b_c4 + c * 4 + p] = -1
                        ipass = True
                        st_status[b_stat + g] = 2
                        dur = gop_dur[gb + g]
                        st_end[b_end + g] = now + np.float64(dur)
                        st_eop[b_c4 + e] = g
                        st_eend[b_c4 + e] = st_end[b_end + g]
                        if gop_is_ddr[gb + g]:
                            # advance_ddr（内联）
                            el2 = now - st_lu[pid]
                            while el2 > 1e-9:
                                na2 = 0
                                for q in range(st_pn[pid]):
                                    if st_pa[b_pool + q] and \
                                            st_pw[b_pool + q] > 1e-9:
                                        st_sa[b_sa + na2] = q
                                        na2 += 1
                                if na2 == 0:
                                    break
                                mw2 = st_pw[b_pool + st_sa[b_sa + 0]]
                                for q in range(1, na2):
                                    if st_pw[b_pool + st_sa[b_sa + q]] < mw2:
                                        mw2 = st_pw[b_pool + st_sa[b_sa + q]]
                                ttf2 = mw2 * na2
                                if ttf2 >= el2 - 1e-9:
                                    sh2 = el2 / na2
                                    for q in range(na2):
                                        st_pw[b_pool + st_sa[b_sa + q]] -= sh2
                                        if st_pw[b_pool + st_sa[b_sa + q]] \
                                                < 0.0:
                                            st_pw[b_pool + st_sa[b_sa + q]] \
                                                = 0.0
                                    break
                                for q in range(na2):
                                    st_pw[b_pool + st_sa[b_sa + q]] -= mw2
                                    if st_pw[b_pool + st_sa[b_sa + q]] < 0.0:
                                        st_pw[b_pool + st_sa[b_sa + q]] = 0.0
                                el2 -= ttf2
                            st_lu[pid] = now
                            s_ = st_pn[pid]
                            st_po[b_pool + s_] = g
                            st_pw[b_pool + s_] = np.float64(dur)
                            st_pa[b_pool + s_] = True
                            st_pslot[b_slot + g] = s_
                            st_pn[pid] += 1
                            # reschedule（内联）
                            nn2 = 0
                            for q in range(st_pn[pid]):
                                if st_pa[b_pool + q]:
                                    w4 = st_pw[b_pool + q]
                                    if w4 < 0.0:
                                        w4 = 0.0
                                    st_sw[b_sw + nn2] = w4
                                    # 存 op（非池下标）
                                    st_sa[b_sa + nn2] = st_po[b_pool + q]
                                    nn2 += 1
                            if nn2 > 0:
                                for qi in range(1, nn2):
                                    ka = st_sa[b_sa + qi]
                                    kw = st_sw[b_sw + qi]
                                    qj = qi - 1
                                    while qj >= 0 and (
                                            st_sw[b_sw + qj] > kw or
                                            (st_sw[b_sw + qj] == kw and
                                             st_sa[b_sa + qj] > ka)):
                                        st_sw[b_sw + qj + 1] = st_sw[b_sw + qj]
                                        st_sa[b_sa + qj + 1] = st_sa[b_sa + qj]
                                        qj -= 1
                                    st_sw[b_sw + qj + 1] = kw
                                    st_sa[b_sa + qj + 1] = ka
                                cur5 = np.float64(now)
                                pv5 = 0.0
                                act5 = nn2
                                qi = 0
                                while qi < nn2:
                                    wk5 = st_sw[b_sw + qi]
                                    cur5 += (wk5 - pv5) * act5
                                    qj = qi
                                    while qj < nn2 and abs(
                                            st_sw[b_sw + qj] - wk5) <= 1e-9:
                                        val5 = cur5 - 1e-9
                                        iv5 = int(val5)
                                        if val5 > iv5:
                                            iv5 += 1
                                        st_end[b_end + st_sa[b_sa + qj]] = \
                                            np.float64(iv5)
                                        qj += 1
                                    act5 -= qj - qi
                                    pv5 = wk5
                                    qi = qj
                                for c5 in range(n_cores):
                                    for p5 in range(4):
                                        e5 = c5 * 4 + p5
                                        if st_eop[b_c4 + e5] != -1:
                                            for q in range(nn2):
                                                if st_sa[b_sa + q] == \
                                                        st_eop[b_c4 + e5]:
                                                    st_eend[b_c4 + e5] = \
                                                        st_end[b_end +
                                                               st_eop[
                                                                   b_c4 + e5]]
                                                    break

        # ---- 时间推进 ----
        has = False
        tn = 0.0
        for e in range(n_cores * 4):
            if st_eop[b_c4 + e] != -1:
                if not has or st_eend[b_c4 + e] < tn:
                    tn = st_eend[b_c4 + e]
                    has = True
        if heap_n[pid] > 0:
            rt = np.float64(heap[b_heap + 0] // HS)
            if not has or rt < tn:
                tn = rt
                has = True
        if not has:
            out_mk[pid] = -2.0
            return
        if tn <= now:
            out_mk[pid] = -8.0
            return
        now = tn

    mk = 0.0
    for g in range(n_gops):
        if st_end[b_end + g] > mk:
            mk = st_end[b_end + g]
    out_mk[pid] = mk
