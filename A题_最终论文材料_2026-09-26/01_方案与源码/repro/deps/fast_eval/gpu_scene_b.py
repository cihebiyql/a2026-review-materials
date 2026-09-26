# -*- coding: utf-8 -*-
"""GPU 场景B：P2/P3 多核事件循环 CUDA 内核（纯索引，单函数）。

与 scene_b_kernel.scene_b_sim 逐位对齐。
差异（vs 场景A）：所有核从 t=0 活跃；跨核 COPY_IN 经释放堆等待源 COPY_OUT + delay。
"""
import numpy as np
from numba import cuda

HS = 4194304  # 2^22 heap stride


@cuda.jit
def scene_b_kernel(n_cores, n_gops, n_links,
                   # 打包数据
                   gop_pipe, gop_dur, gop_is_ddr, gop_core,
                   task_seq_ptr, task_seq_arr,
                   pipe_ptr, pipe_seq_arr, pipe_ptr_base, pipe_seq_base,
                   task_op_base,
                   succ_ptr, succ_arr, pred_cnt,
                   link_src, link_dst,
                   ext_pred_arr,
                   ext_succ_ptr, ext_succ_arr,
                   delay_v,
                   # scratch
                   st_status, st_end, st_pred,
                   st_pcur, st_rdy,
                   st_eop, st_eend,
                   st_po, st_pw, st_pa, st_pn, st_pslot, st_lu,
                   st_sa, st_sw,
                   rel_sched, heap, heap_n, ret_buf,
                   out_mk):
    """单 plan 场景B模拟。out_mk[0] = makespan。"""
    # ---- 初始化 ----
    for g in range(n_gops):
        st_status[g] = 0
        st_end[g] = 0.0
        st_pred[g] = pred_cnt[g]
        st_pslot[g] = -1
        rel_sched[g] = 0
    for e in range(n_cores * 4):
        st_eop[e] = -1
        st_eend[e] = 0.0
    for k in range(n_cores * 4):
        st_pcur[k] = 0
        st_rdy[k] = -1
    st_pn[0] = 0
    st_lu[0] = 0.0
    heap_n[0] = 0
    remaining = n_gops

    # ---- 初始 pass（所有核全活跃）----
    for c in range(n_cores):
        for e in range(task_seq_ptr[c], task_seq_ptr[c + 1]):
            g = task_seq_arr[e]
            # 内联 queue_if_ready
            if st_status[g] != 0 or st_pred[g] != 0:
                continue
            p = gop_pipe[g]
            ck = c * 4 + p  # core*4+pipe（场景B task=core）
            cur = st_pcur[ck]
            pb = pipe_ptr_base[c]
            cnt = pipe_ptr[pb + p + 1] - pipe_ptr[pb + p]
            if cur >= cnt:
                continue
            if pipe_seq_arr[pipe_seq_base[c] + pipe_ptr[pb + p] + cur] \
                    != g - task_op_base[c]:
                continue
            # 外部释放检查
            ep = ext_pred_arr[g]
            if ep != -1:
                if st_status[ep] != 3:
                    continue
                release = int(st_end[ep]) + delay_v
                if release > 0:  # now=0
                    if rel_sched[g] == 0:
                        rel_sched[g] = 1
                        # push to heap（内联）
                        hv = np.int64(release) * HS + g
                        idx = heap_n[0]
                        heap[idx] = hv
                        heap_n[0] += 1
                        while idx > 0:
                            par = (idx - 1) >> 1
                            if heap[par] <= heap[idx]:
                                break
                            tmp = heap[par]
                            heap[par] = heap[idx]
                            heap[idx] = tmp
                            idx = par
                    continue
            rel_sched[g] = 0
            st_status[g] = 1
            st_rdy[ck] = g

    now = 0.0
    it = 0
    while True:
        it += 1
        if it > 3000000:
            out_mk[0] = -3.0
            return

        # ---- advance_ddr ----
        elapsed = now - st_lu[0]
        while elapsed > 1e-9:
            na = 0
            for q in range(st_pn[0]):
                if st_pa[q] and st_pw[q] > 1e-9:
                    st_sa[na] = q
                    na += 1
            if na == 0:
                break
            mw = st_pw[st_sa[0]]
            for q in range(1, na):
                if st_pw[st_sa[q]] < mw:
                    mw = st_pw[st_sa[q]]
            ttf = mw * na
            if ttf >= elapsed - 1e-9:
                sh = elapsed / na
                for q in range(na):
                    st_pw[st_sa[q]] -= sh
                    if st_pw[st_sa[q]] < 0.0:
                        st_pw[st_sa[q]] = 0.0
                break
            for q in range(na):
                st_pw[st_sa[q]] -= mw
                if st_pw[st_sa[q]] < 0.0:
                    st_pw[st_sa[q]] = 0.0
            elapsed -= ttf
        st_lu[0] = now

        # ---- retire ----
        ret_ddr = False
        ret_n = 0
        for c in range(n_cores):
            for p in range(4):
                e = c * 4 + p
                if st_eop[e] != -1 and st_eend[e] <= now + 1e-9:
                    g = st_eop[e]
                    st_eop[e] = -1
                    st_status[g] = 3
                    remaining -= 1
                    ret_buf[ret_n] = g
                    ret_n += 1
                    # pipe cursor
                    pipe = gop_pipe[g]
                    ck = c * 4 + pipe
                    cur = st_pcur[ck]
                    st_pcur[ck] = cur + 1
                    pb = pipe_ptr_base[c]
                    cnt = pipe_ptr[pb + pipe + 1] - pipe_ptr[pb + pipe]
                    if cur + 1 < cnt:
                        nxt = task_op_base[c] + \
                            pipe_seq_arr[pipe_seq_base[c] +
                                         pipe_ptr[pb + pipe] + cur + 1]
                        # 内联 queue_if_ready
                        if st_status[nxt] == 0 and st_pred[nxt] == 0:
                            np2 = gop_pipe[nxt]
                            nck = c * 4 + np2
                            ncur = st_pcur[nck]
                            npb = pipe_ptr_base[c]
                            ncnt = pipe_ptr[npb + np2 + 1] - \
                                pipe_ptr[npb + np2]
                            if ncur < ncnt:
                                nseq = pipe_seq_arr[
                                    pipe_seq_base[c] +
                                    pipe_ptr[npb + np2] + ncur]
                                if nseq == nxt - task_op_base[c]:
                                    ep2 = ext_pred_arr[nxt]
                                    if ep2 != -1:
                                        if st_status[ep2] != 3:
                                            pass  # 等外部
                                        else:
                                            rel2 = int(st_end[ep2]) + delay_v
                                            if rel2 <= now:
                                                st_status[nxt] = 1
                                                st_rdy[nck] = nxt
                                            elif rel_sched[nxt] == 0:
                                                rel_sched[nxt] = 1
                                                hv2 = np.int64(rel2) * HS \
                                                    + nxt
                                                idx2 = heap_n[0]
                                                heap[idx2] = hv2
                                                heap_n[0] += 1
                                                while idx2 > 0:
                                                    par2 = (idx2 - 1) >> 1
                                                    if heap[par2] <= \
                                                            heap[idx2]:
                                                        break
                                                    tmp2 = heap[par2]
                                                    heap[par2] = heap[idx2]
                                                    heap[idx2] = tmp2
                                                    idx2 = par2
                                    else:
                                        st_status[nxt] = 1
                                        st_rdy[nck] = nxt
                    # ddr 退休
                    if st_pslot[g] != -1:
                        st_pa[st_pslot[g]] = False
                        st_pslot[g] = -1
                        ret_ddr = True
                    # succ 唤醒
                    for e2 in range(succ_ptr[g], succ_ptr[g + 1]):
                        s2 = succ_arr[e2]
                        st_pred[s2] -= 1
                        if st_status[s2] == 0 and st_pred[s2] == 0:
                            sp = gop_pipe[s2]
                            sc = gop_core[s2]
                            sck = sc * 4 + sp
                            scur = st_pcur[sck]
                            spb = pipe_ptr_base[sc]
                            scnt = pipe_ptr[spb + sp + 1] - \
                                pipe_ptr[spb + sp]
                            if scur < scnt:
                                sseq = pipe_seq_arr[
                                    pipe_seq_base[sc] +
                                    pipe_ptr[spb + sp] + scur]
                                if sseq == s2 - task_op_base[sc]:
                                    ep3 = ext_pred_arr[s2]
                                    if ep3 != -1:
                                        if st_status[ep3] == 3:
                                            rel3 = int(st_end[ep3]) + \
                                                delay_v
                                            if rel3 <= now:
                                                st_status[s2] = 1
                                                st_rdy[sck] = s2
                                            elif rel_sched[s2] == 0:
                                                rel_sched[s2] = 1
                                                hv3 = np.int64(rel3) * HS \
                                                    + s2
                                                idx3 = heap_n[0]
                                                heap[idx3] = hv3
                                                heap_n[0] += 1
                                                while idx3 > 0:
                                                    par3 = (idx3 - 1) >> 1
                                                    if heap[par3] <= \
                                                            heap[idx3]:
                                                        break
                                                    tmp3 = heap[par3]
                                                    heap[par3] = heap[idx3]
                                                    heap[idx3] = tmp3
                                                    idx3 = par3
                                    else:
                                        st_status[s2] = 1
                                        st_rdy[sck] = s2

        # reschedule_ddr（场景A版：简单投影）
        if ret_ddr:
            nn = 0
            for q in range(st_pn[0]):
                if st_pa[q]:
                    w2 = st_pw[q]
                    if w2 < 0.0:
                        w2 = 0.0
                    st_sw[nn] = w2
                    st_sa[nn] = st_po[q]  # 存 op（非 pool index）
                    nn += 1
            if nn > 0:
                # 按 (work, op) 插入排序
                for qi in range(1, nn):
                    ka = st_sa[qi]
                    kw = st_sw[qi]
                    qj = qi - 1
                    while qj >= 0 and (st_sw[qj] > kw or
                                       (st_sw[qj] == kw and
                                        st_sa[qj] > ka)):
                        st_sw[qj + 1] = st_sw[qj]
                        st_sa[qj + 1] = st_sa[qj]
                        qj -= 1
                    st_sw[qj + 1] = kw
                    st_sa[qj + 1] = ka
                cursor = np.float64(now)
                pv = 0.0
                act = nn
                qi = 0
                while qi < nn:
                    wk = st_sw[qi]
                    cursor += (wk - pv) * act
                    qj = qi
                    while qj < nn and abs(st_sw[qj] - wk) <= 1e-9:
                        val = cursor - 1e-9
                        iv = int(val)
                        if val > iv:
                            iv += 1
                        st_end[st_sa[qj]] = np.float64(iv)
                        qj += 1
                    act -= qj - qi
                    pv = wk
                    qi = qj
                for c in range(n_cores):
                    for p in range(4):
                        e3 = c * 4 + p
                        if st_eop[e3] != -1:
                            for q in range(nn):
                                if st_sa[q] == st_eop[e3]:
                                    st_eend[e3] = st_end[st_eop[e3]]
                                    break

        # ---- 释放堆到期处理 ----
        while heap_n[0] > 0 and heap[0] // HS <= now:
            # pop min
            top = heap[0]
            hn = heap_n[0] - 1
            heap_n[0] = hn
            heap[0] = heap[hn]
            idx = 0
            while True:
                l = 2 * idx + 1
                r = l + 1
                m = idx
                if l < hn and heap[l] < heap[m]:
                    m = l
                if r < hn and heap[r] < heap[m]:
                    m = r
                if m == idx:
                    break
                tmp = heap[m]
                heap[m] = heap[idx]
                heap[idx] = tmp
                idx = m
            g = int(top % HS)
            rel_sched[g] = 0
            # 重新 queue_if_ready
            if st_status[g] == 0 and st_pred[g] == 0:
                p4 = gop_pipe[g]
                c4 = gop_core[g]
                ck4 = c4 * 4 + p4
                cur4 = st_pcur[ck4]
                pb4 = pipe_ptr_base[c4]
                cnt4 = pipe_ptr[pb4 + p4 + 1] - pipe_ptr[pb4 + p4]
                if cur4 < cnt4:
                    sq4 = pipe_seq_arr[pipe_seq_base[c4] +
                                       pipe_ptr[pb4 + p4] + cur4]
                    if sq4 == g - task_op_base[c4]:
                        st_status[g] = 1
                        st_rdy[ck4] = g

        # ---- retired 的外部后继唤醒 ----
        for z in range(ret_n):
            gz = ret_buf[z]
            for e6 in range(ext_succ_ptr[gz], ext_succ_ptr[gz + 1]):
                t2 = ext_succ_arr[e6]
                if st_status[t2] == 0 and st_pred[t2] == 0:
                    tp = gop_pipe[t2]
                    tc = gop_core[t2]
                    tck = tc * 4 + tp
                    tcur = st_pcur[tck]
                    tpb = pipe_ptr_base[tc]
                    tcnt = pipe_ptr[tpb + tp + 1] - pipe_ptr[tpb + tp]
                    if tcur < tcnt:
                        tseq = pipe_seq_arr[pipe_seq_base[tc] +
                                            pipe_ptr[tpb + tp] + tcur]
                        if tseq == t2 - task_op_base[tc]:
                            tep = ext_pred_arr[t2]
                            if tep != -1:
                                if st_status[tep] == 3:
                                    trel = int(st_end[tep]) + delay_v
                                    if trel <= now:
                                        st_status[t2] = 1
                                        st_rdy[tck] = t2
                                    elif rel_sched[t2] == 0:
                                        rel_sched[t2] = 1
                                        thv = np.int64(trel) * HS + t2
                                        tidx = heap_n[0]
                                        heap[tidx] = thv
                                        heap_n[0] += 1
                                        while tidx > 0:
                                            tpar = (tidx - 1) >> 1
                                            if heap[tpar] <= heap[tidx]:
                                                break
                                            ttmp = heap[tpar]
                                            heap[tpar] = heap[tidx]
                                            heap[tidx] = ttmp
                                            tidx = tpar
                            else:
                                st_status[t2] = 1
                                st_rdy[tck] = t2

        if remaining == 0:
            break

        # ---- issue ----
        ipass = True
        while ipass:
            ipass = False
            for c in range(n_cores):
                for p in range(4):
                    e = c * 4 + p
                    while st_eop[e] == -1:
                        g = st_rdy[c * 4 + p]
                        if g == -1:
                            break
                        st_rdy[c * 4 + p] = -1
                        ipass = True
                        st_status[g] = 2
                        dur = gop_dur[g]
                        st_end[g] = now + np.float64(dur)
                        st_eop[e] = g
                        st_eend[e] = st_end[g]
                        if gop_is_ddr[g]:
                            # advance_ddr（内联）
                            el2 = now - st_lu[0]
                            while el2 > 1e-9:
                                na2 = 0
                                for q in range(st_pn[0]):
                                    if st_pa[q] and st_pw[q] > 1e-9:
                                        st_sa[na2] = q
                                        na2 += 1
                                if na2 == 0:
                                    break
                                mw2 = st_pw[st_sa[0]]
                                for q in range(1, na2):
                                    if st_pw[st_sa[q]] < mw2:
                                        mw2 = st_pw[st_sa[q]]
                                ttf2 = mw2 * na2
                                if ttf2 >= el2 - 1e-9:
                                    sh2 = el2 / na2
                                    for q in range(na2):
                                        st_pw[st_sa[q]] -= sh2
                                        if st_pw[st_sa[q]] < 0.0:
                                            st_pw[st_sa[q]] = 0.0
                                    break
                                for q in range(na2):
                                    st_pw[st_sa[q]] -= mw2
                                    if st_pw[st_sa[q]] < 0.0:
                                        st_pw[st_sa[q]] = 0.0
                                el2 -= ttf2
                            st_lu[0] = now
                            s_ = st_pn[0]
                            st_po[s_] = g
                            st_pw[s_] = np.float64(dur)
                            st_pa[s_] = True
                            st_pslot[g] = s_
                            st_pn[0] += 1
                            # reschedule（内联）
                            nn2 = 0
                            for q in range(st_pn[0]):
                                if st_pa[q]:
                                    w4 = st_pw[q]
                                    if w4 < 0.0:
                                        w4 = 0.0
                                    st_sw[nn2] = w4
                                    # 存 op（非 pool index）
                                    st_sa[nn2] = st_po[q]
                                    nn2 += 1
                            if nn2 > 0:
                                for qi in range(1, nn2):
                                    ka = st_sa[qi]
                                    kw = st_sw[qi]
                                    qj = qi - 1
                                    while qj >= 0 and (
                                            st_sw[qj] > kw or
                                            (st_sw[qj] == kw and
                                             st_sa[qj] > ka)):
                                        st_sw[qj + 1] = st_sw[qj]
                                        st_sa[qj + 1] = st_sa[qj]
                                        qj -= 1
                                    st_sw[qj + 1] = kw
                                    st_sa[qj + 1] = ka
                                cur5 = np.float64(now)
                                pv5 = 0.0
                                act5 = nn2
                                qi = 0
                                while qi < nn2:
                                    wk5 = st_sw[qi]
                                    cur5 += (wk5 - pv5) * act5
                                    qj = qi
                                    while qj < nn2 and abs(
                                            st_sw[qj] - wk5) <= 1e-9:
                                        val5 = cur5 - 1e-9
                                        iv5 = int(val5)
                                        if val5 > iv5:
                                            iv5 += 1
                                        st_end[st_sa[qj]] = \
                                            np.float64(iv5)
                                        qj += 1
                                    act5 -= qj - qi
                                    pv5 = wk5
                                    qi = qj
                                for c5 in range(n_cores):
                                    for p5 in range(4):
                                        e5 = c5 * 4 + p5
                                        if st_eop[e5] != -1:
                                            for q in range(nn2):
                                                if st_sa[q] == \
                                                        st_eop[e5]:
                                                    st_eend[e5] = \
                                                        st_end[st_eop[e5]]
                                                    break

        # ---- 时间推进 ----
        has = False
        tn = 0.0
        for e in range(n_cores * 4):
            if st_eop[e] != -1:
                if not has or st_eend[e] < tn:
                    tn = st_eend[e]
                    has = True
        if heap_n[0] > 0:
            rt = np.float64(heap[0] // HS)
            if not has or rt < tn:
                tn = rt
                has = True
        if not has:
            out_mk[0] = -1.0
            return
        if tn <= now:
            out_mk[0] = -2.0
            return
        now = tn

    mk = 0.0
    for g in range(n_gops):
        if st_end[g] > mk:
            mk = st_end[g]
    out_mk[0] = mk
