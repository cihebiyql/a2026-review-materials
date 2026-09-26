# -*- coding: utf-8 -*-
"""GPU 场景A：多核事件循环 CUDA 内核（纯索引，单函数，无外部调用）。

一线程一 plan。与 scene_a_kernel.scene_a_sim 逐位对齐。
所有输入由宿主预计算并打包为平铺数组。
"""
import numpy as np
from numba import cuda


@cuda.jit
def scene_a_kernel(n_cores, n_tasks, n_gops,
                   # 核序/任务归属/任务前驱
                   core_ptr, core_tasks, task_core_arr,
                   task_pred_ptr, task_pred_arr,
                   # step3 输出（全局 op 空间）
                   gop_pipe, gop_dur, gop_is_ddr, gop_task,
                   task_seq_ptr, task_seq_arr,
                   # pipe 序（每任务 5 段偏移 + 局部 op 序）
                   pipe_ptr, pipe_seq_arr, pipe_ptr_base, pipe_seq_base,
                   task_op_base,
                   succ_ptr, succ_arr, pred_cnt,
                   same_wait, cross_wait,
                   # scratch（单 plan，全部一维）
                   st_status, st_end, st_pred,
                   st_pcur, st_rdy,
                   st_eop, st_eend,
                   st_tstatus, st_tend,
                   st_cidx, st_cact, st_cprev,
                   st_po, st_pw, st_pa, st_pn, st_pslot, st_lu,
                   st_sa, st_sw,
                   out_mk):
    """单 plan 场景A模拟。out_mk[0] = makespan。"""
    # ---- 初始化 ----
    for g in range(n_gops):
        st_status[g] = 0
        st_end[g] = 0.0
        st_pred[g] = pred_cnt[g]
        st_pslot[g] = -1
    for k in range(n_tasks):
        st_tstatus[k] = 0
        st_tend[k] = 0
    for c in range(n_cores):
        st_cidx[c] = 0
        st_cact[c] = -1
        st_cprev[c] = -1
    for e in range(n_cores * 4):
        st_eop[e] = -1
        st_eend[e] = 0.0
    for k in range(n_tasks * 4):
        st_pcur[k] = 0
        st_rdy[k] = -1
    st_pn[0] = 0
    st_lu[0] = 0.0

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
        pool_dead = 0
        for c in range(n_cores):
            for p in range(4):
                e = c * 4 + p
                if st_eop[e] != -1 and st_eend[e] <= now + 1e-9:
                    g = st_eop[e]
                    st_eop[e] = -1
                    st_status[g] = 3
                    k = gop_task[g]
                    pipe = gop_pipe[g]
                    ck = k * 4 + pipe
                    cur = st_pcur[ck]
                    st_pcur[ck] = cur + 1
                    # pipe next 唤醒（内联 queue_if_ready）
                    psb = pipe_ptr_base[k]
                    p_cnt = pipe_ptr[psb + pipe + 1] - pipe_ptr[psb + pipe]
                    if cur + 1 < p_cnt:
                        psq_base = pipe_seq_base[k]
                        nxt = task_op_base[k] + \
                            pipe_seq_arr[psq_base + pipe_ptr[psb + pipe]
                                         + cur + 1]
                        if st_status[nxt] == 0 and st_pred[nxt] == 0:
                            np_pipe = gop_pipe[nxt]
                            nk = gop_task[nxt]
                            nck = nk * 4 + np_pipe
                            ncur = st_pcur[nck]
                            ncnt = pipe_ptr[pipe_ptr_base[nk] + np_pipe + 1] \
                                - pipe_ptr[pipe_ptr_base[nk] + np_pipe]
                            if ncur < ncnt:
                                nseq = pipe_seq_arr[
                                    pipe_seq_base[nk] +
                                    pipe_ptr[pipe_ptr_base[nk] + np_pipe]
                                    + ncur]
                                if nseq == nxt - task_op_base[nk]:
                                    st_status[nxt] = 1
                                    st_rdy[nck] = nxt
                    # ddr 退休
                    if st_pslot[g] != -1:
                        st_pa[st_pslot[g]] = False
                        st_pslot[g] = -1
                        ret_ddr = True
                        pool_dead += 1
                    # succ 唤醒
                    for e2 in range(succ_ptr[g], succ_ptr[g + 1]):
                        s2 = succ_arr[e2]
                        st_pred[s2] -= 1
                        if st_status[s2] == 0 and st_pred[s2] == 0:
                            sp_pipe = gop_pipe[s2]
                            sk = gop_task[s2]
                            sck = sk * 4 + sp_pipe
                            scur = st_pcur[sck]
                            scnt = pipe_ptr[pipe_ptr_base[sk] + sp_pipe + 1] \
                                - pipe_ptr[pipe_ptr_base[sk] + sp_pipe]
                            if scur < scnt:
                                sseq = pipe_seq_arr[
                                    pipe_seq_base[sk] +
                                    pipe_ptr[pipe_ptr_base[sk] + sp_pipe]
                                    + scur]
                                if sseq == s2 - task_op_base[sk]:
                                    st_status[s2] = 1
                                    st_rdy[sck] = s2

        # DDR compaction
        if pool_dead > 64:
            w = 0
            for q in range(st_pn[0]):
                if st_pa[q]:
                    if w != q:
                        st_po[w] = st_po[q]
                        st_pw[w] = st_pw[q]
                        st_pa[w] = True
                        st_pslot[st_po[w]] = w
                    w += 1
            st_pn[0] = w

        # reschedule_ddr（场景A：简单投影）
        if ret_ddr:
            nn = 0
            for q in range(st_pn[0]):
                if st_pa[q]:
                    w2 = st_pw[q]
                    if w2 < 0.0:
                        w2 = 0.0
                    st_sa[nn] = q  # pool index
                    st_sw[nn] = w2
                    nn += 1
            if nn > 0:
                for qi in range(1, nn):
                    ka = st_po[st_sa[qi]]
                    kw = st_sw[qi]
                    ka2 = st_sa[qi]
                    qj = qi - 1
                    while qj >= 0 and (st_sw[qj] > kw or
                                       (st_sw[qj] == kw and
                                        st_po[st_sa[qj]] > ka)):
                        st_sw[qj + 1] = st_sw[qj]
                        st_sa[qj + 1] = st_sa[qj]
                        qj -= 1
                    st_sw[qj + 1] = kw
                    st_sa[qj + 1] = ka2
                cursor = np.float64(now)
                pv = 0.0
                act = nn
                qi = 0
                while qi < nn:
                    wk = st_sw[qi]
                    cursor += (wk - pv) * act
                    qj = qi
                    while qj < nn and abs(st_sw[qj] - wk) <= 1e-9:
                        og = st_po[st_sa[qj]]
                        val = cursor - 1e-9
                        iv = int(val)
                        if val > iv:
                            iv += 1
                        st_end[og] = np.float64(iv)
                        qj += 1
                    act -= qj - qi
                    pv = wk
                    qi = qj
                for c in range(n_cores):
                    for p in range(4):
                        e3 = c * 4 + p
                        if st_eop[e3] != -1:
                            for q in range(nn):
                                if st_po[st_sa[q]] == st_eop[e3]:
                                    st_eend[e3] = st_end[st_eop[e3]]
                                    break

        # ---- 任务完成 ----
        for k in range(n_tasks):
            if st_tstatus[k] != 1:
                continue
            all_done = True
            for e in range(task_seq_ptr[k], task_seq_ptr[k + 1]):
                if st_status[task_seq_arr[e]] != 3:
                    all_done = False
                    break
            if all_done:
                st_tstatus[k] = 2
                iv3 = int(now)
                st_tend[k] = iv3
                c = task_core_arr[k]
                st_cact[c] = -1
                st_cprev[c] = iv3
                st_cidx[c] += 1

        # ---- activate ----
        for c in range(n_cores):
            if st_cact[c] != -1:
                continue
            if st_cidx[c] >= core_ptr[c + 1] - core_ptr[c]:
                continue
            k = core_tasks[core_ptr[c] + st_cidx[c]]
            ok_pred = True
            for e in range(task_pred_ptr[k], task_pred_ptr[k + 1]):
                if st_tstatus[task_pred_arr[e]] != 2:
                    ok_pred = False
                    break
            if ok_pred:
                rel = 0
                if st_cprev[c] >= 0:
                    rel = st_cprev[c] + same_wait
                for e in range(task_pred_ptr[k], task_pred_ptr[k + 1]):
                    pk = task_pred_arr[e]
                    if task_core_arr[pk] != c:
                        r2 = st_tend[pk] + cross_wait
                        if r2 > rel:
                            rel = r2
                if rel <= now:
                    st_tstatus[k] = 1
                    st_cact[c] = k
                    # 初始 queue（内联）
                    for e in range(task_seq_ptr[k], task_seq_ptr[k + 1]):
                        g2 = task_seq_arr[e]
                        if st_status[g2] == 0 and st_pred[g2] == 0:
                            p2 = gop_pipe[g2]
                            k2 = gop_task[g2]
                            ck2 = k2 * 4 + p2
                            cur2 = st_pcur[ck2]
                            pb2 = pipe_ptr_base[k2]
                            cnt2 = pipe_ptr[pb2 + p2 + 1] - pipe_ptr[pb2 + p2]
                            if cur2 < cnt2:
                                sq2 = pipe_seq_arr[
                                    pipe_seq_base[k2] +
                                    pipe_ptr[pb2 + p2] + cur2]
                                if sq2 == g2 - task_op_base[k2]:
                                    st_status[g2] = 1
                                    st_rdy[ck2] = g2

        # 全部完成？
        all_done2 = True
        for k in range(n_tasks):
            if st_tstatus[k] != 2:
                all_done2 = False
                break
        if all_done2:
            break

        # ---- issue ----
        ipass = True
        while ipass:
            ipass = False
            for c in range(n_cores):
                k = st_cact[c]
                if k == -1:
                    continue
                for p in range(4):
                    e = c * 4 + p
                    while st_eop[e] == -1:
                        g = st_rdy[k * 4 + p]
                        if g == -1:
                            break
                        st_rdy[k * 4 + p] = -1
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
                            # reschedule（内联，场景A版）
                            nn2 = 0
                            for q in range(st_pn[0]):
                                if st_pa[q]:
                                    w4 = st_pw[q]
                                    if w4 < 0.0:
                                        w4 = 0.0
                                    st_sa[nn2] = q
                                    st_sw[nn2] = w4
                                    nn2 += 1
                            if nn2 > 0:
                                for qi in range(1, nn2):
                                    ka = st_po[st_sa[qi]]
                                    kw = st_sw[qi]
                                    ka2 = st_sa[qi]
                                    qj = qi - 1
                                    while qj >= 0 and (
                                            st_sw[qj] > kw or
                                            (st_sw[qj] == kw and
                                             st_po[st_sa[qj]] > ka)):
                                        st_sw[qj + 1] = st_sw[qj]
                                        st_sa[qj + 1] = st_sa[qj]
                                        qj -= 1
                                    st_sw[qj + 1] = kw
                                    st_sa[qj + 1] = ka2
                                cur3 = np.float64(now)
                                pv3 = 0.0
                                act3 = nn2
                                qi = 0
                                while qi < nn2:
                                    wk3 = st_sw[qi]
                                    cur3 += (wk3 - pv3) * act3
                                    qj = qi
                                    while qj < nn2 and abs(
                                            st_sw[qj] - wk3) <= 1e-9:
                                        og3 = st_po[st_sa[qj]]
                                        val3 = cur3 - 1e-9
                                        iv3 = int(val3)
                                        if val3 > iv3:
                                            iv3 += 1
                                        st_end[og3] = np.float64(iv3)
                                        qj += 1
                                    act3 -= qj - qi
                                    pv3 = wk3
                                    qi = qj
                                for c4 in range(n_cores):
                                    for p4 in range(4):
                                        e4 = c4 * 4 + p4
                                        if st_eop[e4] != -1:
                                            for q in range(nn2):
                                                if st_po[st_sa[q]] == \
                                                        st_eop[e4]:
                                                    st_eend[e4] = \
                                                        st_end[st_eop[e4]]
                                                    break

        # ---- 时间推进 ----
        has = False
        tn = 0.0
        for e in range(n_cores * 4):
            if st_eop[e] != -1:
                if not has or st_eend[e] < tn:
                    tn = st_eend[e]
                    has = True
        for c in range(n_cores):
            if st_cact[c] != -1:
                continue
            if st_cidx[c] >= core_ptr[c + 1] - core_ptr[c]:
                continue
            k = core_tasks[core_ptr[c] + st_cidx[c]]
            ok_p = True
            for e in range(task_pred_ptr[k], task_pred_ptr[k + 1]):
                if st_tstatus[task_pred_arr[e]] != 2:
                    ok_p = False
                    break
            if ok_p:
                rel3 = 0
                if st_cprev[c] >= 0:
                    rel3 = st_cprev[c] + same_wait
                for e in range(task_pred_ptr[k], task_pred_ptr[k + 1]):
                    pk3 = task_pred_arr[e]
                    if task_core_arr[pk3] != c:
                        r4 = st_tend[pk3] + cross_wait
                        if r4 > rel3:
                            rel3 = r4
                if rel3 > now:
                    rt = np.float64(rel3)
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

    mk = 0
    for k in range(n_tasks):
        if st_tend[k] > mk:
            mk = st_tend[k]
    out_mk[0] = np.float64(mk)
