# -*- coding: utf-8 -*-
"""GPU step3：管道调度模拟 CUDA 内核（纯索引，无切片视图，无设备函数调用）。

与 step3_kernel.step3_sim 逐位对齐。单函数内联全部逻辑。
"""
import numpy as np
from numba import cuda


@cuda.jit
def step3_batch_kernel(op_base, ten_base,
                       # 打包数据（全局偏移，按 pair 分段）
                       op_pipe, op_ddr, op_dur,
                       in_ptr, in_ten, out_ptr, out_ten,
                       seq_ext_arr,
                       ten_sz, ten_pl1,
                       cons_ptr, cons_arr, prod_ptr, prod_arr,
                       pipe_sp, pipe_sq,      # 每任务的 5 段偏移 + 局部 op 序
                       pipe_sp_base,           # 每任务在 pipe_sp/sq 中的基址
                       alloc_order_flat, alloc_rank_flat,
                       alloc_base, n_alloc_flat,
                       ord_ip, ord_it, ord_opp, ord_ot,
                       succ_ptr, succ_arr,
                       pred_cnt,
                       cap_l1_v, cap_ub_v,
                       # scratch（按线程分区）
                       st_status, st_end, st_pred,
                       st_pcur, st_rdy, st_ardy, st_eop, st_eend,
                       st_mu, st_mp, st_rcon, st_res,
                       st_crb, st_crt, st_crh, st_crtail,
                       st_po, st_pw, st_pa, st_pn, st_pslot, st_lu,
                       st_sa, st_sw, st_req,
                       out_mk,
                       max_ops, max_ten, cr_cap, pool_cap):
    """一线程一 (plan,task)。out_mk=[mk,peakL1,peakUB] per pair（stride 3）。"""
    i = cuda.grid(1)
    if i >= op_base.size - 1:
        return
    lo = op_base[i]
    hi = op_base[i + 1]
    n = hi - lo
    tb = int(ten_base[i])
    n_ten = int(ten_base[i + 1]) - tb
    kb = i * max_ops       # op scratch 基址
    plb = i * pool_cap     # pool 基址
    psb = int(pipe_sp_base[i])  # pipe_sp/sq 基址

    if n == 0:
        out_mk[i * 3] = 0.0
        out_mk[i * 3 + 1] = 0.0
        out_mk[i * 3 + 2] = 0.0
        return

    # ---- 初始化 ----
    st_mu[i * 2] = 0
    st_mu[i * 2 + 1] = 0
    st_mp[i * 2] = 0
    st_mp[i * 2 + 1] = 0
    st_crh[i * 2] = 0
    st_crh[i * 2 + 1] = 0
    st_crtail[i * 2] = 0
    st_crtail[i * 2 + 1] = 0
    st_pn[i] = 0
    st_lu[i] = 0.0
    for v in range(n):
        st_status[kb + v] = 0
        st_end[kb + v] = 0.0
        st_pred[kb + v] = pred_cnt[lo + v]
        st_pslot[kb + v] = -1
    for p in range(4):
        st_pcur[i * 4 + p] = 0
        st_rdy[i * 4 + p] = -1
        st_ardy[i * 4 + p] = -1
        st_eop[i * 4 + p] = -1
        st_eend[i * 4 + p] = 0.0
    for t in range(n_ten):
        st_res[i * max_ten + t] = False
        st_rcon[i * max_ten + t] = cons_ptr[tb + t + 1] - cons_ptr[tb + t]
    # 初始驻留
    for t in range(n_ten):
        if ten_pl1[tb + t] != 0:
            if prod_ptr[tb + t + 1] - prod_ptr[tb + t] == 0 and \
                    cons_ptr[tb + t + 1] - cons_ptr[tb + t] > 0:
                T = 0 if ten_pl1[tb + t] == 1 else 1
                st_res[i * max_ten + t] = True
                st_mu[i * 2 + T] += ten_sz[tb + t]
                if st_mu[i * 2 + T] > st_mp[i * 2 + T]:
                    st_mp[i * 2 + T] = st_mu[i * 2 + T]
    # VIRGIN credit
    for T in range(2):
        cap = cap_l1_v if T == 0 else cap_ub_v
        unused = cap - st_mu[i * 2 + T]
        if unused > 0:
            h = i * cr_cap * 2 + T * cr_cap + st_crtail[i * 2 + T]
            st_crb[h] = unused
            st_crt[h] = -1
            st_crtail[i * 2 + T] += 1

    ab = int(alloc_base[i])
    n_alloc = int(n_alloc_flat[i])
    next_ar = 0

    # ---- 初始 queue ----
    for v in range(n):
        op_l = int(seq_ext_arr[lo + v]) - lo
        if st_status[kb + op_l] != 0 or st_pred[kb + op_l] != 0:
            continue
        p = op_pipe[lo + op_l]
        # pipe cursor（psb 偏移到该任务的 pipe 数据）
        cnt = pipe_sp[psb + p + 1] - pipe_sp[psb + p]
        cur = st_pcur[i * 4 + p]
        if cur >= cnt or pipe_sq[lo + pipe_sp[psb + p] + cur] != op_l:
            continue
        st_status[kb + op_l] = 1
        if alloc_rank_flat[lo + op_l] >= 0:
            st_ardy[i * 4 + p] = op_l
        else:
            st_rdy[i * 4 + p] = op_l

    now = 0.0
    done = 0
    it = 0
    while True:
        it += 1
        if it > 2000000:
            return
        # ---- advance_ddr ----
        elapsed = now - st_lu[i]
        while elapsed > 1e-9:
            na = 0
            for q in range(st_pn[i]):
                if st_pa[plb + q] and st_pw[plb + q] > 1e-9:
                    st_sa[kb + na] = q
                    na += 1
            if na == 0:
                break
            mw = st_pw[plb + st_sa[kb]]
            for q in range(1, na):
                if st_pw[plb + st_sa[kb + q]] < mw:
                    mw = st_pw[plb + st_sa[kb + q]]
            ttf = mw * na
            if ttf >= elapsed - 1e-9:
                sh = elapsed / na
                for q in range(na):
                    j = plb + st_sa[kb + q]
                    w = st_pw[j] - sh
                    st_pw[j] = 0.0 if w < 0.0 else w
                break
            for q in range(na):
                j = plb + st_sa[kb + q]
                w = st_pw[j] - mw
                st_pw[j] = 0.0 if w < 0.0 else w
            elapsed -= ttf
        st_lu[i] = now

        # ---- retire ----
        ret_ddr = False
        pool_dead = 0
        for p in range(4):
            e = i * 4 + p
            if st_eop[e] != -1 and st_eend[e] <= now + 1e-9:
                op_l = st_eop[e]  # 局部 op
                st_eop[e] = -1
                st_status[kb + op_l] = 3
                done += 1
                # consume_inputs
                for z in range(ord_ip[lo + op_l], ord_ip[lo + op_l + 1]):
                    ti = ord_it[z]
                    if ten_pl1[tb + ti] == 0:
                        continue
                    if st_rcon[i * max_ten + ti] > 0:
                        st_rcon[i * max_ten + ti] -= 1
                        if st_rcon[i * max_ten + ti] == 0:
                            T = 0 if ten_pl1[tb + ti] == 1 else 1
                            st_res[i * max_ten + ti] = False
                            st_mu[i * 2 + T] -= ten_sz[tb + ti]
                            h2 = i * cr_cap * 2 + T * cr_cap + \
                                st_crtail[i * 2 + T]
                            st_crb[h2] = ten_sz[tb + ti]
                            st_crt[h2] = ti
                            st_crtail[i * 2 + T] += 1
                # release_dead_outputs
                for z in range(ord_opp[lo + op_l], ord_opp[lo + op_l + 1]):
                    ti = ord_ot[z]
                    if ten_pl1[tb + ti] != 0 and \
                            st_rcon[i * max_ten + ti] == 0:
                        if st_res[i * max_ten + ti]:
                            T = 0 if ten_pl1[tb + ti] == 1 else 1
                            st_res[i * max_ten + ti] = False
                            st_mu[i * 2 + T] -= ten_sz[tb + ti]
                            h2 = i * cr_cap * 2 + T * cr_cap + \
                                st_crtail[i * 2 + T]
                            st_crb[h2] = ten_sz[tb + ti]
                            st_crt[h2] = ti
                            st_crtail[i * 2 + T] += 1
                # pipe cursor 前进 + 唤醒
                pipe = op_pipe[lo + op_l]
                pc = st_pcur[i * 4 + pipe]
                st_pcur[i * 4 + pipe] = pc + 1
                p_cnt = pipe_sp[psb + pipe + 1] - pipe_sp[psb + pipe]
                if pc + 1 < p_cnt:
                    nxt = pipe_sq[lo + pipe_sp[psb + pipe] + pc + 1]
                    if st_status[kb + nxt] == 0 and \
                            st_pred[kb + nxt] == 0:
                        pp = op_pipe[lo + nxt]
                        cc = st_pcur[i * 4 + pp]
                        pcnt = pipe_sp[psb + pp + 1] - pipe_sp[psb + pp]
                        if cc < pcnt and \
                                pipe_sq[lo + pipe_sp[psb + pp] + cc] == nxt:
                            st_status[kb + nxt] = 1
                            if alloc_rank_flat[lo + nxt] >= 0:
                                st_ardy[i * 4 + pp] = nxt
                            else:
                                st_rdy[i * 4 + pp] = nxt
                # ddr 退休（含池 compaction）
                if st_pslot[kb + op_l] != -1:
                    st_pa[plb + st_pslot[kb + op_l]] = False
                    st_pslot[kb + op_l] = -1
                    ret_ddr = True
                    pool_dead += 1
                    if pool_dead > 64:
                        w = 0
                        for q in range(st_pn[i]):
                            if st_pa[plb + q]:
                                if w != q:
                                    st_po[plb + w] = st_po[plb + q]
                                    st_pw[plb + w] = st_pw[plb + q]
                                    st_pa[plb + w] = True
                                    st_pslot[kb + st_po[plb + w] - lo] = w
                                w += 1
                        st_pn[i] = w
                        pool_dead = 0
                # 后继唤醒
                for e2 in range(succ_ptr[lo + op_l], succ_ptr[lo + op_l + 1]):
                    s2 = succ_arr[e2]
                    st_pred[kb + s2] -= 1
                    if st_status[kb + s2] == 0 and st_pred[kb + s2] == 0:
                        pp = op_pipe[lo + s2]
                        cc = st_pcur[i * 4 + pp]
                        pcnt = pipe_sp[psb + pp + 1] - pipe_sp[psb + pp]
                        if cc < pcnt and \
                                pipe_sq[lo + pipe_sp[psb + pp] + cc] == s2:
                            st_status[kb + s2] = 1
                            if alloc_rank_flat[lo + s2] >= 0:
                                st_ardy[i * 4 + pp] = s2
                            else:
                                st_rdy[i * 4 + pp] = s2

        # reschedule_ddr（step3 版：n/(n-1) 调整投影）
        if ret_ddr:
            nn = 0
            for q in range(st_pn[i]):
                if st_pa[plb + q]:
                    w = st_pw[plb + q]
                    if w < 0.0:
                        w = 0.0
                    st_sa[kb + nn] = plb + q  # 存 pool 绝对索引
                    st_sw[kb + nn] = w
                    nn += 1
            if nn > 0:
                # 插入排序 by (work, op_gid)
                for qi in range(1, nn):
                    ka = st_po[st_sa[kb + qi]]
                    kw = st_sw[kb + qi]
                    ka2 = st_sa[kb + qi]
                    qj = qi - 1
                    while qj >= 0 and (st_sw[kb + qj] > kw or
                                       (st_sw[kb + qj] == kw and
                                        st_po[st_sa[kb + qj]] > ka)):
                        st_sw[kb + qj + 1] = st_sw[kb + qj]
                        st_sa[kb + qj + 1] = st_sa[kb + qj]
                        qj -= 1
                    st_sw[kb + qj + 1] = kw
                    st_sa[kb + qj + 1] = ka2
                # 投影
                bc = np.float64(now)
                ac = np.float64(now)
                pv = 0.0
                act = nn
                qi = 0
                while qi < nn:
                    wk = st_sw[kb + qi]
                    dl = wk - pv
                    if act > 1:
                        bd = dl * (act - 1)
                        ad = bd * act / (act - 1)
                    else:
                        bd = dl
                        ad = dl
                    bc += bd
                    ac += ad
                    qj = qi
                    while qj < nn and abs(st_sw[kb + qj] - wk) <= 1e-9:
                        op_g = st_po[st_sa[kb + qj]]  # 全局 op index
                        op_l2 = op_g - lo
                        _v = ac - 1e-9
                        _iv = int(_v)
                        if _v > _iv:
                            _iv += 1
                        st_end[kb + op_l2] = np.float64(_iv)
                        qj += 1
                    act -= qj - qi
                    pv = wk
                    qi = qj
                # executor 同步
                for p in range(4):
                    e = i * 4 + p
                    if st_eop[e] != -1:
                        for q in range(nn):
                            if st_po[st_sa[kb + q]] == st_eop[e]:
                                st_eend[e] = st_end[kb + st_eop[e] - lo]
                                break

        if done == n:
            break

        # ---- issue ----
        ipass = True
        while ipass:
            ipass = False
            for p in range(4):
                e = i * 4 + p
                while st_eop[e] == -1:
                    op_l = -1
                    # alloc 候选（禁用测试）
                    if next_ar < n_alloc:
                        cand = alloc_order_flat[ab + next_ar]
                        if st_ardy[i * 4 + p] == cand and \
                                op_pipe[lo + cand] == p:
                            # capacity check
                            st_req[i * 2] = 0
                            st_req[i * 2 + 1] = 0
                            for ee in range(out_ptr[lo + cand],
                                            out_ptr[lo + cand + 1]):
                                ti = out_ten[ee] - tb
                                if ten_pl1[tb + ti] == 0 or \
                                        st_res[i * max_ten + ti]:
                                    continue
                                T = 0 if ten_pl1[tb + ti] == 1 else 1
                                st_req[i * 2 + T] += ten_sz[tb + ti]
                            ok = True
                            for T in range(2):
                                cap = cap_l1_v if T == 0 else cap_ub_v
                                if st_mu[i * 2 + T] + \
                                        st_req[i * 2 + T] > cap:
                                    ok = False
                            if ok:
                                st_ardy[i * 4 + p] = -1
                                op_l = cand
                    if op_l == -1 and st_rdy[i * 4 + p] != -1:
                        op_l = st_rdy[i * 4 + p]
                        st_rdy[i * 4 + p] = -1
                    if op_l == -1 and st_ardy[i * 4 + p] != -1:
                        # alloc 就绪但不满足门控——继续等待
                        pass
                    if op_l == -1:
                        break
                    ipass = True
                    if alloc_rank_flat[lo + op_l] >= 0:
                        next_ar += 1
                    st_status[kb + op_l] = 2
                    # allocate outputs
                    for ee in range(out_ptr[lo + op_l],
                                    out_ptr[lo + op_l + 1]):
                        ti = out_ten[ee] - tb
                        if ten_pl1[tb + ti] == 0 or \
                                st_res[i * max_ten + ti]:
                            continue
                        T = 0 if ten_pl1[tb + ti] == 1 else 1
                        st_res[i * max_ten + ti] = True
                        st_mu[i * 2 + T] += ten_sz[tb + ti]
                        if st_mu[i * 2 + T] > st_mp[i * 2 + T]:
                            st_mp[i * 2 + T] = st_mu[i * 2 + T]
                    dur = op_dur[lo + op_l]
                    st_end[kb + op_l] = now + np.float64(dur)
                    st_eop[e] = op_l
                    st_eend[e] = st_end[kb + op_l]
                    if op_ddr[lo + op_l]:
                        # advance_ddr
                        elapsed2 = now - st_lu[i]
                        while elapsed2 > 1e-9:
                            na2 = 0
                            for q in range(st_pn[i]):
                                if st_pa[plb + q] and \
                                        st_pw[plb + q] > 1e-9:
                                    st_sa[kb + na2] = q
                                    na2 += 1
                            if na2 == 0:
                                break
                            mw2 = st_pw[plb + st_sa[kb]]
                            for q in range(1, na2):
                                if st_pw[plb + st_sa[kb + q]] < mw2:
                                    mw2 = st_pw[plb + st_sa[kb + q]]
                            ttf2 = mw2 * na2
                            if ttf2 >= elapsed2 - 1e-9:
                                sh2 = elapsed2 / na2
                                for q in range(na2):
                                    j = plb + st_sa[kb + q]
                                    w = st_pw[j] - sh2
                                    st_pw[j] = 0.0 if w < 0.0 else w
                                break
                            for q in range(na2):
                                j = plb + st_sa[kb + q]
                                w = st_pw[j] - mw2
                                st_pw[j] = 0.0 if w < 0.0 else w
                            elapsed2 -= ttf2
                        st_lu[i] = now
                        # append to pool
                        s_ = st_pn[i]
                        st_po[plb + s_] = lo + op_l
                        st_pw[plb + s_] = np.float64(dur)
                        st_pa[plb + s_] = True
                        st_pslot[kb + op_l] = s_
                        st_pn[i] += 1
                        # reschedule（内联）
                        nn2 = 0
                        for q in range(st_pn[i]):
                            if st_pa[plb + q]:
                                w2 = st_pw[plb + q]
                                if w2 < 0.0:
                                    w2 = 0.0
                                st_sa[kb + nn2] = plb + q
                                st_sw[kb + nn2] = w2
                                nn2 += 1
                        if nn2 > 0:
                            for qi in range(1, nn2):
                                ka = st_po[st_sa[kb + qi]]
                                kw = st_sw[kb + qi]
                                ka2 = st_sa[kb + qi]
                                qj = qi - 1
                                while qj >= 0 and (
                                        st_sw[kb + qj] > kw or
                                        (st_sw[kb + qj] == kw and
                                         st_po[st_sa[kb + qj]] > ka)):
                                    st_sw[kb + qj + 1] = st_sw[kb + qj]
                                    st_sa[kb + qj + 1] = st_sa[kb + qj]
                                    qj -= 1
                                st_sw[kb + qj + 1] = kw
                                st_sa[kb + qj + 1] = ka2
                            bc2 = np.float64(now)
                            ac2 = np.float64(now)
                            pv2 = 0.0
                            act2 = nn2
                            qi = 0
                            while qi < nn2:
                                wk2 = st_sw[kb + qi]
                                dl2 = wk2 - pv2
                                if act2 > 1:
                                    bd2 = dl2 * (act2 - 1)
                                    ad2 = bd2 * act2 / (act2 - 1)
                                else:
                                    bd2 = dl2
                                    ad2 = dl2
                                bc2 += bd2
                                ac2 += ad2
                                qj = qi
                                while qj < nn2 and abs(
                                        st_sw[kb + qj] - wk2) <= 1e-9:
                                    og = st_po[st_sa[kb + qj]]
                                    _v2 = ac2 - 1e-9
                                    _iv2 = int(_v2)
                                    if _v2 > _iv2:
                                        _iv2 += 1
                                    st_end[kb + og - lo] = np.float64(_iv2)
                                    qj += 1
                                act2 -= qj - qi
                                pv2 = wk2
                                qi = qj
                            for p in range(4):
                                e = i * 4 + p
                                if st_eop[e] != -1:
                                    for q in range(nn2):
                                        if st_po[st_sa[kb + q]] == \
                                                lo + st_eop[e]:
                                            st_eend[e] = st_end[
                                                kb + st_eop[e]]
                                            break

        # ---- 时间推进 ----
        has = False
        tn = 0.0
        for p in range(4):
            e = i * 4 + p
            if st_eop[e] != -1:
                if not has or st_eend[e] < tn:
                    tn = st_eend[e]
                    has = True
        if not has:
            out_mk[i * 3] = -1.0
            return
        if tn <= now:
            out_mk[i * 3] = -2.0
            return
        now = tn

    mk = 0.0
    for v in range(n):
        if st_end[kb + v] > mk:
            mk = st_end[kb + v]
    out_mk[i * 3] = mk
    out_mk[i * 3 + 1] = np.float64(st_mp[i * 2])
    out_mk[i * 3 + 2] = np.float64(st_mp[i * 2 + 1])
