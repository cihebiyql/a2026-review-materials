# -*- coding: utf-8 -*-
"""从 gpu_scene_b_batch.py 机械生成 P3 缓存版 gpu_scene_b_cache_batch.py。
生成器本身可删；内核逻辑锚点见 assert。"""

SRC = "gpu_scene_b_batch.py"
DST = "gpu_scene_b_cache_batch.py"

src = open(SRC, encoding="utf-8").read()


def rep(old, new, tag):
    global src
    assert old in src, f"anchor missing: {tag}"
    assert src.count(old) == 1, f"anchor not unique: {tag}"
    src = src.replace(old, new)


# 1) 模块头
head_old = '"""GPU 场景B批量内核：一线程一 plan（纯基址+偏移，[N,1] 发射）。'
head_new = (
    '"""GPU 场景B+L2缓存（P3）批量内核：scene_b_batch + FIFO Cache。\n\n'
    '在 gpu_scene_b_batch 基础上加缓存改动（对齐 CPU scene_b_kernel\n'
    'scene_b_sim use_cache=1 路程）：issue 查命中→dur=ceil(sz/250)→\n'
    'CACHE_READ 独立池；retire 时 COPY_IN 完成→FIFO 插入（按字节逐出）；\n'
    '每轮开头 advance cac_pool；retired_cac 后 resched cac_pool。\n'
    '原 P2 语义保持（n_keys=0 时等价 P2）。')
rep(head_old, head_new, "head")

# 2) 内核名
rep("def scene_b_batch_kernel(", "def scene_b_cache_batch_kernel(", "kname")

# 3) 签名扩展：缓存输入
sig_old = """                         ext_pred_arr, ext_succ_ptr, ext_succ_arr,
                         # scratch（按 plan 分区，槽位 = pid）"""
sig_new = """                         ext_pred_arr, ext_succ_ptr, ext_succ_arr,
                         # 缓存输入（key 段平铺）
                         cache_idx_of_gop, cache_size_of_key, key_off,
                         cache_cap, cache_bw,
                         # scratch（按 plan 分区，槽位 = pid）"""
rep(sig_old, sig_new, "sig-in")

# 4) 签名扩展：缓存 scratch + 步长
sig2_old = """                         rel_sched, heap, heap_n, ret_buf, out_mk,
                         # 分区步长
                         s_ng, s_c4, s_pool, s_sa, s_heap, s_ret):"""
sig2_new = """                         rel_sched, heap, heap_n, ret_buf, out_mk,
                         # 缓存 scratch
                         cac_pool_op, cac_pool_w, cac_pool_alive,
                         cac_pool_n, cac_slot, cac_last_upd,
                         in_cache, fifo_q, fifo_sz,
                         fifo_head, fifo_tail, fifo_n, cache_used,
                         # 分区步长
                         s_ng, s_c4, s_pool, s_sa, s_heap, s_ret, s_key):"""
rep(sig2_old, sig2_new, "sig-scratch")

# 5) 段基址
base_old = """    b_heap = pid * s_heap
    b_ret = pid * s_ret"""
base_new = """    b_heap = pid * s_heap
    b_ret = pid * s_ret
    skb = key_off[pl]           # cache_size_of_key 段
    n_keys = int(key_off[pl + 1] - skb)
    b_key = pid * s_key         # in_cache / fifo_sz
    b_fifo = pid * s_key        # fifo_q（同键只插一次 → tail ≤ n_keys）
    b_cpool = pid * s_pool      # cac_pool_op/w/alive
    b_cslot = pid * s_ng        # cac_slot"""
rep(base_old, base_new, "base")

# 6) 初始化复位
init_old = """    st_pn[pid] = 0
    st_lu[pid] = 0.0
    heap_n[pid] = 0
    remaining = n_gops"""
init_new = """    st_pn[pid] = 0
    st_lu[pid] = 0.0
    heap_n[pid] = 0
    cac_pool_n[pid] = 0
    cac_last_upd[pid] = 0.0
    fifo_head[pid] = 0
    fifo_tail[pid] = 0
    fifo_n[pid] = 0
    cache_used[pid] = 0
    for kk in range(n_keys):
        in_cache[b_key + kk] = 0
    for g0 in range(n_gops):
        cac_slot[b_cslot + g0] = -1
    remaining = n_gops"""
rep(init_old, init_new, "init")

# 7) advance cac_pool（每轮开头）
adv_old = """            elapsed -= ttf
        st_lu[pid] = now

        # ---- retire ----
        ret_ddr = False"""
adv_new = """            elapsed -= ttf
        st_lu[pid] = now

        # ---- advance cac_pool（独立 CACHE_READ 池，250B/cyc）----
        celapsed = now - cac_last_upd[pid]
        while celapsed > 1e-9:
            cna = 0
            for q in range(cac_pool_n[pid]):
                if cac_pool_alive[b_cpool + q] and \\
                        cac_pool_w[b_cpool + q] > 1e-9:
                    st_sa[b_sa + cna] = q
                    cna += 1
            if cna == 0:
                break
            cmw = cac_pool_w[b_cpool + st_sa[b_sa + 0]]
            for q in range(1, cna):
                if cac_pool_w[b_cpool + st_sa[b_sa + q]] < cmw:
                    cmw = cac_pool_w[b_cpool + st_sa[b_sa + q]]
            cttf = cmw * cna
            if cttf >= celapsed - 1e-9:
                csh = celapsed / cna
                for q in range(cna):
                    cac_pool_w[b_cpool + st_sa[b_sa + q]] -= csh
                    if cac_pool_w[b_cpool + st_sa[b_sa + q]] < 0.0:
                        cac_pool_w[b_cpool + st_sa[b_sa + q]] = 0.0
                break
            for q in range(cna):
                cac_pool_w[b_cpool + st_sa[b_sa + q]] -= cmw
                if cac_pool_w[b_cpool + st_sa[b_sa + q]] < 0.0:
                    cac_pool_w[b_cpool + st_sa[b_sa + q]] = 0.0
            celapsed -= cttf
        cac_last_upd[pid] = now

        # ---- retire ----
        ret_ddr = False
        ret_cac = False"""
rep(adv_old, adv_new, "advance")

# 8) retire：cac_slot 释放 + cache insert
ret_old = """                    if st_pslot[b_slot + g] != -1:
                        st_pa[b_pool + st_pslot[b_slot + g]] = False
                        st_pslot[b_slot + g] = -1
                        ret_ddr = True"""
ret_new = """                    if st_pslot[b_slot + g] != -1:
                        st_pa[b_pool + st_pslot[b_slot + g]] = False
                        st_pslot[b_slot + g] = -1
                        ret_ddr = True
                    if cac_slot[b_cslot + g] != -1:
                        cac_pool_alive[b_cpool + cac_slot[b_cslot + g]] \\
                            = False
                        cac_slot[b_cslot + g] = -1
                        ret_cac = True
                    # cache insert（COPY_IN 完成后写缓存，FIFO 按字节逐出）
                    ci = cache_idx_of_gop[gb + g]
                    if ci >= 0:
                        csz = cache_size_of_key[skb + ci]
                        if csz <= cache_cap and in_cache[b_key + ci] != 1:
                            while fifo_n[pid] > 0 and \\
                                    cache_used[pid] + csz > cache_cap:
                                old = fifo_q[b_fifo + fifo_head[pid]]
                                fifo_head[pid] += 1
                                fifo_n[pid] -= 1
                                in_cache[b_key + old] = 0
                                cache_used[pid] -= fifo_sz[b_key + old]
                            fifo_q[b_fifo + fifo_tail[pid]] = ci
                            fifo_sz[b_key + ci] = csz
                            fifo_tail[pid] += 1
                            fifo_n[pid] += 1
                            in_cache[b_key + ci] = 1
                            cache_used[pid] += csz"""
rep(ret_old, ret_new, "retire")

# 9) resched cac_pool（ddr resched 之后、释放堆之前）
rs_old = """        # ---- 释放堆到期处理 ----"""
rs_new = """        if ret_cac:
            cnn = 0
            for q in range(cac_pool_n[pid]):
                if cac_pool_alive[b_cpool + q]:
                    cw2 = cac_pool_w[b_cpool + q]
                    if cw2 < 0.0:
                        cw2 = 0.0
                    st_sw[b_sw + cnn] = cw2
                    st_sa[b_sa + cnn] = cac_pool_op[b_cpool + q]
                    cnn += 1
            if cnn > 0:
                for qi in range(1, cnn):
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
                ccursor = np.float64(now)
                cpv = 0.0
                cact = cnn
                qi = 0
                while qi < cnn:
                    cwk = st_sw[b_sw + qi]
                    ccursor += (cwk - cpv) * cact
                    qj = qi
                    while qj < cnn and abs(st_sw[b_sw + qj] - cwk) <= 1e-9:
                        cval = ccursor - 1e-9
                        civ = int(cval)
                        if cval > civ:
                            civ += 1
                        st_end[b_end + st_sa[b_sa + qj]] = np.float64(civ)
                        qj += 1
                    cact -= qj - qi
                    cpv = cwk
                    qi = qj
                for c in range(n_cores):
                    for p in range(4):
                        e3 = c * 4 + p
                        if st_eop[b_c4 + e3] != -1:
                            for q in range(cnn):
                                if st_sa[b_sa + q] == st_eop[b_c4 + e3]:
                                    st_eend[b_c4 + e3] = \\
                                        st_end[b_end + st_eop[b_c4 + e3]]
                                    break

        # ---- 释放堆到期处理 ----"""
rep(rs_old, rs_new, "resched-cac")

# 10) issue：命中改 dur + cac_pool 路径
iss_old = """                        ipass = True
                        st_status[b_stat + g] = 2
                        dur = gop_dur[gb + g]
                        st_end[b_end + g] = now + np.float64(dur)
                        st_eop[b_c4 + e] = g
                        st_eend[b_c4 + e] = st_end[b_end + g]
                        if gop_is_ddr[gb + g]:"""
iss_new = """                        ipass = True
                        st_status[b_stat + g] = 2
                        dur = gop_dur[gb + g]
                        ci_g = cache_idx_of_gop[gb + g]
                        if ci_g >= 0 and in_cache[b_key + ci_g] == 1:
                            dur = -(-cache_size_of_key[skb + ci_g] //
                                    cache_bw)
                            if dur < 1:
                                dur = 1
                        st_end[b_end + g] = now + np.float64(dur)
                        st_eop[b_c4 + e] = g
                        st_eend[b_c4 + e] = st_end[b_end + g]
                        if ci_g >= 0 and in_cache[b_key + ci_g] == 1:
                            # 命中：进 CACHE_READ 独立池（250B/cyc）
                            cel2 = now - cac_last_upd[pid]
                            while cel2 > 1e-9:
                                cna2 = 0
                                for q in range(cac_pool_n[pid]):
                                    if cac_pool_alive[b_cpool + q] and \\
                                            cac_pool_w[b_cpool + q] > 1e-9:
                                        st_sa[b_sa + cna2] = q
                                        cna2 += 1
                                if cna2 == 0:
                                    break
                                cmw2 = cac_pool_w[b_cpool + st_sa[b_sa + 0]]
                                for q in range(1, cna2):
                                    if cac_pool_w[b_cpool +
                                                  st_sa[b_sa + q]] < cmw2:
                                        cmw2 = cac_pool_w[b_cpool +
                                                          st_sa[b_sa + q]]
                                cttf2 = cmw2 * cna2
                                if cttf2 >= cel2 - 1e-9:
                                    csh2 = cel2 / cna2
                                    for q in range(cna2):
                                        cac_pool_w[b_cpool +
                                                   st_sa[b_sa + q]] -= csh2
                                        if cac_pool_w[b_cpool +
                                                      st_sa[b_sa + q]] < 0.0:
                                            cac_pool_w[b_cpool +
                                                       st_sa[b_sa + q]] = 0.0
                                    break
                                for q in range(cna2):
                                    cac_pool_w[b_cpool +
                                               st_sa[b_sa + q]] -= cmw2
                                    if cac_pool_w[b_cpool +
                                                  st_sa[b_sa + q]] < 0.0:
                                        cac_pool_w[b_cpool +
                                                   st_sa[b_sa + q]] = 0.0
                                cel2 -= cttf2
                            cac_last_upd[pid] = now
                            cs_ = cac_pool_n[pid]
                            cac_pool_op[b_cpool + cs_] = g
                            cac_pool_w[b_cpool + cs_] = np.float64(dur)
                            cac_pool_alive[b_cpool + cs_] = True
                            cac_slot[b_cslot + g] = cs_
                            cac_pool_n[pid] += 1
                            # resched cac_pool（内联）
                            cnn2 = 0
                            for q in range(cac_pool_n[pid]):
                                if cac_pool_alive[b_cpool + q]:
                                    cw4 = cac_pool_w[b_cpool + q]
                                    if cw4 < 0.0:
                                        cw4 = 0.0
                                    st_sw[b_sw + cnn2] = cw4
                                    st_sa[b_sa + cnn2] = \\
                                        cac_pool_op[b_cpool + q]
                                    cnn2 += 1
                            if cnn2 > 0:
                                for qi in range(1, cnn2):
                                    ka = st_sa[b_sa + qi]
                                    kw = st_sw[b_sw + qi]
                                    qj = qi - 1
                                    while qj >= 0 and (
                                            st_sw[b_sw + qj] > kw or
                                            (st_sw[b_sw + qj] == kw and
                                             st_sa[b_sa + qj] > ka)):
                                        st_sw[b_sw + qj + 1] = \\
                                            st_sw[b_sw + qj]
                                        st_sa[b_sa + qj + 1] = \\
                                            st_sa[b_sa + qj]
                                        qj -= 1
                                    st_sw[b_sw + qj + 1] = kw
                                    st_sa[b_sa + qj + 1] = ka
                                ccur5 = np.float64(now)
                                cpv5 = 0.0
                                cact5 = cnn2
                                qi = 0
                                while qi < cnn2:
                                    cwk5 = st_sw[b_sw + qi]
                                    ccur5 += (cwk5 - cpv5) * cact5
                                    qj = qi
                                    while qj < cnn2 and abs(
                                            st_sw[b_sw + qj] - cwk5) <= 1e-9:
                                        cval5 = ccur5 - 1e-9
                                        civ5 = int(cval5)
                                        if cval5 > civ5:
                                            civ5 += 1
                                        st_end[b_end + st_sa[b_sa + qj]] = \\
                                            np.float64(civ5)
                                        qj += 1
                                    cact5 -= qj - qi
                                    cpv5 = cwk5
                                    qi = qj
                                for c5 in range(n_cores):
                                    for p5 in range(4):
                                        e5 = c5 * 4 + p5
                                        if st_eop[b_c4 + e5] != -1:
                                            for q in range(cnn2):
                                                if st_sa[b_sa + q] == \\
                                                        st_eop[b_c4 + e5]:
                                                    st_eend[b_c4 + e5] = \\
                                                        st_end[b_end +
                                                               st_eop[
                                                                   b_c4 + e5]]
                                                    break
                        elif gop_is_ddr[gb + g]:"""
rep(iss_old, iss_new, "issue")

open(DST, "w", encoding="utf-8").write(src)
import ast
ast.parse(src)
print("written + syntax OK,", len(src.splitlines()), "lines")
for probe in ["scene_b_cache_batch_kernel", "fifo_q[b_fifo + fifo_tail",
              "ci_g = cache_idx_of_gop", "advance cac_pool", "s_key",
              "ret_cac = False", "in_cache[b_key + ci] = 1"]:
    assert probe in src, probe
print("anchors OK")
