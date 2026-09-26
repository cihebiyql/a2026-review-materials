# -*- coding: utf-8 -*-
"""GPU step3 对拍：CUDA 管道调度 vs CPU 复刻（stage_step3_one）。

node1（cuda12.1）：
  A2026_ATT=/data1/qlyu/a2026/att CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=v2solver /data/qlyu/anaconda3/envs/cuda12.1/bin/python \
  gpu_step3_gate.py case_019 case_001 case_082 case_050
"""
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "v2solver"))

ATT = os.environ.get("A2026_ATT", r"C:/shumo_live/a_data")
if not os.path.isdir(ATT):
    ATT = "/data1/qlyu/a2026/att"
sys.path.insert(0, os.path.join(ATT, "code"))

from fast_eval_p1 import (GraphCodec, TaskBuild, stage_step1, stage_step2,
                          build_ext_all, stage_step3, _transpose_csr,
                          BW, CAP_L1, CAP_UB)
import stub_multicore_cut_and_schedule as stub
from structure_split import structure_aware_plan


def prep_step3_inputs(tb, seqs, pss, exts):
    """打包所有任务的 step3 输入到平铺数组。"""
    B = tb.n_tasks
    results = []
    for k in range(B):
        ext = exts[k]
        n_ops = ext["n_ext_ops"]
        n_ten = ext["n_ext_ten"]
        # in/out CSR
        in_pairs = sorted(ext["in_pairs"])
        out_pairs = sorted(ext["out_pairs"])
        in_ptr = np.zeros(n_ops + 1, dtype=np.int64)
        for o, _ in in_pairs:
            in_ptr[o + 1] += 1
        in_ptr = np.cumsum(in_ptr)
        in_ten = np.array([t for _, t in in_pairs], dtype=np.int64)
        out_ptr = np.zeros(n_ops + 1, dtype=np.int64)
        for o, _ in out_pairs:
            out_ptr[o + 1] += 1
        out_ptr = np.cumsum(out_ptr)
        out_ten = np.array([t for _, t in out_pairs], dtype=np.int64)
        # op durations / ddr
        op_dur = np.zeros(n_ops, dtype=np.int64)
        op_ddr = np.zeros(n_ops, dtype=np.bool_)
        ten_size = ext["ten_size"]
        ten_pos = ext["ten_pos"]
        op_type = ext["op_type"]
        op_pipe = ext["op_pipe"]
        for o in range(n_ops):
            t = op_type[o]
            if t == 1:
                total = sum(int(ten_size[int(out_ten[e2])])
                                for e2 in range(out_ptr[o], out_ptr[o + 1]))
                op_dur[o] = max(1, -(-total // BW))
            elif t in (2, 3, 4):
                total = sum(int(ten_size[int(in_ten[e2])])
                                for e2 in range(in_ptr[o], in_ptr[o + 1]))
                op_dur[o] = max(1, -(-total // BW))
            else:
                op_dur[o] = int(ext["op_cycles"][o])
            if t in (1, 2, 3, 4):
                for e2 in range(in_ptr[o], in_ptr[o + 1]):
                    if ten_pos[int(in_ten[e2])] == 0:
                        op_ddr[o] = True
                for e2 in range(out_ptr[o], out_ptr[o + 1]):
                    if ten_pos[int(out_ten[e2])] == 0:
                        op_ddr[o] = True
        # succ CSR（数据依赖，不含 mem dep——mem dep 不影响单任务 makespan）
        ten_cons_ptr, ten_cons_arr = _transpose_csr(in_ptr, in_ten,
                                                    n_ops, n_ten)
        es = []
        for o, t in out_pairs:
            for c in range(ten_cons_ptr[t], ten_cons_ptr[t + 1]):
                es.append((o, int(ten_cons_arr[c])))
        for s, d in ext["direct"]:
            es.append((s, d))
        if es:
            comp = np.unique(np.array([s * n_ops + d for s, d in es],
                                      dtype=np.int64))
            esrc = (comp // n_ops).astype(np.int64)
            edst = (comp % n_ops).astype(np.int64)
        else:
            esrc = np.zeros(0, dtype=np.int64)
            edst = np.zeros(0, dtype=np.int64)
        succ_ptr = np.zeros(n_ops + 1, dtype=np.int64)
        np.add.at(succ_ptr, esrc + 1, 1)
        succ_ptr = np.cumsum(succ_ptr)
        succ_arr = edst[np.argsort(esrc * n_ops + edst, kind="stable")]
        pred_cnt = np.zeros(n_ops, dtype=np.int64)
        np.add.at(pred_cnt, edst, 1)
        # pipe seq
        seq_ext = ext["seq_ext"]
        pipe_sp = np.zeros(5, dtype=np.int64)
        for op in seq_ext:
            pipe_sp[op_pipe[op] + 1] += 1
        pipe_sp = np.cumsum(pipe_sp)
        pipe_sq = seq_ext.copy()
        fillp = pipe_sp[:4].copy()
        for op in seq_ext:
            p = op_pipe[op]
            pipe_sq[fillp[p]] = op
            fillp[p] += 1
        # alloc order
        ten_pl1 = np.where(ten_pos == 1, 1, np.where(ten_pos == 2, 2, 0))
        alloc_list = [int(op) for op in seq_ext
                      if any(ten_pl1[int(out_ten[e2])] != 0
                             for e2 in range(out_ptr[op], out_ptr[op + 1]))]
        alloc_rank = np.full(n_ops, -1, dtype=np.int64)
        for r, op in enumerate(alloc_list):
            alloc_rank[op] = r
        # ord_in / ord_out（set 迭代序——用 sorted 简化，对拍仅 makespan）
        ord_ip = in_ptr.copy()
        ord_opp = out_ptr.copy()
        ord_it = in_ten.copy()
        ord_ot = out_ten.copy()
        # tensor prod/cons
        prod_ptr = np.zeros(n_ten + 1, dtype=np.int64)
        for _, t in out_pairs:
            prod_ptr[t + 1] += 1
        prod_ptr = np.cumsum(prod_ptr)
        prod_arr = np.array([o for o, _ in out_pairs], dtype=np.int64)

        results.append({
            "n": n_ops, "n_ten": n_ten,
            "in_ptr": in_ptr, "in_ten": in_ten,
            "out_ptr": out_ptr, "out_ten": out_ten,
            "op_pipe": op_pipe, "op_ddr": op_ddr, "op_dur": op_dur,
            "seq_ext": seq_ext,
            "ten_sz": ten_size, "ten_pl1": ten_pl1,
            "cons_ptr": ten_cons_ptr, "cons_arr": ten_cons_arr,
            "prod_ptr": prod_ptr, "prod_arr": prod_arr,
            "succ_ptr": succ_ptr, "succ_arr": succ_arr,
            "pred_cnt": pred_cnt,
            "pipe_sp": pipe_sp, "pipe_sq": pipe_sq,
            "alloc_order": np.array(alloc_list, dtype=np.int64),
            "alloc_rank": alloc_rank,
            "ord_ip": ord_ip, "ord_it": ord_it,
            "ord_opp": ord_opp, "ord_ot": ord_ot,
        })
    return results


def pack_step3(inputs):
    """拼成平铺数组。"""
    B = len(inputs)
    op_base = np.zeros(B + 1, dtype=np.int64)
    ten_base = np.zeros(B + 1, dtype=np.int64)
    edge_base = np.zeros(B + 1, dtype=np.int64)  # in/out 共用边基址（不对——分开）
    in_edge_base = np.zeros(B + 1, dtype=np.int64)
    out_edge_base = np.zeros(B + 1, dtype=np.int64)
    succ_edge_base = np.zeros(B + 1, dtype=np.int64)
    pipe_base = np.zeros(B + 1, dtype=np.int64)
    pipe_sq_base = np.zeros(B + 1, dtype=np.int64)
    alloc_base = np.zeros(B + 1, dtype=np.int64)
    for i, d in enumerate(inputs):
        op_base[i + 1] = op_base[i] + d["n"]
        ten_base[i + 1] = ten_base[i] + d["n_ten"]
        in_edge_base[i + 1] = in_edge_base[i] + len(d["in_ten"])
        out_edge_base[i + 1] = out_edge_base[i] + len(d["out_ten"])
        succ_edge_base[i + 1] = succ_edge_base[i] + len(d["succ_arr"])
        pipe_base[i + 1] = pipe_base[i] + 5  # 5 段偏移
        pipe_sq_base[i + 1] = pipe_sq_base[i] + d["n"]  # pipe_sq 长度 = n
        alloc_base[i + 1] = alloc_base[i] + len(d["alloc_order"])

    n_ops_all = int(op_base[-1])
    n_ten_all = int(ten_base[-1])
    op_pipe = np.zeros(n_ops_all, dtype=np.int64)
    op_ddr = np.zeros(n_ops_all, dtype=np.bool_)
    op_dur = np.zeros(n_ops_all, dtype=np.int64)
    seq_ext = np.zeros(n_ops_all, dtype=np.int64)
    in_ptr = np.zeros(n_ops_all + 1, dtype=np.int64)
    in_ten = np.zeros(int(in_edge_base[-1]), dtype=np.int64)
    out_ptr = np.zeros(n_ops_all + 1, dtype=np.int64)
    out_ten = np.zeros(int(out_edge_base[-1]), dtype=np.int64)
    ten_sz = np.zeros(n_ten_all, dtype=np.int64)
    ten_pl1 = np.zeros(n_ten_all, dtype=np.int64)
    cons_ptr = np.zeros(n_ten_all + 1, dtype=np.int64)
    cons_arr = np.zeros(int(in_edge_base[-1]), dtype=np.int64)
    prod_ptr = np.zeros(n_ten_all + 1, dtype=np.int64)
    prod_arr = np.zeros(int(out_edge_base[-1]), dtype=np.int64)
    succ_ptr = np.zeros(n_ops_all + 1, dtype=np.int64)
    succ_arr = np.zeros(int(succ_edge_base[-1]), dtype=np.int64)
    pred_cnt = np.zeros(n_ops_all, dtype=np.int64)
    pipe_sp = np.zeros(B * 5, dtype=np.int64)
    pipe_sq = np.zeros(n_ops_all, dtype=np.int64)
    pipe_sp_base = np.zeros(B, dtype=np.int64)
    alloc_order = np.zeros(int(alloc_base[-1]), dtype=np.int64)
    alloc_rank = np.full(n_ops_all, -1, dtype=np.int64)
    n_alloc = np.zeros(B, dtype=np.int64)
    ord_ip = np.zeros(n_ops_all + 1, dtype=np.int64)
    ord_it = np.zeros(int(in_edge_base[-1]), dtype=np.int64)
    ord_opp = np.zeros(n_ops_all + 1, dtype=np.int64)
    ord_ot = np.zeros(int(out_edge_base[-1]), dtype=np.int64)

    for i, d in enumerate(inputs):
        ob, tb = int(op_base[i]), int(ten_base[i])
        ieb, oeb = int(in_edge_base[i]), int(out_edge_base[i])
        seb = int(succ_edge_base[i])
        n, nt = d["n"], d["n_ten"]
        op_pipe[ob:ob + n] = d["op_pipe"]
        op_ddr[ob:ob + n] = d["op_ddr"]
        op_dur[ob:ob + n] = d["op_dur"]
        seq_ext[ob:ob + n] = d["seq_ext"] + ob  # 全局化
        in_ptr[ob:ob + n + 1] = d["in_ptr"] + ieb
        in_ten[ieb:ieb + len(d["in_ten"])] = d["in_ten"] + tb
        out_ptr[ob:ob + n + 1] = d["out_ptr"] + oeb
        out_ten[oeb:oeb + len(d["out_ten"])] = d["out_ten"] + tb
        ten_sz[tb:tb + nt] = d["ten_sz"]
        ten_pl1[tb:tb + nt] = d["ten_pl1"]
        cons_ptr[tb:tb + nt + 1] = d["cons_ptr"] + ieb
        cons_arr[ieb:ieb + len(d["cons_arr"])] = d["cons_arr"] + ob
        prod_ptr[tb:tb + nt + 1] = d["prod_ptr"] + oeb
        prod_arr[oeb:oeb + len(d["prod_arr"])] = d["prod_arr"] + ob
        succ_ptr[ob:ob + n + 1] = d["succ_ptr"] + seb
        succ_arr[seb:seb + len(d["succ_arr"])] = d["succ_arr"]
        pred_cnt[ob:ob + n] = d["pred_cnt"]
        pipe_sp_base[i] = i * 5
        pipe_sp[i * 5:i * 5 + 5] = d["pipe_sp"]
        pipe_sq[ob:ob + n] = d["pipe_sq"]  # 局部 op（不加 ob，内核用 lo 偏移）
        alloc_order[int(alloc_base[i]):int(alloc_base[i]) +
                    len(d["alloc_order"])] = d["alloc_order"]
        alloc_rank[ob:ob + n] = d["alloc_rank"]
        n_alloc[i] = len(d["alloc_order"])
        ord_ip[ob:ob + n + 1] = d["ord_ip"] + ieb
        ord_it[ieb:ieb + len(d["ord_it"])] = d["ord_it"]  # 局部索引
        ord_opp[ob:ob + n + 1] = d["ord_opp"] + oeb
        ord_ot[oeb:oeb + len(d["ord_ot"])] = d["ord_ot"]  # 局部索引

    return (op_base, ten_base, op_pipe, op_ddr, op_dur,
            in_ptr, in_ten, out_ptr, out_ten, seq_ext,
            ten_sz, ten_pl1, cons_ptr, cons_arr, prod_ptr, prod_arr,
            succ_ptr, succ_arr, pred_cnt,
            pipe_sp, pipe_sq, pipe_sp_base,
            alloc_order, alloc_rank, alloc_base, n_alloc,
            ord_ip, ord_it, ord_opp, ord_ot)


def run_case(case, seed, use_v2=False, cpc=1):
    from numba import cuda
    from gpu_step3 import step3_batch_kernel
    graph = json.load(open(os.path.join(ATT, "data", f"{case}.json")))
    if use_v2:
        plan, _ = structure_aware_plan(graph, 4, chunks_per_core=cpc)
    else:
        plan = stub.generate_multicore_plan(graph, num_cores=4, seed=seed,
                                            min_subgraph_size=50,
                                            max_subgraph_size=100)
    gc = GraphCodec(graph)
    pv = stub.derive_multicore_plan(graph, plan)
    tb = TaskBuild(gc, pv)
    seqs = stage_step1(tb)
    pss = stage_step2(tb, seqs)
    exts = build_ext_all(tb, seqs, pss)
    cpu_s3 = stage_step3(exts)
    inputs = prep_step3_inputs(tb, seqs, pss, exts)

    (op_base, ten_base, op_pipe, op_ddr, op_dur,
     in_ptr, in_ten, out_ptr, out_ten, seq_ext,
     ten_sz, ten_pl1, cons_ptr, cons_arr, prod_ptr, prod_arr,
     succ_ptr, succ_arr, pred_cnt,
     pipe_sp, pipe_sq, pipe_sp_base,
     alloc_order, alloc_rank, alloc_base, n_alloc,
     ord_ip, ord_it, ord_opp, ord_ot) = pack_step3(inputs)

    B = len(inputs)
    n_ops_all = int(op_base[-1])
    n_ten_all = int(ten_base[-1])
    max_ops = max(d["n"] for d in inputs)
    max_ten = max(d["n_ten"] for d in inputs)
    cr_cap = max_ten + 8
    pool_cap = max_ops + 8

    d = cuda.to_device
    d_op_b = d(op_base)
    d_ten_b = d(ten_base)
    d_pipe = d(op_pipe)
    d_ddr = d(op_ddr)
    d_dur = d(op_dur)
    d_ip = d(in_ptr)
    d_it = d(in_ten)
    d_op_ptr = d(out_ptr)
    d_ot = d(out_ten)
    d_seq = d(seq_ext)
    d_tsz = d(ten_sz)
    d_tpl1 = d(ten_pl1)
    d_cnp = d(cons_ptr)
    d_cna = d(cons_arr)
    d_ppp = d(prod_ptr)
    d_ppa = d(prod_arr)
    d_sp = d(succ_ptr)
    d_sa_arr = d(succ_arr)
    d_pc = d(pred_cnt)
    d_psp = d(pipe_sp)
    d_psq = d(pipe_sq)
    d_psb = d(pipe_sp_base)
    d_ao = d(alloc_order)
    d_ar = d(alloc_rank)
    d_ab = d(alloc_base)
    d_nal = d(n_alloc)
    d_oip = d(ord_ip)
    d_oit = d(ord_it)
    d_opp = d(ord_opp)
    d_oot = d(ord_ot)

    # scratch
    d_status = d(np.zeros(B * max_ops, dtype=np.int64))
    d_end = d(np.zeros(B * max_ops, dtype=np.float64))
    d_pred = d(np.zeros(B * max_ops, dtype=np.int64))
    d_pcur = d(np.zeros(B * 4, dtype=np.int64))
    d_rdy = d(np.full(B * 4, -1, dtype=np.int64))
    d_ardy = d(np.full(B * 4, -1, dtype=np.int64))
    d_eop = d(np.full(B * 4, -1, dtype=np.int64))
    d_eend = d(np.zeros(B * 4, dtype=np.float64))
    d_mu = d(np.zeros(B * 2, dtype=np.int64))
    d_mp = d(np.zeros(B * 2, dtype=np.int64))
    d_rcon = d(np.zeros(B * max_ten, dtype=np.int64))
    d_res = d(np.zeros(B * max_ten, dtype=np.bool_))
    d_crb = d(np.zeros(B * cr_cap * 2, dtype=np.int64))
    d_crt = d(np.full(B * cr_cap * 2, -1, dtype=np.int64))
    d_crh = d(np.zeros(B * 2, dtype=np.int64))
    d_crtail = d(np.zeros(B * 2, dtype=np.int64))
    d_po = d(np.zeros(B * pool_cap, dtype=np.int64))
    d_pw = d(np.zeros(B * pool_cap, dtype=np.float64))
    d_pa = d(np.zeros(B * pool_cap, dtype=np.bool_))
    d_pn = d(np.zeros(B, dtype=np.int64))
    d_pslot = d(np.full(B * max_ops, -1, dtype=np.int64))
    d_lu = d(np.zeros(B, dtype=np.float64))
    d_sa_scr = d(np.zeros(B * max_ops, dtype=np.int64))
    d_sw_scr = d(np.zeros(B * max_ops, dtype=np.float64))
    d_req = d(np.zeros(B * 2, dtype=np.int64))
    d_out = d(np.zeros(B * 3, dtype=np.float64))

    t0 = time.perf_counter()
    grid = (B + 127) // 128
    step3_batch_kernel[grid, 128](
        d_op_b, d_ten_b,
        d_pipe, d_ddr, d_dur,
        d_ip, d_it, d_op_ptr, d_ot,
        d_seq,
        d_tsz, d_tpl1,
        d_cnp, d_cna, d_ppp, d_ppa,
        d_psp, d_psq, d_psb,
        d_ao, d_ar, d_ab, d_nal,
        d_oip, d_oit, d_opp, d_oot,
        d_sp, d_sa_arr, d_pc,
        np.int64(CAP_L1), np.int64(CAP_UB),
        d_status, d_end, d_pred,
        d_pcur, d_rdy, d_ardy, d_eop, d_eend,
        d_mu, d_mp, d_rcon, d_res,
        d_crb, d_crt, d_crh, d_crtail,
        d_po, d_pw, d_pa, d_pn, d_pslot, d_lu,
        d_sa_scr, d_sw_scr, d_req,
        d_out,
        np.int64(max_ops), np.int64(max_ten),
        np.int64(cr_cap), np.int64(pool_cap))
    cuda.synchronize()
    t1 = time.perf_counter()

    out_h = d_out.copy_to_host()
    n_bad = 0
    for k in range(B):
        cpu_mk = cpu_s3[k]["makespan"]
        gpu_mk = int(round(out_h[k * 3]))
        if cpu_mk != gpu_mk:
            n_bad += 1
            print(f"  MK DIFF task{k}: CPU={cpu_mk} GPU={gpu_mk} "
                  f"(n={inputs[k]['n']})")
    return n_bad, B, t1 - t0


def main():
    cases = sys.argv[1:] or ["case_019"]
    total = 0
    for case in cases:
        for seed in (0, 1):
            n_bad, B, t = run_case(case, seed)
            total += n_bad
            print(f"{case} stub{seed}: {B} tasks, bad={n_bad} "
                  f"gpu={t*1000:.1f}ms", flush=True)
        n_bad, B, t = run_case(case, 0, use_v2=True, cpc=1)
        total += n_bad
        print(f"{case} v2cpc1: {B} tasks, bad={n_bad} "
              f"gpu={t*1000:.1f}ms", flush=True)
    print("GPU_STEP3_GATE:", "PASS" if total == 0 else f"FAIL ({total})")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
