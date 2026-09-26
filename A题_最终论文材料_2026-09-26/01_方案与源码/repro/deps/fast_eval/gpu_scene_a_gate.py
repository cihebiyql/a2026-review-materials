# -*- coding: utf-8 -*-
"""GPU 场景A 对拍：CUDA 多核事件循环 vs CPU 复刻（stage_scene_a）。

node1（cuda12.1）：
  A2026_ATT=/data1/qlyu/a2026/att CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=v2solver /data/qlyu/anaconda3/envs/cuda12.1/bin/python \
  gpu_scene_a_gate.py case_019 case_001 case_082 case_050
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
                          build_ext_all, stage_step3, stage_scene_a,
                          BW, CAP_L1, CAP_UB, SAME_WAIT_A, CROSS_WAIT_A)
import stub_multicore_cut_and_schedule as stub
from structure_split import structure_aware_plan


def prep_scene_a(tb, s3_list, exts):
    """打包场景A的全部输入。"""
    n_tasks = tb.n_tasks
    n_cores = tb.num_cores
    task_op_base = np.zeros(n_tasks + 1, dtype=np.int64)
    for k in range(n_tasks):
        task_op_base[k + 1] = task_op_base[k] + s3_list[k]["n_ops"]
    n_gops = int(task_op_base[-1])

    gop_pipe = np.zeros(n_gops, dtype=np.int64)
    gop_dur = np.zeros(n_gops, dtype=np.int64)
    gop_is_ddr = np.zeros(n_gops, dtype=np.bool_)
    gop_task = np.zeros(n_gops, dtype=np.int64)
    task_seq_parts = []
    succ_parts = []
    pipe_seq_parts = []
    pipe_ptr_parts = []
    pipe_seq_base = np.zeros(n_tasks, dtype=np.int64)
    pipe_ptr_base = np.zeros(n_tasks, dtype=np.int64)

    for k in range(n_tasks):
        s3 = s3_list[k]
        b = task_op_base[k]
        for i in range(s3["n_ops"]):
            gop_pipe[b + i] = int(exts[k]["op_pipe"][i])
            gop_dur[b + i] = int(s3["op_dur"][i])
            gop_is_ddr[b + i] = bool(s3["op_is_ddr"][i])
            gop_task[b + i] = k
        task_seq_parts.append(s3["seq_ext"] + b)
        edges = set()
        for g in range(s3["n_ops"]):
            for e in range(s3["succ_ptr"][g], s3["succ_ptr"][g + 1]):
                edges.add((g, int(s3["succ_arr"][e])))
        for s, d in zip(s3["deps"], s3["depd"]):
            edges.add((int(s), int(d)))
        succ_parts.extend((b + s, b + d) for s, d in edges)
        pipe_seq_parts.append(s3["pipe_seq"])
        pipe_ptr_parts.append(s3["pipe_seq_ptr"])

    task_seq_arr = np.concatenate(task_seq_parts) if task_seq_parts else \
        np.zeros(0, dtype=np.int64)
    task_seq_ptr = np.zeros(n_tasks + 1, dtype=np.int64)
    for k in range(n_tasks):
        task_seq_ptr[k + 1] = task_seq_ptr[k] + len(task_seq_parts[k])
    pipe_seq_arr = np.concatenate(pipe_seq_parts) if pipe_seq_parts else \
        np.zeros(0, dtype=np.int64)
    pipe_ptr = np.concatenate(pipe_ptr_parts) if pipe_ptr_parts else \
        np.zeros(0, dtype=np.int64)
    acc_sq = 0
    acc_pt = 0
    for k in range(n_tasks):
        pipe_seq_base[k] = acc_sq
        pipe_ptr_base[k] = acc_pt
        acc_sq += len(pipe_seq_parts[k])
        acc_pt += len(pipe_ptr_parts[k])

    # succ CSR
    if succ_parts:
        comp = np.unique(np.array([s * n_gops + d for s, d in succ_parts],
                                  dtype=np.int64))
        esrc = (comp // n_gops).astype(np.int64)
        edst = (comp % n_gops).astype(np.int64)
    else:
        esrc = np.zeros(0, dtype=np.int64)
        edst = np.zeros(0, dtype=np.int64)
    succ_ptr = np.zeros(n_gops + 1, dtype=np.int64)
    np.add.at(succ_ptr, esrc + 1, 1)
    succ_ptr = np.cumsum(succ_ptr)
    succ_arr = edst[np.argsort(esrc * n_gops + edst, kind="stable")]
    pred_cnt = np.zeros(n_gops, dtype=np.int64)
    np.add.at(pred_cnt, edst, 1)

    # 核序/任务前驱
    core_ptr = np.zeros(n_cores + 1, dtype=np.int64)
    for c in range(n_cores):
        core_ptr[c + 1] = core_ptr[c] + len(tb.core_orders[c])
    core_tasks = np.array([t for c in range(n_cores) for t in
                           tb.core_orders[c]], dtype=np.int64)
    task_core = np.zeros(n_tasks, dtype=np.int64)
    for k in range(n_tasks):
        task_core[k] = tb.core_of[k]
    tp = tb.task_preds
    task_pred_ptr = np.zeros(n_tasks + 1, dtype=np.int64)
    for k in range(n_tasks):
        task_pred_ptr[k + 1] = task_pred_ptr[k] + len(tp[k])
    task_pred_arr = np.array([p for l in tp for p in l], dtype=np.int64)

    return (n_cores, n_tasks, n_gops,
            core_ptr, core_tasks, task_core,
            task_pred_ptr, task_pred_arr,
            gop_pipe, gop_dur, gop_is_ddr, gop_task,
            task_seq_ptr, task_seq_arr,
            pipe_ptr, pipe_seq_arr, pipe_ptr_base, pipe_seq_base,
            task_op_base,
            succ_ptr, succ_arr, pred_cnt)


def run_case(case, seed, use_v2=False, cpc=1):
    from numba import cuda
    from gpu_scene_a import scene_a_kernel
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
    s3 = stage_step3(exts)

    # CPU 基准
    cpu_mk, _, _ = stage_scene_a(tb, s3, exts)

    # GPU 打包
    (n_cores, n_tasks, n_gops, core_ptr, core_tasks, task_core,
     task_pred_ptr, task_pred_arr, gop_pipe, gop_dur, gop_is_ddr, gop_task,
     task_seq_ptr, task_seq_arr, pipe_ptr, pipe_seq_arr,
     pipe_ptr_base, pipe_seq_base, task_op_base,
     succ_ptr, succ_arr, pred_cnt) = prep_scene_a(tb, s3, exts)

    d = cuda.to_device
    d_nc = np.int64(n_cores)
    d_nt = np.int64(n_tasks)
    d_ng = np.int64(n_gops)
    d_cp = d(core_ptr)
    d_ct = d(core_tasks)
    d_tc = d(task_core)
    d_tpp = d(task_pred_ptr)
    d_tpa = d(task_pred_arr)
    d_gp = d(gop_pipe)
    d_gd = d(gop_dur)
    d_gddr = d(gop_is_ddr)
    d_gt = d(gop_task)
    d_tsp = d(task_seq_ptr)
    d_tsa = d(task_seq_arr)
    d_pp = d(pipe_ptr)
    d_psa = d(pipe_seq_arr)
    d_ppb = d(pipe_ptr_base)
    d_psb = d(pipe_seq_base)
    d_tob = d(task_op_base)
    d_sp = d(succ_ptr)
    d_sa2 = d(succ_arr)
    d_pc = d(pred_cnt)

    # scratch
    pool_cap = n_gops + 8
    st_status = d(np.zeros(n_gops, dtype=np.int64))
    st_end = d(np.zeros(n_gops, dtype=np.float64))
    st_pred = d(np.zeros(n_gops, dtype=np.int64))
    st_pcur = d(np.zeros(n_tasks * 4, dtype=np.int64))
    st_rdy = d(np.full(n_tasks * 4, -1, dtype=np.int64))
    st_eop = d(np.full(n_cores * 4, -1, dtype=np.int64))
    st_eend = d(np.zeros(n_cores * 4, dtype=np.float64))
    st_tstatus = d(np.zeros(n_tasks, dtype=np.int64))
    st_tend = d(np.zeros(n_tasks, dtype=np.int64))
    st_cidx = d(np.zeros(n_cores, dtype=np.int64))
    st_cact = d(np.full(n_cores, -1, dtype=np.int64))
    st_cprev = d(np.full(n_cores, -1, dtype=np.int64))
    st_po = d(np.zeros(pool_cap, dtype=np.int64))
    st_pw = d(np.zeros(pool_cap, dtype=np.float64))
    st_pa = d(np.zeros(pool_cap, dtype=np.bool_))
    st_pn = d(np.zeros(1, dtype=np.int64))
    st_pslot = d(np.full(n_gops, -1, dtype=np.int64))
    st_lu = d(np.zeros(1, dtype=np.float64))
    st_sa = d(np.zeros(n_gops, dtype=np.int64))
    st_sw = d(np.zeros(n_gops, dtype=np.float64))
    d_out = d(np.zeros(1, dtype=np.float64))

    t0 = time.perf_counter()
    scene_a_kernel[1, 1](
        d_nc, d_nt, d_ng,
        d_cp, d_ct, d_tc,
        d_tpp, d_tpa,
        d_gp, d_gd, d_gddr, d_gt,
        d_tsp, d_tsa,
        d_pp, d_psa, d_ppb, d_psb, d_tob,
        d_sp, d_sa2, d_pc,
        np.int64(SAME_WAIT_A), np.int64(CROSS_WAIT_A),
        st_status, st_end, st_pred,
        st_pcur, st_rdy,
        st_eop, st_eend,
        st_tstatus, st_tend,
        st_cidx, st_cact, st_cprev,
        st_po, st_pw, st_pa, st_pn, st_pslot, st_lu,
        st_sa, st_sw,
        d_out)
    cuda.synchronize()
    t1 = time.perf_counter()

    gpu_mk = int(round(d_out.copy_to_host()[0]))
    tag = f"v2cpc{cpc}" if use_v2 else f"stub{seed}"
    ok = cpu_mk == gpu_mk
    print(f"  {tag}: CPU={cpu_mk} GPU={gpu_mk} {'OK' if ok else 'FAIL'} "
          f"({t1 - t0:.1f}s)", flush=True)
    return 0 if ok else 1


def main():
    cases = sys.argv[1:] or ["case_019"]
    total = 0
    for case in cases:
        print(f"== {case}")
        for seed in (0, 1):
            total += run_case(case, seed)
        total += run_case(case, 0, use_v2=True, cpc=1)
    print(f"GPU_SCENE_A_GATE: {'PASS' if total == 0 else 'FAIL'} ({total})")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
