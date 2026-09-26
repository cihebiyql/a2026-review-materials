# -*- coding: utf-8 -*-
"""GPU 场景B 对拍：CUDA P2/P3 多核事件循环 vs CPU 复刻（stage_scene_b）。"""
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

from fast_eval_p1 import GraphCodec, stage_step1, stage_step2, \
    build_ext_all, stage_step3
from fast_eval_p2 import TaskBuildB, _prioritize, stage_scene_b, DELAY_B
import stub_multicore_cut_and_schedule as stub
from structure_split import structure_aware_plan


def run_case(case, seed, use_v2=False, cpc=1):
    from numba import cuda
    from gpu_scene_b import scene_b_kernel
    graph = json.load(open(os.path.join(ATT, "data", f"{case}.json")))
    if use_v2:
        plan, _ = structure_aware_plan(graph, 4, chunks_per_core=cpc)
    else:
        plan = stub.generate_multicore_plan(graph, num_cores=4, seed=seed,
                                            min_subgraph_size=50,
                                            max_subgraph_size=100)
    gc = GraphCodec(graph)
    pv = stub.derive_multicore_plan(graph, plan)
    tb = TaskBuildB(gc, pv)
    seqs = _prioritize(tb, stage_step1(tb))
    pss = stage_step2(tb, seqs)
    exts = build_ext_all(tb, seqs, pss)
    s3 = stage_step3(exts)

    # CPU 基准
    cpu_mk, _ = stage_scene_b(tb, s3, exts, use_cache=False)

    # 打包
    n_tasks = tb.num_cores  # P2: task = core
    n_cores = tb.num_cores
    task_op_base = np.zeros(n_tasks + 1, dtype=np.int64)
    for k in range(n_tasks):
        task_op_base[k + 1] = task_op_base[k] + s3[k]["n_ops"]
    n_gops = int(task_op_base[-1])

    gop_pipe = np.zeros(n_gops, dtype=np.int64)
    gop_dur = np.zeros(n_gops, dtype=np.int64)
    gop_is_ddr = np.zeros(n_gops, dtype=np.bool_)
    gop_core = np.zeros(n_gops, dtype=np.int64)
    task_seq_parts = []
    succ_parts = []
    pipe_seq_parts = []
    pipe_ptr_parts = []
    for k in range(n_tasks):
        s = s3[k]
        b = task_op_base[k]
        for i in range(s["n_ops"]):
            gop_pipe[b + i] = int(exts[k]["op_pipe"][i])
            gop_dur[b + i] = int(s["op_dur"][i])
            gop_is_ddr[b + i] = bool(s["op_is_ddr"][i])
            gop_core[b + i] = k
        task_seq_parts.append(s["seq_ext"] + b)
        edges = set()
        for g in range(s["n_ops"]):
            for e in range(s["succ_ptr"][g], s["succ_ptr"][g + 1]):
                edges.add((g, int(s["succ_arr"][e])))
        for sd, dd in zip(s["deps"], s["depd"]):
            edges.add((int(sd), int(dd)))
        succ_parts.extend((b + s2, b + d2) for s2, d2 in edges)
        pipe_seq_parts.append(s["pipe_seq"])
        pipe_ptr_parts.append(s["pipe_seq_ptr"])

    task_seq_arr = np.concatenate(task_seq_parts)
    task_seq_ptr = np.zeros(n_tasks + 1, dtype=np.int64)
    for k in range(n_tasks):
        task_seq_ptr[k + 1] = task_seq_ptr[k] + len(task_seq_parts[k])
    pipe_seq_arr = np.concatenate(pipe_seq_parts)
    pipe_ptr = np.concatenate(pipe_ptr_parts)
    pipe_seq_base = np.zeros(n_tasks, dtype=np.int64)
    pipe_ptr_base = np.zeros(n_tasks, dtype=np.int64)
    acc_sq = acc_pt = 0
    for k in range(n_tasks):
        pipe_seq_base[k] = acc_sq
        pipe_ptr_base[k] = acc_pt
        acc_sq += len(pipe_seq_parts[k])
        acc_pt += len(pipe_ptr_parts[k])

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

    # 跨核链接
    link_src = np.array([task_op_base[c] + s2 for c, s2 in tb.cross_src_cl],
                        dtype=np.int64)
    link_dst = np.array([task_op_base[c] + s2 for c, s2 in tb.cross_dst_cl],
                        dtype=np.int64)
    n_links = len(link_src)
    ext_pred_arr = np.full(n_gops, -1, dtype=np.int64)
    ext_succ_ptr = np.zeros(n_gops + 1, dtype=np.int64)
    np.add.at(ext_succ_ptr, link_src + 1, 1)
    ext_succ_ptr = np.cumsum(ext_succ_ptr)
    ext_succ_arr = link_dst[np.argsort(link_src * n_gops + link_dst,
                                       kind="stable")]
    for e in range(n_links):
        ext_pred_arr[link_dst[e]] = link_src[e]

    d = cuda.to_device
    pool_cap = n_gops + 8
    heap_cap = n_links + 8
    st_status = d(np.zeros(n_gops, dtype=np.int64))
    st_end = d(np.zeros(n_gops, dtype=np.float64))
    st_pred = d(np.zeros(n_gops, dtype=np.int64))
    st_pcur = d(np.zeros(n_cores * 4, dtype=np.int64))
    st_rdy = d(np.full(n_cores * 4, -1, dtype=np.int64))
    st_eop = d(np.full(n_cores * 4, -1, dtype=np.int64))
    st_eend = d(np.zeros(n_cores * 4, dtype=np.float64))
    st_po = d(np.zeros(pool_cap, dtype=np.int64))
    st_pw = d(np.zeros(pool_cap, dtype=np.float64))
    st_pa = d(np.zeros(pool_cap, dtype=np.bool_))
    st_pn = d(np.zeros(1, dtype=np.int64))
    st_pslot = d(np.full(n_gops, -1, dtype=np.int64))
    st_lu = d(np.zeros(1, dtype=np.float64))
    st_sa = d(np.zeros(n_gops, dtype=np.int64))
    st_sw = d(np.zeros(n_gops, dtype=np.float64))
    rel_sched = d(np.zeros(n_gops, dtype=np.int64))
    heap = d(np.zeros(heap_cap, dtype=np.int64))
    heap_n = d(np.zeros(1, dtype=np.int64))
    ret_buf = d(np.zeros(n_cores * 4 + 8, dtype=np.int64))
    d_out = d(np.zeros(1, dtype=np.float64))

    t0 = time.perf_counter()
    scene_b_kernel[1, 1](
        np.int64(n_cores), np.int64(n_gops), np.int64(n_links),
        d(gop_pipe), d(gop_dur), d(gop_is_ddr), d(gop_core),
        d(task_seq_ptr), d(task_seq_arr),
        d(pipe_ptr), d(pipe_seq_arr), d(pipe_ptr_base), d(pipe_seq_base),
        d(task_op_base),
        d(succ_ptr), d(succ_arr), d(pred_cnt),
        d(link_src), d(link_dst), d(ext_pred_arr),
        d(ext_succ_ptr), d(ext_succ_arr),
        np.int64(DELAY_B),
        st_status, st_end, st_pred,
        st_pcur, st_rdy,
        st_eop, st_eend,
        st_po, st_pw, st_pa, st_pn, st_pslot, st_lu,
        st_sa, st_sw,
        rel_sched, heap, heap_n, ret_buf,
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
    print(f"GPU_SCENE_B_GATE: {'PASS' if total == 0 else 'FAIL'} ({total})")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
