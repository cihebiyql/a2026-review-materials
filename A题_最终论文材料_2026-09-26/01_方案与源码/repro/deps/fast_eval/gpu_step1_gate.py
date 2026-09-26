# -*- coding: utf-8 -*-
"""GPU step1 批量对拍：CUDA 内核 vs CPU 复刻（fast_kernels.step1_seq）。

node1（cuda12.1 环境）运行：
  cd /data1/qlyu/tmp_shumo/fast_eval
  A2026_ATT=/data1/qlyu/a2026/att CUDA_VISIBLE_DEVICES=0 \
    /data/qlyu/anaconda3/envs/cuda12.1/bin/python gpu_step1_gate.py \
    case_019 case_082
批量：每用例 stub 方案 × 2 种子，全部 (plan,task) 对打包上卡，
一线程一对，与 CPU 逐对 bit-exact 对拍。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ATT = Path(__import__("os").environ.get("A2026_ATT",
                                        r"C:/shumo_live/a_data"))
sys.path.insert(0, str(ATT / "code"))

from fast_eval_p1 import GraphCodec, TaskBuild, stage_step1  # noqa: E402
from fast_kernels import fill_task_edges  # noqa: E402
import stub_multicore_cut_and_schedule as stub  # noqa: E402


def build_pairs(tb):
    """逐任务构建 pred/succ CSR（与 stage_step1 相同路径），返回打包件。"""
    from fast_eval_p1 import _transpose_csr
    n_flat_ops = int(tb.t_op_ptr[-1])
    n_flat_ten = int(tb.t_ten_ptr[-1])
    tcp, tca = _transpose_csr(tb.in_ptr, tb.in_ten, n_flat_ops, n_flat_ten)
    tpp, _ = _transpose_csr(tb.out_ptr, tb.out_ten, n_flat_ops, n_flat_ten)
    prod_cnt = np.diff(tpp)
    cons_cnt = np.diff(tcp)
    d_lo = np.searchsorted(tb.direct_task, np.arange(tb.n_tasks))
    d_hi = np.searchsorted(tb.direct_task, np.arange(1, tb.n_tasks + 1))
    pairs = []
    for k in range(tb.n_tasks):
        lo, hi = int(tb.t_op_ptr[k]), int(tb.t_op_ptr[k + 1])
        tlo, thi = int(tb.t_ten_ptr[k]), int(tb.t_ten_ptr[k + 1])
        n = hi - lo
        e_cap = int(np.sum(prod_cnt[tlo:thi] * cons_cnt[tlo:thi])) + int(
            d_hi[k] - d_lo[k]) + 1
        es = np.empty(e_cap, dtype=np.int64)
        ed = np.empty(e_cap, dtype=np.int64)
        cnt, err = fill_task_edges(
            lo, hi, tb.out_ptr, tb.out_ten, tcp, tca,
            int(d_lo[k]), int(d_hi[k]), tb.direct_src, tb.direct_dst, es, ed)
        assert err == 0
        if cnt:
            comp = np.unique(es[:cnt] * n + ed[:cnt])
            src = comp // n
            dst = comp % n
        else:
            src = np.zeros(0, dtype=np.int64)
            dst = np.zeros(0, dtype=np.int64)
        pred_ptr = np.zeros(n + 1, dtype=np.int64)
        succ_ptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(pred_ptr, dst + 1, 1)
        np.add.at(succ_ptr, src + 1, 1)
        pred_ptr = np.cumsum(pred_ptr)
        succ_ptr = np.cumsum(succ_ptr)
        pred_arr = src[np.argsort(dst * n + src, kind="stable")]
        succ_arr = dst[np.argsort(src * n + dst, kind="stable")]
        pairs.append((pred_ptr, pred_arr, succ_ptr, succ_arr))
    return pairs


def pack_batch(pairs):
    """多任务的 CSR 拼成平铺数组（对 i = pair_ptr 的段内偏移做 pred_ptr）。"""
    B = len(pairs)
    seg_ptrs = [p[0] for p in pairs]
    # pred/succ 的 ptr 是局部(0..n)；平铺后加段基址
    all_pred_ptr = []
    all_succ_ptr = []
    all_pred = []
    all_succ = []
    pair_ptr = np.zeros(B + 1, dtype=np.int64)
    for i, (pp, pa, sp, sa) in enumerate(pairs):
        base = pair_ptr[i]
        pair_ptr[i + 1] = base + (pp[-1] if len(pp) > 1 else 0) + \
            (len(pa) and 1 or 1)  # 占位，下面重算
    # 重新精确计算：op 段与 edge 段分开
    op_base = np.zeros(B + 1, dtype=np.int64)
    edge_base = np.zeros(B + 1, dtype=np.int64)
    for i, (pp, pa, sp, sa) in enumerate(pairs):
        op_base[i + 1] = op_base[i] + (len(pp) - 1)
        edge_base[i + 1] = edge_base[i] + len(pa)
    pred_ptr_flat = np.zeros(int(op_base[-1]) + 1, dtype=np.int64)
    succ_ptr_flat = np.zeros(int(op_base[-1]) + 1, dtype=np.int64)
    pred_flat = np.zeros(int(edge_base[-1]), dtype=np.int64)
    succ_flat = np.zeros(int(edge_base[-1]), dtype=np.int64)
    for i, (pp, pa, sp, sa) in enumerate(pairs):
        ob, eb = int(op_base[i]), int(edge_base[i])
        n = len(pp) - 1
        pred_ptr_flat[ob:ob + n + 1] = pp + eb
        succ_ptr_flat[ob:ob + n + 1] = sp + eb
        pred_flat[eb:eb + len(pa)] = pa
        succ_flat[eb:eb + len(sa)] = sa
    return op_base, pred_ptr_flat, pred_flat, succ_ptr_flat, succ_flat


def run_case(case, seed):
    from numba import cuda
    from gpu_batch import step1_batch_kernel
    graph = json.load(open(ATT / "data" / f"{case}.json", encoding="utf-8"))
    plan = stub.generate_multicore_plan(graph, num_cores=4, seed=seed,
                                        min_subgraph_size=50,
                                        max_subgraph_size=100)
    gc = GraphCodec(graph)
    pv = stub.derive_multicore_plan(graph, plan)
    tb = TaskBuild(gc, pv)
    cpu_seqs = stage_step1(tb)
    pairs = build_pairs(tb)
    op_base, pp_flat, pa_flat, sp_flat, sa_flat = pack_batch(pairs)
    B = len(pairs)
    n_ops_all = int(op_base[-1])
    max_ops = int(np.max(np.diff(op_base)))
    is_ci_flat = np.zeros(n_ops_all, dtype=np.int64)
    is_co_flat = np.zeros(n_ops_all, dtype=np.int64)
    for k in range(B):
        lo, hi = int(tb.t_op_ptr[k]), int(tb.t_op_ptr[k + 1])
        ob = int(op_base[k])
        for i in range(lo, hi):
            is_ci_flat[ob + i - lo] = tb.op_type[i] == 1
            is_co_flat[ob + i - lo] = tb.op_type[i] == 2

    t0 = time.perf_counter()
    d = cuda.to_device
    d_op, d_pp = d(op_base), d(pp_flat)
    d_pa, d_sp, d_sa = d(pa_flat), d(sp_flat), d(sa_flat)
    d_ci, d_co = d(is_ci_flat), d(is_co_flat)
    keys = np.zeros(B * max_ops, dtype=np.int64)
    skeys = np.zeros(B * max_ops, dtype=np.int64)
    depth = np.zeros(B * max_ops, dtype=np.int64)
    topo = np.zeros(B * max_ops, dtype=np.int64)
    max_edges = max(int(pp_flat[int(op_base[k+1])] - pp_flat[int(op_base[k])])
                    for k in range(B)) if B else 0
    stack_stride = max_ops + max_edges + 8
    stack = np.zeros(B * stack_stride, dtype=np.int64)
    visited = np.zeros(B * max_ops, dtype=np.bool_)
    seq = np.full(n_ops_all, -1, dtype=np.int64)
    tmp = np.zeros(B * max_ops, dtype=np.int64)
    indeg = np.zeros(B * max_ops, dtype=np.int64)
    d_keys, d_skeys, d_depth = d(keys), d(skeys), d(depth)
    d_topo, d_stack, d_vis, d_seq = d(topo), d(stack), d(visited), d(seq)
    d_tmp, d_indeg = d(tmp), d(indeg)
    block = 128
    grid = (B + block - 1) // block
    t1 = time.perf_counter()
    step1_batch_kernel[grid, block](
        d_op, d_pp, d_pa, d_sp, d_sa, d_ci, d_co,
        d_keys, d_skeys, d_depth, d_topo, d_stack, d_vis, d_seq, d_tmp,
        d_indeg, np.int64(max_ops), np.int64(stack_stride))
    cuda.synchronize()
    t2 = time.perf_counter()
    seq_h = d_seq.copy_to_host()
    # 校验：每段 seq_order（argsort of seq values）== CPU seq
    n_bad = 0
    for k in range(B):
        ob = int(op_base[k])
        n = int(op_base[k + 1]) - ob
        seg = seq_h[ob:ob + n]
        if np.any(seg < 0):
            n_bad += 1
            continue
        gpu_seq = np.argsort(seg, kind="stable")  # 槽位按序号排序 = 序列
        cpu = np.asarray(cpu_seqs[k]) - int(tb.t_op_ptr[k])
        if not np.array_equal(gpu_seq, cpu):
            n_bad += 1
    return n_bad, B, (t1 - t0) + (t2 - t1)


def main():
    cases = sys.argv[1:] or ["case_019"]
    total = 0
    for case in cases:
        for seed in (0, 1):
            n_bad, B, t_gpu = run_case(case, seed)
            total += n_bad
            print(f"{case} seed{seed}: {B} tasks, mismatch={n_bad} "
                  f"gpu={t_gpu*1000:.1f}ms", flush=True)
    print("GPU_STEP1_GATE:", "PASS" if total == 0 else f"FAIL ({total})")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
