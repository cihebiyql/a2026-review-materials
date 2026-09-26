# -*- coding: utf-8 -*-
"""阶段A门禁：任务构造 + step1 与官方逐 task 对拍。

官方侧：patch step1_schedule 捕获每任务图与官方 seq；
patch step2/prepare 跳过后续阶段（本门禁只验构造+step1）。
我方侧：GraphCodec + TaskBuild + stage_step1。

对拍项（每任务）：
  1. op 集合: (gid, is_copy_in, is_copy_out, pipe, cycles)
  2. tensor 集合: (gid, size, pos∈{DDR,L1,UB})
  3. 二部边集合: (op_gid, tensor_gid) 方向对
  4. step1 seq: 官方 op id 序 == 我方 gid 序
  5. 槽序不变量: 每任务平铺序与 gid 升序一致
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fast_eval_p1 import (GraphCodec, TaskBuild, stage_step1,  # noqa: E402
                          BW, T_COPY_IN, T_COPY_OUT)
import stub_multicore_cut_and_schedule as stub  # noqa: E402
import multicore_cut_evaluate_problem_1 as mce  # noqa: E402


def official_capture(graph, plan):
    captured = []
    orig_step1 = mce.step1_schedule
    orig_step2 = mce.step2_spill_insertion
    orig_prep = mce.prepare_step3_execution

    def w_step1(g):
        seq = orig_step1(g)
        captured.append((g, list(seq)))
        return seq

    mce.step1_schedule = w_step1
    mce.step2_spill_insertion = lambda g, s, capacity=None: {
        "spill_records": [], "new_ops": [], "new_tensors": [],
        "ext_edges": g["edges"], "seq_ext": s}
    mce.prepare_step3_execution = lambda g, capacity=None, bandwidth=None: {}
    try:
        mce._build_scene_a_tasks(graph, plan, BW, {"L1": 524288, "UB": 131072})
    finally:
        mce.step1_schedule = orig_step1
        mce.step2_spill_insertion = orig_step2
        mce.prepare_step3_execution = orig_prep
    return captured


def official_task_view(g):
    ops = []
    for o in g["ops"]:
        ops.append((int(o["id"]),
                    o.get("op") == "COPY_IN", o.get("op") == "COPY_OUT",
                    o.get("pipe"), int(o.get("cycles", 1))))
    tens = []
    for t in g["tensors"]:
        tens.append((int(t["id"]), int(t["size"]), t.get("pos", "UB")))
    op_ids = {o[0] for o in ops}
    ten_ids = {t[0] for t in tens}
    edges = set()
    for e in g["edges"]:
        s, d = int(e["source"]), int(e["target"])
        if s in op_ids and d in ten_ids:
            edges.add(("o2t", s, d))
        elif s in ten_ids and d in op_ids:
            edges.add(("t2o", d, s))
    return sorted(ops), sorted(tens), sorted(edges)


def my_task_view(tb, k):
    lo, hi = int(tb.t_op_ptr[k]), int(tb.t_op_ptr[k + 1])
    tlo, thi = int(tb.t_ten_ptr[k]), int(tb.t_ten_ptr[k + 1])
    ops = []
    for i in range(lo, hi):
        ops.append((int(tb.op_gid[i]),
                    tb.op_type[i] == T_COPY_IN, tb.op_type[i] == T_COPY_OUT,
                    ["PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_V"][
                        int(tb.op_pipe[i])],
                    int(tb.op_cycles[i])))
    tens = []
    for j in range(tlo, thi):
        pos = {0: "DDR", 1: "L1", 2: "UB"}[int(tb.ten_pos[j])]
        tens.append((int(tb.ten_gid[j]), int(tb.ten_size[j]), pos))
    edges = set()
    gid_of_flat_op = tb.op_gid
    gid_of_flat_ten = tb.ten_gid
    for i in range(lo, hi):
        for e in range(tb.out_ptr[i], tb.out_ptr[i + 1]):
            edges.add(("o2t", int(gid_of_flat_op[i]),
                       int(gid_of_flat_ten[tb.out_ten[e]])))
        for e in range(tb.in_ptr[i], tb.in_ptr[i + 1]):
            edges.add(("t2o", int(gid_of_flat_op[i]),
                       int(gid_of_flat_ten[tb.in_ten[e]])))
    return sorted(ops), sorted(tens), sorted(edges)


def run_case(case_path, seeds=(0, 1), num_cores=4):
    with open(case_path, encoding="utf-8") as f:
        graph = json.load(f)
    gc = GraphCodec(graph)
    n_fail = 0
    for seed in seeds:
        plan = stub.generate_multicore_plan(
            graph, num_cores=num_cores, seed=seed,
            min_subgraph_size=50, max_subgraph_size=100)
        cap = official_capture(graph, plan)
        plan_view = stub.derive_multicore_plan(graph, plan)
        tb = TaskBuild(gc, plan_view)
        seqs = stage_step1(tb)
        assert len(cap) == tb.n_tasks, (len(cap), tb.n_tasks)
        for k in range(tb.n_tasks):
            g, oseq = cap[k]
            o_ops, o_ten, o_edge = official_task_view(g)
            m_ops, m_ten, m_edge = my_task_view(tb, k)
            tag = f"seed{seed} task{k}"
            if o_ops != m_ops:
                n_fail += 1
                print(f"  [FAIL ops] {tag}: "
                      f"diff={set(o_ops) ^ set(m_ops)}"[:300])
            if o_ten != m_ten:
                n_fail += 1
                print(f"  [FAIL tensors] {tag}")
            if o_edge != m_edge:
                n_fail += 1
                de = set(o_edge) ^ set(m_edge)
                print(f"  [FAIL edges] {tag}: ndiff={len(de)} "
                      f"sample={sorted(de)[:4]}")
            m_seq_gid = [int(tb.op_gid[i]) for i in seqs[k]]
            if oseq != m_seq_gid:
                n_fail += 1
                for z, (a, b) in enumerate(zip(oseq, m_seq_gid)):
                    if a != b:
                        print(f"  [FAIL seq] {tag}: first diff at {z}: "
                              f"official={a} mine={b}")
                        break
                else:
                    print(f"  [FAIL seq] {tag}: length "
                          f"{len(oseq)} vs {len(m_seq_gid)}")
            # 槽序不变量
            ids = tb.op_gid[int(tb.t_op_ptr[k]):int(tb.t_op_ptr[k + 1])]
            if not np.all(np.diff(ids) > 0):
                n_fail += 1
                print(f"  [FAIL slot order] {tag}")
        print(f"  seed={seed}: {tb.n_tasks} tasks compared")
    return n_fail


if __name__ == "__main__":
    import os
    data = Path(os.environ.get("A2026_DATA",
                               r"C:/shumo_live/a_data/data"))
    cases = sys.argv[1:] or ["case_019", "case_001"]
    total = 0
    for c in cases:
        print(f"== {c}")
        total += run_case(data / f"{c}.json")
    print("GATE_A:", "PASS" if total == 0 else f"FAIL ({total})")
    sys.exit(0 if total == 0 else 1)
