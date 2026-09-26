# -*- coding: utf-8 -*-
"""阶段B2门禁：扩展图组装对拍（seq_ext / 新op / 新tensor / ext边集）。"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v2_solver"))

from fast_eval_p1 import (GraphCodec, TaskBuild, stage_step1,  # noqa: E402
                          stage_step2, build_ext_task, BW)
import stub_multicore_cut_and_schedule as stub  # noqa: E402
from gate_b import official_capture  # noqa: E402

TYPE_NAME = {1: "COPY_IN", 2: "COPY_OUT", 3: "COPY_IN", 4: "COPY_OUT"}


def official_ext_views(res):
    seq_ext = list(res["seq_ext"])
    new_ops = sorted(
        (o["id"], o["op"], o["pipe"], int(o["cycles"]))
        for o in res["new_ops"])
    new_tensors = sorted(
        (t["id"], int(t["size"]), t["pos"]) for t in res["new_tensors"])
    edges = sorted(sorted((e["source"], e["target"])) if False else
                   (e["source"], e["target"]) for e in res["ext_edges"])
    return seq_ext, new_ops, new_tensors, edges


def my_ext_views(ext):
    seq_ext = [int(g) for g in ext["op_gid"][ext["seq_ext"]]]
    new_ops = sorted(
        (int(ext["op_gid"][i]), TYPE_NAME[int(ext["op_type"][i])],
         ["PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_V"][int(ext["op_pipe"][i])],
         int(ext["op_cycles"][i]))
        for i in range(ext["n_ext_ops"]) if ext["op_type"][i] in (3, 4))
    new_tensors = sorted(
        (int(ext["ten_gid"][j]), int(ext["ten_size"][j]),
         {0: "DDR", 1: "L1", 2: "UB"}[int(ext["ten_pos"][j])])
        for j in range(ext["n_ten0"], ext["n_ext_ten"]))
    edges = set()
    n0 = None
    for o, t in ext["in_pairs"]:
        edges.add((int(ext["ten_gid"][t]), int(ext["op_gid"][o])))
    for o, t in ext["out_pairs"]:
        edges.add((int(ext["op_gid"][o]), int(ext["ten_gid"][t])))
    for s, d in ext["direct"]:
        edges.add((int(ext["op_gid"][s]), int(ext["op_gid"][d])))
    return seq_ext, new_ops, new_tensors, sorted(edges)


def run_plan(graph, plan, gc, tag):
    cap = official_capture(graph, plan)
    pv = stub.derive_multicore_plan(graph, plan)
    tb = TaskBuild(gc, pv)
    seqs = stage_step1(tb)
    pss = stage_step2(tb, seqs)
    n_fail = 0
    for k in range(tb.n_tasks):
        ext = build_ext_task(tb, k, seqs[k], pss[k])
        o_seq, o_ops, o_ten, o_edge = official_ext_views(cap[k])
        m_seq, m_ops, m_ten, m_edge = my_ext_views(ext)
        tk = f"{tag} task{k}"
        if o_seq != m_seq:
            n_fail += 1
            for z in range(min(len(o_seq), len(m_seq))):
                if o_seq[z] != m_seq[z]:
                    print(f"  [FAIL seq] {tk} at {z}: {o_seq[z]} vs {m_seq[z]}")
                    break
            else:
                print(f"  [FAIL seq] {tk} len {len(o_seq)} vs {len(m_seq)}")
        if o_ops != m_ops:
            n_fail += 1
            print(f"  [FAIL new_ops] {tk}: "
                  f"{sorted(set(o_ops) ^ set(m_ops))[:4]}")
        if o_ten != m_ten:
            n_fail += 1
            print(f"  [FAIL new_tensors] {tk}: "
                  f"{sorted(set(o_ten) ^ set(m_ten))[:4]}")
        if o_edge != m_edge:
            n_fail += 1
            de = set(map(tuple, o_edge)) ^ set(map(tuple, m_edge))
            print(f"  [FAIL edges] {tk}: ndiff={len(de)} {sorted(de)[:4]}")
    print(f"  {tag}: {tb.n_tasks} tasks ext-compared, fail={n_fail}")
    return n_fail


def main():
    from structure_split import structure_aware_plan
    data = Path(r"C:/shumo_live/a_data/data")
    total = 0
    for case in ("case_082", "case_050", "case_019", "case_001"):
        graph = json.load(open(data / f"{case}.json", encoding="utf-8"))
        gc = GraphCodec(graph)
        print(f"== {case}")
        for seed in (0, 1):
            plan = stub.generate_multicore_plan(
                graph, num_cores=4, seed=seed, min_subgraph_size=50,
                max_subgraph_size=100)
            total += run_plan(graph, plan, gc, f"stub{seed}")
        plan, _ = structure_aware_plan(graph, 4, chunks_per_core=1)
        total += run_plan(graph, plan, gc, "v2cpc1")
        plan, _ = structure_aware_plan(graph, 4, chunks_per_core=2)
        total += run_plan(graph, plan, gc, "v2cpc2")
    print("GATE_B2:", "PASS" if total == 0 else f"FAIL ({total})")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
