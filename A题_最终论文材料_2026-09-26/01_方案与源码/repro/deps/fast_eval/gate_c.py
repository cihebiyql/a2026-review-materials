# -*- coding: utf-8 -*-
"""阶段C门禁：step3 local_makespan 与官方逐 task 对拍。"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v2_solver"))

from fast_eval_p1 import (GraphCodec, TaskBuild, stage_step1,  # noqa: E402
                          stage_step2, build_ext_all, stage_step3, BW)
import stub_multicore_cut_and_schedule as stub  # noqa: E402
import multicore_cut_evaluate_problem_1 as mce  # noqa: E402


def official_capture(graph, plan):
    captured = []
    orig = mce.prepare_step3_execution

    def w_prep(g, capacity=None, bandwidth=None):
        res = orig(g, capacity=capacity, bandwidth=bandwidth)
        captured.append(res["step3"]["makespan"])
        return res

    mce.prepare_step3_execution = w_prep
    try:
        mce._build_scene_a_tasks(graph, plan, BW,
                                 {"L1": 524288, "UB": 131072})
    finally:
        mce.prepare_step3_execution = orig
    return captured


def run_plan(graph, plan, gc, tag):
    cap = official_capture(graph, plan)
    pv = stub.derive_multicore_plan(graph, plan)
    tb = TaskBuild(gc, pv)
    seqs = stage_step1(tb)
    pss = stage_step2(tb, seqs)
    exts = build_ext_all(tb, seqs, pss)
    mine = [x["makespan"] for x in stage_step3(exts)]
    n_fail = 0
    for k, (a, b) in enumerate(zip(cap, mine)):
        if a != b:
            n_fail += 1
            print(f"  [FAIL mk] {tag} task{k}: official={a} mine={b} "
                  f"delta={b - a}")
    print(f"  {tag}: {len(cap)} tasks, fail={n_fail}")
    return n_fail


def main():
    from structure_split import structure_aware_plan
    data = Path(r"C:/shumo_live/a_data/data")
    total = 0
    for case in ("case_019", "case_001", "case_082", "case_050"):
        graph = json.load(open(data / f"{case}.json", encoding="utf-8"))
        gc = GraphCodec(graph)
        print(f"== {case}")
        for seed in (0, 1):
            plan = stub.generate_multicore_plan(
                graph, num_cores=4, seed=seed, min_subgraph_size=50,
                max_subgraph_size=100)
            total += run_plan(graph, plan, gc, f"stub{seed}")
        for cpc in (1, 2):
            plan, _ = structure_aware_plan(graph, 4, chunks_per_core=cpc)
            total += run_plan(graph, plan, gc, f"v2cpc{cpc}")
    print("GATE_C:", "PASS" if total == 0 else f"FAIL ({total})")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
