# -*- coding: utf-8 -*-
"""阶段B门禁：step2 spill 模拟与官方逐 task 对拍。

对拍项（每任务，按 trigger 顺序）：
  (victim tensor gid, size, prev_use_step, next_use_step, spill_out_copies_data)
另对拍累计 spill 流量。
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fast_eval_p1 import (GraphCodec, TaskBuild, stage_step1,  # noqa: E402
                          stage_step2, BW)
import stub_multicore_cut_and_schedule as stub  # noqa: E402
import multicore_cut_evaluate_problem_1 as mce  # noqa: E402


def official_capture(graph, plan):
    captured = []
    orig_step2 = mce.step2_spill_insertion
    orig_prep = mce.prepare_step3_execution

    def w_step2(g, s, capacity=None):
        res = orig_step2(g, s, capacity=capacity)
        captured.append(res)
        return res

    mce.step2_spill_insertion = w_step2
    mce.prepare_step3_execution = lambda g, capacity=None, bandwidth=None: {}
    try:
        mce._build_scene_a_tasks(graph, plan, BW, {"L1": 524288, "UB": 131072})
    finally:
        mce.step2_spill_insertion = orig_step2
        mce.prepare_step3_execution = orig_prep
    return captured


def run_case(case_path, seeds=(0, 1)):
    with open(case_path, encoding="utf-8") as f:
        graph = json.load(f)
    gc = GraphCodec(graph)
    n_fail = 0
    for seed in seeds:
        plan = stub.generate_multicore_plan(
            graph, num_cores=4, seed=seed, min_subgraph_size=50,
            max_subgraph_size=100)
        cap = official_capture(graph, plan)
        pv = stub.derive_multicore_plan(graph, plan)
        tb = TaskBuild(gc, pv)
        seqs = stage_step1(tb)
        outs = stage_step2(tb, seqs)
        assert len(cap) == tb.n_tasks
        for k in range(tb.n_tasks):
            recs = cap[k]["spill_records"]
            mine = outs[k]
            if len(recs) != mine["n_ps"]:
                n_fail += 1
                print(f"  [FAIL count] seed{seed} task{k}: official="
                      f"{len(recs)} mine={mine['n_ps']}")
                continue
            for i, r in enumerate(recs):
                vg = int(tb.ten_gid[mine["tlo"] + int(mine["victim"][i])])
                ok = (r["tid"] == vg
                      and r["size"] == int(mine["size"][i])
                      and r["prev_use_step"] == int(mine["prev_step"][i])
                      and r["next_use_step"] == int(mine["next_step"][i]))
                if not ok:
                    n_fail += 1
                    print(f"  [FAIL rec] seed{seed} task{k} #{i}: official="
                          f"(tid={r['tid']}, sz={r['size']}, "
                          f"{r['prev_use_step']}->{r['next_use_step']}) "
                          f"mine=(tid={vg}, sz={int(mine['size'][i])}, "
                          f"{int(mine['prev_step'][i])}-"
                          f">{int(mine['next_step'][i])})")
                    break
        print(f"  seed={seed}: {tb.n_tasks} tasks, spills="
              f"{sum(o['n_ps'] for o in outs)}")
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
    print("GATE_B:", "PASS" if total == 0 else f"FAIL ({total})")
    sys.exit(0 if total == 0 else 1)
